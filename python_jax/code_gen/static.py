import argparse
import importlib
import json
import re
import sys
from pathlib import Path

# Enable 64-bit precision in JAX
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# Add current dir to path for dynamic imports
sys.path.append(str(Path(__file__).parent.resolve()))

# ---------------------------------------------------------------------------
# JAX Solver for Hybrid Decoupled Kernels
# ---------------------------------------------------------------------------

def _strip_jsonc_comments(text):
    return re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)

def _load_json_or_jsonc(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonc":
        text = _strip_jsonc_comments(text)
    return json.loads(text)

def _material_params_from_model(model, pid):
    prop_by_pid = {int(p["pid"]): p for p in model.get("property", [])}
    mat_by_mid = {int(m["mid"]): m for m in model.get("material", [])}
    prop = prop_by_pid[int(pid)]
    mat = mat_by_mid[int(prop["mid"])]
    return jnp.array([float(mat["E"]), float(mat["nu"])])

def _load_kernel_func(module_name, func_name_prefix):
    """Dynamically loads a compute function from a generated module."""
    try:
        mod = importlib.import_module(module_name)
        func_name = f"compute_{func_name_prefix}"
        if not hasattr(mod, func_name):
            raise AttributeError(f"Function '{func_name}' not found in module '{module_name}'")
        return getattr(mod, func_name)
    except ModuleNotFoundError:
        raise FileNotFoundError(f"Generated module '{module_name}.py' not found. Please generate it first.")

def build_solver_data(model):
    nid_to_idx = {int(n["nid"]): i for i, n in enumerate(model["mesh"]["nodes"])}
    coords_list = [[float(n["x"]), float(n["y"]), float(n["z"])] for n in model["mesh"]["nodes"]]
    nodes_arr = jnp.asarray(coords_list, dtype=jnp.float64)

    elements_data = []
    for e in model["mesh"]["elements"]:
        etype = int(e["etype"])
        if etype not in [304, 308]:
            raise ValueError(f"This solver currently only supports Tet4 (304) or Hex8 (308), but got {etype}")
        elements_data.append({
            "pid": int(e["pid"]), 
            "nids": [int(nid) for nid in e["nids"]],
            "etype": etype
        })

    nsid_to_nids = {int(ns["nsid"]): [int(nid) for nid in ns["nids"]] for ns in model.get("nodeset", [])}
    return nodes_arr, elements_data, nid_to_idx, nsid_to_nids

def assemble_global_K_and_F(model, nodes_arr, elements_data, nid_to_idx, nsid_to_nids, kernels):
    """
    Assembles the global stiffness matrix K and load vector F.
    Supports both Tet4 (single kernel) and Hex8 (operator-based).
    """
    ndof = nodes_arr.shape[0] * 3
    K = jnp.zeros((ndof, ndof), dtype=jnp.float64)
    F = jnp.zeros((ndof,), dtype=jnp.float64)

    # Pre-extract kernels
    d_kernel = kernels.get("material_D")
    tet4_ke_kernel = kernels.get("tet4_Ke")
    
    # Hex8 operators
    hex8_op_dN = kernels.get("hex8_op_dN_dnat")
    hex8_op_map = kernels.get("hex8_op_mapping")
    hex8_op_asm = kernels.get("hex8_op_assembly")

    for e in elements_data:
        nids = e["nids"]
        coords = jnp.stack([nodes_arr[nid_to_idx[nid]] for nid in nids])
        mat_params = _material_params_from_model(model, e["pid"])
        d_matrix_flat = jnp.asarray(d_kernel(mat_params))

        etype = e["etype"]
        Ke = None

        if etype == 304: # Tet4
            stiffness_kernel_input = jnp.concatenate([coords.flatten(), d_matrix_flat])
            ke_flat = jnp.asarray(tet4_ke_kernel(stiffness_kernel_input))
            Ke = ke_flat.reshape(12, 12)
            edofs = jnp.array([[nid_to_idx[nid] * 3 + i for i in range(3)] for nid in nids]).flatten()
        
        elif etype == 308: # Hex8
            # 2x2x2 Gauss integration
            gp_val = 1.0 / jnp.sqrt(3.0)
            gps = [-gp_val, gp_val]
            gauss_points = [(x, y, z) for x in gps for y in gps for z in gps]
            
            Ke = jnp.zeros((24, 24), dtype=jnp.float64)
            coords_flat = coords.flatten()

            for pt in gauss_points:
                # 1. dN/dnat
                dN_dnat = jnp.asarray(hex8_op_dN(jnp.array(pt)))
                
                # 2. Mapping
                map_input = jnp.concatenate([coords_flat, dN_dnat])
                map_output = jnp.asarray(hex8_op_map(map_input))
                dN_dx = map_output[0:24]
                detJ = map_output[24]
                
                # 3. Assembly
                asm_input = jnp.concatenate([dN_dx, d_matrix_flat, jnp.array([detJ, 1.0])])
                ke_gp_flat = jnp.asarray(hex8_op_asm(asm_input))
                Ke = Ke + ke_gp_flat.reshape(24, 24)
            
            edofs = jnp.array([[nid_to_idx[nid] * 3 + i for i in range(3)] for nid in nids]).flatten()

        if Ke is not None:
            K = K.at[jnp.ix_(edofs, edofs)].add(Ke)

    # Assemble global load vector F
    for ld in model.get("load", []):
        nsid = int(ld["nsid"])
        value = float(ld["value"])
        s = str(ld["dof"]).lower()
        comps = []
        if "x" in s or "1" in s or "all" in s: comps.append(0)
        if "y" in s or "2" in s or "all" in s: comps.append(1)
        if "z" in s or "3" in s or "all" in s: comps.append(2)
        
        for nid in nsid_to_nids[nsid]:
            if nid in nid_to_idx:
                base_dof = nid_to_idx[nid] * 3
                for comp in comps:
                    F = F.at[base_dof + comp].add(value)

    return K, F

def apply_dirichlet_bc_and_solve(model, K, F, nid_to_idx, nsid_to_nids):
    """Applies zero-displacement boundary conditions and solves the linear system."""
    fixed_dofs = set()
    for bc in model.get("boundary", []):
        if abs(float(bc.get("value", 0.0))) > 1e-9:
            raise ValueError("This solver only supports zero-displacement Dirichlet BCs.")
        
        nsid = int(bc["nsid"])
        s = str(bc["dof"]).lower()
        comps = []
        if "x" in s or "1" in s or "all" in s: comps.append(0)
        if "y" in s or "2" in s or "all" in s: comps.append(1)
        if "z" in s or "3" in s or "all" in s: comps.append(2)

        for nid in nsid_to_nids[nsid]:
            if nid in nid_to_idx:
                base_dof = nid_to_idx[nid] * 3
                for comp in comps:
                    fixed_dofs.add(base_dof + comp)

    ndof = K.shape[0]
    all_dofs_set = set(range(ndof))
    free_dofs_set = all_dofs_set - fixed_dofs
    free_dofs = jnp.array(sorted(list(free_dofs_set)), dtype=jnp.int32)
    
    K_ff = K[jnp.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    
    u_f = jnp.linalg.solve(K_ff, F_f)

    U = jnp.zeros(ndof, dtype=jnp.float64)
    U = U.at[free_dofs].set(u_f)
    return U

def main():
    parser = argparse.ArgumentParser(description="JAX solver using decoupled kernels")
    parser.add_argument("--model", required=True, help="Path to the model file")
    parser.add_argument("--element", default="tet4", help="Element type")
    parser.add_argument("--material", default="isotropic", help="Material type")
    
    args = parser.parse_args()

    # 1. Load generated kernels
    kernels = {}
    
    # Material kernel
    d_mod = f"{args.material}_D_gen"
    kernels["material_D"] = _load_kernel_func(d_mod, f"{args.material}_D")

    # Element kernels/operators
    if args.element == "tet4":
        ke_mod = "tet4_Ke_gen"
        kernels["tet4_Ke"] = _load_kernel_func(ke_mod, "tet4_Ke")
    elif args.element == "hex8":
        kernels["hex8_op_dN_dnat"] = _load_kernel_func("hex8_op_dN_dnat_gen", "hex8_op_dN_dnat")
        kernels["hex8_op_mapping"] = _load_kernel_func("hex8_op_mapping_gen", "hex8_op_mapping")
        kernels["hex8_op_assembly"] = _load_kernel_func("hex8_op_assembly_gen", "hex8_op_assembly")

    print("Successfully loaded generated kernels.")

    # 2. Load model and build solver data
    model = _load_json_or_jsonc(args.model)
    nodes_arr, elements_data, nid_to_idx, nsid_to_nids = build_solver_data(model)

    # 3. Assemble and solve
    K, F = assemble_global_K_and_F(model, nodes_arr, elements_data, nid_to_idx, nsid_to_nids, kernels)
    U = apply_dirichlet_bc_and_solve(model, K, F, nid_to_idx, nsid_to_nids)

    # 4. Print results
    print("\nDisplacement U (all dofs):")
    # print(U)
    
    if nid_to_idx:
        max_nid = max(nid_to_idx.keys())
        max_n_idx = nid_to_idx[max_nid]
        if U.shape[0] > max_n_idx * 3 + 2:
            u_node = U[max_n_idx*3 : max_n_idx*3+3]
            print(f"\nNode {max_nid} displacement (T1, T2, T3): {u_node[0]:.6e}, {u_node[1]:.6e}, {u_node[2]:.6e}")

if __name__ == "__main__":
    main()
