import argparse
import importlib
import json
import re
from pathlib import Path

# Enable 64-bit precision in JAX
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

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
        if int(e["etype"]) != 304:
            raise ValueError(f"This solver currently only supports Tet4 (etype=304), but got {e['etype']}")
        elements_data.append({"pid": int(e["pid"]), "nids": [int(nid) for nid in e["nids"]]})

    nsid_to_nids = {int(ns["nsid"]): [int(nid) for nid in ns["nids"]] for ns in model.get("nodeset", [])}
    return nodes_arr, elements_data, nid_to_idx, nsid_to_nids

def assemble_global_K_and_F(model, nodes_arr, elements_data, nid_to_idx, nsid_to_nids, d_kernel, ke_kernel):
    """
    Assembles the global stiffness matrix K and load vector F.
    This function implements the hybrid decoupled workflow in JAX.
    """
    ndof = nodes_arr.shape[0] * 3
    K = jnp.zeros((ndof, ndof), dtype=jnp.float64)
    F = jnp.zeros((ndof,), dtype=jnp.float64)

    for e in elements_data:
        # 1. Get data for the current element
        nids = e["nids"]
        coords = jnp.stack([nodes_arr[nid_to_idx[nid]] for nid in nids])
        mat_params = _material_params_from_model(model, e["pid"])

        # 2. Call the material kernel to get the D-matrix
        d_matrix_flat = jnp.asarray(d_kernel(mat_params))

        # 3. Prepare inputs and call the stiffness kernel to get Ke
        stiffness_kernel_input = jnp.concatenate([coords.flatten(), d_matrix_flat])
        ke_flat = jnp.asarray(ke_kernel(stiffness_kernel_input))
        Ke = ke_flat.reshape(12, 12)

        # 4. Assemble Ke into the global K matrix
        edofs = jnp.array([[nid_to_idx[nid] * 3 + i for i in range(3)] for nid in nids]).flatten()
        K = K.at[jnp.ix_(edofs, edofs)].add(Ke)

    # Assemble global load vector F
    for ld in model.get("load", []):
        nsid = int(ld["nsid"])
        value = float(ld["value"])
        # Simple parsing for dof "x", "y", "z", "all"
        s = str(ld["dof"]).lower()
        comps = []
        if "x" in s or "1" in s or "all" in s: comps.append(0)
        if "y" in s or "2" in s or "all" in s: comps.append(1)
        if "z" in s or "3" in s or "all" in s: comps.append(2)
        
        for nid in nsid_to_nids[nsid]:
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
            base_dof = nid_to_idx[nid] * 3
            for comp in comps:
                fixed_dofs.add(base_dof + comp)

    ndof = K.shape[0]
    # Correctly calculate free DOFs using standard Python sets
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
    parser = argparse.ArgumentParser(description="JAX solver using decoupled kernels from sympy_codegen.py")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model file (e.g., tet4_mat1_im.jsonc)",
    )
    parser.add_argument("--element", default="tet4", help="Element type (e.g., tet4)")
    parser.add_argument("--material", default="isotropic", help="Material type (e.g., isotropic)")
    
    args = parser.parse_args()

    # 1. Load generated kernels
    d_kernel_module_name = f"{args.material}_D_gen"
    d_kernel_func_prefix = f"{args.material}_D"
    d_kernel = _load_kernel_func(d_kernel_module_name, d_kernel_func_prefix)

    ke_kernel_module_name = f"{args.element}_Ke_gen"
    ke_kernel_func_prefix = f"{args.element}_Ke"
    ke_kernel = _load_kernel_func(ke_kernel_module_name, ke_kernel_func_prefix)

    print("Successfully loaded generated kernels.")

    # 2. Load model and build solver data
    model = _load_json_or_jsonc(args.model)
    nodes_arr, elements_data, nid_to_idx, nsid_to_nids = build_solver_data(model)

    # 3. Assemble and solve
    K, F = assemble_global_K_and_F(model, nodes_arr, elements_data, nid_to_idx, nsid_to_nids, d_kernel, ke_kernel)
    U = apply_dirichlet_bc_and_solve(model, K, F, nid_to_idx, nsid_to_nids)

    # 4. Print results
    print("\nDisplacement U (all dofs):")
    print(U)
    
    # Find a node to print for easy comparison, e.g., the highest node id
    if nid_to_idx:
        max_nid = max(nid_to_idx.keys())
        max_n_idx = nid_to_idx[max_nid]
        if U.shape[0] > max_n_idx * 3 + 2:
            u_node = U[max_n_idx*3 : max_n_idx*3+3]
            print(f"\nNode {max_nid} displacement (T1, T2, T3): {u_node[0]:.6f}, {u_node[1]:.6f}, {u_node[2]:.6f}")

if __name__ == "__main__":
    main()
