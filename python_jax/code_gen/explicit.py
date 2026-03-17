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
from jax import jit

# Add current dir to path for dynamic imports
sys.path.append(str(Path(__file__).parent.resolve()))

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _strip_jsonc_comments(text):
    return re.sub(r"//.*$", "", text, flags=re.MULTILINE)

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
    return jnp.array([float(mat["E"]), float(mat["nu"])]), float(mat["rho"])

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

# ---------------------------------------------------------------------------
# Solver Data Preparation
# ---------------------------------------------------------------------------

def build_solver_data(model, kernels):
    nid_to_idx = {int(n["nid"]): i for i, n in enumerate(model["mesh"]["nodes"])}
    coords_list = [[float(n["x"]), float(n["y"]), float(n["z"])] for n in model["mesh"]["nodes"]]
    nodes_arr = jnp.asarray(coords_list, dtype=jnp.float64)
    num_nodes = nodes_arr.shape[0]

    # Analysis parameters
    analysis = model.get("analysis", [{}])[0]
    endtime = float(analysis.get("endtime", 1.0))
    dt = float(analysis.get("fixed_time_step", 0.001))

    # Global K and M
    K = jnp.zeros((num_nodes * 3, num_nodes * 3), dtype=jnp.float64)
    M_lumped = jnp.zeros((num_nodes, 3), dtype=jnp.float64)

    # Pre-extract kernels
    d_kernel = kernels.get("material_D")
    
    for e in model["mesh"]["elements"]:
        etype = int(e["etype"])
        nids = e["nids"]
        idx_list = [nid_to_idx[nid] for nid in nids]
        coords = nodes_arr[jnp.array(idx_list)]
        coords_flat = coords.flatten()
        
        mat_params_D, rho = _material_params_from_model(model, e["pid"])
        d_matrix_flat = jnp.asarray(d_kernel(mat_params_D))

        Ke = None
        me_lumped_nodes = None

        if etype == 304: # Tet4
            # 1. Stiffness (using operators)
            # For Tet4, we use a single point (centriod) but operators expect GP
            # Actually for Tet4 constant strain, any point is fine. 
            # dN_dnat is constant.
            dN_dnat = jnp.asarray(kernels["tet4_op_dN_dnat"](jnp.array([0.25, 0.25, 0.25])))
            
            map_input = jnp.concatenate([coords_flat, dN_dnat])
            map_output = jnp.asarray(kernels["tet4_op_mapping"](map_input))
            dN_dx = map_output[0:12]
            detJ = map_output[12]
            
            # Assembly (weight for Tet4 is 1/6 for the whole element if using detJ of Jacobian)
            # In our sympy_codegen for Tet4 op_asm, we use Abs(detJ) * weight.
            # Volume of Tet4 = Abs(detJ)/6. So weight should be 1/6.
            asm_input = jnp.concatenate([dN_dx, d_matrix_flat, jnp.array([detJ, 1.0/6.0])])
            ke_flat = jnp.asarray(kernels["tet4_op_assembly"](asm_input))
            Ke = ke_flat.reshape(12, 12)
            
            # 2. Mass
            mass_input = jnp.concatenate([coords_flat, jnp.array([rho])])
            me_lumped_nodes = jnp.asarray(kernels["tet4_op_lumped_mass"](mass_input))

        elif etype == 308: # Hex8
            # 1. Stiffness (2x2x2 Gauss)
            gp_val = 1.0 / jnp.sqrt(3.0)
            gps = [-gp_val, gp_val]
            gauss_points = [(x, y, z) for x in gps for y in gps for z in gps]
            
            Ke = jnp.zeros((24, 24), dtype=jnp.float64)
            for pt in gauss_points:
                dN_dnat = jnp.asarray(kernels["hex8_op_dN_dnat"](jnp.array(pt)))
                map_input = jnp.concatenate([coords_flat, dN_dnat])
                map_output = jnp.asarray(kernels["hex8_op_mapping"](map_input))
                dN_dx = map_output[0:24]
                detJ = map_output[24]
                
                asm_input = jnp.concatenate([dN_dx, d_matrix_flat, jnp.array([detJ, 1.0])])
                ke_gp_flat = jnp.asarray(kernels["hex8_op_assembly"](asm_input))
                Ke = Ke + ke_gp_flat.reshape(24, 24)
            
            # 2. Mass (TODO: need hex8_op_lumped_mass)
            # For now Hex8 mass is not implemented in generators
            raise NotImplementedError("Hex8 mass operators not yet generated.")

        if Ke is not None:
            edofs = jnp.array([[idx * 3 + i for i in range(3)] for idx in idx_list]).flatten()
            K = K.at[jnp.ix_(edofs, edofs)].add(Ke)
        
        if me_lumped_nodes is not None:
            for i, idx in enumerate(idx_list):
                M_lumped = M_lumped.at[idx, :].add(me_lumped_nodes[i])

    # BC Mask and External Load (Same as before)
    bc_mask = jnp.ones((num_nodes, 3), dtype=jnp.float64)
    nsid_to_nids = {int(ns["nsid"]): [int(nid) for nid in ns["nids"]] for ns in model.get("nodeset", [])}
    for bc in model.get("boundary", []):
        nsid = int(bc["nsid"])
        s = str(bc["dof"]).lower()
        comps = []
        if "x" in s or "1" in s or "all" in s: comps.append(0)
        if "y" in s or "2" in s or "all" in s: comps.append(1)
        if "z" in s or "3" in s or "all" in s: comps.append(2)
        for nid in nsid_to_nids[nsid]:
            if nid in nid_to_idx:
                idx = nid_to_idx[nid]
                for comp in comps: bc_mask = bc_mask.at[idx, comp].set(0.0)

    F_ext_base = jnp.zeros((num_nodes, 3), dtype=jnp.float64)
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
                idx = nid_to_idx[nid]
                for comp in comps: F_ext_base = F_ext_base.at[idx, comp].add(value)

    return nodes_arr, K, M_lumped, bc_mask, F_ext_base, nid_to_idx, endtime, dt

# ---------------------------------------------------------------------------
# Main Solver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="JAX explicit dynamics solver (Kernel-based)")
    parser.add_argument("--model", required=True, help="Path to the model file")
    parser.add_argument("--element", default="tet4", help="Element type")
    parser.add_argument("--material", default="isotropic", help="Material type")
    args = parser.parse_args()

    # 1. Load generated kernels
    kernels = {}
    d_mod = f"{args.material}_D_gen"
    kernels["material_D"] = _load_kernel_func(d_mod, f"{args.material}_D")

    if args.element == "tet4":
        kernels["tet4_op_dN_dnat"] = _load_kernel_func("tet4_op_dN_dnat_gen", "tet4_op_dN_dnat")
        kernels["tet4_op_mapping"] = _load_kernel_func("tet4_op_mapping_gen", "tet4_op_mapping")
        kernels["tet4_op_assembly"] = _load_kernel_func("tet4_op_assembly_gen", "tet4_op_assembly")
        kernels["tet4_op_lumped_mass"] = _load_kernel_func("tet4_op_lumped_mass_gen", "tet4_op_lumped_mass")
    elif args.element == "hex8":
        kernels["hex8_op_dN_dnat"] = _load_kernel_func("hex8_op_dN_dnat_gen", "hex8_op_dN_dnat")
        kernels["hex8_op_mapping"] = _load_kernel_func("hex8_op_mapping_gen", "hex8_op_mapping")
        kernels["hex8_op_assembly"] = _load_kernel_func("hex8_op_assembly_gen", "hex8_op_assembly")
        # kernels["hex8_op_lumped_mass"] = ...

    # 2. Load model and build solver data
    model = _load_json_or_jsonc(args.model)
    nodes_arr, K, M_lumped, bc_mask, F_ext_base, nid_to_idx, endtime, dt = build_solver_data(model, kernels)
    num_nodes = nodes_arr.shape[0]

    @jit
    def step_update(u, v_half, t):
        f_int = (K @ u.flatten()).reshape((num_nodes, 3))
        f_ext = F_ext_base * (t / endtime)
        a = (f_ext - f_int) / M_lumped * bc_mask
        v_half_new = v_half + a * dt
        u_new = u + v_half_new * dt
        return u_new, v_half_new

    def solve():
        num_steps = int(endtime / dt)
        u, v_half = jnp.zeros((num_nodes, 3)), jnp.zeros((num_nodes, 3))
        print(f"Starting explicit simulation: {args.element}, steps={num_steps}")
        for s in range(num_steps):
            u, v_half = step_update(u, v_half, (s + 1) * dt)
            if (s + 1) % max(1, num_steps // 10) == 0:
                print(f"Step {s+1}/{num_steps}")
        return u

    final_u = solve()

    print("\nDisplacement U (all nodes):")
    sorted_nids = sorted(nid_to_idx.keys())
    print(f"{'NodeID':>8} | {'T1':>12} | {'T2':>12} | {'T3':>12}")
    print("-" * 50)
    for nid in sorted_nids:
        idx = nid_to_idx[nid]
        u_node = final_u[idx]
        print(f"{nid:8d} | {u_node[0]:12.6e} | {u_node[1]:12.6e} | {u_node[2]:12.6e}")

if __name__ == "__main__":
    main()
