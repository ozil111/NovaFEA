import argparse
import json
import re
import sys
from pathlib import Path

# Enable 64-bit precision in JAX
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit

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
    return {
        "E": float(mat["E"]),
        "nu": float(mat["nu"]),
        "rho": float(mat["rho"])
    }

# ---------------------------------------------------------------------------
# Element Calculation (Tet4)
# ---------------------------------------------------------------------------

def compute_tet4_ke_and_mass(coords, mat_params):
    """
    Computes Tet4 stiffness matrix Ke and lumped mass for one element.
    coords: (4, 3) array of node coordinates
    mat_params: dict with E, nu, rho
    """
    E, nu, rho = mat_params["E"], mat_params["nu"], mat_params["rho"]
    
    # 1. Volume and Shape Function Derivatives
    M = jnp.concatenate([jnp.ones((4, 1)), coords], axis=1)
    V = jnp.abs(jnp.linalg.det(M)) / 6.0
    invM = jnp.linalg.inv(M)
    dNdx = invM[1, :]
    dNdy = invM[2, :]
    dNdz = invM[3, :]

    # 2. B Matrix (6 x 12)
    B = jnp.zeros((6, 12))
    for i in range(4):
        B = B.at[0, i*3+0].set(dNdx[i])
        B = B.at[1, i*3+1].set(dNdy[i])
        B = B.at[2, i*3+2].set(dNdz[i])
        B = B.at[3, i*3+0].set(dNdy[i])
        B = B.at[3, i*3+1].set(dNdx[i])
        B = B.at[4, i*3+1].set(dNdz[i])
        B = B.at[4, i*3+2].set(dNdy[i])
        B = B.at[5, i*3+0].set(dNdz[i])
        B = B.at[5, i*3+2].set(dNdx[i])

    # 3. D Matrix (6 x 6)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = jnp.array([
        [lam + 2*mu, lam, lam, 0, 0, 0],
        [lam, lam + 2*mu, lam, 0, 0, 0],
        [lam, lam, lam + 2*mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu]
    ])

    # 4. Ke Matrix
    Ke = B.T @ D @ B * V
    
    # 5. Lumped Mass
    me_lumped = jnp.full((4,), rho * V / 4.0)
    
    return Ke, me_lumped

# ---------------------------------------------------------------------------
# Solver Data Preparation
# ---------------------------------------------------------------------------

def build_solver_data(model):
    nid_to_idx = {int(n["nid"]): i for i, n in enumerate(model["mesh"]["nodes"])}
    coords_list = [[float(n["x"]), float(n["y"]), float(n["z"])] for n in model["mesh"]["nodes"]]
    nodes_arr = jnp.asarray(coords_list, dtype=jnp.float64)
    num_nodes = nodes_arr.shape[0]

    # Analysis parameters
    analysis = model.get("analysis", [{}])[0]
    endtime = float(analysis.get("endtime", 1.0))
    dt = float(analysis.get("fixed_time_step", 0.001))

    # Assemble Global Stiffness K and Lumped Mass M
    K = jnp.zeros((num_nodes * 3, num_nodes * 3), dtype=jnp.float64)
    M_lumped = jnp.zeros((num_nodes, 3), dtype=jnp.float64)

    for e in model["mesh"]["elements"]:
        etype = int(e["etype"])
        if etype != 304:
            raise ValueError(f"Explicit solver currently only supports Tet4 (304), but got {etype}")
        
        nids = e["nids"]
        idx_list = [nid_to_idx[nid] for nid in nids]
        coords = nodes_arr[jnp.array(idx_list)]
        
        mat_params = _material_params_from_model(model, e["pid"])
        Ke, me_lumped = compute_tet4_ke_and_mass(coords, mat_params)
        
        # Assemble Ke into global K
        edofs = jnp.array([[idx * 3 + i for i in range(3)] for idx in idx_list]).flatten()
        K = K.at[jnp.ix_(edofs, edofs)].add(Ke)
        
        # Assemble lumped mass
        for i, idx in enumerate(idx_list):
            M_lumped = M_lumped.at[idx, :].add(me_lumped[i])

    # Boundary Conditions Mask
    # 1 where DOF is FREE, 0 where DOF is FIXED
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
                for comp in comps:
                    bc_mask = bc_mask.at[idx, comp].set(0.0)

    # External Load Vector F_ext (Base values)
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
                for comp in comps:
                    F_ext_base = F_ext_base.at[idx, comp].add(value)

    return nodes_arr, K, M_lumped, bc_mask, F_ext_base, nid_to_idx, endtime, dt

# ---------------------------------------------------------------------------
# Main Solver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="JAX explicit dynamics solver for fast validation")
    parser.add_argument("--model", required=True, help="Path to the model file (.json or .jsonc)")
    args = parser.parse_args()

    # 1. Load model and build solver data
    model = _load_json_or_jsonc(args.model)
    nodes_arr, K, M_lumped, bc_mask, F_ext_base, nid_to_idx, endtime, dt = build_solver_data(model)
    num_nodes = nodes_arr.shape[0]

    # --- 显式时间步更新函数 ---
    @jit
    def step_update(u, v_half, t):
        # 1. 内力计算
        f_int_flat = K @ u.flatten()
        f_int = f_int_flat.reshape((num_nodes, 3))
        
        # 2. 外力计算 (线性斜坡加载)
        # Note: 也可以根据需要修改为 Step 加载
        scale = t / endtime
        f_ext = F_ext_base * scale
        
        # 3. 求解加速度并施加边界条件
        # a = (F_ext - F_int) / M
        a = (f_ext - f_int) / M_lumped
        a = a * bc_mask  # Fix prescribed DOFs
        
        # 4. 显式半步积分 (Central Difference)
        v_half_new = v_half + a * dt
        u_new = u + v_half_new * dt
        return u_new, v_half_new

    # --- 主求解循环 ---
    def solve():
        num_steps = int(endtime / dt)
        u = jnp.zeros((num_nodes, 3))
        v_half = jnp.zeros((num_nodes, 3)) 
        
        print(f"Starting explicit simulation: endtime={endtime}, dt={dt}, steps={num_steps}")
        
        for s in range(num_steps):
            t = (s + 1) * dt
            u, v_half = step_update(u, v_half, t)
            
            if (s + 1) % max(1, num_steps // 10) == 0:
                # 打印一个参考点的位移，比如最后一个加载点的 Z 位移
                print(f"Step {s+1}/{num_steps}, Time {t:.3f}s")

        return u

    final_u = solve()

    # 4. Print results
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
