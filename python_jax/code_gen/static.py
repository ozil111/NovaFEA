import argparse
import importlib
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# 小型隐式 static linear 求解器
# ---------------------------------------------------------------------------
def _strip_jsonc_comments(text):
    return re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)

def _load_json_or_jsonc(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonc":
        text = _strip_jsonc_comments(text)
    return json.loads(text)

def _resolve_model_input(input_path):
    p = Path(input_path).resolve()
    data = _load_json_or_jsonc(p)
    if isinstance(data, dict) and "mesh" in data:
        return p
    if isinstance(data, dict) and "test_cases" in data:
        tc = data["test_cases"][0]
        args = tc.get("args", [])
        model_rel = None
        for i, a in enumerate(args):
            if a == "-i" and i + 1 < len(args):
                model_rel = args[i + 1]
                break
        if model_rel is None:
            raise ValueError("test_cases.json 中未找到 -i 模型路径")
        return (p.parent / model_rel).resolve()
    raise ValueError(f"无法识别输入格式: {p}")

def _get_dof_component_indices(dof_spec):
    s = str(dof_spec).lower()
    if s == "all": return [0, 1, 2]
    if s == "x": return [0]
    if s == "y": return [1]
    if s == "z": return [2]
    out = []
    if "1" in s: out.append(0)
    if "2" in s: out.append(1)
    if "3" in s: out.append(2)
    if not out: raise ValueError(f"不支持的 dof 描述: {dof_spec}")
    return out

def _material_params_from_model(model, pid):
    prop_by_pid = {int(p["pid"]): p for p in model.get("property", [])}
    mat_by_mid = {int(m["mid"]): m for m in model.get("material", [])}
    prop = prop_by_pid[int(pid)]
    mat = mat_by_mid[int(prop["mid"])]
    # 返回参数列表，顺序必须与符号模型中一致
    return [float(mat["E"]), float(mat["nu"])]

def _make_generated_ke_from_module(module_name, element, material):
    """
    从生成器输出模块加载 compute_..._Ke(in_flat)。
    返回统一接口 ke_func(coords, E, nu)->(12,12)。
    """
    func_name = f"compute_{element}_{material}_Ke"
    mod = importlib.import_module(module_name)
    if not hasattr(mod, func_name):
        raise AttributeError(f"模块 {module_name} 中未找到 {func_name}")
    raw_func = getattr(mod, func_name)

    def ke_func(coords, *mat_params):
        coords_flat = jnp.asarray(coords).flatten(order="C")
        params_flat = jnp.asarray(mat_params)
        in_flat = jnp.concatenate([coords_flat, params_flat])
        out = raw_func(in_flat)
        
        arr = jnp.asarray(out)
        if arr.shape == (12, 12):
            return arr
        return arr.reshape(12, 12, order="C")

    return ke_func

def build_solver_data(model):
    nid_to_idx = {}
    coords_list = []
    for i, n in enumerate(model["mesh"]["nodes"]):
        nid = int(n["nid"])
        nid_to_idx[nid] = i
        coords_list.append([float(n["x"]), float(n["y"]), float(n["z"])])
    nodes_arr = jnp.asarray(coords_list, dtype=jnp.float64)

    elements_data = []
    for e in model["mesh"]["elements"]:
        if int(e["etype"]) != 304:
            raise ValueError(f"当前仅支持 Tet4(etype=304)，收到 etype={e['etype']}")
        elements_data.append({"eid": int(e["eid"]), "pid": int(e["pid"]), "nids": [int(x) for x in e["nids"]]})

    nsid_to_nids = {int(ns["nsid"]): [int(x) for x in ns["nids"]] for ns in model.get("nodeset", [])}
    return nodes_arr, elements_data, nid_to_idx, nsid_to_nids

def assemble_global_K_and_F(model, ke_func):
    """
    组装全局刚度 K 与载荷 F。
    ke_func: (coords(4x3), E, nu) -> Ke(12x12)
    """
    nodes_arr, elements_data, nid_to_idx, nsid_to_nids = build_solver_data(model)
    ndof = nodes_arr.shape[0] * 3
    K = jnp.zeros((ndof, ndof), dtype=jnp.float64)
    F = jnp.zeros((ndof,), dtype=jnp.float64)

    for e in elements_data:
        nids = e["nids"]
        coords = jnp.stack([nodes_arr[nid_to_idx[nid]] for nid in nids], axis=0)
        mat_params = _material_params_from_model(model, e["pid"])
        Ke = ke_func(coords, *mat_params)
        
        edofs = []
        for nid in nids:
            base = nid_to_idx[nid] * 3
            edofs.extend([base, base + 1, base + 2])
        edofs = jnp.asarray(edofs, dtype=jnp.int32)
        
        K = K.at[jnp.ix_(edofs, edofs)].add(Ke)

    for ld in model.get("load", []):
        nsid = int(ld["nsid"])
        value = float(ld["value"])
        comps = _get_dof_component_indices(ld["dof"])
        for nid in nsid_to_nids[nsid]:
            base = nid_to_idx[nid] * 3
            for c in comps:
                F = F.at[base + c].add(value)

    return K, F, nid_to_idx, nsid_to_nids

def apply_dirichlet_bc_and_solve(model, K, F, nid_to_idx, nsid_to_nids):
    fixed = set()
    for bc in model.get("boundary", []):
        nsid = int(bc["nsid"])
        if abs(float(bc.get("value", 0.0))) > 1e-9:
            raise ValueError("当前求解器仅支持零位移 Dirichlet 边界 (value=0)")
        comps = _get_dof_component_indices(bc["dof"])
        for nid in nsid_to_nids[nsid]:
            base = nid_to_idx[nid] * 3
            for c in comps:
                fixed.add(base + c)

    ndof = K.shape[0]
    all_dofs = jnp.arange(ndof)
    fixed_dofs = jnp.asarray(sorted(list(fixed)), dtype=jnp.int32)
    free_mask = jnp.ones(ndof, dtype=bool).at[fixed_dofs].set(False)
    free_dofs = all_dofs[free_mask]

    K_ff = K[jnp.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    u_f = jnp.linalg.solve(K_ff, F_f)

    U = jnp.zeros(ndof, dtype=jnp.float64)
    U = U.at[free_dofs].set(u_f)
    return U

def solve_static_linear(model_path, element, material):
    """
    加载模型，使用生成的JAX内核进行求解，并返回位移。
    """
    model_file = _resolve_model_input(model_path)
    model = _load_json_or_jsonc(model_file)

    # 始终使用生成的内核
    module_name = f"{element}_{material}_Ke_gen"
    ke_func = _make_generated_ke_from_module(module_name, element, material)

    K, F, nid_to_idx, nsid_to_nids = assemble_global_K_and_F(model, ke_func)
    U = apply_dirichlet_bc_and_solve(model, K, F, nid_to_idx, nsid_to_nids)
    return U

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX 求解器，使用由 sympy_codegen.py 生成的内核")
    parser.add_argument(
        "--model",
        default=str((Path(__file__).resolve().parents[2] / "test_case" / "tet4_mat1_im" / "tet4_mat1_im.jsonc")),
        help="模型文件路径（json/jsonc），或 test_case/test_cases.json",
    )
    parser.add_argument("--element", default="tet4", help="单元类型 (e.g., tet4)")
    parser.add_argument("--material", default="isotropic", help="材料类型 (e.g., isotropic)")
    
    args = parser.parse_args()

    U = solve_static_linear(
        model_path=args.model,
        element=args.element,
        material=args.material,
    )
    print("Displacement U (all dof):")
    print(U)
    if U.shape[0] >= 12:
        print("Node 4 displacement (T1, T2, T3):", U[9], U[10], U[11])
