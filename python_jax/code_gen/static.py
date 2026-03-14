import argparse
import importlib
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import jacfwd

# 1. 定义母单元形函数 (Natural Coordinates: xi, eta, zeta)
def get_N(xi, eta, zeta):
    # 对于 Tet4，形函数极其简单
    return jnp.array([
        1 - xi - eta - zeta,
        xi,
        eta,
        zeta
    ])

# 2. 自动计算形函数对天然坐标的梯度 (dN/d_xi)
# jacfwd 会生成一个函数，返回 dN/d_xi, dN/d_eta, dN/d_zeta
dN_dxi_func = jacfwd(get_N, argnums=(0, 1, 2))

# ---------------------------------------------------------------------------
# 材料本构（与单元运动学解耦）：返回 D 矩阵 (6x6 Voigt)
# ---------------------------------------------------------------------------
def material_isotropic(E, nu):
    """线性各向同性弹性：get_constitutive_matrix(params) 形式，返回 D。"""
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
    return D


def compute_tet4_K_decoupled(coords, D):
    """
    单元刚度：仅几何（B、体积），接受本构矩阵 D 作为输入。
    coords: 单元节点坐标 (4x3)
    D: 本构矩阵 (6x6)，由材料模型提供，如 material_isotropic(E, nu)
    """
    # 由于 Tet4 是线性单元，梯度是常数，我们直接在 (0,0,0) 处计算
    # dN_dxi_raw 的 shape 为 (4, 3)
    grads = dN_dxi_func(0.0, 0.0, 0.0)
    dN_dxi = jnp.stack(grads, axis=1) 

    # 计算雅可比矩阵 J = dx/d_xi = coords^T * dN/d_xi
    # J 的 shape 为 (3, 3)
    J = coords.T @ dN_dxi
    
    # 单元体积 V = 1/6 * det(J)
    detJ = jnp.linalg.det(J)
    vol = jnp.abs(detJ) / 6.0

    # 计算全局导数 dN/dx = dN/dxi * J^-1
    # B 矩阵组装的核心
    invJ = jnp.linalg.inv(J)
    dN_dx = dN_dxi @ invJ  # (4, 3)

    # 组装 B 矩阵 (6 x 12)
    B = jnp.zeros((6, 12))
    for i in range(4):
        # 对应节点 i 的三个分量
        B = B.at[0, 3*i].set(dN_dx[i, 0])
        B = B.at[1, 3*i+1].set(dN_dx[i, 1])
        B = B.at[2, 3*i+2].set(dN_dx[i, 2])
        B = B.at[3, 3*i].set(dN_dx[i, 1])
        B = B.at[3, 3*i+1].set(dN_dx[i, 0])
        B = B.at[4, 3*i+1].set(dN_dx[i, 2])
        B = B.at[4, 3*i+2].set(dN_dx[i, 1])
        B = B.at[5, 3*i].set(dN_dx[i, 2])
        B = B.at[5, 3*i+2].set(dN_dx[i, 0])

    # 计算刚度矩阵 K = B^T * D * B * Vol
    K = B.T @ D @ B * vol
    return K


def compute_tet4_K(coords, E, nu):
    """
    兼容接口：coords + 材料参数 E, nu，内部使用 material_isotropic 与 compute_tet4_K_decoupled。
    """
    D = material_isotropic(E, nu)
    print("coords:")
    print(coords)
    return compute_tet4_K_decoupled(coords, D)


def verify_tet4_Ke(coords, D, generated_ke_func):
    """用 compute_tet4_K_decoupled 与生成器模块的 Ke 函数对比验证。"""
    K_decoupled = compute_tet4_K_decoupled(coords, D)
    Ke_gen = generated_ke_func(coords, D)
    diff = jnp.abs(K_decoupled - Ke_gen)
    max_diff = jnp.max(diff)
    return max_diff, K_decoupled, Ke_gen


# ---------------------------------------------------------------------------
# 小型隐式 static linear 求解器（Tet4）
# 输入可为：
# 1) 直接模型 json/jsonc（如 tet4_mat1_im.jsonc）
# 2) test_case/test_cases.json（自动解析 -i 指向的模型路径）
# ---------------------------------------------------------------------------
def _strip_jsonc_comments(text):
    """移除 // 行注释，满足当前用例的 jsonc 解析。"""
    return re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)


def _load_json_or_jsonc(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonc":
        text = _strip_jsonc_comments(text)
    return json.loads(text)


def _resolve_model_input(input_path):
    """
    支持直接传模型 json/jsonc，或传 test_cases.json（自动提取 -i 参数）。
    返回模型文件的绝对路径。
    """
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
    if s == "all":
        return [0, 1, 2]
    if s == "x":
        return [0]
    if s == "y":
        return [1]
    if s == "z":
        return [2]
    # 兼容 "123"/"12" 这种表达
    out = []
    if "1" in s:
        out.append(0)
    if "2" in s:
        out.append(1)
    if "3" in s:
        out.append(2)
    if not out:
        raise ValueError(f"不支持的 dof 描述: {dof_spec}")
    return out


def _material_D_from_model(model, pid):
    prop_by_pid = {int(p["pid"]): p for p in model.get("property", [])}
    mat_by_mid = {int(m["mid"]): m for m in model.get("material", [])}

    prop = prop_by_pid[int(pid)]
    mat = mat_by_mid[int(prop["mid"])]
    return material_isotropic(float(mat["E"]), float(mat["nu"]))


def _make_generated_ke_from_module(module_name):
    """
    从生成器输出模块加载 compute_tet4_Ke(in_flat)。
    返回统一接口 ke_func(coords, D)->(12,12)。
    """
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "compute_tet4_Ke"):
        raise AttributeError(f"模块 {module_name} 中未找到 compute_tet4_Ke")
    raw = getattr(mod, "compute_tet4_Ke")

    def ke_func(coords, D):
        coords_flat = jnp.asarray(coords).flatten(order="C")
        D_flat = jnp.asarray(D).flatten(order="C")
        in_flat = jnp.concatenate([coords_flat, D_flat])
        out = raw(in_flat)
        # 兼容 tuple/list 输出（sympy 直接生成）和数组输出（本文件内嵌版本）
        if isinstance(out, (tuple, list)):
            return jnp.asarray(out).reshape(12, 12, order="C")
        arr = jnp.asarray(out)
        if arr.shape == (12, 12):
            return arr
        return arr.reshape(12, 12, order="C")

    return ke_func


def build_solver_data(model):
    """把模型字典转换为求解所需的数据结构。"""
    nid_to_idx = {}
    coords_list = []
    for i, n in enumerate(model["mesh"]["nodes"]):
        nid = int(n["nid"])
        nid_to_idx[nid] = i
        coords_list.append([float(n["x"]), float(n["y"]), float(n["z"])])
    nodes_arr = jnp.asarray(coords_list, dtype=jnp.float32)

    elements_data = []
    for e in model["mesh"]["elements"]:
        if int(e["etype"]) != 304:
            raise ValueError(f"当前仅支持 Tet4(etype=304)，收到 etype={e['etype']}")
        nids = [int(x) for x in e["nids"]]
        elements_data.append({"eid": int(e["eid"]), "pid": int(e["pid"]), "nids": nids})

    nsid_to_nids = {int(ns["nsid"]): [int(x) for x in ns["nids"]] for ns in model.get("nodeset", [])}
    return nodes_arr, elements_data, nid_to_idx, nsid_to_nids


def assemble_global_K_and_F(model, ke_func):
    """
    组装全局刚度 K 与载荷 F。
    ke_func: (coords(4x3), D(6x6)) -> Ke(12x12)
    """
    nodes_arr, elements_data, nid_to_idx, nsid_to_nids = build_solver_data(model)
    ndof = nodes_arr.shape[0] * 3
    K = jnp.zeros((ndof, ndof), dtype=jnp.float32)
    F = jnp.zeros((ndof,), dtype=jnp.float32)

    for e in elements_data:
        nids = e["nids"]
        coords = jnp.stack([nodes_arr[nid_to_idx[nid]] for nid in nids], axis=0)
        D = _material_D_from_model(model, e["pid"])
        Ke = ke_func(coords, D)
        edofs = []
        for nid in nids:
            base = nid_to_idx[nid] * 3
            edofs.extend([base, base + 1, base + 2])
        edofs = jnp.asarray(edofs, dtype=jnp.int32)
        for a in range(12):
            ia = edofs[a]
            for b in range(12):
                ib = edofs[b]
                K = K.at[ia, ib].add(Ke[a, b])

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
        value = float(bc.get("value", 0.0))
        if abs(value) > 0.0:
            raise ValueError("当前求解器仅支持零位移 Dirichlet 边界 (value=0)")
        comps = _get_dof_component_indices(bc["dof"])
        for nid in nsid_to_nids[nsid]:
            base = nid_to_idx[nid] * 3
            for c in comps:
                fixed.add(base + c)

    ndof = K.shape[0]
    all_dofs = jnp.arange(ndof, dtype=jnp.int32)
    fixed_dofs = jnp.asarray(sorted(fixed), dtype=jnp.int32)
    free_mask = jnp.ones((ndof,), dtype=bool).at[fixed_dofs].set(False)
    free_dofs = all_dofs[free_mask]

    K_ff = K[jnp.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    u_f = jnp.linalg.solve(K_ff, F_f)

    U = jnp.zeros((ndof,), dtype=jnp.float32)
    U = U.at[free_dofs].set(u_f)
    return U


def solve_static_linear(model_path, kernel_mode="generated", generated_module="tet4_gen"):
    """
    输入模型，输出全局位移向量 U。
    kernel_mode:
      - generated: 调用生成器输出模块（如 tet4_gen.compute_tet4_Ke）
      - decoupled: 调用内置人类可读版 compute_tet4_K_decoupled
    """
    model_file = _resolve_model_input(model_path)
    model = _load_json_or_jsonc(model_file)

    if kernel_mode == "generated":
        ke_func = _make_generated_ke_from_module(generated_module)
    elif kernel_mode == "decoupled":
        ke_func = compute_tet4_K_decoupled
    else:
        raise ValueError(f"未知 kernel_mode: {kernel_mode}")

    K, F, nid_to_idx, nsid_to_nids = assemble_global_K_and_F(model, ke_func)
    U = apply_dirichlet_bc_and_solve(model, K, F, nid_to_idx, nsid_to_nids)
    return U


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tet4 JAX 验证与小型 static linear 求解器")
    parser.add_argument(
        "--model",
        default=str((Path(__file__).resolve().parents[2] / "test_case" / "tet4_mat1_im" / "tet4_mat1_im.jsonc")),
        help="模型文件路径（json/jsonc），或 test_case/test_cases.json",
    )
    parser.add_argument(
        "--kernel",
        choices=["generated", "decoupled"],
        default="generated",
        help="单元核来源：generated(外部tet4_gen.py) / decoupled(内置人类可读核)",
    )
    parser.add_argument(
        "--generated-module",
        default="tet4_gen",
        help="当 --kernel=generated 时，Python 模块名（默认 tet4_gen）",
    )
    parser.add_argument(
        "--verify-ke",
        action="store_true",
        help="先在内置 tet4_mat1 上做 Ke 一致性验证（生成核 vs 解析核）",
    )
    args = parser.parse_args()

    if args.verify_ke:
        model_file = _resolve_model_input(args.model)
        model = _load_json_or_jsonc(model_file)
        nodes_arr, elements_data, nid_to_idx, _ = build_solver_data(model)
        if not elements_data:
            raise ValueError("模型中无单元，无法做 Ke 验证")
        e = elements_data[0]
        coords = jnp.stack([nodes_arr[nid_to_idx[nid]] for nid in e["nids"]], axis=0)
        D = _material_D_from_model(model, e["pid"])
        gen_ke = _make_generated_ke_from_module(args.generated_module)
        max_diff, _, _ = verify_tet4_Ke(coords, D, gen_ke)
        print("--- 生成器验证 (tet4_gen vs compute_tet4_K_decoupled) ---")
        print("max |K_decoupled - K_gen| =", float(max_diff))
        print("验证通过" if float(max_diff) < 1e-5 else "验证未通过")
        print()

    U = solve_static_linear(
        model_path=args.model,
        kernel_mode=args.kernel,
        generated_module=args.generated_module,
    )
    print("Displacement U (all dof):")
    print(U)
    if U.shape[0] >= 12:
        print("Node 4 displacement (T1, T2, T3):", U[9], U[10], U[11])