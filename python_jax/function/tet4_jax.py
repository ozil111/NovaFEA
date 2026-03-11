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

# ---------------------------------------------------------------------------
# tet4_mat1 算例：从 BDF 提取的网格、材料、载荷与边界条件
# ---------------------------------------------------------------------------

# 节点坐标 (GRID 1..4)
nodes = jnp.array([
    [0.0, 0.0, 0.0],  # node 1
    [1.0, 0.0, 0.0],  # node 2
    [0.0, 1.0, 0.0],  # node 3
    [0.0, 0.0, 1.0],  # node 4
])

# 单元连接：CTETRA 1 的节点顺序为 4,1,3,2 → 0-based [3,0,2,1]
elements = jnp.array([[3, 0, 2, 1]])

# 材料 MAT1: BDF 写为 12.0+7 → 12e7；f06 中为 2E+07，若要对齐 f06 位移可取 E=2e7
E, nu = 2.0e7, 0.3

# 边界条件 SPC1: 节点 1,2,3 固定 (123456)
# 自由自由度：节点 4 的 3 个平移 → dof 9,10,11
fixed_dofs = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # 节点 1,2,3 的 x,y,z
free_dofs = jnp.array([9, 10, 11])

# 载荷 FORCE: 节点 4 上 (0, 0, 100000)
F_global = jnp.zeros(12)
F_global = F_global.at[11].set(100000.0)  # 节点 4 的 z 向


def run_tet4_mat1():
    """组装并求解 tet4_mat1 静力问题，返回位移 U 与单元刚度 K_e."""
    # 单元 1 的节点坐标 (按 BDF 单元节点顺序: 4,1,3,2)
    coords = nodes[elements[0]]
    K_e = compute_tet4_K(coords, E, nu)
    print("K_e (12x12, 列主序数组):")
    K_e_flat = jnp.asarray(K_e).flatten(order='F')
    print(K_e_flat)
    print()

    # 局部→全局 dof 映射：局部节点 0,1,2,3 → 全局节点 4,1,3,2 → dof [9,10,11], [0,1,2], [6,7,8], [3,4,5]
    loc_to_glob = jnp.array([9, 10, 11, 0, 1, 2, 6, 7, 8, 3, 4, 5])
    glob_to_loc = jnp.zeros(12, dtype=jnp.int32).at[loc_to_glob].set(jnp.arange(12))
    K_global = K_e[glob_to_loc, :][:, glob_to_loc]

    # 仅自由自由度求解: K_ff @ u_f = F_f
    K_ff = K_global[jnp.ix_(free_dofs, free_dofs)]
    F_f = F_global[free_dofs]
    u_f = jnp.linalg.solve(K_ff, F_f)

    U = jnp.zeros(12)
    U = U.at[free_dofs].set(u_f)

    return U, K_e


if __name__ == "__main__":
    # 打印 D 矩阵（按列主序）
    D = material_isotropic(E, nu)
    print("本构矩阵 D (6x6, 列主序数组):")
    D_flat = jnp.asarray(D).flatten(order='F')
    print(D_flat)
    print()
    
    U, K_e = run_tet4_mat1()
    print("Tet4 stiffness matrix K shape:", K_e.shape)
    print("\nDisplacement U (12 dof):")
    print(U)
    print("\nNode 4 displacement (T1, T2, T3):", U[9], U[10], U[11])