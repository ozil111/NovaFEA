import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 本构模型 (对应 Fortran: N3 Hyperelastic)
# ==========================================
def n3_hyperelastic_energy(F, C10, C20, C30, D1, D2, D3):
    """N3 超弹性模型的应变能密度函数"""
    J = jnp.linalg.det(F)
    C = F.T @ F
    
    # 增加微小偏置防止 J 刚好为 0 或负数时 log/幂函数 NaN
    J = jnp.clip(J, a_min=1e-6)
    
    J_minus_2_3 = J ** (-2.0 / 3.0)
    C_bar = J_minus_2_3 * C
    I1_bar = jnp.trace(C_bar)
    I1_bar_minus_3 = I1_bar - 3.0
    
    # 偏斜部分 (Deviatoric)
    W_dev = (C10 * I1_bar_minus_3 + 
             C20 * (I1_bar_minus_3 ** 2) + 
             C30 * (I1_bar_minus_3 ** 3))
    
    # 体积部分 (Volumetric)
    J_minus_1 = J - 1.0
    W_vol = (1.0 / D1) * (J_minus_1 ** 2)
    W_vol += jnp.where(D2 > 0, (1.0 / (D2 + 1e-16)) * (J_minus_1 ** 4), 0.0)
    W_vol += jnp.where(D3 > 0, (1.0 / (D3 + 1e-16)) * (J_minus_1 ** 6), 0.0)
    
    return W_dev + W_vol

# ==========================================
# 2. 单元内核与能量定义 (Total Lagrangian + Hourglass)
# ==========================================
def element_total_energy(u_elem, X_elem, B0, V0, gammas, props):
    """
    计算单个 C3D8R 单元的总能量 = 材料变形能 + 沙漏稳定能
    """
    C10, C20, C30, D1, D2, D3, k_hg = props
    
    # 1. 物理变形梯度 (在中心点单点积分)
    # x_elem = X_elem + u_elem  (8x3)
    # F = x^T * B0 (3x8 @ 8x3 = 3x3)
    x_elem = X_elem + u_elem
    F = x_elem.T @ B0
    
    # 2. 材料应变能
    W_mat = n3_hyperelastic_energy(F, C10, C20, C30, D1, D2, D3)
    
    # 3. 沙漏控制能 (Flanagan-Belytschko 形式的能量惩罚)
    # q 是沙漏模式的幅值向量 (3x4)
    q = u_elem.T @ gammas
    W_hg = 0.5 * k_hg * jnp.sum(q ** 2)
    
    # 总能量 = (应变能密度 + 沙漏能密度) * 初始体积
    return (W_mat + W_hg) * V0

# 使用 jax.grad 对位移求导，自动获得准确的节点力！完全取代了繁杂的 Jacobian 和 DMAT 推导
get_element_internal_force = jax.grad(element_total_energy, argnums=0)

def element_force_kernel(u_elem, X_elem, B0, V0, gammas, props):
    # f_int = dE/du, 我们返回 -f_int 作为右端项 RHS
    return -get_element_internal_force(u_elem, X_elem, B0, V0, gammas, props)

# ==========================================
# 3. 时间积分显式求解器
# ==========================================
@jax.jit
def explicit_step(state, _):
    (u, v, X, conn, B0_all, V0_all, gammas_all, inv_mass, bc_mask, props, dt) = state
    
    u_elems = u[conn]
    X_elems = X[conn]

    # vmap 并行计算所有单元的节点力
    f_elems = jax.vmap(element_force_kernel, in_axes=(0, 0, 0, 0, 0, None))(
        u_elems, X_elems, B0_all, V0_all, gammas_all, props
    )

    # 组装全局力向量
    global_forces = jnp.zeros_like(u)
    global_forces = global_forces.at[conn].add(f_elems)

    # a = F/m
    a = global_forces * inv_mass[:, None]
    
    # 施加边界条件 (固定节点的加速度和速度清零)
    a = a * bc_mask
    v_new = (v + a * dt) * bc_mask
    u_new = u + v_new * dt

    new_state = (u_new, v_new, X, conn, B0_all, V0_all, gammas_all, inv_mass, bc_mask, props, dt)
    return new_state, u_new

# ==========================================
# 4. 前处理与 C3D8R 几何计算
# ==========================================
def build_single_hex_mesh():
    """创建一个 1x1x1 的六面体单元"""
    X = jnp.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], # 底部 0 1 2 3
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]  # 顶部 4 5 6 7
    ])
    connectivity = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    return X, connectivity

def preprocess_hex_mesh(X, conn, rho=1.0):
    n_elems = conn.shape[0]
    n_nodes = X.shape[0]
    
    # 中心点 (-1 到 1 的等参坐标中心) 的形函数导数 dN/dxi
    xi_eta_zeta = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
    ])
    dN_dxi_0 = xi_eta_zeta / 8.0  # (8, 3)
    
    # 对应的 4 个沙漏正交模式向量 h (参考 Fortran 代码)
    h_vectors = np.array([
        [ 1, -1,  1, -1,  1, -1,  1, -1], # h1
        [ 1, -1, -1,  1, -1,  1,  1, -1], # h2
        [ 1,  1, -1, -1, -1, -1,  1,  1], # h3
        [-1,  1, -1,  1,  1, -1,  1, -1]  # h4
    ]).T # (8, 4)

    B0_list, V0_list, gammas_list = [], [], []
    mass_vec = np.zeros(n_nodes)
    
    for e in range(n_elems):
        X_e = X[conn[e]]
        
        # J0 = X^T * dN/dxi
        J0 = X_e.T @ dN_dxi_0
        detJ0 = np.linalg.det(J0)
        invJ0 = np.linalg.inv(J0)
        
        # B0 = dN/dxi * J0^-1 (求导的链式法则)
        B0 = dN_dxi_0 @ invJ0
        
        # 自然域体积为 8 (2x2x2)，故 V0 = 8 * det(J0)
        V0 = 8.0 * detJ0
        
        # 计算沙漏修正向量 gammas
        h_dot_x = X_e.T @ h_vectors      # (3, 4)
        B0_h_dot_x = B0 @ h_dot_x        # (8, 4)
        gammas = (h_vectors - B0_h_dot_x) / 8.0  # (8, 4)
        
        B0_list.append(B0)
        V0_list.append(V0)
        gammas_list.append(gammas)
        
        # 节点质量集中分配 (1/8分配)
        m_e = rho * V0
        for i in range(8):
            mass_vec[conn[e, i]] += m_e / 8.0

    return (jnp.array(B0_list), jnp.array(V0_list), jnp.array(gammas_list), jnp.array(1.0 / mass_vec))

# ==========================================
# 5. 可视化模块
# ==========================================
def plot_hex_results(trajectory, X_ref, dt, skip_step=500):
    trajectory = np.asarray(trajectory)
    X_ref = np.asarray(X_ref)
    num_steps, n_nodes, _ = trajectory.shape
    time = np.arange(num_steps) * dt

    # 六面体的 12 条边用于绘图
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0], # 底面
        [4, 5], [5, 6], [6, 7], [7, 4], # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 柱边
    ]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    frames = trajectory[::skip_step]
    
    ax.set_xlim(-0.2, 1.5); ax.set_ylim(-0.2, 1.5); ax.set_zlim(-0.2, 1.5)
    ax.set_title("C3D8R Hexahedral Element - Total Lagrangian explicit")

    scat = ax.scatter([], [], [], c="b", s=30)
    lines = [ax.plot([], [], [], "k-", alpha=0.6)[0] for _ in edges]

    def update(frame_idx):
        curr_X = X_ref + frames[frame_idx]
        ax.set_title(f"Time: {frame_idx * skip_step * dt:.4f} s")
        scat._offsets3d = (curr_X[:, 0], curr_X[:, 1], curr_X[:, 2])
        for line, edge in zip(lines, edges):
            p1, p2 = curr_X[edge[0]], curr_X[edge[1]]
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])
        return [scat] + lines

    print("Generating animation...")
    ani = FuncAnimation(fig, update, frames=len(frames), interval=30, blit=False)
    plt.show()

# ==========================================
# 主流程
# ==========================================
def main():
    jax.config.update("jax_enable_x64", True)

    X, conn = build_single_hex_mesh()
    B0_all, V0_all, gammas_all, inv_mass = preprocess_hex_mesh(X, conn, rho=1.0)

    # 边界条件：固定 X=0 的四个节点 (0, 3, 4, 7)
    bc_mask = jnp.ones((X.shape[0], 3))
    fixed_nodes = jnp.array([0, 3, 4, 7])
    bc_mask = bc_mask.at[fixed_nodes, :].set(0.0)

    # 材料参数 (对应 Fortran)
    C10, C20, C30 = 1.0, 0.0, 0.0   # Neo-Hookean 退化
    D1, D2, D3 = 1e-3, 0.0, 0.0     # 微小可压缩性
    k_hg = 0.5                      # 沙漏惩罚刚度因子
    props = jnp.array([C10, C20, C30, D1, D2, D3, k_hg])

    u0 = jnp.zeros_like(X)
    v0 = jnp.zeros_like(X)
    
    # 给 X=1 侧的节点 (1, 2, 5, 6) 施加沿 X 方向的初始拉伸速度
    pulled_nodes = jnp.array([1, 2, 5, 6])
    v0 = v0.at[pulled_nodes, 0].set(0.5)

    dt = 1e-4
    state = (u0, v0, X, conn, B0_all, V0_all, gammas_all, inv_mass, bc_mask, props, dt)
    
    num_steps = 20_000
    print("Running JAX Scan explicit solver...")
    final_state, trajectory = jax.lax.scan(explicit_step, state, None, length=num_steps)
    
    print("Simulation finished. Max Displacement X:", float(jnp.max(trajectory[:, pulled_nodes, 0])))
    
    plot_hex_results(trajectory, X, dt, skip_step=200)

if __name__ == "__main__":
    main()