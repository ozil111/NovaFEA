"""
Unified FEM Demo: Neo-Hookean + PANN
统一材料参数结构，通过 model_type 和 jax.lax.cond 分发
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 0. 全局配置
# ==========================================
jax.config.update("jax_enable_x64", True)

# 实验数据 (PANN 训练用)
uniaxial_data = {
    'stress': jnp.array([1.5506E5, 2.4367E5, 3.1013E5, 4.2089E5, 5.3165E5, 5.9810E5, 6.8671E5, 8.8608E5, 10.6329E5, 12.4051E5, 16.1709E5, 19.9367E5, 23.4810E5, 27.4684E5, 31.0127E5, 34.5570E5, 38.3228E5, 42.0886E5, 45.6329E5, 49.3987E5, 53.1646E5, 56.9304E5, 64.2405E5]),
    'strain': jnp.array([0.1338, 0.2675, 0.3567, 0.6242, 0.8917, 1.1592, 1.4268, 2.0510, 2.5860, 3.0318, 3.7898, 4.3694, 4.8153, 5.1720, 5.4395, 5.7070, 5.9299, 6.0637, 6.1975, 6.3312, 6.4650, 6.5541, 6.6433])
}
biaxial_data = {
    'stress': jnp.array([0.9384E5, 1.5900E5, 2.4087E5, 2.6220E5, 3.3240E5, 4.4278E5, 5.1830E5, 6.6024E5, 7.7794E5, 9.7857E5, 12.6351E5, 14.6804E5, 17.4000E5, 20.1058E5, 22.4502E5, 24.6530E5]),
    'strain': jnp.array([0.0200, 0.0600, 0.1100, 0.1400, 0.2000, 0.3100, 0.4200, 0.6800, 0.9400, 1.4900, 2.0300, 2.4300, 2.7500, 3.0700, 3.2600, 3.4500])
}
planar_data = {
    'stress': jnp.array([0.6000E5, 1.6000E5, 2.4000E5, 3.3600E5, 4.2000E5, 6.0000E5, 7.8000E5, 9.6000E5, 11.1200E5, 12.9600E5, 14.8800E5, 16.5800E5, 18.2000E5]),
    'strain': jnp.array([0.0690, 0.1034, 0.1724, 0.2828, 0.4276, 0.8483, 1.3862, 2.0000, 2.4897, 3.0345, 3.4483, 3.7793, 4.0621])
}
volumetric_data = {
    'pressure': jnp.array([60.E5, 118.2E5, 175.2E5, 231.1E5]),
    'j': jnp.array([0.9703, 0.9412, 0.9127, 0.8847])
}

STRESS_SCALE = 1.0e5  # PANN 归一化缩放

# ==========================================
# 1. 统一材料参数结构
# ==========================================

def make_mat_params_neo(mu=1.0, bulk=10.0, scale=1.0):
    """Neo-Hookean 材料参数。含 PANN 占位字段以满足 jax.lax.cond 两分支都可追踪。"""
    return {
        'model_type': 0,
        'mu': mu,
        'bulk': bulk,
        'scale': scale,
        'layers': [],  # 占位，neo 分支不访问
        'k_vol': jnp.array([0.0]),
        'normalization_constant': jnp.array([0.0])
    }

def make_mat_params_pann(layers, k_vol, normalization_constant, scale=1.0e5):
    """PANN 材料参数。含 Neo-Hookean 占位字段以满足 jax.lax.cond 两分支都可追踪。"""
    return {
        'model_type': 1,
        'mu': 0.0,  # 占位，pann 分支不访问
        'bulk': 0.0,
        'layers': layers,
        'k_vol': jnp.asarray(k_vol) if not isinstance(k_vol, jnp.ndarray) else k_vol,
        'normalization_constant': jnp.asarray(normalization_constant) if not isinstance(normalization_constant, jnp.ndarray) else normalization_constant,
        'scale': scale
    }

# ==========================================
# 2. PANN 模型组件
# ==========================================

def init_pann_params(key, hidden_dims=[32, 32], input_dim=4):
    layers = []
    dims = [input_dim] + hidden_dims + [1]
    for i in range(len(dims) - 1):
        key, k1 = jax.random.split(key)
        lim = jnp.sqrt(6.0 / (dims[i] + dims[i+1]))
        w = jax.random.uniform(k1, (dims[i+1], dims[i]), minval=0.0, maxval=lim)
        b = jnp.zeros((dims[i+1],))
        layers.append((w, b))
    return {
        'layers': layers,
        'k_vol': jnp.array([10.0]),
        'normalization_constant': jnp.array([0.0])
    }

def compute_invariants(F):
    C = F.T @ F
    I1 = jnp.trace(C)
    C2 = C @ C
    I2 = 0.5 * (I1**2 - jnp.trace(C2))
    I3 = jnp.linalg.det(C)
    I3 = jnp.maximum(I3, 1e-10)
    J = jnp.sqrt(I3)
    I1_star = -2.0 * J
    return jnp.stack([I1, I2, I3, I1_star]), J

def icnn_forward(params, x):
    layers = params['layers']
    for w, b in layers:
        w_pos = jnp.maximum(w, 0.0)
        x = x @ w_pos.T + b
        x = jax.nn.softplus(x)
    return x[0]

def _pann_energy_inner(F, params):
    """PANN 能量 (归一化空间)，供 get_energy 调用"""
    inputs, J = compute_invariants(F)
    psi_nn = icnn_forward(params, inputs)
    k_vol = params['k_vol']
    J_term = J + 1.0/J - 2.0
    psi_growth = k_vol * (J_term**2)
    n_norm = params['normalization_constant']
    psi_norm = -n_norm * (J - 1.0)
    return jnp.squeeze(psi_nn + psi_growth + psi_norm)

# ==========================================
# 3. 统一应力内核 (jax.lax.cond 分发)
# ==========================================

def get_energy(F, mat_params):
    """统一能量泛函接口"""

    def neo_hookean_case(F):
        J = jnp.linalg.det(F)
        I1 = jnp.trace(F.T @ F)
        mu, bulk = mat_params['mu'], mat_params['bulk']
        return 0.5 * mu * (I1 - 3.0) - mu * jnp.log(J) + 0.5 * bulk * (J - 1.0)**2

    def pann_case(F):
        return _pann_energy_inner(F, mat_params)

    return jax.lax.cond(
        mat_params['model_type'] == 0,
        neo_hookean_case,
        pann_case,
        F
    )

def compute_pk1_stress(F, mat_params):
    """统一 PK1 应力计算"""
    P_unit = jax.grad(get_energy, argnums=0)(F, mat_params)
    return P_unit * mat_params['scale']

# ==========================================
# 4. 单元力核
# ==========================================

def element_force_kernel(u_elem, X_elem, Dm_inv, vol, mat_params):
    x_elem = X_elem + u_elem
    Ds = (x_elem[1:] - x_elem[0]).T
    F = Ds @ Dm_inv
    P = compute_pk1_stress(F, mat_params)
    H = vol * Dm_inv.T
    f_nodes_123 = P @ H
    f_node_0 = -jnp.sum(f_nodes_123, axis=1)
    return -jnp.vstack([f_node_0, f_nodes_123.T])

# ==========================================
# 5. 时间积分
# ==========================================

@jax.jit
def explicit_step(state, _):
    (u, v, X, connectivity, Dm_inv, vols, inv_mass_vec, bc_mask, mat_params, step_dt) = state
    u_elems = u[connectivity]
    X_elems = X[connectivity]

    f_elems = jax.vmap(element_force_kernel, in_axes=(0, 0, 0, 0, None))(
        u_elems, X_elems, Dm_inv, vols, mat_params
    )

    global_forces = jnp.zeros_like(u)
    global_forces = global_forces.at[connectivity].add(f_elems)

    a = global_forces * inv_mass_vec[:, None]
    a = a * bc_mask
    v_new = v + a * step_dt
    v_new = v_new * bc_mask
    u_new = u + v_new * step_dt

    return (u_new, v_new, X, connectivity, Dm_inv, vols, inv_mass_vec, bc_mask, mat_params, step_dt), u_new

# ==========================================
# 6. 网格与前处理
# ==========================================

def build_single_tet_mesh():
    X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    connectivity = jnp.array([[0, 1, 2, 3]])
    return X, connectivity

def preprocess_mesh(X, connectivity, rho=1.0):
    X_np = np.array(X)
    conn_np = np.array(connectivity)
    n_nodes = X_np.shape[0]
    n_elems = conn_np.shape[0]
    Dm_inv_list = []
    vol_list = []
    for e in range(n_elems):
        X_e = X_np[conn_np[e]]
        Dm = (X_e[1:] - X_e[0]).T
        vol = np.linalg.det(Dm) / 6.0
        Dm_inv_list.append(np.linalg.inv(Dm))
        vol_list.append(vol)
    Dm_inv = jnp.array(np.stack(Dm_inv_list))
    vols = jnp.array(vol_list)
    mass_vec = np.zeros(n_nodes)
    for e in range(n_elems):
        m_e = rho * vol_list[e]
        for i in range(4):
            mass_vec[conn_np[e, i]] += m_e / 4.0
    return Dm_inv, vols, jnp.array(1.0 / mass_vec)

def run_simulation(initial_state, num_steps):
    _, trajectory = jax.lax.scan(explicit_step, initial_state, None, length=num_steps)
    return trajectory

# ==========================================
# 7. PANN 训练
# ==========================================

def update_normalization(params):
    def stress_at_ref(p):
        F_ref = jnp.eye(3)
        temp = {**p, 'normalization_constant': jnp.array([0.0])}
        P = jax.grad(_pann_energy_inner, argnums=0)(F_ref, temp)
        return jnp.trace(P) / 3.0
    n_new = stress_at_ref(params)
    return {**params, 'normalization_constant': jnp.array([n_new])}

def prepare_data():
    F_list, Target_list, Type_list = [], [], []

    def add_data(stress, strain, type_id):
        for s, e in zip(stress, strain):
            if type_id == 1:
                lam = 1.0 + e
                F = jnp.diag(jnp.array([lam, lam**-0.5, lam**-0.5]))
            elif type_id == 2:
                lam = 1.0 + e
                F = jnp.diag(jnp.array([lam, lam, lam**-2]))
            elif type_id == 3:
                lam = 1.0 + e
                F = jnp.diag(jnp.array([lam, 1.0, lam**-1]))
            F_list.append(F)
            Target_list.append(s / STRESS_SCALE)
            Type_list.append(float(type_id))

    add_data(uniaxial_data['stress'], uniaxial_data['strain'], 1)
    add_data(biaxial_data['stress'], biaxial_data['strain'], 2)
    add_data(planar_data['stress'], planar_data['strain'], 3)
    for p, j_val in zip(volumetric_data['pressure'], volumetric_data['j']):
        lam = j_val**(1.0/3.0)
        F = jnp.diag(jnp.array([lam, lam, lam]))
        F_list.append(F)
        Target_list.append(p / STRESS_SCALE)
        Type_list.append(4.0)
    return jnp.stack(F_list), jnp.stack(Target_list), jnp.stack(Type_list)

F_train, S_train, Type_train = prepare_data()

@jax.jit
def loss_fn(params, F_batch, Target_batch, Type_batch):
    P_preds = jax.vmap(lambda F, p: jax.grad(_pann_energy_inner, argnums=0)(F, p), in_axes=(0, None))(F_batch, params)

    def get_pred_val(P, F, type_id):
        val_stress = P[0, 0]
        lam = F[0, 0]
        J = jnp.linalg.det(F)
        sigma_11 = P[0, 0] * lam / J
        val_pressure = -sigma_11
        return jax.lax.cond(type_id == 4.0, lambda: val_pressure, lambda: val_stress)

    preds = jax.vmap(get_pred_val)(P_preds, F_batch, Type_batch)
    return jnp.mean((preds - Target_batch)**2)

def train_pann_model(steps=5000, learning_rate=0.01):
    key = jax.random.PRNGKey(42)
    params = init_pann_params(key)
    params = update_normalization(params)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    grad_fn = jax.value_and_grad(loss_fn)
    print("\nStart Training PANN...")
    for step in range(steps):
        if step % 100 == 0:
            params = update_normalization(params)
        loss_val, grads = grad_fn(params, F_train, S_train, Type_train)
        m = jax.tree_util.tree_map(lambda m_i, g_i: beta1 * m_i + (1 - beta1) * g_i, m, grads)
        v = jax.tree_util.tree_map(lambda v_i, g_i: beta2 * v_i + (1 - beta2) * (g_i**2), v, grads)
        m_hat = jax.tree_util.tree_map(lambda m_i: m_i / (1 - beta1**(step + 1)), m)
        v_hat = jax.tree_util.tree_map(lambda v_i: v_i / (1 - beta2**(step + 1)), v)
        params = jax.tree_util.tree_map(
            lambda p_i, m_h, v_h: p_i - learning_rate * m_h / (jnp.sqrt(v_h) + eps),
            params, m_hat, v_hat
        )
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss_val:.6f}")
    params = update_normalization(params)
    return params

def save_params(params, filename="pann_params.pkl"):
    params_np = jax.tree_util.tree_map(lambda x: np.array(x), params)
    with open(filename, "wb") as f:
        pickle.dump(params_np, f)
    print(f"Saved to {filename}")

def load_params(filename="pann_params.pkl"):
    with open(filename, "rb") as f:
        params_np = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params_np)

# ==========================================
# 8. 可视化
# ==========================================

def plot_pann_fittings(params, scale):
    """PANN 拟合效果"""
    plt.figure(figsize=(15, 10))
    for idx, (name, data, F_fn) in enumerate([
        ("Uniaxial", uniaxial_data, lambda e: jnp.diag(jnp.array([1+e, (1+e)**-0.5, (1+e)**-0.5]))),
        ("Biaxial", biaxial_data, lambda e: jnp.diag(jnp.array([1+e, 1+e, (1+e)**-2]))),
        ("Planar", planar_data, lambda e: jnp.diag(jnp.array([1+e, 1.0, (1+e)**-1]))),
    ]):
        plt.subplot(2, 2, idx + 1)
        plt.plot(data['strain'], data['stress'], 'ko', label='Exp')
        e_range = jnp.linspace(0, float(data['strain'].max()), 50)
        s_pred = [float(jax.grad(_pann_energy_inner, 0)(F_fn(e), params)[0, 0] * scale) for e in e_range]
        plt.plot(e_range, s_pred, 'r-', linewidth=2, label='PANN')
        plt.title(name)
        plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 4)
    plt.plot(volumetric_data['j'], volumetric_data['pressure'], 'ko', label='Exp')
    j_range = jnp.linspace(float(volumetric_data['j'].min()), 1.0, 50)
    p_pred = []
    for j_val in j_range:
        F = jnp.diag(jnp.array([j_val**(1/3)]*3))
        P = jax.grad(_pann_energy_inner, 0)(F, params) * scale
        p_pred.append(-float(P[0, 0] * j_val**(1/3) / j_val))
    plt.plot(j_range, p_pred, 'r-', linewidth=2)
    plt.title("Volumetric")
    plt.xlabel("J")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_results(trajectory, X_ref, dt, skip_step=500):
    trajectory = np.asarray(trajectory)
    X_ref = np.asarray(X_ref)
    num_steps, n_nodes, _ = trajectory.shape
    time = np.arange(num_steps) * dt

    plt.figure(figsize=(10, 4))
    disp_x = trajectory[:, 1, 0]
    plt.plot(time, disp_x, label="Node 1 (X-displacement)")
    plt.axhline(0, color="k", linestyle="--", alpha=0.3)
    plt.title(f"Node 1 (Max Disp: {disp_x.max():.4f})")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement X")
    plt.grid(True)
    plt.legend()
    plt.savefig("displacement_curve.png")
    print("Saved displacement_curve.png")
    plt.close()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    frames = trajectory[::skip_step]
    edges = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]
    lines = [ax.plot([], [], [], "k-", alpha=0.5)[0] for _ in edges]
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.2, 1.2)
    ax.set_zlim(-0.2, 1.2)
    scat = ax.scatter([], [], [], c="b", s=50)
    scat_node1 = ax.scatter([], [], [], c="r", s=80, marker="*")

    def update(frame_idx):
        curr_X = X_ref + frames[frame_idx]
        ax.set_title(f"Time: {frame_idx * skip_step * dt:.3f} s")
        scat._offsets3d = (curr_X[:, 0], curr_X[:, 1], curr_X[:, 2])
        scat_node1._offsets3d = ([curr_X[1, 0]], [curr_X[1, 1]], [curr_X[1, 2]])
        for line, (i, j) in zip(lines, edges):
            p1, p2 = curr_X[i], curr_X[j]
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])
        return [scat, scat_node1] + lines

    ani = FuncAnimation(fig, update, frames=len(frames), interval=30, blit=False)
    plt.show()

# ==========================================
# 9. Main
# ==========================================

def main():
    # 选择材料: 'neo' 或 'pann'
    MATERIAL = 'pann'  # 可改为 'pann'

    X, connectivity = build_single_tet_mesh()
    rho = 1000.0 if MATERIAL == 'pann' else 1.0
    Dm_inv, vols, inv_mass_vec = preprocess_mesh(X, connectivity, rho=rho)

    n_nodes = X.shape[0]
    bc_mask = jnp.ones((n_nodes, 3))
    bc_mask = bc_mask.at[jnp.array([0, 2, 3]), :].set(0.0)

    if MATERIAL == 'neo':
        mat_params = make_mat_params_neo(mu=1.0, bulk=10.0, scale=1.0)
        dt = 1e-5
        num_steps = 100_000
        v0 = jnp.zeros((n_nodes, 3))
        v0 = v0.at[1].set(jnp.array([0.1, 0.0, 0.0]))
        print("Using Neo-Hookean material")
    else:
        if os.path.exists("pann_params.pkl"):
            trained = load_params("pann_params.pkl")
        else:
            trained = train_pann_model(steps=5000, learning_rate=0.01)
            save_params(trained)
        mat_params = make_mat_params_pann(
            trained['layers'], trained['k_vol'], trained['normalization_constant'], scale=STRESS_SCALE
        )
        plot_pann_fittings(trained, STRESS_SCALE)
        dt = 1e-6
        num_steps = 5000
        v0 = jnp.zeros((n_nodes, 3))
        v0 = v0.at[1].set(jnp.array([10.0, 0.0, 0.0]))
        print("Using PANN material")

    u0 = jnp.zeros((n_nodes, 3))
    state = (u0, v0, X, connectivity, Dm_inv, vols, inv_mass_vec, bc_mask, mat_params, dt)

    print("\nRunning FEM simulation...")
    trajectory = run_simulation(state, num_steps)
    trajectory_np = np.asarray(trajectory)
    X_np = np.asarray(X)

    if jnp.isnan(trajectory).any():
        print("Warning: Simulation exploded to NaN!")
    else:
        print("Simulation finished successfully.")

    plot_results(trajectory_np, X_np, dt, skip_step=max(1, num_steps // 200))

if __name__ == "__main__":
    main()
