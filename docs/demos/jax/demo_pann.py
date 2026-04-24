import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# ==========================================
# 0. 全局配置与数据
# ==========================================
jax.config.update("jax_enable_x64", True)

# 实验数据 (保持不变)
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

# --- 关键修改 1: 计算 Scaling Factor ---
# 选取一个特征值（例如 1 MPa）作为缩放因子
STRESS_SCALE = 1.0e5 
print(f"Using Stress Scale: {STRESS_SCALE}")

# ==========================================
# 1. PANN 模型 (修改了 k_vol 初始化)
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
        # 修改点：k_vol 初始化为小数值 (10.0)，因为现在我们在归一化空间训练
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

def pann_energy(F, params):
    inputs, J = compute_invariants(F)
    psi_nn = icnn_forward(params, inputs)
    
    k_vol = params['k_vol']
    J_term = J + 1.0/J - 2.0
    psi_growth = k_vol * (J_term**2)
    
    n_norm = params['normalization_constant']
    psi_norm = -n_norm * (J - 1.0)
    
    # 修改点：你加的 squeeze 是对的，防止维度错误
    return jnp.squeeze(psi_nn + psi_growth + psi_norm)

# 基础应力计算 (输出是归一化的应力)
get_pk1_stress_normalized = jax.grad(pann_energy, argnums=0)

def update_normalization(params):
    F_ref = jnp.eye(3)
    temp_params = params.copy()
    temp_params['normalization_constant'] = jnp.array([0.0])
    P_ref = get_pk1_stress_normalized(F_ref, temp_params)
    n_new = jnp.trace(P_ref) / 3.0
    params['normalization_constant'] = jnp.array([n_new])
    return params

# ==========================================
# 2. 训练准备 (应用 Scaling)
# ==========================================

def prepare_data():
    F_list, Target_list, Type_list = [], [], []

    def add_data(stress, strain, type_id):
        for s, e in zip(stress, strain):
            if type_id == 1: # Uniaxial
                lam = 1.0 + e
                F = jnp.diag(jnp.array([lam, lam**-0.5, lam**-0.5]))
            elif type_id == 2: # Biaxial
                lam = 1.0 + e
                F = jnp.diag(jnp.array([lam, lam, lam**-2]))
            elif type_id == 3: # Planar
                lam = 1.0 + e
                F = jnp.diag(jnp.array([lam, 1.0, lam**-1]))
            
            F_list.append(F)
            # 修改点：目标值除以 SCALE
            Target_list.append(s / STRESS_SCALE)
            Type_list.append(float(type_id))

    add_data(uniaxial_data['stress'], uniaxial_data['strain'], 1)
    add_data(biaxial_data['stress'], biaxial_data['strain'], 2)
    add_data(planar_data['stress'], planar_data['strain'], 3)
    
    # Volumetric
    for p, j_val in zip(volumetric_data['pressure'], volumetric_data['j']):
        lam = j_val**(1.0/3.0)
        F = jnp.diag(jnp.array([lam, lam, lam]))
        F_list.append(F)
        # 修改点：压力也要除以 SCALE
        Target_list.append(p / STRESS_SCALE)
        Type_list.append(4.0)

    return jnp.stack(F_list), jnp.stack(Target_list), jnp.stack(Type_list)

F_train, S_train, Type_train = prepare_data()

@jax.jit
def loss_fn(params, F_batch, Target_batch, Type_batch):
    # 计算归一化的预测应力
    P_preds = jax.vmap(get_pk1_stress_normalized, in_axes=(0, None))(F_batch, params)
    
    def get_pred_val(P, F, type_id):
        val_stress = P[0, 0] # Uniaxial/Biaxial/Planar
        
        # Volumetric Pressure calculation
        lam = F[0, 0]
        J = jnp.linalg.det(F)
        sigma_11 = P[0, 0] * lam / J
        val_pressure = -sigma_11
        
        return jax.lax.cond(type_id == 4.0, lambda: val_pressure, lambda: val_stress)

    preds = jax.vmap(get_pred_val)(P_preds, F_batch, Type_batch)
    return jnp.mean((preds - Target_batch)**2)

# ==========================================
# 3. 训练循环 (Adam)
# ==========================================

def train_pann_model(steps=5000, learning_rate=0.01):
    key = jax.random.PRNGKey(42)
    params = init_pann_params(key)
    params = update_normalization(params) # 初始更新一次
    
    # Adam Optimizer State
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    grad_fn = jax.value_and_grad(loss_fn)
    
    print("\nStart Training PANN (Scaled Data)...")
    for step in range(steps):
        if step % 100 == 0:
            params = update_normalization(params)

        loss_val, grads = grad_fn(params, F_train, S_train, Type_train)
        
        # Adam Update
        m = jax.tree_util.tree_map(lambda m_i, g_i: beta1 * m_i + (1 - beta1) * g_i, m, grads)
        v = jax.tree_util.tree_map(lambda v_i, g_i: beta2 * v_i + (1 - beta2) * (g_i**2), v, grads)
        
        m_hat = jax.tree_util.tree_map(lambda m_i: m_i / (1 - beta1**(step + 1)), m)
        v_hat = jax.tree_util.tree_map(lambda v_i: v_i / (1 - beta2**(step + 1)), v)
        
        params = jax.tree_util.tree_map(
            lambda p_i, m_h, v_h: p_i - learning_rate * m_h / (jnp.sqrt(v_h) + eps), 
            params, m_hat, v_hat
        )
        
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss_val:.6f}") # Loss 现在应该是小数值 (e.g., 0.1 ~ 5.0)
            
    params = update_normalization(params)
    return params

# ==========================================
# 4. FEM 仿真 (集成 Scaling)
# ==========================================

def element_force_kernel(u_elem, X_elem, Dm_inv, vol, pann_params):
    x_elem = X_elem + u_elem
    Ds = (x_elem[1:] - x_elem[0]).T
    F = Ds @ Dm_inv
    
    # --- 关键修改：获取归一化应力并乘回 Scale ---
    P_norm = get_pk1_stress_normalized(F, pann_params)
    P_real = P_norm * STRESS_SCALE 
    # ----------------------------------------
    
    H = vol * Dm_inv.T
    f_nodes_123 = P_real @ H
    f_node_0 = -jnp.sum(f_nodes_123, axis=1)
    f_local = jnp.vstack([f_node_0, f_nodes_123.T])
    return -f_local

@jax.jit
def explicit_step(state, _):
    (u, v, X, connectivity, Dm_inv, vols, inv_mass_vec, bc_mask, pann_params, step_dt) = state
    u_elems = u[connectivity]
    X_elems = X[connectivity]

    f_elems = jax.vmap(element_force_kernel, in_axes=(0, 0, 0, 0, None))(
        u_elems, X_elems, Dm_inv, vols, pann_params
    )

    global_forces = jnp.zeros_like(u)
    global_forces = global_forces.at[connectivity].add(f_elems)

    a = global_forces * inv_mass_vec[:, None]
    a = a * bc_mask
    
    v_new = v + a * step_dt
    v_new = v_new * bc_mask
    u_new = u + v_new * step_dt

    return (u_new, v_new, X, connectivity, Dm_inv, vols, inv_mass_vec, bc_mask, pann_params, step_dt), u_new

# --- 网格与运行 (保持不变) ---
def build_single_tet_mesh():
    X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    connectivity = jnp.array([[0, 1, 2, 3]])
    return X, connectivity

def preprocess_mesh(X, connectivity, rho=1000.0):
    X_np = np.array(X)
    conn_np = np.array(connectivity)
    n_elems = conn_np.shape[0]
    Dm_inv_list = [np.linalg.inv((X_np[conn_np[e, 1:]] - X_np[conn_np[e, 0]]).T) for e in range(n_elems)]
    vol_list = [np.linalg.det((X_np[conn_np[e, 1:]] - X_np[conn_np[e, 0]]).T) / 6.0 for e in range(n_elems)]
    
    Dm_inv = jnp.array(np.stack(Dm_inv_list))
    vols = jnp.array(vol_list)
    
    mass_vec = np.zeros(X_np.shape[0])
    for e in range(n_elems):
        m_e = rho * vol_list[e]
        for i in range(4): mass_vec[conn_np[e, i]] += m_e / 4.0
    
    return Dm_inv, vols, jnp.array(1.0 / mass_vec)

def run_simulation(initial_state, num_steps):
    final_state, trajectory = jax.lax.scan(explicit_step, initial_state, None, length=num_steps)
    return trajectory

def save_params(params, filename="pann_params.pkl"):
    """保存训练好的参数"""
    params_np = jax.tree_util.tree_map(lambda x: np.array(x), params)
    with open(filename, "wb") as f:
        pickle.dump(params_np, f)
    print(f"Model parameters saved to {filename}")


def load_params(filename="pann_params.pkl"):
    """加载参数"""
    with open(filename, "rb") as f:
        params_np = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params_np)


def plot_all_fittings(params, scale):
    """绘制所有工况的拟合效果"""
    plt.figure(figsize=(15, 10))

    # 1. Uniaxial
    plt.subplot(2, 2, 1)
    plt.plot(uniaxial_data['strain'], uniaxial_data['stress'], 'ko', label='Exp')
    e_range = jnp.linspace(0, float(uniaxial_data['strain'].max()), 50)
    s_pred = []
    for e in e_range:
        lam = 1.0 + e
        F = jnp.diag(jnp.array([lam, lam**-0.5, lam**-0.5]))
        P = get_pk1_stress_normalized(F, params) * scale
        s_pred.append(P[0, 0])
    plt.plot(e_range, s_pred, 'r-', linewidth=2, label='PANN')
    plt.title("Uniaxial Tension")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. Biaxial
    plt.subplot(2, 2, 2)
    plt.plot(biaxial_data['strain'], biaxial_data['stress'], 'ko', label='Exp')
    e_range = jnp.linspace(0, float(biaxial_data['strain'].max()), 50)
    s_pred = []
    for e in e_range:
        lam = 1.0 + e
        F = jnp.diag(jnp.array([lam, lam, lam**-2]))
        P = get_pk1_stress_normalized(F, params) * scale
        s_pred.append(P[0, 0])
    plt.plot(e_range, s_pred, 'r-', linewidth=2)
    plt.title("Biaxial Tension")
    plt.grid(True, alpha=0.3)

    # 3. Planar
    plt.subplot(2, 2, 3)
    plt.plot(planar_data['strain'], planar_data['stress'], 'ko', label='Exp')
    e_range = jnp.linspace(0, float(planar_data['strain'].max()), 50)
    s_pred = []
    for e in e_range:
        lam = 1.0 + e
        F = jnp.diag(jnp.array([lam, 1.0, lam**-1]))
        P = get_pk1_stress_normalized(F, params) * scale
        s_pred.append(P[0, 0])
    plt.plot(e_range, s_pred, 'r-', linewidth=2)
    plt.title("Planar Tension")
    plt.grid(True, alpha=0.3)

    # 4. Volumetric
    plt.subplot(2, 2, 4)
    plt.plot(volumetric_data['j'], volumetric_data['pressure'], 'ko', label='Exp')
    j_range = jnp.linspace(float(volumetric_data['j'].min()), 1.0, 50)
    p_pred = []
    for j_val in j_range:
        lam = j_val**(1.0 / 3.0)
        F = jnp.diag(jnp.array([lam, lam, lam]))
        P = get_pk1_stress_normalized(F, params) * scale
        sigma = P[0, 0] * lam / j_val
        p_pred.append(-sigma)
    plt.plot(j_range, p_pred, 'r-', linewidth=2)
    plt.title("Volumetric Compression")
    plt.xlabel("Volume Ratio J")
    plt.ylabel("Pressure")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    # --- 1. 训练或加载模型 ---
    if os.path.exists("pann_params.pkl"):
        print("Found saved parameters. Loading...")
        trained_params = load_params("pann_params.pkl")
    else:
        print("No saved parameters found. Training new model...")
        trained_params = train_pann_model(steps=5000, learning_rate=0.01)
        save_params(trained_params)

    # --- 2. 验证拟合效果 (多图) ---
    print("Plotting fitting results...")
    plot_all_fittings(trained_params, STRESS_SCALE)

    # --- 3. 运行 FEM ---
    X, connectivity = build_single_tet_mesh()
    Dm_inv, vols, inv_mass_vec = preprocess_mesh(X, connectivity)

    n_nodes = X.shape[0]
    bc_mask = jnp.ones((n_nodes, 3))
    bc_mask = bc_mask.at[jnp.array([0, 2, 3]), :].set(0.0)

    u0 = jnp.zeros((n_nodes, 3))
    v0 = jnp.zeros((n_nodes, 3))
    v0 = v0.at[1].set(jnp.array([10.0, 0.0, 0.0]))

    dt = 1e-6
    num_steps = 5000
    state = (u0, v0, X, connectivity, Dm_inv, vols, inv_mass_vec, bc_mask, trained_params, dt)

    print("\nRunning FEM simulation...")
    trajectory = run_simulation(state, num_steps)

    plt.figure()
    disp_x = np.array(trajectory)[:, 1, 0]
    plt.plot(np.arange(num_steps) * dt, disp_x)
    plt.title("FEM Node 1 Displacement (X)")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()