import jax
import jax.numpy as jnp
from jax import jit

# 1. 模型与材料参数
E = 2.0e7
nu = 0.3
rho = 1000.0
dt = 0.001
endtime = 1.0
f_ext_max = 10000.0  # 最终载荷

nodes = jnp.array([
    [0.0, 0.0, 0.0], # Node 1
    [1.0, 0.0, 0.0], # Node 2
    [0.0, 1.0, 0.0], # Node 3
    [0.0, 0.0, 1.0]  # Node 4
], dtype=jnp.float32)

# 2. 计算体积和形状函数导数 (B矩阵的基石)
M = jnp.concatenate([jnp.ones((4, 1)), nodes], axis=1)
V = jnp.abs(jnp.linalg.det(M)) / 6.0
invM = jnp.linalg.inv(M)
dNdx = invM[1, :]
dNdy = invM[2, :]
dNdz = invM[3, :]

# 3. 组装 B 矩阵 (6 x 12)
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

# 4. 组装弹性本构 D 矩阵 (6 x 6)
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

# 5. 计算真实的单元刚度矩阵 K (12 x 12)
K = B.T @ D @ B * V

# 集中质量矩阵
mass_total = rho * V
m_lumped = jnp.full((4, 3), mass_total / 4.0)

# 边界条件掩码 (节点 1, 2, 3 固定)
bc_mask = jnp.array([[0,0,0], [0,0,0], [0,0,0], [1,1,1]])

# --- 显式时间步更新函数 ---
@jit
def step_update(u, v_half, t):
    # 1. 内力计算：真实刚度矩阵乘以位移向量
    f_int_flat = K @ u.flatten()
    f_int = f_int_flat.reshape((4, 3))
    
    # 2. 外力计算：根据 CURVE1 (Amplitude) 进行线性斜坡加载
    current_f = f_ext_max * (t / endtime)
    f_ext = jnp.zeros((4, 3)).at[3, 2].set(current_f)
    
    # 3. 求解加速度并施加边界条件
    a = (f_ext - f_int) / m_lumped * bc_mask
    
    # 4. 显式半步积分
    v_half_new = v_half + a * dt
    u_new = u + v_half_new * dt
    return u_new, v_half_new

# --- 主求解循环 ---
def solve():
    num_steps = int(endtime / dt)
    u = jnp.zeros((4, 3))
    v_half = jnp.zeros((4, 3)) # 初始化半步速度
    
    for s in range(num_steps):
        t = (s + 1) * dt # 当前时间
        u, v_half = step_update(u, v_half, t)
        
        if (s+1) % 100 == 0:
            print(f"Step {s+1}, Time {t:.3f}s: Node 4 Z-Disp = {u[3, 2]:.6e}")

    return u

final_u = solve()
print("\n最终位移结果:\n", final_u)