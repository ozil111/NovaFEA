"""
材料库：统一签名为 (F, props) -> energy_density。
新增材料只需实现能量函数并注册到 MATERIAL_FACTORY。
"""
import jax
import jax.numpy as jnp


def neo_hookean_energy(F, props):
    """Neo-Hookean 应变能密度。props = [mu, bulk]。"""
    mu, bulk = props[0], props[1]
    J = jnp.linalg.det(F)
    C = F.T @ F
    I1 = jnp.trace(C)
    return 0.5 * mu * (I1 - 3.0) - mu * jnp.log(J) + 0.5 * bulk * (J - 1.0) ** 2


def n3_hyperelastic_energy(F, props):
    """N3 超弹性应变能密度。props 至少前 6 个为 [C10, C20, C30, D1, D2, D3]（C3D8R 时最后可带 k_hg，此处只用前 6 个）。"""
    C10, C20, C30, D1, D2, D3 = props[0], props[1], props[2], props[3], props[4], props[5]
    J = jnp.linalg.det(F)
    C = F.T @ F
    J = jnp.clip(J, a_min=1e-6)
    J_minus_2_3 = J ** (-2.0 / 3.0)
    C_bar = J_minus_2_3 * C
    I1_bar = jnp.trace(C_bar)
    I1_bar_minus_3 = I1_bar - 3.0
    W_dev = (
        C10 * I1_bar_minus_3
        + C20 * (I1_bar_minus_3 ** 2)
        + C30 * (I1_bar_minus_3 ** 3)
    )
    J_minus_1 = J - 1.0
    W_vol = (1.0 / D1) * (J_minus_1 ** 2)
    W_vol += jnp.where(D2 > 0, (1.0 / (D2 + 1e-16)) * (J_minus_1 ** 4), 0.0)
    W_vol += jnp.where(D3 > 0, (1.0 / (D3 + 1e-16)) * (J_minus_1 ** 6), 0.0)
    return W_dev + W_vol


MATERIAL_FACTORY = {
    "neo_hookean": neo_hookean_energy,
    "n3_hyperelastic": n3_hyperelastic_energy,
}
