"""
单元库：统一签名为 (u_elem, X_elem, material_energy_fn, props, elem_data) -> total_energy。
elem_data: Tet4 为 (Dm_inv, vol)，C3D8R 为 (B0, V0, gammas)。
新增单元只需实现能量函数并注册到 ELEMENT_FACTORY。
"""
import jax
import jax.numpy as jnp


def tet4_energy(u_elem, X_elem, material_energy_fn, props, elem_data):
    """Tet4 单元总能量。elem_data = (Dm_inv, vol)。"""
    Dm_inv, vol = elem_data
    x_elem = X_elem + u_elem
    Ds = (x_elem[1:] - x_elem[0]).T
    F = Ds @ Dm_inv
    W_mat = material_energy_fn(F, props)
    return W_mat * vol


def c3d8r_energy(u_elem, X_elem, material_energy_fn, props, elem_data):
    """C3D8R 单元总能量 = 材料能 + 沙漏能。elem_data = (B0, V0, gammas)，k_hg = props[-1]。"""
    B0, V0, gammas = elem_data
    # 材料参数不含沙漏刚度，最后一项留给 k_hg
    mat_props = props[:-1]
    k_hg = props[-1]
    x_elem = X_elem + u_elem
    F = x_elem.T @ B0
    W_mat = material_energy_fn(F, mat_props)
    q = u_elem.T @ gammas
    W_hg = 0.5 * k_hg * jnp.sum(q ** 2)
    return (W_mat + W_hg) * V0


ELEMENT_FACTORY = {
    "tet4": tet4_energy,
    "c3d8r": c3d8r_energy,
}
