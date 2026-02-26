"""
构建器工厂：根据 element_type 和 material_type 组合材料与单元，
通过闭包绑定后使用 jax.grad 生成通用节点力内核，并返回 JIT 显式积分步函数。
"""
import jax
import jax.numpy as jnp

try:
    from .materials import MATERIAL_FACTORY
    from .elements import ELEMENT_FACTORY
except ImportError:
    # 支持直接运行
    from materials import MATERIAL_FACTORY
    from elements import ELEMENT_FACTORY

# 各单元 elem_data 中数组个数，用于构建 vmap in_axes
_ELEM_DATA_LEN = {"tet4": 2, "c3d8r": 3}


def build_solver_step(element_type: str, material_type: str):
    """
    根据单元类型和材料类型，动态构建被 JIT 编译的显式积分步函数。
    返回的 step_fn(state, _) 中 state = (u, v, X, conn, inv_mass, bc_mask, props, dt, elem_data_tuple)。
    """
    mat_fn = MATERIAL_FACTORY[material_type]
    elem_fn = ELEMENT_FACTORY[element_type]
    n_elem_data = _ELEM_DATA_LEN[element_type]
    in_axes = (0, 0, None) + (0,) * n_elem_data

    def compute_element_energy(u_elem, X_elem, props, *elem_data):
        return elem_fn(u_elem, X_elem, mat_fn, props, elem_data)

    compute_internal_force = jax.grad(compute_element_energy, argnums=0)

    def force_kernel(u_elem, X_elem, props, *elem_data):
        return -compute_internal_force(u_elem, X_elem, props, *elem_data)

    @jax.jit
    def explicit_step(state, _):
        # 支持可选的 external_forces 参数
        if len(state) == 10:
            (u, v, X, conn, inv_mass, bc_mask, props, dt, elem_data_tuple, external_forces) = state
        else:
            (u, v, X, conn, inv_mass, bc_mask, props, dt, elem_data_tuple) = state
            external_forces = jnp.zeros_like(u)
        
        u_elems = u[conn]
        X_elems = X[conn]
        f_elems = jax.vmap(force_kernel, in_axes=in_axes)(
            u_elems, X_elems, props, *elem_data_tuple
        )
        global_forces = jnp.zeros_like(u)
        global_forces = global_forces.at[conn].add(f_elems)
        # 添加外力
        global_forces = global_forces + external_forces
        a = global_forces * inv_mass[:, None]
        a = a * bc_mask
        v_new = (v + a * dt) * bc_mask
        u_new = u + v_new * dt
        
        if len(state) == 10:
            new_state = (
                u_new,
                v_new,
                X,
                conn,
                inv_mass,
                bc_mask,
                props,
                dt,
                elem_data_tuple,
                external_forces,
            )
        else:
            new_state = (
                u_new,
                v_new,
                X,
                conn,
                inv_mass,
                bc_mask,
                props,
                dt,
                elem_data_tuple,
            )
        return new_state, u_new

    return explicit_step
