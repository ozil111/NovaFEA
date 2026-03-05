import jax
import jax.numpy as jnp
from jax.export import export

def get_integration_rule(rule_type_idx):
    gp = 1.0 / jnp.sqrt(3.0)
    full_points = jnp.array([
        [-gp, -gp, -gp], [gp, -gp, -gp], [gp, gp, -gp], [-gp, gp, -gp],
        [-gp, -gp,  gp], [gp, -gp,  gp], [gp, gp,  gp], [-gp, gp,  gp]
    ], dtype=jnp.float32)
    full_weights = jnp.ones(8, dtype=jnp.float32)

    reduced_points = jnp.zeros((8, 3), dtype=jnp.float32)
    reduced_weights = jnp.array([8.0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)

    points = jax.lax.cond(rule_type_idx == 0, lambda _: full_points, lambda _: reduced_points, operand=None)
    weights = jax.lax.cond(rule_type_idx == 0, lambda _: full_weights, lambda _: reduced_weights, operand=None)
    return points, weights

def hex8_lumped_mass_export(node_coords, density, rule_type_idx):
    xi_n = jnp.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=jnp.float32)
    eta_n = jnp.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=jnp.float32)
    zeta_n = jnp.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=jnp.float32)

    gauss_points, weights = get_integration_rule(rule_type_idx)

    def compute_det_jacobian(xi_coords):
        xi, eta, zeta = xi_coords
        dN_dxi = 0.125 * xi_n * (1 + eta * eta_n) * (1 + zeta * zeta_n)
        dN_deta = 0.125 * eta_n * (1 + xi * xi_n) * (1 + zeta * zeta_n)
        dN_dzeta = 0.125 * zeta_n * (1 + xi * xi_n) * (1 + eta * eta_n)
        dN_dnatural = jnp.stack([dN_dxi, dN_deta, dN_dzeta])
        jacobian = dN_dnatural @ node_coords
        return jnp.abs(jnp.linalg.det(jacobian))

    # 使用 vmap 进行 8 个点的并行计算
    det_js = jax.vmap(compute_det_jacobian)(gauss_points)
    volume = jnp.sum(det_js * weights)
    return jnp.full((8,), (density * volume) / 8.0)

if __name__ == "__main__":
    print("正在将 Hex8 逻辑导出为 MLIR...")
    
    shape_coords = jax.ShapeDtypeStruct((8, 3), jnp.float32)
    shape_rho = jax.ShapeDtypeStruct((), jnp.float32)
    shape_idx = jax.ShapeDtypeStruct((), jnp.int32)
    
    # 1. 执行导出
    exp = export(jax.jit(hex8_lumped_mass_export))(shape_coords, shape_rho, shape_idx)
    
    # 2. 写入文件：核心修正点在于 exp.mlir_module() 后面的括号
    with open("hex8.mlir", "w", encoding="utf-8") as f:
        # 必须加上 () 来调用方法，获取真正的 MLIR 对象，然后再转成字符串
        mlir_text = str(exp.mlir_module()) 
        f.write(mlir_text)
    
    print("--------------------------------------------------")
    print("✅ 导出成功！生成的 hex8.mlir 现在是正确的文本格式了。")
    print("下一步：请在 PowerShell 中重新运行 iree-compile。")
    print("--------------------------------------------------")