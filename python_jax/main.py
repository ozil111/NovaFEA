"""
主程序：支持命令行参数和JSON配置的显式动力学求解器
"""
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .builder import build_solver_step
    from .config import load_config, validate_config, get_material_props, get_boundary_conditions, get_initial_conditions
    from .mesh_builder import build_mesh_from_config, preprocess_tet_mesh, preprocess_hex_mesh, build_mesh_from_inp, build_mesh_from_simdroid
    from .visualization import plot_tet_results, plot_hex_results
except ImportError:
    # 支持直接运行
    from builder import build_solver_step
    from config import load_config, validate_config, get_material_props, get_boundary_conditions, get_initial_conditions
    from mesh_builder import build_mesh_from_config, preprocess_tet_mesh, preprocess_hex_mesh, build_mesh_from_inp, build_mesh_from_simdroid
    from visualization import plot_tet_results, plot_hex_results


def run_simulation(config_path: str = None, inp_path: str = None, simdroid_path: str = None,
                   element_type: str = None, material_type: str = None, 
                   num_steps: int = None, dt: float = None, visualize: bool = True):
    """
    运行仿真
    
    Args:
        config_path: JSON配置文件路径（如果提供，将覆盖其他参数）
        inp_path: ABAQUS INP文件路径
        simdroid_path: Simdroid案例目录路径（包含control.json和mesh.dat）
        element_type: 单元类型 ("tet4" | "c3d8r")
        material_type: 材料类型 ("neo_hookean" | "n3_hyperelastic")
        num_steps: 时间步数
        dt: 时间步长
        visualize: 是否可视化结果
    """
    jax.config.update("jax_enable_x64", True)
    
    # 从 Simdroid 格式加载
    if simdroid_path:
        model_data = build_mesh_from_simdroid(simdroid_path)
        X = model_data['nodes']
        conn = model_data['connectivity']
        element_type = model_data['element_type']
        material_type = 'n3_hyperelastic'  # Simdroid 使用超弹性材料
        
        # 材料参数
        mat_props = model_data['material_properties']
        props = jnp.array([
            mat_props.get('C10', 0.0),
            mat_props.get('C20', 0.0),
            mat_props.get('C30', 0.0),
            mat_props.get('D1', 0.0),
            mat_props.get('D2', 0.0),
            mat_props.get('D3', 0.0),
            mat_props.get('k_hg', 0.5)  # 沙漏控制参数
        ])
        rho = mat_props.get('density', 1000.0)
        
        # 边界条件
        bc_config = model_data['boundary_conditions']
        n_nodes = X.shape[0]
        bc_mask = jnp.ones((n_nodes, 3))
        if bc_config.get('fixed_nodes'):
            fixed_nodes = jnp.array(bc_config['fixed_nodes'])
            bc_mask = bc_mask.at[fixed_nodes, :].set(0.0)
        if bc_config.get('fixed_dofs'):
            for dof_config in bc_config['fixed_dofs']:
                node_idx = dof_config['node']
                dof = dof_config['dof']
                bc_mask = bc_mask.at[node_idx, dof].set(0.0)
        
        # 初始条件
        u0 = jnp.zeros((n_nodes, 3))
        v0 = jnp.zeros((n_nodes, 3))
        
        # 载荷（作为外力施加）
        external_forces = jnp.zeros((n_nodes, 3))
        loads = model_data.get('loads', [])
        for load in loads:
            node_idx = load['node']
            dof = load['dof']
            magnitude = load['magnitude']
            external_forces = external_forces.at[node_idx, dof].set(magnitude)
        
        # 时间积分参数
        time_config = model_data.get('time_integration', {})
        dt = time_config.get('dt', dt) if dt is None else dt
        num_steps = time_config.get('num_steps', num_steps) if num_steps is None else num_steps
        
        # 预处理网格
        print(f"[Simulation] 开始预处理网格...")
        if element_type == "tet4":
            Dm_inv, vols, inv_mass = preprocess_tet_mesh(X, conn, rho=rho)
            elem_data_tuple = (Dm_inv, vols)
            print(f"[Simulation] 四面体网格预处理完成: 单元数={len(vols)}")
        else:
            B0_all, V0_all, gammas_all, inv_mass = preprocess_hex_mesh(X, conn, rho=rho)
            elem_data_tuple = (B0_all, V0_all, gammas_all)
            print(f"[Simulation] 六面体网格预处理完成: 单元数={B0_all.shape[0]}")
        
        # 生成求解器并运行
        step_fn = build_solver_step(element_type, material_type)
        state = (u0, v0, X, conn, inv_mass, bc_mask, props, dt, elem_data_tuple, external_forces)
        
        print(f"\n[Simulation] ========== 开始计算 ==========")
        print(f"[Simulation] 单元类型: {element_type}")
        print(f"[Simulation] 材料类型: {material_type}")
        print(f"[Simulation] 时间步数: {num_steps}")
        print(f"[Simulation] 时间步长: {dt:.6e}")
        print(f"[Simulation] 总时间: {num_steps * dt:.6e}")
        print(f"[Simulation] 节点数: {X.shape[0]}")
        print(f"[Simulation] 单元数: {conn.shape[0]}")
        print(f"[Simulation] 初始位移范围: [{float(jnp.min(u0)):.6e}, {float(jnp.max(u0)):.6e}]")
        print(f"[Simulation] 初始速度范围: [{float(jnp.min(v0)):.6e}, {float(jnp.max(v0)):.6e}]")
        if len(external_forces.shape) > 0:
            max_force = float(jnp.max(jnp.abs(external_forces)))
            print(f"[Simulation] 最大外力: {max_force:.6e}")
        
        print(f"[Simulation] 开始时间积分...")
        final_state, trajectory = jax.lax.scan(step_fn, state, None, length=num_steps)
        
        print(f"[Simulation] 时间积分完成")
        
        # 从trajectory中提取关键信息
        u_final = trajectory[-1]
        u_mid = trajectory[num_steps // 2] if num_steps > 1 else trajectory[0]
        u_quarter = trajectory[num_steps // 4] if num_steps > 3 else trajectory[0]
        
        print(f"\n[Simulation] ========== 计算结果 ==========")
        if element_type == "tet4":
            print(f"[Simulation] 节点1位移 X (最终): {float(u_final[1, 0]):.6e}")
            print(f"[Simulation] 节点1位移 X (1/4): {float(u_quarter[1, 0]):.6e}")
            print(f"[Simulation] 节点1位移 X (1/2): {float(u_mid[1, 0]):.6e}")
        else:
            max_disp_final = float(jnp.max(jnp.abs(u_final)))
            max_disp_quarter = float(jnp.max(jnp.abs(u_quarter)))
            max_disp_mid = float(jnp.max(jnp.abs(u_mid)))
            print(f"[Simulation] 最大位移 (1/4步): {max_disp_quarter:.6e}")
            print(f"[Simulation] 最大位移 (1/2步): {max_disp_mid:.6e}")
            print(f"[Simulation] 最大位移 (最终): {max_disp_final:.6e}")
            print(f"[Simulation] 最大位移 X方向: {float(jnp.max(trajectory[:, :, 0])):.6e}")
            print(f"[Simulation] 最大位移 Y方向: {float(jnp.max(trajectory[:, :, 1])):.6e}")
            print(f"[Simulation] 最大位移 Z方向: {float(jnp.max(trajectory[:, :, 2])):.6e}")
        
        # 提取最终状态的速度信息
        u_final_state, v_final_state = final_state[0], final_state[1]
        max_vel = float(jnp.max(jnp.abs(v_final_state)))
        print(f"[Simulation] 最终最大速度: {max_vel:.6e}")
        
        print(f"[Simulation] ========== 计算完成 ==========\n")
        
        # 可视化
        if visualize:
            if element_type == "tet4":
                plot_tet_results(np.asarray(trajectory), np.asarray(X), dt, skip_step=500)
            else:
                plot_hex_results(trajectory, X, dt, skip_step=200, conn=conn)
        
        return trajectory, final_state
    
    # 从 INP 文件加载
    if inp_path:
        model_data = build_mesh_from_inp(inp_path)
        X = model_data['nodes']
        conn = model_data['connectivity']
        element_type = model_data['element_type']
        material_type = 'n3_hyperelastic'  # INP 文件使用超弹性材料
        
        # 材料参数
        mat_props = model_data['material_properties']
        props = jnp.array([
            mat_props.get('C10', 0.0),
            mat_props.get('C20', 0.0),
            mat_props.get('C30', 0.0),
            mat_props.get('D1', 0.0),
            mat_props.get('D2', 0.0),
            mat_props.get('D3', 0.0),
            0.5  # k_hg (沙漏控制参数)
        ])
        rho = mat_props.get('density', 1000.0)
        
        # 边界条件
        bc_config = model_data['boundary_conditions']
        n_nodes = X.shape[0]
        bc_mask = jnp.ones((n_nodes, 3))
        if bc_config.get('fixed_nodes'):
            fixed_nodes = jnp.array(bc_config['fixed_nodes'])
            bc_mask = bc_mask.at[fixed_nodes, :].set(0.0)
        if bc_config.get('fixed_dofs'):
            for dof_config in bc_config['fixed_dofs']:
                node_idx = dof_config['node']
                dof = dof_config['dof']
                bc_mask = bc_mask.at[node_idx, dof].set(0.0)
        
        # 初始条件
        u0 = jnp.zeros((n_nodes, 3))
        v0 = jnp.zeros((n_nodes, 3))
        
        # 载荷（作为外力施加）
        external_forces = jnp.zeros((n_nodes, 3))
        loads = model_data.get('loads', [])
        for load in loads:
            node_idx = load['node']
            dof = load['dof']
            magnitude = load['magnitude']
            external_forces = external_forces.at[node_idx, dof].set(magnitude)
        
        # 时间积分参数
        time_config = model_data.get('time_integration', {})
        dt = time_config.get('dt', dt) if dt is None else dt
        num_steps = time_config.get('num_steps', num_steps) if num_steps is None else num_steps
        
        # 预处理网格
        if element_type == "tet4":
            Dm_inv, vols, inv_mass = preprocess_tet_mesh(X, conn, rho=rho)
            elem_data_tuple = (Dm_inv, vols)
        else:
            B0_all, V0_all, gammas_all, inv_mass = preprocess_hex_mesh(X, conn, rho=rho)
            elem_data_tuple = (B0_all, V0_all, gammas_all)
        
        # 生成求解器并运行
        step_fn = build_solver_step(element_type, material_type)
        state = (u0, v0, X, conn, inv_mass, bc_mask, props, dt, elem_data_tuple, external_forces)
        
        print(f"Running simulation from INP file: element={element_type}, material={material_type}, steps={num_steps}, dt={dt}")
        final_state, trajectory = jax.lax.scan(step_fn, state, None, length=num_steps)
        
        print("Simulation finished.")
        
        # 输出结果
        if element_type == "tet4":
            print("Node 1 displacement X (sample):", float(trajectory[-1, 1, 0]))
        else:
            print("Max displacement X:", float(jnp.max(trajectory[:, :, 0])))
        
        # 可视化
        if visualize:
            if element_type == "tet4":
                plot_tet_results(np.asarray(trajectory), np.asarray(X), dt, skip_step=500)
            else:
                plot_hex_results(trajectory, X, dt, skip_step=200, conn=conn)
        
        return trajectory, final_state
    
    # 加载配置
    if config_path:
        config = load_config(config_path)
        validate_config(config)
        element_type = config['element_type']
        material_type = config['material_type']
        mesh_config = config['mesh']
        time_config = config['time_integration']
        num_steps = time_config.get('num_steps', num_steps)
        dt = time_config.get('dt', dt)
        rho = config.get('density', 1.0)
    else:
        # 使用命令行参数或默认值
        if element_type is None:
            element_type = "tet4"
        if material_type is None:
            material_type = "neo_hookean"
        if num_steps is None:
            num_steps = 10000
        if dt is None:
            dt = 1e-5
        rho = 1.0
        mesh_config = {'type': element_type}
        config = {
            'element_type': element_type,
            'material_type': material_type,
            'mesh': mesh_config,
            'boundary_conditions': {},
            'material_properties': {},
            'initial_conditions': {},
            'time_integration': {'num_steps': num_steps, 'dt': dt}
        }
    
    # 构建网格
    X, conn = build_mesh_from_config(mesh_config)
    n_nodes = X.shape[0]
    
    # 预处理网格
    if element_type == "tet4":
        Dm_inv, vols, inv_mass = preprocess_tet_mesh(X, conn, rho=rho)
        elem_data_tuple = (Dm_inv, vols)
    else:
        B0_all, V0_all, gammas_all, inv_mass = preprocess_hex_mesh(X, conn, rho=rho)
        elem_data_tuple = (B0_all, V0_all, gammas_all)
    
    # 边界条件
    if config_path:
        bc_mask = get_boundary_conditions(config, n_nodes)
    else:
        # 默认边界条件
        bc_mask = jnp.ones((n_nodes, 3))
        if element_type == "tet4":
            fixed_nodes = jnp.array([0, 2, 3])
        else:
            fixed_nodes = jnp.array([0, 3, 4, 7])
        bc_mask = bc_mask.at[fixed_nodes, :].set(0.0)
    
    # 材料参数
    if config_path:
        props = get_material_props(config)
    else:
        if element_type == "tet4":
            props = jnp.array([1.0, 10.0])  # neo_hookean: [mu, bulk]
        else:
            props = jnp.array([1.0, 0.0, 0.0, 1e-3, 0.0, 0.0, 0.5])  # n3_hyperelastic + k_hg
    
    # 初始条件
    if config_path:
        u0, v0 = get_initial_conditions(config, X)
    else:
        u0 = jnp.zeros((n_nodes, 3))
        v0 = jnp.zeros((n_nodes, 3))
        if element_type == "tet4":
            v0 = v0.at[1].set(jnp.array([0.1, 0.0, 0.0]))
        else:
            v0 = v0.at[jnp.array([1, 2, 5, 6]), 0].set(0.5)
    
    # 生成求解器并运行
    step_fn = build_solver_step(element_type, material_type)
    state = (u0, v0, X, conn, inv_mass, bc_mask, props, dt, elem_data_tuple)
    
    print(f"Running simulation: element={element_type}, material={material_type}, steps={num_steps}, dt={dt}")
    final_state, trajectory = jax.lax.scan(step_fn, state, None, length=num_steps)
    
    print("Simulation finished.")
    
    # 输出结果
    if element_type == "tet4":
        print("Node 1 displacement X (sample):", float(trajectory[-1, 1, 0]))
    else:
        print("Max displacement X:", float(jnp.max(trajectory[:, [1, 2, 5, 6], 0])))
    
    # 可视化
    if visualize:
        if element_type == "tet4":
            plot_tet_results(np.asarray(trajectory), np.asarray(X), dt, skip_step=500)
        else:
            plot_hex_results(trajectory, X, dt, skip_step=200, conn=conn)
    
    return trajectory, final_state


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='显式动力学有限元求解器')
    parser.add_argument('--config', type=str, help='JSON配置文件路径')
    parser.add_argument('--inp', type=str, help='ABAQUS INP文件路径')
    parser.add_argument('--simdroid', type=str, help='Simdroid案例目录路径（包含control.json和mesh.dat）')
    parser.add_argument('--element', type=str, choices=['tet4', 'c3d8r'], help='单元类型')
    parser.add_argument('--material', type=str, choices=['neo_hookean', 'n3_hyperelastic'], help='材料类型')
    parser.add_argument('--steps', type=int, help='时间步数')
    parser.add_argument('--dt', type=float, help='时间步长')
    parser.add_argument('--no-viz', action='store_true', help='不显示可视化')
    
    args = parser.parse_args()
    
    run_simulation(
        config_path=args.config,
        inp_path=args.inp,
        simdroid_path=args.simdroid,
        element_type=args.element,
        material_type=args.material,
        num_steps=args.steps,
        dt=args.dt,
        visualize=not args.no_viz
    )


if __name__ == "__main__":
    main()
