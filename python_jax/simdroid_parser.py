"""
Simdroid 格式解析器：从 Simdroid 格式文件提取模型信息
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import jax.numpy as jnp
import numpy as np


def _extract_braced_block(content: str, keyword: str, start_pos: int = 0) -> Optional[str]:
    """
    提取 keyword { ... } 块的内容，正确处理嵌套大括号。
    返回大括号内的内容字符串，找不到返回 None。
    """
    pattern = re.compile(rf'{keyword}\s*\{{', re.DOTALL)
    m = pattern.search(content, start_pos)
    if not m:
        return None
    
    depth = 1
    i = m.end()
    while i < len(content) and depth > 0:
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
        i += 1
    
    if depth == 0:
        return content[m.end():i - 1]
    return None


def parse_mesh_dat(mesh_path: str) -> Dict[str, Any]:
    """
    解析 mesh.dat 文件
    
    Returns:
        包含节点、单元、集合等信息的字典
    """
    print(f"[Parser] 开始解析 mesh.dat: {mesh_path}")
    with open(mesh_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    result = {
        'nodes': [],
        'elements': [],
        'node_sets': {},
        'element_sets': {},
        'surface_sets': {}
    }
    
    # 解析节点 —— Node 是顶层块，内部无嵌套
    node_block = _extract_braced_block(content, 'Node')
    if node_block:
        for line in node_block.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                match = re.match(r'(\d+)\s*\[([^\]]+)\]', line)
                if match:
                    node_id = int(match.group(1))
                    coords = [float(x.strip()) for x in match.group(2).split(',')]
                    if len(coords) >= 3:
                        result['nodes'].append((node_id, coords[0], coords[1], coords[2]))
    
    # 解析单元 —— Element { Hex8 { ... } } 有嵌套
    elem_block = _extract_braced_block(content, 'Element')
    if elem_block:
        hex8_block = _extract_braced_block(elem_block, 'Hex8')
        if hex8_block:
            for line in hex8_block.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    match = re.match(r'(\d+)\s*\[([^\]]+)\]', line)
                    if match:
                        elem_id = int(match.group(1))
                        node_ids = [int(x.strip()) for x in match.group(2).split(',') if x.strip()]
                        if len(node_ids) > 8:
                            node_ids = node_ids[:8]
                        if len(node_ids) == 8:
                            result['elements'].append({
                                'id': elem_id,
                                'type': 'Hex8',
                                'nodes': node_ids
                            })
    
    # 解析集合 —— Set { Node { ... }  Element { ... } } 有嵌套
    set_block = _extract_braced_block(content, 'Set')
    if set_block:
        # 解析节点集
        node_set_block = _extract_braced_block(set_block, 'Node')
        if node_set_block:
            for line in node_set_block.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('[', 1)
                    if len(parts) == 2:
                        set_name = parts[0].strip()
                        ids_str = parts[1].rstrip(']').strip()
                        
                        node_ids = []
                        if ':' in ids_str:
                            range_parts = ids_str.split(':')
                            if len(range_parts) == 3:
                                start, end, step = int(range_parts[0]), int(range_parts[1]), int(range_parts[2])
                                node_ids = list(range(start, end + 1, step))
                        else:
                            node_ids = [int(x.strip()) for x in ids_str.split(',') if x.strip()]
                        
                        if node_ids:
                            result['node_sets'][set_name] = node_ids
        
        # 解析单元集
        elem_set_block = _extract_braced_block(set_block, 'Element')
        if elem_set_block:
            for line in elem_set_block.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('[', 1)
                    if len(parts) == 2:
                        set_name = parts[0].strip()
                        ids_str = parts[1].rstrip(']').strip()
                        
                        elem_ids = []
                        if ':' in ids_str:
                            range_parts = ids_str.split(':')
                            if len(range_parts) == 3:
                                start, end, step = int(range_parts[0]), int(range_parts[1]), int(range_parts[2])
                                elem_ids = list(range(start, end + 1, step))
                        else:
                            elem_ids = [int(x.strip()) for x in ids_str.split(',') if x.strip()]
                        
                        if elem_ids:
                            result['element_sets'][set_name] = elem_ids
    
    print(f"[Parser] 解析完成: 节点数={len(result['nodes'])}, 单元数={len(result['elements'])}, "
          f"节点集数={len(result['node_sets'])}, 单元集数={len(result['element_sets'])}")
    if result['node_sets']:
        print(f"[Parser] 节点集: {list(result['node_sets'].keys())}")
    if result['element_sets']:
        print(f"[Parser] 单元集: {list(result['element_sets'].keys())}")
    
    return result


def parse_amp_file(amp_path: str) -> Dict[str, Any]:
    """
    解析 AMP 时间曲线文件
    
    Returns:
        包含时间-值对列表的字典
    """
    print(f"[Parser] 开始解析 AMP 文件: {amp_path}")
    with open(amp_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    time_values = []
    in_data_section = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('#DATA'):
            in_data_section = True
            continue
        if in_data_section and line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    time = float(parts[0])
                    value = float(parts[1])
                    time_values.append((time, value))
                except ValueError:
                    pass
    
    print(f"[Parser] AMP 文件解析完成: 时间点数量={len(time_values)}")
    if time_values:
        print(f"[Parser] AMP 时间范围: [{time_values[0][0]}, {time_values[-1][0]}], "
              f"值范围: [{min(v[1] for v in time_values)}, {max(v[1] for v in time_values)}]")
    
    return {'time_values': time_values}


def parse_control_json(control_path: str) -> Dict[str, Any]:
    """
    解析 control.json 文件
    
    Returns:
        包含所有配置信息的字典
    """
    with open(control_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_simdroid_to_jax_model(control_path: str, mesh_path: str, 
                                   amp_path: Optional[str] = None,
                                   case_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    将 Simdroid 格式转换为 JAX 模型格式
    
    Args:
        control_path: control.json 文件路径
        mesh_path: mesh.dat 文件路径
        amp_path: AMP 时间曲线文件路径（可选）
        case_dir: 案例目录（用于查找 AMP 文件）
    
    Returns:
        包含节点坐标、单元连接、材料参数、边界条件等的字典
    """
    print(f"[Parser] ========== 开始转换 Simdroid 格式 ==========")
    print(f"[Parser] control.json: {control_path}")
    print(f"[Parser] mesh.dat: {mesh_path}")
    
    # 解析文件
    control = parse_control_json(control_path)
    mesh_data = parse_mesh_dat(mesh_path)
    
    # 解析 AMP 文件（如果提供）
    amp_data = None
    if amp_path:
        amp_data = parse_amp_file(amp_path)
    elif case_dir:
        # 尝试从 control.json 中查找 AMP 文件名
        functions = control.get('Function', {})
        for func_name, func_data in functions.items():
            if func_data.get('Type') == 'Tabular':
                # 尝试查找对应的 AMP 文件
                amp_file = Path(case_dir) / f"{func_name}.txt"
                if amp_file.exists():
                    amp_data = parse_amp_file(str(amp_file))
                    break
    
    # 提取节点并排序
    nodes_dict = {node_id: (x, y, z) for node_id, x, y, z in mesh_data['nodes']}
    sorted_node_ids = sorted(nodes_dict.keys())
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}
    
    # 构建节点坐标数组（0-based索引）
    X = jnp.array([[nodes_dict[nid][0], nodes_dict[nid][1], nodes_dict[nid][2]] 
                    for nid in sorted_node_ids])
    
    print(f"[Parser] 节点坐标数组形状: {X.shape}")
    print(f"[Parser] 节点坐标范围: X=[{float(jnp.min(X[:, 0])):.6f}, {float(jnp.max(X[:, 0])):.6f}], "
          f"Y=[{float(jnp.min(X[:, 1])):.6f}, {float(jnp.max(X[:, 1])):.6f}], "
          f"Z=[{float(jnp.min(X[:, 2])):.6f}, {float(jnp.max(X[:, 2])):.6f}]")
    
    # 构建单元连接关系（转换为0-based索引）
    connectivity = []
    for elem in mesh_data['elements']:
        node_ids = elem['nodes']
        # Simdroid 使用 0-based，转换为节点索引
        conn = []
        for nid in node_ids:
            if nid in node_id_to_idx:
                conn.append(node_id_to_idx[nid])
            else:
                # 如果节点 ID 不在映射中，可能是索引本身
                conn.append(nid)
        if len(conn) == 8:
            connectivity.append(conn)
    
    if len(connectivity) == 0:
        raise ValueError(f"No valid elements found. Parsed {len(mesh_data['elements'])} elements from mesh.dat")
    
    connectivity = jnp.array(connectivity)
    print(f"[Parser] 单元连接数组形状: {connectivity.shape}")
    
    # 提取材料参数
    material_props = {}
    materials = control.get('Material', {})
    print(f"[Parser] 开始提取材料参数...")
    if materials:
        mat_name = list(materials.keys())[0]
        mat_data = materials[mat_name]
        print(f"[Parser] 材料名称: {mat_name}")
        
        density = mat_data.get('Density', 1000.0)
        material_props['density'] = density
        print(f"[Parser] 密度: {density}")
        
        # 检查材料类型
        mat_type = mat_data.get('MaterialType', '')
        print(f"[Parser] 材料类型: {mat_type}")
        if mat_type == 'IsotropicElastic':
            # 线性弹性材料，需要转换为超弹性参数
            # 如果 control.json 中没有 E 和 nu，使用默认值
            E = mat_data.get('YoungModulus', mat_data.get('E', 1e6))
            nu = mat_data.get('PoissonRatio', mat_data.get('nu', 0.3))
            print(f"[Parser] 弹性模量 E: {E}, 泊松比 nu: {nu}")
            
            # 转换为超弹性参数（Reduced Polynomial n=1 近似）
            # 对于小变形，neo-Hookean 可以近似线性弹性
            # mu = E / (2 * (1 + nu))  (剪切模量)
            # K = E / (3 * (1 - 2*nu))  (体积模量)
            mu = E / (2 * (1 + nu))
            K = E / (3 * (1 - 2 * nu))
            
            # Reduced Polynomial n=1: W = C10*(I1-3) + D1*(J-1)^2
            # 对于小变形，C10 ≈ mu/2, D1 ≈ 2/K
            material_props['C10'] = mu / 2.0
            material_props['C20'] = 0.0
            material_props['C30'] = 0.0
            material_props['D1'] = 2.0 / K if K > 0 else 1e-6
            material_props['D2'] = 0.0
            material_props['D3'] = 0.0
            print(f"[Parser] 转换后的超弹性参数: C10={material_props['C10']:.6e}, D1={material_props['D1']:.6e}")
        elif mat_type == 'ReducedPolynomial':
            # Reduced Polynomial 材料类型
            # 从 MaterialConstants 中提取参数
            mat_constants = mat_data.get('MaterialConstants', {})
            order = mat_constants.get('Order', 1)
            const_array = mat_constants.get('Const', [])
            
            print(f"[Parser] ReducedPolynomial Order: {order}, Const数组长度: {len(const_array)}")
            
            # Const数组格式: [C10, C20, C30, ..., D1, D2, D3, ...]
            # 前order个是C参数，后order个是D参数
            if len(const_array) >= 2 * order:
                # 提取C参数
                for i in range(min(order, 3)):  # 最多支持C10, C20, C30
                    material_props[f'C{i+1}0'] = const_array[i]
                
                # 提取D参数
                for i in range(min(order, 3)):  # 最多支持D1, D2, D3
                    material_props[f'D{i+1}'] = const_array[order + i]
                
                # 如果order小于3，填充0
                for i in range(order, 3):
                    material_props[f'C{i+1}0'] = 0.0
                    material_props[f'D{i+1}'] = 0.0
                
                print(f"[Parser] 提取的超弹性参数:")
                print(f"[Parser]   C10={material_props['C10']:.6e}, C20={material_props['C20']:.6e}, C30={material_props['C30']:.6e}")
                print(f"[Parser]   D1={material_props['D1']:.6e}, D2={material_props['D2']:.6e}, D3={material_props['D3']:.6e}")
            else:
                print(f"[Parser] 警告: Const数组长度不足，期望至少{2*order}个值，实际{len(const_array)}个")
                # 使用默认值
                material_props['C10'] = const_array[0] if len(const_array) > 0 else 1e5
                material_props['C20'] = const_array[1] if len(const_array) > 1 else 0.0
                material_props['C30'] = const_array[2] if len(const_array) > 2 else 0.0
                material_props['D1'] = const_array[order] if len(const_array) > order else 1e-6
                material_props['D2'] = const_array[order + 1] if len(const_array) > order + 1 else 0.0
                material_props['D3'] = const_array[order + 2] if len(const_array) > order + 2 else 0.0
                print(f"[Parser] 使用部分参数: C10={material_props['C10']:.6e}, D1={material_props['D1']:.6e}")
        else:
            # 其他材料类型，尝试从材料数据中提取超弹性参数
            # 检查是否有直接的超弹性参数
            if 'C10' in mat_data:
                material_props['C10'] = mat_data.get('C10', 0.0)
                material_props['C20'] = mat_data.get('C20', 0.0)
                material_props['C30'] = mat_data.get('C30', 0.0)
                material_props['D1'] = mat_data.get('D1', 0.0)
                material_props['D2'] = mat_data.get('D2', 0.0)
                material_props['D3'] = mat_data.get('D3', 0.0)
                print(f"[Parser] 直接读取超弹性参数: C10={material_props['C10']:.6e}, D1={material_props['D1']:.6e}")
            else:
                # 默认超弹性参数（使用较小的值以避免数值问题）
                material_props['C10'] = 1e5
                material_props['C20'] = 0.0
                material_props['C30'] = 0.0
                material_props['D1'] = 1e-6
                material_props['D2'] = 0.0
                material_props['D3'] = 0.0
                print(f"[Parser] 使用默认超弹性参数: C10={material_props['C10']:.6e}, D1={material_props['D1']:.6e}")
    
    # 提取边界条件
    bc_config = {'fixed_nodes': [], 'fixed_dofs': []}
    constraints = control.get('Constraint', {})
    boundary = constraints.get('Boundary', {})
    print(f"[Parser] 开始提取边界条件...")
    print(f"[Parser] 边界条件数量: {len(boundary)}")
    
    for bc_name, bc_data in boundary.items():
        nset_name = bc_data.get('NodeSet', '')
        dofs = bc_data.get('Dofs', [])
        print(f"[Parser] 边界条件 '{bc_name}': 节点集={nset_name}, DOFs={dofs}")
        
        if nset_name in mesh_data['node_sets']:
            node_ids = mesh_data['node_sets'][nset_name]
            node_indices = [node_id_to_idx.get(nid, nid) for nid in node_ids]
            print(f"[Parser]   节点集 '{nset_name}' 包含 {len(node_ids)} 个节点")
            
            # Dofs 格式: [dof1, dof2, dof3, dof4, dof5, dof6]
            # dof1-3: x, y, z 平移
            # dof4-6: rx, ry, rz 旋转
            # 1 表示固定，0 表示自由
            fixed_count = 0
            for node_idx in node_indices:
                if dofs[0] == 1:  # x 方向固定
                    bc_config['fixed_dofs'].append({'node': node_idx, 'dof': 0})
                    fixed_count += 1
                if dofs[1] == 1:  # y 方向固定
                    bc_config['fixed_dofs'].append({'node': node_idx, 'dof': 1})
                    fixed_count += 1
                if dofs[2] == 1:  # z 方向固定
                    bc_config['fixed_dofs'].append({'node': node_idx, 'dof': 2})
                    fixed_count += 1
            print(f"[Parser]   添加了 {fixed_count} 个固定DOF约束")
        else:
            print(f"[Parser]   警告: 节点集 '{nset_name}' 未找到")
    
    # 去重
    bc_config['fixed_dofs'] = list({(d['node'], d['dof']): d for d in bc_config['fixed_dofs']}.values())
    print(f"[Parser] 边界条件提取完成: 固定DOF总数={len(bc_config['fixed_dofs'])}")
    
    # 提取载荷
    loads = []
    load_configs = control.get('Load', {})
    print(f"[Parser] 开始提取载荷...")
    print(f"[Parser] 载荷数量: {len(load_configs)}")
    
    for load_name, load_data in load_configs.items():
        load_type = load_data.get('Type', '')
        print(f"[Parser] 载荷 '{load_name}': 类型={load_type}")
        if load_type == 'Force':
            nset_name = load_data.get('NodeSet', '')
            dof_str = load_data.get('Dof', '')
            value = load_data.get('Value', 0.0)
            time_curve = load_data.get('TimeCurve', '')
            print(f"[Parser]   节点集={nset_name}, DOF={dof_str}, 初始值={value}, 时间曲线={time_curve}")
            
            # 转换 dof 字符串到索引
            dof_map = {'x': 0, 'y': 1, 'z': 2}
            dof = dof_map.get(dof_str.lower(), 0)
            
            # 应用时间曲线（如果有）
            if time_curve and amp_data:
                # 这里简化处理，使用第一个值
                # 实际应该根据时间插值
                if amp_data['time_values']:
                    value = value * amp_data['time_values'][-1][1]  # 使用最后一个值
                    print(f"[Parser]   应用时间曲线后值={value}")
            
            if nset_name in mesh_data['node_sets']:
                node_ids = mesh_data['node_sets'][nset_name]
                node_indices = [node_id_to_idx.get(nid, nid) for nid in node_ids if nid in node_id_to_idx]
                print(f"[Parser]   节点集 '{nset_name}' 包含 {len(node_indices)} 个节点")
                
                # 将载荷分配到每个节点
                if node_indices:
                    load_per_node = value / len(node_indices)
                    for node_idx in node_indices:
                        loads.append({
                            'node': node_idx,
                            'dof': dof,
                            'magnitude': load_per_node
                        })
                    print(f"[Parser]   分配到 {len(node_indices)} 个节点, 每个节点载荷={load_per_node:.6e}")
            else:
                print(f"[Parser]   警告: 节点集 '{nset_name}' 未找到")
    
    print(f"[Parser] 载荷提取完成: 载荷点总数={len(loads)}")
    
    # 时间积分参数
    time_config = {}
    analysis_control = control.get('AnalysisControl', {})
    time_step_control = analysis_control.get('TimeStepControl', {})
    if time_step_control:
        end_time = time_step_control.get('EndTime', 0.02)
        fixed_dt = time_step_control.get('FixedDt', None)
        dt_min = time_step_control.get('Dtmin', 0.0)
        dt_scale = time_step_control.get('DtScale', 0.9)
        
        # 优先使用 FixedDt，然后是 Dtmin，最后估算
        if fixed_dt is not None and fixed_dt > 0:
            dt = fixed_dt
        elif dt_min > 0:
            dt = dt_min
        else:
            # 根据网格尺寸估算稳定时间步长
            # dt = min_edge_length / wave_speed
            # 简化处理：使用一个合理的默认值
            # 对于显式动力学，通常需要 dt < L_min / c，其中 c 是波速
            # 这里使用一个保守的估计值
            if X.shape[0] > 0 and len(connectivity) > 0:
                # 计算最小边长
                X_np = np.array(X)
                conn_np = np.array(connectivity)
                min_edge_length = float('inf')
                for elem_nodes in conn_np:
                    nodes_elem = X_np[elem_nodes]
                    # 计算单元的所有边长
                    for i in range(len(nodes_elem)):
                        for j in range(i+1, len(nodes_elem)):
                            edge_len = np.linalg.norm(nodes_elem[i] - nodes_elem[j])
                            min_edge_length = min(min_edge_length, edge_len)
                
                # 估算波速（简化：假设材料参数）
                rho = material_props.get('density', 1000.0)
                E_est = material_props.get('C10', 1e5) * 4.0  # 粗略估算
                c = np.sqrt(E_est / rho)  # 波速
                dt = (min_edge_length / c) * 0.1  # 使用 0.1 的安全系数
                dt = max(dt, 1e-7)  # 最小时间步长
                dt = min(dt, 1e-4)  # 最大时间步长
            else:
                dt = 1e-5  # 默认值
        
        num_steps = int(end_time / dt) if dt > 0 else 1000
        
        time_config = {
            'dt': dt,
            'total_time': end_time,
            'num_steps': num_steps
        }
    
    # 提取截面属性（用于沙漏控制参数）
    cross_sections = control.get('CrossSection', {})
    k_hg = 0.5  # 默认值
    if cross_sections:
        section_name = list(cross_sections.keys())[0]
        section_data = cross_sections[section_name]
        # 可以从 QuadraticViscosity 和 LinearViscosity 计算 k_hg
        # 这里简化处理
        k_hg = section_data.get('QuadraticViscosity', 1.2) / 2.0
    
    # 添加 k_hg 到材料属性
    material_props['k_hg'] = k_hg
    print(f"[Parser] 沙漏控制参数 k_hg: {k_hg}")
    
    print(f"[Parser] ========== 转换完成 ==========")
    print(f"[Parser] 总结:")
    print(f"[Parser]   - 节点数: {X.shape[0]}")
    print(f"[Parser]   - 单元数: {connectivity.shape[0]}")
    print(f"[Parser]   - 固定DOF数: {len(bc_config['fixed_dofs'])}")
    print(f"[Parser]   - 载荷点数: {len(loads)}")
    print(f"[Parser]   - 时间步长: {time_config.get('dt', 'N/A')}")
    print(f"[Parser]   - 总时间: {time_config.get('total_time', 'N/A')}")
    print(f"[Parser]   - 时间步数: {time_config.get('num_steps', 'N/A')}")
    
    return {
        'nodes': X,
        'connectivity': connectivity,
        'material_properties': material_props,
        'boundary_conditions': bc_config,
        'loads': loads,
        'time_integration': time_config,
        'element_type': 'c3d8r'  # Simdroid 使用 Hex8RPH，对应 c3d8r
    }
