"""
ABAQUS INP 文件解析器：从 INP 文件提取模型信息
"""
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import jax.numpy as jnp
import numpy as np


def parse_inp_file(inp_path: str) -> Dict[str, Any]:
    """
    解析 ABAQUS INP 文件
    
    Returns:
        包含节点、单元、材料、边界条件、载荷等信息的字典
    """
    with open(inp_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    result = {
        'nodes': [],
        'elements': [],
        'node_sets': {},
        'element_sets': {},
        'materials': {},
        'boundary_conditions': [],
        'loads': [],
        'time_integration': {}
    }
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 解析节点
        if line.startswith('*NODE'):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('*'):
                line = lines[i].strip()
                if line and not line.startswith('**'):
                    # 节点格式: node_id, x, y, z
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        node_id = int(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        result['nodes'].append((node_id, x, y, z))
                i += 1
            continue
        
        # 解析单元
        elif line.startswith('*ELEMENT'):
            # 提取单元类型
            elem_type = None
            if 'TYPE=' in line:
                match = re.search(r'TYPE=(\w+)', line)
                if match:
                    elem_type = match.group(1)
            
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('*'):
                line = lines[i].strip()
                if line and not line.startswith('**'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        elem_id = int(parts[0])
                        node_ids = [int(p) for p in parts[1:]]
                        result['elements'].append({
                            'id': elem_id,
                            'type': elem_type,
                            'nodes': node_ids
                        })
                i += 1
            continue
        
        # 解析节点集
        elif line.startswith('*NSET'):
            nset_name = None
            generate = False
            if 'NSET=' in line:
                match = re.search(r'NSET=([^,\s]+)', line)
                if match:
                    nset_name = match.group(1)
            if 'GENERATE' in line:
                generate = True
            
            node_ids = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('*'):
                line = lines[i].strip()
                if line and not line.startswith('**'):
                    if generate:
                        # GENERATE 格式: start, end, step
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
                            node_ids.extend(range(start, end + 1, step))
                    else:
                        # 普通格式或引用其他节点集
                        parts = [p.strip() for p in line.split(',')]
                        for p in parts:
                            if p:
                                # 先检查是否是节点集引用
                                if p in result['node_sets']:
                                    node_ids.extend(result['node_sets'][p])
                                else:
                                    try:
                                        node_ids.append(int(p))
                                    except ValueError:
                                        # 忽略无法解析的部分
                                        pass
                i += 1  # 重要：推进到下一行
            if nset_name:
                result['node_sets'][nset_name] = node_ids
            continue
        
        # 解析单元集
        elif line.startswith('*ELSET'):
            elset_name = None
            if 'ELSET=' in line:
                match = re.search(r'ELSET=([^,\s]+)', line)
                if match:
                    elset_name = match.group(1)
            
            elem_ids = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('*'):
                line = lines[i].strip()
                if line and not line.startswith('**'):
                    parts = [p.strip() for p in line.split(',')]
                    for p in parts:
                        if p:
                            try:
                                elem_ids.append(int(p))
                            except ValueError:
                                pass
                i += 1  # 重要：推进到下一行
            if elset_name:
                result['element_sets'][elset_name] = elem_ids
            continue
        
        # 解析材料
        elif line.startswith('*MATERIAL'):
            mat_name = None
            if 'NAME=' in line:
                match = re.search(r'NAME=([^,\s]+)', line)
                if match:
                    mat_name = match.group(1)
            
            mat_data = {}
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('*'):
                line = lines[i].strip()
                
                if line.startswith('*Density'):
                    i += 1
                    # 读取密度值（单行，可能以逗号结尾）
                    if i < len(lines):
                        density_line = lines[i].strip()
                        # 移除末尾逗号和空白
                        density_line = density_line.rstrip(',').strip()
                        if density_line and not density_line.startswith('**'):
                            try:
                                # 可能包含多个值，取第一个
                                density_val = density_line.split(',')[0].strip()
                                if density_val:
                                    mat_data['density'] = float(density_val)
                            except ValueError:
                                pass
                    i += 1  # 推进到下一行
                    continue
                
                elif line.startswith('*Hyperelastic'):
                    # 解析超弹性参数
                    # 格式: *Hyperelastic, n=3, reduced polynomial
                    n = 3
                    if 'n=' in line:
                        match = re.search(r'n=(\d+)', line)
                        if match:
                            n = int(match.group(1))
                    
                    i += 1
                    # 读取参数值（单行，逗号分隔）
                    if i < len(lines):
                        params_line = lines[i].strip()
                        if params_line and not params_line.startswith('**'):
                            # 移除末尾逗号
                            params_line = params_line.rstrip(',').strip()
                            if params_line:
                                try:
                                    # 按逗号或空格分割
                                    params = []
                                    for p in params_line.replace(',', ' ').split():
                                        p = p.strip()
                                        if p:
                                            params.append(float(p))
                                    if params:
                                        mat_data['hyperelastic'] = {
                                            'type': 'reduced_polynomial',
                                            'n': n,
                                            'params': params
                                        }
                                except ValueError:
                                    pass
                    i += 1  # 推进到下一行
                    continue
                
                i += 1  # 对于其他行，正常推进
            
            if mat_name:
                result['materials'][mat_name] = mat_data
            continue
        
        # 解析边界条件
        elif line.startswith('*BOUNDARY'):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('*'):
                line = lines[i].strip()
                if line and not line.startswith('**'):
                    # 格式: nset_name, dof_start, dof_end, value
                    # 注意：dof_end 可能为空，表示到6
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        nset_name = parts[0]
                        dof_start = int(parts[1]) if parts[1] else 1
                        # 如果 dof_end 为空，默认为 6（所有平移自由度）
                        dof_end = int(parts[2]) if len(parts) > 2 and parts[2] else 6
                        value = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
                        result['boundary_conditions'].append({
                            'nset': nset_name,
                            'dof_start': dof_start,
                            'dof_end': dof_end,
                            'value': value
                        })
                i += 1
            continue
        
        # 解析集中载荷
        elif line.startswith('*CLOAD'):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('*'):
                line = lines[i].strip()
                if line and not line.startswith('**'):
                    # 格式: nset_name, dof, magnitude
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        nset_name = parts[0]
                        dof = int(parts[1])
                        magnitude = float(parts[2])
                        result['loads'].append({
                            'nset': nset_name,
                            'dof': dof,
                            'magnitude': magnitude
                        })
                i += 1
            continue
        
        # 解析时间积分参数
        elif line.startswith('*DYNAMIC'):
            if 'EXPLICIT' in line:
                i += 1
                if i < len(lines):
                    time_line = lines[i].strip()
                    parts = [p.strip() for p in time_line.split(',')]
                    if len(parts) >= 2:
                        try:
                            dt = float(parts[0])
                            total_time = float(parts[1])
                            num_steps = int(total_time / dt)
                            result['time_integration'] = {
                                'dt': dt,
                                'total_time': total_time,
                                'num_steps': num_steps
                            }
                        except (ValueError, IndexError):
                            pass
        
        i += 1
    
    return result


def convert_inp_to_jax_model(inp_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 INP 数据转换为 JAX 模型格式
    
    Returns:
        包含节点坐标、单元连接、材料参数、边界条件等的字典
    """
    # 提取节点并排序
    nodes_dict = {node_id: (x, y, z) for node_id, x, y, z in inp_data['nodes']}
    sorted_node_ids = sorted(nodes_dict.keys())
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}
    
    # 构建节点坐标数组（0-based索引）
    X = jnp.array([[nodes_dict[nid][0], nodes_dict[nid][1], nodes_dict[nid][2]] 
                    for nid in sorted_node_ids])
    
    # 构建单元连接关系（转换为0-based索引）
    connectivity = []
    for elem in inp_data['elements']:
        node_ids = elem['nodes']
        # ABAQUS 使用 1-based，转换为 0-based
        conn = [node_id_to_idx[nid] for nid in node_ids if nid in node_id_to_idx]
        if len(conn) > 0:
            connectivity.append(conn)
    
    connectivity = jnp.array(connectivity)
    
    # 提取材料参数
    material_props = {}
    if inp_data['materials']:
        mat_name = list(inp_data['materials'].keys())[0]
        mat_data = inp_data['materials'][mat_name]
        
        density = mat_data.get('density', 1000.0)
        material_props['density'] = density
        
        if 'hyperelastic' in mat_data:
            hyp_data = mat_data['hyperelastic']
            params = hyp_data['params']
            n = hyp_data['n']
            
            # Reduced polynomial 格式转换
            # ABAQUS: C10, C20, C30, D1, D2, D3 (如果 n=3)
            # 我们的格式: C10, C20, C30, D1, D2, D3
            if len(params) >= 6:
                material_props['C10'] = params[0]
                material_props['C20'] = params[1] if len(params) > 1 else 0.0
                material_props['C30'] = params[2] if len(params) > 2 else 0.0
                material_props['D1'] = params[3] if len(params) > 3 else 0.0
                material_props['D2'] = params[4] if len(params) > 4 else 0.0
                material_props['D3'] = params[5] if len(params) > 5 else 0.0
            else:
                # 填充默认值
                material_props['C10'] = params[0] if len(params) > 0 else 0.0
                material_props['C20'] = 0.0
                material_props['C30'] = 0.0
                material_props['D1'] = 0.0
                material_props['D2'] = 0.0
                material_props['D3'] = 0.0
    
    # 提取边界条件
    bc_config = {'fixed_nodes': [], 'fixed_dofs': []}
    for bc in inp_data['boundary_conditions']:
        nset_name = bc['nset']
        if nset_name in inp_data['node_sets']:
            node_ids = inp_data['node_sets'][nset_name]
            dof_start = bc['dof_start']
            dof_end = bc['dof_end']
            
            # 转换为 0-based 节点索引
            node_indices = [node_id_to_idx[nid] for nid in node_ids if nid in node_id_to_idx]
            
            if dof_start == 1 and dof_end == 6:
                # 固定所有自由度
                bc_config['fixed_nodes'].extend(node_indices)
            else:
                # 固定特定自由度
                for node_idx in node_indices:
                    for dof in range(dof_start - 1, dof_end):  # 转换为 0-based
                        bc_config['fixed_dofs'].append({
                            'node': node_idx,
                            'dof': dof
                        })
    
    # 去重
    bc_config['fixed_nodes'] = list(set(bc_config['fixed_nodes']))
    
    # 提取载荷
    loads = []
    for load in inp_data['loads']:
        nset_name = load['nset']
        if nset_name in inp_data['node_sets']:
            node_ids = inp_data['node_sets'][nset_name]
            dof = load['dof'] - 1  # 转换为 0-based (1=x, 2=y, 3=z)
            magnitude = load['magnitude']
            
            # 转换为 0-based 节点索引
            node_indices = [node_id_to_idx[nid] for nid in node_ids if nid in node_id_to_idx]
            
            # 将载荷分配到每个节点
            load_per_node = magnitude / len(node_indices) if node_indices else 0.0
            for node_idx in node_indices:
                loads.append({
                    'node': node_idx,
                    'dof': dof,
                    'magnitude': load_per_node
                })
    
    # 时间积分参数
    time_config = inp_data.get('time_integration', {})
    
    return {
        'nodes': X,
        'connectivity': connectivity,
        'material_properties': material_props,
        'boundary_conditions': bc_config,
        'loads': loads,
        'time_integration': time_config,
        'element_type': 'c3d8r' if len(connectivity) > 0 and len(connectivity[0]) == 8 else 'tet4'
    }
