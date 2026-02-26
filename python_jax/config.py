"""
配置系统：从JSON文件加载仿真配置
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import jax.numpy as jnp
import numpy as np


def load_config(config_path: str) -> Dict[str, Any]:
    """从JSON文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """验证配置文件的必需字段"""
    required_fields = ['element_type', 'material_type', 'mesh', 'boundary_conditions', 
                      'material_properties', 'initial_conditions', 'time_integration']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")


def get_material_props(config: Dict[str, Any]) -> jnp.ndarray:
    """从配置中提取材料参数并转换为JAX数组"""
    mat_type = config['material_type']
    props_dict = config['material_properties']
    
    if mat_type == 'neo_hookean':
        return jnp.array([props_dict['mu'], props_dict['bulk']])
    elif mat_type == 'n3_hyperelastic':
        # C3D8R需要包含k_hg
        elem_type = config['element_type']
        if elem_type == 'c3d8r':
            return jnp.array([
                props_dict['C10'], props_dict['C20'], props_dict['C30'],
                props_dict['D1'], props_dict['D2'], props_dict['D3'],
                props_dict.get('k_hg', 0.5)
            ])
        else:
            return jnp.array([
                props_dict['C10'], props_dict['C20'], props_dict['C30'],
                props_dict['D1'], props_dict['D2'], props_dict['D3']
            ])
    else:
        raise ValueError(f"Unknown material type: {mat_type}")


def get_boundary_conditions(config: Dict[str, Any], n_nodes: int) -> jnp.ndarray:
    """从配置中构建边界条件mask"""
    bc_mask = jnp.ones((n_nodes, 3))
    bc_config = config['boundary_conditions']
    
    if 'fixed_nodes' in bc_config:
        fixed_nodes = jnp.array(bc_config['fixed_nodes'])
        bc_mask = bc_mask.at[fixed_nodes, :].set(0.0)
    
    if 'fixed_dofs' in bc_config:
        for node_dof in bc_config['fixed_dofs']:
            node_idx = node_dof['node']
            dof = node_dof['dof']  # 0=x, 1=y, 2=z
            bc_mask = bc_mask.at[node_idx, dof].set(0.0)
    
    return bc_mask


def get_initial_conditions(config: Dict[str, Any], X: jnp.ndarray) -> tuple:
    """从配置中获取初始位移和速度"""
    ic = config['initial_conditions']
    n_nodes = X.shape[0]
    
    u0 = jnp.zeros((n_nodes, 3))
    v0 = jnp.zeros((n_nodes, 3))
    
    if 'initial_displacement' in ic:
        for disp in ic['initial_displacement']:
            node_idx = disp['node']
            u0 = u0.at[node_idx].set(jnp.array(disp['value']))
    
    if 'initial_velocity' in ic:
        for vel in ic['initial_velocity']:
            node_idx = vel['node']
            v0 = v0.at[node_idx].set(jnp.array(vel['value']))
    
    return u0, v0
