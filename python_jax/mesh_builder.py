"""
网格构建模块：从配置或函数构建网格
"""
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

try:
    from .inp_parser import parse_inp_file, convert_inp_to_jax_model
    from .simdroid_parser import convert_simdroid_to_jax_model
except ImportError:
    from inp_parser import parse_inp_file, convert_inp_to_jax_model
    from simdroid_parser import convert_simdroid_to_jax_model


def build_single_tet_mesh() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """创建一个简单的四面体单元网格"""
    X = jnp.array([
        [0.0, 0.0, 0.0], 
        [1.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0], 
        [0.0, 0.0, 1.0]
    ])
    connectivity = jnp.array([[0, 1, 2, 3]])
    return X, connectivity


def build_single_hex_mesh() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """创建一个简单的六面体单元网格"""
    X = jnp.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],  # 底部 0 1 2 3
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]   # 顶部 4 5 6 7
    ])
    connectivity = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    return X, connectivity


def build_mesh_from_inp(inp_path: str) -> Dict[str, Any]:
    """从 INP 文件构建网格和模型信息"""
    inp_data = parse_inp_file(inp_path)
    model_data = convert_inp_to_jax_model(inp_data)
    return model_data


def build_mesh_from_simdroid(case_dir: str) -> Dict[str, Any]:
    """
    从 Simdroid 格式构建网格和模型信息
    
    Args:
        case_dir: Simdroid 案例目录路径（包含 control.json 和 mesh.dat）
    
    Returns:
        包含节点、单元、材料、边界条件等的字典
    """
    case_path = Path(case_dir)
    control_path = case_path / "control.json"
    mesh_path = case_path / "mesh.dat"
    
    if not control_path.exists():
        raise FileNotFoundError(f"control.json not found in {case_dir}")
    if not mesh_path.exists():
        raise FileNotFoundError(f"mesh.dat not found in {case_dir}")
    
    # 查找 AMP 文件
    amp_path = None
    for amp_file in case_path.glob("AMP-*.txt"):
        amp_path = str(amp_file)
        break
    
    model_data = convert_simdroid_to_jax_model(
        str(control_path),
        str(mesh_path),
        amp_path=amp_path,
        case_dir=str(case_path)
    )
    return model_data


def build_mesh_from_config(mesh_config: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """从配置构建网格"""
    mesh_type = mesh_config.get('type', 'tet4')
    
    if mesh_type == 'tet4':
        return build_single_tet_mesh()
    elif mesh_type == 'c3d8r':
        return build_single_hex_mesh()
    elif mesh_type == 'custom':
        # 从配置中读取节点坐标和连接关系
        nodes = jnp.array(mesh_config['nodes'])
        connectivity = jnp.array(mesh_config['connectivity'])
        return nodes, connectivity
    elif mesh_type == 'inp':
        # 从 INP 文件构建
        inp_path = mesh_config['inp_path']
        model_data = build_mesh_from_inp(inp_path)
        return model_data['nodes'], model_data['connectivity']
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")


def preprocess_tet_mesh(X, conn, rho=1.0):
    """预处理四面体网格：计算Dm_inv, vols, inv_mass"""
    X_np = np.array(X)
    conn_np = np.array(conn)
    n_nodes = X_np.shape[0]
    n_elems = conn_np.shape[0]

    Dm_inv_list = []
    vol_list = []
    for e in range(n_elems):
        X_e = X_np[conn_np[e]]
        Dm = (X_e[1:] - X_e[0]).T
        vol = np.linalg.det(Dm) / 6.0
        Dm_inv_list.append(np.linalg.inv(Dm))
        vol_list.append(vol)

    Dm_inv = jnp.array(np.stack(Dm_inv_list))
    vols = jnp.array(vol_list)

    mass_vec = np.zeros(n_nodes)
    for e in range(n_elems):
        vol = float(vol_list[e])
        m_e = rho * vol
        for i in range(4):
            mass_vec[conn_np[e, i]] += m_e / 4.0
    
    inv_mass_vec = jnp.array(1.0 / mass_vec)
    return Dm_inv, vols, inv_mass_vec


def preprocess_hex_mesh(X, conn, rho=1.0):
    """预处理六面体网格：计算B0, V0, gammas, inv_mass"""
    n_elems = conn.shape[0]
    n_nodes = X.shape[0]
    
    # 中心点的形函数导数 dN/dxi
    xi_eta_zeta = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
    ])
    dN_dxi_0 = xi_eta_zeta / 8.0
    
    # 沙漏正交模式向量
    h_vectors = np.array([
        [ 1, -1,  1, -1,  1, -1,  1, -1],
        [ 1, -1, -1,  1, -1,  1,  1, -1],
        [ 1,  1, -1, -1, -1, -1,  1,  1],
        [-1,  1, -1,  1,  1, -1,  1, -1]
    ]).T
    
    B0_list, V0_list, gammas_list = [], [], []
    mass_vec = np.zeros(n_nodes)
    
    for e in range(n_elems):
        X_e = X[conn[e]]
        
        J0 = X_e.T @ dN_dxi_0
        detJ0 = np.linalg.det(J0)
        invJ0 = np.linalg.inv(J0)
        
        B0 = dN_dxi_0 @ invJ0
        V0 = 8.0 * detJ0
        
        h_dot_x = X_e.T @ h_vectors
        B0_h_dot_x = B0 @ h_dot_x
        gammas = (h_vectors - B0_h_dot_x) / 8.0
        
        B0_list.append(B0)
        V0_list.append(V0)
        gammas_list.append(gammas)
        
        m_e = rho * V0
        for i in range(8):
            mass_vec[conn[e, i]] += m_e / 8.0

    return (jnp.array(B0_list), jnp.array(V0_list), jnp.array(gammas_list), 
            jnp.array(1.0 / mass_vec))
