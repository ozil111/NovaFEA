# 使用说明

## 项目结构

项目已从 `demo/` 目录重构到 `python_jax/` 目录，包含以下模块：

- `builder.py`: 求解器构建器
- `elements.py`: 单元库
- `materials.py`: 材料库
- `config.py`: 配置系统
- `mesh_builder.py`: 网格构建模块
- `visualization.py`: 可视化模块
- `main.py`: 主程序（支持命令行参数）
- `run.py`: 可执行脚本入口

## 使用方法

### 方法1: 使用JSON配置文件

```bash
# 使用示例配置文件运行
python python_jax/run.py --config python_jax/config_example_tet4.json

# 或
python python_jax/run.py --config python_jax/config_example_c3d8r.json
```

### 方法2: 使用命令行参数

```bash
# 基本用法
python python_jax/run.py --element tet4 --material neo_hookean --steps 10000 --dt 1e-5

# 六面体单元
python python_jax/run.py --element c3d8r --material n3_hyperelastic --steps 20000 --dt 1e-4

# 不显示可视化
python python_jax/run.py --element tet4 --material neo_hookean --no-viz
```

### 方法3: 作为Python模块调用

```python
from python_jax import run_simulation

# 使用配置文件
trajectory, final_state = run_simulation(config_path="config_example_tet4.json")

# 使用参数
trajectory, final_state = run_simulation(
    element_type="tet4",
    material_type="neo_hookean",
    num_steps=10000,
    dt=1e-5,
    visualize=True
)
```

## JSON配置文件格式

### Tet4 + Neo-Hookean 示例

```json
{
  "element_type": "tet4",
  "material_type": "neo_hookean",
  "density": 1.0,
  "mesh": {
    "type": "tet4"
  },
  "boundary_conditions": {
    "fixed_nodes": [0, 2, 3]
  },
  "material_properties": {
    "mu": 1.0,
    "bulk": 10.0
  },
  "initial_conditions": {
    "initial_velocity": [
      {
        "node": 1,
        "value": [0.1, 0.0, 0.0]
      }
    ]
  },
  "time_integration": {
    "dt": 1e-5,
    "num_steps": 100000
  }
}
```

### C3D8R + N3 Hyperelastic 示例

```json
{
  "element_type": "c3d8r",
  "material_type": "n3_hyperelastic",
  "density": 1.0,
  "mesh": {
    "type": "c3d8r"
  },
  "boundary_conditions": {
    "fixed_nodes": [0, 3, 4, 7]
  },
  "material_properties": {
    "C10": 1.0,
    "C20": 0.0,
    "C30": 0.0,
    "D1": 1e-3,
    "D2": 0.0,
    "D3": 0.0,
    "k_hg": 0.5
  },
  "initial_conditions": {
    "initial_velocity": [
      {
        "node": 1,
        "value": [0.5, 0.0, 0.0]
      }
    ]
  },
  "time_integration": {
    "dt": 1e-4,
    "num_steps": 20000
  }
}
```

## 配置选项说明

- `element_type`: 单元类型，可选 `"tet4"` 或 `"c3d8r"`
- `material_type`: 材料类型，可选 `"neo_hookean"` 或 `"n3_hyperelastic"`
- `density`: 材料密度
- `mesh.type`: 网格类型，与 `element_type` 对应
- `boundary_conditions.fixed_nodes`: 固定节点列表
- `boundary_conditions.fixed_dofs`: 固定自由度列表（可选）
- `material_properties`: 材料参数（根据材料类型不同）
- `initial_conditions.initial_velocity`: 初始速度
- `initial_conditions.initial_displacement`: 初始位移（可选）
- `time_integration.dt`: 时间步长
- `time_integration.num_steps`: 时间步数
