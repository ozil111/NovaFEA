# JAX + IREE：高性能结构求解器开发工作流

本手册旨在指导如何将力学算法（以 Hex8 单元质量矩阵为例）从 JAX 导出并集成至 C++ 环境。

## 1. 核心链路概览

该工作流分为四个阶段，利用 **MLIR** 作为中间桥梁，实现了算法逻辑与底层硬件的解耦。

1. **算法设计 (JAX/Python)**：编写可微分、可并行的力学算子。
2. **算子硬化 (MLIR/StableHLO)**：将动态 Python 逻辑导出为静态计算图。
3. **底层编译 (IREE Compiler)**：针对 CPU (AVX/ARM) 或 GPU (CUDA/Vulkan) 生成机器码。
4. **工程集成 (IREE Runtime/C++)**：在求解器主程序中加载并调用二进制算子。

------

## 2. 阶段一：JAX 算法定义与导出

对于 Hex8 单元，其节点集中质量 $m_i$ 的计算公式为：

$$m_i = \frac{1}{8} \int_{\Omega_e} \rho |J| d\xi d\eta d\zeta$$

在 Python 中，我们利用 `jax.vmap` 实现高斯积分点的并行计算，并使用 `jax.export` 进行导出。

```Python
import jax
import jax.numpy as jnp
from jax.export import export

def hex8_mass_kernel(node_coords, density, rule_idx):
    # 逻辑：计算雅可比 -> 数值积分 -> 质量分配
    # 使用 jax.lax.cond 处理静态分支 (Full/Reduced integration)
    ... 
    return lumped_mass_vector  # Shape: (8,)

# 导出逻辑
shape_specs = (
    jax.ShapeDtypeStruct((8, 3), jnp.float32), # 坐标
    jax.ShapeDtypeStruct((), jnp.float32),    # 密度
    jax.ShapeDtypeStruct((), jnp.int32)       # 积分类型索引
)
exp = export(jax.jit(hex8_mass_kernel))(*shape_specs)

with open("hex8.mlir", "w") as f:
    f.write(str(exp.mlir_module()))
```

------

## 3. 阶段二：AOT 编译 (Ahead-of-Time)

使用 IREE 编译器将硬件无关的 `.mlir` 文件编译为特定硬件优化的 `.vmfb`（Virtual Machine FlatBuffer）。

- **针对 CPU 优化**（利用 AVX/SSE 指令集）：

  ```PowerShell
  iree-compile --iree-hal-target-backends=llvm-cpu `
               --iree-llvmcpu-target-cpu=host `
               hex8.mlir -o hex8_solver.vmfb
  ```

- **针对 NVIDIA GPU 优化**：

  ```PowerShell
  iree-compile --iree-hal-target-backends=cuda hex8.mlir -o hex8_solver_gpu.vmfb
  ```

------

## 4. 阶段三：C++ 集成与调用

在 C++ 端，通过 IREE 提供的轻量级 C API 加载算子。这使得求解器可以像插件一样热加载不同的单元逻辑。

### 核心步骤：

1. **创建 Instance & Device**：初始化计算环境。
2. **加载 Module**：将 `.vmfb` 映射至内存。
3. **封装 Buffer**：将 C++ 数组（如网格坐标）封装为 `iree_hal_buffer_view`。
4. **调用函数**：通过函数名（如 `module.main`）触发计算。

> **提示**：为实现极致性能，建议在 C++ 中通过 **Zero-copy** 模式直接绑定求解器的内存地址，避免数据在宿主空间反复拷贝。

------

## 5. 性能提升关键：批量化 (Batching)

在实际的 **xFEM** 或显式动力学计算中，单次只计算一个单元会产生巨大的上下文切换开销。

- **错误做法**：在 C++ 里写 `for` 循环，循环 100 万次调用 IREE。
- **正确做法**：修改 JAX 函数使用 `vmap` 支持批量输入 `(N, 8, 3)`。在 C++ 中一次性传入 10 万个单元的数据，让 IREE 在底层自动进行多线程并行。

| **策略**              | **实现难度**  | **预期性能**             |
| --------------------- | ------------- | ------------------------ |
| 单单元顺序调用        | 低            | 差 (受限于 API 开销)     |
| **批量化调用 (Vmap)** | **中 (推荐)** | **极高 (接近原生 CUDA)** |
| 异步流式计算          | 高            | 极高 (适合超大规模并行)  |

------

## 总结

通过这套流程，你无需精通 `CUDA` 编程，也能为 **hyperFEM** 编写出运行在 GPU 上的高性能单元算子。



## 如何从 IREE Python 运行时提取 DLL 和 LIB 文件

本指南说明了如何将 Python 的 `.pyd` 扩展模块转换为标准的 C++ 开发所需的 `.dll` 和 `.lib`（导入库）文件。

### 1. 准备 DLL 文件

IREE 的运行时核心通常以 `.pyd` 文件形式存在于 Python 包中。由于 `.pyd` 本质上就是 Windows 的动态链接库（DLL），我们可以直接重命名。

1. 进入 IREE 运行时的安装目录： `D:\anaconda3\envs\jax\Lib\site-packages\iree\_runtime_libs`
2. 找到文件：`_runtime.cp312-win_amd64.pyd`
3. 将其复制到你的工作目录（例如 `D:\Download`）。
4. 将副本重命名为：**`iree_runtime.dll`**

------

### 2. 生成导出定义文件 (.def)

由于我们只有二进制 DLL，需要通过 Visual Studio 的工具手动创建一个导出函数清单。

1. 打开 **Developer Command Prompt for VS** (在开始菜单搜索即可)。

2. 导航至你的工作目录：

   Bash

   ```
   cd /d D:\Download
   ```

3. 执行 `dumpbin` 命令查看导出函数：

   Bash

   ```
   dumpbin /exports iree_runtime.dll > iree_runtime.def
   ```

------

### 3. 数据清洗 (手动编辑 .def)

打开生成的 `iree_runtime.def`，将其修改为符合规范的导出格式。你需要删除所有描述性文本，只保留 **LIBRARY** 声明和 **EXPORTS** 下的函数修饰名（Mangled Names）。

**最终的 `iree_runtime.def` 内容应如下：**

Plaintext

```
LIBRARY iree_runtime
EXPORTS
   ??0builtin_exception@nanobind@@QEAA@$$QEAV01@@Z
   ??0builtin_exception@nanobind@@QEAA@AEBV01@@Z
   ??0builtin_exception@nanobind@@QEAA@W4exception_type@1@PEBD@Z
   ??0python_error@nanobind@@QEAA@$$QEAV01@@Z
   ??0python_error@nanobind@@QEAA@AEBV01@@Z
   ??0python_error@nanobind@@QEAA@XZ
   ??1builtin_exception@nanobind@@UEAA@XZ
   ??1python_error@nanobind@@UEAA@XZ
   ??_7builtin_exception@nanobind@@6B@
   ??_7python_error@nanobind@@6B@
   ?discard_as_unraisable@python_error@nanobind@@QEAAXPEBD@Z
   ?discard_as_unraisable@python_error@nanobind@@QEAAXVhandle@2@@Z
   ?matches@python_error@nanobind@@QEBA_NVhandle@2@@Z
   ?restore@python_error@nanobind@@QEAAXXZ
   ?trace@python_error@nanobind@@QEBA?AVobject@2@XZ
   ?traceback@python_error@nanobind@@QEBA?AVobject@2@XZ
   ?type@builtin_exception@nanobind@@QEBA?AW4exception_type@2@XZ
   ?type@python_error@nanobind@@QEBA?AVhandle@2@XZ
   ?value@python_error@nanobind@@QEBA?AVhandle@2@XZ
   ?what@python_error@nanobind@@UEBAPEBDXZ
   PyInit__runtime
```

> **注意：** 必须保留带 `?` 和 `@` 的完整字符串，这些是 C++ 编译器识别函数所必需的“修饰名”。

------

### 4. 生成导入库 (.lib)

在开发人员命令提示符中执行以下命令，根据 `.def` 文件生成 `.lib`：

Bash

```
lib /def:iree_runtime.def /machine:x64 /out:iree_runtime.lib
```

执行成功后，文件夹中会出现两个新文件：

- **`iree_runtime.lib`**: 用于在 C++ 项目中链接。
- **`iree_runtime.exp`**: 导出文件（通常开发中不需要，可以忽略）。

------

### 5. 在 C++ 项目中使用

1. **链接**：在 Visual Studio 的项目属性 -> **链接器 -> 输入 -> 附加依赖项** 中添加 `iree_runtime.lib` 的路径。
2. **运行**：确保 `iree_runtime.dll` 位于生成的 `.exe` 相同目录下，或者已添加到系统的 `PATH` 环境变量中。