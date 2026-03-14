# hyperFEM

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/your-username/hyperFEM)
[![Language](https://img.shields.io/badge/language-C%2B%2B%20%26%20Python-orange.svg)]()

**hyperFEM** 是一个从零开始构建的、现代、高性能的有限元分析（FEA）引擎。它诞生于对数值模拟和先进软件架构的探索热情，旨在融合传统高性能计算与现代AI驱动的开发范式。

## 核心特性

- **高性能C++核心**：项目主体采用现代C++（17/20）编写，注重性能和代码组织。利用CMake和Vcpkg进行构建和依赖管理，确保跨平台兼容性。
- **声明式输入文件**：使用 `.jsonc` 格式作为主要的输入文件，支持注释，使得算例定义更加清晰、易读和可扩展。
- **符号代码生成 (Symbolic Code Generation)**：项目正在探索一个革命性的开发模式，利用 `Python/SymPy` 进行符号数学定义，自动生成高度优化的C++、CUDA或JAX计算内核。这是项目的核心创新点和未来方向。
- **Python/JAX生态**：拥有一个并行的Python开发环境，利用JAX进行快速算法原型设计、机器学习模型集成和求解器验证。

## 架构愿景：从“手工打造”到“自动生成”

`hyperFEM` 的发展体现了一条清晰的演进路径，这解释了当前代码库中并存的两种模式：

1.  **当前坚实的基础 (The Proven Path)**：项目的大部分功能（位于 `system/` 和 `data_center/`）是经过精心设计和手工实现的C++代码。这个核心框架是稳定、可靠且高性能的，它验证了整个FEA流程的正确性。

2.  **未来的方向 (The Future Vision)**：我们认识到，手工编写复杂的单元和材料本构关系既耗时又容易出错。因此，项目的未来正全面转向以 `python_jax/code_gen` 为核心的开发模式。

**最终目标是：**

> 让 `hyperFEM` 的C++核心框架消费（consume）由Python中的高层符号数学定义自动生成的、高度优化的计算内核。开发者将不再需要编写底层的C++计算代码，而是专注于更高层次的数学和物理模型定义。

当前代码库中的“冗余”部分，正是这条从过去到未来演进道路上留下的宝贵足迹。

## 安装与构建

项目包含C++和Python两部分，需要分别配置环境。

### 1. C++ 环境

**依赖:**
- C++ 17/20 编译器 (MSVC, GCC, Clang)
- CMake 3.20+
- Git

**步骤:**
1. 克隆仓库：
   ```bash
   git clone https://github.com/your-username/hyperFEM.git
   cd hyperFEM
   ```
2. 安装Vcpkg依赖：
   项目使用 `vcpkg.json` 管理C++依赖。运行 `build_scripts` 下的脚本来自动安装。
   ```bash
   # Windows
   .\build_scripts\vcpkg_install.bat

   # Linux / macOS
   ./build_scripts/vcpkg_install.sh
   ```
3. 构建项目：
   使用 `build_scripts` 下的脚本进行构建。
   ```bash
   # Windows
   .\build_scripts\build.bat

   # Linux / macOS
   ./build_scripts/build.sh
   ```
   可执行文件将生成在 `build/bin/` 目录下。

### 2. Python 环境

**依赖:**
- Python 3.9+

**步骤:**
1. 进入 `python_jax` 目录：
   ```bash
   cd python_jax
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 快速开始

### 运行一个C++仿真

1. 确保项目已成功构建。
2. 使用生成的可执行文件运行一个位于 `case/` 目录下的算例。
   ```bash
   # 假设在项目根目录
   ./build/bin/hyperfem.exe -i ./case/test_new_syntax.jsonc
   ```
   *注意：请根据实际生成的可执行文件名和路径进行调整。*

### 探索代码生成和JAX求解器

1. 进入 `python_jax/code_gen` 目录。
2. **生成代码**：可以尝试运行代码生成器。
   ```bash
   # 生成 tet4 单元的C++刚度矩阵内核
   python sympy_codegen.py --task stiffness --element tet4 --target cpp
   ```
3. **运行JAX求解器**：`static.py` 是一个使用生成内核的完整JAX求解器。
   ```bash
   # 需要先为 tet4 和 isotropic 生成 jax 内核
   python sympy_codegen.py --task stiffness --element tet4 --target jax
   python sympy_codegen.py --task constitutive --material isotropic --target jax

   # 运行求解器 (需要一个适配的 .jsonc 文件)
   python static.py --model path/to/your/model.jsonc
   ```

## 目录结构

```
.
├── system/             # C++核心系统层 (求解器、装配、IO等)
├── data_center/        # C++核心数据层 (网格、组件、上下文等)
├── python_jax/         # Python/JAX 生态系统
│   └── code_gen/       # 核心的符号代码生成器
├── case/               # 仿真算例输入文件
├── docs/               # 项目设计文档和参考资料
├── build_scripts/      # 跨平台的构建和安装脚本
├── test/               # C++ 单元测试
├── CMakeLists.txt      # 主CMake构建文件
└── vcpkg.json          # C++依赖项定义
```

## 贡献

本项目目前主要由个人驱动，但欢迎任何形式的交流和贡献。如果您有任何想法、建议或发现Bug，请随时提交Issue。

## 许可

本项目采用 [MIT](LICENSE) 许可。
