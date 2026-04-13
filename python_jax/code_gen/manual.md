# Finite Element Code Generator - User Manual

## 1. Introduction

This tool generates optimized C++, CUDA, and JAX computational kernels for Finite Element Analysis (FEA). It converts high-level mathematical definitions into highly efficient, low-level functions.

The generator supports four main types of tasks:
-   **Constitutive Kernels**: Generate material D-matrices.
-   **Stiffness Kernels**: Generate element stiffness matrices (`Ke`), either as a single kernel or a set of **Decoupled Operators**.
-   **Mass Kernels**: Generate element mass matrices (e.g., lumped mass for explicit dynamics).
-   **Custom Kernels**: Generate arbitrary mathematical models from a standalone Python script.

## 2. Command-Line Usage

The primary script is `sympy_codegen.py`.

### Arguments

-   `--task {constitutive,stiffness,mass,custom}`: **(Required)**
    -   `constitutive`: Material D-matrix. Requires `--material`.
    -   `stiffness`: Element Ke matrix (or operators). Requires `--element`.
    -   `mass`: Element mass matrix (or lumped mass). Requires `--element`.
    -   `custom`: Arbitrary model. Requires `--script`.
-   `--material <name>`: Material name (e.g., `isotropic`).
-   `--element <name>`: Element name (e.g., `tet4`, `hex8`).
-   `--script <path>`: Path to a Python script for custom tasks.
-   `--target {cpp,cuda,jax,peachpy,all}`: **(Required)** Target language.
-   `--output <path>`: (Optional) Output file or directory.
-   `--test`: (Optional) Generate CI test assets alongside kernel code (C++/Fortran test wrappers, `test_driver.py`, build scripts).
-   `--test-output-dir <path>`: (Optional) Directory for test assets (defaults to `--output` if omitted).

## 3. Advanced Features

### 3.1 Operator-Based Decoupling
For complex elements, a single monolithic kernel can be slow to compile and hard to vectorize. The generator supports splitting the calculation into modular **operators**:
1.  **dN_dnat**: Shape function derivatives in natural coordinates.
2.  **Mapping**: Jacobian calculation and physical coordinate derivatives (`dN_dx`).
3.  **Assembly**: Integration point contribution ($B^T D B \det(J) W$).
4.  **Lumped Mass**: Element-level mass distribution for explicit dynamics.

### 3.2 Fast Validation Solvers (JAX)
Two scripts are provided for rapid verification of generated JAX kernels:
-   `static.py`: Solves linear static problems using implicit integration.
-   `explicit.py`: Solves dynamic problems using the Central Difference method (explicit).

**Example Usage (Explicit):**
```bash
python explicit.py --model test_case/tet4_mat1_ex/tet4_mat1_ex.jsonc --element tet4 --material isotropic
```

## 4. Performance Optimizations

-   **Chunked CSE**: Common Subexpression Elimination is performed in row-level chunks. This reduces generation time for large matrices (like 24x24) from hours to seconds.
-   **Memory Alignment**: C++ outputs are structured to be "compiler-friendly" for auto-vectorization.
-   **JAX Unpacking**: JAX kernels automatically unpack `in_flat` into named variables for readability and performance.

## 5. Integration Example (Explicit Operator Mode)

Using decoupled operators in a JAX-based explicit solver:

```python
# 1. dN/dnat (Constant for Tet4)
dN_dnat = kernels["tet4_op_dN_dnat"](jnp.array([0.25, 0.25, 0.25]))

# 2. Mapping to physical space
map_output = kernels["tet4_op_mapping"](jnp.concatenate([coords_flat, dN_dnat]))
dN_dx, detJ = map_output[0:12], map_output[12]

# 3. Assemble Stiffness
Ke = kernels["tet4_op_assembly"](jnp.concatenate([dN_dx, d_matrix, jnp.array([detJ, 1/6])]))

# 4. Compute Lumped Mass
Me_lumped = kernels["tet4_op_lumped_mass"](jnp.concatenate([coords_flat, jnp.array([rho])]))
```

## 6. CI Testing Workflow

The `--test` flag generates a complete set of test assets that allow cross-backend numerical validation: **SymPy (reference) vs C++ vs Fortran**.

### 6.1 Generate Test Assets

Add `--test` to any code generation command. Use `--test-output-dir` to specify where test files go (defaults to `--output`).

**Example — Constitutive model (isotropic D-matrix):**
```bash
python sympy_codegen.py --task constitutive --material isotropic --target all --output generated/isotropic_D --test --test-output-dir generated/isotropic_D
```

**Example — Stiffness operators (tet4 element):**
```bash
python sympy_codegen.py --task stiffness --element tet4 --target all --output generated/tet4 --test --test-output-dir generated/tet4
```

This produces, for each model/operator, a subdirectory containing:

| File | Description |
|------|-------------|
| `kernel.cpp` | Generated C++ kernel |
| `kernel.f90` | Generated Fortran kernel |
| `main.cpp` | C++ test wrapper (reads stdin, writes stdout) |
| `main.f90` | Fortran test wrapper |
| `test_driver.py` | Python test driver (SymPy reference + subprocess comparison) |
| `build.sh` | Linux/macOS build script |
| `build.bat` | Windows build script |

### 6.2 Build

Enter the test directory and run the build script:

**Linux / macOS:**
```bash
cd generated/tet4/tet4_op_dN_dnat
bash build.sh
```

**Windows:**
```cmd
cd generated\tet4\tet4_op_dN_dnat
build.bat
```

The build script auto-detects available compilers:
- C++: tries `clang++`, `g++`, or MSVC `cl` (in that order)
- Fortran: uses `gfortran`

Compiler flags: `-O2 -fno-fast-math` (ensures IEEE 754 compliance for reproducible floating-point behavior).

Output executables: `kernel_cpp.exe` (or `kernel_cpp` on Linux) and `kernel_f90.exe` (or `kernel_f90`).

### 6.3 Run Tests

The `test_driver.py` generates random inputs, computes the reference result via SymPy lambdify, then feeds the same input to the compiled C++/Fortran executables and compares outputs.

**Basic usage (both backends):**
```bash
python test_driver.py --cpp-exe kernel_cpp.exe --f90-exe kernel_f90.exe
```

**C++ only:**
```bash
python test_driver.py --cpp-exe kernel_cpp.exe
```

**Custom tolerance and run count:**
```bash
python test_driver.py --cpp-exe kernel_cpp.exe --f90-exe kernel_f90.exe --n-runs 5000 --atol 1e-9 --rtol 1e-10
```

### 6.4 Test Driver CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-runs` | 1000 | Number of random test cases |
| `--atol` | 1e-10 | Absolute tolerance for `np.allclose` |
| `--rtol` | 1e-11 | Relative tolerance for `np.allclose` |
| `--seed` | 42 | Random seed for reproducibility |
| `--cpp-exe` | (none) | Path to compiled C++ executable |
| `--f90-exe` | (none) | Path to compiled Fortran executable |
| `--input-range` | 0.1 2.0 | Range for uniform random inputs |

At least one of `--cpp-exe` or `--f90-exe` must be provided.

### 6.5 Interpreting Results

On success:
```
Results: 1000/1000 passed, 0 failed (atol=1e-10, rtol=1e-11)
```

On failure, a detailed debug dump is printed showing each output value from all backends and the maximum difference, making it easy to identify which output diverged and by how much.

### 6.6 Tolerance Guidelines

- **atol=1e-10, rtol=1e-11** (defaults): Suitable for most operators. The `rtol` accounts for large dynamic range computations (e.g., matrix inversion producing values ~500, where absolute error ~1e-9 is within expected floating-point reordering precision).
- For operators with very small output magnitudes, you may need to relax `atol`. For very large magnitudes, `rtol` is the primary control.

