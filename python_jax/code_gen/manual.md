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

## 6. PeachPy Workflow (Custom Model)

Use this flow to generate a PeachPy script from a custom SymPy model, then assemble it into a `.obj` and C header.

### 6.1 Generate PeachPy Python Script

```bash
python .\sympy_codegen.py --task custom --script .\test_polystress.py --target peachpy
```

Output example:
- `polystress_nomullins_peachpy.py`

> Note: for `--task custom`, the script must define `get_model()`.

### 6.2 Assemble to Object + Header

```bash
python -m peachpy.x86_64 .\polystress_nomullins_peachpy.py -mabi=ms -mimage-format=ms-coff -o polystress_kernel.obj -emit-c-header polystress_kernel.h
```

Output example:
- `polystress_kernel.obj`
- `polystress_kernel.h`
