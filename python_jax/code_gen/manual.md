# Finite Element Code Generator - User Manual

## 1. Introduction

This tool generates optimized C++, CUDA, and JAX computational kernels for Finite Element Analysis (FEA). It takes high-level mathematical definitions of elements and materials and converts them into low-level, ready-to-use functions.

The generator operates under a "Hybrid Decoupling" architecture, creating two distinct types of functions:
-   **Constitutive Kernels**: Calculate a material's D-matrix.
-   **Stiffness Kernels**: Calculate an element's stiffness matrix (`Ke`) using a given D-matrix.

## 2. Prerequisites

-   Python 3.9+
-   SymPy: `pip install sympy`
-   JAX (optional, for JAX target): `pip install jax jaxlib`

## 3. Usage

The primary script is `sympy_codegen.py`. It is a command-line tool that dispatches generation tasks.

### Command-Line Arguments

-   `--task {constitutive,stiffness}`: **(Required)** The type of kernel to generate.
    -   `constitutive`: Generate a material D-matrix kernel. Requires `--material`.
    -   `stiffness`: Generate an element stiffness matrix (`Ke`) kernel. Requires `--element`.
-   `--material <name>`: The name of the material to use (e.g., `isotropic`). Required for `--task=constitutive`.
-   `--element <name>`: The name of the element to use (e.g., `tet4`). Required for `--task=stiffness`.
-   `--target {cpp,cuda,jax,all}`: **(Required)** The target programming language.
-   `--output <path>`: (Optional) The path for the output file. If not provided, a default name is generated (e.g., `isotropic_D_gen.cpp`).

## 4. Code Generation Examples

### Task 1: Generate a Material Kernel

This task creates a function that computes the 6x6 constitutive matrix (D-matrix).

**Command:**
```bash
python sympy_codegen.py --task constitutive --material isotropic --target cpp
```

**Output:**
This command generates a file named `isotropic_D_gen.cpp` containing the following function:

```cpp
/**
 * @brief Computes the isotropic_D kernel.
 * 
 * @param in Input array (const double*). Layout:
 *   - in[0]: E
 *   - in[1]: nu
 * 
 * @param out Output array (double*). Layout:
 *   - out[0..35]: Flattened result, row-major order.
 */
inline void compute_isotropic_D(const double* in, double* out) {
    // ... optimized code to compute 36 components of D ...
}
```

### Task 2: Generate an Element Stiffness Kernel

This task creates a function that computes the element stiffness matrix (`Ke`).

**Command:**
```bash
python sympy_codegen.py --task stiffness --element tet4 --target cpp
```

**Output:**
This command generates a file named `tet4_Ke_gen.cpp` containing the following function:

```cpp
/**
 * @brief Computes the tet4_Ke kernel.
 * 
 * @param in Input array (const double*). Layout:
 *   - in[0]: coord[0][0]
 *   ...
 *   - in[11]: coord[3][2]
 *   - in[12]: D[0][0]
 *   ...
 *   - in[47]: D[5][5]
 * 
 * @param out Output array (double*). Layout:
 *   - out[0..143]: Flattened result, row-major order.
 */
inline void compute_tet4_Ke(const double* in, double* out) {
    // ... optimized code for B^T * D * B integration ...
}
```

## 5. Integration Example (Conceptual C++)

To use the generated kernels in your main C++ application (e.g., `hyperfem`), you would include or link the generated files and call the functions in sequence.

```cpp
// main_fea_solver.cpp

#include "isotropic_D_gen.cpp" // Or link the compiled object
#include "tet4_Ke_gen.cpp"     // Or link the compiled object

void calculate_element_stiffness() {
    // 1. Prepare input data for a single element
    double node_coords[12] = { ... };      // Nodal coordinates
    double material_params[2] = { E, nu }; // Material properties

    // 2. Prepare storage for intermediate and final results
    double D_matrix[36];
    double Ke_matrix[144];

    // 3. Call the material kernel to compute the D-matrix
    compute_isotropic_D(material_params, D_matrix);

    // 4. Prepare the input array for the stiffness kernel
    double stiffness_kernel_input[48];
    for (int i = 0; i < 12; ++i) stiffness_kernel_input[i] = node_coords[i];
    for (int i = 0; i < 36; ++i) stiffness_kernel_input[i+12] = D_matrix[i];

    // 5. Call the stiffness kernel to compute Ke
    compute_tet4_Ke(stiffness_kernel_input, Ke_matrix);

    // 6. Now Ke_matrix can be used for global assembly
    // ...
}
```
