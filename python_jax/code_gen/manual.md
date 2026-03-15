# Finite Element Code Generator - User Manual

## 1. Introduction

This tool generates optimized C++, CUDA, and JAX computational kernels for Finite Element Analysis (FEA). It converts high-level mathematical definitions into highly efficient, low-level functions.

The generator supports three main types of tasks:
-   **Constitutive Kernels**: Generate material D-matrices.
-   **Stiffness Kernels**: Generate element stiffness matrices (`Ke`), either as a single kernel or a set of **Decoupled Operators**.
-   **Custom Kernels**: Generate arbitrary mathematical models from a standalone Python script.

## 2. Command-Line Usage

The primary script is `sympy_codegen.py`.

### Arguments

-   `--task {constitutive,stiffness,custom}`: **(Required)**
    -   `constitutive`: Material D-matrix. Requires `--material`.
    -   `stiffness`: Element Ke matrix (or operators). Requires `--element`.
    -   `custom`: Arbitrary model. Requires `--script`.
-   `--material <name>`: Material name (e.g., `isotropic`).
-   `--element <name>`: Element name (e.g., `hex8`).
-   `--script <path>`: Path to a Python script for custom tasks.
-   `--target {cpp,cuda,jax,all}`: **(Required)** Target language.
-   `--output <path>`: (Optional) Output file or directory.

## 3. Advanced Features

### 3.1 Operator-Based Decoupling (New!)
For complex elements like `Hex8`, a single monolithic kernel can be slow to compile and hard to vectorize. The generator now supports splitting the calculation into modular **operators**:
1.  **dN_dnat**: Shape function derivatives in natural coordinates.
2.  **Mapping**: Jacobian calculation and physical coordinate derivatives (`dN_dx`).
3.  **Assembly**: Integration point contribution ($B^T D B \det(J) W$).

This approach enables:
-   **SoA (Structure of Arrays) Layout**: Output data is arranged for optimal SIMD vectorization.
-   **Vectorization Hints**: C++ kernels include `__restrict__` and `[[gnu::always_inline]]`.

### 3.2 Custom Task Interface
You can generate code for any formula by providing a script with a `get_model()` function:

**Example `my_formula.py`:**
```python
import sympy as sp
from sympy_codegen import MathModel

def get_model():
    x, y = sp.symbols("x y")
    return MathModel(inputs=[x, y], outputs=[x*y, x+y], name="my_add_mul")
```
**Command:**
```bash
python sympy_codegen.py --task custom --script my_formula.py --target cpp
```

## 4. Performance Optimizations

-   **Chunked CSE**: Common Subexpression Elimination is performed in row-level chunks. This reduces generation time for large matrices (like 24x24) from hours to seconds.
-   **Memory Alignment**: C++ outputs are structured to be "compiler-friendly" for auto-vectorization.
-   **JAX Unpacking**: JAX kernels automatically unpack `in_flat` into named variables for readability and performance.

## 5. Integration Example (Operator Mode)

Using decoupled operators in a C++ solver:

```cpp
#include "hex8_op_dN_dnat_gen.cpp"
#include "hex8_op_mapping_gen.cpp"
#include "hex8_op_assembly_gen.cpp"

void compute_hex8_stiffness() {
    double Ke[576] = {0}; // 24x24
    
    // Gauss Loop (2x2x2)
    for (int gp = 0; gp < 8; ++gp) {
        double dN_dnat[24], dN_dx[24], detJ;
        
        // 1. Natural derivatives
        compute_hex8_op_dN_dnat(gp_coords[gp], dN_dnat);
        
        // 2. Map to physical space (SoA output)
        double map_in[48]; // coords + dN_dnat
        double map_out[25]; // dN_dx + detJ
        compute_hex8_op_mapping(map_in, map_out);
        
        // 3. Assemble point contribution
        double asm_out[576];
        compute_hex8_op_assembly(asm_in, asm_out);
        
        // Sum contributions...
    }
}
```
