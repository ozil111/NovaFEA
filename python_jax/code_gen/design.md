# Finite Element Code Generator - Design Document

## 1. Overview

This project is a software tool for the automated generation of efficient, low-level FEA computational kernels (C++, CUDA, JAX). It uses SymPy as a symbolic engine to convert high-level formulas into optimized source code.

## 2. Architecture: Decoupled Operators

The project has evolved from a simple **Hybrid Decoupling** (Materials vs. Elements) to a more granular **Decoupled Operator** architecture for elements.

### 2.1 Why Operators?
Monolithic kernels for higher-order elements (like Hex8) lead to:
-   **Compilation Time**: SymPy CSE can take hours on a 24x24 matrix.
-   **Vectorization Bottlenecks**: Complex data layouts (AoS) are hard for compilers to vectorize.

### 2.2 Operator Structure
A stiffness calculation is now split into three decoupled phases:
1.  **Natural Coordinate Operator (`dN_dnat`)**: Handles the shape function derivatives.
2.  **Mapping Operator (`J` & `B`)**: Computes the Jacobian, its inverse, and maps derivatives to physical space.
3.  **Assembly Operator (`BᵀDB`)**: Computes the final matrix contribution at a single integration point.

This design enables **SoA (Structure of Arrays)** output for mapping, allowing SIMD vectorization in the C++ assembly loop.

## 3. High-Performance Optimizations

### 3.1 Row-level Chunked CSE
The `FEACompiler` performs Common Subexpression Elimination (CSE) in chunks. Instead of processing 576 elements simultaneously, it processes row-by-row (e.g., 24 elements at a time). This provides:
-   **Linear Scalability**: Generation time scales linearly with matrix size.
-   **Memory Efficiency**: Reduced memory usage during the symbolic optimization phase.

### 3.2 Compiler-Friendly C++
-   **SIMD Ready**: Output data layouts prioritize contiguous memory access.
-   **Optimization Hints**: Kernels use `__restrict__` and `[[gnu::always_inline]]` to assist compiler auto-vectorization and inter-procedural analysis.

## 4. Custom Task Extensibility

The tool now supports a generic `--task custom` mode. Users can provide a standalone Python script containing a `get_model()` function. This decouples the code generator's optimization and printing engine from the FEA-specific `Element`/`Material` classes.

## 5. How to Extend

### 5.1 Implement a New Operator-based Element
To support operators, implement `get_stiffness_operators(self)` in your element class:
```python
def get_stiffness_operators(self):
    op_dN = MathModel(...)
    op_map = MathModel(...)
    op_asm = MathModel(...)
    return [op_dN, op_map, op_asm]
```

### 5.2 Standalone Mathematical Script
Create a script (e.g., `tensor_ops.py`) that returns one or more `MathModel` objects:
```python
def get_model():
    # Symbolic tensor rotation logic...
    return MathModel(inputs, outputs, name="rotate_tensor")
```
Run with: `python sympy_codegen.py --task custom --script tensor_ops.py`
