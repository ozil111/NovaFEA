# Finite Element Code Generator - Design Document

## 1. Overview

This project is a software tool for the automated generation of efficient, low-level FEA computational kernels (C++, CUDA, JAX). It uses SymPy as a symbolic engine to convert high-level formulas into optimized source code.

## 2. Architecture: Decoupled Operators

The project uses a **Decoupled Operator** architecture, splitting monolithic calculations into modular, reusable kernels.

### 2.1 Why Operators?
-   **Compilation Time**: SymPy CSE can take hours on monolithic 24x24 or larger matrices. Chunks are much faster.
-   **Vectorization**: Modular kernels with SoA (Structure of Arrays) layouts are easier for compilers to auto-vectorize.
-   **Reusability**: Operators like `dN_dnat` or `Mapping` can be reused across different solver types (Static, Explicit, Modal).

### 2.2 Operator Structure for Stiffness
A stiffness calculation is split into three decoupled phases:
1.  **Natural Coordinate Operator (`dN_dnat`)**: Handles the shape function derivatives.
2.  **Mapping Operator (`J` & `B`)**: Computes the Jacobian, its inverse, and maps derivatives to physical space.
3.  **Assembly Operator (`BᵀDB`)**: Computes the final matrix contribution at a single integration point.

### 2.3 Support for Explicit Dynamics
To support explicit solvers (e.g., Central Difference), the architecture includes:
-   **Mass Operators**: Specifically `op_lumped_mass` to compute the nodal distribution of mass based on element geometry and density.
-   **Fast Validation Solvers**: A JAX-based `explicit.py` script that utilizes these operators to perform time-marching simulations without the overhead of a full C++ system.

## 3. High-Performance Optimizations

### 3.1 Row-level Chunked CSE
The `FEACompiler` performs Common Subexpression Elimination (CSE) in chunks. Instead of processing the entire output simultaneously, it processes row-by-row (e.g., 12 or 24 elements). This provides **linear scalability** in generation time relative to matrix size.

### 3.2 Compiler-Friendly C++
-   **Memory Alignment**: Contiguous layouts for SIMD-ready access.
-   **Optimization Hints**: Use of `__restrict__` and `[[gnu::always_inline]]`.

## 4. Hybrid Decoupling (Material vs. Element)
The compiler separates the **Material Constitutive Law** (D-matrix) from the **Element Kinematics** (B-matrix). The D-matrix is computed once per element (or per integration point) and passed as a flat array to the element's assembly operator.

## 5. How to Extend

### 5.1 Implement a New Operator-based Element
Implement the following methods in the `Element` class:
-   `get_stiffness_operators(self)`: Return `[op_dN, op_map, op_asm]`.
-   `get_mass_operators(self)`: Return `[op_mass]` (e.g., `op_lumped_mass`).

### 5.2 Standalone Mathematical Script
Use `--task custom` with a script containing `get_model()` to generate arbitrary kernels.
