import sympy as sp
from sympy_codegen import MathModel

def get_model():
    """
    Implement the Fortran subroutine JACOBIAN_CENTER logic:
    JAC = transpose(matmul(transpose(XiI), COORD) * (1/8))

    COORD is an 8x3 nodal coordinate matrix.
    XiI is the 8x3 constant derivative-sign matrix in natural coordinates.
    """
    
    # 1. Define input symbols: COORD (8x3)
    # COORD_0_0, COORD_0_1, ..., COORD_7_2
    coord_syms = [sp.Symbol(f"COORD_{i}_{j}", real=True) for i in range(8) for j in range(3)]
    COORD = sp.Matrix(8, 3, coord_syms)
    
    # 2. Define constant matrix XiI (8x3)
    # Sign pattern of shape-function derivatives for a standard 8-node hex element at center.
    xi_val = [
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
    ]
    XiI = sp.Matrix(xi_val)
    
    one_over_eight = sp.Rational(1, 8)
    
    # 3. Equivalent form of the Fortran two-step expression:
    # transpose((XiI.T * COORD) * 1/8) == (COORD.T * XiI) * 1/8
    JAC = (COORD.T * XiI) * one_over_eight
    
    # 4. Define output (flattened 3x3 matrix)
    jac_flat = [JAC[i, j] for i in range(3) for j in range(3)]
    
    # 5. Map input names
    input_names = [str(sym) for sym in coord_syms]
    
    return MathModel(
        inputs=coord_syms,
        outputs=jac_flat,
        name="jacobian_center",
        input_names=input_names,
        is_operator=True
    )