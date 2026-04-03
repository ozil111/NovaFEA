import sympy as sp
from sympy_codegen import MathModel

def get_model():
    """
    Defines a generic 3x2 matrix multiplied by a 2x3 matrix.
    A (3x2) * B (2x3) = C (3x3)
    """
    # 1. Define inputs
    A_syms = [sp.Symbol(f"A_{i}_{j}", real=True) for i in range(3) for j in range(2)]
    B_syms = [sp.Symbol(f"B_{i}_{j}", real=True) for i in range(2) for j in range(3)]
    
    A = sp.Matrix(3, 2, A_syms)
    B = sp.Matrix(2, 3, B_syms)
    
    # 2. Define computation
    C = A * B
    
    # 3. Define outputs
    C_flat = [C[i, j] for i in range(3) for j in range(3)]
    
    # 4. Map to flat inputs
    all_inputs = A_syms + B_syms
    input_names = [str(sym) for sym in all_inputs]
    
    # 5. Create output names (row-major order)
    output_names = [f"C_{i}_{j}" for i in range(3) for j in range(3)]
    
    return MathModel(
        inputs=all_inputs, 
        outputs=C_flat, 
        name="custom_matmul", 
        input_names=input_names,
        output_names=output_names,
        is_operator=True
    )
