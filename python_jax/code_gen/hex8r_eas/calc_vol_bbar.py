import sympy as sp
from sympy_codegen import MathModel

def get_model_vol_bbar():
    # 1. 定义输入 (B1I 和 X 各 8 个分量)
    b1i_syms = [sp.Symbol(f"B1I_{i}", real=True) for i in range(8)]
    x_syms = [sp.Symbol(f"X_{i}", real=True) for i in range(8)]
    
    B1I = sp.Matrix(8, 1, b1i_syms)
    X = sp.Matrix(8, 1, x_syms)
    
    # 2. 计算 V = B1I · X
    V = X.dot(B1I)
    
    return MathModel(
        inputs=b1i_syms + x_syms,
        outputs=[V],
        name="calc_vol_bbar",
        input_names=[str(s) for s in b1i_syms + x_syms],
        is_operator=True
    )