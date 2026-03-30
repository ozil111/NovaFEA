import sympy as sp
from sympy_codegen import MathModel

def get_model():
    """
    实现 Fortran 子程序 HUGHES_WINGET_ROTATION 的逻辑:
    1. 计算反对称张量 DeltaW = 0.5 * (dL - dL.T) * dt
    2. 计算 Cayley 变换: DeltaR = (I - 0.5*DeltaW)^-1 * (I + 0.5*DeltaW)
    """
    
    # 1. 定义输入符号: dL (3x3) 和 dt (标量)
    dl_syms = [sp.Symbol(f"dL_{i}_{j}", real=True) for i in range(3) for j in range(3)]
    dt = sp.Symbol("dt", real=True)
    
    DL = sp.Matrix(3, 3, dl_syms)
    I = sp.eye(3) # 3x3 单位矩阵
    
    # 2. 按照算法逻辑计算
    # 计算反对称部分 (Skew-symmetric part)
    # 注意: Fortran 代码中 DeltaW 已经包含了 dt
    half = sp.Rational(1, 2)
    DeltaW = half * (DL - DL.T) * dt
    
    # 计算中间矩阵
    A = I - half * DeltaW
    B = I + half * DeltaW
    
    # 计算增量旋转张量 DeltaR
    # SymPy 的 .inv() 会自动处理 3x3 矩阵求逆的代数展开
    DeltaR = A.inv() * B
    
    # 3. 定义输出: 展平的 3x3 矩阵
    deltar_flat = [DeltaR[i, j] for i in range(3) for j in range(3)]
    
    # 4. 映射输入
    all_inputs = dl_syms + [dt]
    input_names = [str(sym) for sym in all_inputs]
    
    return MathModel(
        inputs=all_inputs,
        outputs=deltar_flat,
        name="hughes_winget_rotation",
        input_names=input_names,
        is_operator=True
    )