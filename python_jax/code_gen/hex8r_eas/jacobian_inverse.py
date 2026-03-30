import sympy as sp
from sympy_codegen import MathModel

def get_model():
    """
    实现 Fortran 子程序 JACOBIAN_INVERSE 的逻辑:
    1. 计算 3x3 矩阵 JAC 的行列式 DETJ
    2. 计算 JAC 的逆矩阵 JACINV
    """
    
    # 1. 定义输入符号: JAC (3x3)
    # 按照行优先顺序定义: JAC_0_0, JAC_0_1 ... JAC_2_2
    jac_syms = [sp.Symbol(f"JAC_{i}_{j}", real=True) for i in range(3) for j in range(3)]
    JAC = sp.Matrix(3, 3, jac_syms)
    
    # 2. 定义计算
    # 直接使用 SymPy 内置函数计算行列式和逆矩阵
    # SymPy 会自动推导出与你 Fortran 代码中一致的代数表达式
    det_j = JAC.det()
    JAC_inv = JAC.inv()
    
    # 3. 定义输出
    # 输出包含: 行列式 (1个值) + 逆矩阵展平 (9个值)
    jac_inv_flat = [JAC_inv[i, j] for i in range(3) for j in range(3)]
    outputs = [det_j] + jac_inv_flat
    
    # 4. 映射输入名称
    input_names = [str(sym) for sym in jac_syms]
    
    return MathModel(
        inputs=jac_syms,
        outputs=outputs,
        name="jacobian_inverse",
        input_names=input_names,
        is_operator=True
    )