import sympy as sp
from sympy_codegen import MathModel

def get_model():
    """
    实现 Fortran 子程序 CALC_MISES_STRESS 的逻辑:
    1. 构造应力张量 S
    2. 计算静水压力 P = tr(S)/3 和偏应力张量 S_dev = S - P*I
    3. 计算 Von Mises 应力 = sqrt(1.5 * (S_dev : S_dev))
    """
    
    # 1. 定义输入符号: stress(6)
    # 顺序: sig_xx, sig_yy, sig_zz, tau_xy, tau_yz, tau_xz
    stress_syms = [sp.Symbol(f"stress_{i}", real=True) for i in range(6)]
    
    # 2. 构造对称应力矩阵 S
    S = sp.Matrix([
        [stress_syms[0], stress_syms[3], stress_syms[5]],
        [stress_syms[3], stress_syms[1], stress_syms[4]],
        [stress_syms[5], stress_syms[4], stress_syms[2]]
    ])
    
    # 3. 计算偏应力张量 (Deviatoric Stress)
    # P = (sig_xx + sig_yy + sig_zz) / 3
    hydrostatic_pressure = S.trace() / 3
    I = sp.eye(3)
    S_dev = S - hydrostatic_pressure * I
    
    # 4. 计算 Von Mises 应力
    # 公式: sigma_v = sqrt(3/2 * s_ij * s_ij)
    # 在 SymPy 中，s_ij * s_ij 等价于矩阵所有元素平方和
    # 我们利用矩阵的 Frobenius 范数平方: trace(S_dev * S_dev)
    inner_product = (S_dev * S_dev).trace()
    mises_stress = sp.sqrt(sp.Rational(3, 2) * inner_product)
    
    # 5. 定义输出 (标量)
    outputs = [mises_stress]
    
    # 6. 映射输入名称
    input_names = [str(sym) for sym in stress_syms]
    
    return MathModel(
        inputs=stress_syms,
        outputs=outputs,
        name="calc_mises_stress",
        input_names=input_names,
        is_operator=True
    )