import sympy as sp
from sympy_codegen import MathModel

def get_model():
    """
    实现 Fortran 子程序 STRESS_ROTATE 的逻辑:
    1. 将 6 分量应力向量还原为 3x3 对称矩阵 S
    2. 计算自旋张量 (Spin Tensor) W = 0.5 * (dudx - transpose(dudx))
    3. 计算应力增量 dS = W*S - S*W (基于 Jaumann 速率)
    4. 更新应力 S_new = S + dS * dt
    """
    
    # 1. 定义输入符号
    # stress: 6个分量
    stress_syms = [sp.Symbol(f"stress_{i}", real=True) for i in range(6)]
    # dudx: 3x3 矩阵分量
    dudx_syms = [sp.Symbol(f"dudx_{i}_{j}", real=True) for i in range(3) for j in range(3)]
    # dt: 时间步长
    dt = sp.Symbol("dt", real=True)
    
    # 2. 构造矩阵
    # 还原对称应力张量 S (注意 Fortran 索引与 Python 索引的对应)
    # stress: [sig11, sig22, sig33, sig12, sig23, sig13]
    S = sp.Matrix([
        [stress_syms[0], stress_syms[3], stress_syms[5]],
        [stress_syms[3], stress_syms[1], stress_syms[4]],
        [stress_syms[5], stress_syms[4], stress_syms[2]]
    ])
    
    DUDX = sp.Matrix(3, 3, dudx_syms)
    
    # 3. 计算旋转逻辑
    # 计算反对称部分的自旋张量 W
    W = sp.Rational(1, 2) * (DUDX - DUDX.T)
    
    # 计算应力变化率 (交换子 [W, S])
    dS = W * S - S * W
    
    # 更新应力
    S_new = S + dS * dt
    
    # 4. 提取输出分量 (写回 6 分量格式)
    stress_new_vec = [
        S_new[0, 0], # stress(1)
        S_new[1, 1], # stress(2)
        S_new[2, 2], # stress(3)
        S_new[0, 1], # stress(4)
        S_new[1, 2], # stress(5)
        S_new[0, 2]  # stress(6)
    ]
    
    # 5. 组装模型
    all_inputs = stress_syms + dudx_syms + [dt]
    input_names = [str(sym) for sym in all_inputs]
    
    return MathModel(
        inputs=all_inputs,
        outputs=stress_new_vec,
        name="stress_rotate",
        input_names=input_names,
        is_operator=True
    )