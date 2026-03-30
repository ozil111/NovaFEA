import sympy as sp
from python_jax.code_gen.sympy_codegen import MathModel

def get_model():
    """
    实现 Fortran 子程序 GET_DSTRAIN 的逻辑:
    1. 计算应变率张量 D = 0.5 * (dudx + transpose(dudx))
    2. 计算增量应变向量 dstrain = D * dt (其中剪切分量乘 2 转换为工程应变)
    """
    
    # 1. 定义输入符号: dudx (3x3) 和 dt (标量)
    dudx_syms = [sp.Symbol(f"dudx_{i}_{j}", real=True) for i in range(3) for j in range(3)]
    dt = sp.Symbol("dt", real=True)
    
    DUDX = sp.Matrix(3, 3, dudx_syms)
    
    # 2. 定义计算
    # 使用 SymPy 矩阵操作定义对称部分
    D = sp.Rational(1, 2) * (DUDX + DUDX.T)
    
    # 3. 定义输出向量 dstrain (长度为 6)
    # 对应关系 (Fortran 1-based -> Python 0-based):
    # dstrain(1)=D(1,1), (2)=D(2,2), (3)=D(3,3) -> 正应变
    # dstrain(4)=2*D(1,2), (5)=2*D(2,3), (6)=2*D(1,3) -> 工程剪切应变
    dstrain = [
        D[0, 0] * dt,      # dstrain[0]
        D[1, 1] * dt,      # dstrain[1]
        D[2, 2] * dt,      # dstrain[2]
        2 * D[0, 1] * dt,  # dstrain[3] -> gamma_xy
        2 * D[1, 2] * dt,  # dstrain[4] -> gamma_yz
        2 * D[0, 2] * dt   # dstrain[5] -> gamma_zx
    ]
    
    # 4. 映射输入
    all_inputs = dudx_syms + [dt]
    input_names = [str(sym) for sym in all_inputs]
    
    return MathModel(
        inputs=all_inputs,
        outputs=dstrain,
        name="get_dstrain",
        input_names=input_names,
        is_operator=True
    )