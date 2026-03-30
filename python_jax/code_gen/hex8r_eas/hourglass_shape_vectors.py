import sympy as sp
from python_jax.code_gen.sympy_codegen import MathModel

def get_model():
    """
    实现 Fortran 子程序 HOURGLASS_SHAPE_VECTORS 的逻辑:
    gammas = (SCALE_GAMMA / 8) * (h - BiI * (COORD^T * h))
    
    输入:
    - BiI: 形状函数导数 (8x3)
    - COORD: 节点物理坐标 (8x3)
    - h: 沙漏基向量 (8x4)
    - SCALE_GAMMA: 缩放因子 (标量)
    """
    
    # 1. 定义输入符号
    bii_syms = [sp.Symbol(f"BiI_{i}_{j}", real=True) for i in range(8) for j in range(3)]
    coord_syms = [sp.Symbol(f"COORD_{i}_{j}", real=True) for i in range(8) for j in range(3)]
    h_syms = [sp.Symbol(f"h_{i}_{j}", real=True) for i in range(8) for j in range(4)]
    scale_gamma = sp.Symbol("SCALE_GAMMA", real=True)
    
    BiI = sp.Matrix(8, 3, bii_syms)
    COORD = sp.Matrix(8, 3, coord_syms)
    H = sp.Matrix(8, 4, h_syms)
    
    # 2. 核心算法逻辑 (向量化处理 4 个沙漏模式)
    # Fortran 中的 h_dot_x = transpose(COORD) * h_i
    # 其中 h_i 是 h 矩阵的第 i 列 (8x1)
    
    one_over_eight = sp.Rational(1, 8)
    gammas_list = []
    
    for i in range(4):
        # 提取第 i 个沙漏模式向量 (8x1)
        h_i = H.col(i)
        
        # 计算 h_dot_x (3x1 向量): 对应 Fortran 中的 h_dot_x 累加循环
        # h_dot_x(j) = sum_{A=1}^8 h(A,i) * COORD(A,j)
        h_dot_x = COORD.T * h_i
        
        # 计算 gamma 向量 (8x1): 对应 Fortran 中的第二个 A 循环
        # gamma_i = (h_i - BiI * h_dot_x) * constant
        gamma_i = scale_gamma * one_over_eight * (h_i - BiI * h_dot_x)
        
        gammas_list.append(gamma_i)
    
    # 将 4 个 8x1 向量合并回 8x4 矩阵
    GAMMAS = sp.BlockMatrix([gammas_list]).as_explicit()
    
    # 3. 定义输出: 展平的 8x4 矩阵 (64个元素)
    # 按照 Fortran 顺序 (A=1..8, i=1..4)
    gammas_flat = [GAMMAS[a, i] for i in range(4) for a in range(8)]
    
    # 4. 映射输入
    all_inputs = bii_syms + coord_syms + h_syms + [scale_gamma]
    input_names = [str(sym) for sym in all_inputs]
    
    return MathModel(
        inputs=all_inputs,
        outputs=gammas_flat,
        name="hourglass_shape_vectors",
        input_names=input_names,
        is_operator=True
    )