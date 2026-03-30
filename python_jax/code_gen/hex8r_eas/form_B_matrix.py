import sympy as sp
from sympy_codegen import MathModel

def get_model():
    """
    实现 Fortran 子程序 FORM_B_MATRIX 的逻辑:
    将形状函数在物理坐标系下的导数 BiI (8x3) 组装成应变-位移矩阵 B (6x24)。
    """
    
    # 1. 定义输入符号: BiI (8x3)
    # 代表 8 个节点对 x, y, z 的偏导数
    bii_syms = [sp.Symbol(f"BiI_{i}_{j}", real=True) for i in range(8) for j in range(3)]
    BiI = sp.Matrix(8, 3, bii_syms)
    
    # 2. 初始化 B 矩阵 (6行 x 24列)
    B = sp.zeros(6, 24)
    
    # 3. 填充 B 矩阵逻辑
    # 遍历 8 个节点
    for k in range(8):
        # 对应 Fortran 中的列偏移: 3*K-2, 3*K-1, 3*K
        # 转换到 Python 0-based 索引: col_x, col_y, col_z
        cx = 3 * k
        cy = 3 * k + 1
        cz = 3 * k + 2
        
        # 正应变分量 (Normal Strains)
        B[0, cx] = BiI[k, 0] # epsilon_xx
        B[1, cy] = BiI[k, 1] # epsilon_yy
        B[2, cz] = BiI[k, 2] # epsilon_zz
        
        # 剪切应变分量 (Shear Strains - 工程应变)
        # gamma_xy
        B[3, cx] = BiI[k, 1]
        B[3, cy] = BiI[k, 0]
        
        # gamma_yz
        B[4, cy] = BiI[k, 2]
        B[4, cz] = BiI[k, 1]
        
        # gamma_zx
        B[5, cx] = BiI[k, 2]
        B[5, cz] = BiI[k, 0]
        
    # 4. 定义输出: 展平的 6x24 矩阵 (共 144 个元素)
    b_flat = [B[i, j] for i in range(6) for j in range(24)]
    
    # 5. 映射输入名称
    input_names = [str(sym) for sym in bii_syms]
    
    return MathModel(
        inputs=bii_syms,
        outputs=b_flat,
        name="form_b_matrix",
        input_names=input_names,
        is_operator=True
    )