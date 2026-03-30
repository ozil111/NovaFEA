import sympy as sp
from python_jax.code_gen.sympy_codegen import MathModel

def get_model():
    """
    实现 Fortran 子程序 CALC_B_BAR 的逻辑:
    计算 8 节点单元的几何常数向量 BiI (通常对应 dN/dx, dN/dy, dN/dz 的几何部分)。
    输入为两个坐标分量数组 y(8) 和 z(8)。
    """
    
    # 1. 定义输入符号 (0-based indexing)
    y = [sp.Symbol(f"y_{i}", real=True) for i in range(8)]
    z = [sp.Symbol(f"z_{i}", real=True) for i in range(8)]
    
    inv12 = sp.Rational(1, 12)
    
    # 2. 定义计算公式 (将 Fortran 1-based 转换为 Python 0-based)
    # 虽然公式较长，但我们保持其结构清晰，SymPy 会在生成代码时自动进行 CSE 优化
    
    bii = [sp.S.Zero] * 8
    
    bii[0] = -inv12 * (y[1]*(z[2]+z[3]-z[4]-z[5]) + y[2]*(-z[1]+z[3]) + 
                       y[3]*(-z[1]-z[2]+z[4]+z[7]) + y[4]*(z[1]-z[3]+z[5]-z[7]) + 
                       y[5]*(z[1]-z[4]) + y[7]*(-z[3]+z[4]))
    
    bii[1] =  inv12 * (y[0]*(z[2]+z[3]-z[4]-z[5]) + y[2]*(-z[0]-z[3]+z[5]+z[6]) + 
                       y[3]*(-z[0]+z[2]) + y[4]*(z[0]-z[5]) + 
                       y[5]*(z[0]-z[2]+z[4]-z[6]) + y[6]*(-z[2]+z[5]))
    
    bii[2] = -inv12 * (y[0]*(z[1]-z[3]) + y[1]*(-z[0]-z[3]+z[5]+z[6]) + 
                       y[3]*(z[0]+z[1]-z[6]-z[7]) + y[5]*(-z[1]+z[6]) + 
                       y[6]*(-z[1]+z[3]-z[5]+z[7]) + y[7]*(z[3]-z[6]))
    
    bii[3] = -inv12 * (y[0]*(z[1]+z[2]-z[4]-z[7]) + y[1]*(-z[0]+z[2]) + 
                       y[2]*(-z[0]-z[1]+z[6]+z[7]) + y[4]*(z[0]-z[7]) + 
                       y[6]*(-z[2]+z[7]) + y[7]*(z[0]-z[2]+z[4]-z[6]))
    
    bii[4] =  inv12 * (y[0]*(z[1]-z[3]+z[5]-z[7]) + y[1]*(-z[0]+z[5]) + 
                       y[3]*(z[0]-z[7]) + y[5]*(-z[0]-z[1]+z[6]+z[7]) + 
                       y[6]*(-z[5]+z[7]) + y[7]*(z[0]+z[3]-z[5]-z[6]))
    
    bii[5] =  inv12 * (y[0]*(z[1]-z[4]) + y[1]*(-z[0]+z[2]-z[4]+z[6]) + 
                       y[2]*(-z[1]+z[6]) + y[4]*(z[0]+z[1]-z[6]-z[7]) + 
                       y[6]*(-z[1]-z[2]+z[4]+z[7]) + y[7]*(z[4]-z[6]))
    
    bii[6] =  inv12 * (y[1]*(z[2]-z[5]) + y[2]*(-z[1]+z[3]-z[5]+z[7]) + 
                       y[3]*(-z[2]+z[7]) + y[4]*(z[5]-z[7]) + 
                       y[5]*(z[1]+z[2]-z[4]-z[7]) + y[7]*(-z[2]-z[3]+z[4]+z[5]))
    
    bii[7] = -inv12 * (y[0]*(z[3]-z[4]) + y[2]*(-z[3]+z[6]) + 
                       y[3]*(-z[0]+z[2]-z[4]+z[6]) + y[4]*(z[0]+z[3]-z[5]-z[6]) + 
                       y[5]*(z[4]-z[6]) + y[6]*(-z[2]-z[3]+z[4]+z[5]))

    # 3. 定义输出
    input_names = [str(s) for s in y] + [str(s) for s in z]
    
    return MathModel(
        inputs=y + z,
        outputs=bii,
        name="calc_b_bar",
        input_names=input_names,
        is_operator=True
    )