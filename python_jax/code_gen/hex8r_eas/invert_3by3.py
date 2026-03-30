import sympy as sp
from python_jax.code_gen.sympy_codegen import MathModel  # pyright: ignore[reportImplicitRelativeImport]

def get_model():
    """
    实现 Fortran 子程序 INVERT_3X3 的逻辑:
    使用 SymPy 内置的 .inv() 自动推导 3x3 矩阵 A 的逆矩阵 AINV。
    """
    
    # 1. 定义输入符号: A (3x3)
    # A_0_0, A_0_1, ..., A_2_2
    a_syms = [sp.Symbol(f"A_{i}_{j}", real=True) for i in range(3) for j in range(3)]
    A = sp.Matrix(3, 3, a_syms)
    
    # 2. 定义计算
    # SymPy 会自动通过伴随矩阵和行列式计算逆矩阵，
    # 其生成的代数式在数学上等价于 Fortran 中手写的克莱姆法则展开。
    AINV = A.inv()
    
    # 3. 定义输出 (展平 3x3 矩阵 AINV)
    ainv_flat = [AINV[i, j] for i in range(3) for j in range(3)]
    
    # 4. 映射输入名称
    input_names = [str(sym) for sym in a_syms]
    
    return MathModel(
        inputs=a_syms,
        outputs=ainv_flat,
        name="invert_3x3",
        input_names=input_names,
        is_operator=True
    )