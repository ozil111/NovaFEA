import sympy as sp
from sympy.printing.c import ccode
import time

def generate_tet4_cpp():
    # 1. 定义节点坐标符号 (设为 real=True 方便 sympy 简化代数式)
    x0, y0, z0 = sp.symbols('x0 y0 z0', real=True)
    x1, y1, z1 = sp.symbols('x1 y1 z1', real=True)
    x2, y2, z2 = sp.symbols('x2 y2 z2', real=True)
    x3, y3, z3 = sp.symbols('x3 y3 z3', real=True)

    # 2. 定义 6x6 本构矩阵符号 (D_0_0 到 D_5_5)
    D = sp.Matrix(6, 6, lambda i, j: sp.Symbol(f'D_{i}_{j}', real=True))

    # 3. 单元几何与雅可比矩阵计算 (与 JAX 逻辑完全一一对应)
    coords = sp.Matrix([
        [x0, y0, z0],
        [x1, y1, z1],
        [x2, y2, z2],
        [x3, y3, z3]
    ])

    # 母单元形函数对天然坐标的梯度 (dN/d_xi)
    dN_dxi = sp.Matrix([
        [-1, -1, -1],
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1]
    ])

    # J = coords^T * dN_dxi (注意 SymPy 中的矩阵乘法方向)
    J = coords.T * dN_dxi
    detJ = J.det()
    vol = detJ / 6.0
    
    # 获取全局导数 dN/dx = dN/dxi * J^-1
    invJ = J.inv()
    dN_dx = dN_dxi * invJ

    # 4. 组装 B 矩阵 (6 x 12)
    B = sp.zeros(6, 12)
    for i in range(4):
        B[0, 3*i]   = dN_dx[i, 0]
        B[1, 3*i+1] = dN_dx[i, 1]
        B[2, 3*i+2] = dN_dx[i, 2]
        B[3, 3*i]   = dN_dx[i, 1]
        B[3, 3*i+1] = dN_dx[i, 0]
        B[4, 3*i+1] = dN_dx[i, 2]
        B[4, 3*i+2] = dN_dx[i, 1]
        B[5, 3*i]   = dN_dx[i, 2]
        B[5, 3*i+2] = dN_dx[i, 0]

    # 5. 符号计算刚度矩阵 K = B^T * D * B * vol
    # 这一步会产生极其庞大的符号表达式
    K = B.T * D * B * vol

    # 将 12x12 矩阵展平为一维列表 (行主序 Row-major，对应 C++ 的 1D 数组)
    K_flat = [K[i, j] for i in range(12) for j in range(12)]

    # 6. 核心：公共子表达式消除 (CSE, Common Subexpression Elimination)
    # 自动寻找重复计算的项，提取为 x4, x5...
    # start=4 是为了贴合你 C++ 代码中从 x4 开始的习惯
    replacements, reduced_exprs = sp.cse(K_flat, symbols=sp.numbered_symbols("x", start=4))

    # 7. 格式化输出为 C++ 代码
    cpp_src = "    // --- 3. 预计算若干与坐标相关的中间量（由 SymPy 自动生成） ---\n"
    for var, expr in replacements:
        cpp_src += f"    double {var} = {ccode(expr)};\n"

    cpp_src += "\n    // --- 5. 组装 12x12 单元刚度矩阵 Ke ---\n"
    for i, expr in enumerate(reduced_exprs):
        cpp_src += f"    Ke[{i}] = {ccode(expr)};\n"

    return cpp_src

if __name__ == "__main__":
    print("🚀 正在启动 SymPy 符号推导与 C++ 代码生成，请稍候...")
    start_time = time.time()
    
    code = generate_tet4_cpp()
    
    end_time = time.time()
    output_path = "tet4_sympy_codegen.cpp"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"✅ 生成完毕！耗时: {end_time - start_time:.2f} 秒")
    print(f"已写入文件: {output_path}")