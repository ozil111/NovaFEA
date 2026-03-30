import sympy as sp
from python_jax.code_gen.sympy_codegen import MathModel

def get_model():
    xi, eta, zeta = sp.symbols('xi eta zeta', real=True)
    
    # 定义 8 个节点的局部坐标
    corners = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
    ]
    
    # 自动生成 N_i = 1/8 * (1 + xi*xi_i) * (1 + eta*eta_i) * (1 + zeta*zeta_i)
    N = []
    for c in corners:
        Ni = sp.Rational(1, 8) * (1 + c[0]*xi) * (1 + c[1]*eta) * (1 + c[2]*zeta)
        N.append(Ni)
        
    return MathModel(
        inputs=[xi, eta, zeta],
        outputs=N,
        name="calc_shape_functions",
        input_names=["xi", "eta", "zeta"],
        is_operator=True
    )