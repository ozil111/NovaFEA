import sympy as sp
from definitions.abc import Material
from sympy_codegen import MathModel

class Isotropic(Material):
    """Linear isotropic elastic material."""
    def __init__(self):
        super().__init__("isotropic")

    def get_constitutive_model(self):
        """
        Defines the symbolic model for a linear isotropic material's D-matrix.
        """
        # 1. Define inputs (material parameters)
        mat_params = list(sp.symbols("E nu", real=True))
        E, nu = mat_params

        # 2. Define symbolic computation
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        
        D = sp.Matrix([
            [lam + 2*mu, lam,        lam,        0,  0,  0],
            [lam,        lam + 2*mu, lam,        0,  0,  0],
            [lam,        lam,        lam + 2*mu, 0,  0,  0],
            [0,          0,          0,          mu, 0,  0],
            [0,          0,          0,          0,  mu, 0],
            [0,          0,          0,          0,  0,  mu]
        ])

        # 3. Define outputs
        D_flat = [D[i, j] for i in range(6) for j in range(6)]
        
        # 4. Create and return the MathModel
        # Rename inputs to "in[i]" for C-style array access
        in_syms = [sp.Symbol(f"in[{i}]") for i in range(len(mat_params))]
        subs_map = dict(zip(mat_params, in_syms))
        
        outputs_subd = [expr.subs(subs_map) for expr in D_flat]
        model_name = f"{self.name}_D"
        
        return MathModel(inputs=in_syms, outputs=outputs_subd, name=model_name, input_names=[str(p) for p in mat_params])

