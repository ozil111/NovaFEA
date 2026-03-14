import sympy as sp
from definitions.abc import Material

class Isotropic(Material):
    """Linear isotropic elastic material."""
    def __init__(self):
        super().__init__("isotropic")

    def get_symbolic_model(self):
        """
        Defines the symbolic model for a linear isotropic material.

        Returns:
            - params ([E, nu]): List of SymPy symbols.
            - D (sympy.Matrix): 6x6 constitutive matrix.
        """
        E, nu = sp.symbols("E nu", real=True)
        
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
        
        return [E, nu], D
