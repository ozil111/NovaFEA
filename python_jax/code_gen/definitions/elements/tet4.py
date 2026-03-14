import sympy as sp
from definitions.abc import Element
from sympy_codegen import MathModel

class Tet4(Element):
    """4-node tetrahedral element."""
    def __init__(self):
        super().__init__("tet4")

    def get_stiffness_model(self):
        """
        Creates the symbolic model for a Tet4 stiffness kernel.
        This kernel takes nodal coordinates and D-matrix components as flat input.
        """
        # 1. Define inputs: 12 for coordinates, 36 for D-matrix
        coord_syms = [sp.Symbol(f"c{i}", real=True) for i in range(12)]
        D_syms = [sp.Symbol(f"D{i}", real=True) for i in range(36)]
        
        all_inputs = coord_syms + D_syms
        input_names = [f"coord[{i//3}][{i%3}]" for i in range(12)] + [f"D[{i//6}][{i%6}]" for i in range(36)]

        # Rename symbols to "in[i]" for C-style array access
        in_syms = [sp.Symbol(f"in[{i}]", real=True) for i in range(len(all_inputs))]
        subs_map = dict(zip(all_inputs, in_syms))

        coords = sp.Matrix(4, 3, lambda i, j: coord_syms[i * 3 + j]).subs(subs_map)
        D = sp.Matrix(6, 6, lambda i, j: D_syms[i * 6 + j]).subs(subs_map)

        # 2. Finite Element Formulation
        xi, eta, zeta = sp.symbols("xi eta zeta", real=True)
        N_list = [1 - xi - eta - zeta, xi, eta, zeta]
        dN_dxi = sp.Matrix(4, 3, lambda i, j: sp.diff(N_list[i], (xi, eta, zeta)[j]))

        J = coords.T * dN_dxi
        detJ = J.det()
        vol = sp.Abs(detJ) / 6.0

        invJ = J.inv()
        dN_dx = dN_dxi * invJ

        B = sp.zeros(6, 12)
        for i in range(4):
            B[0, 3 * i]     = dN_dx[i, 0]
            B[1, 3 * i + 1] = dN_dx[i, 1]
            B[2, 3 * i + 2] = dN_dx[i, 2]
            B[3, 3 * i]     = dN_dx[i, 1]
            B[3, 3 * i + 1] = dN_dx[i, 0]
            B[4, 3 * i + 1] = dN_dx[i, 2]
            B[4, 3 * i + 2] = dN_dx[i, 1]
            B[5, 3 * i]     = dN_dx[i, 2]
            B[5, 3 * i + 2] = dN_dx[i, 0]

        K = B.T * D * B * vol
        K_flat = [K[i, j] for i in range(12) for j in range(12)]

        # 3. Create and return the MathModel
        model_name = f"{self.name}_Ke"
        return MathModel(inputs=in_syms, outputs=K_flat, name=model_name, input_names=input_names)

