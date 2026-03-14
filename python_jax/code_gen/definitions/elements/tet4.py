import sympy as sp
from definitions.abc import Element, Material
from sympy_codegen import MathModel

class Tet4(Element):
    """4-node tetrahedral element."""
    def __init__(self):
        super().__init__("tet4")

    def get_symbolic_model(self, material: Material):
        """
        Creates the symbolic model for a Tet4 element with a given material.
        The generated kernel will take nodal coordinates and material parameters as flat input.
        """
        # 1. Get material's symbolic model
        mat_params, D_sym = material.get_symbolic_model()

        # 2. Define kinematic inputs (nodal coordinates)
        coord_syms = [sp.Symbol(f"c{i}", real=True) for i in range(12)]
        coords = sp.Matrix(4, 3, lambda i, j: coord_syms[i * 3 + j])

        # 3. Combine all inputs for the kernel
        # The final C++/JAX function will take one flat array: [coords..., mat_params...]
        all_inputs = coord_syms + mat_params
        input_names = [f"coord[{i//3}][{i%3}]" for i in range(12)] + [str(p) for p in mat_params]
        
        # Rename symbols to "in[i]" to match the C-style array access in the generated code
        in_syms = [sp.Symbol(f"in[{i}]", real=True) for i in range(len(all_inputs))]
        subs_map = dict(zip(all_inputs, in_syms))

        D = D_sym.subs(subs_map)
        coords = coords.subs(subs_map)

        # 4. Finite Element Formulation (same as before)
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

        # 5. Create and return the MathModel
        model_name = f"{self.name}_{material.name}_Ke"
        return MathModel(inputs=in_syms, outputs=K_flat, name=model_name, input_names=input_names)
