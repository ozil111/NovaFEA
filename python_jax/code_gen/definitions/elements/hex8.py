import sympy as sp
from definitions.abc import Element
from sympy_codegen import MathModel

class Hex8(Element):
    """8-node hexahedral element (Linear interpolation)."""
    def __init__(self):
        super().__init__("hex8")

    def get_stiffness_model(self):
        """
        Traditional single-kernel model for Hex8. 
        Now benefits from chunked CSE in the compiler.
        """
        coord_syms = [sp.Symbol(f"c{i}", real=True) for i in range(24)]
        D_syms = [sp.Symbol(f"D{i}", real=True) for i in range(36)]
        in_syms = [sp.Symbol(f"in[{i}]", real=True) for i in range(60)]
        subs_map = dict(zip(coord_syms + D_syms, in_syms))

        coords = sp.Matrix(8, 3, lambda i, j: coord_syms[i * 3 + j]).subs(subs_map)
        D = sp.Matrix(6, 6, lambda i, j: D_syms[i * 6 + j]).subs(subs_map)

        xi, eta, zeta = sp.symbols("xi eta zeta", real=True)
        node_coords = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1)
        ]
        N = [0.125 * (1 + nc[0]*xi) * (1 + nc[1]*eta) * (1 + nc[2]*zeta) for nc in node_coords]
        dN_dnat = sp.Matrix(8, 3, lambda i, j: sp.diff(N[i], (xi, eta, zeta)[j]))

        gp_val = 1.0 / sp.sqrt(3)
        gauss_points = [
            (x, y, z) for x in [-gp_val, gp_val] for y in [-gp_val, gp_val] for z in [-gp_val, gp_val]
        ]

        K = sp.zeros(24, 24)
        for pt in gauss_points:
            dN_dnat_gp = dN_dnat.subs({xi: pt[0], eta: pt[1], zeta: pt[2]})
            J = coords.T * dN_dnat_gp
            detJ = J.det()
            invJ = J.inv()
            dN_dx = dN_dnat_gp * invJ
            
            B = sp.zeros(6, 24)
            for i in range(8):
                g = dN_dx[i, :]
                B[0, 3*i], B[1, 3*i+1], B[2, 3*i+2] = g[0], g[1], g[2]
                B[3, 3*i], B[3, 3*i+1] = g[1], g[0]
                B[4, 3*i+1], B[4, 3*i+2] = g[2], g[1]
                B[5, 3*i], B[5, 3*i+2] = g[2], g[0]
            
            K += B.T * D * B * detJ

        K_flat = [K[i, j] for i in range(24) for j in range(24)]
        return MathModel(inputs=in_syms, outputs=K_flat, name=f"{self.name}_Ke")

    def get_stiffness_operators(self):
        """
        Returns decoupled operators for Hex8: dN_dnat, mapping, and assembly.
        """
        # 1. dN_dnat operator
        xi, eta, zeta = sp.symbols("xi eta zeta", real=True)
        node_coords = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1)
        ]
        N = []
        for nc in node_coords:
            N.append(0.125 * (1 + nc[0]*xi) * (1 + nc[1]*eta) * (1 + nc[2]*zeta))

        dN_dnat = sp.Matrix(8, 3, lambda i, j: sp.diff(N[i], (xi, eta, zeta)[j]))
        dN_dnat_flat = [dN_dnat[i, j] for i in range(8) for j in range(3)]

        op_dN = MathModel(
            inputs=[xi, eta, zeta],
            outputs=dN_dnat_flat,
            name=f"{self.name}_op_dN_dnat",
            input_names=["xi", "eta", "zeta"],
            is_operator=True
        )

        # 2. Mapping operator (Jacobian & dN/dx)
        coord_syms = [sp.Symbol(f"c{i}", real=True) for i in range(24)]
        dN_dnat_syms = [sp.Symbol(f"dN_dnat{i}", real=True) for i in range(24)]

        coords = sp.Matrix(8, 3, lambda i, j: coord_syms[i * 3 + j])
        dN_dnat_mat = sp.Matrix(8, 3, lambda i, j: dN_dnat_syms[i * 3 + j])

        J = coords.T * dN_dnat_mat
        detJ = J.det()
        invJ = J.inv()

        dN_dx = dN_dnat_mat * invJ
        # SoA Layout: [dN1/dx, dN2/dx, ..., dN8/dx, dN1/dy, ..., dN8/dy, dN1/dz, ..., dN8/dz]
        dN_dx_flat = [dN_dx[i, j] for j in range(3) for i in range(8)]

        op_map = MathModel(
            inputs=coord_syms + dN_dnat_syms,
            outputs=dN_dx_flat + [detJ],
            name=f"{self.name}_op_mapping",
            input_names=[f"coord[{i}]" for i in range(24)] + [f"dN_dnat[{i}]" for i in range(24)],
            is_operator=True
        )

        # 3. Assembly operator (B^T * D * B)
        # Input dN_dx is in SoA format: 8 dx, then 8 dy, then 8 dz
        dN_dx_syms = [sp.Symbol(f"dN_dx{i}", real=True) for i in range(24)]
        D_syms = [sp.Symbol(f"D{i}", real=True) for i in range(36)]
        detJ_sym = sp.Symbol("detJ", real=True)
        weight_sym = sp.Symbol("weight", real=True)

        D = sp.Matrix(6, 6, lambda i, j: D_syms[i * 6 + j])
        B = sp.zeros(6, 24)
        for i in range(8):
            # Access SoA layout
            g0 = dN_dx_syms[i]      # dx for node i
            g1 = dN_dx_syms[i + 8]  # dy for node i
            g2 = dN_dx_syms[i + 16] # dz for node i

            B[0, 3 * i]     = g0
            B[1, 3 * i + 1] = g1
            B[2, 3 * i + 2] = g2
            B[3, 3 * i]     = g1
            B[3, 3 * i + 1] = g0
            B[4, 3 * i + 1] = g2
            B[4, 3 * i + 2] = g1
            B[5, 3 * i]     = g2
            B[5, 3 * i + 2] = g0
        Ke_gp = B.T * D * B * detJ_sym * weight_sym
        Ke_flat = [Ke_gp[i, j] for i in range(24) for j in range(24)]

        op_asm = MathModel(
            inputs=dN_dx_syms + D_syms + [detJ_sym, weight_sym],
            outputs=Ke_flat,
            name=f"{self.name}_op_assembly",
            input_names=[f"dN_dx[{i}]" for i in range(24)] + [f"D[{i}]" for i in range(36)] + ["detJ", "weight"],
            is_operator=True
        )

        return [op_dN, op_map, op_asm]