import sympy as sp
from definitions.abc import Element
from sympy_codegen import MathModel

class Tet4(Element):
    """4-node tetrahedral element."""
    def __init__(self):
        super().__init__("tet4")

    def get_stiffness_operators(self):
        """
        Returns decoupled operators for Tet4: dN_dnat, mapping, and assembly.
        """
        # 1. dN_dnat operator (Constant for Tet4, but we follow the pattern)
        xi, eta, zeta = sp.symbols("xi eta zeta", real=True)
        N_list = [1 - xi - eta - zeta, xi, eta, zeta]
        dN_dnat = sp.Matrix(4, 3, lambda i, j: sp.diff(N_list[i], (xi, eta, zeta)[j]))
        dN_dnat_flat = [dN_dnat[i, j] for i in range(4) for j in range(3)]

        op_dN = MathModel(
            inputs=[xi, eta, zeta],
            outputs=dN_dnat_flat,
            name=f"{self.name}_op_dN_dnat",
            input_names=["xi", "eta", "zeta"],
            output_names=[f"dN{i+1}_d{j+1}" for i in range(4) for j in range(3)],
            is_operator=True
        )

        # 2. Mapping operator (Jacobian & dN/dx)
        coord_syms = [sp.Symbol(f"c{i}", real=True) for i in range(12)]
        dN_dnat_syms = [sp.Symbol(f"dN_dnat{i}", real=True) for i in range(12)]

        coords = sp.Matrix(4, 3, lambda i, j: coord_syms[i * 3 + j])
        dN_dnat_mat = sp.Matrix(4, 3, lambda i, j: dN_dnat_syms[i * 3 + j])

        J = coords.T * dN_dnat_mat
        detJ = J.det()
        invJ = J.inv()

        dN_dx = dN_dnat_mat * invJ
        # SoA Layout: [dN1/dx, dN2/dx, ..., dN4/dx, dN1/dy, ..., dN4/dy, dN1/dz, ..., dN4/dz]
        dN_dx_flat = [dN_dx[i, j] for j in range(3) for i in range(4)]

        op_map = MathModel(
            inputs=coord_syms + dN_dnat_syms,
            outputs=dN_dx_flat + [detJ],
            name=f"{self.name}_op_mapping",
            input_names=[f"coord[{i}]" for i in range(12)] + [f"dN_dnat[{i}]" for i in range(12)],
            output_names=[f"dN{i+1}_dx" for i in range(4)] + [f"dN{i+1}_dy" for i in range(4)] + [f"dN{i+1}_dz" for i in range(4)] + ["detJ"],
            is_operator=True
        )

        # 3. Assembly operator (B^T * D * B)
        dN_dx_syms = [sp.Symbol(f"dN_dx{i}", real=True) for i in range(12)]
        D_syms = [sp.Symbol(f"D{i}", real=True) for i in range(36)]
        detJ_sym = sp.Symbol("detJ", real=True)
        weight_sym = sp.Symbol("weight", real=True)

        D = sp.Matrix(6, 6, lambda i, j: D_syms[i * 6 + j])
        B = sp.zeros(6, 12)
        for i in range(4):
            # SoA Access
            g0 = dN_dx_syms[i]      # dx
            g1 = dN_dx_syms[i + 4]  # dy
            g2 = dN_dx_syms[i + 8]  # dz

            B[0, 3 * i]     = g0
            B[1, 3 * i + 1] = g1
            B[2, 3 * i + 2] = g2
            B[3, 3 * i]     = g1
            B[3, 3 * i + 1] = g0
            B[4, 3 * i + 1] = g2
            B[4, 3 * i + 2] = g1
            B[5, 3 * i]     = g2
            B[5, 3 * i + 2] = g0

        Ke_gp = B.T * D * B * sp.Abs(detJ_sym) * weight_sym
        Ke_flat = [Ke_gp[i, j] for i in range(12) for j in range(12)]

        op_asm = MathModel(
            inputs=dN_dx_syms + D_syms + [detJ_sym, weight_sym],
            outputs=Ke_flat,
            name=f"{self.name}_op_assembly",
            input_names=[f"dN_dx[{i}]" for i in range(12)] + [f"D[{i}]" for i in range(36)] + ["detJ", "weight"],
            output_names=[f"Ke_{i}_{j}" for i in range(12) for j in range(12)],
            is_operator=True
        )

        return [op_dN, op_map, op_asm]

    def get_stiffness_model(self):
        """
        Traditional single-kernel model for Tet4.
        """
        coord_syms = [sp.Symbol(f"c{i}", real=True) for i in range(12)]
        D_syms = [sp.Symbol(f"D{i}", real=True) for i in range(36)]
        
        all_inputs = coord_syms + D_syms
        in_syms = [sp.Symbol(f"in[{i}]", real=True) for i in range(len(all_inputs))]
        subs_map = dict(zip(all_inputs, in_syms))

        coords = sp.Matrix(4, 3, lambda i, j: coord_syms[i * 3 + j]).subs(subs_map)
        D = sp.Matrix(6, 6, lambda i, j: D_syms[i * 6 + j]).subs(subs_map)

        xi, eta, zeta = sp.symbols("xi eta zeta", real=True)
        N_list = [1 - xi - eta - zeta, xi, eta, zeta]
        dN_dxi = sp.Matrix(4, 3, lambda i, j: sp.diff(N_list[i], (xi, eta, zeta)[j]))

        J = coords.T * dN_dxi
        vol = sp.Abs(J.det()) / 6.0
        dN_dx = dN_dxi * J.inv()

        B = sp.zeros(6, 12)
        for i in range(4):
            g = dN_dx[i, :]
            B[0, 3*i], B[1, 3*i+1], B[2, 3*i+2] = g[0], g[1], g[2]
            B[3, 3*i], B[3, 3*i+1] = g[1], g[0]
            B[4, 3*i+1], B[4, 3*i+2] = g[2], g[1]
            B[5, 3*i], B[5, 3*i+2] = g[2], g[0]

        K = B.T * D * B * vol
        K_flat = [K[i, j] for i in range(12) for j in range(12)]

        output_names = [f"Ke_{i}_{j}" for i in range(12) for j in range(12)]
        
        return MathModel(
            inputs=in_syms, 
            outputs=K_flat, 
            name=f"{self.name}_Ke",
            output_names=output_names
        )

    def get_mass_operators(self):
        """
        Returns decoupled operators for Tet4 mass calculation.
        For explicit, we typically use lumped mass.
        """
        # 1. Node coordinates and density
        coord_syms = [sp.Symbol(f"c{i}", real=True) for i in range(12)]
        rho_sym = sp.Symbol("rho", real=True)

        coords = sp.Matrix(4, 3, lambda i, j: coord_syms[i * 3 + j])
        
        # Volume calculation for Tet4
        M = sp.Matrix(4, 4, lambda i, j: 1 if j == 0 else coords[i, j-1])
        vol = sp.Abs(M.det()) / 6.0
        
        # Lumped mass: rho * vol / 4 for each node
        m_node = rho_sym * vol / 4.0
        m_lumped = [m_node] * 4

        op_mass = MathModel(
            inputs=coord_syms + [rho_sym],
            outputs=m_lumped,
            name=f"{self.name}_op_lumped_mass",
            input_names=[f"coord[{i}]" for i in range(12)] + ["rho"],
            output_names=[f"mass_node{i+1}" for i in range(4)],
            is_operator=True
        )
        return [op_mass]
