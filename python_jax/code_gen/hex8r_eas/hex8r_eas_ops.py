"""Element-related operators for hex8r EAS element.

This module contains geometry helpers, kinematic operators, and EAS
stiffness-matrix operators.  It is self-contained and exposes all
element operators through ``get_model()``.

Usage::

    python sympy_codegen.py --task custom --script ./hex8r_eas/hex8r_eas_ops.py
"""

import itertools

import sympy as sp

try:
    from sympy_codegen import MathModel
except ImportError:
    # Fallback for cases where the generator is imported from a package path.
    from .sympy_codegen import MathModel


ONE_OVER_EIGHT = sp.Rational(1, 8)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def mat_symbols(prefix: str, rows: int, cols: int):
    """Create a rows x cols SymPy Matrix with readable scalar symbols."""
    return sp.Matrix(rows, cols, lambda i, j: sp.Symbol(f"{prefix}_{i+1}_{j+1}", real=True))


def vec_symbols(prefix: str, n: int):
    """Create an n-vector with readable scalar symbols."""
    return sp.Matrix([sp.Symbol(f"{prefix}_{i+1}", real=True) for i in range(n)])


def flatten_row_major(mat: sp.Matrix):
    """Flatten a SymPy Matrix in row-major order."""
    return [mat[i, j] for i in range(mat.rows) for j in range(mat.cols)]


def flatten_nd_row_major(tensor):
    """Flatten a SymPy N-D tensor in row-major order."""
    return [tensor[idx] for idx in itertools.product(*[range(n) for n in tensor.shape])]


def mat3_det(A: sp.Matrix):
    return sp.simplify(A.det())


def inv3x3_with_det(A: sp.Matrix):
    detA = mat3_det(A)
    Ainv = sp.simplify(A.inv())
    return detA, Ainv


def hex8_h_vectors():
    return sp.Matrix([
        [1, 1, 1, -1],
        [-1, -1, 1, 1],
        [1, -1, -1, -1],
        [-1, 1, -1, 1],
        [1, -1, -1, 1],
        [-1, 1, -1, -1],
        [1, 1, 1, 1],
        [-1, -1, 1, -1],
    ])


# -----------------------------------------------------------------------------
# Hex8 geometry helpers
# -----------------------------------------------------------------------------

def hex8_shape(xi, eta, zeta):
    """Standard Hex8 shape functions on [-1,1]^3."""
    return sp.Matrix([
        (1 - xi) * (1 - eta) * (1 - zeta) / 8,
        (1 + xi) * (1 - eta) * (1 - zeta) / 8,
        (1 + xi) * (1 + eta) * (1 - zeta) / 8,
        (1 - xi) * (1 + eta) * (1 - zeta) / 8,
        (1 - xi) * (1 - eta) * (1 + zeta) / 8,
        (1 + xi) * (1 - eta) * (1 + zeta) / 8,
        (1 + xi) * (1 + eta) * (1 + zeta) / 8,
        (1 - xi) * (1 + eta) * (1 + zeta) / 8,
    ])


def hex8_shape_gradient(xi, eta, zeta):
    """Return 8x3 shape function gradients with respect to natural coordinates."""
    N = hex8_shape(xi, eta, zeta)
    return sp.Matrix([
        [sp.diff(N[I], xi), sp.diff(N[I], eta), sp.diff(N[I], zeta)]
        for I in range(8)
    ])


def hex8_shape_gradient_at_center():
    xi, eta, zeta = sp.symbols("xi eta zeta", real=True)
    return sp.simplify(hex8_shape_gradient(xi, eta, zeta).subs({xi: 0, eta: 0, zeta: 0}))


def jacobian_from_shape_gradient(coord, dN_dxi):
    return sp.simplify(coord.T * dN_dxi)


def hex8_face_parametrizations():
    a, b = sp.symbols("a b", real=True)
    return [
        ("zeta=-1", {sp.Symbol("xi"): a, sp.Symbol("eta"): b, sp.Symbol("zeta"): -1}, (a, b), -1),
        ("zeta=+1", {sp.Symbol("xi"): a, sp.Symbol("eta"): b, sp.Symbol("zeta"): 1}, (a, b), 1),
        ("eta=-1", {sp.Symbol("xi"): a, sp.Symbol("eta"): -1, sp.Symbol("zeta"): b}, (a, b), 1),
        ("eta=+1", {sp.Symbol("xi"): a, sp.Symbol("eta"): 1, sp.Symbol("zeta"): b}, (a, b), -1),
        ("xi=-1", {sp.Symbol("xi"): -1, sp.Symbol("eta"): a, sp.Symbol("zeta"): b}, (a, b), -1),
        ("xi=+1", {sp.Symbol("xi"): 1, sp.Symbol("eta"): a, sp.Symbol("zeta"): b}, (a, b), 1),
    ]


def restricted_hex8_shape_on_face(face_subs):
    xi = sp.Symbol("xi")
    eta = sp.Symbol("eta")
    zeta = sp.Symbol("zeta")
    N = hex8_shape(xi, eta, zeta)
    return sp.Matrix([sp.simplify(N[i].subs(face_subs)) for i in range(8)])


def face_mapping_from_hex8(coord, face_subs):
    Nf = restricted_hex8_shape_on_face(face_subs)
    return coord.T * Nf


def face_area_vector_from_mapping(xvec, local_vars, orientation_sign=1):
    a, b = local_vars
    dx_da = sp.diff(xvec, a)
    dx_db = sp.diff(xvec, b)
    return sp.simplify(orientation_sign * dx_da.cross(dx_db))


def integrate_face_surface_vector(coord, face_subs, local_vars, orientation_sign):
    a, b = local_vars
    Nf = restricted_hex8_shape_on_face(face_subs)
    xvec = face_mapping_from_hex8(coord, face_subs)
    avec = face_area_vector_from_mapping(xvec, local_vars, orientation_sign)

    Gf = sp.Matrix.zeros(8, 3)
    for I in range(8):
        for j in range(3):
            Gf[I, j] = sp.simplify(
                sp.integrate(
                    sp.integrate(Nf[I] * avec[j], (a, -1, 1)),
                    (b, -1, 1),
                )
            )
    return Gf


def calc_b_bar_surface_vector(coord):
    G = sp.Matrix.zeros(8, 3)
    for _, face_subs, local_vars, orientation_sign in hex8_face_parametrizations():
        G += integrate_face_surface_vector(coord, face_subs, local_vars, orientation_sign)
    return sp.simplify(G)


def calc_volume_from_surface_vector(coord, G):
    return sp.expand(coord[:, 0].dot(G[:, 0]))


# -----------------------------------------------------------------------------
# Hyperelastic kinematics (internal helper)
# -----------------------------------------------------------------------------

def build_hyperelastic_kinematics(F: sp.Matrix):
    """Compute kinematic quantities from deformation gradient F.

    Returns a dict with J, J_minus_2_3, B, B_bar, I1_bar_B,
    C, Cinv, C_bar, I1_bar_C, detC.
    """
    J = F.det()
    J_minus_2_3 = J**sp.Rational(-2, 3)

    B = F * F.T
    B_bar = J_minus_2_3 * B
    I1_bar_B = sp.trace(B_bar)

    C = F.T * F
    detC = C.det()
    Cinv = C.inv()
    C_bar = J_minus_2_3 * C
    I1_bar_C = sp.trace(C_bar)

    return {
        "J": J,
        "J_minus_2_3": J_minus_2_3,
        "B": B,
        "B_bar": B_bar,
        "I1_bar_B": I1_bar_B,
        "C": C,
        "Cinv": Cinv,
        "C_bar": C_bar,
        "I1_bar_C": I1_bar_C,
        "detC": detC,
    }


# -----------------------------------------------------------------------------
# Element operators
# -----------------------------------------------------------------------------

def build_hex8r_op_bbar_grad():
    coord = mat_symbols("coord", 8, 3)
    G = calc_b_bar_surface_vector(coord)
    vol = calc_volume_from_surface_vector(coord, G)
    BiI = sp.simplify(G / vol)

    inputs = flatten_row_major(coord)
    outputs = flatten_row_major(BiI) + [vol]
    input_names = [f"COORD({i+1},{j+1})" for i in range(8) for j in range(3)]
    output_names = [f"BiI({i+1},{j+1})" for i in range(8) for j in range(3)] + ["VOL"]
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_bbar_grad",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


def build_hex8r_op_jacobian_center():
    coord = mat_symbols("coord", 8, 3)
    dN_dxi_center = hex8_shape_gradient_at_center()
    J = jacobian_from_shape_gradient(coord, dN_dxi_center)
    detJ, Jinv = inv3x3_with_det(J)

    inputs = flatten_row_major(coord)
    outputs = flatten_row_major(J) + [detJ] + flatten_row_major(Jinv)
    input_names = [f"COORD({i+1},{j+1})" for i in range(8) for j in range(3)]
    output_names = (
        [f"J({i+1},{j+1})" for i in range(3) for j in range(3)]
        + ["detJ"]
        + [f"Jinv({i+1},{j+1})" for i in range(3) for j in range(3)]
    )
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_jacobian_center",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


def strain_displacement_block(gradN):
    bx, by, bz = gradN
    return sp.Matrix([
        [bx, 0, 0],
        [0, by, 0],
        [0, 0, bz],
        [by, bx, 0],
        [0, bz, by],
        [bz, 0, bx],
    ])


def form_B_matrix(BiI: sp.Matrix):
    blocks = [strain_displacement_block(BiI[I, :]) for I in range(8)]
    return sp.Matrix.hstack(*blocks)


def build_hex8r_op_form_B():
    BiI = mat_symbols("BiI", 8, 3)
    B = form_B_matrix(BiI)

    inputs = flatten_row_major(BiI)
    outputs = flatten_row_major(B)
    input_names = [f"BiI({i+1},{j+1})" for i in range(8) for j in range(3)]
    output_names = [f"B({i+1},{j+1})" for i in range(6) for j in range(24)]
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_form_B",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


def build_hex8r_op_internal_force():
    B = mat_symbols("B", 6, 24)
    stress = vec_symbols("stress", 6)
    detJ = sp.Symbol("detJ", real=True)
    weight = sp.Symbol("weight", real=True)
    fint = B.T * stress * detJ * weight

    inputs = flatten_row_major(B) + list(stress) + [detJ, weight]
    outputs = list(fint)
    input_names = (
        [f"B({i+1},{j+1})" for i in range(6) for j in range(24)]
        + [f"stress({i+1})" for i in range(6)]
        + ["detJ", "weight"]
    )
    output_names = [f"fint({i+1})" for i in range(24)]
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_internal_force",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


def build_hex8r_op_mass_gp():
    coord = mat_symbols("coord", 8, 3)
    xi, eta, zeta = sp.symbols("xi eta zeta", real=True)
    weight = sp.Symbol("weight", real=True)
    rho = sp.Symbol("rho", real=True)

    N = hex8_shape(xi, eta, zeta)
    dN_dxi = hex8_shape_gradient(xi, eta, zeta)
    J = jacobian_from_shape_gradient(coord, dN_dxi)
    detJ = J.det()
    mij = weight * detJ * rho * (N * N.T)

    inputs = flatten_row_major(coord) + [xi, eta, zeta, weight, rho]
    outputs = flatten_row_major(mij) + list(N) + [detJ]
    input_names = (
        [f"COORD({i+1},{j+1})" for i in range(8) for j in range(3)]
        + ["xi", "eta", "zeta", "weight", "rho"]
    )
    output_names = (
        [f"m_{i+1}_{j+1}" for i in range(8) for j in range(8)]
        + [f"N({i+1})" for i in range(8)]
        + ["detJ"]
    )
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_mass_gp",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


def build_hex8r_op_hourglass_gamma():
    BiI = mat_symbols("BiI", 8, 3)
    coord = mat_symbols("coord", 8, 3)
    h_vectors = hex8_h_vectors()
    gamma = sp.Matrix.zeros(8, 4)

    for i in range(4):
        h_dot_x = coord.T * h_vectors[:, i]
        for A in range(8):
            gamma[A, i] = sp.simplify(
                ONE_OVER_EIGHT * (h_vectors[A, i] - sum(h_dot_x[k] * BiI[A, k] for k in range(3)))
            )

    inputs = flatten_row_major(BiI) + flatten_row_major(coord)
    outputs = flatten_row_major(gamma)
    input_names = (
        [f"BiI({i+1},{j+1})" for i in range(8) for j in range(3)]
        + [f"COORD({i+1},{j+1})" for i in range(8) for j in range(3)]
    )
    output_names = [f"gamma({i+1},{j+1})" for i in range(8) for j in range(4)]
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_hourglass_gamma",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


# -----------------------------------------------------------------------------
# Kinematics operator (F → J, B, B_bar, C, C_bar, Cinv, I1_bar, ...)
# -----------------------------------------------------------------------------

def build_hex8r_op_kinematics():
    """Compute hyperelastic kinematic quantities from the deformation gradient.

    Outputs are the intermediate quantities needed by the material stress
    and tangent-modulus operators, so that those operators no longer need
    direct access to F.
    """
    F = mat_symbols("F", 3, 3)
    kin = build_hyperelastic_kinematics(F)

    outputs = (
        [kin["J"]]
        + flatten_row_major(kin["B"])
        + flatten_row_major(kin["B_bar"])
        + [kin["I1_bar_B"]]
        + flatten_row_major(kin["C"])
        + flatten_row_major(kin["C_bar"])
        + flatten_row_major(kin["Cinv"])
        + [kin["I1_bar_C"]]
        + [kin["J_minus_2_3"]]
    )
    input_names = [f"F({i+1},{j+1})" for i in range(3) for j in range(3)]
    output_names = (
        ["J"]
        + [f"B({i+1},{j+1})" for i in range(3) for j in range(3)]
        + [f"B_bar({i+1},{j+1})" for i in range(3) for j in range(3)]
        + ["I1_bar_B"]
        + [f"C({i+1},{j+1})" for i in range(3) for j in range(3)]
        + [f"C_bar({i+1},{j+1})" for i in range(3) for j in range(3)]
        + [f"Cinv({i+1},{j+1})" for i in range(3) for j in range(3)]
        + ["I1_bar_C"]
        + ["J_minus_2_3"]
    )
    return MathModel(
        inputs=flatten_row_major(F),
        outputs=outputs,
        name="hex8r_op_kinematics",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


# -----------------------------------------------------------------------------
# EAS K-matrices operator
# -----------------------------------------------------------------------------

def build_hex8r_op_k_matrices():
    C = mat_symbols("C", 6, 6)
    VOL = sp.Symbol("VOL", real=True)

    Kmat = sp.MutableDenseNDimArray.zeros(4, 4, 3, 3)
    K_alpha_u = sp.MutableDenseNDimArray.zeros(4, 6, 3)
    K_alpha_alpha = sp.Matrix.zeros(6, 6)

    factor_K123 = sp.Rational(8, 3)
    factor_K4 = sp.Rational(8, 9)
    factor_Kau = sp.Rational(8, 3)

    Kmat[0, 0, 0, 0] = factor_K123 * C[0, 0]
    Kmat[0, 0, 0, 2] = factor_K123 * C[0, 5]
    Kmat[0, 0, 1, 1] = factor_K123 * C[1, 1]
    Kmat[0, 0, 1, 2] = factor_K123 * C[1, 4]
    Kmat[0, 0, 2, 0] = factor_K123 * C[5, 0]
    Kmat[0, 0, 2, 1] = factor_K123 * C[4, 1]
    Kmat[0, 0, 2, 2] = factor_K123 * (C[4, 4] + C[5, 5])

    Kmat[1, 1, 0, 0] = factor_K123 * C[0, 0]
    Kmat[1, 1, 0, 1] = factor_K123 * C[0, 3]
    Kmat[1, 1, 1, 0] = factor_K123 * C[3, 0]
    Kmat[1, 1, 1, 1] = factor_K123 * (C[4, 4] + C[3, 3])
    Kmat[1, 1, 1, 2] = factor_K123 * C[4, 2]
    Kmat[1, 1, 2, 1] = factor_K123 * C[2, 4]
    Kmat[1, 1, 2, 2] = factor_K123 * C[2, 2]

    Kmat[2, 2, 0, 0] = factor_K123 * (C[5, 5] + C[3, 3])
    Kmat[2, 2, 0, 1] = factor_K123 * C[3, 1]
    Kmat[2, 2, 0, 2] = factor_K123 * C[5, 2]
    Kmat[2, 2, 1, 0] = factor_K123 * C[1, 3]
    Kmat[2, 2, 1, 1] = factor_K123 * C[1, 1]
    Kmat[2, 2, 2, 0] = factor_K123 * C[2, 5]
    Kmat[2, 2, 2, 2] = factor_K123 * C[2, 2]

    Kmat[3, 3, 0, 0] = factor_K4 * C[0, 0]
    Kmat[3, 3, 1, 1] = factor_K4 * C[1, 1]
    Kmat[3, 3, 2, 2] = factor_K4 * C[2, 2]

    Kmat[0, 1, 1, 1] = factor_K123 * C[1, 4]
    Kmat[0, 1, 1, 2] = factor_K123 * C[1, 2]
    Kmat[0, 1, 2, 1] = factor_K123 * C[4, 4]
    Kmat[0, 1, 2, 2] = factor_K123 * C[4, 2]

    Kmat[0, 2, 0, 0] = factor_K123 * C[0, 5]
    Kmat[0, 2, 0, 2] = factor_K123 * C[0, 2]
    Kmat[0, 2, 2, 0] = factor_K123 * C[5, 5]
    Kmat[0, 2, 2, 2] = factor_K123 * C[5, 2]

    Kmat[1, 0, 1, 1] = factor_K123 * C[4, 1]
    Kmat[1, 0, 1, 2] = factor_K123 * C[4, 4]
    Kmat[1, 0, 2, 1] = factor_K123 * C[2, 1]
    Kmat[1, 0, 2, 2] = factor_K123 * C[2, 4]

    Kmat[1, 2, 0, 0] = factor_K123 * C[0, 3]
    Kmat[1, 2, 0, 1] = factor_K123 * C[0, 1]
    Kmat[1, 2, 1, 0] = factor_K123 * C[3, 3]
    Kmat[1, 2, 1, 1] = factor_K123 * C[3, 1]

    Kmat[2, 0, 0, 0] = factor_K123 * C[5, 0]
    Kmat[2, 0, 0, 2] = factor_K123 * C[5, 5]
    Kmat[2, 0, 2, 0] = factor_K123 * C[2, 0]
    Kmat[2, 0, 2, 2] = factor_K123 * C[2, 5]

    Kmat[2, 1, 0, 0] = factor_K123 * C[3, 0]
    Kmat[2, 1, 0, 1] = factor_K123 * C[3, 3]
    Kmat[2, 1, 1, 0] = factor_K123 * C[1, 0]
    Kmat[2, 1, 1, 1] = factor_K123 * C[1, 3]

    K_alpha_u[0, 0, 1] = factor_Kau * C[0, 1]
    K_alpha_u[0, 0, 2] = factor_Kau * C[0, 4]
    K_alpha_u[0, 1, 0] = factor_Kau * C[1, 0]
    K_alpha_u[0, 1, 2] = factor_Kau * C[1, 5]

    K_alpha_u[1, 0, 1] = factor_Kau * C[0, 4]
    K_alpha_u[1, 0, 2] = factor_Kau * C[0, 2]
    K_alpha_u[1, 2, 0] = factor_Kau * C[2, 0]
    K_alpha_u[1, 2, 1] = factor_Kau * C[2, 3]

    K_alpha_u[2, 1, 0] = factor_Kau * C[1, 5]
    K_alpha_u[2, 1, 2] = factor_Kau * C[1, 2]
    K_alpha_u[2, 2, 0] = factor_Kau * C[2, 3]
    K_alpha_u[2, 2, 1] = factor_Kau * C[2, 1]

    H43 = C[0, 2] + C[1, 2] + C[2, 2]
    H51 = C[0, 0] + C[1, 0] + C[2, 0]
    H62 = C[0, 1] + C[1, 1] + C[2, 1]
    K_alpha_u[3, 3, 2] = factor_K4 * H43
    K_alpha_u[3, 4, 0] = factor_K4 * H51
    K_alpha_u[3, 5, 1] = factor_K4 * H62

    H = C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[1, 2] + C[0, 2])
    K_alpha_alpha[0, 0] = factor_Kau * C[0, 0]
    K_alpha_alpha[1, 1] = factor_Kau * C[1, 1]
    K_alpha_alpha[2, 2] = factor_Kau * C[2, 2]
    K_alpha_alpha[3, 3] = factor_Kau * H / 3
    K_alpha_alpha[4, 4] = factor_Kau * H / 3
    K_alpha_alpha[5, 5] = factor_Kau * H / 3

    inputs = flatten_row_major(C) + [VOL]
    outputs = flatten_nd_row_major(Kmat) + flatten_nd_row_major(K_alpha_u) + flatten_row_major(K_alpha_alpha)
    input_names = [f"C({i+1},{j+1})" for i in range(6) for j in range(6)] + ["VOL"]
    output_names = (
        [f"Kmat_{i}_{j}_{k}_{l}" for i in range(4) for j in range(4) for k in range(3) for l in range(3)]
        + [f"K_alpha_u_{i}_{j}_{k}" for i in range(4) for j in range(6) for k in range(3)]
        + [f"K_alpha_alpha({i+1},{j+1})" for i in range(6) for j in range(6)]
    )
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_k_matrices",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


# -----------------------------------------------------------------------------
# Public entry for --task custom
# -----------------------------------------------------------------------------

def get_model():
    return [
        build_hex8r_op_bbar_grad(),
        build_hex8r_op_jacobian_center(),
        build_hex8r_op_form_B(),
        build_hex8r_op_internal_force(),
        build_hex8r_op_mass_gp(),
        build_hex8r_op_hourglass_gamma(),
        build_hex8r_op_kinematics(),
        build_hex8r_op_k_matrices(),
    ]
