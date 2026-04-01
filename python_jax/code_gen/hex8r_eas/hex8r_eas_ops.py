import itertools

import sympy as sp

try:
    from sympy_codegen import MathModel
except ImportError:
    # Fallback for cases where the generator is imported from a package path.
    from .sympy_codegen import MathModel


ONE_OVER_EIGHT = sp.Rational(1, 8)
ONE_OVER_THREE = sp.Rational(1, 3)
TWO_OVER_THREE = sp.Rational(2, 3)
FOUR_OVER_THREE = sp.Rational(4, 3)
TWO = sp.Integer(2)
SIX = sp.Integer(6)
TWELVE = sp.Integer(12)
THIRTY = sp.Integer(30)
VOL_TOL = sp.Float("1.0e-20")


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


def identity3():
    return sp.eye(3)


def kronecker(i: int, j: int):
    return sp.Integer(1) if i == j else sp.Integer(0)


def mat3_det(A: sp.Matrix):
    return sp.simplify(A.det())


def inv3x3_with_det(A: sp.Matrix):
    detA = mat3_det(A)
    Ainv = sp.simplify(A.inv())
    return detA, Ainv


def symmetric_matrix_to_voigt(mat: sp.Matrix):
    return sp.Matrix([
        mat[0, 0],
        mat[1, 1],
        mat[2, 2],
        mat[0, 1],
        mat[1, 2],
        mat[0, 2],
    ])


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


def volumetric_pressure(J, D1, D2, D3):
    j_minus_1 = J - 1
    d2_term = sp.Piecewise(
        (4 * j_minus_1**3 / D2, D2 > VOL_TOL),
        (0, True),
    )
    d3_term = sp.Piecewise(
        (6 * j_minus_1**5 / D3, D3 > VOL_TOL),
        (0, True),
    )
    return 2 * j_minus_1 / D1 + d2_term + d3_term


def volumetric_pressure_tilde(J, D1, D2, D3):
    j_minus_1 = J - 1
    d2_tilde = sp.Piecewise(
        (12 * j_minus_1**2 / D2, D2 > VOL_TOL),
        (0, True),
    )
    d3_tilde = sp.Piecewise(
        (30 * j_minus_1**4 / D3, D3 > VOL_TOL),
        (0, True),
    )
    return (
        volumetric_pressure(J, D1, D2, D3)
        + J * (2 / D1 + d2_tilde + d3_tilde)
    )


def n3_deviatoric_energy_derivatives(I1_bar, C10, C20, C30):
    i1_shift = I1_bar - 3
    w1 = C10 + 2 * C20 * i1_shift + 3 * C30 * i1_shift**2
    w11 = 2 * C20 + 6 * C30 * i1_shift
    return sp.simplify(w1), sp.simplify(w11)


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
# Existing operators
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


# -----------------------------------------------------------------------------
# New priority operators
# -----------------------------------------------------------------------------

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


def build_hex8r_op_constitutive_linear():
    E = sp.Symbol("E", real=True)
    nu = sp.Symbol("nu", real=True)
    c1 = E / ((1 + nu) * (1 - 2 * nu))
    c2 = E / (2 * (1 + nu))

    D = sp.Matrix.zeros(6, 6)
    D[0, 0] = c1 * (1 - nu)
    D[0, 1] = c1 * nu
    D[0, 2] = c1 * nu
    D[1, 0] = c1 * nu
    D[1, 1] = c1 * (1 - nu)
    D[1, 2] = c1 * nu
    D[2, 0] = c1 * nu
    D[2, 1] = c1 * nu
    D[2, 2] = c1 * (1 - nu)
    D[3, 3] = c2
    D[4, 4] = c2
    D[5, 5] = c2

    return MathModel(
        inputs=[E, nu],
        outputs=flatten_row_major(D),
        name="hex8r_op_constitutive_linear",
        input_names=["E", "nu"],
        output_names=[f"D({i+1},{j+1})" for i in range(6) for j in range(6)],
        is_operator=True,
    )


def build_voigt_transform(J0inv: sp.Matrix):
    j11, j12, j13 = J0inv[0, 0], J0inv[0, 1], J0inv[0, 2]
    j21, j22, j23 = J0inv[1, 0], J0inv[1, 1], J0inv[1, 2]
    j31, j32, j33 = J0inv[2, 0], J0inv[2, 1], J0inv[2, 2]

    T = sp.Matrix.zeros(6, 6)
    T[0, 0] = j11 * j11
    T[0, 1] = j21 * j21
    T[0, 2] = j31 * j31
    T[0, 3] = j11 * j21
    T[0, 4] = j21 * j31
    T[0, 5] = j11 * j31

    T[1, 0] = j12 * j12
    T[1, 1] = j22 * j22
    T[1, 2] = j32 * j32
    T[1, 3] = j12 * j22
    T[1, 4] = j22 * j32
    T[1, 5] = j12 * j32

    T[2, 0] = j13 * j13
    T[2, 1] = j23 * j23
    T[2, 2] = j33 * j33
    T[2, 3] = j13 * j23
    T[2, 4] = j23 * j33
    T[2, 5] = j13 * j33

    T[3, 0] = 2 * j11 * j12
    T[3, 1] = 2 * j21 * j22
    T[3, 2] = 2 * j31 * j32
    T[3, 3] = j11 * j22 + j21 * j12
    T[3, 4] = j21 * j32 + j31 * j22
    T[3, 5] = j11 * j32 + j31 * j12

    T[4, 0] = 2 * j12 * j13
    T[4, 1] = 2 * j22 * j23
    T[4, 2] = 2 * j32 * j33
    T[4, 3] = j12 * j23 + j22 * j13
    T[4, 4] = j22 * j33 + j32 * j23
    T[4, 5] = j12 * j33 + j32 * j13

    T[5, 0] = 2 * j13 * j11
    T[5, 1] = 2 * j23 * j21
    T[5, 2] = 2 * j33 * j31
    T[5, 3] = j13 * j21 + j23 * j11
    T[5, 4] = j23 * j31 + j33 * j21
    T[5, 5] = j13 * j31 + j33 * j11
    return T


def build_hex8r_op_rot_dmtx():
    D = mat_symbols("D", 6, 6)
    J0inv = mat_symbols("J0inv", 3, 3)
    rj = sp.Symbol("rj", real=True)

    transform = build_voigt_transform(J0inv)
    D_rotated = sp.simplify(rj * transform.T * D * transform)

    inputs = flatten_row_major(D) + flatten_row_major(J0inv) + [rj]
    outputs = flatten_row_major(D_rotated)
    input_names = (
        [f"D({i+1},{j+1})" for i in range(6) for j in range(6)]
        + [f"J0inv({i+1},{j+1})" for i in range(3) for j in range(3)]
        + ["rj"]
    )
    output_names = [f"D_rot({i+1},{j+1})" for i in range(6) for j in range(6)]
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_rot_dmtx",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


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
# Hyperelastic operators
# -----------------------------------------------------------------------------

def build_hyperelastic_kinematics(F: sp.Matrix):
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


def build_hex8r_op_stress_cauchy_n3():
    F = mat_symbols("F", 3, 3)
    C10, C20, C30, D1, D2, D3 = sp.symbols("C10 C20 C30 D1 D2 D3", real=True)

    kin = build_hyperelastic_kinematics(F)
    J = kin["J"]
    B = kin["B"]
    B_bar = kin["B_bar"]
    I1_bar = kin["I1_bar_B"]

    W1, _ = n3_deviatoric_energy_derivatives(I1_bar, C10, C20, C30)
    pressure = volumetric_pressure(J, D1, D2, D3)
    B_bar_dev = B_bar - (I1_bar / 3) * identity3()
    sigma_matrix = (2 / J) * W1 * B_bar_dev + pressure * identity3()
    stress_voigt = symmetric_matrix_to_voigt(sigma_matrix)

    inputs = flatten_row_major(F) + [C10, C20, C30, D1, D2, D3]
    outputs = list(stress_voigt) + [J] + flatten_row_major(B) + flatten_row_major(B_bar) + [I1_bar, W1]
    input_names = [f"F({i+1},{j+1})" for i in range(3) for j in range(3)] + ["C10", "C20", "C30", "D1", "D2", "D3"]
    output_names = (
        [f"sigma({i+1})" for i in range(6)]
        + ["J"]
        + [f"B({i+1},{j+1})" for i in range(3) for j in range(3)]
        + [f"B_bar({i+1},{j+1})" for i in range(3) for j in range(3)]
        + ["I1_bar", "W1"]
    )
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_stress_cauchy_n3",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


def build_hex8r_op_stress_pk2_n3():
    F = mat_symbols("F", 3, 3)
    C10, C20, C30, D1, D2, D3 = sp.symbols("C10 C20 C30 D1 D2 D3", real=True)

    kin = build_hyperelastic_kinematics(F)
    J = kin["J"]
    J_minus_2_3 = kin["J_minus_2_3"]
    C = kin["C"]
    Cinv = kin["Cinv"]
    C_bar = kin["C_bar"]
    I1_bar = kin["I1_bar_C"]

    W1, _ = n3_deviatoric_energy_derivatives(I1_bar, C10, C20, C30)
    p_vol = volumetric_pressure(J, D1, D2, D3)
    S_matrix = 2 * J_minus_2_3 * W1 * (identity3() - (I1_bar / 3) * Cinv) + p_vol * J * Cinv
    stress_voigt = symmetric_matrix_to_voigt(S_matrix)

    inputs = flatten_row_major(F) + [C10, C20, C30, D1, D2, D3]
    outputs = (
        list(stress_voigt)
        + [J]
        + flatten_row_major(C)
        + flatten_row_major(C_bar)
        + flatten_row_major(Cinv)
        + [I1_bar, W1]
    )
    input_names = [f"F({i+1},{j+1})" for i in range(3) for j in range(3)] + ["C10", "C20", "C30", "D1", "D2", "D3"]
    output_names = (
        [f"S({i+1})" for i in range(6)]
        + ["J"]
        + [f"C({i+1},{j+1})" for i in range(3) for j in range(3)]
        + [f"C_bar({i+1},{j+1})" for i in range(3) for j in range(3)]
        + [f"Cinv({i+1},{j+1})" for i in range(3) for j in range(3)]
        + ["I1_bar", "W1"]
    )
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_stress_pk2_n3",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


def build_hex8r_op_dmat_n3():
    F = mat_symbols("F", 3, 3)
    B_bar = mat_symbols("B_bar", 3, 3)
    J = sp.Symbol("J", real=True)
    I1_bar = sp.Symbol("I1_bar", real=True)
    C10, C20, C30, D1, D2, D3 = sp.symbols("C10 C20 C30 D1 D2 D3", real=True)

    W1, W11 = n3_deviatoric_energy_derivatives(I1_bar, C10, C20, C30)
    p = volumetric_pressure(J, D1, D2, D3)
    p_tilde = volumetric_pressure_tilde(J, D1, D2, D3)
    two_over_J = 2 / J
    coeff1 = two_over_J * (W1 + I1_bar * W11)
    coeff2 = two_over_J * W11

    voigt_map = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
    DMAT = sp.Matrix.zeros(6, 6)

    for m, (i, j_tensor) in enumerate(voigt_map):
        for n, (k, l) in enumerate(voigt_map):
            term1 = kronecker(i, j_tensor) * kronecker(k, l)
            term2 = sp.Rational(1, 2) * (kronecker(i, k) * kronecker(j_tensor, l) + kronecker(i, l) * kronecker(j_tensor, k))
            c_vol = p_tilde * term1 - 2 * p * term2

            c_dev = 0
            if i == j_tensor and k == l:
                c_dev += coeff1 * TWO_OVER_THREE
            if i == j_tensor:
                c_dev -= coeff1 * B_bar[k, l]
            if k == l:
                c_dev -= coeff1 * B_bar[i, j_tensor]
            c_dev += coeff2 * B_bar[i, j_tensor] * B_bar[k, l]

            shear_term = (
                kronecker(i, k) * B_bar[j_tensor, l]
                + kronecker(j_tensor, l) * B_bar[i, k]
                + kronecker(i, l) * B_bar[j_tensor, k]
                + kronecker(j_tensor, k) * B_bar[i, l]
            )
            c_dev += (two_over_J * W1) * sp.Rational(1, 2) * shear_term
            DMAT[m, n] = c_vol + c_dev

    inputs = flatten_row_major(F) + flatten_row_major(B_bar) + [J, I1_bar, C10, C20, C30, D1, D2, D3]
    outputs = flatten_row_major(DMAT)
    input_names = (
        [f"F({i+1},{j+1})" for i in range(3) for j in range(3)]
        + [f"B_bar({i+1},{j+1})" for i in range(3) for j in range(3)]
        + ["J", "I1_bar", "C10", "C20", "C30", "D1", "D2", "D3"]
    )
    output_names = [f"DMAT({i+1},{j+1})" for i in range(6) for j in range(6)]
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_dmat_n3",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


def build_hex8r_op_dmat_pk2_n3():
    F = mat_symbols("F", 3, 3)
    C_bar = mat_symbols("C_bar", 3, 3)
    Cinv = mat_symbols("Cinv", 3, 3)
    J = sp.Symbol("J", real=True)
    I1_bar = sp.Symbol("I1_bar", real=True)
    C10, C20, C30, D1, D2, D3 = sp.symbols("C10 C20 C30 D1 D2 D3", real=True)

    J_minus_2_3 = J**sp.Rational(-2, 3)
    W1, W11 = n3_deviatoric_energy_derivatives(I1_bar, C10, C20, C30)
    p = volumetric_pressure(J, D1, D2, D3)
    p_tilde = volumetric_pressure_tilde(J, D1, D2, D3)

    voigt_map = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
    DMAT = sp.Matrix.zeros(6, 6)
    Iden = identity3()
    proj = Iden - (I1_bar / 3) * Cinv

    for m, (i, j_tensor) in enumerate(voigt_map):
        for n, (k, l) in enumerate(voigt_map):
            c_vol = (
                p_tilde * J * Cinv[i, j_tensor] * Cinv[k, l]
                - p * J * (Cinv[i, k] * Cinv[j_tensor, l] + Cinv[i, l] * Cinv[j_tensor, k])
            )

            coeff1 = 2 * J_minus_2_3 * (W1 + I1_bar * W11)
            coeff2 = 2 * J_minus_2_3 * W11
            c_dev = 0
            if i == j_tensor and k == l:
                c_dev += coeff1 * J_minus_2_3
            if i == j_tensor:
                c_dev -= coeff1 * J_minus_2_3 * (I1_bar / 3) * Cinv[k, l]
            if k == l:
                c_dev -= coeff1 * J_minus_2_3 * (I1_bar / 3) * Cinv[i, j_tensor]
            c_dev += coeff2 * J_minus_2_3 * J_minus_2_3 * proj[i, j_tensor] * proj[k, l]

            term2 = -sp.Rational(1, 2) * (
                kronecker(i, k) * Cinv[j_tensor, l]
                + kronecker(j_tensor, l) * Cinv[i, k]
                + kronecker(i, l) * Cinv[j_tensor, k]
                + kronecker(j_tensor, k) * Cinv[i, l]
            )
            c_dev -= 2 * J_minus_2_3 * W1 * (I1_bar / 3) * term2
            DMAT[m, n] = c_vol + c_dev

    inputs = (
        flatten_row_major(F)
        + flatten_row_major(C_bar)
        + flatten_row_major(Cinv)
        + [J, I1_bar, C10, C20, C30, D1, D2, D3]
    )
    outputs = flatten_row_major(DMAT)
    input_names = (
        [f"F({i+1},{j+1})" for i in range(3) for j in range(3)]
        + [f"C_bar({i+1},{j+1})" for i in range(3) for j in range(3)]
        + [f"Cinv({i+1},{j+1})" for i in range(3) for j in range(3)]
        + ["J", "I1_bar", "C10", "C20", "C30", "D1", "D2", "D3"]
    )
    output_names = [f"DMAT({i+1},{j+1})" for i in range(6) for j in range(6)]
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_dmat_pk2_n3",
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
        build_hex8r_op_constitutive_linear(),
        build_hex8r_op_rot_dmtx(),
        build_hex8r_op_k_matrices(),
        build_hex8r_op_stress_cauchy_n3(),
        build_hex8r_op_stress_pk2_n3(),
        build_hex8r_op_dmat_n3(),
        build_hex8r_op_dmat_pk2_n3(),
    ]
