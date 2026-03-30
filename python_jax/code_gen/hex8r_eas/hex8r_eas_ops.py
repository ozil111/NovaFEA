import sympy as sp

try:
    from sympy_codegen import MathModel
except ImportError:
    # Fallback for cases where the generator is imported from a package path.
    from .sympy_codegen import MathModel


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


def inv3x3_with_det(A: sp.Matrix):
    """
    High-level readable 3x3 inverse using SymPy built-in methods.
    Returns (detA, Ainv).
    """
    detA = A.det()
    Ainv = A.inv()
    return detA, Ainv



# -----------------------------------------------------------------------------
# Operator 1: B-bar averaged nodal gradient
# -----------------------------------------------------------------------------

def hex8_shape(xi, eta, zeta):
    """
    Standard Hex8 shape functions on [-1,1]^3.
    Return 8x1 vector N, node order consistent with the current file:
        1 (-1,-1,-1)
        2 ( 1,-1,-1)
        3 ( 1, 1,-1)
        4 (-1, 1,-1)
        5 (-1,-1, 1)
        6 ( 1,-1, 1)
        7 ( 1, 1, 1)
        8 (-1, 1, 1)
    """
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


def hex8_face_parametrizations():
    """
    Six faces of the reference Hex8.
    Each face is represented by:
        name, substitution dict, local variables (a, b), orientation_sign

    We parametrize each face using two local coordinates (a, b) in [-1,1]^2.
    The outward normal direction is controlled by orientation_sign.
    """
    a, b = sp.symbols("a b", real=True)

    return [
        # zeta = -1, outward normal should point toward -z
        ("zeta=-1", {sp.Symbol("xi"): a, sp.Symbol("eta"): b, sp.Symbol("zeta"): -1}, (a, b), -1),

        # zeta = +1, outward normal should point toward +z
        ("zeta=+1", {sp.Symbol("xi"): a, sp.Symbol("eta"): b, sp.Symbol("zeta"):  1}, (a, b), +1),

        # eta = -1, outward normal should point toward -y
        ("eta=-1",  {sp.Symbol("xi"): a, sp.Symbol("eta"): -1, sp.Symbol("zeta"): b}, (a, b), +1),

        # eta = +1, outward normal should point toward +y
        ("eta=+1",  {sp.Symbol("xi"): a, sp.Symbol("eta"):  1, sp.Symbol("zeta"): b}, (a, b), -1),

        # xi = -1, outward normal should point toward -x
        ("xi=-1",   {sp.Symbol("xi"): -1, sp.Symbol("eta"): a, sp.Symbol("zeta"): b}, (a, b), -1),

        # xi = +1, outward normal should point toward +x
        ("xi=+1",   {sp.Symbol("xi"):  1, sp.Symbol("eta"): a, sp.Symbol("zeta"): b}, (a, b), +1),
    ]


def restricted_hex8_shape_on_face(face_subs):
    """
    Restrict the 8 Hex8 shape functions onto one face.
    Return 8x1 vector.
    """
    xi = sp.Symbol("xi")
    eta = sp.Symbol("eta")
    zeta = sp.Symbol("zeta")
    N = hex8_shape(xi, eta, zeta)
    return sp.Matrix([sp.simplify(N[i].subs(face_subs)) for i in range(8)])


def face_mapping_from_hex8(coord, face_subs):
    """
    Physical mapping x(a,b) of a face induced from the parent Hex8 interpolation:
        x(a,b) = sum_I N_I|face * X_I
    coord: 8x3 nodal coordinate matrix
    Return:
        xvec(a,b): 3x1 vector
    """
    Nf = restricted_hex8_shape_on_face(face_subs)   # 8x1
    return coord.T * Nf                             # 3x1


def face_area_vector_from_mapping(xvec, local_vars, orientation_sign=1):
    """
    Surface area vector:
        a_vec = (dx/da) x (dx/db)
    and optionally flipped by orientation_sign to enforce outward normal.
    """
    a, b = local_vars
    dx_da = sp.diff(xvec, a)
    dx_db = sp.diff(xvec, b)
    return sp.simplify(orientation_sign * dx_da.cross(dx_db))


def integrate_face_surface_vector(coord, face_subs, local_vars, orientation_sign):
    """
    Face contribution to:
        G[I,:] += ∫_face N_I * n dS
               = ∫_{-1}^{1}∫_{-1}^{1} N_I(a,b) * a_vec(a,b) da db
    Return 8x3 matrix.
    """
    a, b = local_vars
    Nf = restricted_hex8_shape_on_face(face_subs)                   # 8x1
    xvec = face_mapping_from_hex8(coord, face_subs)                 # 3x1
    avec = face_area_vector_from_mapping(xvec, local_vars, orientation_sign)  # 3x1

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
    """
    Most high-level FE definition of the B-bar numerator:

        G[I,:] = ∫_{∂Ω} N_I n dS

    where:
        G is 8x3
        G[:,0] corresponds to the old b_x_raw
        G[:,1] corresponds to the old b_y_raw
        G[:,2] corresponds to the old b_z_raw

    Then average gradient is:
        gradN_bar = G / V
    """
    G = sp.Matrix.zeros(8, 3)

    for _, face_subs, local_vars, orientation_sign in hex8_face_parametrizations():
        G += integrate_face_surface_vector(coord, face_subs, local_vars, orientation_sign)

    return sp.simplify(G)


def calc_volume_from_surface_vector(coord, G):
    """
    Volume reconstructed from the boundary integral result.

    By consistency with the original code:
        vol = x · b_x_raw
    Here:
        x = coord[:,0]
        b_x_raw = G[:,0]

    So:
        vol = coord[:,0].dot(G[:,0])
    """
    x = coord[:, 0]
    return sp.expand(x.dot(G[:, 0]))


def build_hex8r_op_bbar_grad():
    coord = mat_symbols("coord", 8, 3)

    G = calc_b_bar_surface_vector(coord)      # 8x3
    vol = calc_volume_from_surface_vector(coord, G)
    BiI = sp.simplify(G / vol)

    inputs = flatten_row_major(coord)
    outputs = flatten_row_major(BiI) + [vol]

    input_names = [f"COORD({i+1},{j+1})" for i in range(8) for j in range(3)]
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_bbar_grad",
        input_names=input_names,
        is_operator=True,
    )


# -----------------------------------------------------------------------------
# Operator 2: Jacobian at center
# -----------------------------------------------------------------------------

def build_hex8r_op_jacobian_center():
    coord = mat_symbols("coord", 8, 3)

    # Natural coordinates of Hex8 nodes.
    XiI = sp.Matrix([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ])

    J = (XiI.T * coord * sp.Rational(1, 8)).T
    detJ, Jinv = inv3x3_with_det(J)

    inputs = flatten_row_major(coord)
    outputs = flatten_row_major(J) + [detJ] + flatten_row_major(Jinv)

    input_names = [f"COORD({i+1},{j+1})" for i in range(8) for j in range(3)]
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_jacobian_center",
        input_names=input_names,
        is_operator=True,
    )


# -----------------------------------------------------------------------------
# Operator 3: Form B matrix from averaged gradient
# -----------------------------------------------------------------------------

def form_B_matrix(BiI: sp.Matrix):
    """Readable SymPy version of VUEL FORM_B_MATRIX."""
    B = sp.zeros(6, 24)
    for k in range(8):
        c = 3 * k
        bx = BiI[k, 0]
        by = BiI[k, 1]
        bz = BiI[k, 2]

        B[0, c + 0] = bx
        B[1, c + 1] = by
        B[2, c + 2] = bz

        B[3, c + 0] = by
        B[3, c + 1] = bx

        B[4, c + 1] = bz
        B[4, c + 2] = by

        B[5, c + 0] = bz
        B[5, c + 2] = bx
    return B



def build_hex8r_op_form_B():
    BiI = mat_symbols("BiI", 8, 3)
    B = form_B_matrix(BiI)

    inputs = flatten_row_major(BiI)
    outputs = flatten_row_major(B)

    input_names = [f"BiI({i+1},{j+1})" for i in range(8) for j in range(3)]
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_form_B",
        input_names=input_names,
        is_operator=True,
    )


# -----------------------------------------------------------------------------
# Operator 4: Internal force from B and stress
# -----------------------------------------------------------------------------

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
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="hex8r_op_internal_force",
        input_names=input_names,
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
    ]
