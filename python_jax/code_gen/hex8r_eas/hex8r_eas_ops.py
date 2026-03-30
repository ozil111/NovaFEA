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

def calc_b_bar_component(y: sp.Matrix, z: sp.Matrix):
    """
    SymPy transcription of VUEL CALC_B_BAR.
    Input:
        y, z : 8-vectors
    Output:
        b    : 8-vector

    This intentionally follows the original algebraic structure, but is kept in
    symbolic vector form for readability.
    """
    y1, y2, y3, y4, y5, y6, y7, y8 = list(y)
    z1, z2, z3, z4, z5, z6, z7, z8 = list(z)
    one_over_twelve = sp.Rational(1, 12)

    b = sp.Matrix([
        -(
            y2 * (z3 + z4 - z5 - z6)
            + y3 * (-z2 + z4)
            + y4 * (-z2 - z3 + z5 + z8)
            + y5 * (z2 - z4 + z6 - z8)
            + y6 * (z2 - z5)
            + y8 * (-z4 + z5)
        ) * one_over_twelve,
        (
            y1 * (z3 + z4 - z5 - z6)
            + y3 * (-z1 - z4 + z6 + z7)
            + y4 * (-z1 + z3)
            + y5 * (z1 - z6)
            + y6 * (z1 - z3 + z5 - z7)
            + y7 * (-z3 + z6)
        ) * one_over_twelve,
        -(
            y1 * (z2 - z4)
            + y2 * (-z1 - z4 + z6 + z7)
            + y4 * (z1 + z2 - z7 - z8)
            + y6 * (-z2 + z7)
            + y7 * (-z2 + z4 - z6 + z8)
            + y8 * (z4 - z7)
        ) * one_over_twelve,
        -(
            y1 * (z2 + z3 - z5 - z8)
            + y2 * (-z1 + z3)
            + y3 * (-z1 - z2 + z7 + z8)
            + y5 * (z1 - z8)
            + y7 * (-z3 + z8)
            + y8 * (z1 - z3 + z5 - z7)
        ) * one_over_twelve,
        (
            y1 * (z2 - z4 + z6 - z8)
            + y2 * (-z1 + z6)
            + y4 * (z1 - z8)
            + y6 * (-z1 - z2 + z7 + z8)
            + y7 * (-z6 + z8)
            + y8 * (z1 + z4 - z6 - z7)
        ) * one_over_twelve,
        (
            y1 * (z2 - z5)
            + y2 * (-z1 + z3 - z5 + z7)
            + y3 * (-z2 + z7)
            + y5 * (z1 + z2 - z7 - z8)
            + y7 * (-z2 - z3 + z5 + z8)
            + y8 * (z5 - z7)
        ) * one_over_twelve,
        (
            y2 * (z3 - z6)
            + y3 * (-z2 + z4 - z6 + z8)
            + y4 * (-z3 + z8)
            + y5 * (z6 - z8)
            + y6 * (z2 + z3 - z5 - z8)
            + y8 * (-z3 - z4 + z5 + z6)
        ) * one_over_twelve,
        -(
            y1 * (z4 - z5)
            + y3 * (-z4 + z7)
            + y4 * (-z1 + z3 - z5 + z7)
            + y5 * (z1 + z4 - z6 - z7)
            + y6 * (z5 - z7)
            + y7 * (-z3 - z4 + z5 + z6)
        ) * one_over_twelve,
    ])
    return b


def calc_vol_bbar(b1: sp.Matrix, x: sp.Matrix):
    """SymPy version of VUEL CALC_VOL_BBAR."""
    return sp.expand(x.dot(b1))


def build_hex8r_op_bbar_grad():
    coord = mat_symbols("coord", 8, 3)

    x = coord[:, 0]
    y = coord[:, 1]
    z = coord[:, 2]

    b_x_raw = calc_b_bar_component(y, z)
    b_y_raw = calc_b_bar_component(z, x)
    b_z_raw = calc_b_bar_component(x, y)

    vol = calc_vol_bbar(b_x_raw, x)
    BiI = sp.Matrix.hstack(b_x_raw / vol, b_y_raw / vol, b_z_raw / vol)

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
