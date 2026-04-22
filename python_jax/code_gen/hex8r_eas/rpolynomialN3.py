"""Material-related operators for hex8r EAS element.

This module is self-contained: it defines its own utility functions and
material helpers, and exposes all material operators through ``get_model()``.

Usage::

    python sympy_codegen.py --task custom --script ./hex8r_eas/hex8r_eas_material_ops.py
"""

import sympy as sp

try:
    from sympy_codegen import MathModel
except ImportError:
    from .sympy_codegen import MathModel


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

VOL_TOL = sp.Float("1.0e-20")
TWO_OVER_THREE = sp.Rational(2, 3)


# -----------------------------------------------------------------------------
# Small utilities (self-contained copies for independence)
# -----------------------------------------------------------------------------

def mat_symbols(prefix: str, rows: int, cols: int):
    """Create a rows x cols SymPy Matrix with readable scalar symbols."""
    return sp.Matrix(rows, cols, lambda i, j: sp.Symbol(f"{prefix}_{i+1}_{j+1}", real=True))


def flatten_row_major(mat: sp.Matrix):
    """Flatten a SymPy Matrix in row-major order."""
    return [mat[i, j] for i in range(mat.rows) for j in range(mat.cols)]


def identity3():
    return sp.eye(3)


def kronecker(i: int, j: int):
    return sp.Integer(1) if i == j else sp.Integer(0)


def symmetric_matrix_to_voigt(mat: sp.Matrix):
    return sp.Matrix([
        mat[0, 0],
        mat[1, 1],
        mat[2, 2],
        mat[0, 1],
        mat[1, 2],
        mat[0, 2],
    ])


# -----------------------------------------------------------------------------
# Material-specific helpers
# -----------------------------------------------------------------------------

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


def _deviatoric_energy_n3(I1_bar, C10, C20, C30):
    """3rd-order reduced polynomial deviatoric strain energy W_dev(I1_bar)."""
    i1_shift = I1_bar - 3
    return C10 * i1_shift + C20 * i1_shift**2 + C30 * i1_shift**3


def _volumetric_energy_n3(J, D1, D2, D3):
    """3rd-order volumetric strain energy W_vol(J)."""
    j_minus_1 = J - 1
    d2_term = sp.Piecewise((j_minus_1**4 / D2, D2 > VOL_TOL), (0, True))
    d3_term = sp.Piecewise((j_minus_1**6 / D3, D3 > VOL_TOL), (0, True))
    return j_minus_1**2 / D1 + d2_term + d3_term


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


# -----------------------------------------------------------------------------
# Material operators
# -----------------------------------------------------------------------------

def build_mat_op_constitutive_linear():
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
        name="mat_op_constitutive_linear",
        input_names=["E", "nu"],
        output_names=[f"D({i+1},{j+1})" for i in range(6) for j in range(6)],
        is_operator=True,
    )


def build_mat_op_rot_dmtx():
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
        name="mat_op_rot_dmtx",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


def build_mat_op_stress_cauchy_n3():
    """N3 reduced polynomial Cauchy stress.

    Inputs are kinematic quantities (from ``mat_op_kinematics``) and
    material parameters — no deformation gradient F is needed.
    """
    J = sp.Symbol("J", real=True)
    B_bar = mat_symbols("B_bar", 3, 3)
    C10, C20, C30, D1, D2, D3 = sp.symbols("C10 C20 C30 D1 D2 D3", real=True)

    I1_bar = sp.trace(B_bar)
    B = J**sp.Rational(2, 3) * B_bar

    W1, _ = n3_deviatoric_energy_derivatives(I1_bar, C10, C20, C30)
    pressure = volumetric_pressure(J, D1, D2, D3)
    B_bar_dev = B_bar - (I1_bar / 3) * identity3()
    sigma_matrix = (2 / J) * W1 * B_bar_dev + pressure * identity3()
    stress_voigt = symmetric_matrix_to_voigt(sigma_matrix)

    inputs = [J] + flatten_row_major(B_bar) + [C10, C20, C30, D1, D2, D3]
    outputs = list(stress_voigt) + [J] + flatten_row_major(B) + flatten_row_major(B_bar) + [I1_bar, W1]
    input_names = (
        ["J"]
        + [f"B_bar({i+1},{j+1})" for i in range(3) for j in range(3)]
        + ["C10", "C20", "C30", "D1", "D2", "D3"]
    )
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
        name="mat_op_stress_cauchy_n3",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


def build_mat_op_stress_pk2_n3():
    """N3 reduced polynomial 2nd Piola-Kirchhoff stress.

    Inputs are kinematic quantities (from ``mat_op_kinematics``) and
    material parameters — no deformation gradient F is needed.
    """
    J = sp.Symbol("J", real=True)
    J_minus_2_3 = sp.Symbol("J_minus_2_3", real=True)
    Cinv = mat_symbols("Cinv", 3, 3)
    C_bar = mat_symbols("C_bar", 3, 3)
    I1_bar = sp.Symbol("I1_bar", real=True)
    C10, C20, C30, D1, D2, D3 = sp.symbols("C10 C20 C30 D1 D2 D3", real=True)

    C = J**sp.Rational(2, 3) * C_bar

    W1, _ = n3_deviatoric_energy_derivatives(I1_bar, C10, C20, C30)
    p_vol = volumetric_pressure(J, D1, D2, D3)
    S_matrix = 2 * J_minus_2_3 * W1 * (identity3() - (I1_bar / 3) * Cinv) + p_vol * J * Cinv
    stress_voigt = symmetric_matrix_to_voigt(S_matrix)

    inputs = (
        [J, J_minus_2_3]
        + flatten_row_major(Cinv)
        + flatten_row_major(C_bar)
        + [I1_bar, C10, C20, C30, D1, D2, D3]
    )
    outputs = (
        list(stress_voigt)
        + [J]
        + flatten_row_major(C)
        + flatten_row_major(C_bar)
        + flatten_row_major(Cinv)
        + [I1_bar, W1]
    )
    input_names = (
        ["J", "J_minus_2_3"]
        + [f"Cinv({i+1},{j+1})" for i in range(3) for j in range(3)]
        + [f"C_bar({i+1},{j+1})" for i in range(3) for j in range(3)]
        + ["I1_bar", "C10", "C20", "C30", "D1", "D2", "D3"]
    )
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
        name="mat_op_stress_pk2_n3",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


def build_mat_op_dmat_n3():
    """N3 reduced polynomial Cauchy tangent modulus (material).

    F has been removed from inputs as it was unused in the computation.
    """
    B_bar = mat_symbols("B_bar", 3, 3)
    J = sp.Symbol("J", real=True)
    I1_bar = sp.Symbol("I1_bar", real=True)
    C10, C20, C30, D1, D2, D3 = sp.symbols("C10 C20 C30 D1 D2 D3", real=True)

    # Automatic differentiation from potential energy functions
    W_dev = _deviatoric_energy_n3(I1_bar, C10, C20, C30)
    W1 = sp.diff(W_dev, I1_bar)
    W11 = sp.diff(W_dev, I1_bar, 2)

    W_vol = _volumetric_energy_n3(J, D1, D2, D3)
    p = sp.diff(W_vol, J)
    # p_tilde = d(J*p)/dJ = p + J*dp/dJ
    p_tilde = sp.diff(J * p, J)

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

    inputs = flatten_row_major(B_bar) + [J, I1_bar, C10, C20, C30, D1, D2, D3]
    outputs = flatten_row_major(DMAT)
    input_names = (
        [f"B_bar({i+1},{j+1})" for i in range(3) for j in range(3)]
        + ["J", "I1_bar", "C10", "C20", "C30", "D1", "D2", "D3"]
    )
    output_names = [f"DMAT({i+1},{j+1})" for i in range(6) for j in range(6)]
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="mat_op_dmat_n3",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


def build_mat_op_dmat_pk2_n3():
    """N3 reduced polynomial PK2 tangent modulus (material).

    F has been removed from inputs as it was unused in the computation.
    """
    C_bar = mat_symbols("C_bar", 3, 3)
    Cinv = mat_symbols("Cinv", 3, 3)
    J = sp.Symbol("J", real=True)
    I1_bar = sp.Symbol("I1_bar", real=True)
    C10, C20, C30, D1, D2, D3 = sp.symbols("C10 C20 C30 D1 D2 D3", real=True)

    J_minus_2_3 = sp.Symbol("J_minus_2_3", real=True)
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
        flatten_row_major(C_bar)
        + flatten_row_major(Cinv)
        + [J, I1_bar, J_minus_2_3, C10, C20, C30, D1, D2, D3]
    )
    outputs = flatten_row_major(DMAT)
    input_names = (
        [f"C_bar({i+1},{j+1})" for i in range(3) for j in range(3)]
        + [f"Cinv({i+1},{j+1})" for i in range(3) for j in range(3)]
        + ["J", "I1_bar", "J_minus_2_3", "C10", "C20", "C30", "D1", "D2", "D3"]
    )
    output_names = [f"DMAT({i+1},{j+1})" for i in range(6) for j in range(6)]
    return MathModel(
        inputs=inputs,
        outputs=outputs,
        name="mat_op_dmat_pk2_n3",
        input_names=input_names,
        output_names=output_names,
        is_operator=True,
    )


# -----------------------------------------------------------------------------
# Public entry for --task custom
# -----------------------------------------------------------------------------

def get_model():
    return [
        build_mat_op_constitutive_linear(),
        build_mat_op_rot_dmtx(),
        build_mat_op_stress_cauchy_n3(),
        build_mat_op_stress_pk2_n3(),
        build_mat_op_dmat_n3(),
        build_mat_op_dmat_pk2_n3(),
    ]
