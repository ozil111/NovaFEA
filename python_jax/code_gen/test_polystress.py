import sympy as sp

from compiler_core import IRGenerator, MathModel, PeachPyBackend, RegisterAllocator


def get_polystress_model() -> MathModel:
    b11, b22, b33, b12, b23, b31 = sp.symbols("b11 b22 b33 b12 b23 b31")
    C10, C01, C20, C11, C02, C30, C21, C12, C03 = sp.symbols(
        "C10 C01 C20 C11 C02 C30 C21 C12 C03"
    )
    D1, D2, D3 = sp.symbols("D1 D2 D3")

    # Determinant of a symmetric 3x3 tensor MATB.
    detb = (
        b11 * b22 * b33
        + 2 * b12 * b23 * b31
        - b11 * b23 * b23
        - b22 * b31 * b31
        - b33 * b12 * b12
    )
    J = sp.sqrt(detb)

    i1 = b11 + b22 + b33
    i2 = b11 * b22 + b11 * b33 + b22 * b33 - (b12 * b12 + b23 * b23 + b31 * b31)

    bi1 = i1 / sp.cbrt(J * J)
    bi2 = i2 / sp.cbrt(J**4)

    bi1m3 = bi1 - 3
    bi2m3 = bi2 - 3

    dphidi1 = (
        C10
        + 2 * C20 * bi1m3
        + 3 * C30 * bi1m3 * bi1m3
        + C11 * bi2m3
        + C12 * bi2m3 * bi2m3
        + 2 * C21 * bi1m3 * bi2m3
    )
    dphidi2 = (
        C01
        + 2 * C02 * bi2m3
        + 3 * C03 * bi2m3 * bi2m3
        + C11 * bi1m3
        + C21 * bi1m3 * bi1m3
        + 2 * C12 * bi1m3 * bi2m3
    )

    jm1 = J - 1
    dphidj = jm1 * (2 * D1 + jm1 * jm1 * (4 * D2 + jm1 * jm1 * 6 * D3))

    inv2j = 2 / J
    j2third = 1 / sp.cbrt(J * J)
    j4third = 1 / sp.cbrt(J**4)

    AA = (dphidi1 + dphidi2 * bi1) * inv2j * j2third
    BB = dphidi2 * inv2j * j4third
    CC = sp.Rational(1, 3) * inv2j * (bi1 * dphidi1 + 2 * bi2 * dphidi2)

    sig11 = AA * b11 - BB * b11 * b11 - CC + dphidj
    sig22 = AA * b22 - BB * b22 * b22 - CC + dphidj
    sig33 = AA * b33 - BB * b33 * b33 - CC + dphidj
    sig12 = AA * b12 - BB * b12 * b12
    sig23 = AA * b23 - BB * b23 * b23
    sig31 = AA * b31 - BB * b31 * b31

    inputs = [
        b11,
        b22,
        b33,
        b12,
        b23,
        b31,
        C10,
        C01,
        C20,
        C11,
        C02,
        C30,
        C21,
        C12,
        C03,
        D1,
        D2,
        D3,
    ]
    outputs = [sig11, sig22, sig33, sig12, sig23, sig31]
    model = MathModel(name="polystress_nomullins", inputs=inputs, outputs=outputs)
    # Keep compatibility with python_jax.code_gen.sympy_codegen.FEACompiler expectations.
    model.input_names = [str(s) for s in inputs]
    model.is_operator = False
    return model


def get_model() -> MathModel:
    return get_polystress_model()


def generate_polystress2_peachpy(output_file: str = "polystress2_gen.py") -> str:
    model = get_polystress_model()
    ir = IRGenerator().generate_ir(model)
    allocated = RegisterAllocator().allocate(ir)
    backend = PeachPyBackend()
    out_path = backend.render(model, allocated, output_file)
    return str(out_path)


if __name__ == "__main__":
    path = generate_polystress2_peachpy("polystress2_gen.py")
    print(f"Generated PeachPy kernel: {path}")
