"""
Test driver generator for cross-backend numerical validation.

Generates a standalone test_driver.py that:
1. Uses SymPy lambdify as the reference implementation
2. Runs compiled C++/Fortran executables via subprocess
3. Compares outputs with atol=1e-10
4. Supports --n-runs, --atol, --seed, --cpp-exe, --f90-exe CLI args
5. Provides debug dump on failure

Strategy: The MathModel is serialized via pickle/base64 and embedded directly
in the generated test_driver.py. The unpickling needs the sympy_codegen module,
so we add the project's code_gen directory to sys.path before deserializing.
"""


_TEST_DRIVER_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated test driver for compute_{{MODEL_NAME}}.
Cross-backend numerical validation: SymPy vs C++ vs Fortran.
"""
import argparse
import base64
import os
import pickle
import subprocess
import sys

# Add code_gen directory to sys.path for unpickling MathModel
# Search upward from this script's directory to find python_jax/code_gen
_script_dir = os.path.dirname(os.path.abspath(__file__))
_search_dir = _script_dir
for _ in range(5):  # search up to 5 levels
    _candidate = os.path.join(_search_dir, 'python_jax', 'code_gen')
    if os.path.isdir(_candidate):
        sys.path.insert(0, _candidate)
        break
    _search_dir = os.path.dirname(_search_dir)

import numpy as np
import sympy as sp

# Deserialize the embedded MathModel
_model = pickle.loads(base64.b64decode({{PICKLED_MODEL}}))

# ---------------------------------------------------------------------------
# Model metadata (embedded at generation time)
# ---------------------------------------------------------------------------
MODEL_NAME = {{MODEL_NAME_REPR}}
N_IN = {{N_IN}}
N_OUT = {{N_OUT}}
INPUT_NAMES = {{INPUT_NAMES}}
OUTPUT_NAMES = {{OUTPUT_NAMES}}

# Build the lambdify reference function once using the loaded model
_sym_func = sp.lambdify(_model.inputs, _model.outputs, modules="numpy")


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------
def run_sympy_ref(x: np.ndarray) -> np.ndarray:
    """Compute reference output using SymPy lambdify."""
    result = _sym_func(*x)
    # lambdify may return a scalar or tuple depending on output count
    if N_OUT == 1:
        return np.array([float(result)])
    return np.array([float(v) for v in result])


def run_exec(cmd: list, x: np.ndarray) -> np.ndarray:
    """Run a compiled executable, feed x via stdin, parse stdout."""
    inp = "\\n".join(f"{v:.17g}" for v in x)
    try:
        proc = subprocess.run(
            cmd,
            input=inp.encode(),
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print(f"ERROR: {cmd[0]} timed out", file=sys.stderr)
        raise
    if proc.returncode != 0:
        print(f"ERROR: {cmd[0]} exited with code {proc.returncode}", file=sys.stderr)
        print(f"stderr: {proc.stderr.decode()}", file=sys.stderr)
        raise RuntimeError(f"{cmd[0]} failed")
    output = proc.stdout.decode()
    return np.fromstring(output, sep=" ")


# ---------------------------------------------------------------------------
# Debug dump
# ---------------------------------------------------------------------------
def debug_dump(x, y_sym, y_cpp, y_f90, atol, rtol):
    """Print detailed debug information on test failure."""
    print("=" * 70)
    print(f"TEST FAILED for model {MODEL_NAME} (atol={atol}, rtol={rtol})")
    print("=" * 70)
    print(f"Input ({N_IN} values):")
    for i, name in enumerate(INPUT_NAMES):
        print(f"  {name} = {x[i]:.17g}")
    print()
    print(f"{'Index':<6} {'Output':<20} {'SymPy':<25} {'C++':<25} {'Fortran':<25} {'MaxDiff':<15}")
    print("-" * 120)
    for i in range(N_OUT):
        s = y_sym[i]
        c = y_cpp[i] if y_cpp is not None else float("nan")
        f = y_f90[i] if y_f90 is not None else float("nan")
        diff = max(
            abs(s - c) if y_cpp is not None else 0,
            abs(s - f) if y_f90 is not None else 0,
        )
        name = OUTPUT_NAMES[i] if i < len(OUTPUT_NAMES) else f"out[{i}]"
        print(f"{i:<6} {name:<20} {s:<25.17g} {c:<25.17g} {f:<25.17g} {diff:<15.2e}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Test logic
# ---------------------------------------------------------------------------
def test_once(x, cpp_exe, f90_exe, atol, rtol):
    """Run a single test: generate random input, compare all backends."""
    y_sym = run_sympy_ref(x)

    y_cpp = None
    if cpp_exe:
        y_cpp = run_exec([cpp_exe], x)

    y_f90 = None
    if f90_exe:
        y_f90 = run_exec([f90_exe], x)

    # Validate shapes
    assert y_sym.shape == (N_OUT,), f"SymPy output shape: {y_sym.shape}, expected ({N_OUT},)"
    if y_cpp is not None:
        assert y_cpp.shape == (N_OUT,), f"C++ output shape: {y_cpp.shape}, expected ({N_OUT},)"
    if y_f90 is not None:
        assert y_f90.shape == (N_OUT,), f"Fortran output shape: {y_f90.shape}, expected ({N_OUT},)"

    # Compare
    passed = True
    if y_cpp is not None and not np.allclose(y_sym, y_cpp, atol=atol, rtol=rtol):
        passed = False
    if y_f90 is not None and not np.allclose(y_sym, y_f90, atol=atol, rtol=rtol):
        passed = False

    if not passed:
        debug_dump(x, y_sym, y_cpp, y_f90, atol, rtol)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description=f"Cross-backend numerical test for compute_{MODEL_NAME}"
    )
    parser.add_argument("--n-runs", type=int, default=1000,
                        help="Number of random test runs (default: 1000)")
    parser.add_argument("--atol", type=float, default=1e-10,
                        help="Absolute tolerance (default: 1e-10)")
    parser.add_argument("--rtol", type=float, default=1e-11,
                        help="Relative tolerance (default: 1e-11)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--cpp-exe", type=str, default=None,
                        help="Path to compiled C++ executable")
    parser.add_argument("--f90-exe", type=str, default=None,
                        help="Path to compiled Fortran executable")
    parser.add_argument("--input-range", type=float, nargs=2, default=[0.1, 2.0],
                        help="Range for random inputs (default: 0.1 2.0)")
    args = parser.parse_args()

    if not args.cpp_exe and not args.f90_exe:
        parser.error("At least one of --cpp-exe or --f90-exe is required")

    rng = np.random.default_rng(args.seed)
    n_passed = 0
    n_failed = 0

    for run_idx in range(args.n_runs):
        x = rng.uniform(args.input_range[0], args.input_range[1], size=N_IN)
        try:
            if test_once(x, args.cpp_exe, args.f90_exe, args.atol, args.rtol):
                n_passed += 1
            else:
                n_failed += 1
        except Exception as e:
            print(f"RUN {run_idx}: EXCEPTION: {e}")
            n_failed += 1

    print(f"\\nResults: {n_passed}/{args.n_runs} passed, {n_failed} failed (atol={args.atol}, rtol={args.rtol})")
    if n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
'''


def generate_test_driver(model, task=None, model_name=None) -> str:
    """
    Generate a complete test_driver.py for the given MathModel.

    The MathModel is embedded as a pickled+base64 string, making the
    generated script self-contained. The unpickler needs sympy_codegen
    on sys.path, which is added at the top of the generated script.

    Args:
        model: MathModel instance
        task: CLI task type (unused, kept for API compatibility)
        model_name: model/material/element name (unused, kept for API compatibility)
    """
    import base64
    import pickle

    pickled = base64.b64encode(pickle.dumps(model)).decode('ascii')

    subs = {
        "MODEL_NAME": model.name,
        "MODEL_NAME_REPR": repr(model.name),
        "N_IN": str(len(model.inputs)),
        "N_OUT": str(len(model.outputs)),
        "INPUT_NAMES": repr(model.input_names),
        "OUTPUT_NAMES": repr(model.output_names),
        "PICKLED_MODEL": repr(pickled),
    }

    code = _TEST_DRIVER_TEMPLATE
    for key, value in subs.items():
        code = code.replace("{{" + key + "}}", value)
    return code
