"""
GitHub Actions workflow generator for CI testing.

Generates .github/workflows/codegen_test.yml that automates:
codegen -> compile -> test
"""


_WORKFLOW_TEMPLATE = '''\
name: Codegen Numerical Test

on:
  push:
    branches: [ main, develop, codegen ]
    paths:
      - 'python_jax/code_gen/**'
      - '.github/workflows/codegen_test.yml'
  pull_request:
    branches: [ main, develop ]
    paths:
      - 'python_jax/code_gen/**'
      - '.github/workflows/codegen_test.yml'
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        model:
          - task: constitutive
            name: isotropic
          - task: stiffness
            name: tet4
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python {{PYTHON_VERSION}}
        uses: actions/setup-python@v5
        with:
          python-version: '{{PYTHON_VERSION}}'

      - name: Install Python dependencies
        run: |
          pip install sympy numpy

      - name: Install compilers
        run: |
          sudo apt-get update
          sudo apt-get install -y clang gfortran

      - name: Generate code + test assets
        run: |
          cd python_jax/code_gen
          if [ "${{ matrix.model.task }}" = "constitutive" ]; then
            FLAG="--material"
          else
            FLAG="--element"
          fi
          python sympy_codegen.py --task ${{ matrix.model.task }} $FLAG ${{ matrix.model.name }} --target all --test --test-output-dir ../../generated/${{ matrix.model.name }}

      - name: Compile
        run: |
          cd generated/${{ matrix.model.name }}
          bash build.sh

      - name: Run numerical tests
        run: |
          cd generated/${{ matrix.model.name }}
          python test_driver.py --cpp-exe ./kernel_cpp --f90-exe ./kernel_f90 --n-runs {{N_RUNS}} --atol {{ATOL}}
'''


def generate_github_actions_workflow(
    model_name: str = "all",
    python_version: str = "3.11",
    n_runs: int = 1000,
    atol: str = "1e-10",
) -> str:
    """
    Generate a GitHub Actions workflow YAML for codegen testing.

    Args:
        model_name: Name of the model to test, or "all" for everything
        python_version: Python version for the CI environment
        n_runs: Number of test runs
        atol: Absolute tolerance for numerical comparison
    """
    code = _WORKFLOW_TEMPLATE
    code = code.replace("{{PYTHON_VERSION}}", python_version)
    code = code.replace("{{N_RUNS}}", str(n_runs))
    code = code.replace("{{ATOL}}", atol)
    return code
