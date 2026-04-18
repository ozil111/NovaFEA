"""
GitHub Actions workflow generator for CI testing.

Generates .github/workflows/codegen_test.yml that automates:
codegen -> compile -> test

Supports task types:
  - constitutive: --task constitutive --material <name>
  - stiffness:    --task stiffness --element <name>
  - mass:         --task mass --element <name>
  - custom:       --task custom --script <script_path>
"""


# ---------------------------------------------------------------------------
# Default matrix entries
# ---------------------------------------------------------------------------
DEFAULT_MATRIX_ENTRIES = [
    {"task": "constitutive", "name": "isotropic"},
    {"task": "stiffness",    "name": "tet4"},
]


def _format_matrix_yaml(entries):
    """Format matrix entries as YAML, 10-space indented (under 'model:')."""
    lines = []
    for entry in entries:
        lines.append(f"          - task: {entry['task']}")
        lines.append(f"            name: {entry['name']}")
        if "script" in entry:
            lines.append(f"            script: {entry['script']}")
    return "\n".join(lines)


_WORKFLOW_TEMPLATE = """\
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
      fail-fast: false
      matrix:
        model:
{{MATRIX_BLOCK}}
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
          TASK="${{ matrix.model.task }}"
          NAME="${{ matrix.model.name }}"
          SCRIPT="${{ matrix.model.script }}"
          OUTPUT_DIR="../../generated/${NAME}"

          if [ "$TASK" = "custom" ]; then
            if [ -z "$SCRIPT" ]; then
              echo "ERROR: matrix.model.script is required for custom task"
              exit 1
            fi
            python sympy_codegen.py --task custom --script "$SCRIPT" --target all --test --test-output-dir "$OUTPUT_DIR"
          elif [ "$TASK" = "constitutive" ]; then
            python sympy_codegen.py --task constitutive --material "$NAME" --target all --test --test-output-dir "$OUTPUT_DIR"
          elif [ "$TASK" = "stiffness" ] || [ "$TASK" = "mass" ]; then
            python sympy_codegen.py --task "$TASK" --element "$NAME" --target all --test --test-output-dir "$OUTPUT_DIR"
          else
            echo "ERROR: Unknown task: $TASK"
            exit 1
          fi

      - name: Compile all operators
        run: |
          cd generated/${{ matrix.model.name }}
          # If there are subdirectories (multi-operator), iterate; otherwise build directly
          if ls -d */ > /dev/null 2>&1; then
            for dir in */; do
              dir="${dir%/}"
              if [ -f "$dir/build.sh" ]; then
                echo "=== Building $dir ==="
                cd "$dir"
                bash build.sh
                cd ..
              fi
            done
          else
            bash build.sh
          fi

      - name: Run numerical tests
        run: |
          cd generated/${{ matrix.model.name }}
          FAILED=0
          # If there are subdirectories (multi-operator), iterate; otherwise test directly
          if ls -d */ > /dev/null 2>&1; then
            for dir in */; do
              dir="${dir%/}"
              if [ -f "$dir/test_driver.py" ]; then
                echo "=== Testing $dir ==="
                cd "$dir"
                python test_driver.py --cpp-exe ./kernel_cpp --f90-exe ./kernel_f90 --n-runs {{N_RUNS}} --atol {{ATOL}} || FAILED=1
                cd ..
              fi
            done
          else
            python test_driver.py --cpp-exe ./kernel_cpp --f90-exe ./kernel_f90 --n-runs {{N_RUNS}} --atol {{ATOL}} || FAILED=1
          fi
          if [ "$FAILED" -ne 0 ]; then
            echo "Some tests failed!"
            exit 1
          fi
"""


def generate_github_actions_workflow(
    matrix_entries=None,
    python_version: str = "3.11",
    n_runs: int = 1000,
    atol: str = "1e-10",
) -> str:
    """
    Generate a GitHub Actions workflow YAML for codegen testing.

    Args:
        matrix_entries: List of dicts defining the matrix. Each dict has:
            - task: "constitutive", "stiffness", "mass", or "custom"
            - name: model/material/element name (used for output dir)
            - script: (required for custom) path to the custom script
            If None, DEFAULT_MATRIX_ENTRIES is used.
            Example:
                [
                    {"task": "constitutive", "name": "isotropic"},
                    {"task": "stiffness",    "name": "tet4"},
                    {"task": "custom",       "name": "hex8", "script": "./hex8r_eas/hex8r_eas_ops.py"},
                ]
        python_version: Python version for the CI environment
        n_runs: Number of test runs
        atol: Absolute tolerance for numerical comparison
    """
    if matrix_entries is None:
        matrix_entries = DEFAULT_MATRIX_ENTRIES

    matrix_block = _format_matrix_yaml(matrix_entries)

    code = _WORKFLOW_TEMPLATE
    code = code.replace("{{MATRIX_BLOCK}}", matrix_block)
    code = code.replace("{{PYTHON_VERSION}}", python_version)
    code = code.replace("{{N_RUNS}}", str(n_runs))
    code = code.replace("{{ATOL}}", atol)
    return code
