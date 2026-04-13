"""
Build script generators for CI testing.

Generates build.sh (Linux/macOS/CI) and build.bat (Windows)
that compile C++ and Fortran kernels with their wrappers.

Note: C++ wrapper includes kernel.cpp via #include, so only main.cpp needs
to be compiled. Fortran compiles both kernel.f90 and main.f90 together.
"""
import textwrap


def generate_build_sh(model, kernel_cpp="kernel.cpp", kernel_f90="kernel.f90",
                      main_cpp="main.cpp", main_f90="main.f90") -> str:
    """
    Generate a build.sh script for Linux/macOS/CI.

    Compiles:
    - C++: main.cpp only (it #includes kernel.cpp), with -O2 -fno-fast-math
    - Fortran: kernel.f90 + main.f90, with -O2 -fno-fast-math
    """
    name = model.name

    code = textwrap.dedent(f'''\
        #!/usr/bin/env bash
        # Auto-generated build script for compute_{name}
        set -euo pipefail

        KERNEL_F90="{kernel_f90}"
        MAIN_CPP="{main_cpp}"
        MAIN_F90="{main_f90}"
        OUTPUT_CPP="kernel_cpp"
        OUTPUT_F90="kernel_f90"

        echo "=== Building C++ kernel ==="
        if command -v clang++ &> /dev/null; then
            CXX="clang++"
        elif command -v g++ &> /dev/null; then
            CXX="g++"
        else
            echo "ERROR: No C++ compiler found (tried clang++, g++)"
            exit 1
        fi
        echo "Using C++ compiler: $CXX"
        # main.cpp #includes kernel.cpp directly, so only compile main.cpp
        "$CXX" -O2 -fno-fast-math "$MAIN_CPP" -lm -o "$OUTPUT_CPP"
        echo "Built: $OUTPUT_CPP"

        echo ""
        echo "=== Building Fortran kernel ==="
        if command -v gfortran &> /dev/null; then
            FC="gfortran"
        else
            echo "ERROR: gfortran not found"
            exit 1
        fi
        echo "Using Fortran compiler: $FC"
        "$FC" -O2 -fno-fast-math "$KERNEL_F90" "$MAIN_F90" -o "$OUTPUT_F90"
        echo "Built: $OUTPUT_F90"

        echo ""
        echo "=== Build complete ==="
    ''')
    return code


def generate_build_bat(model, kernel_cpp="kernel.cpp", kernel_f90="kernel.f90",
                       main_cpp="main.cpp", main_f90="main.f90") -> str:
    """
    Generate a build.bat script for Windows.

    Compiles:
    - C++: main.cpp only (it #includes kernel.cpp), with -O2 -fno-fast-math
    - Fortran: kernel.f90 + main.f90, with -O2 -fno-fast-math
    """
    name = model.name

    code = textwrap.dedent(f'''\
        @echo off
        REM Auto-generated build script for compute_{name}
        setlocal enabledelayedexpansion

        set KERNEL_F90={kernel_f90}
        set MAIN_CPP={main_cpp}
        set MAIN_F90={main_f90}
        set OUTPUT_CPP=kernel_cpp.exe
        set OUTPUT_F90=kernel_f90.exe

        echo === Building C++ kernel ===
        where cl >nul 2>nul
        if not errorlevel 1 (
            echo Using MSVC
            REM main.cpp #includes kernel.cpp directly, so only compile main.cpp
            cl /O2 /fp:precise "%MAIN_CPP%" /Fe:"%OUTPUT_CPP%" /link
        ) else (
            where clang++ >nul 2>nul
            if not errorlevel 1 (
                echo Using clang++
                clang++ -O2 -fno-fast-math "%MAIN_CPP%" -o "%OUTPUT_CPP%"
            ) else (
                where g++ >nul 2>nul
                if not errorlevel 1 (
                    echo Using g++
                    g++ -O2 -fno-fast-math "%MAIN_CPP%" -o "%OUTPUT_CPP%"
                ) else (
                    echo ERROR: No C++ compiler found
                    exit /b 1
                )
            )
        )
        echo Built: %OUTPUT_CPP%

        echo.
        echo === Building Fortran kernel ===
        where gfortran >nul 2>nul
        if not errorlevel 1 (
            echo Using gfortran
            gfortran -O2 -fno-fast-math "%KERNEL_F90%" "%MAIN_F90%" -o "%OUTPUT_F90%"
            echo Built: %OUTPUT_F90%
        ) else (
            echo WARNING: gfortran not found, skipping Fortran build
        )

        echo.
        echo === Build complete ===
    ''')
    return code
