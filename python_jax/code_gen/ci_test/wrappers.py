"""
C++ and Fortran wrapper generators for CI testing.

Generates main.cpp / main.f90 that read inputs from stdin,
call compute_xxx(), and write outputs to stdout.
"""
import textwrap


def generate_cpp_main(model) -> str:
    """
    Generate a C++ main.cpp wrapper for the compute kernel.

    The wrapper:
    1. Includes kernel.cpp directly (needed because kernel uses FEA_ALWAYS_INLINE)
    2. Reads N doubles from stdin
    3. Calls compute_{name}(in, out)
    4. Writes M doubles to stdout (space-separated, single line)
    """
    n_in = len(model.inputs)
    n_out = len(model.outputs)
    name = model.name

    code = textwrap.dedent(f"""\
        // Auto-generated C++ wrapper for compute_{name}
        // Reads {n_in} doubles from stdin, writes {n_out} doubles to stdout
        #include <cmath>
        #include <iostream>
        #include <iomanip>

        // Include kernel source directly (kernel uses FEA_ALWAYS_INLINE)
        #include "kernel.cpp"

        int main() {{
            double in[{n_in}];
            double out[{n_out}];

            for (int i = 0; i < {n_in}; i++) {{
                std::cin >> in[i];
            }}

            compute_{name}(in, out);

            // Use high precision output (17 significant digits for double)
            std::cout << std::setprecision(17);
            for (int i = 0; i < {n_out}; i++) {{
                std::cout << out[i];
                if (i < {n_out} - 1) std::cout << " ";
            }}
            std::cout << std::endl;

            return 0;
        }}
    """)
    return code


def generate_f90_main(model) -> str:
    """
    Generate a Fortran main.f90 wrapper for the compute kernel.

    The wrapper:
    1. Reads N doubles from stdin (one per line)
    2. Calls compute_{name}(in_vec, out_vec)
    3. Writes M doubles to stdout (one per line)
    """
    n_in = len(model.inputs)
    n_out = len(model.outputs)
    name = model.name

    code = textwrap.dedent(f"""\
        ! Auto-generated Fortran wrapper for compute_{name}
        ! Reads {n_in} doubles from stdin, writes {n_out} doubles to stdout
        program main
            implicit none
            double precision :: in_vec({n_in})
            double precision :: out_vec({n_out})
            integer :: i

            do i = 1, {n_in}
                read(*, *) in_vec(i)
            end do

            call compute_{name}(in_vec, out_vec)

            do i = 1, {n_out}
                write(*, '(ES25.16)') out_vec(i)
            end do
        end program main
    """)
    return code
