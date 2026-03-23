/**
 * @brief Computes the tet4_op_dN_dnat kernel.
 * @note This is an optimized operator kernel.
 * 
 * @param in Input array (const double*). Layout:
 *   - in[0]: xi
 *   - in[1]: eta
 *   - in[2]: zeta
 * 
 * @param out Output array (double*). Layout:
 *   - out[0..11]: Flattened result, row-major order.
 */
[[gnu::always_inline]] inline void compute_tet4_op_dN_dnat(const double* __restrict__ in, double* __restrict__ out) { 
    const double& xi = in[0];
    const double& eta = in[1];
    const double& zeta = in[2];
    (void)xi;
    (void)eta;
    (void)zeta;

    // --- Chunk 0 ---
    out[0] = -1;
    out[1] = -1;
    out[2] = -1;
    out[3] = 1;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[7] = 1;
    out[8] = 0;
    out[9] = 0;
    out[10] = 0;
    out[11] = 1;
}