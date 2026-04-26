
#if defined(__CUDACC__)
  #define FEA_DEVICE __device__
  #define FEA_HOST __host__
  #define FEA_HOST_DEVICE __host__ __device__
  #define FEA_RESTRICT __restrict__
#else
  #define FEA_DEVICE
  #define FEA_HOST
  #define FEA_HOST_DEVICE
  #if defined(_WIN32) || defined(_WIN64)
    #if defined(_MSC_VER)
      #define FEA_RESTRICT __restrict
    #else
      #define FEA_RESTRICT __restrict__
    #endif
  #else
    #if defined(__GNUC__) || defined(__clang__)
      #define FEA_RESTRICT __restrict__
    #else
      #define FEA_RESTRICT
    #endif
  #endif
#endif

#if defined(_MSC_VER)
  #define FEA_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
  #define FEA_ALWAYS_INLINE inline __attribute__((always_inline))
#else
  #define FEA_ALWAYS_INLINE inline
#endif

/**
 * @brief Computes the isotropic_D kernel.
 * 
 * @param in Input array (const double*). Layout:
 *   - in[0]: E
 *   - in[1]: nu
 * 
 * @param out Output array (double*). Layout:
 *   - out[0]: D_0_0
 *   - out[1]: D_0_1
 *   - out[2]: D_0_2
 *   - out[3]: D_0_3
 *   - out[4]: D_0_4
 *   - out[5]: D_0_5
 *   - out[6]: D_1_0
 *   - out[7]: D_1_1
 *   - out[8]: D_1_2
 *   - out[9]: D_1_3
 *   - out[10]: D_1_4
 *   - out[11]: D_1_5
 *   - out[12]: D_2_0
 *   - out[13]: D_2_1
 *   - out[14]: D_2_2
 *   - out[15]: D_2_3
 *   - out[16]: D_2_4
 *   - out[17]: D_2_5
 *   - out[18]: D_3_0
 *   - out[19]: D_3_1
 *   - out[20]: D_3_2
 *   - out[21]: D_3_3
 *   - out[22]: D_3_4
 *   - out[23]: D_3_5
 *   - out[24]: D_4_0
 *   - out[25]: D_4_1
 *   - out[26]: D_4_2
 *   - out[27]: D_4_3
 *   - out[28]: D_4_4
 *   - out[29]: D_4_5
 *   - out[30]: D_5_0
 *   - out[31]: D_5_1
 *   - out[32]: D_5_2
 *   - out[33]: D_5_3
 *   - out[34]: D_5_4
 *   - out[35]: D_5_5
 */
FEA_ALWAYS_INLINE void compute_isotropic_D(const double* FEA_RESTRICT in, double* FEA_RESTRICT out) { 


    // --- Chunk 0 ---
    double v_0_0 = 2*in[1];
    double v_0_1 = in[0]/(v_0_0 + 2);
    double v_0_2 = in[0]*in[1]/((1 - v_0_0)*(in[1] + 1));
    double v_0_3 = 2*v_0_1 + v_0_2;
    out[0] = v_0_3;
    out[1] = v_0_2;
    out[2] = v_0_2;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = v_0_2;
    out[7] = v_0_3;
    out[8] = v_0_2;
    out[9] = 0;
    out[10] = 0;
    out[11] = 0;
    out[12] = v_0_2;
    out[13] = v_0_2;
    out[14] = v_0_3;
    out[15] = 0;
    out[16] = 0;
    out[17] = 0;
    out[18] = 0;
    out[19] = 0;
    out[20] = 0;
    out[21] = v_0_1;
    out[22] = 0;
    out[23] = 0;

    // --- Chunk 1 ---
    double v_1_0 = in[0]/(2*in[1] + 2);
    out[24] = 0;
    out[25] = 0;
    out[26] = 0;
    out[27] = 0;
    out[28] = v_1_0;
    out[29] = 0;
    out[30] = 0;
    out[31] = 0;
    out[32] = 0;
    out[33] = 0;
    out[34] = 0;
    out[35] = v_1_0;
}