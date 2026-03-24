/**
 * @brief Computes the tet4_op_assembly kernel.
 * @note This is an optimized operator kernel.
 * 
 * @param in Input array (const double*). Layout:
 *   - in[0]: dN_dx[0]
 *   - in[1]: dN_dx[1]
 *   - in[2]: dN_dx[2]
 *   - in[3]: dN_dx[3]
 *   - in[4]: dN_dx[4]
 *   - in[5]: dN_dx[5]
 *   - in[6]: dN_dx[6]
 *   - in[7]: dN_dx[7]
 *   - in[8]: dN_dx[8]
 *   - in[9]: dN_dx[9]
 *   - in[10]: dN_dx[10]
 *   - in[11]: dN_dx[11]
 *   - in[12]: D[0]
 *   - in[13]: D[1]
 *   - in[14]: D[2]
 *   - in[15]: D[3]
 *   - in[16]: D[4]
 *   - in[17]: D[5]
 *   - in[18]: D[6]
 *   - in[19]: D[7]
 *   - in[20]: D[8]
 *   - in[21]: D[9]
 *   - in[22]: D[10]
 *   - in[23]: D[11]
 *   - in[24]: D[12]
 *   - in[25]: D[13]
 *   - in[26]: D[14]
 *   - in[27]: D[15]
 *   - in[28]: D[16]
 *   - in[29]: D[17]
 *   - in[30]: D[18]
 *   - in[31]: D[19]
 *   - in[32]: D[20]
 *   - in[33]: D[21]
 *   - in[34]: D[22]
 *   - in[35]: D[23]
 *   - in[36]: D[24]
 *   - in[37]: D[25]
 *   - in[38]: D[26]
 *   - in[39]: D[27]
 *   - in[40]: D[28]
 *   - in[41]: D[29]
 *   - in[42]: D[30]
 *   - in[43]: D[31]
 *   - in[44]: D[32]
 *   - in[45]: D[33]
 *   - in[46]: D[34]
 *   - in[47]: D[35]
 *   - in[48]: detJ
 *   - in[49]: weight
 * 
 * @param out Output array (double*). Layout:
 *   - out[0..143]: Flattened result, row-major order.
 */
#include <cmath>
[[gnu::always_inline]] inline void compute_tet4_op_assembly(const double* __restrict__ in, double* __restrict__ out) { 
    const double& dN_dx0 = in[0];
    const double& dN_dx1 = in[1];
    const double& dN_dx2 = in[2];
    const double& dN_dx3 = in[3];
    const double& dN_dx4 = in[4];
    const double& dN_dx5 = in[5];
    const double& dN_dx6 = in[6];
    const double& dN_dx7 = in[7];
    const double& dN_dx8 = in[8];
    const double& dN_dx9 = in[9];
    const double& dN_dx10 = in[10];
    const double& dN_dx11 = in[11];

    const double& D0 = in[12];
    const double& D1 = in[13];
    const double& D2 = in[14];
    const double& D3 = in[15];
    const double& D4 = in[16];
    const double& D5 = in[17];
    const double& D6 = in[18];
    const double& D7 = in[19];
    const double& D8 = in[20];
    const double& D9 = in[21];
    const double& D10 = in[22];
    const double& D11 = in[23];
    const double& D12 = in[24];
    const double& D13 = in[25];
    const double& D14 = in[26];
    const double& D15 = in[27];
    const double& D16 = in[28];
    const double& D17 = in[29];
    const double& D18 = in[30];
    const double& D19 = in[31];
    const double& D20 = in[32];
    const double& D21 = in[33];
    const double& D22 = in[34];
    const double& D23 = in[35];
    const double& D24 = in[36];
    const double& D25 = in[37];
    const double& D26 = in[38];
    const double& D27 = in[39];
    const double& D28 = in[40];
    const double& D29 = in[41];
    const double& D30 = in[42];
    const double& D31 = in[43];
    const double& D32 = in[44];
    const double& D33 = in[45];
    const double& D34 = in[46];
    const double& D35 = in[47];
    const double& detJ = in[48];
    const double& weight = in[49];

    // --- Chunk 0 ---
    double v_0_0 = D0*dN_dx0 + D18*dN_dx4 + D30*dN_dx8;
    double v_0_1 = D21*dN_dx4 + D3*dN_dx0 + D33*dN_dx8;
    double v_0_2 = D23*dN_dx4 + D35*dN_dx8 + D5*dN_dx0;
    double v_0_3 = weight*fabs(detJ);
    double v_0_4 = D1*dN_dx0 + D19*dN_dx4 + D31*dN_dx8;
    double v_0_5 = D22*dN_dx4 + D34*dN_dx8 + D4*dN_dx0;
    double v_0_6 = D2*dN_dx0 + D20*dN_dx4 + D32*dN_dx8;
    double v_0_7 = D18*dN_dx0 + D24*dN_dx8 + D6*dN_dx4;
    double v_0_8 = D21*dN_dx0 + D27*dN_dx8 + D9*dN_dx4;
    double v_0_9 = D11*dN_dx4 + D23*dN_dx0 + D29*dN_dx8;
    double v_0_10 = D19*dN_dx0 + D25*dN_dx8 + D7*dN_dx4;
    double v_0_11 = D10*dN_dx4 + D22*dN_dx0 + D28*dN_dx8;
    double v_0_12 = D20*dN_dx0 + D26*dN_dx8 + D8*dN_dx4;
    out[0] = v_0_3*(dN_dx0*v_0_0 + dN_dx4*v_0_1 + dN_dx8*v_0_2);
    out[1] = v_0_3*(dN_dx0*v_0_1 + dN_dx4*v_0_4 + dN_dx8*v_0_5);
    out[2] = v_0_3*(dN_dx0*v_0_2 + dN_dx4*v_0_5 + dN_dx8*v_0_6);
    out[3] = v_0_3*(dN_dx1*v_0_0 + dN_dx5*v_0_1 + dN_dx9*v_0_2);
    out[4] = v_0_3*(dN_dx1*v_0_1 + dN_dx5*v_0_4 + dN_dx9*v_0_5);
    out[5] = v_0_3*(dN_dx1*v_0_2 + dN_dx5*v_0_5 + dN_dx9*v_0_6);
    out[6] = v_0_3*(dN_dx10*v_0_2 + dN_dx2*v_0_0 + dN_dx6*v_0_1);
    out[7] = v_0_3*(dN_dx10*v_0_5 + dN_dx2*v_0_1 + dN_dx6*v_0_4);
    out[8] = v_0_3*(dN_dx10*v_0_6 + dN_dx2*v_0_2 + dN_dx6*v_0_5);
    out[9] = v_0_3*(dN_dx11*v_0_2 + dN_dx3*v_0_0 + dN_dx7*v_0_1);
    out[10] = v_0_3*(dN_dx11*v_0_5 + dN_dx3*v_0_1 + dN_dx7*v_0_4);
    out[11] = v_0_3*(dN_dx11*v_0_6 + dN_dx3*v_0_2 + dN_dx7*v_0_5);
    out[12] = v_0_3*(dN_dx0*v_0_7 + dN_dx4*v_0_8 + dN_dx8*v_0_9);
    out[13] = v_0_3*(dN_dx0*v_0_8 + dN_dx4*v_0_10 + dN_dx8*v_0_11);
    out[14] = v_0_3*(dN_dx0*v_0_9 + dN_dx4*v_0_11 + dN_dx8*v_0_12);
    out[15] = v_0_3*(dN_dx1*v_0_7 + dN_dx5*v_0_8 + dN_dx9*v_0_9);
    out[16] = v_0_3*(dN_dx1*v_0_8 + dN_dx5*v_0_10 + dN_dx9*v_0_11);
    out[17] = v_0_3*(dN_dx1*v_0_9 + dN_dx5*v_0_11 + dN_dx9*v_0_12);
    out[18] = v_0_3*(dN_dx10*v_0_9 + dN_dx2*v_0_7 + dN_dx6*v_0_8);
    out[19] = v_0_3*(dN_dx10*v_0_11 + dN_dx2*v_0_8 + dN_dx6*v_0_10);
    out[20] = v_0_3*(dN_dx10*v_0_12 + dN_dx2*v_0_9 + dN_dx6*v_0_11);
    out[21] = v_0_3*(dN_dx11*v_0_9 + dN_dx3*v_0_7 + dN_dx7*v_0_8);
    out[22] = v_0_3*(dN_dx11*v_0_11 + dN_dx3*v_0_8 + dN_dx7*v_0_10);
    out[23] = v_0_3*(dN_dx11*v_0_12 + dN_dx3*v_0_9 + dN_dx7*v_0_11);

    // --- Chunk 1 ---
    double v_1_0 = D12*dN_dx8 + D24*dN_dx4 + D30*dN_dx0;
    double v_1_1 = D15*dN_dx8 + D27*dN_dx4 + D33*dN_dx0;
    double v_1_2 = D17*dN_dx8 + D29*dN_dx4 + D35*dN_dx0;
    double v_1_3 = weight*fabs(detJ);
    double v_1_4 = D13*dN_dx8 + D25*dN_dx4 + D31*dN_dx0;
    double v_1_5 = D16*dN_dx8 + D28*dN_dx4 + D34*dN_dx0;
    double v_1_6 = D14*dN_dx8 + D26*dN_dx4 + D32*dN_dx0;
    double v_1_7 = D0*dN_dx1 + D18*dN_dx5 + D30*dN_dx9;
    double v_1_8 = D21*dN_dx5 + D3*dN_dx1 + D33*dN_dx9;
    double v_1_9 = D23*dN_dx5 + D35*dN_dx9 + D5*dN_dx1;
    double v_1_10 = D1*dN_dx1 + D19*dN_dx5 + D31*dN_dx9;
    double v_1_11 = D22*dN_dx5 + D34*dN_dx9 + D4*dN_dx1;
    double v_1_12 = D2*dN_dx1 + D20*dN_dx5 + D32*dN_dx9;
    out[24] = v_1_3*(dN_dx0*v_1_0 + dN_dx4*v_1_1 + dN_dx8*v_1_2);
    out[25] = v_1_3*(dN_dx0*v_1_1 + dN_dx4*v_1_4 + dN_dx8*v_1_5);
    out[26] = v_1_3*(dN_dx0*v_1_2 + dN_dx4*v_1_5 + dN_dx8*v_1_6);
    out[27] = v_1_3*(dN_dx1*v_1_0 + dN_dx5*v_1_1 + dN_dx9*v_1_2);
    out[28] = v_1_3*(dN_dx1*v_1_1 + dN_dx5*v_1_4 + dN_dx9*v_1_5);
    out[29] = v_1_3*(dN_dx1*v_1_2 + dN_dx5*v_1_5 + dN_dx9*v_1_6);
    out[30] = v_1_3*(dN_dx10*v_1_2 + dN_dx2*v_1_0 + dN_dx6*v_1_1);
    out[31] = v_1_3*(dN_dx10*v_1_5 + dN_dx2*v_1_1 + dN_dx6*v_1_4);
    out[32] = v_1_3*(dN_dx10*v_1_6 + dN_dx2*v_1_2 + dN_dx6*v_1_5);
    out[33] = v_1_3*(dN_dx11*v_1_2 + dN_dx3*v_1_0 + dN_dx7*v_1_1);
    out[34] = v_1_3*(dN_dx11*v_1_5 + dN_dx3*v_1_1 + dN_dx7*v_1_4);
    out[35] = v_1_3*(dN_dx11*v_1_6 + dN_dx3*v_1_2 + dN_dx7*v_1_5);
    out[36] = v_1_3*(dN_dx0*v_1_7 + dN_dx4*v_1_8 + dN_dx8*v_1_9);
    out[37] = v_1_3*(dN_dx0*v_1_8 + dN_dx4*v_1_10 + dN_dx8*v_1_11);
    out[38] = v_1_3*(dN_dx0*v_1_9 + dN_dx4*v_1_11 + dN_dx8*v_1_12);
    out[39] = v_1_3*(dN_dx1*v_1_7 + dN_dx5*v_1_8 + dN_dx9*v_1_9);
    out[40] = v_1_3*(dN_dx1*v_1_8 + dN_dx5*v_1_10 + dN_dx9*v_1_11);
    out[41] = v_1_3*(dN_dx1*v_1_9 + dN_dx5*v_1_11 + dN_dx9*v_1_12);
    out[42] = v_1_3*(dN_dx10*v_1_9 + dN_dx2*v_1_7 + dN_dx6*v_1_8);
    out[43] = v_1_3*(dN_dx10*v_1_11 + dN_dx2*v_1_8 + dN_dx6*v_1_10);
    out[44] = v_1_3*(dN_dx10*v_1_12 + dN_dx2*v_1_9 + dN_dx6*v_1_11);
    out[45] = v_1_3*(dN_dx11*v_1_9 + dN_dx3*v_1_7 + dN_dx7*v_1_8);
    out[46] = v_1_3*(dN_dx11*v_1_11 + dN_dx3*v_1_8 + dN_dx7*v_1_10);
    out[47] = v_1_3*(dN_dx11*v_1_12 + dN_dx3*v_1_9 + dN_dx7*v_1_11);

    // --- Chunk 2 ---
    double v_2_0 = D18*dN_dx1 + D24*dN_dx9 + D6*dN_dx5;
    double v_2_1 = D21*dN_dx1 + D27*dN_dx9 + D9*dN_dx5;
    double v_2_2 = D11*dN_dx5 + D23*dN_dx1 + D29*dN_dx9;
    double v_2_3 = weight*fabs(detJ);
    double v_2_4 = D19*dN_dx1 + D25*dN_dx9 + D7*dN_dx5;
    double v_2_5 = D10*dN_dx5 + D22*dN_dx1 + D28*dN_dx9;
    double v_2_6 = D20*dN_dx1 + D26*dN_dx9 + D8*dN_dx5;
    double v_2_7 = D12*dN_dx9 + D24*dN_dx5 + D30*dN_dx1;
    double v_2_8 = D15*dN_dx9 + D27*dN_dx5 + D33*dN_dx1;
    double v_2_9 = D17*dN_dx9 + D29*dN_dx5 + D35*dN_dx1;
    double v_2_10 = D13*dN_dx9 + D25*dN_dx5 + D31*dN_dx1;
    double v_2_11 = D16*dN_dx9 + D28*dN_dx5 + D34*dN_dx1;
    double v_2_12 = D14*dN_dx9 + D26*dN_dx5 + D32*dN_dx1;
    out[48] = v_2_3*(dN_dx0*v_2_0 + dN_dx4*v_2_1 + dN_dx8*v_2_2);
    out[49] = v_2_3*(dN_dx0*v_2_1 + dN_dx4*v_2_4 + dN_dx8*v_2_5);
    out[50] = v_2_3*(dN_dx0*v_2_2 + dN_dx4*v_2_5 + dN_dx8*v_2_6);
    out[51] = v_2_3*(dN_dx1*v_2_0 + dN_dx5*v_2_1 + dN_dx9*v_2_2);
    out[52] = v_2_3*(dN_dx1*v_2_1 + dN_dx5*v_2_4 + dN_dx9*v_2_5);
    out[53] = v_2_3*(dN_dx1*v_2_2 + dN_dx5*v_2_5 + dN_dx9*v_2_6);
    out[54] = v_2_3*(dN_dx10*v_2_2 + dN_dx2*v_2_0 + dN_dx6*v_2_1);
    out[55] = v_2_3*(dN_dx10*v_2_5 + dN_dx2*v_2_1 + dN_dx6*v_2_4);
    out[56] = v_2_3*(dN_dx10*v_2_6 + dN_dx2*v_2_2 + dN_dx6*v_2_5);
    out[57] = v_2_3*(dN_dx11*v_2_2 + dN_dx3*v_2_0 + dN_dx7*v_2_1);
    out[58] = v_2_3*(dN_dx11*v_2_5 + dN_dx3*v_2_1 + dN_dx7*v_2_4);
    out[59] = v_2_3*(dN_dx11*v_2_6 + dN_dx3*v_2_2 + dN_dx7*v_2_5);
    out[60] = v_2_3*(dN_dx0*v_2_7 + dN_dx4*v_2_8 + dN_dx8*v_2_9);
    out[61] = v_2_3*(dN_dx0*v_2_8 + dN_dx4*v_2_10 + dN_dx8*v_2_11);
    out[62] = v_2_3*(dN_dx0*v_2_9 + dN_dx4*v_2_11 + dN_dx8*v_2_12);
    out[63] = v_2_3*(dN_dx1*v_2_7 + dN_dx5*v_2_8 + dN_dx9*v_2_9);
    out[64] = v_2_3*(dN_dx1*v_2_8 + dN_dx5*v_2_10 + dN_dx9*v_2_11);
    out[65] = v_2_3*(dN_dx1*v_2_9 + dN_dx5*v_2_11 + dN_dx9*v_2_12);
    out[66] = v_2_3*(dN_dx10*v_2_9 + dN_dx2*v_2_7 + dN_dx6*v_2_8);
    out[67] = v_2_3*(dN_dx10*v_2_11 + dN_dx2*v_2_8 + dN_dx6*v_2_10);
    out[68] = v_2_3*(dN_dx10*v_2_12 + dN_dx2*v_2_9 + dN_dx6*v_2_11);
    out[69] = v_2_3*(dN_dx11*v_2_9 + dN_dx3*v_2_7 + dN_dx7*v_2_8);
    out[70] = v_2_3*(dN_dx11*v_2_11 + dN_dx3*v_2_8 + dN_dx7*v_2_10);
    out[71] = v_2_3*(dN_dx11*v_2_12 + dN_dx3*v_2_9 + dN_dx7*v_2_11);

    // --- Chunk 3 ---
    double v_3_0 = D0*dN_dx2 + D18*dN_dx6 + D30*dN_dx10;
    double v_3_1 = D21*dN_dx6 + D3*dN_dx2 + D33*dN_dx10;
    double v_3_2 = D23*dN_dx6 + D35*dN_dx10 + D5*dN_dx2;
    double v_3_3 = weight*fabs(detJ);
    double v_3_4 = D1*dN_dx2 + D19*dN_dx6 + D31*dN_dx10;
    double v_3_5 = D22*dN_dx6 + D34*dN_dx10 + D4*dN_dx2;
    double v_3_6 = D2*dN_dx2 + D20*dN_dx6 + D32*dN_dx10;
    double v_3_7 = D18*dN_dx2 + D24*dN_dx10 + D6*dN_dx6;
    double v_3_8 = D21*dN_dx2 + D27*dN_dx10 + D9*dN_dx6;
    double v_3_9 = D11*dN_dx6 + D23*dN_dx2 + D29*dN_dx10;
    double v_3_10 = D19*dN_dx2 + D25*dN_dx10 + D7*dN_dx6;
    double v_3_11 = D10*dN_dx6 + D22*dN_dx2 + D28*dN_dx10;
    double v_3_12 = D20*dN_dx2 + D26*dN_dx10 + D8*dN_dx6;
    out[72] = v_3_3*(dN_dx0*v_3_0 + dN_dx4*v_3_1 + dN_dx8*v_3_2);
    out[73] = v_3_3*(dN_dx0*v_3_1 + dN_dx4*v_3_4 + dN_dx8*v_3_5);
    out[74] = v_3_3*(dN_dx0*v_3_2 + dN_dx4*v_3_5 + dN_dx8*v_3_6);
    out[75] = v_3_3*(dN_dx1*v_3_0 + dN_dx5*v_3_1 + dN_dx9*v_3_2);
    out[76] = v_3_3*(dN_dx1*v_3_1 + dN_dx5*v_3_4 + dN_dx9*v_3_5);
    out[77] = v_3_3*(dN_dx1*v_3_2 + dN_dx5*v_3_5 + dN_dx9*v_3_6);
    out[78] = v_3_3*(dN_dx10*v_3_2 + dN_dx2*v_3_0 + dN_dx6*v_3_1);
    out[79] = v_3_3*(dN_dx10*v_3_5 + dN_dx2*v_3_1 + dN_dx6*v_3_4);
    out[80] = v_3_3*(dN_dx10*v_3_6 + dN_dx2*v_3_2 + dN_dx6*v_3_5);
    out[81] = v_3_3*(dN_dx11*v_3_2 + dN_dx3*v_3_0 + dN_dx7*v_3_1);
    out[82] = v_3_3*(dN_dx11*v_3_5 + dN_dx3*v_3_1 + dN_dx7*v_3_4);
    out[83] = v_3_3*(dN_dx11*v_3_6 + dN_dx3*v_3_2 + dN_dx7*v_3_5);
    out[84] = v_3_3*(dN_dx0*v_3_7 + dN_dx4*v_3_8 + dN_dx8*v_3_9);
    out[85] = v_3_3*(dN_dx0*v_3_8 + dN_dx4*v_3_10 + dN_dx8*v_3_11);
    out[86] = v_3_3*(dN_dx0*v_3_9 + dN_dx4*v_3_11 + dN_dx8*v_3_12);
    out[87] = v_3_3*(dN_dx1*v_3_7 + dN_dx5*v_3_8 + dN_dx9*v_3_9);
    out[88] = v_3_3*(dN_dx1*v_3_8 + dN_dx5*v_3_10 + dN_dx9*v_3_11);
    out[89] = v_3_3*(dN_dx1*v_3_9 + dN_dx5*v_3_11 + dN_dx9*v_3_12);
    out[90] = v_3_3*(dN_dx10*v_3_9 + dN_dx2*v_3_7 + dN_dx6*v_3_8);
    out[91] = v_3_3*(dN_dx10*v_3_11 + dN_dx2*v_3_8 + dN_dx6*v_3_10);
    out[92] = v_3_3*(dN_dx10*v_3_12 + dN_dx2*v_3_9 + dN_dx6*v_3_11);
    out[93] = v_3_3*(dN_dx11*v_3_9 + dN_dx3*v_3_7 + dN_dx7*v_3_8);
    out[94] = v_3_3*(dN_dx11*v_3_11 + dN_dx3*v_3_8 + dN_dx7*v_3_10);
    out[95] = v_3_3*(dN_dx11*v_3_12 + dN_dx3*v_3_9 + dN_dx7*v_3_11);

    // --- Chunk 4 ---
    double v_4_0 = D12*dN_dx10 + D24*dN_dx6 + D30*dN_dx2;
    double v_4_1 = D15*dN_dx10 + D27*dN_dx6 + D33*dN_dx2;
    double v_4_2 = D17*dN_dx10 + D29*dN_dx6 + D35*dN_dx2;
    double v_4_3 = weight*fabs(detJ);
    double v_4_4 = D13*dN_dx10 + D25*dN_dx6 + D31*dN_dx2;
    double v_4_5 = D16*dN_dx10 + D28*dN_dx6 + D34*dN_dx2;
    double v_4_6 = D14*dN_dx10 + D26*dN_dx6 + D32*dN_dx2;
    double v_4_7 = D0*dN_dx3 + D18*dN_dx7 + D30*dN_dx11;
    double v_4_8 = D21*dN_dx7 + D3*dN_dx3 + D33*dN_dx11;
    double v_4_9 = D23*dN_dx7 + D35*dN_dx11 + D5*dN_dx3;
    double v_4_10 = D1*dN_dx3 + D19*dN_dx7 + D31*dN_dx11;
    double v_4_11 = D22*dN_dx7 + D34*dN_dx11 + D4*dN_dx3;
    double v_4_12 = D2*dN_dx3 + D20*dN_dx7 + D32*dN_dx11;
    out[96] = v_4_3*(dN_dx0*v_4_0 + dN_dx4*v_4_1 + dN_dx8*v_4_2);
    out[97] = v_4_3*(dN_dx0*v_4_1 + dN_dx4*v_4_4 + dN_dx8*v_4_5);
    out[98] = v_4_3*(dN_dx0*v_4_2 + dN_dx4*v_4_5 + dN_dx8*v_4_6);
    out[99] = v_4_3*(dN_dx1*v_4_0 + dN_dx5*v_4_1 + dN_dx9*v_4_2);
    out[100] = v_4_3*(dN_dx1*v_4_1 + dN_dx5*v_4_4 + dN_dx9*v_4_5);
    out[101] = v_4_3*(dN_dx1*v_4_2 + dN_dx5*v_4_5 + dN_dx9*v_4_6);
    out[102] = v_4_3*(dN_dx10*v_4_2 + dN_dx2*v_4_0 + dN_dx6*v_4_1);
    out[103] = v_4_3*(dN_dx10*v_4_5 + dN_dx2*v_4_1 + dN_dx6*v_4_4);
    out[104] = v_4_3*(dN_dx10*v_4_6 + dN_dx2*v_4_2 + dN_dx6*v_4_5);
    out[105] = v_4_3*(dN_dx11*v_4_2 + dN_dx3*v_4_0 + dN_dx7*v_4_1);
    out[106] = v_4_3*(dN_dx11*v_4_5 + dN_dx3*v_4_1 + dN_dx7*v_4_4);
    out[107] = v_4_3*(dN_dx11*v_4_6 + dN_dx3*v_4_2 + dN_dx7*v_4_5);
    out[108] = v_4_3*(dN_dx0*v_4_7 + dN_dx4*v_4_8 + dN_dx8*v_4_9);
    out[109] = v_4_3*(dN_dx0*v_4_8 + dN_dx4*v_4_10 + dN_dx8*v_4_11);
    out[110] = v_4_3*(dN_dx0*v_4_9 + dN_dx4*v_4_11 + dN_dx8*v_4_12);
    out[111] = v_4_3*(dN_dx1*v_4_7 + dN_dx5*v_4_8 + dN_dx9*v_4_9);
    out[112] = v_4_3*(dN_dx1*v_4_8 + dN_dx5*v_4_10 + dN_dx9*v_4_11);
    out[113] = v_4_3*(dN_dx1*v_4_9 + dN_dx5*v_4_11 + dN_dx9*v_4_12);
    out[114] = v_4_3*(dN_dx10*v_4_9 + dN_dx2*v_4_7 + dN_dx6*v_4_8);
    out[115] = v_4_3*(dN_dx10*v_4_11 + dN_dx2*v_4_8 + dN_dx6*v_4_10);
    out[116] = v_4_3*(dN_dx10*v_4_12 + dN_dx2*v_4_9 + dN_dx6*v_4_11);
    out[117] = v_4_3*(dN_dx11*v_4_9 + dN_dx3*v_4_7 + dN_dx7*v_4_8);
    out[118] = v_4_3*(dN_dx11*v_4_11 + dN_dx3*v_4_8 + dN_dx7*v_4_10);
    out[119] = v_4_3*(dN_dx11*v_4_12 + dN_dx3*v_4_9 + dN_dx7*v_4_11);

    // --- Chunk 5 ---
    double v_5_0 = D18*dN_dx3 + D24*dN_dx11 + D6*dN_dx7;
    double v_5_1 = D21*dN_dx3 + D27*dN_dx11 + D9*dN_dx7;
    double v_5_2 = D11*dN_dx7 + D23*dN_dx3 + D29*dN_dx11;
    double v_5_3 = weight*fabs(detJ);
    double v_5_4 = D19*dN_dx3 + D25*dN_dx11 + D7*dN_dx7;
    double v_5_5 = D10*dN_dx7 + D22*dN_dx3 + D28*dN_dx11;
    double v_5_6 = D20*dN_dx3 + D26*dN_dx11 + D8*dN_dx7;
    double v_5_7 = D12*dN_dx11 + D24*dN_dx7 + D30*dN_dx3;
    double v_5_8 = D15*dN_dx11 + D27*dN_dx7 + D33*dN_dx3;
    double v_5_9 = D17*dN_dx11 + D29*dN_dx7 + D35*dN_dx3;
    double v_5_10 = D13*dN_dx11 + D25*dN_dx7 + D31*dN_dx3;
    double v_5_11 = D16*dN_dx11 + D28*dN_dx7 + D34*dN_dx3;
    double v_5_12 = D14*dN_dx11 + D26*dN_dx7 + D32*dN_dx3;
    out[120] = v_5_3*(dN_dx0*v_5_0 + dN_dx4*v_5_1 + dN_dx8*v_5_2);
    out[121] = v_5_3*(dN_dx0*v_5_1 + dN_dx4*v_5_4 + dN_dx8*v_5_5);
    out[122] = v_5_3*(dN_dx0*v_5_2 + dN_dx4*v_5_5 + dN_dx8*v_5_6);
    out[123] = v_5_3*(dN_dx1*v_5_0 + dN_dx5*v_5_1 + dN_dx9*v_5_2);
    out[124] = v_5_3*(dN_dx1*v_5_1 + dN_dx5*v_5_4 + dN_dx9*v_5_5);
    out[125] = v_5_3*(dN_dx1*v_5_2 + dN_dx5*v_5_5 + dN_dx9*v_5_6);
    out[126] = v_5_3*(dN_dx10*v_5_2 + dN_dx2*v_5_0 + dN_dx6*v_5_1);
    out[127] = v_5_3*(dN_dx10*v_5_5 + dN_dx2*v_5_1 + dN_dx6*v_5_4);
    out[128] = v_5_3*(dN_dx10*v_5_6 + dN_dx2*v_5_2 + dN_dx6*v_5_5);
    out[129] = v_5_3*(dN_dx11*v_5_2 + dN_dx3*v_5_0 + dN_dx7*v_5_1);
    out[130] = v_5_3*(dN_dx11*v_5_5 + dN_dx3*v_5_1 + dN_dx7*v_5_4);
    out[131] = v_5_3*(dN_dx11*v_5_6 + dN_dx3*v_5_2 + dN_dx7*v_5_5);
    out[132] = v_5_3*(dN_dx0*v_5_7 + dN_dx4*v_5_8 + dN_dx8*v_5_9);
    out[133] = v_5_3*(dN_dx0*v_5_8 + dN_dx4*v_5_10 + dN_dx8*v_5_11);
    out[134] = v_5_3*(dN_dx0*v_5_9 + dN_dx4*v_5_11 + dN_dx8*v_5_12);
    out[135] = v_5_3*(dN_dx1*v_5_7 + dN_dx5*v_5_8 + dN_dx9*v_5_9);
    out[136] = v_5_3*(dN_dx1*v_5_8 + dN_dx5*v_5_10 + dN_dx9*v_5_11);
    out[137] = v_5_3*(dN_dx1*v_5_9 + dN_dx5*v_5_11 + dN_dx9*v_5_12);
    out[138] = v_5_3*(dN_dx10*v_5_9 + dN_dx2*v_5_7 + dN_dx6*v_5_8);
    out[139] = v_5_3*(dN_dx10*v_5_11 + dN_dx2*v_5_8 + dN_dx6*v_5_10);
    out[140] = v_5_3*(dN_dx10*v_5_12 + dN_dx2*v_5_9 + dN_dx6*v_5_11);
    out[141] = v_5_3*(dN_dx11*v_5_9 + dN_dx3*v_5_7 + dN_dx7*v_5_8);
    out[142] = v_5_3*(dN_dx11*v_5_11 + dN_dx3*v_5_8 + dN_dx7*v_5_10);
    out[143] = v_5_3*(dN_dx11*v_5_12 + dN_dx3*v_5_9 + dN_dx7*v_5_11);
}