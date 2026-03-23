/**
 * @brief Computes the tet4_op_mapping kernel.
 * @note This is an optimized operator kernel.
 * 
 * @param in Input array (const double*). Layout:
 *   - in[0]: coord[0]
 *   - in[1]: coord[1]
 *   - in[2]: coord[2]
 *   - in[3]: coord[3]
 *   - in[4]: coord[4]
 *   - in[5]: coord[5]
 *   - in[6]: coord[6]
 *   - in[7]: coord[7]
 *   - in[8]: coord[8]
 *   - in[9]: coord[9]
 *   - in[10]: coord[10]
 *   - in[11]: coord[11]
 *   - in[12]: dN_dnat[0]
 *   - in[13]: dN_dnat[1]
 *   - in[14]: dN_dnat[2]
 *   - in[15]: dN_dnat[3]
 *   - in[16]: dN_dnat[4]
 *   - in[17]: dN_dnat[5]
 *   - in[18]: dN_dnat[6]
 *   - in[19]: dN_dnat[7]
 *   - in[20]: dN_dnat[8]
 *   - in[21]: dN_dnat[9]
 *   - in[22]: dN_dnat[10]
 *   - in[23]: dN_dnat[11]
 * 
 * @param out Output array (double*). Layout:
 *   - out[0..12]: Flattened result, row-major order.
 */
[[gnu::always_inline]] inline void compute_tet4_op_mapping(const double* __restrict__ in, double* __restrict__ out) { 
    const double& c0 = in[0];
    const double& c1 = in[1];
    const double& c2 = in[2];
    const double& c3 = in[3];
    const double& c4 = in[4];
    const double& c5 = in[5];
    const double& c6 = in[6];
    const double& c7 = in[7];
    const double& c8 = in[8];
    const double& c9 = in[9];
    const double& c10 = in[10];
    const double& c11 = in[11];

    const double& dN_dnat0 = in[12];
    const double& dN_dnat1 = in[13];
    const double& dN_dnat2 = in[14];
    const double& dN_dnat3 = in[15];
    const double& dN_dnat4 = in[16];
    const double& dN_dnat5 = in[17];
    const double& dN_dnat6 = in[18];
    const double& dN_dnat7 = in[19];
    const double& dN_dnat8 = in[20];
    const double& dN_dnat9 = in[21];
    const double& dN_dnat10 = in[22];
    const double& dN_dnat11 = in[23];

    // --- Chunk 0 ---
    double v_0_0 = c5*dN_dnat5;
    double v_0_1 = c10*dN_dnat10;
    double v_0_2 = v_0_0*v_0_1;
    double v_0_3 = c0*dN_dnat0;
    double v_0_4 = c10*dN_dnat11;
    double v_0_5 = c5*dN_dnat3;
    double v_0_6 = v_0_4*v_0_5;
    double v_0_7 = c0*dN_dnat1;
    double v_0_8 = c5*dN_dnat4;
    double v_0_9 = c10*dN_dnat9;
    double v_0_10 = v_0_8*v_0_9;
    double v_0_11 = c0*dN_dnat2;
    double v_0_12 = c8*dN_dnat8;
    double v_0_13 = v_0_1*v_0_12;
    double v_0_14 = c8*dN_dnat6;
    double v_0_15 = v_0_14*v_0_4;
    double v_0_16 = c8*dN_dnat7;
    double v_0_17 = v_0_16*v_0_9;
    double v_0_18 = c11*dN_dnat11;
    double v_0_19 = c4*dN_dnat4;
    double v_0_20 = v_0_18*v_0_19;
    double v_0_21 = c4*dN_dnat5;
    double v_0_22 = c11*dN_dnat9;
    double v_0_23 = v_0_21*v_0_22;
    double v_0_24 = c11*dN_dnat10;
    double v_0_25 = c4*dN_dnat3;
    double v_0_26 = v_0_24*v_0_25;
    double v_0_27 = c7*dN_dnat7;
    double v_0_28 = v_0_18*v_0_27;
    double v_0_29 = c7*dN_dnat8;
    double v_0_30 = v_0_22*v_0_29;
    double v_0_31 = c7*dN_dnat6;
    double v_0_32 = v_0_24*v_0_31;
    double v_0_33 = v_0_12*v_0_19;
    double v_0_34 = v_0_14*v_0_21;
    double v_0_35 = v_0_16*v_0_25;
    double v_0_36 = v_0_0*v_0_27;
    double v_0_37 = v_0_29*v_0_5;
    double v_0_38 = v_0_31*v_0_8;
    double v_0_39 = c1*dN_dnat0;
    double v_0_40 = v_0_24*v_0_39;
    double v_0_41 = c3*dN_dnat5;
    double v_0_42 = c1*dN_dnat1;
    double v_0_43 = v_0_18*v_0_42;
    double v_0_44 = c3*dN_dnat3;
    double v_0_45 = c1*dN_dnat2;
    double v_0_46 = v_0_22*v_0_45;
    double v_0_47 = c3*dN_dnat4;
    double v_0_48 = c6*dN_dnat8;
    double v_0_49 = c6*dN_dnat6;
    double v_0_50 = c6*dN_dnat7;
    double v_0_51 = v_0_16*v_0_39;
    double v_0_52 = v_0_12*v_0_42;
    double v_0_53 = v_0_14*v_0_45;
    double v_0_54 = v_0_39*v_0_8;
    double v_0_55 = v_0_0*v_0_42;
    double v_0_56 = v_0_45*v_0_5;
    double v_0_57 = c9*dN_dnat11;
    double v_0_58 = c9*dN_dnat9;
    double v_0_59 = c9*dN_dnat10;
    double v_0_60 = c2*dN_dnat0;
    double v_0_61 = v_0_4*v_0_60;
    double v_0_62 = c2*dN_dnat1;
    double v_0_63 = v_0_62*v_0_9;
    double v_0_64 = c2*dN_dnat2;
    double v_0_65 = v_0_1*v_0_64;
    double v_0_66 = v_0_29*v_0_60;
    double v_0_67 = v_0_31*v_0_62;
    double v_0_68 = v_0_27*v_0_64;
    double v_0_69 = v_0_21*v_0_60;
    double v_0_70 = v_0_25*v_0_62;
    double v_0_71 = v_0_19*v_0_64;
    double v_0_72 = v_0_4*v_0_8;
    double v_0_73 = v_0_0*v_0_9;
    double v_0_74 = v_0_1*v_0_5;
    double v_0_75 = v_0_16*v_0_4;
    double v_0_76 = v_0_12*v_0_9;
    double v_0_77 = v_0_1*v_0_14;
    double v_0_78 = v_0_21*v_0_24;
    double v_0_79 = v_0_18*v_0_25;
    double v_0_80 = v_0_19*v_0_22;
    double v_0_81 = v_0_24*v_0_29;
    double v_0_82 = v_0_18*v_0_31;
    double v_0_83 = v_0_22*v_0_27;
    double v_0_84 = v_0_16*v_0_21;
    double v_0_85 = v_0_12*v_0_25;
    double v_0_86 = v_0_14*v_0_19;
    double v_0_87 = v_0_29*v_0_8;
    double v_0_88 = v_0_0*v_0_31;
    double v_0_89 = v_0_27*v_0_5;
    double v_0_90 = v_0_18*v_0_39;
    double v_0_91 = v_0_22*v_0_42;
    double v_0_92 = v_0_24*v_0_45;
    double v_0_93 = v_0_12*v_0_39;
    double v_0_94 = v_0_14*v_0_42;
    double v_0_95 = v_0_16*v_0_45;
    double v_0_96 = v_0_0*v_0_39;
    double v_0_97 = v_0_42*v_0_5;
    double v_0_98 = v_0_45*v_0_8;
    double v_0_99 = v_0_1*v_0_60;
    double v_0_100 = v_0_4*v_0_62;
    double v_0_101 = v_0_64*v_0_9;
    double v_0_102 = v_0_27*v_0_60;
    double v_0_103 = v_0_29*v_0_62;
    double v_0_104 = v_0_31*v_0_64;
    double v_0_105 = v_0_19*v_0_60;
    double v_0_106 = v_0_21*v_0_62;
    double v_0_107 = v_0_25*v_0_64;
    double v_0_108 = v_0_10*v_0_11 + v_0_10*v_0_48 - v_0_100*v_0_44 - v_0_100*v_0_49 - v_0_101*v_0_47 - v_0_101*v_0_50 - v_0_102*v_0_41 - v_0_102*v_0_57 - v_0_103*v_0_44 - v_0_103*v_0_58 - v_0_104*v_0_47 - v_0_104*v_0_59 - v_0_105*v_0_48 - v_0_105*v_0_57 - v_0_106*v_0_49 - v_0_106*v_0_58 - v_0_107*v_0_50 - v_0_107*v_0_59 + v_0_11*v_0_17 + v_0_11*v_0_26 + v_0_11*v_0_32 + v_0_11*v_0_35 + v_0_11*v_0_38 - v_0_11*v_0_74 - v_0_11*v_0_77 - v_0_11*v_0_80 - v_0_11*v_0_83 - v_0_11*v_0_86 - v_0_11*v_0_89 + v_0_13*v_0_3 + v_0_13*v_0_44 + v_0_15*v_0_47 + v_0_15*v_0_7 + v_0_17*v_0_41 + v_0_2*v_0_3 + v_0_2*v_0_49 + v_0_20*v_0_3 + v_0_20*v_0_49 + v_0_23*v_0_50 + v_0_23*v_0_7 + v_0_26*v_0_48 + v_0_28*v_0_3 + v_0_28*v_0_44 + v_0_3*v_0_33 + v_0_3*v_0_36 - v_0_3*v_0_72 - v_0_3*v_0_75 - v_0_3*v_0_78 - v_0_3*v_0_81 - v_0_3*v_0_84 - v_0_3*v_0_87 + v_0_30*v_0_47 + v_0_30*v_0_7 + v_0_32*v_0_41 + v_0_33*v_0_58 + v_0_34*v_0_59 + v_0_34*v_0_7 + v_0_35*v_0_57 + v_0_36*v_0_58 + v_0_37*v_0_59 + v_0_37*v_0_7 + v_0_38*v_0_57 + v_0_40*v_0_41 + v_0_40*v_0_48 + v_0_41*v_0_51 + v_0_41*v_0_63 + v_0_41*v_0_67 - v_0_41*v_0_77 - v_0_41*v_0_83 - v_0_41*v_0_91 - v_0_41*v_0_94 - v_0_41*v_0_99 + v_0_43*v_0_44 + v_0_43*v_0_49 + v_0_44*v_0_52 + v_0_44*v_0_65 + v_0_44*v_0_68 - v_0_44*v_0_75 - v_0_44*v_0_81 - v_0_44*v_0_92 - v_0_44*v_0_95 + v_0_46*v_0_47 + v_0_46*v_0_50 + v_0_47*v_0_53 + v_0_47*v_0_61 + v_0_47*v_0_66 - v_0_47*v_0_76 - v_0_47*v_0_82 - v_0_47*v_0_90 - v_0_47*v_0_93 + v_0_48*v_0_54 + v_0_48*v_0_63 + v_0_48*v_0_70 - v_0_48*v_0_74 - v_0_48*v_0_80 - v_0_48*v_0_91 - v_0_48*v_0_97 - v_0_48*v_0_99 + v_0_49*v_0_55 + v_0_49*v_0_65 + v_0_49*v_0_71 - v_0_49*v_0_72 - v_0_49*v_0_78 - v_0_49*v_0_92 - v_0_49*v_0_98 + v_0_50*v_0_56 + v_0_50*v_0_6 + v_0_50*v_0_61 + v_0_50*v_0_69 - v_0_50*v_0_73 - v_0_50*v_0_79 - v_0_50*v_0_90 - v_0_50*v_0_96 + v_0_51*v_0_57 + v_0_52*v_0_58 + v_0_53*v_0_59 + v_0_54*v_0_57 + v_0_55*v_0_58 + v_0_56*v_0_59 + v_0_57*v_0_67 + v_0_57*v_0_70 - v_0_57*v_0_86 - v_0_57*v_0_89 - v_0_57*v_0_94 - v_0_57*v_0_97 + v_0_58*v_0_68 + v_0_58*v_0_71 - v_0_58*v_0_84 - v_0_58*v_0_87 - v_0_58*v_0_95 - v_0_58*v_0_98 + v_0_59*v_0_66 + v_0_59*v_0_69 - v_0_59*v_0_85 - v_0_59*v_0_88 - v_0_59*v_0_93 - v_0_59*v_0_96 + v_0_6*v_0_7 - v_0_7*v_0_73 - v_0_7*v_0_76 - v_0_7*v_0_79 - v_0_7*v_0_82 - v_0_7*v_0_85 - v_0_7*v_0_88;
    double v_0_109 = 1.0/v_0_108;
    double v_0_110 = v_0_109*(-v_0_100 - v_0_103 - v_0_106 + v_0_13 + v_0_2 + v_0_20 + v_0_28 + v_0_33 + v_0_36 + v_0_43 + v_0_52 + v_0_55 + v_0_65 + v_0_68 + v_0_71 - v_0_72 - v_0_75 - v_0_78 - v_0_81 - v_0_84 - v_0_87 - v_0_92 - v_0_95 - v_0_98);
    double v_0_111 = v_0_109*(c1*c11*dN_dnat2*dN_dnat9 + c1*c5*dN_dnat2*dN_dnat3 + c1*c8*dN_dnat2*dN_dnat6 + c10*c2*dN_dnat0*dN_dnat11 + c10*c5*dN_dnat11*dN_dnat3 + c10*c8*dN_dnat11*dN_dnat6 + c11*c4*dN_dnat5*dN_dnat9 + c11*c7*dN_dnat8*dN_dnat9 + c2*c4*dN_dnat0*dN_dnat5 + c2*c7*dN_dnat0*dN_dnat8 + c4*c8*dN_dnat5*dN_dnat6 + c5*c7*dN_dnat3*dN_dnat8 - v_0_101 - v_0_104 - v_0_107 - v_0_73 - v_0_76 - v_0_79 - v_0_82 - v_0_85 - v_0_88 - v_0_90 - v_0_93 - v_0_96);
    double v_0_112 = v_0_109*(v_0_10 - v_0_102 - v_0_105 + v_0_17 + v_0_26 + v_0_32 + v_0_35 + v_0_38 + v_0_40 + v_0_51 + v_0_54 + v_0_63 + v_0_67 + v_0_70 - v_0_74 - v_0_77 - v_0_80 - v_0_83 - v_0_86 - v_0_89 - v_0_91 - v_0_94 - v_0_97 - v_0_99);
    double v_0_113 = v_0_109*(c0*c11*dN_dnat10*dN_dnat2 + c0*c5*dN_dnat2*dN_dnat4 + c0*c8*dN_dnat2*dN_dnat7 + c11*c3*dN_dnat10*dN_dnat5 + c11*c6*dN_dnat10*dN_dnat8 + c2*c3*dN_dnat1*dN_dnat5 + c2*c6*dN_dnat1*dN_dnat8 + c2*c9*dN_dnat1*dN_dnat11 + c3*c8*dN_dnat5*dN_dnat7 + c5*c6*dN_dnat4*dN_dnat8 + c5*c9*dN_dnat11*dN_dnat4 + c8*c9*dN_dnat11*dN_dnat7 - v_0_0*v_0_50 - v_0_0*v_0_59 - v_0_0*v_0_7 - v_0_12*v_0_47 - v_0_12*v_0_59 - v_0_12*v_0_7 - v_0_18*v_0_47 - v_0_18*v_0_50 - v_0_18*v_0_7 - v_0_47*v_0_64 - v_0_50*v_0_64 - v_0_59*v_0_64);
    double v_0_114 = v_0_109*(v_0_0*v_0_3 + v_0_0*v_0_49 + v_0_0*v_0_58 - v_0_11*v_0_14 - v_0_11*v_0_22 - v_0_11*v_0_5 + v_0_12*v_0_3 + v_0_12*v_0_44 + v_0_12*v_0_58 - v_0_14*v_0_41 - v_0_14*v_0_57 + v_0_18*v_0_3 + v_0_18*v_0_44 + v_0_18*v_0_49 - v_0_22*v_0_41 - v_0_22*v_0_48 - v_0_41*v_0_60 + v_0_44*v_0_64 - v_0_48*v_0_5 - v_0_48*v_0_60 + v_0_49*v_0_64 - v_0_5*v_0_57 - v_0_57*v_0_60 + v_0_58*v_0_64);
    double v_0_115 = v_0_109*(c0*c11*dN_dnat1*dN_dnat9 + c0*c5*dN_dnat1*dN_dnat3 + c0*c8*dN_dnat1*dN_dnat6 + c11*c3*dN_dnat4*dN_dnat9 + c11*c6*dN_dnat7*dN_dnat9 + c2*c3*dN_dnat0*dN_dnat4 + c2*c6*dN_dnat0*dN_dnat7 + c2*c9*dN_dnat0*dN_dnat10 + c3*c8*dN_dnat4*dN_dnat6 + c5*c6*dN_dnat3*dN_dnat7 + c5*c9*dN_dnat10*dN_dnat3 + c8*c9*dN_dnat10*dN_dnat6 - v_0_16*v_0_3 - v_0_16*v_0_44 - v_0_16*v_0_58 - v_0_24*v_0_3 - v_0_24*v_0_44 - v_0_24*v_0_49 - v_0_3*v_0_8 - v_0_44*v_0_62 - v_0_49*v_0_62 - v_0_49*v_0_8 - v_0_58*v_0_62 - v_0_58*v_0_8);
    double v_0_116 = v_0_109*(-v_0_1*v_0_11 - v_0_1*v_0_41 - v_0_1*v_0_48 - v_0_11*v_0_19 - v_0_11*v_0_27 - v_0_19*v_0_48 - v_0_19*v_0_57 + v_0_21*v_0_50 + v_0_21*v_0_59 + v_0_21*v_0_7 - v_0_27*v_0_41 - v_0_27*v_0_57 + v_0_29*v_0_47 + v_0_29*v_0_59 + v_0_29*v_0_7 + v_0_4*v_0_47 + v_0_4*v_0_50 + v_0_4*v_0_7 - v_0_41*v_0_42 - v_0_42*v_0_48 - v_0_42*v_0_57 + v_0_45*v_0_47 + v_0_45*v_0_50 + v_0_45*v_0_59);
    double v_0_117 = v_0_109*(c0*c10*dN_dnat2*dN_dnat9 + c0*c4*dN_dnat2*dN_dnat3 + c0*c7*dN_dnat2*dN_dnat6 + c1*c3*dN_dnat0*dN_dnat5 + c1*c6*dN_dnat0*dN_dnat8 + c1*c9*dN_dnat0*dN_dnat11 + c10*c3*dN_dnat5*dN_dnat9 + c10*c6*dN_dnat8*dN_dnat9 + c3*c7*dN_dnat5*dN_dnat6 + c4*c6*dN_dnat3*dN_dnat8 + c4*c9*dN_dnat11*dN_dnat3 + c7*c9*dN_dnat11*dN_dnat6 - v_0_21*v_0_3 - v_0_21*v_0_49 - v_0_21*v_0_58 - v_0_29*v_0_3 - v_0_29*v_0_44 - v_0_29*v_0_58 - v_0_3*v_0_4 - v_0_4*v_0_44 - v_0_4*v_0_49 - v_0_44*v_0_45 - v_0_45*v_0_49 - v_0_45*v_0_58);
    double v_0_118 = v_0_109*(v_0_1*v_0_3 + v_0_1*v_0_44 + v_0_1*v_0_49 + v_0_19*v_0_3 + v_0_19*v_0_49 + v_0_19*v_0_58 - v_0_25*v_0_50 - v_0_25*v_0_59 - v_0_25*v_0_7 + v_0_27*v_0_3 + v_0_27*v_0_44 + v_0_27*v_0_58 - v_0_31*v_0_47 - v_0_31*v_0_59 - v_0_31*v_0_7 - v_0_39*v_0_47 - v_0_39*v_0_50 - v_0_39*v_0_59 + v_0_42*v_0_44 + v_0_42*v_0_49 + v_0_42*v_0_58 - v_0_47*v_0_9 - v_0_50*v_0_9 - v_0_7*v_0_9);
    out[0] = dN_dnat0*v_0_110 + dN_dnat1*v_0_111 + dN_dnat2*v_0_112;
    out[1] = dN_dnat3*v_0_110 + dN_dnat4*v_0_111 + dN_dnat5*v_0_112;
    out[2] = dN_dnat6*v_0_110 + dN_dnat7*v_0_111 + dN_dnat8*v_0_112;
    out[3] = dN_dnat10*v_0_111 + dN_dnat11*v_0_112 + dN_dnat9*v_0_110;
    out[4] = dN_dnat0*v_0_113 + dN_dnat1*v_0_114 + dN_dnat2*v_0_115;
    out[5] = dN_dnat3*v_0_113 + dN_dnat4*v_0_114 + dN_dnat5*v_0_115;
    out[6] = dN_dnat6*v_0_113 + dN_dnat7*v_0_114 + dN_dnat8*v_0_115;
    out[7] = dN_dnat10*v_0_114 + dN_dnat11*v_0_115 + dN_dnat9*v_0_113;
    out[8] = dN_dnat0*v_0_116 + dN_dnat1*v_0_117 + dN_dnat2*v_0_118;
    out[9] = dN_dnat3*v_0_116 + dN_dnat4*v_0_117 + dN_dnat5*v_0_118;
    out[10] = dN_dnat6*v_0_116 + dN_dnat7*v_0_117 + dN_dnat8*v_0_118;
    out[11] = dN_dnat10*v_0_117 + dN_dnat11*v_0_118 + dN_dnat9*v_0_116;
    out[12] = v_0_108;
}