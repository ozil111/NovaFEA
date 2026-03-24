// Tet4StiffnessMatrix.cpp
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */

#include "Tet4StiffnessMatrix.h"
#include "components/mesh_components.h"
#include "spdlog/spdlog.h"
#include <cmath>

void compute_tet4_stiffness_matrix(
    entt::registry& registry,
    entt::entity element_entity,
    const Eigen::Matrix<double, 6, 6>& D,
    Eigen::MatrixXd& Ke_out
) {
    // Validate connectivity
    if (!registry.all_of<Component::Connectivity>(element_entity)) {
        throw std::runtime_error("Tet4 stiffness: element missing Connectivity");
    }
    const auto& conn = registry.get<Component::Connectivity>(element_entity);
    if (conn.nodes.size() != 4) {
        throw std::runtime_error("Tet4 stiffness: Connectivity is not 4 nodes");
    }

    // Gather coordinates
    double coords[12];
    for (int a = 0; a < 4; ++a) {
        const entt::entity n = conn.nodes[static_cast<size_t>(a)];
        if (!registry.all_of<Component::Position>(n)) {
            throw std::runtime_error("Tet4 stiffness: node missing Position");
        }
        const auto& p = registry.get<Component::Position>(n);
        const int base = 3 * a;
        coords[base + 0] = p.x;
        coords[base + 1] = p.y;
        coords[base + 2] = p.z;
    }

    double Ke_raw[12 * 12];
    const double volume = compute_tet4_stiffness(coords, D.data(), Ke_raw);

    if (!(volume > 0.0) || !std::isfinite(volume)) {
        throw std::runtime_error("Tet4 stiffness: invalid volume");
    }

    Ke_out.resize(12, 12);
    Ke_out = Eigen::Map<const Eigen::Matrix<double, 12, 12, Eigen::RowMajor>>(Ke_raw);
 }

double compute_tet4_stiffness(const double* coords, const double* D, double* Ke) {
    // --- 1. 读取 4 个节点坐标（[x0,y0,z0, x1,y1,z1, ...] 展开)---
    double x0 = coords[0];  double y0 = coords[1];  double z0 = coords[2];   // 节点 1
    double x1 = coords[3];  double y1 = coords[4];  double z1 = coords[5];   // 节点 2
    double x2 = coords[6];  double y2 = coords[7];  double z2 = coords[8];   // 节点 3
    double x3 = coords[9];  double y3 = coords[10]; double z3 = coords[11];  // 节点 4

    // --- 2. 6x6 本构矩阵 D 拆成标量（D 按列主序存储，D(i,j) = D[j*6+i])---
    double D_0_0 = D[0];  double D_0_1 = D[6];  double D_0_2 = D[12]; double D_0_3 = D[18]; double D_0_4 = D[24]; double D_0_5 = D[30];
    double D_1_0 = D[1];  double D_1_1 = D[7];  double D_1_2 = D[13]; double D_1_3 = D[19]; double D_1_4 = D[25]; double D_1_5 = D[31];
    double D_2_0 = D[2];  double D_2_1 = D[8];  double D_2_2 = D[14]; double D_2_3 = D[20]; double D_2_4 = D[26]; double D_2_5 = D[32];
    double D_3_0 = D[3];  double D_3_1 = D[9];  double D_3_2 = D[15]; double D_3_3 = D[21]; double D_3_4 = D[27]; double D_3_5 = D[33];
    double D_4_0 = D[4];  double D_4_1 = D[10]; double D_4_2 = D[16]; double D_4_3 = D[22]; double D_4_4 = D[28]; double D_4_5 = D[34];
    double D_5_0 = D[5];  double D_5_1 = D[11]; double D_5_2 = D[17]; double D_5_3 = D[23]; double D_5_4 = D[29]; double D_5_5 = D[35];

    // --- 3. 预计算若干与坐标相关的中间量（用于雅可比与形函数梯度)---
    double x4 = y1*z2;
    double x5 = x0*x4;
    double x6 = y2*z3;
    double x7 = x0*x6;
    double x8 = y3*z1;
    double x9 = x0*x8;
    double x10 = y0*z3;
    double x11 = x1*x10;
    double x12 = y2*z0;
    double x13 = x1*x12;
    double x14 = y3*z2;
    double x15 = x1*x14;
    double x16 = y0*z1;
    double x17 = x16*x2;
    double x18 = y1*z3;
    double x19 = x18*x2;
    double x20 = y3*z0;
    double x21 = x2*x20;
    double x22 = y0*z2;
    double x23 = x22*x3;
    double x24 = y1*z0;
    double x25 = x24*x3;
    double x26 = y2*z1;
    double x27 = x26*x3;
    // x28 = detJ / 6.0 = 四面体体积（带符号）
    double x28 = (1.0/6.0)*x0*y1*z3 + (1.0/6.0)*x0*y2*z1 + (1.0/6.0)*x0*y3*z2 + (1.0/6.0)*x1*y0*z2 + (1.0/6.0)*x1*y2*z3 + (1.0/6.0)*x1*y3*z0 - 1.0/6.0*x11 - 1.0/6.0*x13 - 1.0/6.0*x15 - 1.0/6.0*x17 - 1.0/6.0*x19 + (1.0/6.0)*x2*y0*z3 + (1.0/6.0)*x2*y1*z0 + (1.0/6.0)*x2*y3*z1 - 1.0/6.0*x21 - 1.0/6.0*x23 - 1.0/6.0*x25 - 1.0/6.0*x27 + (1.0/6.0)*x3*y0*z1 + (1.0/6.0)*x3*y1*z2 + (1.0/6.0)*x3*y2*z0 - 1.0/6.0*x5 - 1.0/6.0*x7 - 1.0/6.0*x9;
    double x29 = x16 - x24;
    // x30 是1/detJ，后续乘上不同组合得到4 个节点的形函数梯
    double x30 = 1.0/(-x0*x14 - x0*x18 - x0*x26 - x1*x20 - x1*x22 - x1*x6 - x10*x2 + x11 - x12*x3 + x13 + x15 - x16*x3 + x17 + x19 - x2*x24 - x2*x8 + x21 + x23 + x25 + x27 - x3*x4 + x5 + x7 + x9);
    double x31 = x30*(-x12 + x22 - x29 - x4 + y2*z1);
    double x32 = -x10 + x20;
    double x33 = x30*(x18 + x29 + x32 - x8);
    double x34 = x30*(x12 - x22 - x32 - x6 + y3*z2);
    double x35 = -x31 - x33 - x34;
    double x36 = x0*z1 - x1*z0;
    double x37 = -x0*z3 + x3*z0;
    double x38 = x30*(-x1*z3 + x3*z1 - x36 - x37);
    double x39 = x2*z0;
    double x40 = x0*z2;
    double x41 = x30*(x1*z2 - x2*z1 + x36 + x39 - x40);
    double x42 = x30*(x2*z3 - x3*z2 + x37 - x39 + x40);
    // (x35, x43, x51) 对应于第 1 个节点的形函数梯度（x,y,z 上的分量）
    double x43 = -x38 - x41 - x42;
    double x44 = x2*y0;
    double x45 = x0*y2;
    double x46 = x0*y1 - x1*y0;
    double x47 = x30*(-x1*y2 + x2*y1 - x44 + x45 - x46);
    double x48 = -x0*y3 + x3*y0;
    double x49 = x30*(x1*y3 - x3*y1 + x46 + x48);
    double x50 = x30*(-x2*y3 + x3*y2 + x44 - x45 - x48);
    double x51 = -x47 - x49 - x50;

    // --- 4. 将形函数梯度与本构矩阵D 组合，得到等价的 B^T D（展开后的 3x3 块形式） ---
    double x52 = D_0_0*x35 + D_3_0*x43 + D_5_0*x51;
    double x53 = D_0_3*x35 + D_3_3*x43 + D_5_3*x51;
    double x54 = D_0_5*x35 + D_3_5*x43 + D_5_5*x51;
    double x55 = D_0_1*x35 + D_3_1*x43 + D_5_1*x51;
    double x56 = D_0_4*x35 + D_3_4*x43 + D_5_4*x51;
    double x57 = D_0_2*x35 + D_3_2*x43 + D_5_2*x51;
    double x58 = D_1_0*x43 + D_3_0*x35 + D_4_0*x51;
    double x59 = D_1_3*x43 + D_3_3*x35 + D_4_3*x51;
    double x60 = D_1_5*x43 + D_3_5*x35 + D_4_5*x51;
    double x61 = D_1_1*x43 + D_3_1*x35 + D_4_1*x51;
    double x62 = D_1_4*x43 + D_3_4*x35 + D_4_4*x51;
    double x63 = D_1_2*x43 + D_3_2*x35 + D_4_2*x51;
    double x64 = D_2_0*x51 + D_4_0*x43 + D_5_0*x35;
    double x65 = D_2_3*x51 + D_4_3*x43 + D_5_3*x35;
    double x66 = D_2_5*x51 + D_4_5*x43 + D_5_5*x35;
    double x67 = D_2_1*x51 + D_4_1*x43 + D_5_1*x35;
    double x68 = D_2_4*x51 + D_4_4*x43 + D_5_4*x35;
    double x69 = D_2_2*x51 + D_4_2*x43 + D_5_2*x35;
    double x70 = D_0_5*x34 + D_3_5*x42 + D_5_5*x50;
    double x71 = D_0_3*x34 + D_3_3*x42 + D_5_3*x50;
    double x72 = D_0_0*x34 + D_3_0*x42 + D_5_0*x50;
    double x73 = D_0_4*x34 + D_3_4*x42 + D_5_4*x50;
    double x74 = D_0_1*x34 + D_3_1*x42 + D_5_1*x50;
    double x75 = D_0_2*x34 + D_3_2*x42 + D_5_2*x50;
    double x76 = D_1_5*x42 + D_3_5*x34 + D_4_5*x50;
    double x77 = D_1_3*x42 + D_3_3*x34 + D_4_3*x50;
    double x78 = D_1_0*x42 + D_3_0*x34 + D_4_0*x50;
    double x79 = D_1_4*x42 + D_3_4*x34 + D_4_4*x50;
    double x80 = D_1_1*x42 + D_3_1*x34 + D_4_1*x50;
    double x81 = D_1_2*x42 + D_3_2*x34 + D_4_2*x50;
    double x82 = D_2_5*x50 + D_4_5*x42 + D_5_5*x34;
    double x83 = D_2_3*x50 + D_4_3*x42 + D_5_3*x34;
    double x84 = D_2_0*x50 + D_4_0*x42 + D_5_0*x34;
    double x85 = D_2_4*x50 + D_4_4*x42 + D_5_4*x34;
    double x86 = D_2_1*x50 + D_4_1*x42 + D_5_1*x34;
    double x87 = D_2_2*x50 + D_4_2*x42 + D_5_2*x34;
    double x88 = D_0_5*x33 + D_3_5*x38 + D_5_5*x49;
    double x89 = D_0_3*x33 + D_3_3*x38 + D_5_3*x49;
    double x90 = D_0_0*x33 + D_3_0*x38 + D_5_0*x49;
    double x91 = D_0_4*x33 + D_3_4*x38 + D_5_4*x49;
    double x92 = D_0_1*x33 + D_3_1*x38 + D_5_1*x49;
    double x93 = D_0_2*x33 + D_3_2*x38 + D_5_2*x49;
    double x94 = D_1_5*x38 + D_3_5*x33 + D_4_5*x49;
    double x95 = D_1_3*x38 + D_3_3*x33 + D_4_3*x49;
    double x96 = D_1_0*x38 + D_3_0*x33 + D_4_0*x49;
    double x97 = D_1_4*x38 + D_3_4*x33 + D_4_4*x49;
    double x98 = D_1_1*x38 + D_3_1*x33 + D_4_1*x49;
    double x99 = D_1_2*x38 + D_3_2*x33 + D_4_2*x49;
    double x100 = D_2_5*x49 + D_4_5*x38 + D_5_5*x33;
    double x101 = D_2_3*x49 + D_4_3*x38 + D_5_3*x33;
    double x102 = D_2_0*x49 + D_4_0*x38 + D_5_0*x33;
    double x103 = D_2_4*x49 + D_4_4*x38 + D_5_4*x33;
    double x104 = D_2_1*x49 + D_4_1*x38 + D_5_1*x33;
    double x105 = D_2_2*x49 + D_4_2*x38 + D_5_2*x33;
    double x106 = D_0_5*x31 + D_3_5*x41 + D_5_5*x47;
    double x107 = D_0_3*x31 + D_3_3*x41 + D_5_3*x47;
    double x108 = D_0_0*x31 + D_3_0*x41 + D_5_0*x47;
    double x109 = D_0_4*x31 + D_3_4*x41 + D_5_4*x47;
    double x110 = D_0_1*x31 + D_3_1*x41 + D_5_1*x47;
    double x111 = D_0_2*x31 + D_3_2*x41 + D_5_2*x47;
    double x112 = D_1_5*x41 + D_3_5*x31 + D_4_5*x47;
    double x113 = D_1_3*x41 + D_3_3*x31 + D_4_3*x47;
    double x114 = D_1_0*x41 + D_3_0*x31 + D_4_0*x47;
    double x115 = D_1_4*x41 + D_3_4*x31 + D_4_4*x47;
    double x116 = D_1_1*x41 + D_3_1*x31 + D_4_1*x47;
    double x117 = D_1_2*x41 + D_3_2*x31 + D_4_2*x47;
    double x118 = D_2_5*x47 + D_4_5*x41 + D_5_5*x31;
    double x119 = D_2_3*x47 + D_4_3*x41 + D_5_3*x31;
    double x120 = D_2_0*x47 + D_4_0*x41 + D_5_0*x31;
    double x121 = D_2_4*x47 + D_4_4*x41 + D_5_4*x31;
    double x122 = D_2_1*x47 + D_4_1*x41 + D_5_1*x31;
    double x123 = D_2_2*x47 + D_4_2*x41 + D_5_2*x31;
    // --- 5. 组装 12x12 单元刚度矩阵 Ke（按节点 1..4，每节点 3 自由度） ---
    Ke[0] = x28*(x35*x52 + x43*x53 + x51*x54);
    Ke[1] = x28*(x35*x53 + x43*x55 + x51*x56);
    Ke[2] = x28*(x35*x54 + x43*x56 + x51*x57);
    Ke[3] = x28*(x34*x52 + x42*x53 + x50*x54);
    Ke[4] = x28*(x34*x53 + x42*x55 + x50*x56);
    Ke[5] = x28*(x34*x54 + x42*x56 + x50*x57);
    Ke[6] = x28*(x33*x52 + x38*x53 + x49*x54);
    Ke[7] = x28*(x33*x53 + x38*x55 + x49*x56);
    Ke[8] = x28*(x33*x54 + x38*x56 + x49*x57);
    Ke[9] = x28*(x31*x52 + x41*x53 + x47*x54);
    Ke[10] = x28*(x31*x53 + x41*x55 + x47*x56);
    Ke[11] = x28*(x31*x54 + x41*x56 + x47*x57);
    Ke[12] = x28*(x35*x58 + x43*x59 + x51*x60);
    Ke[13] = x28*(x35*x59 + x43*x61 + x51*x62);
    Ke[14] = x28*(x35*x60 + x43*x62 + x51*x63);
    Ke[15] = x28*(x34*x58 + x42*x59 + x50*x60);
    Ke[16] = x28*(x34*x59 + x42*x61 + x50*x62);
    Ke[17] = x28*(x34*x60 + x42*x62 + x50*x63);
    Ke[18] = x28*(x33*x58 + x38*x59 + x49*x60);
    Ke[19] = x28*(x33*x59 + x38*x61 + x49*x62);
    Ke[20] = x28*(x33*x60 + x38*x62 + x49*x63);
    Ke[21] = x28*(x31*x58 + x41*x59 + x47*x60);
    Ke[22] = x28*(x31*x59 + x41*x61 + x47*x62);
    Ke[23] = x28*(x31*x60 + x41*x62 + x47*x63);
    Ke[24] = x28*(x35*x64 + x43*x65 + x51*x66);
    Ke[25] = x28*(x35*x65 + x43*x67 + x51*x68);
    Ke[26] = x28*(x35*x66 + x43*x68 + x51*x69);
    Ke[27] = x28*(x34*x64 + x42*x65 + x50*x66);
    Ke[28] = x28*(x34*x65 + x42*x67 + x50*x68);
    Ke[29] = x28*(x34*x66 + x42*x68 + x50*x69);
    Ke[30] = x28*(x33*x64 + x38*x65 + x49*x66);
    Ke[31] = x28*(x33*x65 + x38*x67 + x49*x68);
    Ke[32] = x28*(x33*x66 + x38*x68 + x49*x69);
    Ke[33] = x28*(x31*x64 + x41*x65 + x47*x66);
    Ke[34] = x28*(x31*x65 + x41*x67 + x47*x68);
    Ke[35] = x28*(x31*x66 + x41*x68 + x47*x69);
    Ke[36] = x28*(x35*x72 + x43*x71 + x51*x70);
    Ke[37] = x28*(x35*x71 + x43*x74 + x51*x73);
    Ke[38] = x28*(x35*x70 + x43*x73 + x51*x75);
    Ke[39] = x28*(x34*x72 + x42*x71 + x50*x70);
    Ke[40] = x28*(x34*x71 + x42*x74 + x50*x73);
    Ke[41] = x28*(x34*x70 + x42*x73 + x50*x75);
    Ke[42] = x28*(x33*x72 + x38*x71 + x49*x70);
    Ke[43] = x28*(x33*x71 + x38*x74 + x49*x73);
    Ke[44] = x28*(x33*x70 + x38*x73 + x49*x75);
    Ke[45] = x28*(x31*x72 + x41*x71 + x47*x70);
    Ke[46] = x28*(x31*x71 + x41*x74 + x47*x73);
    Ke[47] = x28*(x31*x70 + x41*x73 + x47*x75);
    Ke[48] = x28*(x35*x78 + x43*x77 + x51*x76);
    Ke[49] = x28*(x35*x77 + x43*x80 + x51*x79);
    Ke[50] = x28*(x35*x76 + x43*x79 + x51*x81);
    Ke[51] = x28*(x34*x78 + x42*x77 + x50*x76);
    Ke[52] = x28*(x34*x77 + x42*x80 + x50*x79);
    Ke[53] = x28*(x34*x76 + x42*x79 + x50*x81);
    Ke[54] = x28*(x33*x78 + x38*x77 + x49*x76);
    Ke[55] = x28*(x33*x77 + x38*x80 + x49*x79);
    Ke[56] = x28*(x33*x76 + x38*x79 + x49*x81);
    Ke[57] = x28*(x31*x78 + x41*x77 + x47*x76);
    Ke[58] = x28*(x31*x77 + x41*x80 + x47*x79);
    Ke[59] = x28*(x31*x76 + x41*x79 + x47*x81);
    Ke[60] = x28*(x35*x84 + x43*x83 + x51*x82);
    Ke[61] = x28*(x35*x83 + x43*x86 + x51*x85);
    Ke[62] = x28*(x35*x82 + x43*x85 + x51*x87);
    Ke[63] = x28*(x34*x84 + x42*x83 + x50*x82);
    Ke[64] = x28*(x34*x83 + x42*x86 + x50*x85);
    Ke[65] = x28*(x34*x82 + x42*x85 + x50*x87);
    Ke[66] = x28*(x33*x84 + x38*x83 + x49*x82);
    Ke[67] = x28*(x33*x83 + x38*x86 + x49*x85);
    Ke[68] = x28*(x33*x82 + x38*x85 + x49*x87);
    Ke[69] = x28*(x31*x84 + x41*x83 + x47*x82);
    Ke[70] = x28*(x31*x83 + x41*x86 + x47*x85);
    Ke[71] = x28*(x31*x82 + x41*x85 + x47*x87);
    Ke[72] = x28*(x35*x90 + x43*x89 + x51*x88);
    Ke[73] = x28*(x35*x89 + x43*x92 + x51*x91);
    Ke[74] = x28*(x35*x88 + x43*x91 + x51*x93);
    Ke[75] = x28*(x34*x90 + x42*x89 + x50*x88);
    Ke[76] = x28*(x34*x89 + x42*x92 + x50*x91);
    Ke[77] = x28*(x34*x88 + x42*x91 + x50*x93);
    Ke[78] = x28*(x33*x90 + x38*x89 + x49*x88);
    Ke[79] = x28*(x33*x89 + x38*x92 + x49*x91);
    Ke[80] = x28*(x33*x88 + x38*x91 + x49*x93);
    Ke[81] = x28*(x31*x90 + x41*x89 + x47*x88);
    Ke[82] = x28*(x31*x89 + x41*x92 + x47*x91);
    Ke[83] = x28*(x31*x88 + x41*x91 + x47*x93);
    Ke[84] = x28*(x35*x96 + x43*x95 + x51*x94);
    Ke[85] = x28*(x35*x95 + x43*x98 + x51*x97);
    Ke[86] = x28*(x35*x94 + x43*x97 + x51*x99);
    Ke[87] = x28*(x34*x96 + x42*x95 + x50*x94);
    Ke[88] = x28*(x34*x95 + x42*x98 + x50*x97);
    Ke[89] = x28*(x34*x94 + x42*x97 + x50*x99);
    Ke[90] = x28*(x33*x96 + x38*x95 + x49*x94);
    Ke[91] = x28*(x33*x95 + x38*x98 + x49*x97);
    Ke[92] = x28*(x33*x94 + x38*x97 + x49*x99);
    Ke[93] = x28*(x31*x96 + x41*x95 + x47*x94);
    Ke[94] = x28*(x31*x95 + x41*x98 + x47*x97);
    Ke[95] = x28*(x31*x94 + x41*x97 + x47*x99);
    Ke[96] = x28*(x100*x51 + x101*x43 + x102*x35);
    Ke[97] = x28*(x101*x35 + x103*x51 + x104*x43);
    Ke[98] = x28*(x100*x35 + x103*x43 + x105*x51);
    Ke[99] = x28*(x100*x50 + x101*x42 + x102*x34);
    Ke[100] = x28*(x101*x34 + x103*x50 + x104*x42);
    Ke[101] = x28*(x100*x34 + x103*x42 + x105*x50);
    Ke[102] = x28*(x100*x49 + x101*x38 + x102*x33);
    Ke[103] = x28*(x101*x33 + x103*x49 + x104*x38);
    Ke[104] = x28*(x100*x33 + x103*x38 + x105*x49);
    Ke[105] = x28*(x100*x47 + x101*x41 + x102*x31);
    Ke[106] = x28*(x101*x31 + x103*x47 + x104*x41);
    Ke[107] = x28*(x100*x31 + x103*x41 + x105*x47);
    Ke[108] = x28*(x106*x51 + x107*x43 + x108*x35);
    Ke[109] = x28*(x107*x35 + x109*x51 + x110*x43);
    Ke[110] = x28*(x106*x35 + x109*x43 + x111*x51);
    Ke[111] = x28*(x106*x50 + x107*x42 + x108*x34);
    Ke[112] = x28*(x107*x34 + x109*x50 + x110*x42);
    Ke[113] = x28*(x106*x34 + x109*x42 + x111*x50);
    Ke[114] = x28*(x106*x49 + x107*x38 + x108*x33);
    Ke[115] = x28*(x107*x33 + x109*x49 + x110*x38);
    Ke[116] = x28*(x106*x33 + x109*x38 + x111*x49);
    Ke[117] = x28*(x106*x47 + x107*x41 + x108*x31);
    Ke[118] = x28*(x107*x31 + x109*x47 + x110*x41);
    Ke[119] = x28*(x106*x31 + x109*x41 + x111*x47);
    Ke[120] = x28*(x112*x51 + x113*x43 + x114*x35);
    Ke[121] = x28*(x113*x35 + x115*x51 + x116*x43);
    Ke[122] = x28*(x112*x35 + x115*x43 + x117*x51);
    Ke[123] = x28*(x112*x50 + x113*x42 + x114*x34);
    Ke[124] = x28*(x113*x34 + x115*x50 + x116*x42);
    Ke[125] = x28*(x112*x34 + x115*x42 + x117*x50);
    Ke[126] = x28*(x112*x49 + x113*x38 + x114*x33);
    Ke[127] = x28*(x113*x33 + x115*x49 + x116*x38);
    Ke[128] = x28*(x112*x33 + x115*x38 + x117*x49);
    Ke[129] = x28*(x112*x47 + x113*x41 + x114*x31);
    Ke[130] = x28*(x113*x31 + x115*x47 + x116*x41);
    Ke[131] = x28*(x112*x31 + x115*x41 + x117*x47);
    Ke[132] = x28*(x118*x51 + x119*x43 + x120*x35);
    Ke[133] = x28*(x119*x35 + x121*x51 + x122*x43);
    Ke[134] = x28*(x118*x35 + x121*x43 + x123*x51);
    Ke[135] = x28*(x118*x50 + x119*x42 + x120*x34);
    Ke[136] = x28*(x119*x34 + x121*x50 + x122*x42);
    Ke[137] = x28*(x118*x34 + x121*x42 + x123*x50);
    Ke[138] = x28*(x118*x49 + x119*x38 + x120*x33);
    Ke[139] = x28*(x119*x33 + x121*x49 + x122*x38);
    Ke[140] = x28*(x118*x33 + x121*x38 + x123*x49);
    Ke[141] = x28*(x118*x47 + x119*x41 + x120*x31);
    Ke[142] = x28*(x119*x31 + x121*x47 + x122*x41);
    Ke[143] = x28*(x118*x31 + x121*x41 + x123*x47);

    // 返回单元体积（带符号），方便外层做体积检查或后处理
    return x28;
}