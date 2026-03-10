// Tet4StiffnessMatrix.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 hyperFEM. All rights reserved.
 * Author: Xiaotong Wang (or hyperFEM Team)
 */
#pragma once

#include "entt/entt.hpp"
#include <Eigen/Dense>

/**
 * @brief Compute linear Tet4 element stiffness matrix (constant strain).
 * @param registry EnTT registry (expects Position + Connectivity on element)
 * @param element_entity element handle
 * @param D material matrix (6x6, Voigt order: xx,yy,zz,xy,yz,xz)
 * @param Ke_out output stiffness matrix (12x12)
 */
void compute_tet4_stiffness_matrix(
    entt::registry& registry,
    entt::entity element_entity,
    const Eigen::Matrix<double, 6, 6>& D,
    Eigen::MatrixXd& Ke_out
);

/**
 * @brief Low-level Tet4 stiffness routine using raw coordinates.
 * @param coords Node coordinates array [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3].
 * @param D Material matrix (6x6).
 * @param Ke Output stiffness matrix (flattened 12x12, row-major).
 * @return Element volume (detJ / 6).
 */
double compute_tet4_stiffness(
    const double* coords,
    const Eigen::Matrix<double, 6, 6>& D,
    double* Ke
);

