// LinearElasticMatrixSystem.cpp
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#include "LinearElasticMatrixSystem.h"
#include "isotropic_D_gen.cpp"
#include <spdlog/spdlog.h>

void compute_single_linear_elastic_matrix(entt::registry& registry, entt::entity entity){
    const auto& params = registry.get<const Component::LinearElasticParams>(entity);
    double E = params.E;
    double nu = params.nu;

    double in[2] = {E, nu};
    double out[36] = {};
    compute_isotropic_D(in, out);

    Eigen::Matrix<double, 6, 6> D;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            D(i, j) = out[i * 6 + j];

    auto& matrix_comp = registry.get_or_emplace<Component::LinearElasticMatrix>(entity);
    matrix_comp.D = D;
    matrix_comp.is_initialized = true;
}