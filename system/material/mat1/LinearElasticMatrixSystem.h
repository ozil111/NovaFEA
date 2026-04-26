// LinearElasticMatrixSystem.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include <entt/entt.hpp>
#include "../../../data_center/components/material_components.h"

// -------------------------------------------------------------------
// **材料系统 (Material Systems)**
// 这个类是无状态的，所有函数都是静态的。
// 它操作EnTT registry，从中读取材料参数组件，
// 计算并生成派生的材料矩阵组件。
// -------------------------------------------------------------------

void compute_single_linear_elastic_matrix(entt::registry& registry, entt::entity entity);


