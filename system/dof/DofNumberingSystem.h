// DofNumberingSystem.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include <entt/entt.hpp>
#include "../../data_center/DofMap.h"

/**
 * @brief DOF 编号系统 (DOF Numbering System)
 * @details 
 *   - 负责构建节点到全局自由度的映射
 *   - 应该�?Assembly、Solver 等其他系统之前运�?
 *   - 将映射结果存储在 registry.ctx() 中，供所有系统共�?
 */
class DofNumberingSystem {
public:
    /**
     * @brief 构建 DOF 映射并存储到 Context �?
     * @param registry EnTT registry
     * @details 
     *   - 遍历所有节点实体（具有 Position 组件�?
     *   - 为每个节点分配连续的全局自由度编�?
     *   - 将映射存储在 registry.ctx<DofMap>() �?
     *   - 默认假设每个节点�?3 个自由度（x, y, z�?
     */
    static void build_dof_map(entt::registry& registry);
};

