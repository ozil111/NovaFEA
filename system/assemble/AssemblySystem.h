// AssemblySystem.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include <entt/entt.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>

// -------------------------------------------------------------------
// **组装系统 (Assembly System)**
// 负责将单元刚度矩阵组装到全局刚度矩阵中�?
// 包含分发器（Dispatcher）和组装循环（Assembly Loop）�?
// -------------------------------------------------------------------

class AssemblySystem {
public:
    // 使用 Eigen::SparseMatrix 作为全局刚度矩阵类型
    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using Triplet = Eigen::Triplet<double>;

    static constexpr int MAX_ELEMENT_DOFS = 60;

    /**
     * @brief [Dispatcher] 根据单元类型分发到相应的刚度矩阵计算函数
     * @param registry EnTT registry
     * @param element_entity 单元实体句柄
     * @param Ke_raw 输出的单元刚度矩阵缓冲区（row-major, 至少 MAX_ELEMENT_DOFS^2�?
     * @param element_dofs 输出的单元自由度�?
     * @return true 如果成功计算，false 如果不支持的单元类型
     */
    static bool compute_element_stiffness_dispatcher(
        entt::registry& registry,
        entt::entity element_entity,
        double* Ke_raw,
        int& element_dofs
    );

    /**
     * @brief [Assembly] 组装全局刚度矩阵
     * @param registry EnTT registry
     * @param K_global 输出的全局刚度矩阵（稀疏矩阵）
     * @details 
     *   - 遍历所有单元，调用 dispatcher 获取单元刚度矩阵
     *   - 将单元刚度矩阵组装到全局矩阵�?
     *   - 使用 registry.ctx<DofMap>() 中的映射（需要先运行 DofNumberingSystem�?
     * 
     * @pre 必须事先调用 DofNumberingSystem::build_dof_map(registry)
     */
    static void assemble_stiffness(
        entt::registry& registry,
        SparseMatrix& K_global
    );
};

