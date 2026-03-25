// DofMap.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "entt/entt.hpp"

/**
 * @brief DOF Mapping Resource
 * @details 
 *   - Stores globally unique data in registry.ctx() (Context)
 *   - Provides fast mapping from node entities to global DOF indices
 *   - Built once before analysis starts, then shared by multiple systems
 * 
 * Architectural advantages:
 *   - Decoupling: StiffnessSystem, MassSystem, ForceSystem, etc. can all use it
 *   - Caching: DOF numbering remains constant during nonlinear iterations
 *   - Performance: Uses vector for O(1) access (entity ID directly as index)
 */
struct DofMap {
    /**
     * @brief Core mapping: Entity ID -> Global DOF Start Index
     * @details 
     *   - Index: entity ID (converted via static_cast<uint32_t>(entity))
     *   - Value: the starting global DOF index for this node
     *   - Each node typically has 3 DOFs (x, y, z), so node DOF range is [index, index+2]
     *   - If value is -1, this entity ID is not a node or not assigned DOF
     */
    std::vector<int> node_to_dof_index;

    /**
     * @brief Total number of DOFs (i.e., size of system equations)
     */
    int num_total_dofs = 0;

    /**
     * @brief Number of DOFs per node (usually 3, for 3D solid elements)
     * @details Can be extended in the future to support different node types (e.g., beam nodes have 6 DOFs)
     */
    int dofs_per_node = 3;

    // ------------------------------------------------------------------
    // Curve mapping: unified storage for time curves / material test curves
    // ------------------------------------------------------------------
    std::unordered_map<std::string, entt::entity> curve_name_to_entity;
    std::unordered_map<entt::entity, std::string> curve_entity_to_name;

    /**
     * @brief Check if entity is in the mapping
     */
    bool has_node(entt::entity node_entity) const {
        uint32_t entity_id = static_cast<uint32_t>(node_entity);
        if (entity_id >= node_to_dof_index.size()) {
            return false;
        }
        return node_to_dof_index[entity_id] != -1;
    }

    /**
     * @brief Get node's global DOF index (safe version, with bounds checking)
     * @param node_entity node entity
     * @param dof DOF direction (0=x, 1=y, 2=z)
     * @return global DOF index, -1 if node does not exist
     */
    int get_dof_index(entt::entity node_entity, int dof) const {
        uint32_t entity_id = static_cast<uint32_t>(node_entity);
        if (entity_id >= node_to_dof_index.size()) {
            return -1;
        }
        int base_index = node_to_dof_index[entity_id];
        if (base_index == -1) {
            return -1;
        }
        return base_index + dof;
    }

    /**
     * @brief Fast get node's global DOF index (unsafe version, no bounds checking)
     * @param entity_id node entity ID (already converted to uint32_t)
     * @param dof DOF direction (0=x, 1=y, 2=z)
     * @return global DOF index
     * @details 
     *   - Performance optimization: skip bounds checking, direct array access
     *   - Usage prerequisite: ensure entity_id is valid and DofMap is correctly constructed
     *   - Use only in hot loops, Assembly phase usually meets prerequisites
     */
    inline int get_dof_index_unsafe(uint32_t entity_id, int dof) const {
        return node_to_dof_index[entity_id] + dof;
    }

    /**
     * @brief Get pointer to underlying array (for ultimate performance optimization)
     * @return pointer to node_to_dof_index array
     * @details Allows direct array access, avoiding function call overhead
     */
    const int* get_dof_array_ptr() const {
        return node_to_dof_index.data();
    }
};

