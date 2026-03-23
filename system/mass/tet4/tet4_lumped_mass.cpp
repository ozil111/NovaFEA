// tet4_lumped_mass.cpp
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#include "Tet4Mass.h"
#include "../../mesh/TopologySystems.h"
#include "../../../data_center/TopologyData.h"
#include "../../../data_center/components/mesh_components.h"
#include "../../../data_center/components/simdroid_components.h"
#include "../../../data_center/components/material_components.h"
#include "spdlog/spdlog.h"
#include <cmath>

namespace {

/**
 * @brief Computes the tet4_op_lumped_mass kernel.
 * @note This is an optimized operator kernel.
 *
 * @param in Input array (const double*). Layout:
 *   - in[0..11]:  four nodes × (x, y, z) in order
 *   - in[12]: rho
 *
 * @param out Output array (double*). Layout:
 *   - out[0..3]: lumped mass contribution per node (generated code uses same value for all four)
 */
static inline void compute_tet4_op_lumped_mass(const double* __restrict__ in, double* __restrict__ out) {
    const double c0 = in[0];
    const double c1 = in[1];
    const double c2 = in[2];
    const double c3 = in[3];
    const double c4 = in[4];
    const double c5 = in[5];
    const double c6 = in[6];
    const double c7 = in[7];
    const double c8 = in[8];
    const double c9 = in[9];
    const double c10 = in[10];
    const double c11 = in[11];
    const double rho = in[12];

    double v_0_0 = c0 * c10;
    double v_0_1 = c0 * c11;
    double v_0_2 = c4 * c8;
    double v_0_3 = c1 * c11;
    double v_0_4 = c1 * c5;
    double v_0_5 = c1 * c8;
    double v_0_6 = c10 * c2;
    double v_0_7 = c4 * c6;
    double v_0_8 = c3 * c7;
    double v_0_9 = c2 * c9;
    double v_0_10 = c5 * c7;
    double v_0_11 = 0.041666666666666664 * rho * std::fabs(
        c0 * v_0_10 - c0 * v_0_2 - c10 * c3 * c8 + c10 * c5 * c6 - c11 * v_0_7 + c11 * v_0_8 + c2 * v_0_7 - c2 * v_0_8
        - c3 * v_0_3 + c3 * v_0_5 + c3 * v_0_6 + c4 * v_0_1 - c4 * v_0_9 - c5 * v_0_0 + c6 * v_0_3 - c6 * v_0_4 - c6 * v_0_6
        - c7 * v_0_1 + c7 * v_0_9 + c8 * v_0_0 - c9 * v_0_10 + c9 * v_0_2 + c9 * v_0_4 - c9 * v_0_5);
    out[0] = v_0_11;
    out[1] = v_0_11;
    out[2] = v_0_11;
    out[3] = v_0_11;
}

}  // namespace

bool compute_tet4_mass(entt::registry& registry, entt::entity element_entity) {
    const auto& connectivity = registry.get<Component::Connectivity>(element_entity);

    if (connectivity.nodes.size() != 4) {
        spdlog::warn("Element has {} nodes, expected 4 for Tet4. Skipping.", connectivity.nodes.size());
        return false;
    }

    if (!registry.all_of<Component::ElementID>(element_entity)) {
        spdlog::warn("Element missing ElementID. Skipping mass calculation.");
        return false;
    }
    if (!registry.ctx().contains<std::unique_ptr<TopologyData>>()) {
        spdlog::info("TopologyData not found. Building topology before Tet4 mass...");
        TopologySystems::extract_topology(registry);
    }
    auto& topology = *registry.ctx().get<std::unique_ptr<TopologyData>>();
    int eid = registry.get<Component::ElementID>(element_entity).value;
    if (eid < 0 || static_cast<size_t>(eid) >= topology.element_uid_to_part_map.size()) {
        spdlog::warn("Element ID out of range. Skipping mass calculation.");
        return false;
    }
    entt::entity part_entity = topology.element_uid_to_part_map[static_cast<size_t>(eid)];
    if (part_entity == entt::null || !registry.all_of<Component::SimdroidPart>(part_entity)) {
        spdlog::warn("No Part for element. Skipping mass calculation.");
        return false;
    }
    entt::entity material_entity = registry.get<Component::SimdroidPart>(part_entity).material;

    if (!registry.all_of<Component::LinearElasticParams>(material_entity)) {
        spdlog::warn("Material missing LinearElasticParams. Skipping mass calculation.");
        return false;
    }

    const auto& material_params = registry.get<Component::LinearElasticParams>(material_entity);
    double rho = material_params.rho;

    double in[13];
    for (size_t n = 0; n < 4; ++n) {
        entt::entity node_entity = connectivity.nodes[n];
        if (!registry.all_of<Component::Position>(node_entity)) {
            spdlog::warn("Node missing Position component. Skipping element.");
            return false;
        }
        const auto& pos = registry.get<Component::Position>(node_entity);
        in[3 * n + 0] = pos.x;
        in[3 * n + 1] = pos.y;
        in[3 * n + 2] = pos.z;
    }
    in[12] = rho;

    double out[4];
    compute_tet4_op_lumped_mass(in, out);

    for (size_t i = 0; i < 4; ++i) {
        entt::entity node_entity = connectivity.nodes[i];
        double nodal_mass = out[i];
        if (!registry.all_of<Component::Mass>(node_entity)) {
            registry.emplace<Component::Mass>(node_entity, nodal_mass);
        } else {
            registry.get<Component::Mass>(node_entity).value += nodal_mass;
        }
    }

    return true;
}
