// Tet4InternalForce.cpp
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2026 NovaFEA. All rights reserved.
 */
#include "Tet4InternalForce.h"
#include "../../../data_center/TopologyData.h"
#include "../../../data_center/components/material_components.h"
#include "../../../data_center/components/mesh_components.h"
#include "../../../data_center/components/simdroid_components.h"

#include <array>

#include "tet4_assembly.cpp"
#include "tet4_dN_dnat.cpp"
#include "tet4_mapping.cpp"

bool compute_tet4_internal_forces(entt::registry& registry, entt::entity element_entity) {
    if (!registry.all_of<Component::Connectivity, Component::ElementType>(element_entity)) {
        return false;
    }

    const auto& connectivity = registry.get<Component::Connectivity>(element_entity);
    if (connectivity.nodes.size() != 4) {
        return false;
    }

    if (!registry.all_of<Component::ElementID>(element_entity)) {
        return false;
    }
    if (!registry.ctx().contains<std::unique_ptr<TopologyData>>()) {
        return false;
    }

    auto& topology = *registry.ctx().get<std::unique_ptr<TopologyData>>();
    int eid = registry.get<Component::ElementID>(element_entity).value;
    if (eid < 0 || static_cast<size_t>(eid) >= topology.element_uid_to_part_map.size()) {
        return false;
    }

    entt::entity part_entity = topology.element_uid_to_part_map[static_cast<size_t>(eid)];
    if (part_entity == entt::null || !registry.all_of<Component::SimdroidPart>(part_entity)) {
        return false;
    }

    entt::entity material_entity = registry.get<Component::SimdroidPart>(part_entity).material;
    if (!registry.all_of<Component::LinearElasticMatrix>(material_entity)) {
        return false;
    }

    const auto& material_matrix = registry.get<Component::LinearElasticMatrix>(material_entity);
    if (!material_matrix.is_initialized) {
        return false;
    }

    std::array<double, 12> coords_current{};
    std::array<double, 12> u_e{};
    for (size_t i = 0; i < 4; ++i) {
        entt::entity node_entity = connectivity.nodes[i];
        if (!registry.all_of<Component::Position>(node_entity)) {
            return false;
        }

        const auto& pos = registry.get<Component::Position>(node_entity);
        double x0 = pos.x;
        double y0 = pos.y;
        double z0 = pos.z;
        if (registry.all_of<Component::InitialPosition>(node_entity)) {
            const auto& pos0 = registry.get<Component::InitialPosition>(node_entity);
            x0 = pos0.x0;
            y0 = pos0.y0;
            z0 = pos0.z0;
        }

        coords_current[3 * i + 0] = pos.x;
        coords_current[3 * i + 1] = pos.y;
        coords_current[3 * i + 2] = pos.z;

        u_e[3 * i + 0] = pos.x - x0;
        u_e[3 * i + 1] = pos.y - y0;
        u_e[3 * i + 2] = pos.z - z0;
    }

    // 1-point integration at centroid for linear TET4.
    std::array<double, 3> dnat_in{0.25, 0.25, 0.25};
    std::array<double, 12> dN_dnat{};
    compute_tet4_op_dN_dnat(dnat_in.data(), dN_dnat.data());

    std::array<double, 24> mapping_in{};
    for (size_t i = 0; i < 12; ++i) {
        mapping_in[i] = coords_current[i];
        mapping_in[12 + i] = dN_dnat[i];
    }
    std::array<double, 13> mapping_out{};
    compute_tet4_op_mapping(mapping_in.data(), mapping_out.data());

    std::array<double, 50> assembly_in{};
    for (size_t i = 0; i < 12; ++i) {
        assembly_in[i] = mapping_out[i];
    }

    const auto& D = material_matrix.D;
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            assembly_in[12 + 6 * r + c] = D[r * 6 + c];
        }
    }

    assembly_in[48] = mapping_out[12];
    assembly_in[49] = 1.0 / 6.0;

    std::array<double, 144> K_e{};
    compute_tet4_op_assembly(assembly_in.data(), K_e.data());

    std::array<double, 12> f_e{};
    for (int i = 0; i < 12; ++i) {
        double acc = 0.0;
        for (int j = 0; j < 12; ++j) {
            acc += K_e[12 * i + j] * u_e[j];
        }
        f_e[i] = acc;
    }

    for (size_t i = 0; i < 4; ++i) {
        entt::entity node_entity = connectivity.nodes[i];
        if (!registry.all_of<Component::InternalForce>(node_entity)) {
            registry.emplace<Component::InternalForce>(node_entity, 0.0, 0.0, 0.0);
        }

        auto& internal_force = registry.get<Component::InternalForce>(node_entity);
        internal_force.fx += f_e[3 * i + 0];
        internal_force.fy += f_e[3 * i + 1];
        internal_force.fz += f_e[3 * i + 2];
    }

    return true;
}

