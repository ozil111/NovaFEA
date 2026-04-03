// data_center/components/load_components.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include <string>
#include <vector>
#include <entt/entt.hpp>

/**
 * @namespace Component
 * @brief ECS components - Load and Boundary (loads and boundary conditions) section
 * @details These components are divided into two categories:
 *   1. Definition components: Attached to Load/Boundary entities, store abstract definitions of loads/boundary conditions
 *   2. Application components: Attached to Node/Element entities, point to Load/Boundary entities applied to them
 */
namespace Component {

    // ===================================================================
    // ID components - for identifying different types of entities, avoid ID conflicts
    // ===================================================================

    /**
     * @brief [New] Attached to Load entity, stores its user-defined ID (lid)
     * @details Used to identify Load entity, avoiding ID conflicts with other types of entities
     */
    struct LoadID {
        int value;
    };

    /**
     * @brief [New] Attached to Boundary entity, stores its user-defined ID (bid)
     * @details Used to identify Boundary entity, avoiding ID conflicts with other types of entities
     */
    struct BoundaryID {
        int value;
    };

    // ===================================================================
    // Definition components - attached to Load/Boundary entities
    // ===================================================================

    /**
     * @brief [New] Attached to Load entity, stores definition of nodal loads
     * @details Corresponds to "load" object in JSON
     * A Load entity represents an abstract load definition, can be applied to multiple nodes
     * When curve_entity is not null, load value will be scaled by the curve over time
     */
    struct NodalLoad {
        int type_id;           // Load type ID
        std::string dof;       // Degrees of freedom: "all", "x", "y", "z", "xy", etc.
        double value;          // Load value
        entt::entity curve_entity = entt::null;  // Optional time curve entity, entt::null means no scaling
    };

    /**
     * @brief [New] Attached to Load entity, stores definition of base/gravity acceleration (Base Acceleration)
     * @details Corresponds to Type="BaseAcceleration" (GRAV) in Simdroid JSON.
     *          Each direction can specify constant value and time curve separately (1 independent + 1 dependent variable function).
     *          If curve_entity is not null, acceleration in that direction scales with time according to the curve.
     *          coord_sys is optional local coordinate system name/ID (Skew_ID), empty means global coordinate system.
     */
    struct BaseAccelerationLoad {
        double ax = 0.0;
        double ay = 0.0;
        double az = 0.0;
        entt::entity x_curve_entity = entt::null;
        entt::entity y_curve_entity = entt::null;
        entt::entity z_curve_entity = entt::null;
        std::string coord_sys;
    };

    /**
     * @brief [Deprecated] Please use NodalLoad::curve_entity. Retained only for compatibility with old data.
     * @details If present, load value will be scaled according to curve and time.
     */
    struct CurveRef {
        entt::entity curve_entity;  // Reference to Curve entity
    };

    /**
     * @brief [New] Attached to Boundary entity, stores definition of single-point constraint (SPC)
     * @details Corresponds to "boundary" object in JSON
     * A Boundary entity represents an abstract boundary condition definition, can be applied to multiple nodes
     */
    struct BoundarySPC {
        int type_id;        // Boundary condition type ID
        std::string dof;    // Constrained degrees of freedom: "all", "x", "y", "z", "xy", etc.
        double value;       // Constraint value (usually 0.0 means fixed)
    };

    /**
     * @brief [New] Attached to Curve entity, stores curve ID
     * @details Used to identify Curve entity
     */
    struct CurveID {
        int value;
    };

    /**
     * @brief [New] Attached to Curve entity, stores curve definition
     * @details Corresponds to "curve" object in JSON
     * Supports different types of curves (e.g. linear), used to scale loads over time
     */
    struct Curve {
        std::string type;   // Curve type, e.g. "linear"
        std::vector<double> x;  // x-coordinate array (usually time)
        std::vector<double> y;  // y-coordinate array (usually scaling factor)
    };

    // ===================================================================
    // Application components - attached to Node/Element entities
    // ===================================================================

    /**
     * @brief [New] Attached to Node entity, contains references to all Load entities applied to this node
     * @details This is the core of Plan B: implement "many-to-one" relationship through references.
     * Using vector allows a single node to reference multiple load definitions (e.g. ForceX and ForceY coexist).
     * 
     * Usage example:
     * // In solver, iterate over all loaded nodes
     * auto view = registry.view<Component::AppliedLoadRef, Component::Position>();
     * for(auto [node_entity, load_ref, pos] : view.each()) {
     * for(const auto load_entity : load_ref.load_entities) {
     * // Get load definition
     * const auto& load = registry.get<Component::NodalLoad>(load_entity);
     * // Apply load to node
     * apply_force(node_entity, load.dof, load.value);
     * }
     * }
     */
    struct AppliedLoadRef {
        std::vector<entt::entity> load_entities;
    };

    /**
     * @brief [New] Attached to Node entity, contains references to all Boundary entities applied to this node
     * @details Similar to AppliedLoadRef, but for boundary conditions
     */
    struct AppliedBoundaryRef {
        std::vector<entt::entity> boundary_entities;
    };

    // Future extensions: can add other types of loads and boundary conditions
    // struct AppliedPressureRef { entt::entity pressure_entity; };
    // struct AppliedTemperatureRef { entt::entity temperature_entity; };

} // namespace Component

