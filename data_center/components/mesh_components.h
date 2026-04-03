// mesh_components.h
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
#include <entt/entt.hpp>

// Type aliases for clarity and consistency
using NodeID = int;
using ElementID = int;

/**
 * @namespace Component
 * @brief Contains all ECS components for mesh representation
 * @details Components are organized by domain:
 *   - Geometric components: Position
 *   - Topological components: Connectivity, ElementType
 *   - Identification components: OriginalID
 *   - Set components: SetName, NodeSetMembers, ElementSetMembers
 */
namespace Component {

    // ===================================================================
    // Core Geometric & Topological Components
    // ===================================================================

    /**
     * @brief 3D position component for node entities
     * @details Attached to entities representing mesh nodes
     */
    struct Position {
        double x, y, z;
    };

    /**
     * @brief [Deprecated] Stores the original ID from the input file
     * @details To maintain consistency and avoid ID conflicts, now use dedicated ID components:
     *   - Node uses NodeID
     *   - Element uses ElementID
     *   This component is retained for backward compatibility, but new code should use dedicated ID components
     * @deprecated Use NodeID and ElementID instead
     */
    struct OriginalID {
        int value;
    };

    /**
     * @brief [New] Attached to Node entity, stores its user-defined ID (nid)
     * @details Used to identify Node entity, avoiding ID conflicts with other types of entities
     */
    struct NodeID {
        int value;
    };

    /**
     * @brief [New] Attached to Element entity, stores its user-defined ID (eid)
     * @details Used to identify Element entity, avoiding ID conflicts with other types of entities
     */
    struct ElementID {
        int value;
    };

    /**
     * @brief Element type identifier (e.g., 308 for Hexa8, 304 for Tetra4)
     * @details Attached to element entities. Used to:
     *   - Determine element topology
     *   - Look up properties in ElementRegistry
     *   - Extract faces for topology analysis
     */
    struct ElementType {
        int type_id;
    };

    /**
     * @brief Element-to-node connectivity
     * @details Attached to element entities. Stores direct entity handles
     * to node entities, enabling fast traversal without ID lookups.
     */
    struct Connectivity {
        std::vector<entt::entity> nodes;  // Direct handles to node entities
    };

    // ===================================================================
    // Set-Related Components
    // ===================================================================
    // Strategy: Each set (node set or element set) is represented as
    // a separate entity with SetName and member components attached.

    /**
     * @brief Name identifier for a set entity
     * @details Attached to set entities. Used for:
     *   - User-friendly identification
     *   - File export
     *   - Command-line queries
     */
    struct SetName {
        std::string value;
    };

    /**
     * @brief Members of a node set
     * @details Attached to node set entities. Contains entity handles
     * to all node entities that are members of this set.
     */
    struct NodeSetMembers {
        std::vector<entt::entity> members;
    };

    /**
     * @brief Members of an element set
     * @details Attached to element set entities. Contains entity handles
     * to all element entities that are members of this set.
     */
    struct ElementSetMembers {
        std::vector<entt::entity> members;
    };

    // ===================================================================
    // Surface (Face) Components (Simdroid)
    // ===================================================================
    // Note:
    // - "Surface" in Simdroid mesh.dat is a list of boundary faces/edges with stable IDs.
    // - We represent each surface entry as its own ECS entity, using dedicated components
    //   to avoid mixing them with volume/shell elements (which use Component::Connectivity).

    /**
     * @brief Surface ID component for surface entities (sid)
     * @details Attached to surface entities parsed from / exported to Simdroid "Surface {" block.
     */
    struct SurfaceID {
        int value;
    };

    /**
     * @brief Surface connectivity (nodes on the face/edge)
     * @details Uses direct node entity handles; does NOT reuse Component::Connectivity
     * to avoid being treated as a volume/shell element by other systems.
     */
    struct SurfaceConnectivity {
        std::vector<entt::entity> nodes;
    };

    /**
     * @brief Parent element reference for a surface entity
     * @details Simdroid surface lines append the parent element ID at the end.
     */
    struct SurfaceParentElement {
        entt::entity element;
    };

    /**
     * @brief Members of a surface set (SurfaceSet)
     * @details Attached to set entities representing Simdroid "Set { Surface { ... } }".
     */
    struct SurfaceSetMembers {
        std::vector<entt::entity> members;
    };

    /**
     * @brief [New] Attached to NodeSet entity, stores its user-defined ID (nsid)
     * @details Used to identify NodeSet entity, avoiding ID conflicts with other types of entities
     */
    struct NodeSetID {
        int value;
    };

    /**
     * @brief [New] Attached to EleSet entity, stores its user-defined ID (esid)
     * @details Used to identify EleSet entity, avoiding ID conflicts with other types of entities
     */
    struct EleSetID {
        int value;
    };

    // ===================================================================
    // Reference Components (Plan B: Entity-to-Entity References)
    // ===================================================================
    // These components establish relationships between entities by storing
    // entt::entity handles, enabling flexible and memory-efficient references.

    /**
     * @brief [New] Attached to Element entity, points to its associated Property (section) entity
     * @details Section parameters (integration, hourglass, etc.) are obtained from here; material is bound through SimdroidPart.
     * Get Part from TopologyData.element_uid_to_part_map, then get part.material.
     * 
     * Usage example (section):
     *   auto section_entity = registry.get<Component::PropertyRef>(element_entity).property_entity;
     *   const auto& property = registry.get<Component::SolidProperty>(section_entity);
     * Usage example (material, requires TopologyData and Part):
     *   auto part_entity = topology.element_uid_to_part_map[eid];
     *   entt::entity material_entity = registry.get<Component::SimdroidPart>(part_entity).material;
     */
    struct PropertyRef {
        entt::entity property_entity;
    };

    // ===================================================================
    // Explicit Dynamics Components
    // ===================================================================
    // Components for explicit time integration solver

    /**
     * @brief Node velocity component (for explicit dynamics)
     * @details Attached to Node entity, stores velocity components in three directions
     */
    struct Velocity {
        double vx, vy, vz;
    };

    /**
     * @brief Node acceleration component (for explicit dynamics)
     * @details Attached to Node entity, stores acceleration components in three directions
     */
    struct Acceleration {
        double ax, ay, az;
    };

    /**
     * @brief [New] Node base acceleration (gravity/base acceleration) component
     * @details Distinguished from Component::BaseAccelerationLoad (load definition).
     *          This component is used in the solving phase to cache "base acceleration already applied to nodes".
     *          Parsing phase usually only creates BaseAccelerationLoad and associates to nodes via AppliedLoadRef.
     */
    struct BaseAcceleration {
        double ax, ay, az;
    };

    /**
     * @brief Node displacement component (for explicit dynamics)
     * @details Attached to Node entity, stores displacement components in three directions
     */
    struct Displacement {
        double dx, dy, dz;
    };

    /**
     * @brief Node lumped mass component (for explicit dynamics)
     * @details Attached to Node entity, stores the node's lumped mass.
     * Obtained by MassSystem from element mass distribution
     */
    struct Mass {
        double value;
    };

    /**
     * @brief Node external force component (for explicit dynamics)
     * @details Attached to Node entity, stores external load components in three directions
     * Calculated by LoadSystem and AppliedLoadRef
     */
    struct ExternalForce {
        double fx, fy, fz;
    };

    /**
     * @brief Node internal force component (for explicit dynamics)
     * @details Attached to Node entity, stores internal force components in three directions from element stresses
     * Calculated by InternalForceSystem from element stresses
     */
    struct InternalForce {
        double fx, fy, fz;
    };

    /**
     * @brief Initial position component (for explicit dynamics)
     * @details Attached to Node entity, stores the node's initial position, used to calculate displacement increments
     * Copied from Position during solver initialization
     */
    struct InitialPosition {
        double x0, y0, z0;
    };

} // namespace Component

