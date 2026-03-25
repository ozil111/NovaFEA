// TopologyData.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include <vector>
#include <unordered_map>
#include "entt/entt.hpp"

// --- Type aliases to enhance code clarity ---
using NodeID = int;       // External ID from input file
using ElementID = int;    // External ID from input file

using FaceID = size_t;       // Internal index for face entities (0 to N-1)
using BodyID = int;          // ID for contiguous mesh bodies

// Definition of a face: composed of its node external IDs sorted to ensure uniqueness and stability
using FaceKey = std::vector<NodeID>;

// --- Hash function to allow FaceKey to be used in unordered_map ---
// For std::vector<int> as key, a custom hash function is needed
struct VectorHasher {
    std::size_t operator()(const std::vector<int>& v) const {
        std::size_t seed = v.size();
        for(int i : v) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// -------------------------------------------------------------------
// **Core data structure - derived/accelerated data**
// This struct contains only topological data, no logic
// It serves as derived data computed from basic components in the EnTT registry
// Stored in the registry's context for use by subsequent systems
// -------------------------------------------------------------------
struct TopologyData {
    // --- Core topological entities ---

    // 1. Face entities
    // The index of `faces` is the FaceID
    std::vector<FaceKey> faces;
    // Fast lookup table from FaceKey -> FaceID
    std::unordered_map<FaceKey, FaceID, VectorHasher> face_key_to_id;

    // --- Relationship mappings ---
    // Note: entt::entity is used here instead of indices because entity is a stable handle

    // 2. Bidirectional lookup between elements and faces
    // `element_to_faces[entity]` -> get all faces owned by this element entity (FaceID)
    std::unordered_map<entt::entity, std::vector<FaceID>> element_to_faces;
    // `face_to_elements[face_id]` -> get all element entities sharing this face
    std::vector<std::vector<entt::entity>> face_to_elements;

    // 3. Relationship between elements and contiguous bodies
    // `element_to_body[entity]` -> get the contiguous body this element entity belongs to (BodyID)
    std::unordered_map<entt::entity, BodyID> element_to_body;
    // `body_to_elements[body_id]` -> get all element entities contained in this contiguous body
    std::unordered_map<BodyID, std::vector<entt::entity>> body_to_elements;

    // ================= Simdroid extensions =================

    // 1. [Reverse lookup] External Element ID -> Part Entity
    // Allows O(1) query of which Part any element belongs to
    // Index: ElementID (external ID)
    std::vector<entt::entity> element_uid_to_part_map;

    // 2. [Reverse lookup] External Node ID -> List of Part Entities
    // Allows O(1) query of which Parts share any node (for contact analysis)
    // Index: NodeID (external ID)
    std::vector<std::vector<entt::entity>> node_uid_to_parts_map;

    // --- Construction and cleanup ---
    TopologyData() = default;

    // Cleanup/reset helper function
    void clear_simdroid_maps() {
        element_uid_to_part_map.clear();
        node_uid_to_parts_map.clear();
    }

    // Pre-allocate memory (call before parsing)
    void reserve_simdroid_maps(size_t max_element_id, size_t max_node_id) {
        if (max_element_id >= element_uid_to_part_map.size()) {
            element_uid_to_part_map.resize(max_element_id + 1, entt::null);
        }
        if (max_node_id >= node_uid_to_parts_map.size()) {
            node_uid_to_parts_map.resize(max_node_id + 1);
        }
    }

    void clear() {
        faces.clear();
        face_key_to_id.clear();
        element_to_faces.clear();
        face_to_elements.clear();
        element_to_body.clear();
        body_to_elements.clear();
        clear_simdroid_maps();
    }
};