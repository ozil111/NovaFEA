// mesh.h
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
#include <string> // Added for Set Name/ID mapping if needed later

// Use type aliases to enhance code readability and maintainability
using NodeID = int;
using ElementID = int;
using SetID = int; // Add new type alias for Set ID

// This is a pure data aggregate describing a specific mesh instance
struct Mesh {
    // --- Node data (SoA format) ---
    std::vector<double> node_coordinates;         // [x1, y1, z1, x2, y2, z2, ...]
    std::unordered_map<NodeID, size_t> node_id_to_index; // External ID -> internal index
    std::vector<NodeID> node_index_to_id;         // Internal index -> external ID

    // --- Element data (SoA format) ---
    std::vector<int> element_types;               // [type_e1, type_e2, ...] stores type ID for each element
    std::vector<int> element_connectivity;        // All element node IDs stored contiguously
    std::vector<size_t> element_offsets;          // Starting position of each element's node list in connectivity
    std::unordered_map<ElementID, size_t> element_id_to_index; // External ID -> internal index
    std::vector<ElementID> element_index_to_id;   // Internal index -> external ID

    // --- Set data (Set Data) ---
    // Bi-directional mapping for Set Names and IDs
    SetID next_set_id = 0; // Simple counter to generate unique SetIDs
    std::unordered_map<std::string, SetID> set_name_to_id;
    std::vector<std::string> set_id_to_name;

    // Containers for the actual set data, using efficient integer IDs
    std::unordered_map<SetID, std::vector<NodeID>>       node_sets;
    std::unordered_map<SetID, std::vector<ElementID>>    element_sets; 

    // --- Helper functions ---
    size_t getNodeCount() const {
        return node_index_to_id.size();
    }
    
    size_t getElementCount() const {
        return element_index_to_id.size();
    }

    // --- Management functions ---
    void clear() {
        // Clear node data
        node_coordinates.clear();
        node_id_to_index.clear();
        node_index_to_id.clear();

        // Clear element data
        element_types.clear();
        element_connectivity.clear();
        element_offsets.clear();
        element_id_to_index.clear();
        element_index_to_id.clear();

        // Clear set data
        next_set_id = 0;
        set_name_to_id.clear();
        set_id_to_name.clear();
        node_sets.clear();
        element_sets.clear();
    }
};