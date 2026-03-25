// PartGraph.h
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
#include <unordered_map>

enum class ConnectionType {
    Contact,        // Explicitly defined contact (Tie, Surface-to-Surface)
    SharedNode,     // Implicit topological connection (Mesh Topology)
    MPC             // Rigid body connection
};

struct EdgeInfo {
    std::string target_part;
    ConnectionType type;
    // Detailed connection type, e.g. "Tie", "Type7", "Type24"
    std::string sub_type;
    double weight;
    int count; // Number of shared nodes or contact definitions
};

struct PartNode {
    std::string name;
    bool is_load_part = false;
    bool is_constraint_part = false;
    // 用于在可视化中显示的材料/属性信�?
    std::string material_info;
    std::string property_info;
    std::vector<EdgeInfo> edges;
};

class PartGraph {
public:
    std::unordered_map<std::string, PartNode> nodes;

    void add_node(const std::string& name) {
        if (nodes.find(name) == nodes.end()) {
            nodes[name] = {name};
        }
    }

    // sub_type is used to distinguish different contact algorithms or connection subclasses
    // For example: "Tie" / "Type7" / "Type24" under ConnectionType::Contact
    void add_edge(const std::string& src,
                  const std::string& tgt,
                  ConnectionType type,
                  double weight,
                  int count = 1,
                  const std::string& sub_type = {}) {
        if (nodes.find(src) == nodes.end()) add_node(src);
        if (nodes.find(tgt) == nodes.end()) add_node(tgt);

        // Check if an edge of the same type + same sub_type already exists, if so accumulate the count
        auto& edges = nodes[src].edges;
        for (auto& edge : edges) {
            if (edge.target_part == tgt && edge.type == type && edge.sub_type == sub_type) {
                edge.count += count;
                // Take the minimum weight (lower impedance means tighter connection)
                if (weight < edge.weight) edge.weight = weight;
                return;
            }
        }
        edges.push_back({tgt, type, sub_type, weight, count});
    }
};