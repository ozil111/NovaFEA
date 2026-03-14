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
    Contact,        // 显式定义的接�?(Tie, Surface-to-Surface)
    SharedNode,     // 隐式拓扑连接 (Mesh Topology)
    MPC             // 刚体连接
};

struct EdgeInfo {
    std::string target_part;
    ConnectionType type;
    // 细分连接类型，例�?"Tie", "Type7", "Type24" �?
    std::string sub_type;
    double weight;
    int count; // 共享节点数量 �?接触定义数量
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

    // sub_type 用于区分不同的接触算法或连接子类�?
    // 例如：ConnectionType::Contact 下的 "Tie" / "Type7" / "Type24"
    void add_edge(const std::string& src,
                  const std::string& tgt,
                  ConnectionType type,
                  double weight,
                  int count = 1,
                  const std::string& sub_type = {}) {
        if (nodes.find(src) == nodes.end()) add_node(src);
        if (nodes.find(tgt) == nodes.end()) add_node(tgt);

        // 检查是否已存在相同类型 + 相同子类�?的边，如果是则累加计�?
        auto& edges = nodes[src].edges;
        for (auto& edge : edges) {
            if (edge.target_part == tgt && edge.type == type && edge.sub_type == sub_type) {
                edge.count += count;
                // 取最小权�?(阻抗越小连接越紧�?
                if (weight < edge.weight) edge.weight = weight;
                return;
            }
        }
        edges.push_back({tgt, type, sub_type, weight, count});
    }
};