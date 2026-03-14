/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once
#include "PartGraph.h"
#include <queue>
#include <set>
#include <algorithm>

struct GraphAnalysisResult {
    // 关键路径上的节点集合 (Load -> Constraint)
    std::unordered_set<std::string> critical_path_nodes;
    // 连通分�?(每个 vector 代表一组相互连接的零件)
    std::vector<std::vector<std::string>> components;
};

class GraphAnalyzer {
public:
    static GraphAnalysisResult analyze(const PartGraph& graph) {
        GraphAnalysisResult result;
        
        // 1. 寻找所有连通分�?(解决图太�?太长的问�?
        std::unordered_set<std::string> visited;
        for (const auto& [name, node] : graph.nodes) {
            if (visited.find(name) == visited.end()) {
                std::vector<std::string> component;
                std::queue<std::string> q;
                
                q.push(name);
                visited.insert(name);
                while (!q.empty()) {
                    std::string curr = q.front();
                    q.pop();
                    component.push_back(curr);
                    
                    if (graph.nodes.count(curr)) {
                        for (const auto& edge : graph.nodes.at(curr).edges) {
                            if (visited.find(edge.target_part) == visited.end()) {
                                visited.insert(edge.target_part);
                                q.push(edge.target_part);
                            }
                        }
                    }
                }
                result.components.push_back(component);
            }
        }
        
        // 排序分量：包�?Load �?Constraint 的分量排在前�?
        std::sort(result.components.begin(), result.components.end(), [&](const auto& a, const auto& b) {
            bool a_important = has_load_or_fix(graph, a);
            bool b_important = has_load_or_fix(graph, b);
            if (a_important != b_important) return a_important > b_important;
            return a.size() > b.size(); // 大的分量排前�?
        });

        // 2. (可�? 寻找关键传力路径 (Dijkstra �?BFS)
        // 这里的简化逻辑：标记所有在 Load 分量中的节点�?Critical
        for (const auto& comp : result.components) {
            if (has_load_or_fix(graph, comp)) {
                for(const auto& node : comp) result.critical_path_nodes.insert(node);
            }
        }

        return result;
    }

public:
    static bool has_load_or_fix(const PartGraph& g, const std::vector<std::string>& nodes) {
        for (const auto& n : nodes) {
            // is_load_part: includes nodal loads (Force/Moment) and BaseAcceleration (/GRAV) via AppliedLoadRef
            if (g.nodes.at(n).is_load_part || g.nodes.at(n).is_constraint_part) return true;
        }
        return false;
    }
};