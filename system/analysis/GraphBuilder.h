/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 hyperFEM. All rights reserved.
 * Author: Xiaotong Wang (or hyperFEM Team)
 */
#pragma once
#include "PartGraph.h"
#include "entt/entt.hpp"
#include "../simdroid/SimdroidInspector.h"
#include "../../data_center/components/simdroid_components.h"
#include "../../data_center/components/load_components.h"
#include <set>
#include <map>
#include <algorithm>

namespace Component {
    struct NodeID;
}

class GraphBuilder {
public:
    static PartGraph build(entt::registry& registry, SimdroidInspector& inspector) {
        PartGraph graph;

        // 确保索引已构建
        if (!inspector.is_built) {
            inspector.build(registry);
        }

        // 1. 初始化节点 (Parts)
        auto view_parts = registry.view<const Component::SimdroidPart>();
        for (auto entity : view_parts) {
            const auto& part = view_parts.get<const Component::SimdroidPart>(entity);
            graph.add_node(part.name);
        }

        // -------------------------------------------------------
        // 2. 处理 Contact (显式连接)
        //    同时读取 ContactTypeTag，用于细分接触类型 (Tie / Type7 / Type24 等)
        // -------------------------------------------------------
        auto view_contacts = registry.view<const Component::ContactBase, const Component::ContactTypeTag>();
        for (auto entity : view_contacts) {
            const auto& contact = view_contacts.get<const Component::ContactBase>(entity);
            const auto& type_tag = view_contacts.get<const Component::ContactTypeTag>(entity);

            // 将 ContactInterType 映射为更具体的算法名称
            // /INTER/TYPE2 (Tie), /INTER/TYPE7 (N-S), /INTER/TYPE24 (General)
            std::string sub_type;
            switch (type_tag.type) {
                case Component::ContactInterType::Tie:
                    sub_type = "Tie";
                    break;
                case Component::ContactInterType::NodeToSurface:
                    // 使用 Radioss 习惯名称，便于在可视化中显示为 Contact (Type7)
                    sub_type = "Type7";
                    break;
                case Component::ContactInterType::General:
                    sub_type = "Type24";
                    break;
                default:
                    sub_type = "Unknown";
                    break;
            }

            auto master_parts = get_parts_from_set(registry, inspector, contact.master_entity);
            auto slave_parts = get_parts_from_set(registry, inspector, contact.slave_entity);

            for (const auto& m : master_parts) {
                for (const auto& s : slave_parts) {
                    if (m != s) {
                        // Contact 连接通常被视为“强连接”，权重设低一点 (1.0)
                        graph.add_edge(m, s, ConnectionType::Contact, 1.0, 1, sub_type);
                        graph.add_edge(s, m, ConnectionType::Contact, 1.0, 1, sub_type);
                    }
                }
            }
        }

        // -------------------------------------------------------
        // 3. 处理 Shared Nodes (隐式拓扑连接) - 核心算法
        // -------------------------------------------------------
        // 逻辑：遍历每个节点 -> 找出该节点属于哪些 Part -> 如果 >1 个 Part，则这些 Part 互联
        
        // 临时存储：pair<PartA, PartB> -> SharedNodeCount
        std::map<std::pair<std::string, std::string>, int> shared_topology_map;

        for (const auto& [nid, elem_ids] : inspector.nid_to_elems) {
            // 如果一个节点只被一个单元引用，或者只被同一个 Part 的单元引用，则无连接
            if (elem_ids.empty()) continue;

            // 收集该节点涉及的所有 Part
            std::vector<std::string> parts_sharing_this_node;
            parts_sharing_this_node.reserve(4); // 预留少量空间，通常不会超过4个

            for (int eid : elem_ids) {
                // 利用 Inspector 的 O(1) 查找
                if (inspector.eid_to_part.count(eid)) {
                    const std::string& p_name = inspector.eid_to_part.at(eid);
                    // 避免重复添加 (std::unique 需要排序，这里手动检查更快)
                    bool already_added = false;
                    for (const auto& existing : parts_sharing_this_node) {
                        if (existing == p_name) { already_added = true; break; }
                    }
                    if (!already_added) {
                        parts_sharing_this_node.push_back(p_name);
                    }
                }
            }

            // 如果该节点被多个 Part 共享，建立两两连接
            if (parts_sharing_this_node.size() > 1) {
                // 排序以确保 pair 的 key 一致性 (A, B) vs (B, A)
                std::sort(parts_sharing_this_node.begin(), parts_sharing_this_node.end());
                
                for (size_t i = 0; i < parts_sharing_this_node.size(); ++i) {
                    for (size_t j = i + 1; j < parts_sharing_this_node.size(); ++j) {
                        shared_topology_map[{parts_sharing_this_node[i], parts_sharing_this_node[j]}]++;
                    }
                }
            }
        }

        // 将统计结果写入 Graph
        for (const auto& [pair, count] : shared_topology_map) {
            // 阈值过滤：比如共享节点少于 3 个的可能是噪点？这里暂时全部保留
            // Shared Node 权重逻辑：共享节点越多，连接越紧密 (权重越低)
            double weight = (count > 100) ? 0.1 : (count > 10 ? 0.5 : 2.0);
            
            graph.add_edge(pair.first, pair.second, ConnectionType::SharedNode, weight, count);
            graph.add_edge(pair.second, pair.first, ConnectionType::SharedNode, weight, count);
        }

        // -------------------------------------------------------
        // 4. 标记 Load / Constraint 所属的 Part
        //    通过节点 -> 元素 -> Part 的映射，将有载荷/约束的节点
        //    归属到对应 PartNode 上。
        // -------------------------------------------------------
        {
            // 4.1 处理 Nodal Load -> is_load_part
            auto view_node_load = registry.view<const Component::AppliedLoadRef, const Component::NodeID>();
            for (auto entity : view_node_load) {
                const int nid = view_node_load.get<const Component::NodeID>(entity).value;
                auto it_ne = inspector.nid_to_elems.find(nid);
                if (it_ne == inspector.nid_to_elems.end()) continue;

                for (int eid : it_ne->second) {
                    auto it_part = inspector.eid_to_part.find(eid);
                    if (it_part == inspector.eid_to_part.end()) continue;
                    auto node_it = graph.nodes.find(it_part->second);
                    if (node_it != graph.nodes.end()) {
                        node_it->second.is_load_part = true;
                    }
                }
            }

            // 4.2 处理 Boundary 条件 -> is_constraint_part
            auto view_node_fix = registry.view<const Component::AppliedBoundaryRef, const Component::NodeID>();
            for (auto entity : view_node_fix) {
                const int nid = view_node_fix.get<const Component::NodeID>(entity).value;
                auto it_ne = inspector.nid_to_elems.find(nid);
                if (it_ne == inspector.nid_to_elems.end()) continue;

                for (int eid : it_ne->second) {
                    auto it_part = inspector.eid_to_part.find(eid);
                    if (it_part == inspector.eid_to_part.end()) continue;
                    auto node_it = graph.nodes.find(it_part->second);
                    if (node_it != graph.nodes.end()) {
                        node_it->second.is_constraint_part = true;
                    }
                }
            }
        }

        return graph;
    }

private:
    // 辅助函数：从 Set Entity 解析出包含的 Parts
    static std::vector<std::string> get_parts_from_set(entt::registry& reg, SimdroidInspector& insp, entt::entity set_entity) {
        std::vector<std::string> parts;
        if (!reg.valid(set_entity)) return parts;

        std::set<std::string> unique_parts;

        // 1) Element Set: 直接通过 ElementID / OriginalID -> Part
        if (reg.all_of<Component::ElementSetMembers>(set_entity)) {
            const auto& members = reg.get<Component::ElementSetMembers>(set_entity).members;
            for (auto ent : members) {
                int eid = -1;
                if (reg.all_of<Component::ElementID>(ent)) {
                    eid = reg.get<Component::ElementID>(ent).value;
                } else if (reg.all_of<Component::OriginalID>(ent)) {
                    eid = reg.get<Component::OriginalID>(ent).value;
                }
                if (eid >= 0 && insp.eid_to_part.count(eid)) {
                    unique_parts.insert(insp.eid_to_part.at(eid));
                }
            }
        }

        // 2) Node Set: 通过 NodeID -> nid_to_elems -> eid_to_part
        if (reg.all_of<Component::NodeSetMembers>(set_entity)) {
            const auto& members = reg.get<Component::NodeSetMembers>(set_entity).members;
            for (auto node_ent : members) {
                if (!reg.valid(node_ent) || !reg.all_of<Component::NodeID>(node_ent)) continue;
                int nid = reg.get<Component::NodeID>(node_ent).value;
                auto it = insp.nid_to_elems.find(nid);
                if (it == insp.nid_to_elems.end()) continue;
                for (int eid : it->second) {
                    if (insp.eid_to_part.count(eid)) {
                        unique_parts.insert(insp.eid_to_part.at(eid));
                    }
                }
            }
        }

        // 3) Surface Set: 通过 SurfaceParentElement -> ElementID -> eid_to_part
        if (reg.all_of<Component::SurfaceSetMembers>(set_entity)) {
            const auto& members = reg.get<Component::SurfaceSetMembers>(set_entity).members;
            for (auto surf_ent : members) {
                if (!reg.valid(surf_ent) || !reg.all_of<Component::SurfaceParentElement>(surf_ent)) continue;
                entt::entity parent_elem = reg.get<Component::SurfaceParentElement>(surf_ent).element;
                if (!reg.valid(parent_elem)) continue;

                int eid = -1;
                if (reg.all_of<Component::ElementID>(parent_elem)) {
                    eid = reg.get<Component::ElementID>(parent_elem).value;
                } else if (reg.all_of<Component::OriginalID>(parent_elem)) {
                    eid = reg.get<Component::OriginalID>(parent_elem).value;
                }
                if (eid >= 0 && insp.eid_to_part.count(eid)) {
                    unique_parts.insert(insp.eid_to_part.at(eid));
                }
            }
        }

        parts.assign(unique_parts.begin(), unique_parts.end());
        return parts;
    }
};