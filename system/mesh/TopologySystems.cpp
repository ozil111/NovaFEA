// TopologySystems.cpp
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#include "TopologySystems.h"
#include "components/simdroid_components.h"
#include <algorithm> // for std::sort
#include <stdexcept>
#include <queue>     // for std::queue in flood fill
#include <unordered_set> // for visited tracking
#include "spdlog/spdlog.h"

// -------------------------------------------------------------------
// **System 1: 拓扑提取**
// -------------------------------------------------------------------
void TopologySystems::extract_topology(entt::registry& registry) {
    spdlog::info("TopologySystems: Starting topology extraction...");
    
    // Create a new TopologyData instance
    auto topology_ptr = std::make_unique<TopologyData>();
    TopologyData& topology = *topology_ptr;

    // Get a view of all element entities with Connectivity and ElementType
    auto element_view = registry.view<const Component::Connectivity, const Component::ElementType>();
    
    // Count elements for logging
    size_t element_count = 0;
    for (auto element_entity : element_view) {
        ++element_count;
    }
    spdlog::debug("Processing {} element entities...", element_count);

    // Process each element entity
    for (auto element_entity : element_view) {
        const auto& connectivity = element_view.get<const Component::Connectivity>(element_entity);
        const auto& elem_type = element_view.get<const Component::ElementType>(element_entity);

        // 1. 获取当前单元的节�?(需要转换为外部ID)
        std::vector<NodeID> element_node_ids;
        element_node_ids.reserve(connectivity.nodes.size());
        
        for (entt::entity node_entity : connectivity.nodes) {
            // NOTE: JSON parser uses Component::NodeID, legacy uses Component::OriginalID.
            if (registry.all_of<Component::NodeID>(node_entity)) {
                const auto& nid = registry.get<Component::NodeID>(node_entity);
                element_node_ids.push_back(nid.value);
            } else if (registry.all_of<Component::OriginalID>(node_entity)) {
                const auto& oid = registry.get<Component::OriginalID>(node_entity);
                element_node_ids.push_back(oid.value);
            } else {
                throw std::runtime_error("TopologySystems::extract_topology: node entity missing NodeID/OriginalID");
            }
        }

        // 2. 从单元中提取所有的�?
        auto element_faces = get_faces_from_element(element_node_ids, elem_type.type_id);

        for (auto& face_key : element_faces) {
            // 3. 对面的节点ID排序，创建唯一�?FaceKey
            std::sort(face_key.begin(), face_key.end());

            FaceID face_id;
            // 4. 检查这个面是否已经存在
            auto it = topology.face_key_to_id.find(face_key);
            if (it == topology.face_key_to_id.end()) {
                // 如果是新面：注册它，并分配一个新的FaceID
                face_id = topology.faces.size();
                topology.faces.push_back(face_key);
                topology.face_key_to_id[face_key] = face_id;
                topology.face_to_elements.emplace_back(); // 为新面创建一个空的单元列�?
            } else {
                // 如果是已存在的面：获取它的FaceID
                face_id = it->second;
            }

            // 5. 构建双向关系 - 使用entity而不是索�?
            topology.element_to_faces[element_entity].push_back(face_id);
            topology.face_to_elements[face_id].push_back(element_entity);
        }
    }

    spdlog::info("Topology extraction complete. Found {} unique faces.", topology.faces.size());

    // 填充 Simdroid 映射：element_uid_to_part_map（单�?-> Part�?
    size_t max_eid = 0;
    size_t max_nid = 0;
    for (auto e : registry.view<const Component::ElementID>()) {
        size_t uid = static_cast<size_t>(registry.get<Component::ElementID>(e).value);
        if (uid > max_eid) max_eid = uid;
    }
    for (auto e : registry.view<const Component::NodeID>()) {
        size_t uid = static_cast<size_t>(registry.get<Component::NodeID>(e).value);
        if (uid > max_nid) max_nid = uid;
    }
    if (max_nid == 0) {
        for (auto e : registry.view<const Component::OriginalID>()) {
            size_t uid = static_cast<size_t>(registry.get<Component::OriginalID>(e).value);
            if (uid > max_nid) max_nid = uid;
        }
    }
    topology.reserve_simdroid_maps(max_eid, max_nid);

    auto part_view = registry.view<const Component::SimdroidPart>();
    for (auto part_entity : part_view) {
        const auto& part = part_view.get<const Component::SimdroidPart>(part_entity);
        if (part.element_set == entt::null) continue;
        if (!registry.all_of<Component::ElementSetMembers>(part.element_set)) continue;
        const auto& members = registry.get<Component::ElementSetMembers>(part.element_set).members;
        for (entt::entity elem_entity : members) {
            if (!registry.all_of<Component::ElementID>(elem_entity)) continue;
            ElementID eid = registry.get<Component::ElementID>(elem_entity).value;
            if (eid >= 0 && static_cast<size_t>(eid) < topology.element_uid_to_part_map.size()) {
                topology.element_uid_to_part_map[static_cast<size_t>(eid)] = part_entity;
            }
        }
    }

    // Store the topology data in the registry's context
    registry.ctx().emplace<std::unique_ptr<TopologyData>>(std::move(topology_ptr));
}

// -------------------------------------------------------------------
// **System 2: 连续体查�?(洪水填充)**
// -------------------------------------------------------------------
void TopologySystems::find_continuous_bodies(entt::registry& registry) {
    spdlog::info("TopologySystems: Finding continuous bodies...");
    
    // Get the topology data from the registry context
    if (!registry.ctx().contains<std::unique_ptr<TopologyData>>()) {
        spdlog::error("Topology has not been built. Please run 'extract_topology' first.");
        throw std::runtime_error("TopologyData not found in registry context");
    }

    auto& topology_ptr = registry.ctx().get<std::unique_ptr<TopologyData>>();
    TopologyData& topology = *topology_ptr;

    // Clear previous body data
    topology.element_to_body.clear();
    topology.body_to_elements.clear();
    
    BodyID current_body_id = 0;

    // Get all element entities
    auto element_view = registry.view<const Component::Connectivity>();
    
    // Track which entities have been visited
    std::unordered_set<entt::entity> visited;

    for (auto element_entity : element_view) {
        // If this element hasn't been assigned to any body yet
        if (visited.find(element_entity) == visited.end()) {
            // Start a new flood fill from this element
            std::queue<entt::entity> q;
            q.push(element_entity);
            visited.insert(element_entity);
            topology.element_to_body[element_entity] = current_body_id;

            while (!q.empty()) {
                entt::entity current_elem_entity = q.front();
                q.pop();

                // Add to the body's element list
                topology.body_to_elements[current_body_id].push_back(current_elem_entity);

                // Find all neighbor elements through shared faces
                auto faces_it = topology.element_to_faces.find(current_elem_entity);
                if (faces_it == topology.element_to_faces.end()) {
                    continue; // This element has no faces (shouldn't happen)
                }
                
                const auto& faces_of_current_elem = faces_it->second;
                for (FaceID face_id : faces_of_current_elem) {
                    const auto& elements_sharing_face = topology.face_to_elements[face_id];
                    
                    // Only internal faces (shared by exactly 2 elements) connect neighbors
                    if (elements_sharing_face.size() == 2) {
                        // Find the other element sharing this face
                        entt::entity neighbor_elem_entity = (elements_sharing_face[0] == current_elem_entity)
                                                           ? elements_sharing_face[1]
                                                           : elements_sharing_face[0];
                        
                        // If neighbor hasn't been visited, mark it and add to queue
                        if (visited.find(neighbor_elem_entity) == visited.end()) {
                            visited.insert(neighbor_elem_entity);
                            topology.element_to_body[neighbor_elem_entity] = current_body_id;
                            q.push(neighbor_elem_entity);
                        }
                    }
                }
            }
            current_body_id++; // Prepare a new ID for the next discovered body
        }
    }
    
    spdlog::info("Found {} continuous body/bodies.", topology.body_to_elements.size());
}

// -------------------------------------------------------------------
// **辅助函数: 从单元提取面 (根据单元库定�?**
// -------------------------------------------------------------------
std::vector<FaceKey> TopologySystems::get_faces_from_element(
    const std::vector<NodeID>& nodes, int element_type) {
    
    // 对于梁单元，它们�?D实体，没有“面”的概念，因此返回空列表�?
    // 它们的连接性需要通过其他方式（如节点共享）来定义，而不是面共享�?
    // if (element_type == 102 || element_type == 103) {
    //     return {};
    // }

    switch (element_type) {
        case 102:
            return {{nodes[0]}, {nodes[1]}};
            break;
        case 103:
            return {{nodes[0]}, {nodes[1]}};
            break;
        // --- 2D 单元 (面是�? ---
        case 203: // 3节点平面三角�?
            if (nodes.size() == 3) {
                return { {nodes[0], nodes[1]}, {nodes[1], nodes[2]}, {nodes[2], nodes[0]} };
            }
            break;
        case 204: // 4节点平面四边�?
            if (nodes.size() == 4) {
                return { {nodes[0], nodes[1]}, {nodes[1], nodes[2]}, {nodes[2], nodes[3]}, {nodes[3], nodes[0]} };
            }
            break;
        case 208: // 8节点二阶平面四边�?
            // **关键**: 只使用角点节�?(�?�? 来定义拓扑边界�?
            if (nodes.size() == 8) {
                return { {nodes[0], nodes[1]}, {nodes[1], nodes[2]}, {nodes[2], nodes[3]}, {nodes[3], nodes[0]} };
            }
            break;

        // --- 3D 单元 (面是2D多边�? ---
        case 304: // 4节点四面�?
            if (nodes.size() == 4) {
                return {
                    {nodes[0], nodes[1], nodes[2]},
                    {nodes[0], nodes[3], nodes[1]},
                    {nodes[1], nodes[3], nodes[2]},
                    {nodes[2], nodes[3], nodes[0]}
                };
            }
            break;
        case 306: // 6节点楔形单元 (Penta)
            if (nodes.size() == 6) {
                return {
                    {nodes[0], nodes[1], nodes[2]},          // 底面三角�?
                    {nodes[3], nodes[4], nodes[5]},          // 顶面三角�?
                    {nodes[0], nodes[1], nodes[4], nodes[3]},  // 侧面四边�?1
                    {nodes[1], nodes[2], nodes[5], nodes[4]},  // 侧面四边�?2
                    {nodes[2], nodes[0], nodes[3], nodes[5]}   // 侧面四边�?3
                };
            }
            break;
        case 308: // 8节点六面�?
            if (nodes.size() == 8) {
                return {
                    {nodes[0], nodes[1], nodes[2], nodes[3]}, // 底面
                    {nodes[4], nodes[5], nodes[6], nodes[7]}, // 顶面
                    {nodes[0], nodes[1], nodes[5], nodes[4]}, // 前面
                    {nodes[3], nodes[2], nodes[6], nodes[7]}, // 后面
                    {nodes[0], nodes[3], nodes[7], nodes[4]}, // 左面
                    {nodes[1], nodes[2], nodes[6], nodes[5]}  // 右面
                };
            }
            break;
        case 310: // 10节点二阶四面�?
            // **关键**: 只使用角点节�?(�?�? 来定义拓扑面�?
            if (nodes.size() == 10) {
                 return {
                    {nodes[0], nodes[1], nodes[2]},
                    {nodes[0], nodes[3], nodes[1]},
                    {nodes[1], nodes[3], nodes[2]},
                    {nodes[2], nodes[3], nodes[0]}
                };
            }
            break;
        case 320: // 20节点二阶六面�?
            // **关键**: 只使用角点节�?(�?�? 来定义拓扑面�?
            if (nodes.size() == 20) {
                return {
                    {nodes[0], nodes[1], nodes[2], nodes[3]}, // 底面
                    {nodes[4], nodes[5], nodes[6], nodes[7]}, // 顶面
                    {nodes[0], nodes[1], nodes[5], nodes[4]}, // 前面
                    {nodes[3], nodes[2], nodes[6], nodes[7]}, // 后面
                    {nodes[0], nodes[3], nodes[7], nodes[4]}, // 左面
                    {nodes[1], nodes[2], nodes[6], nodes[5]}  // 右面
                };
            }
            break;

        default:
            // 对于未知的单元类型，打印警告或直接返回空列表
            spdlog::warn("Warning: Unknown element type encountered: {}", element_type);
            return {};
    }

    // 如果因为节点数量不匹配而没有进入if语句，则返回空列�?
    spdlog::warn("Warning: Node count mismatch for element type {}", element_type);
    return {};
}