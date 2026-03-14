// BlockHandler.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include "entt/entt.hpp"
#include "GenericLineParser.h"
#include "string_utils.h"
#include "components/mesh_components.h"
#include <functional>
#include <map>
#include <unordered_map>
#include <fstream>

namespace Parser {

// BlockHandler的基类，以便在FemParser中统一管理
class IBlockHandler {
public:
    virtual ~IBlockHandler() = default;
    virtual void process(std::ifstream& file) = 0;
};

/**
 * @class BlockHandler
 * @brief 声明式的块处理器 - 框架的核�?
 * @details 通过声明字段索引到组件成员的映射，自动完成：
 *   1. 从文件读取行
 *   2. 解析字段
 *   3. 创建实体
 *   4. 附加组件
 * 
 * 使用示例:
 *   auto handler = std::make_unique<BlockHandler>(registry);
 *   handler->map<Component::OriginalID>(0, [](auto& comp, const Field& f){ 
 *       comp.value = std::get<int>(f); 
 *   });
 */
class BlockHandler : public IBlockHandler {
public:
    explicit BlockHandler(entt::registry& reg) : registry(reg) {}

    /**
     * @brief 声明一个字段如何映射到一个组�?
     * @tparam ComponentType 要映射的组件类型
     * @param field_index 字段在行中的索引（从0开始）
     * @param setter Lambda函数，接收组件引用和字段值，负责赋�?
     */
    template<typename ComponentType>
    void map(int field_index, std::function<void(ComponentType&, const Field&)> setter) {
        // 创建一个包装函�?
        auto wrapper = [setter](entt::registry& reg, entt::entity e, const Field& f) {
            // 如果组件不存在，则先创建
            auto& component = reg.get_or_emplace<ComponentType>(e);
            setter(component, f);
        };
        field_mappers[field_index] = wrapper;
    }

    /**
     * @brief 主处理逻辑 - 读取块内所有行并创建实�?
     * @param file 输入文件�?
     */
    void process(std::ifstream& file) override {
        std::string line;
        while (get_logical_line(file, line)) {
            // �?*end 关键字结�?
            if (line.find(" end") != std::string::npos) {
                break;
            }
            if (line.empty()) continue;

            try {
                auto fields = parse_line_to_fields(line);
                const entt::entity entity = registry.create();

                // 应用所有已配置的字段映�?
                for (const auto& [index, mapper_func] : field_mappers) {
                    if (index < static_cast<int>(fields.size())) {
                        mapper_func(registry, entity, fields[index]);
                    } else {
                        spdlog::warn("Missing field at index {} for line: {}", index, line);
                    }
                }
                
                // 调用后处理钩子（如果有）
                if (post_entity_hook) {
                    post_entity_hook(entity, fields);
                }

            } catch (const std::exception& e) {
                spdlog::warn("Skipping line due to error: '{}'. Details: {}", line, e.what());
            }
        }
    }

    /**
     * @brief 设置实体创建后的钩子函数
     * @details 用于处理需要额外逻辑的情况，如建立ID映射�?
     * @param hook Lambda函数，接收创建的实体和解析的字段
     */
    void set_post_entity_hook(std::function<void(entt::entity, const std::vector<Field>&)> hook) {
        post_entity_hook = hook;
    }

private:
    entt::registry& registry;
    // 存储�?字段索引 -> 组件设置逻辑 的映�?
    std::map<int, std::function<void(entt::registry&, entt::entity, const Field&)>> field_mappers;
    // 后处理钩�?
    std::function<void(entt::entity, const std::vector<Field>&)> post_entity_hook;
};

/**
 * @class SetBlockHandler
 * @brief 专门用于处理集合（Set）的块处理器
 * @details 集合需要特殊处理，因为它们引用其他实体的ID，需要ID->entity映射
 */
class SetBlockHandler : public IBlockHandler {
public:
    SetBlockHandler(
        entt::registry& reg,
        std::unordered_map<int, entt::entity>& id_map,
        bool is_node_set
    ) : registry(reg), id_to_entity_map(id_map), is_node_set(is_node_set) {}

    void process(std::ifstream& file) override {
        std::string line;
        spdlog::debug("--> Entering {} SetBlockHandler", is_node_set ? "Node" : "Element");

        while (get_logical_line(file, line)) {
            if (line.find(" end") != std::string::npos) {
                break;
            }
            if (line.empty()) continue;

            try {
                auto fields = parse_line_to_fields(line);
                
                if (fields.size() < 3) {
                    spdlog::warn("Set line has insufficient fields: {}", line);
                    continue;
                }

                // 字段0: ID (可忽�?
                // 字段1: Set Name
                // 字段2: Member IDs (vector<int>)
                
                std::string set_name = std::get<std::string>(fields[1]);
                const auto& member_ids = std::get<std::vector<int>>(fields[2]);

                // 检查集合是否已存在
                entt::entity set_entity = find_set_by_name(set_name);
                if (set_entity == entt::null) {
                    // 创建新集合实�?
                    set_entity = registry.create();
                    registry.emplace<Component::SetName>(set_entity, set_name);
                    
                    if (is_node_set) {
                        registry.emplace<Component::NodeSetMembers>(set_entity);
                    } else {
                        registry.emplace<Component::ElementSetMembers>(set_entity);
                    }
                    
                    spdlog::debug("Created new {} set: '{}'", 
                                  is_node_set ? "node" : "element", set_name);
                } else {
                    spdlog::warn("{} set '{}' already exists. Appending members.", 
                                 is_node_set ? "Node" : "Element", set_name);
                }

                // 转换ID到entity并添加到集合
                if (is_node_set) {
                    auto& members = registry.get<Component::NodeSetMembers>(set_entity);
                    for (int id : member_ids) {
                        auto it = id_to_entity_map.find(id);
                        if (it != id_to_entity_map.end()) {
                            members.members.push_back(it->second);
                        } else {
                            spdlog::warn("Node set '{}' references undefined node ID: {}", 
                                         set_name, id);
                        }
                    }
                } else {
                    auto& members = registry.get<Component::ElementSetMembers>(set_entity);
                    for (int id : member_ids) {
                        auto it = id_to_entity_map.find(id);
                        if (it != id_to_entity_map.end()) {
                            members.members.push_back(it->second);
                        } else {
                            spdlog::warn("Element set '{}' references undefined element ID: {}", 
                                         set_name, id);
                        }
                    }
                }

            } catch (const std::exception& e) {
                spdlog::warn("Skipping set line due to error: '{}'. Details: {}", line, e.what());
            }
        }
        
        spdlog::debug("<-- Exiting {} SetBlockHandler", is_node_set ? "Node" : "Element");
    }

private:
    entt::registry& registry;
    std::unordered_map<int, entt::entity>& id_to_entity_map;
    bool is_node_set;

    // 辅助函数：按名称查找集合实体
    entt::entity find_set_by_name(const std::string& set_name) {
        auto view = registry.view<const Component::SetName>();
        for (auto entity : view) {
            if (view.get<const Component::SetName>(entity).value == set_name) {
                return entity;
            }
        }
        return entt::null;
    }
};

/**
 * @class ElementBlockHandler
 * @brief 专门用于处理单元（Element）的块处理器
 * @details 单元需要特殊处理，因为它们的连接性引用节点实�?
 */
class ElementBlockHandler : public IBlockHandler {
public:
    ElementBlockHandler(
        entt::registry& reg,
        std::unordered_map<int, entt::entity>& node_map,
        std::unordered_map<int, entt::entity>& element_map
    ) : registry(reg), node_id_to_entity(node_map), element_id_to_entity(element_map) {}

    void process(std::ifstream& file) override {
        std::string line;
        spdlog::debug("--> Entering ElementBlockHandler");

        while (get_logical_line(file, line)) {
            if (line.find(" end") != std::string::npos) {
                break;
            }
            if (line.empty()) continue;

            try {
                auto fields = parse_line_to_fields(line);
                
                if (fields.size() < 3) {
                    spdlog::warn("Element line has insufficient fields: {}", line);
                    continue;
                }

                // 字段0: Element ID
                // 字段1: Element Type
                // 字段2: Node IDs (vector<int>)
                
                int element_id = std::get<int>(fields[0]);
                int element_type = std::get<int>(fields[1]);
                const auto& node_ids = std::get<std::vector<int>>(fields[2]);

                // 检查重复ID
                if (element_id_to_entity.count(element_id)) {
                    spdlog::warn("Duplicate element ID {}. Skipping.", element_id);
                    continue;
                }

                // 创建单元实体
                const entt::entity element_entity = registry.create();
                
                // 附加组件
                registry.emplace<Component::OriginalID>(element_entity, element_id);
                registry.emplace<Component::ElementType>(element_entity, element_type);
                
                // 构建连接�?
                auto& connectivity = registry.emplace<Component::Connectivity>(element_entity);
                connectivity.nodes.reserve(node_ids.size());
                
                for (int node_id : node_ids) {
                    auto it = node_id_to_entity.find(node_id);
                    if (it == node_id_to_entity.end()) {
                        throw std::runtime_error("Element references undefined node ID: " + 
                                                 std::to_string(node_id));
                    }
                    connectivity.nodes.push_back(it->second);
                }
                
                // 填充外部�?element_id_to_entity 映射
                element_id_to_entity[element_id] = element_entity;

            } catch (const std::exception& e) {
                spdlog::warn("Skipping element line due to error: '{}'. Details: {}", 
                             line, e.what());
            }
        }
        
        spdlog::debug("<-- Exiting ElementBlockHandler. Created {} element entities.", 
                      element_id_to_entity.size());
    }

private:
    entt::registry& registry;
    std::unordered_map<int, entt::entity>& node_id_to_entity;
    std::unordered_map<int, entt::entity>& element_id_to_entity;
};

} // namespace Parser

