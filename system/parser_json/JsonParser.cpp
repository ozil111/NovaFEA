// system/parser_json/JsonParser.cpp
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#include "parser_json/JsonParser.h"
#include "components/mesh_components.h"
#include "components/material_components.h"
#include "components/property_components.h"
#include "components/simdroid_components.h"
#include "components/load_components.h"
#include "components/analysis_component.h"
#include "nlohmann/json.hpp"
#include "spdlog/spdlog.h"
#include <fstream>
#include <stdexcept>
#include "JsonParser.h"

using json = nlohmann::json;

// ============================================================================
// 主解析入�?
// ============================================================================
bool JsonParser::parse(const std::string& filepath, DataContext& data_context) {
    spdlog::debug("JsonParser started for file: {}", filepath);

    // 1. 加载 JSON 文件
    std::ifstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("JsonParser could not open file: {}", filepath);
        return false;
    }

    json j;
    try {
        // 启用 nlohmann::json 的注释支�?
        // 注意：需�?nlohmann::json 3.10.0+ 版本
        j = json::parse(file, nullptr, true, true);  // 最后一个参数启用注释忽�?
    } catch (const json::exception& e) {
        spdlog::error("JSON parsing error: {}", e.what());
        return false;
    }

    // 2. 清空 DataContext，从干净状态开�?
    data_context.clear();
    auto& registry = data_context.registry;

    // 3. 准备所有的 ID -> entity 映射�?
    std::unordered_map<int, entt::entity> material_id_map;
    std::unordered_map<int, entt::entity> property_id_map;
    std::unordered_map<int, entt::entity> node_id_map;
    std::unordered_map<int, entt::entity> element_id_map;
    std::unordered_map<int, entt::entity> nodeset_id_map;
    std::unordered_map<int, entt::entity> eleset_id_map;
    std::unordered_map<int, entt::entity> load_id_map;
    std::unordered_map<int, entt::entity> boundary_id_map;
    std::unordered_map<int, entt::entity> curve_id_map;
    std::unordered_map<int, entt::entity> analysis_id_map;
    std::unordered_map<int, entt::entity> output_id_map;

    // 4. 按照严格的依赖顺序执�?N-Step 解析
    try {
        // 步骤 1: Material (无依�?
        if (j.contains("material")) {
            parse_materials(j, registry, material_id_map);
        }

        // 步骤 2: Property (依赖 Material)
        std::unordered_map<int, entt::entity> property_id_to_material;
        if (j.contains("property")) {
            parse_properties(j, registry, material_id_map, property_id_map, property_id_to_material);
        }

        // 步骤 3: Node (无依�?
        if (j.contains("mesh") && j["mesh"].contains("nodes")) {
            parse_nodes(j, registry, node_id_map);
        }

        // 步骤 4: Element (依赖 Node, Property)
        if (j.contains("mesh") && j["mesh"].contains("elements")) {
            parse_elements(j, registry, node_id_map, property_id_map, element_id_map);
        }

        // 步骤 4.5: �?Property 构建 SimdroidPart 与单元集（材料通过 Part 绑定�?
        if (j.contains("property") && j.contains("mesh") && j["mesh"].contains("elements")) {
            build_parts_from_properties(registry, property_id_map, property_id_to_material, element_id_map);
        }

        // 步骤 5: NodeSet (依赖 Node)
        if (j.contains("nodeset")) {
            parse_nodesets(j, registry, node_id_map, nodeset_id_map);
        }

        // 步骤 6: EleSet (依赖 Element)
        if (j.contains("eleset")) {
            parse_elesets(j, registry, element_id_map, eleset_id_map);
        }

        // 步骤 6.5: Curve (无依赖，需要在Load之前解析)
        if (j.contains("curve")) {
            parse_curves(j, registry, curve_id_map);
        }

        // 步骤 7: Load (依赖 Curve)
        if (j.contains("load")) {
            parse_loads(j, registry, load_id_map, curve_id_map);
        }

        // 步骤 8: Boundary (无依�?
        if (j.contains("boundary")) {
            parse_boundaries(j, registry, boundary_id_map);
        }

        // 步骤 9: 应用 Load (依赖 Load, NodeSet)
        if (j.contains("load")) {
            apply_loads(j, registry, load_id_map, nodeset_id_map);
        }

        // 步骤 10: 应用 Boundary (依赖 Boundary, NodeSet)
        if (j.contains("boundary")) {
            apply_boundaries(j, registry, boundary_id_map, nodeset_id_map);
        }

        // 步骤 11: 解析 Analysis (无依赖，但应在最后解�?
        if (j.contains("analysis") && j["analysis"].is_array() && !j["analysis"].empty()) {
            parse_analysis(j, registry, analysis_id_map);
            // 取第一个分析配置对应的 entity 同步�?DataContext
            const auto& analysis_config = j["analysis"][0];
            if (analysis_config.contains("aid") && analysis_config["aid"].is_number_integer()) {
                int first_aid = analysis_config["aid"].get<int>();
                auto it = analysis_id_map.find(first_aid);
                if (it != analysis_id_map.end()) {
                    data_context.analysis_entity = it->second;
                    spdlog::info("Analysis entity set (aid={}).", first_aid);
                }
            }
        } else {
            spdlog::debug("No 'analysis' field found, defaulting to 'static' analysis");
        }

        // 步骤 12: 解析 Output (无依赖，但应在最后解�?
        if (j.contains("output")) {
            parse_output(j, registry, output_id_map);
            auto it = output_id_map.find(0);
            if (it != output_id_map.end()) {
                data_context.output_entity = it->second;
                spdlog::info("Output entity set.");
            }
        }

    } catch (const std::exception& e) {
        spdlog::error("A critical parsing error occurred: {}", e.what());
        return false;
    }

    // 5. 统计并报�?
    auto node_count = registry.view<Component::Position>().size();
    auto element_count = registry.view<Component::Connectivity>().size();
    auto material_count = material_id_map.size();
    auto property_count = property_id_map.size();
    
    spdlog::info("JsonParser finished. Materials: {}, Properties: {}, Nodes: {}, Elements: {}", 
                 material_count, property_count, node_count, element_count);
    
    return true;
}

// ============================================================================
// 步骤 1: 解析 Material
// ============================================================================
void JsonParser::parse_materials(
    const json& j,
    entt::registry& registry,
    std::unordered_map<int, entt::entity>& material_id_map
) {
    spdlog::debug("--> Parsing Materials...");
    
    for (const auto& mat : j["material"]) {
        int mid = mat["mid"];
        int type_id = mat["typeid"];

        // 检查重�?ID
        if (material_id_map.count(mid)) {
            spdlog::warn("Duplicate material ID {}. Skipping.", mid);
            continue;
        }

        entt::entity e = registry.create();
        registry.emplace<Component::MaterialID>(e, mid);

        // 根据 type_id 附加不同的参数组�?
        switch (type_id) {
            case 1: { // 线弹性材�?
                Component::LinearElasticParams params;
                params.rho = mat["rho"];
                params.E = mat["E"];
                params.nu = mat["nu"];
                registry.emplace<Component::LinearElasticParams>(e, params);
                spdlog::debug("  Created LinearElastic Material {}: E={}, nu={}", mid, params.E, params.nu);
                break;
            }
            // 未来可以添加其他材料类型
            // case 101: { /* Polynomial */ break; }
            // case 102: { /* ReducedPolynomial */ break; }
            // case 103: { /* Ogden */ break; }
            default:
                spdlog::warn("Unknown material typeid: {}. Skipping parameters.", type_id);
                break;
        }

        material_id_map[mid] = e;
    }

    spdlog::debug("<-- Materials parsed: {} entities created.", material_id_map.size());
}

// ============================================================================
// 步骤 2: 解析 Property
// ============================================================================
void JsonParser::parse_properties(
    const json& j,
    entt::registry& registry,
    const std::unordered_map<int, entt::entity>& material_id_map,
    std::unordered_map<int, entt::entity>& property_id_map,
    std::unordered_map<int, entt::entity>& property_id_to_material
) {
    spdlog::debug("--> Parsing Properties...");

    for (const auto& prop : j["property"]) {
        int pid = prop["pid"];
        int mid = prop["mid"];
        int type_id = prop["typeid"];

        // 检查重�?ID
        if (property_id_map.count(pid)) {
            spdlog::warn("Duplicate property ID {}. Skipping.", pid);
            continue;
        }

        // 检查引用的 Material 是否存在
        auto mat_it = material_id_map.find(mid);
        if (mat_it == material_id_map.end()) {
            spdlog::error("Property {} references undefined Material ID {}. Skipping.", pid, mid);
            continue;
        }

        entt::entity e = registry.create();
        registry.emplace<Component::PropertyID>(e, pid);

        // 根据 type_id 附加不同的属性组�?
        switch (type_id) {
            case 1: { // 固体单元属�?
                Component::SolidProperty solid_prop;
                solid_prop.type_id = type_id;
                solid_prop.integration_network = prop["integration_network"];
                solid_prop.hourglass_control = prop["hourglass_control"];
                registry.emplace<Component::SolidProperty>(e, solid_prop);
                spdlog::debug("  Created SolidProperty {}: integration={}, hourglass={}", 
                              pid, solid_prop.integration_network, solid_prop.hourglass_control);
                break;
            }
            // 未来可以添加其他属性类�?
            // case 2: { /* Shell Property */ break; }
            default:
                spdlog::warn("Unknown property typeid: {}. Skipping parameters.", type_id);
                break;
        }

        // 材料通过 SimdroidPart 绑定，此处仅记录 pid -> material 供后续创�?Part 使用
        property_id_to_material[pid] = mat_it->second;

        property_id_map[pid] = e;
    }

    spdlog::debug("<-- Properties parsed: {} entities created.", property_id_map.size());
}

// ============================================================================
// 步骤 3: 解析 Node
// ============================================================================
void JsonParser::parse_nodes(
    const json& j,
    entt::registry& registry,
    std::unordered_map<int, entt::entity>& node_id_map
) {
    spdlog::debug("--> Parsing Nodes...");

    for (const auto& node : j["mesh"]["nodes"]) {
        int nid = node["nid"];

        // 检查重�?ID
        if (node_id_map.count(nid)) {
            spdlog::warn("Duplicate node ID {}. Skipping.", nid);
            continue;
        }

        entt::entity e = registry.create();
        registry.emplace<Component::NodeID>(e, nid);
        
        Component::Position pos;
        pos.x = node["x"];
        pos.y = node["y"];
        pos.z = node["z"];
        registry.emplace<Component::Position>(e, pos);

        node_id_map[nid] = e;
    }

    spdlog::debug("<-- Nodes parsed: {} entities created.", node_id_map.size());
}

// ============================================================================
// 步骤 4: 解析 Element
// ============================================================================
void JsonParser::parse_elements(
    const json& j,
    entt::registry& registry,
    const std::unordered_map<int, entt::entity>& node_id_map,
    const std::unordered_map<int, entt::entity>& property_id_map,
    std::unordered_map<int, entt::entity>& element_id_map
) {
    spdlog::debug("--> Parsing Elements...");

    for (const auto& elem : j["mesh"]["elements"]) {
        int eid = elem["eid"];
        int etype = elem["etype"];
        int pid = elem["pid"];

        // 检查重�?ID
        if (element_id_map.count(eid)) {
            spdlog::warn("Duplicate element ID {}. Skipping.", eid);
            continue;
        }

        // 检查引用的 Property 是否存在
        auto prop_it = property_id_map.find(pid);
        if (prop_it == property_id_map.end()) {
            spdlog::error("Element {} references undefined Property ID {}. Skipping.", eid, pid);
            continue;
        }

        entt::entity e = registry.create();
        registry.emplace<Component::ElementID>(e, eid);
        registry.emplace<Component::ElementType>(e, etype);

        // 建立连接�?
        auto& conn = registry.emplace<Component::Connectivity>(e);
        for (int nid : elem["nids"]) {
            auto node_it = node_id_map.find(nid);
            if (node_it == node_id_map.end()) {
                spdlog::error("Element {} references undefined Node ID {}.", eid, nid);
                throw std::runtime_error("Element references undefined node");
            }
            conn.nodes.push_back(node_it->second);
        }

        // 建立�?Property 的引用（核心！）
        registry.emplace<Component::PropertyRef>(e, prop_it->second);

        element_id_map[eid] = e;
    }

    spdlog::debug("<-- Elements parsed: {} entities created.", element_id_map.size());
}

// ============================================================================
// 步骤 4.5: �?Property 构建 SimdroidPart 与单元集
// ============================================================================
void JsonParser::build_parts_from_properties(
    entt::registry& registry,
    const std::unordered_map<int, entt::entity>& property_id_map,
    const std::unordered_map<int, entt::entity>& property_id_to_material,
    const std::unordered_map<int, entt::entity>& element_id_map
) {
    spdlog::debug("--> Building SimdroidPart from properties...");

    for (const auto& [pid, section_entity] : property_id_map) {
        auto mat_it = property_id_to_material.find(pid);
        if (mat_it == property_id_to_material.end()) {
            continue;
        }
        entt::entity material_entity = mat_it->second;

        // 收集使用�?pid 的所有单�?
        std::vector<entt::entity> members;
        for (const auto& [eid, element_entity] : element_id_map) {
            if (!registry.all_of<Component::PropertyRef>(element_entity)) {
                continue;
            }
            entt::entity prop_entity = registry.get<Component::PropertyRef>(element_entity).property_entity;
            if (!registry.all_of<Component::PropertyID>(prop_entity)) {
                continue;
            }
            if (registry.get<Component::PropertyID>(prop_entity).value == pid) {
                members.push_back(element_entity);
            }
        }

        if (members.empty()) {
            continue;
        }

        // 创建单元�?
        entt::entity ele_set_entity = registry.create();
        registry.emplace<Component::SetName>(ele_set_entity, "Part_pid_" + std::to_string(pid));
        auto& set_members = registry.emplace<Component::ElementSetMembers>(ele_set_entity);
        set_members.members = std::move(members);

        // 创建 Part（绑�?几何/截面/材料�?
        Component::SimdroidPart part;
        part.name = "Part_pid_" + std::to_string(pid);
        part.element_set = ele_set_entity;
        part.material = material_entity;
        part.section = section_entity;
        entt::entity part_entity = registry.create();
        registry.emplace<Component::SimdroidPart>(part_entity, std::move(part));
    }

    spdlog::debug("<-- SimdroidPart build done.");
}

// ============================================================================
// 步骤 5: 解析 NodeSet
// ============================================================================
void JsonParser::parse_nodesets(
    const json& j,
    entt::registry& registry,
    const std::unordered_map<int, entt::entity>& node_id_map,
    std::unordered_map<int, entt::entity>& nodeset_id_map
) {
    spdlog::debug("--> Parsing NodeSets...");

    for (const auto& nset : j["nodeset"]) {
        int nsid = nset["nsid"];
        std::string name = nset["name"];

        // 检查重�?ID
        if (nodeset_id_map.count(nsid)) {
            spdlog::warn("Duplicate nodeset ID {}. Skipping.", nsid);
            continue;
        }

        entt::entity e = registry.create();
        registry.emplace<Component::NodeSetID>(e, nsid);
        registry.emplace<Component::SetName>(e, name);

        // 建立 NodeSetMembers
        auto& members = registry.emplace<Component::NodeSetMembers>(e);
        for (int nid : nset["nids"]) {
            auto node_it = node_id_map.find(nid);
            if (node_it == node_id_map.end()) {
                spdlog::warn("NodeSet '{}' references undefined Node ID {}.", name, nid);
                continue;
            }
            members.members.push_back(node_it->second);
        }

        nodeset_id_map[nsid] = e;
        spdlog::debug("  Created NodeSet '{}' with {} members.", name, members.members.size());
    }

    spdlog::debug("<-- NodeSets parsed: {} entities created.", nodeset_id_map.size());
}

// ============================================================================
// 步骤 6: 解析 EleSet
// ============================================================================
void JsonParser::parse_elesets(
    const json& j,
    entt::registry& registry,
    const std::unordered_map<int, entt::entity>& element_id_map,
    std::unordered_map<int, entt::entity>& eleset_id_map
) {
    spdlog::debug("--> Parsing EleSets...");

    for (const auto& eset : j["eleset"]) {
        int esid = eset["esid"];
        std::string name = eset["name"];

        // 检查重�?ID
        if (eleset_id_map.count(esid)) {
            spdlog::warn("Duplicate eleset ID {}. Skipping.", esid);
            continue;
        }

        entt::entity e = registry.create();
        registry.emplace<Component::EleSetID>(e, esid);
        registry.emplace<Component::SetName>(e, name);

        // 建立 ElementSetMembers
        auto& members = registry.emplace<Component::ElementSetMembers>(e);
        for (int eid : eset["eids"]) {
            auto elem_it = element_id_map.find(eid);
            if (elem_it == element_id_map.end()) {
                spdlog::warn("EleSet '{}' references undefined Element ID {}.", name, eid);
                continue;
            }
            members.members.push_back(elem_it->second);
        }

        eleset_id_map[esid] = e;
        spdlog::debug("  Created EleSet '{}' with {} members.", name, members.members.size());
    }

    spdlog::debug("<-- EleSets parsed: {} entities created.", eleset_id_map.size());
}

// ============================================================================
// 步骤 6.5: 解析 Curve（曲线定义）
// ============================================================================
void JsonParser::parse_curves(
    const json& j,
    entt::registry& registry,
    std::unordered_map<int, entt::entity>& curve_id_map
) {
    spdlog::debug("--> Parsing Curves...");

    for (const auto& curve : j["curve"]) {
        int cid = curve["cid"];
        std::string type = curve["type"];

        // 检查重�?ID
        if (curve_id_map.count(cid)) {
            spdlog::warn("Duplicate curve ID {}. Skipping.", cid);
            continue;
        }

        entt::entity e = registry.create();
        registry.emplace<Component::CurveID>(e, cid);

        Component::Curve curve_data;
        curve_data.type = type;
        
        // 解析x和y数组
        if (curve.contains("x") && curve["x"].is_array()) {
            for (const auto& x_val : curve["x"]) {
                curve_data.x.push_back(x_val.get<double>());
            }
        }
        if (curve.contains("y") && curve["y"].is_array()) {
            for (const auto& y_val : curve["y"]) {
                curve_data.y.push_back(y_val.get<double>());
            }
        }

        // 验证数组长度
        if (curve_data.x.size() != curve_data.y.size()) {
            spdlog::warn("Curve {} has mismatched x/y array sizes. Skipping.", cid);
            registry.destroy(e);
            continue;
        }

        if (curve_data.x.empty()) {
            spdlog::warn("Curve {} has empty data. Skipping.", cid);
            registry.destroy(e);
            continue;
        }

        registry.emplace<Component::Curve>(e, curve_data);
        curve_id_map[cid] = e;
        spdlog::debug("  Created Curve {}: type={}, points={}", 
                      cid, type, curve_data.x.size());
    }

    spdlog::debug("<-- Curves parsed: {} entities created.", curve_id_map.size());
}

// ============================================================================
// 步骤 7: 解析 Load（抽象定义）
// ============================================================================
void JsonParser::parse_loads(
    const json& j,
    entt::registry& registry,
    std::unordered_map<int, entt::entity>& load_id_map,
    std::unordered_map<int, entt::entity>& curve_id_map
) {
    spdlog::debug("--> Parsing Loads...");

    for (const auto& load : j["load"]) {
        int lid = load["lid"];
        int type_id = load["typeid"];

        // 检查重�?ID
        if (load_id_map.count(lid)) {
            spdlog::warn("Duplicate load ID {}. Skipping.", lid);
            continue;
        }

        entt::entity e = registry.create();
        registry.emplace<Component::LoadID>(e, lid);

        // 根据 type_id 附加不同的载荷组�?
        switch (type_id) {
            case 1: { // 节点载荷
                Component::NodalLoad nodal_load;
                nodal_load.type_id = type_id;
                nodal_load.dof = load["dof"];
                nodal_load.value = load["value"];
                // curve_entity 在下方解�?curve �?patch 写入
                registry.emplace<Component::NodalLoad>(e, nodal_load);
                spdlog::debug("  Created NodalLoad {}: dof={}, value={}",
                              lid, nodal_load.dof, nodal_load.value);
                break;
            }
            // 未来可以添加其他载荷类型
            // case 2: { /* Pressure Load */ break; }
            default:
                spdlog::warn("Unknown load typeid: {}. Skipping parameters.", type_id);
                break;
        }

        // 解析curve字段：如果未指定，使用默认curve (cid=0)
        entt::entity curve_entity = entt::null;
        
        if (load.contains("curve") && !load["curve"].is_null()) {
            // 使用指定的curve
            int curve_id = load["curve"];
            auto curve_it = curve_id_map.find(curve_id);
            if (curve_it != curve_id_map.end()) {
                curve_entity = curve_it->second;
                spdlog::debug("  Load {} linked to Curve {}", lid, curve_id);
            } else {
                spdlog::warn("Load {} references undefined Curve ID {}. Ignoring curve.", lid, curve_id);
            }
        }
        
        // 如果没有指定curve或指定的curve不存在，使用默认curve (cid=0)
        if (curve_entity == entt::null) {
            auto default_curve_it = curve_id_map.find(0);
            if (default_curve_it != curve_id_map.end()) {
                // 默认curve已存在，使用�?
                curve_entity = default_curve_it->second;
                spdlog::debug("  Load {} using default Curve 0", lid);
            } else {
                // 创建默认curve: {"cid":0,"type":"linear","x":[0.0,1.0],"y":[0.0,1.0]}
                entt::entity default_curve = registry.create();
                registry.emplace<Component::CurveID>(default_curve, 0);
                
                Component::Curve default_curve_data;
                default_curve_data.type = "linear";
                default_curve_data.x = {0.0, 1.0};
                default_curve_data.y = {1.0, 1.0};
                registry.emplace<Component::Curve>(default_curve, default_curve_data);
                
                curve_id_map[0] = default_curve;
                curve_entity = default_curve;
                spdlog::debug("  Created default Curve 0 for Load {}", lid);
            }
        }
        
        // �?curve 写入 NodalLoad.curve_entity（仅节点载荷�?
        if (curve_entity != entt::null && registry.all_of<Component::NodalLoad>(e)) {
            registry.patch<Component::NodalLoad>(e, [curve_entity](auto& nl) { nl.curve_entity = curve_entity; });
        }

        load_id_map[lid] = e;
    }

    spdlog::debug("<-- Loads parsed: {} entities created.", load_id_map.size());
}

// ============================================================================
// 步骤 8: 解析 Boundary（抽象定义）
// ============================================================================
void JsonParser::parse_boundaries(
    const json& j,
    entt::registry& registry,
    std::unordered_map<int, entt::entity>& boundary_id_map
) {
    spdlog::debug("--> Parsing Boundaries...");

    for (const auto& bnd : j["boundary"]) {
        int bid = bnd["bid"];
        int type_id = bnd["typeid"];

        // 检查重�?ID
        if (boundary_id_map.count(bid)) {
            spdlog::warn("Duplicate boundary ID {}. Skipping.", bid);
            continue;
        }

        entt::entity e = registry.create();
        registry.emplace<Component::BoundaryID>(e, bid);

        // 根据 type_id 附加不同的边界组�?
        switch (type_id) {
            case 1: { // 单点约束 (SPC)
                Component::BoundarySPC spc;
                spc.type_id = type_id;
                spc.dof = bnd["dof"];
                spc.value = bnd["value"];
                registry.emplace<Component::BoundarySPC>(e, spc);
                spdlog::debug("  Created BoundarySPC {}: dof={}, value={}", 
                              bid, spc.dof, spc.value);
                break;
            }
            // 未来可以添加其他边界类型
            default:
                spdlog::warn("Unknown boundary typeid: {}. Skipping parameters.", type_id);
                break;
        }

        boundary_id_map[bid] = e;
    }

    spdlog::debug("<-- Boundaries parsed: {} entities created.", boundary_id_map.size());
}

// ============================================================================
// 步骤 9: 应用 Load �?Node（建立引用关系）
// ============================================================================
void JsonParser::apply_loads(
    const json& j,
    entt::registry& registry,
    const std::unordered_map<int, entt::entity>& load_id_map,
    const std::unordered_map<int, entt::entity>& nodeset_id_map
) {
    spdlog::debug("--> Applying Loads to Nodes...");

    for (const auto& load : j["load"]) {
        int lid = load["lid"];
        int nsid = load["nsid"];

        // 1. 找到 Load 实体
        auto load_it = load_id_map.find(lid);
        if (load_it == load_id_map.end()) {
            spdlog::error("Load application references undefined Load ID {}.", lid);
            continue;
        }

        // 2. 找到 NodeSet 实体
        auto nodeset_it = nodeset_id_map.find(nsid);
        if (nodeset_it == nodeset_id_map.end()) {
            spdlog::error("Load {} references undefined NodeSet ID {}.", lid, nsid);
            continue;
        }

        // 3. 获取�?Set 的所�?Node 成员
        const auto& members = registry.get<Component::NodeSetMembers>(nodeset_it->second);

        // 4. �?Load 引用附加到每�?Node 实体上（核心！）
        for (entt::entity node_e : members.members) {
            // 允许一个节点应用多个载荷（1-to-Many�?
            auto& applied = registry.get_or_emplace<Component::AppliedLoadRef>(node_e);
            applied.load_entities.push_back(load_it->second);
        }

        spdlog::debug("  Applied Load {} to {} nodes.", lid, members.members.size());
    }

    spdlog::debug("<-- Load application complete.");
}

// ============================================================================
// 步骤 10: 应用 Boundary �?Node（建立引用关系）
// ============================================================================
void JsonParser::apply_boundaries(
    const json& j,
    entt::registry& registry,
    const std::unordered_map<int, entt::entity>& boundary_id_map,
    const std::unordered_map<int, entt::entity>& nodeset_id_map
) {
    spdlog::debug("--> Applying Boundaries to Nodes...");

    for (const auto& bnd : j["boundary"]) {
        int bid = bnd["bid"];
        int nsid = bnd["nsid"];

        // 1. 找到 Boundary 实体
        auto bnd_it = boundary_id_map.find(bid);
        if (bnd_it == boundary_id_map.end()) {
            spdlog::error("Boundary application references undefined Boundary ID {}.", bid);
            continue;
        }

        // 2. 找到 NodeSet 实体
        auto nodeset_it = nodeset_id_map.find(nsid);
        if (nodeset_it == nodeset_id_map.end()) {
            spdlog::error("Boundary {} references undefined NodeSet ID {}.", bid, nsid);
            continue;
        }

        // 3. 获取�?Set 的所�?Node 成员
        const auto& members = registry.get<Component::NodeSetMembers>(nodeset_it->second);

        // 4. �?Boundary 引用附加到每�?Node 实体上（核心！）
        for (entt::entity node_e : members.members) {
            auto& applied = registry.get_or_emplace<Component::AppliedBoundaryRef>(node_e);
            applied.boundary_entities.push_back(bnd_it->second);
        }

        spdlog::debug("  Applied Boundary {} to {} nodes.", bid, members.members.size());
    }

    spdlog::debug("<-- Boundary application complete.");
}

// ============================================================================
// 步骤 11: 解析 Analysis 实体
// ============================================================================
void JsonParser::parse_analysis(const nlohmann::json& j, entt::registry& registry, std::unordered_map<int, entt::entity>& analysis_id_map)
{
    spdlog::debug("--> Parsing Analysis...");

    for (const auto& a : j["analysis"]) {
        int aid = a["aid"];
        if (analysis_id_map.count(aid)) {
            spdlog::warn("Duplicate analysis ID {}. Skipping.", aid);
            continue;
        }

        entt::entity e = registry.create();
        registry.emplace<Component::AnalysisID>(e, aid);

        std::string analysis_type_str = a.contains("analysis_type") && a["analysis_type"].is_string()
            ? a["analysis_type"].get<std::string>() : "static";
        registry.emplace<Component::AnalysisType>(e, analysis_type_str);

        if (a.contains("endtime") && a["endtime"].is_number()) {
            registry.emplace<Component::EndTime>(e, a["endtime"].get<double>());
        }
        if (a.contains("fixed_time_step") && a["fixed_time_step"].is_number()) {
            registry.emplace<Component::FixedTimeStep>(e, a["fixed_time_step"].get<double>());
        }

        analysis_id_map[aid] = e;
        spdlog::debug("  Created Analysis {}: type={}", aid, analysis_type_str);
    }

    spdlog::debug("<-- Analysis parsed: {} entities created.", analysis_id_map.size());
}

void JsonParser::parse_output(const nlohmann::json &j, entt::registry &registry, std::unordered_map<int, entt::entity> &output_id_map)
{
    spdlog::debug("--> Parsing Output...");

    const auto& o = j["output"];
    entt::entity e = registry.create();

    if (o.contains("node_output") && o["node_output"].is_array()) {
        std::vector<std::string> node_out;
        for (const auto& s : o["node_output"]) {
            if (s.is_string())
                node_out.push_back(s.get<std::string>());
        }
        registry.emplace<Component::NodeOutput>(e, Component::NodeOutput{std::move(node_out)});
    }
    if (o.contains("element_output") && o["element_output"].is_array()) {
        std::vector<std::string> elem_out;
        for (const auto& s : o["element_output"]) {
            if (s.is_string())
                elem_out.push_back(s.get<std::string>());
        }
        registry.emplace<Component::ElementOutput>(e, Component::ElementOutput{std::move(elem_out)});
    }
    if (o.contains("interval_time") && o["interval_time"].is_number()) {
        registry.emplace<Component::OutputIntervalTime>(e, o["interval_time"].get<double>());
    }

    output_id_map[0] = e;
    spdlog::debug("  Created Output (single global)");
    spdlog::debug("<-- Output parsed: 1 entity created.");
}
