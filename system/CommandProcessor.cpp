/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 hyperFEM. All rights reserved.
 * Author: Xiaotong Wang (or hyperFEM Team)
 */
#include "CommandProcessor.h"
#include "spdlog/spdlog.h"
#include "parser_base/parserBase.h"
#include "parser_json/JsonParser.h"
#include "exporter_base/exporterBase.h"
#include "DataContext.h"
#include "components/mesh_components.h"
#include "TopologyData.h"
#include "mesh/TopologySystems.h"
#include "parser_simdroid/SimdroidParser.h"
#include "exporter_simdroid/SimdroidExporter.h"
#include "analysis/GraphBuilder.h"
#include "analysis/MermaidReporter.h"
#include "tui/ComponentTUI.h"
#include <filesystem>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <algorithm>

namespace {

// 基础查找辅助：节点 / 单元 / 集合
entt::entity find_node_by_id(entt::registry& registry, int nid) {
    auto view = registry.view<const Component::NodeID>();
    for (auto e : view) {
        if (view.get<const Component::NodeID>(e).value == nid) {
            return e;
        }
    }
    return entt::null;
}

entt::entity find_element_by_id(entt::registry& registry, int eid) {
    auto view = registry.view<const Component::ElementID>();
    for (auto e : view) {
        if (view.get<const Component::ElementID>(e).value == eid) {
            return e;
        }
    }
    return entt::null;
}

entt::entity find_set_by_name(entt::registry& registry, const std::string& name) {
    auto view = registry.view<const Component::SetName>();
    for (auto e : view) {
        if (view.get<const Component::SetName>(e).value == name) {
            return e;
        }
    }
    return entt::null;
}

entt::entity get_or_create_set_entity(entt::registry& registry, const std::string& name) {
    entt::entity e = find_set_by_name(registry, name);
    if (e != entt::null) return e;
    e = registry.create();
    registry.emplace<Component::SetName>(e, name);
    return e;
}

int infer_element_type_from_node_count(std::size_t count) {
    if (count == 2) return 102;   // Line2
    if (count == 3) return 203;   // Tri3
    if (count == 4) return 304;   // Tet4 / Quad4，默认按体单元处理
    if (count == 8) return 308;   // Hex8
    if (count == 10) return 310;  // Tet10
    if (count == 20) return 320;  // Hex20
    return 0;
}

void invalidate_topology_if_needed(AppSession& session) {
    auto& registry = session.data.registry;
    if (registry.ctx().contains<std::unique_ptr<TopologyData>>()) {
        registry.ctx().erase<std::unique_ptr<TopologyData>>();
    }
    session.topology_built = false;
}

void rebuild_inspector_if_mesh_loaded(AppSession& session) {
    if (!session.mesh_loaded) return;
    session.inspector.build(session.data.registry);
}

} // anonymous namespace

void process_command(const std::string& command_line, AppSession& session) {
    std::stringstream ss(command_line);
    std::string command;
    ss >> command;

    if (command == "quit" || command == "exit") {
        session.is_running = false;
        spdlog::info("Exiting hyperFEM. Goodbye!");
    }
    else if (command == "help") {
        spdlog::info("Available commands: import, import_simdroid, export_simdroid, "
                     "info, build_topology, list_bodies, show_body, "
                     "list_parts, delete_part, graph, validate_constraints, list_constraint_warnings, "
                     "panel node <nid>, panel elem <eid>, panel part <name>, panel set <name>, "
                     "node, node_add, node_move, node_delete, "
                     "elem, elem_add, elem_delete, "
                     "list_sets, set_info, set_addnode, set_addelem, set_removenode, set_removeelem, "
                     "save, help, quit");
    }
    else if (command == "import") {
        std::string file_path;
        ss >> file_path;
        if (file_path.empty()) {
            spdlog::error("Usage: import <path_to_file>");
            return;
        }
        
        // 检查文件是否存在
        if (!std::filesystem::exists(file_path)) {
            spdlog::error("File does not exist: {}", file_path);
            return;
        }
        
        session.clear_data();
        spdlog::info("Importing mesh from: {}", file_path);
        
        // 根据文件扩展名自动选择解析器
        std::filesystem::path path(file_path);
        std::string extension = path.extension().string();
        bool parse_success = false;
        
        if (extension == ".json" || extension == ".jsonc") {
            spdlog::info("Detected JSON format, using JsonParser...");
            parse_success = JsonParser::parse(file_path, session.data);
        } else if (extension == ".xfem") {
            spdlog::info("Detected XFEM format, using FemParser (legacy)...");
            parse_success = FemParser::parse(file_path, session.data);
        } else {
            spdlog::error("Unsupported file format: {}. Supported: .json, .jsonc, .xfem", extension);
            return;
        }
        
        if (parse_success) {
            session.mesh_loaded = true;
            // Count entities using views
            auto node_count = session.data.registry.view<Component::Position>().size();
            auto element_count = session.data.registry.view<Component::Connectivity>().size();
            spdlog::info("Successfully imported mesh. {} nodes, {} elements.", node_count, element_count);
        } else {
            spdlog::error("Failed to import mesh from: {}", file_path);
        }
    }
    // =======================================================
    // 新增: Simdroid 导入命令
    // =======================================================
    else if (command == "import_simdroid") {
        std::string control_path_str;
        ss >> control_path_str;
        if (control_path_str.empty()) {
            spdlog::error("Usage: import_simdroid <path_to_control.json>");
            return;
        }

        std::filesystem::path control_path(control_path_str);
        if (!std::filesystem::exists(control_path)) {
            spdlog::error("Control file not found: {}", control_path_str);
            return;
        }

        // 自动推导 mesh.dat 路径 (假设在同级目录)
        std::filesystem::path mesh_path = control_path.parent_path() / "mesh.dat";
        if (!std::filesystem::exists(mesh_path)) {
            spdlog::error("Mesh file not found at expected location: {}", mesh_path.string());
            spdlog::info("Tip: mesh.dat must be in the same directory as control.json");
            return;
        }

        spdlog::info("Importing Simdroid model...");
        spdlog::info("  Control: {}", control_path.string());
        spdlog::info("  Mesh:    {}", mesh_path.string());

        session.clear_data();

        // 调用 Parser
        try {
            if (SimdroidParser::parse(mesh_path.string(), control_path.string(), session.data)) {
                session.mesh_loaded = true;

                // 核心步骤：导入成功后，立即构建 Inspector 索引
                session.inspector.build(session.data.registry);

                spdlog::info("Simdroid import successful. Entered Simdroid Interactive Mode.");
            } else {
                spdlog::error("Simdroid import failed.");
            }
        } catch (const std::exception& e) {
            spdlog::error("Exception during import: {}", e.what());
        }
    }
    // =======================================================
    // 新增: Simdroid 导出命令 (Blueprint Strategy)
    // =======================================================
    else if (command == "export_simdroid") {
        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded. Please 'import' or 'import_simdroid' first.");
            return;
        }

        std::string arg1;
        std::string arg2;
        ss >> arg1 >> arg2;

        if (arg1.empty()) {
            spdlog::error("Usage: export_simdroid <output_dir | mesh.dat | control.json> [control.json]");
            return;
        }

        try {
            std::filesystem::path mesh_path;
            std::filesystem::path control_path;

            if (!arg2.empty()) {
                // 显式指定两个路径: export_simdroid <mesh.dat> <control.json>
                mesh_path = std::filesystem::path(arg1);
                control_path = std::filesystem::path(arg2);
            } else {
                // 只给一个参数时，按扩展名推导
                std::filesystem::path out(arg1);
                const std::string ext = out.extension().string();

                if (ext == ".json" || ext == ".jsonc") {
                    control_path = out;
                    mesh_path = out.parent_path() / "mesh.dat";
                } else if (ext == ".dat") {
                    mesh_path = out;
                    control_path = out.parent_path() / "control.json";
                } else {
                    // 当作输出目录
                    mesh_path = out / "mesh.dat";
                    control_path = out / "control.json";
                }
            }

            if (!mesh_path.parent_path().empty()) {
                std::filesystem::create_directories(mesh_path.parent_path());
            }
            if (!control_path.parent_path().empty()) {
                std::filesystem::create_directories(control_path.parent_path());
            }

            spdlog::info("Exporting Simdroid project...");
            spdlog::info("  Mesh:    {}", mesh_path.string());
            spdlog::info("  Control: {}", control_path.string());

            if (SimdroidExporter::save(mesh_path.string(), control_path.string(), session.data)) {
                spdlog::info("Simdroid export successful.");
            } else {
                spdlog::error("Simdroid export failed.");
            }

        } catch (const std::exception& e) {
            spdlog::error("Exception during export: {}", e.what());
        }
    }
    else if (command == "build_topology") {
        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded. Please 'import' a mesh first.");
            return;
        }
        spdlog::info("Building topology data...");
        TopologySystems::extract_topology(session.data.registry);
        session.topology_built = true;
        
        // Get the topology from context to report statistics
        auto& topology = *session.data.registry.ctx().get<std::unique_ptr<TopologyData>>();
        spdlog::info("Topology built successfully. Found {} unique faces.", topology.faces.size());
    }
    else if (command == "list_bodies") {
        if (!session.topology_built) {
            spdlog::error("Topology not built. Please run 'build_topology' first.");
            return;
        }
        spdlog::info("Finding continuous bodies...");
        TopologySystems::find_continuous_bodies(session.data.registry);
        
        // Get the topology from context
        auto& topology = *session.data.registry.ctx().get<std::unique_ptr<TopologyData>>();
        spdlog::info("Found {} continuous body/bodies:", topology.body_to_elements.size());
        for (const auto& pair : topology.body_to_elements) {
            spdlog::info("  - Body {}: {} elements", pair.first, pair.second.size());
        }
    }
    else if (command == "show_body") {
        if (!session.topology_built) {
            spdlog::error("Topology not built. Please run 'build_topology' first.");
            return;
        }

        int body_id_to_show;
        if (!(ss >> body_id_to_show)) {
            spdlog::error("Usage: show_body <body_id>");
            return;
        }

        // Get the topology from context
        auto& topology = *session.data.registry.ctx().get<std::unique_ptr<TopologyData>>();
        
        // Check if the requested BodyID exists
        auto it = topology.body_to_elements.find(body_id_to_show);
        if (it == topology.body_to_elements.end()) {
            spdlog::error("Body with ID {} not found. Use 'list_bodies' to see available bodies.", body_id_to_show);
            return;
        }

        const std::vector<entt::entity>& element_entities = it->second;
        
        // Build the output list
        std::stringstream element_list_ss;
        for (size_t i = 0; i < element_entities.size(); ++i) {
            entt::entity elem_entity = element_entities[i];
            // Get the OriginalID component to display external ID
            const auto& orig_id = session.data.registry.get<Component::OriginalID>(elem_entity);
            element_list_ss << orig_id.value << (i == element_entities.size() - 1 ? "" : ", ");
        }

        spdlog::info("Elements in Body {}:", body_id_to_show);
        spdlog::info("{}", element_list_ss.str());
    }
    else if (command == "save") {
        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded to save. Please 'import' a mesh first.");
            return;
        }
        std::string file_path;
        ss >> file_path;
        if (file_path.empty()) {
            spdlog::error("Usage: save <path_to_output_file.xfem>");
            return;
        }
        spdlog::info("Exporting mesh data to: {}", file_path);
        if (FemExporter::save(file_path, session.data)) {
            spdlog::info("Successfully exported mesh data.");
        } else {
            spdlog::error("Failed to export mesh data to: {}", file_path);
        }
    }
    else if (command == "info") {
        if (!session.mesh_loaded) {
            spdlog::warn("No mesh loaded.");
        } else {
            // Count entities using views
            auto node_count = session.data.registry.view<Component::Position>().size();
            auto element_count = session.data.registry.view<Component::Connectivity>().size();
            auto set_count = session.data.registry.view<Component::SetName>().size();
            
            spdlog::info("Mesh loaded: {} nodes, {} elements, {} sets",
                         node_count, element_count, set_count);
            
            if (session.topology_built) {
                auto& topology = *session.data.registry.ctx().get<std::unique_ptr<TopologyData>>();
                spdlog::info("Topology built: {} unique faces, {} bodies",
                             topology.faces.size(),
                             topology.body_to_elements.size());
            } else {
                spdlog::info("Topology not built yet.");
            }
        }
    }
    // =======================================================
    // 新增: Simdroid 交互式调试面板
    // =======================================================
    else if (command == "list_parts") {
        if (!session.mesh_loaded) { spdlog::warn("No mesh loaded."); return; }
        session.inspector.list_parts(session.data.registry);
    }
    else if (command == "delete_part") {
        std::vector<std::string> part_names;
        for (std::string name; ss >> name;) {
            part_names.push_back(name);
        }
        if (part_names.empty()) {
            spdlog::error("Usage: delete_part <part_name> [part_name2 ...]");
            return;
        }

        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded. Please 'import_simdroid' first.");
            return;
        }

        size_t deleted = 0;
        size_t failed = 0;
        for (const auto& part_name : part_names) {
            // delete_part() 会清空 inspector 索引；为保证多次删除稳定，这里每次都先重建索引
            session.inspector.build(session.data.registry);

            if (session.inspector.delete_part(session.data.registry, part_name)) {
                spdlog::info("Part '{}' deleted successfully.", part_name);
                ++deleted;
            } else {
                spdlog::error("Failed to delete part '{}'. Part not found?", part_name);
                ++failed;
            }
        }

        if (deleted > 0) {
            // 删除后必须重建索引，否则 eid_to_part 等映射会失效导致 Crash
            session.inspector.build(session.data.registry);
            // 拓扑数据会因实体删除而失效，清理以避免后续误用
            if (session.data.registry.ctx().contains<std::unique_ptr<TopologyData>>()) {
                session.data.registry.ctx().erase<std::unique_ptr<TopologyData>>();
            }
            session.topology_built = false;
        }
        spdlog::info("delete_part done. Deleted={}, Failed={}", deleted, failed);
    }
    else if (command == "graph") {
        if (!session.mesh_loaded) {
            spdlog::warn("No mesh loaded.");
            return;
        }

        std::string output_filename;
        ss >> output_filename;
        if (output_filename.empty()) output_filename = "connectivity.html";

        spdlog::info("Analyzing connectivity...");

        // 1. 构建图
        PartGraph graph = GraphBuilder::build(session.data.registry, session.inspector);

        // 2. (可选) 进行一些统计分析，比如打印孤立节点
        int isolated_count = 0;
        for (const auto& [n, node] : graph.nodes) {
            if (node.edges.empty()) isolated_count++;
        }
        spdlog::info("Analysis complete. Parts: {}, Isolated: {}", graph.nodes.size(), isolated_count);

        // 3. 生成报告
        MermaidReporter::generate_interactive_html(graph, output_filename);

        // 4. (可选) 尝试自动打开浏览器 (Windows/Linux/Mac)
#ifdef _WIN32
        std::string cmd = "start " + output_filename;
        system(cmd.c_str());
#endif
    }
    else if (command == "validate_constraints") {
        if (!session.mesh_loaded) {
            spdlog::warn("No mesh loaded. Please 'import_simdroid' first.");
            return;
        }

        spdlog::info("Re-running Simdroid constraint/contact validations...");
        auto& registry = session.data.registry;
        SimdroidParser::validate_constraints(registry);
    }
    else if (command == "list_constraint_warnings") {
        if (!session.mesh_loaded) {
            spdlog::warn("No mesh loaded. Please 'import_simdroid' first.");
            return;
        }
        auto& registry = session.data.registry;
        SimdroidParser::list_constraint_warnings(registry);
    }
    // =======================================================
    // 基础节点 / 单元操作
    // =======================================================
    else if (command == "node_add") {
        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded. Please 'import' or 'import_simdroid' first.");
            return;
        }
        int nid;
        double x, y, z;
        if (!(ss >> nid >> x >> y >> z)) {
            spdlog::error("Usage: node_add <nid> <x> <y> <z>");
            return;
        }
        auto& registry = session.data.registry;
        if (find_node_by_id(registry, nid) != entt::null) {
            spdlog::error("Node {} already exists.", nid);
            return;
        }
        auto e = registry.create();
        registry.emplace<Component::Position>(e, x, y, z);
        registry.emplace<Component::NodeID>(e, nid);
        registry.emplace<Component::OriginalID>(e, nid);
        spdlog::info("Node {} created at ({}, {}, {}).", nid, x, y, z);
        invalidate_topology_if_needed(session);
        rebuild_inspector_if_mesh_loaded(session);
    }
    else if (command == "node_move") {
        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded. Please 'import' or 'import_simdroid' first.");
            return;
        }
        int nid;
        double x, y, z;
        if (!(ss >> nid >> x >> y >> z)) {
            spdlog::error("Usage: node_move <nid> <x> <y> <z>");
            return;
        }
        auto& registry = session.data.registry;
        entt::entity e = find_node_by_id(registry, nid);
        if (e == entt::null) {
            spdlog::error("Node {} not found.", nid);
            return;
        }
        auto& pos = registry.get<Component::Position>(e);
        pos.x = x;
        pos.y = y;
        pos.z = z;
        spdlog::info("Node {} moved to ({}, {}, {}).", nid, x, y, z);
        // 拓扑结构不变，仅坐标变化，不必重建拓扑 / 索引
    }
    else if (command == "node_delete") {
        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded. Please 'import' or 'import_simdroid' first.");
            return;
        }
        int nid;
        if (!(ss >> nid)) {
            spdlog::error("Usage: node_delete <nid>");
            return;
        }
        auto& registry = session.data.registry;
        entt::entity node_e = find_node_by_id(registry, nid);
        if (node_e == entt::null) {
            spdlog::error("Node {} not found.", nid);
            return;
        }
        // 安全检查：如果节点仍被任何单元使用，则拒绝删除
        auto view_elems = registry.view<const Component::ElementID, const Component::Connectivity>();
        for (auto elem_e : view_elems) {
            const auto& conn = view_elems.get<const Component::Connectivity>(elem_e);
            if (std::find(conn.nodes.begin(), conn.nodes.end(), node_e) != conn.nodes.end()) {
                int eid = view_elems.get<const Component::ElementID>(elem_e).value;
                spdlog::error("Cannot delete node {}: used by element {}.", nid, eid);
                return;
            }
        }
        registry.destroy(node_e);
        spdlog::info("Node {} deleted.", nid);
        invalidate_topology_if_needed(session);
        rebuild_inspector_if_mesh_loaded(session);
    }
    else if (command == "elem_add") {
        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded. Please 'import' or 'import_simdroid' first.");
            return;
        }
        int eid;
        if (!(ss >> eid)) {
            spdlog::error("Usage: elem_add <eid> <nid1> <nid2> ...");
            return;
        }
        std::vector<int> node_ids;
        for (int nid; ss >> nid; ) {
            node_ids.push_back(nid);
        }
        if (node_ids.size() < 2) {
            spdlog::error("elem_add requires at least 2 node IDs.");
            return;
        }
        auto& registry = session.data.registry;
        if (find_element_by_id(registry, eid) != entt::null) {
            spdlog::error("Element {} already exists.", eid);
            return;
        }
        std::vector<entt::entity> node_entities;
        node_entities.reserve(node_ids.size());
        for (int nid : node_ids) {
            entt::entity ne = find_node_by_id(registry, nid);
            if (ne == entt::null) {
                spdlog::error("Node {} not found. Aborting elem_add.", nid);
                return;
            }
            node_entities.push_back(ne);
        }
        int type_id = infer_element_type_from_node_count(node_entities.size());
        if (type_id == 0) {
            spdlog::error("Unsupported element node count {} in elem_add.", node_entities.size());
            return;
        }
        auto e = registry.create();
        registry.emplace<Component::ElementID>(e, eid);
        registry.emplace<Component::OriginalID>(e, eid);
        registry.emplace<Component::ElementType>(e, type_id);
        auto& conn = registry.emplace<Component::Connectivity>(e);
        conn.nodes = std::move(node_entities);
        spdlog::info("Element {} created with {} nodes (type_id={}).", eid, conn.nodes.size(), type_id);
        invalidate_topology_if_needed(session);
        rebuild_inspector_if_mesh_loaded(session);
    }
    else if (command == "elem_delete") {
        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded. Please 'import' or 'import_simdroid' first.");
            return;
        }
        int eid;
        if (!(ss >> eid)) {
            spdlog::error("Usage: elem_delete <eid>");
            return;
        }
        auto& registry = session.data.registry;
        entt::entity elem_e = find_element_by_id(registry, eid);
        if (elem_e == entt::null) {
            spdlog::error("Element {} not found.", eid);
            return;
        }
        // 移除与该单元关联的 Surface
        {
            std::vector<entt::entity> surfaces_to_delete;
            auto surf_view = registry.view<const Component::SurfaceParentElement>();
            for (auto se : surf_view) {
                const auto& pe = surf_view.get<const Component::SurfaceParentElement>(se).element;
                if (pe == elem_e) {
                    surfaces_to_delete.push_back(se);
                }
            }
            if (!surfaces_to_delete.empty()) {
                // 从所有 SurfaceSetMembers 中移除
                std::unordered_set<entt::entity> surf_set(surfaces_to_delete.begin(), surfaces_to_delete.end());
                auto sset_view = registry.view<Component::SurfaceSetMembers>();
                for (auto set_e : sset_view) {
                    auto& mem = registry.get<Component::SurfaceSetMembers>(set_e).members;
                    mem.erase(
                        std::remove_if(mem.begin(), mem.end(), [&](entt::entity x) {
                            return surf_set.find(x) != surf_set.end();
                        }),
                        mem.end()
                    );
                }
                for (auto se : surfaces_to_delete) {
                    if (registry.valid(se)) registry.destroy(se);
                }
            }
        }
        // 从所有 ElementSetMembers 中移除该单元
        {
            auto eset_view = registry.view<Component::ElementSetMembers>();
            for (auto set_e : eset_view) {
                auto& mem = registry.get<Component::ElementSetMembers>(set_e).members;
                mem.erase(
                    std::remove(mem.begin(), mem.end(), elem_e),
                    mem.end()
                );
            }
        }
        registry.destroy(elem_e);
        spdlog::info("Element {} deleted.", eid);
        invalidate_topology_if_needed(session);
        rebuild_inspector_if_mesh_loaded(session);
    }
    // =======================================================
    // 基础 Set 操作
    // =======================================================
    else if (command == "list_sets") {
        if (!session.mesh_loaded) {
            spdlog::warn("No mesh loaded.");
            return;
        }
        auto& registry = session.data.registry;
        auto view = registry.view<const Component::SetName>();
        spdlog::info("Sets (total {}):", view.size());
        for (auto e : view) {
            const auto& name = view.get<const Component::SetName>(e).value;
            bool has_node = registry.all_of<Component::NodeSetMembers>(e);
            bool has_elem = registry.all_of<Component::ElementSetMembers>(e);
            bool has_surf = registry.all_of<Component::SurfaceSetMembers>(e);
            std::string type = "generic";
            std::size_t count = 0;
            if (has_node && !has_elem && !has_surf) {
                type = "node";
                count = registry.get<Component::NodeSetMembers>(e).members.size();
            } else if (!has_node && has_elem && !has_surf) {
                type = "element";
                count = registry.get<Component::ElementSetMembers>(e).members.size();
            } else if (!has_node && !has_elem && has_surf) {
                type = "surface";
                count = registry.get<Component::SurfaceSetMembers>(e).members.size();
            } else if (has_node || has_elem || has_surf) {
                type = "mixed";
            }
            spdlog::info("  - {} (type={}, size={})", name, type, count);
        }
    }
    else if (command == "set_info") {
        if (!session.mesh_loaded) {
            spdlog::warn("No mesh loaded.");
            return;
        }
        std::string set_name;
        ss >> set_name;
        if (set_name.empty()) {
            spdlog::error("Usage: set_info <set_name>");
            return;
        }
        auto& registry = session.data.registry;
        entt::entity set_e = find_set_by_name(registry, set_name);
        if (set_e == entt::null) {
            spdlog::error("Set '{}' not found.", set_name);
            return;
        }
        spdlog::info("Set '{}':", set_name);
        if (registry.all_of<Component::NodeSetMembers>(set_e)) {
            const auto& mem = registry.get<Component::NodeSetMembers>(set_e).members;
            spdlog::info("  Node members ({}):", mem.size());
            std::string line;
            int printed = 0;
            for (auto n : mem) {
                if (!registry.valid(n) || !registry.all_of<Component::NodeID>(n)) continue;
                int nid = registry.get<Component::NodeID>(n).value;
                if (!line.empty()) line += ", ";
                line += std::to_string(nid);
                ++printed;
                if (printed >= 32) {
                    spdlog::info("    {}", line);
                    line.clear();
                    printed = 0;
                }
            }
            if (!line.empty()) spdlog::info("    {}", line);
        }
        if (registry.all_of<Component::ElementSetMembers>(set_e)) {
            const auto& mem = registry.get<Component::ElementSetMembers>(set_e).members;
            spdlog::info("  Element members ({}):", mem.size());
            std::string line;
            int printed = 0;
            for (auto el : mem) {
                if (!registry.valid(el) || !registry.all_of<Component::ElementID>(el)) continue;
                int eid = registry.get<Component::ElementID>(el).value;
                if (!line.empty()) line += ", ";
                line += std::to_string(eid);
                ++printed;
                if (printed >= 32) {
                    spdlog::info("    {}", line);
                    line.clear();
                    printed = 0;
                }
            }
            if (!line.empty()) spdlog::info("    {}", line);
        }
    }
    else if (command == "set_addnode") {
        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded. Please 'import' or 'import_simdroid' first.");
            return;
        }
        std::string set_name;
        ss >> set_name;
        if (set_name.empty()) {
            spdlog::error("Usage: set_addnode <set_name> <nid1> [nid2 ...]");
            return;
        }
        std::vector<int> node_ids;
        for (int nid; ss >> nid; ) {
            node_ids.push_back(nid);
        }
        if (node_ids.empty()) {
            spdlog::error("set_addnode requires at least one node id.");
            return;
        }
        auto& registry = session.data.registry;
        entt::entity set_e = get_or_create_set_entity(registry, set_name);
        auto& mem = registry.get_or_emplace<Component::NodeSetMembers>(set_e);
        std::size_t added = 0;
        for (int nid : node_ids) {
            entt::entity ne = find_node_by_id(registry, nid);
            if (ne == entt::null) {
                spdlog::warn("Node {} not found. Skipped in set_addnode.", nid);
                continue;
            }
            if (std::find(mem.members.begin(), mem.members.end(), ne) == mem.members.end()) {
                mem.members.push_back(ne);
                ++added;
            }
        }
        spdlog::info("set_addnode '{}' : added {} nodes.", set_name, added);
    }
    else if (command == "set_addelem") {
        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded. Please 'import' or 'import_simdroid' first.");
            return;
        }
        std::string set_name;
        ss >> set_name;
        if (set_name.empty()) {
            spdlog::error("Usage: set_addelem <set_name> <eid1> [eid2 ...]");
            return;
        }
        std::vector<int> elem_ids;
        for (int eid; ss >> eid; ) {
            elem_ids.push_back(eid);
        }
        if (elem_ids.empty()) {
            spdlog::error("set_addelem requires at least one element id.");
            return;
        }
        auto& registry = session.data.registry;
        entt::entity set_e = get_or_create_set_entity(registry, set_name);
        auto& mem = registry.get_or_emplace<Component::ElementSetMembers>(set_e);
        std::size_t added = 0;
        for (int eid : elem_ids) {
            entt::entity ee = find_element_by_id(registry, eid);
            if (ee == entt::null) {
                spdlog::warn("Element {} not found. Skipped in set_addelem.", eid);
                continue;
            }
            if (std::find(mem.members.begin(), mem.members.end(), ee) == mem.members.end()) {
                mem.members.push_back(ee);
                ++added;
            }
        }
        spdlog::info("set_addelem '{}' : added {} elements.", set_name, added);
    }
    else if (command == "set_removenode") {
        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded. Please 'import' or 'import_simdroid' first.");
            return;
        }
        std::string set_name;
        ss >> set_name;
        if (set_name.empty()) {
            spdlog::error("Usage: set_removenode <set_name> <nid1> [nid2 ...]");
            return;
        }
        std::vector<int> node_ids;
        for (int nid; ss >> nid; ) {
            node_ids.push_back(nid);
        }
        if (node_ids.empty()) {
            spdlog::error("set_removenode requires at least one node id.");
            return;
        }
        auto& registry = session.data.registry;
        entt::entity set_e = find_set_by_name(registry, set_name);
        if (set_e == entt::null || !registry.all_of<Component::NodeSetMembers>(set_e)) {
            spdlog::error("Node set '{}' not found.", set_name);
            return;
        }
        auto& mem = registry.get<Component::NodeSetMembers>(set_e).members;
        std::size_t removed = 0;
        for (int nid : node_ids) {
            entt::entity ne = find_node_by_id(registry, nid);
            if (ne == entt::null) continue;
            auto it = std::remove(mem.begin(), mem.end(), ne);
            if (it != mem.end()) {
                removed += static_cast<std::size_t>(mem.end() - it);
                mem.erase(it, mem.end());
            }
        }
        spdlog::info("set_removenode '{}' : removed {} entries.", set_name, removed);
    }
    else if (command == "set_removeelem") {
        if (!session.mesh_loaded) {
            spdlog::error("No mesh loaded. Please 'import' or 'import_simdroid' first.");
            return;
        }
        std::string set_name;
        ss >> set_name;
        if (set_name.empty()) {
            spdlog::error("Usage: set_removeelem <set_name> <eid1> [eid2 ...]");
            return;
        }
        std::vector<int> elem_ids;
        for (int eid; ss >> eid; ) {
            elem_ids.push_back(eid);
        }
        if (elem_ids.empty()) {
            spdlog::error("set_removeelem requires at least one element id.");
            return;
        }
        auto& registry = session.data.registry;
        entt::entity set_e = find_set_by_name(registry, set_name);
        if (set_e == entt::null || !registry.all_of<Component::ElementSetMembers>(set_e)) {
            spdlog::error("Element set '{}' not found.", set_name);
            return;
        }
        auto& mem = registry.get<Component::ElementSetMembers>(set_e).members;
        std::size_t removed = 0;
        for (int eid : elem_ids) {
            entt::entity ee = find_element_by_id(registry, eid);
            if (ee == entt::null) continue;
            auto it = std::remove(mem.begin(), mem.end(), ee);
            if (it != mem.end()) {
                removed += static_cast<std::size_t>(mem.end() - it);
                mem.erase(it, mem.end());
            }
        }
        spdlog::info("set_removeelem '{}' : removed {} entries.", set_name, removed);
    }
    else if (command == "node") {
        int nid;
        if (ss >> nid) {
            session.inspector.inspect_node(session.data.registry, nid);
        } else {
            spdlog::error("Usage: node <node_id>");
        }
    }
    else if (command == "elem" || command == "element") {
        int eid;
        if (ss >> eid) {
            session.inspector.inspect_element(session.data.registry, eid);
        } else {
            spdlog::error("Usage: elem <element_id>");
        }
    }
    else if (command == "panel") {
        std::string type, id_or_name;
        if (!(ss >> type >> id_or_name)) {
            spdlog::error("Usage: panel <type> <id_or_name>  (type: node|elem|element|part|set)");
            return;
        }
        tui::PanelEntityKind kind = tui::PanelEntityKind::Unknown;
        std::string display_id;
        entt::entity e = tui::resolve_panel_entity(
            session.data.registry, &session.inspector, type, id_or_name, &kind, &display_id);
        if (e == entt::null) {
            spdlog::error("Panel: '{}' '{}' not found. Ensure mesh is loaded and index built.", type, id_or_name);
            return;
        }
        tui::render_panel(session.data.registry, e, &session.inspector, kind, display_id);
    }
    else {
        spdlog::warn("Unknown command: '{}'. Type 'help' for a list of commands.", command);
    }
}
