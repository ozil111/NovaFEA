#pragma once
#include "DataContext.h"
#include <entt/entt.hpp>
#include "nlohmann/json.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

// Forward declarations for pointer/reference parameters in private methods
class DofMap;

class SimdroidParser {
    public:
        static bool parse(const std::string& mesh_path, 
                          const std::string& control_path, 
                          DataContext& context);

        // Re-run post-parse validations (contacts/rigid bodies/cross-constraints).
        // Intended for interactive mode commands.
        static void validate_constraints(entt::registry& registry);

        // Print cached cross-constraint conflict details collected during last validation.
        // If none were collected, prints a short "no conflicts" message.
        static void list_constraint_warnings(entt::registry& registry);
    private:
        // 使用 Vector 作为 O(1) 查找表
        static std::vector<entt::entity> node_lookup;
        static std::vector<entt::entity> element_lookup;
        static std::vector<entt::entity> surface_lookup;
        
        // 辅助函数
        static void parse_control_json(const std::string& path, DataContext& ctx);
        static void parse_mesh_dat(const std::string& path, DataContext& ctx);

        // New Parsing Helpers (Hierarchical Constraints + Loads)
        static void parse_boundary_conditions(const nlohmann::json& j, entt::registry& registry);
        static void parse_rigid_bodies(const nlohmann::json& j, entt::registry& registry);
        static void parse_loads(const nlohmann::json& j, entt::registry& registry);

        // [新增] Core parsing helpers
        static void parse_initial_conditions(const nlohmann::json& j, entt::registry& registry);
        static void parse_rigid_walls(const nlohmann::json& j, entt::registry& registry);
        static void parse_analysis_settings(const nlohmann::json& j, entt::registry& registry, DataContext& ctx);
        
        // [新增] Radioss /RBODY rigid bodies
        static void parse_radioss_rigid_bodies(const nlohmann::json& j, entt::registry& registry);
        static void validate_rigid_bodies(entt::registry& registry);
        static void validate_cross_constraints(entt::registry& registry);

        // Helper to find a set entity by name
        static entt::entity find_set_by_name(entt::registry& registry, const std::string& name);

        // Post-parse: validate contact entity references
        static void validate_contacts(entt::registry& registry);

        // --- Sub-parsers extracted from parse_control_json() ---
        static void parse_cross_sections(const nlohmann::json& j, entt::registry& registry,
                                         std::unordered_map<std::string, entt::entity>& cross_section_map,
                                         DofMap* dof_map);
        static void parse_materials(const nlohmann::json& j, entt::registry& registry,
                                    DofMap* dof_map, const std::filesystem::path& control_dir);
        static void parse_part_properties(const nlohmann::json& j, entt::registry& registry,
                                         const std::unordered_map<std::string, entt::entity>& cross_section_map);
        static void parse_contacts(const nlohmann::json& j, entt::registry& registry,
                                   DofMap* dof_map);

        // Curve entity helpers (extracted from lambdas in parse_control_json)
        static entt::entity get_or_create_curve_entity(
            const std::string& fname, entt::registry& registry,
            DofMap* dof_map, const std::filesystem::path& control_dir);
        static entt::entity resolve_curve_entity(
            const std::string& raw_name, entt::registry& registry, DofMap* dof_map);

    };