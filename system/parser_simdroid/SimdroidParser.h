#pragma once
#include "DataContext.h"
#include "entt/entt.hpp"
#include "nlohmann/json.hpp"
#include <string>
#include <vector>

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
    };