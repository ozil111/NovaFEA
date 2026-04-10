#include "SimdroidParser.h"
#include "SimdroidParserDetail.h"

#include "../../data_center/DofMap.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

using json = nlohmann::json;

// Static member definitions
std::vector<entt::entity> SimdroidParser::node_lookup{};
std::vector<entt::entity> SimdroidParser::element_lookup{};
std::vector<entt::entity> SimdroidParser::surface_lookup{};

bool SimdroidParser::parse(const std::string& mesh_path, 
                          const std::string& control_path, 
                          DataContext& ctx) {
    // 1. Parse Mesh DAT (sets/members must exist before applying loads/constraints)
    try {
        parse_mesh_dat(mesh_path, ctx);
    } catch (const std::exception& e) {
        spdlog::error("Error parsing mesh.dat: {}", e.what());
        return false;
    }

    // 2. Parse Control JSON
    try {
        parse_control_json(control_path, ctx);
    } catch (const std::exception& e) {
        spdlog::error("Error parsing control.json: {}", e.what());
        return false;
    }

    return true;
}
