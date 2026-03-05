#include "SimdroidParser.h"

#include "components/mesh_components.h"
#include "components/analysis_component.h"
#include "components/simdroid_components.h"
#include "components/material_components.h"
#include "../../data_center/components/load_components.h"
#include "../../data_center/components/property_components.h"
#include "../../data_center/DofMap.h"
#include "../parser_base/string_utils.h"
#include "nlohmann/json.hpp"
#include "spdlog/spdlog.h"

#include <filesystem>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using json = nlohmann::json;

// Static member definitions
std::vector<entt::entity> SimdroidParser::node_lookup{};
std::vector<entt::entity> SimdroidParser::element_lookup{};
std::vector<entt::entity> SimdroidParser::surface_lookup{};

namespace {

struct IdRange {
    int start = 0;
    int end = 0;
    int step = 1;
};

inline std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::optional<std::string> extract_prefix_before_bracket(const std::string& s) {
    const auto lb = s.find('[');
    if (lb == std::string::npos) return std::nullopt;
    auto prefix = s.substr(0, lb);
    trim(prefix); 
    if (prefix.empty()) return std::nullopt;
    return prefix;
}

inline std::vector<std::string> split_ws_and_commas(std::string s) {
    for (char& c : s) {
        if (c == ',' || c == '[' || c == ']') c = ' ';
    }
    std::vector<std::string> out;
    std::istringstream iss(s);
    for (std::string tok; iss >> tok;) out.push_back(tok);
    return out;
}

inline std::vector<IdRange> parse_id_ranges(const std::string& id_string) {
    std::vector<IdRange> ranges;
    for (const auto& tok : split_ws_and_commas(id_string)) {
        const auto colon1 = tok.find(':');
        if (colon1 == std::string::npos) {
            try {
                const int v = std::stoi(tok);
                ranges.push_back(IdRange{v, v, 1});
            } catch (...) {}
            continue;
        }

        const auto colon2 = tok.find(':', colon1 + 1);
        try {
            if (colon2 == std::string::npos) {
                const int a = std::stoi(tok.substr(0, colon1));
                const int b = std::stoi(tok.substr(colon1 + 1));
                ranges.push_back(IdRange{a, b, 1});
            } else {
                const int a = std::stoi(tok.substr(0, colon1));
                const int b = std::stoi(tok.substr(colon1 + 1, colon2 - (colon1 + 1)));
                const int c = std::stoi(tok.substr(colon2 + 1));
                ranges.push_back(IdRange{a, b, c == 0 ? 1 : c});
            }
        } catch (...) {}
    }
    return ranges;
}

inline entt::entity get_or_create_set_entity(entt::registry& registry, const std::string& name) {
    auto view = registry.view<const Component::SetName>();
    for (auto e : view) {
        if (view.get<const Component::SetName>(e).value == name) return e;
    }
    const auto e = registry.create();
    registry.emplace<Component::SetName>(e, name);
    return e;
}

// ---------------------------------------------------------------
// String-to-Enum mapping tables for Contact parameters
// ---------------------------------------------------------------

const std::unordered_map<std::string, Component::ContactInterType> kContactTypeMap = {
    {"nodetosurfacetie",     Component::ContactInterType::Tie},
    {"surfacetosurfacetie",  Component::ContactInterType::Tie},
    {"nodetosurface",        Component::ContactInterType::NodeToSurface},
    {"nodetosurfacecontact", Component::ContactInterType::NodeToSurface},
    {"surfacetosurface",     Component::ContactInterType::NodeToSurface},
    {"generalcontact",       Component::ContactInterType::General},
};

const std::unordered_map<std::string, Component::ContactFormulationType> kFormulationMap = {
    {"standard",           Component::ContactFormulationType::Standard},
    {"optimized",          Component::ContactFormulationType::Optimized},
    {"failshbrick",        Component::ContactFormulationType::FailShBrick},
    {"failsh",             Component::ContactFormulationType::FailSh},
    {"failbrick",          Component::ContactFormulationType::FailBrick},
    {"penalty",            Component::ContactFormulationType::Penalty},
    {"standswitchpenal",   Component::ContactFormulationType::StandSwitchPenal},
    {"optimswitchpenal",   Component::ContactFormulationType::OptimSwitchPenal},
};

const std::unordered_map<std::string, Component::InterfaceStiffnessType> kInterfaceStiffnessMap = {
    {"default", Component::InterfaceStiffnessType::Default},
    {"main",    Component::InterfaceStiffnessType::Main},
    {"maximum", Component::InterfaceStiffnessType::Maximum},
    {"minimum", Component::InterfaceStiffnessType::Minimum},
};

const std::unordered_map<std::string, Component::SearchMethodType> kSearchMethodMap = {
    {"box",     Component::SearchMethodType::Box},
    {"segment", Component::SearchMethodType::Segment},
};

const std::unordered_map<std::string, Component::FrictionLawType> kFrictionLawMap = {
    {"coulomb",  Component::FrictionLawType::Coulomb},
    {"viscous",  Component::FrictionLawType::Viscous},
    {"darmstad", Component::FrictionLawType::Darmstad},
    {"renard",   Component::FrictionLawType::Renard},
    {"expdecay", Component::FrictionLawType::ExpDecay},
};

const std::unordered_map<std::string, Component::GapFlagType> kGapFlagMap = {
    {"variable",         Component::GapFlagType::Variable},
    {"variablescale",    Component::GapFlagType::VariableScale},
    {"variablescalepen", Component::GapFlagType::VariableScalePen},
    {"gapmin",           Component::GapFlagType::Gapmin},
};

const std::unordered_map<std::string, Component::IgnoreType> kIgnoreMap = {
    {"length",        Component::IgnoreType::Length},
    {"mainthick",     Component::IgnoreType::MainThick},
    {"mainthickzero", Component::IgnoreType::MainThickZero},
};

const std::unordered_map<std::string, Component::FailModeType> kFailModeMap = {
    {"max",  Component::FailModeType::Max},
    {"quad", Component::FailModeType::Quad},
};

const std::unordered_map<std::string, Component::RuptureFlagType> kRuptureFlagMap = {
    {"tractioncompress", Component::RuptureFlagType::TractionCompress},
    {"traction",         Component::RuptureFlagType::Traction},
};

template <typename EnumT>
EnumT lookup_enum(const std::unordered_map<std::string, EnumT>& map,
                  const std::string& raw, EnumT fallback) {
    auto it = map.find(to_lower_copy(raw));
    return it != map.end() ? it->second : fallback;
}

// Stiffness form int mapping (TYPE7/TYPE24): string -> Istf code
const std::unordered_map<std::string, int> kStiffnessFormMap = {
    {"stiffnessfactor", 1}, {"average", 2}, {"maximum", 3},
    {"minimum", 4}, {"masteronly", 1000},
};

struct MeshSetDefs {
    std::unordered_map<std::string, std::vector<IdRange>> element_sets;
    std::unordered_map<std::string, std::vector<IdRange>> parts_ranges;
    std::unordered_map<std::string, std::vector<IdRange>> node_sets;
    std::unordered_map<std::string, std::vector<IdRange>> surface_sets;
};

void collect_set_definitions_from_file(const std::string& path, MeshSetDefs& defs) {
    std::ifstream file(path);
    if (!file.is_open()) return;

    std::string current_block; 
    std::string current_name;
    std::string current_ids;

    auto flush_current = [&](const std::string& block_type) {
        if (current_name.empty()) return;
        const auto ranges = parse_id_ranges(current_ids);
        if (ranges.empty()) return;

        if (block_type == "element") {
            auto& v = defs.element_sets[current_name];
            v.insert(v.end(), ranges.begin(), ranges.end());
        } else if (block_type == "part") {
            auto& v = defs.parts_ranges[current_name];
            v.insert(v.end(), ranges.begin(), ranges.end());
        } else if (block_type == "node") {
            auto& v = defs.node_sets[current_name];
            v.insert(v.end(), ranges.begin(), ranges.end());
        } else if (block_type == "surface") {
            auto& v = defs.surface_sets[current_name];
            v.insert(v.end(), ranges.begin(), ranges.end());
        }
        current_ids.clear(); 
    };

    std::string line;
    int brace_level = 0;
    enum State { IDLE, IN_SET_BLOCK, IN_PART_BLOCK };
    State state = IDLE;

    while (std::getline(file, line)) {
        preprocess_line(line);
        if (line.empty()) continue;

        if (state == IDLE) {
            if (line == "Set {") {
                state = IN_SET_BLOCK;
                brace_level = 1;
                current_block.clear();
                current_name.clear();
                current_ids.clear();
            } else if (line == "Part {") {
                state = IN_PART_BLOCK;
                brace_level = 1;
                current_block = "part"; 
                current_name.clear();
                current_ids.clear();
            }
            continue;
        }

        if (ends_with(line, "{")) {
            ++brace_level;
            if (state == IN_SET_BLOCK) {
                auto b = line.substr(0, line.size() - 1);
                trim(b);
                current_block = to_lower_copy(b);
            }
            continue;
        }

        if (line == "}") {
            flush_current(current_block);
            current_name.clear();
            --brace_level;
            if (brace_level <= 0) {
                state = IDLE;
                current_block.clear();
            } else if (state == IN_SET_BLOCK && brace_level == 1) {
                current_block.clear();
            }
            continue;
        }

        if (state == IN_SET_BLOCK) {
            bool is_supported = (current_block == "element" || current_block == "part" || current_block == "node" || current_block == "surface");
            if (!is_supported) continue;
        }

        bool has_open_bracket = line.find('[') != std::string::npos;
        if (has_open_bracket) {
            flush_current(current_block);
            current_name.clear(); 
            auto name_opt = extract_prefix_before_bracket(line);
            if (name_opt) current_name = *name_opt;
            
            size_t lb = line.find('[');
            std::string content = line.substr(lb + 1);
            size_t rb = content.rfind(']');
            if (rb != std::string::npos) {
                current_ids = content.substr(0, rb);
                flush_current(current_block); 
                current_name.clear();
            } else {
                current_ids = content; 
            }
        } else if (!current_name.empty()) {
            std::string segment = line;
            size_t rb = segment.rfind(']');
            if (rb != std::string::npos) {
                current_ids += " " + segment.substr(0, rb);
                flush_current(current_block);
                current_name.clear();
            } else {
                current_ids += " " + segment;
            }
        }
    }
}

} // namespace

namespace {

struct CrossConstraintConflictDetail {
    int node_id = -1;
    std::vector<std::string> as_master;
    std::vector<std::string> as_slave;
};

struct ConstraintWarningReport {
    std::vector<CrossConstraintConflictDetail> cross_conflicts;
};

ConstraintWarningReport& get_constraint_warning_report(entt::registry& registry) {
    auto& ctx = registry.ctx();
    if (!ctx.contains<ConstraintWarningReport>()) {
        return ctx.emplace<ConstraintWarningReport>();
    }
    return ctx.get<ConstraintWarningReport>();
}

} // namespace

// Helper to normalize vector
static std::tuple<double, double, double> normalize(double x, double y, double z) {
    const double len = std::sqrt(x * x + y * y + z * z);
    if (len < 1e-9) return {0.0, 0.0, 0.0};
    return {x / len, y / len, z / len};
}

bool SimdroidParser::parse(const std::string& mesh_path, const std::string& control_path, DataContext& ctx) {
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

void SimdroidParser::parse_control_json(const std::string& path, DataContext& ctx) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("cannot open control file");
    json j = json::parse(file, nullptr, true, true);
    
    // [Blueprint Strategy] 将原始 JSON 保存到 DataContext 中作为蓝图
    // Export 时将 ECS 修改回写到此蓝图，保留所有未解析的字段
    ctx.simdroid_blueprint = j;
    spdlog::info("Simdroid blueprint saved. Unknown fields will be preserved during export.");
    
    auto& registry = ctx.registry;

    // ---------------------------------------------------------------------
    // 预解析：Function 名称列表 -> Curve 实体
    // 用于超弹性材料的 TestCurve-* 列表 (HyperelasticMode::*_funcs)
    //  - 统一复用 Component::Curve / CurveID
    //  - 曲线名 -> entity 映射存放在 registry.ctx().get<DofMap>()
    //  - 曲线数据直接从同名 .txt 中读取（格式参考 convert_function.py）
    // ---------------------------------------------------------------------
    DofMap* dof_map = nullptr;
    if (registry.ctx().contains<DofMap>()) {
        dof_map = &registry.ctx().get<DofMap>();
    } else {
        dof_map = &registry.ctx().emplace<DofMap>();
    }

    // 控制文件所在目录，用于定位同名 .txt
    std::filesystem::path control_dir = std::filesystem::path(path).parent_path();

    auto get_or_create_curve_entity_by_name = [&](const std::string& fname) -> entt::entity {
        // 已存在映射则直接返回
        auto it = dof_map->curve_name_to_entity.find(fname);
        if (it != dof_map->curve_name_to_entity.end()) {
            return it->second;
        }

        // 创建新的 Curve 实体
        entt::entity curve_e = registry.create();

        // 目前 CurveID 仅用于调试/区分，这里使用当前映射大小作为简单 ID
        int cid = static_cast<int>(dof_map->curve_name_to_entity.size());
        registry.emplace<Component::CurveID>(curve_e, cid);

        Component::Curve curve{};
        curve.type = "tabular"; // 通用 1D 数据点

        // 从同名 .txt 读取数据：第一列 strain(x), 第二列 stress(y)，忽略以 '#' 开头的行
        std::filesystem::path txt_path = control_dir / (fname + ".txt");
        std::ifstream fin(txt_path);
        if (!fin.is_open()) {
            spdlog::warn("Function '{}' expects txt file '{}', but it cannot be opened.", fname, txt_path.string());
        } else {
            std::string line;
            while (std::getline(fin, line)) {
                if (line.empty()) continue;
                if (!line.empty() && line[0] == '#') continue;

                std::istringstream iss(line);
                double col1 = 0.0, col2 = 0.0;
                if (!(iss >> col1 >> col2)) continue;

                // 约定：x = strain, y = stress
                curve.x.push_back(col1);
                curve.y.push_back(col2);
            }
            if (curve.x.empty()) {
                spdlog::warn("Txt file '{}' for Function '{}' contains no valid data rows.", txt_path.string(), fname);
            }
        }

        registry.emplace<Component::Curve>(curve_e, std::move(curve));
        dof_map->curve_name_to_entity.emplace(fname, curve_e);
        dof_map->curve_entity_to_name.emplace(curve_e, fname);

        return curve_e;
    };

    auto resolve_curve_entity_by_name = [&](const std::string& raw_name) -> entt::entity {
        if (!dof_map) return entt::null;
        std::string fname = raw_name;
        trim(fname);
        if (fname.empty()) return entt::null;
        auto it = dof_map->curve_name_to_entity.find(fname);
        if (it != dof_map->curve_name_to_entity.end()) {
            return it->second;
        }
        return get_or_create_curve_entity_by_name(fname);
    };

    // 预先根据 Function 块创建对应的 Curve 实体（如果存在）
    if (j.contains("Function") && j["Function"].is_object()) {
        for (auto& [fname, fval] : j["Function"].items()) {
            (void)fval;
            (void)get_or_create_curve_entity_by_name(fname);
        }
        spdlog::info("Simdroid Functions parsed: {} entries mapped to Curve entities.", dof_map->curve_name_to_entity.size());
    }

    // ---------------------------------------------------------------------
    // CrossSection: 截面/属性定义 -> Property 实体
    //  - 使用 SetName = CrossSection 名称
    //  - 按 Type 附加不同的 Property 组件
    // ---------------------------------------------------------------------
    std::unordered_map<std::string, entt::entity> cross_section_map;

    if (j.contains("CrossSection") && j["CrossSection"].is_object()) {
        for (auto& [cs_name, cs_val] : j["CrossSection"].items()) {
            const entt::entity cs_entity = registry.create();
            registry.emplace<Component::SetName>(cs_entity, cs_name);
            cross_section_map.emplace(cs_name, cs_entity);

            const std::string type_str = cs_val.value("Type", "");
            const std::string type_l = to_lower_copy(type_str);

            // --- Solid / SolidOrthotropic ---
            if (type_l == "solid" || type_l == "solidorthotropic") {
                Component::SolidAdvancedProperty prop{};

                if (cs_val.contains("Formulation") && cs_val["Formulation"].is_array() && !cs_val["Formulation"].empty()) {
                    prop.formulation = cs_val["Formulation"].front().get<std::string>();
                }
                prop.small_strain = cs_val.value("SmallStrain", "");
                prop.const_pressure = cs_val.value("ConstPressure", "");
                prop.co_rotation_flag = cs_val.value("CoRotationFlag", "");

                if (cs_val.contains("ViscoHourglassK")) {
                    prop.visco_hourglass_k = cs_val["ViscoHourglassK"].get<double>();
                } else if (cs_val.contains("hm")) {
                    prop.visco_hourglass_k = cs_val["hm"].get<double>();
                }
                if (cs_val.contains("QuadraticViscosity")) {
                    prop.bulk_viscosity.quadratic = cs_val["QuadraticViscosity"].get<double>();
                }
                if (cs_val.contains("LinearViscosity")) {
                    prop.bulk_viscosity.linear = cs_val["LinearViscosity"].get<double>();
                }
                if (cs_val.contains("dtmin") && cs_val["dtmin"].is_number()) {
                    prop.dtmin = cs_val["dtmin"].get<double>();
                }
                if (cs_val.contains("DampingNumeri") && cs_val["DampingNumeri"].is_number()) {
                    prop.numeric_damping = cs_val["DampingNumeri"].get<double>();
                }
                if (cs_val.contains("DistortionControl") && cs_val["DistortionControl"].is_boolean()) {
                    prop.distortion_control = cs_val["DistortionControl"].get<bool>();
                }
                if (cs_val.contains("DistortionControlCoeffs") && cs_val["DistortionControlCoeffs"].is_array()) {
                    const auto& arr = cs_val["DistortionControlCoeffs"];
                    for (size_t i = 0; i < std::min<std::size_t>(3, arr.size()); ++i) {
                        prop.distortion_coeffs[i] = arr[i].get<double>();
                    }
                }
                if (cs_val.contains("DispHourglassFactor") && cs_val["DispHourglassFactor"].is_number()) {
                    prop.disp_hourglass_factor = cs_val["DispHourglassFactor"].get<double>();
                }
                if (cs_val.contains("HourglassType") && cs_val["HourglassType"].is_string()) {
                    prop.hourglass_type = cs_val["HourglassType"].get<std::string>();
                }
                if (cs_val.contains("EleCharacLength") && cs_val["EleCharacLength"].is_boolean()) {
                    prop.ele_charac_length = cs_val["EleCharacLength"].get<bool>();
                }

                registry.emplace<Component::SolidAdvancedProperty>(cs_entity, std::move(prop));
            }
            // --- Shell / SandwichShell ---
            else if (type_l == "shell" || type_l == "sandwichshell") {
                Component::ShellProperty prop{};
                prop.type_id = (type_l == "shell") ? 1 : 11;

                if (cs_val.contains("Thickness") && cs_val["Thickness"].is_array()) {
                    const auto& arr = cs_val["Thickness"];
                    for (size_t i = 0; i < std::min<std::size_t>(4, arr.size()); ++i) {
                        prop.thickness[i] = arr[i].get<double>();
                    }
                }
                if (cs_val.contains("ThicknessChange") && cs_val["ThicknessChange"].is_boolean()) {
                    prop.thickness_change = cs_val["ThicknessChange"].get<bool>();
                }
                if (cs_val.contains("DrillDof") && cs_val["DrillDof"].is_boolean()) {
                    prop.drill_dof = cs_val["DrillDof"].get<bool>();
                }
                if (cs_val.contains("ShearFactor") && cs_val["ShearFactor"].is_number()) {
                    prop.shear_factor = cs_val["ShearFactor"].get<double>();
                }
                if (cs_val.contains("InpNum") && cs_val["InpNum"].is_number_integer()) {
                    prop.integration_points = cs_val["InpNum"].get<int>();
                }
                if (cs_val.contains("InpRule") && cs_val["InpRule"].is_string()) {
                    prop.inp_rule = cs_val["InpRule"].get<std::string>();
                }
                if (cs_val.contains("FailThick") && cs_val["FailThick"].is_number()) {
                    prop.fail_thick = cs_val["FailThick"].get<double>();
                }
                if (cs_val.contains("HourglassCofs") && cs_val["HourglassCofs"].is_array()) {
                    const auto& arr = cs_val["HourglassCofs"];
                    for (size_t i = 0; i < std::min<std::size_t>(3, arr.size()); ++i) {
                        prop.hourglass_coefs[i] = arr[i].get<double>();
                    }
                }
                if (cs_val.contains("PlasticPlaneStressReturn") && cs_val["PlasticPlaneStressReturn"].is_string()) {
                    prop.plastic_plane_stress_return = cs_val["PlasticPlaneStressReturn"].get<std::string>();
                }
                if (cs_val.contains("MidShellFlag") && cs_val["MidShellFlag"].is_string()) {
                    prop.mid_shell_flag = cs_val["MidShellFlag"].get<std::string>();
                }

                registry.emplace<Component::ShellProperty>(cs_entity, std::move(prop));
            }
            // --- SolidShell / SolidShComp (厚壳 / 复合厚壳) ---
            else if (type_l == "solidshell") {
                Component::SolidShellProperty prop{};
                if (cs_val.contains("Formulation") && cs_val["Formulation"].is_string()) {
                    prop.formulation.value = cs_val["Formulation"].get<std::string>();
                }
                if (cs_val.contains("SmallStrain") && cs_val["SmallStrain"].is_string()) {
                    prop.small_strain.value = cs_val["SmallStrain"].get<std::string>();
                }
                if (cs_val.contains("InpNum") && cs_val["InpNum"].is_array()) {
                    const auto& arr = cs_val["InpNum"];
                    for (size_t i = 0; i < std::min<std::size_t>(3, arr.size()); ++i) {
                        prop.integration_points[i] = arr[i].get<int>();
                    }
                }
                if (cs_val.contains("ViscoHourglassK") && cs_val["ViscoHourglassK"].is_number()) {
                    prop.visco_hourglass_k = cs_val["ViscoHourglassK"].get<double>();
                }
                if (cs_val.contains("QuadraticViscosity") && cs_val["QuadraticViscosity"].is_number()) {
                    prop.bulk_viscosity.quadratic = cs_val["QuadraticViscosity"].get<double>();
                }
                if (cs_val.contains("LinearViscosity") && cs_val["LinearViscosity"].is_number()) {
                    prop.bulk_viscosity.linear = cs_val["LinearViscosity"].get<double>();
                }
                if (cs_val.contains("dtmin") && cs_val["dtmin"].is_number()) {
                    prop.dtmin = cs_val["dtmin"].get<double>();
                }
                if (cs_val.contains("ThicknessPenaltyFactor") && cs_val["ThicknessPenaltyFactor"].is_number()) {
                    prop.thickness_penalty = cs_val["ThicknessPenaltyFactor"].get<double>();
                }
                if (cs_val.contains("DistortionControlCoeffs") && cs_val["DistortionControlCoeffs"].is_array()) {
                    const auto& arr = cs_val["DistortionControlCoeffs"];
                    for (size_t i = 0; i < std::min<std::size_t>(3, arr.size()); ++i) {
                        prop.distortion_coeffs[i] = arr[i].get<double>();
                    }
                }
                registry.emplace<Component::SolidShellProperty>(cs_entity, std::move(prop));
            }
            else if (type_l == "solidshcomp") {
                Component::SolidShCompProperty prop{};
                if (cs_val.contains("Formulation") && cs_val["Formulation"].is_string()) {
                    prop.formulation.value = cs_val["Formulation"].get<std::string>();
                }
                if (cs_val.contains("SmallStrain") && cs_val["SmallStrain"].is_string()) {
                    prop.small_strain.value = cs_val["SmallStrain"].get<std::string>();
                }
                if (cs_val.contains("InpNum") && cs_val["InpNum"].is_array()) {
                    const auto& arr = cs_val["InpNum"];
                    for (size_t i = 0; i < std::min<std::size_t>(3, arr.size()); ++i) {
                        prop.integration_points[i] = arr[i].get<int>();
                    }
                }
                if (cs_val.contains("DampingNumeri") && cs_val["DampingNumeri"].is_number()) {
                    prop.numeric_damping = cs_val["DampingNumeri"].get<double>();
                }
                if (cs_val.contains("QuadraticViscosity") && cs_val["QuadraticViscosity"].is_number()) {
                    prop.bulk_viscosity.quadratic = cs_val["QuadraticViscosity"].get<double>();
                }
                if (cs_val.contains("LinearViscosity") && cs_val["LinearViscosity"].is_number()) {
                    prop.bulk_viscosity.linear = cs_val["LinearViscosity"].get<double>();
                }
                if (cs_val.contains("ThicknessPenaltyFactor") && cs_val["ThicknessPenaltyFactor"].is_number()) {
                    prop.thickness_penalty = cs_val["ThicknessPenaltyFactor"].get<double>();
                }
                if (cs_val.contains("CoordSys") && cs_val["CoordSys"].is_string()) {
                    prop.coord_sys.value = cs_val["CoordSys"].get<std::string>();
                }
                if (cs_val.contains("Angles") && cs_val["Angles"].is_array()) {
                    prop.layer_angles = cs_val["Angles"].get<std::vector<double>>();
                }
                if (cs_val.contains("Thicks") && cs_val["Thicks"].is_array()) {
                    prop.layer_thicks = cs_val["Thicks"].get<std::vector<double>>();
                }
                if (cs_val.contains("Positions") && cs_val["Positions"].is_array()) {
                    prop.layer_positions = cs_val["Positions"].get<std::vector<double>>();
                }
                if (cs_val.contains("Materials") && cs_val["Materials"].is_array()) {
                    prop.layer_materials = cs_val["Materials"].get<std::vector<std::string>>();
                }
                if (cs_val.contains("PositionFlag") && cs_val["PositionFlag"].is_array()) {
                    prop.position_flags = cs_val["PositionFlag"].get<std::vector<std::string>>();
                }
                registry.emplace<Component::SolidShCompProperty>(cs_entity, std::move(prop));
            }
            // --- Beam / FiberBeam / Cohesive / Springs ---
            else if (type_l == "generalbeam" || type_l == "beam") {
                Component::BeamProperty prop{};
                if (cs_val.contains("SmallStrain") && cs_val["SmallStrain"].is_string()) {
                    prop.small_strain.value = cs_val["SmallStrain"].get<std::string>();
                }
                if (cs_val.contains("Area") && cs_val["Area"].is_number()) {
                    prop.area = cs_val["Area"].get<double>();
                }
                if (cs_val.contains("Ixx") && cs_val["Ixx"].is_number()) {
                    prop.ixx = cs_val["Ixx"].get<double>();
                }
                if (cs_val.contains("Iyy") && cs_val["Iyy"].is_number()) {
                    prop.iyy = cs_val["Iyy"].get<double>();
                }
                if (cs_val.contains("Izz") && cs_val["Izz"].is_number()) {
                    prop.izz = cs_val["Izz"].get<double>();
                }
                if (cs_val.contains("ShearFlag") && cs_val["ShearFlag"].is_boolean()) {
                    prop.shear_flag = cs_val["ShearFlag"].get<bool>();
                }
                registry.emplace<Component::BeamProperty>(cs_entity, std::move(prop));
            }
            else if (type_l == "fiberbeam") {
                Component::FiberBeamProperty prop{};
                if (cs_val.contains("Pattern") && cs_val["Pattern"].is_string()) {
                    prop.pattern = cs_val["Pattern"].get<std::string>();
                }
                if (cs_val.contains("SmallStrain") && cs_val["SmallStrain"].is_string()) {
                    prop.small_strain.value = cs_val["SmallStrain"].get<std::string>();
                }
                if (cs_val.contains("InpNum") && cs_val["InpNum"].is_number_integer()) {
                    prop.integration_points = cs_val["InpNum"].get<int>();
                }
                if (cs_val.contains("Yi") && cs_val["Yi"].is_array()) {
                    prop.yi = cs_val["Yi"].get<std::vector<double>>();
                }
                if (cs_val.contains("Zi") && cs_val["Zi"].is_array()) {
                    prop.zi = cs_val["Zi"].get<std::vector<double>>();
                }
                if (cs_val.contains("Areai") && cs_val["Areai"].is_array()) {
                    prop.areai = cs_val["Areai"].get<std::vector<double>>();
                }
                if (cs_val.contains("Dj") && cs_val["Dj"].is_array()) {
                    prop.dj = cs_val["Dj"].get<std::vector<double>>();
                }
                registry.emplace<Component::FiberBeamProperty>(cs_entity, std::move(prop));
            }
            else if (type_l == "cohesive") {
                Component::CohesiveProperty prop{};
                if (cs_val.contains("SmallStrain") && cs_val["SmallStrain"].is_string()) {
                    prop.small_strain.value = cs_val["SmallStrain"].get<std::string>();
                }
                if (cs_val.contains("Thickness") && cs_val["Thickness"].is_number()) {
                    prop.thickness = cs_val["Thickness"].get<double>();
                }
                registry.emplace<Component::CohesiveProperty>(cs_entity, std::move(prop));
            }
            else if (type_l == "axialspringdamper") {
                Component::AxialSpringDamperProperty prop{};
                if (cs_val.contains("Mass") && cs_val["Mass"].is_number()) {
                    prop.mass = cs_val["Mass"].get<double>();
                }
                if (cs_val.contains("Stiffness") && cs_val["Stiffness"].is_number()) {
                    prop.stiffness = cs_val["Stiffness"].get<double>();
                }
                if (cs_val.contains("DampingCoefficient") && cs_val["DampingCoefficient"].is_number()) {
                    prop.damping = cs_val["DampingCoefficient"].get<double>();
                }
                if (cs_val.contains("NonlinearSpring") && cs_val["NonlinearSpring"].is_boolean()) {
                    prop.nonlinear_spring = cs_val["NonlinearSpring"].get<bool>();
                }
                if (cs_val.contains("NonlinearDamper") && cs_val["NonlinearDamper"].is_boolean()) {
                    prop.nonlinear_damper = cs_val["NonlinearDamper"].get<bool>();
                }
                if (cs_val.contains("HardeningFlag") && cs_val["HardeningFlag"].is_string()) {
                    prop.hardening_flag = cs_val["HardeningFlag"].get<std::string>();
                }
                if (cs_val.contains("Load_DeflectionCurve") && cs_val["Load_DeflectionCurve"].is_string()) {
                    prop.stiffness_curve = resolve_curve_entity_by_name(cs_val["Load_DeflectionCurve"].get<std::string>());
                }
                if (cs_val.contains("DampingCurve") && cs_val["DampingCurve"].is_string()) {
                    prop.damping_curve = resolve_curve_entity_by_name(cs_val["DampingCurve"].get<std::string>());
                }
                registry.emplace<Component::AxialSpringDamperProperty>(cs_entity, std::move(prop));
            }
            else if (type_l == "beamspring") {
                Component::BeamSpringProperty prop{};
                if (cs_val.contains("Mass") && cs_val["Mass"].is_number()) {
                    prop.mass = cs_val["Mass"].get<double>();
                }
                if (cs_val.contains("Inertia") && cs_val["Inertia"].is_number()) {
                    prop.inertia = cs_val["Inertia"].get<double>();
                }
                if (cs_val.contains("CoordSys") && cs_val["CoordSys"].is_string()) {
                    prop.coord_sys.value = cs_val["CoordSys"].get<std::string>();
                }
                if (cs_val.contains("FailureCriteria") && cs_val["FailureCriteria"].is_string()) {
                    prop.failure_criteria = cs_val["FailureCriteria"].get<std::string>();
                }
                if (cs_val.contains("LengthFlag") && cs_val["LengthFlag"].is_string()) {
                    prop.length_flag = cs_val["LengthFlag"].get<std::string>();
                }
                if (cs_val.contains("FailureModel") && cs_val["FailureModel"].is_string()) {
                    prop.failure_model = cs_val["FailureModel"].get<std::string>();
                }

                auto fill_array_double = [&](const char* key, std::array<double, 6>& dst) {
                    if (!cs_val.contains(key) || !cs_val[key].is_array()) return;
                    const auto& arr = cs_val[key];
                    for (size_t i = 0; i < std::min<std::size_t>(6, arr.size()); ++i) {
                        dst[i] = arr[i].get<double>();
                    }
                };
                auto fill_array_string = [&](const char* key, std::array<std::string, 6>& dst) {
                    if (!cs_val.contains(key) || !cs_val[key].is_array()) return;
                    const auto& arr = cs_val[key];
                    for (size_t i = 0; i < std::min<std::size_t>(6, arr.size()); ++i) {
                        if (arr[i].is_string()) dst[i] = arr[i].get<std::string>();
                    }
                };
                auto fill_array_curve = [&](const char* key, std::array<entt::entity, 6>& dst) {
                    if (!cs_val.contains(key) || !cs_val[key].is_array()) return;
                    const auto& arr = cs_val[key];
                    for (size_t i = 0; i < std::min<std::size_t>(6, arr.size()); ++i) {
                        if (arr[i].is_string()) {
                            dst[i] = resolve_curve_entity_by_name(arr[i].get<std::string>());
                        }
                    }
                };

                fill_array_double("LinearStiffness", prop.linear_stiffness);
                fill_array_double("LinearDamping", prop.linear_damping);
                fill_array_double("NonStiffFacA", prop.non_stiff_fac_a);
                fill_array_double("NonStiffFacB", prop.non_stiff_fac_b);
                fill_array_double("NonStiffFacD", prop.non_stiff_fac_d);
                fill_array_string("HardeningFlag", prop.hardening_flag);

                fill_array_curve("NonlinearStiffness", prop.nonlinear_stiffness);
                fill_array_curve("ForOrMomWithVelCurve", prop.for_or_mom_with_vel);
                fill_array_curve("HardenRelatedCurve", prop.harden_related_curve);
                fill_array_curve("NonlinearDamping", prop.nonlinear_damping);

                fill_array_double("UpperFailureLimit", prop.upper_failure_limit);
                fill_array_double("LowerFailureLimit", prop.lower_failure_limit);
                fill_array_double("AbscScaleDamp", prop.absc_scale_damp);
                fill_array_double("OrdinaScaleDamp", prop.ordina_scale_damp);
                fill_array_double("AbscScaleStiff", prop.absc_scale_stiff);
                fill_array_double("OrdinaScaleStiff", prop.ordina_scale_stiff);

                if (cs_val.contains("RefTranVel") && cs_val["RefTranVel"].is_number()) {
                    prop.ref_tran_vel = cs_val["RefTranVel"].get<double>();
                }
                if (cs_val.contains("RefRotVel") && cs_val["RefRotVel"].is_number()) {
                    prop.ref_rot_vel = cs_val["RefRotVel"].get<double>();
                }
                if (cs_val.contains("SmoothStrRate") && cs_val["SmoothStrRate"].is_boolean()) {
                    prop.smooth_strain_rate = cs_val["SmoothStrRate"].get<bool>();
                }
                fill_array_double("RelaVecCoeff", prop.rela_vec_coeff);
                fill_array_double("RelaVecExp", prop.rela_vec_exp);
                fill_array_double("FailureScale", prop.failure_scale);
                fill_array_double("FailureExp", prop.failure_exp);

                registry.emplace<Component::BeamSpringProperty>(cs_entity, std::move(prop));
            }
            // 其它类型暂不做细化，仅保留 SetName 以便后续扩展
        }
    }

    if (j.contains("Material") && j["Material"].is_object()) {
        for (auto& [key, val] : j["Material"].items()) {
            const entt::entity mat_e = registry.create();

            // Material type (prefer Simdroid's "MaterialType", fallback to legacy "Type")
            const std::string mat_type = val.value("MaterialType", val.value("Type", ""));
            if (!mat_type.empty()) {
                registry.emplace<Component::MaterialModel>(mat_e, mat_type);
            }

            // Common: density + linear elastic constants (many plastic laws still require E, nu)
            Component::LinearElasticParams params{};
            if (val.contains("Density")) params.rho = val["Density"].get<double>();

            if (val.contains("MaterialConstants") && val["MaterialConstants"].is_object()) {
                const auto& cons = val["MaterialConstants"];

                // Support both canonical keys and older variants.
                if (cons.contains("ElasticModulus")) params.E = cons["ElasticModulus"].get<double>();
                else if (cons.contains("E")) params.E = cons["E"].get<double>();
                else if (cons.contains("YoungModulus")) params.E = cons["YoungModulus"].get<double>();

                if (cons.contains("PoissonRatio")) params.nu = cons["PoissonRatio"].get<double>();
                else if (cons.contains("Nu")) params.nu = cons["Nu"].get<double>();
                else if (cons.contains("nu")) params.nu = cons["nu"].get<double>();

                registry.emplace<Component::LinearElasticParams>(mat_e, params);

                // LAW2: IsotropicPlasticJC (Johnson-Cook)
                if (mat_type == "IsotropicPlasticJC") {
                    Component::IsotropicPlasticParams pl{};
                    if (cons.contains("YieldStress")) pl.yield_stress_A = cons["YieldStress"].get<double>();
                    if (cons.contains("HardeningCoefB")) pl.hardening_coef_B = cons["HardeningCoefB"].get<double>();
                    if (cons.contains("HardeningExpN")) pl.hardening_exp_n = cons["HardeningExpN"].get<double>();
                    if (cons.contains("RateCoef")) pl.rate_coef_C = cons["RateCoef"].get<double>();
                    if (cons.contains("HardeningMode")) pl.hardening_mode = cons["HardeningMode"].get<double>();
                    if (cons.contains("TemperatureExp")) pl.temperature_exp_m = cons["TemperatureExp"].get<double>();
                    if (cons.contains("MeltTemperature")) pl.melt_temperature = cons["MeltTemperature"].get<double>();
                    if (cons.contains("EnvTemperature")) pl.env_temperature = cons["EnvTemperature"].get<double>();
                    if (cons.contains("RefStrainRate")) pl.ref_strain_rate = cons["RefStrainRate"].get<double>();
                    if (cons.contains("SpecificHeat")) pl.specific_heat = cons["SpecificHeat"].get<double>();
                    registry.emplace<Component::IsotropicPlasticParams>(mat_e, std::move(pl));
                }

                // LAW36: RateDependentPlastic
                if (mat_type == "RateDependentPlastic") {
                    Component::RateDependentPlasticParams rdp{};
                    if (cons.contains("HardeningMode")) rdp.hardening_mode = cons["HardeningMode"].get<double>();
                    if (cons.contains("FailurePlasticStrain")) rdp.failure_plastic_strain = cons["FailurePlasticStrain"].get<double>();
                    if (cons.contains("FailBeginTensileStrain")) rdp.fail_begin_tensile_strain = cons["FailBeginTensileStrain"].get<double>();
                    if (cons.contains("FailEndTensileStrain")) rdp.fail_end_tensile_strain = cons["FailEndTensileStrain"].get<double>();
                    if (cons.contains("ElemDelTensileStrain")) rdp.elem_del_tensile_strain = cons["ElemDelTensileStrain"].get<double>();
                    if (cons.contains("StrainRateType")) rdp.strain_rate_type = cons["StrainRateType"].get<std::string>();
                    if (cons.contains("StrainAndStrainRateYieldCurve") && cons["StrainAndStrainRateYieldCurve"].is_array()) {
                        rdp.yield_curves = cons["StrainAndStrainRateYieldCurve"].get<std::vector<std::string>>();
                    }
                    if (cons.contains("StrainRate") && cons["StrainRate"].is_array()) {
                        rdp.strain_rates = cons["StrainRate"].get<std::vector<double>>();
                    }
                    registry.emplace<Component::RateDependentPlasticParams>(mat_e, std::move(rdp));
                }

                // -----------------------------------------------------------------
                // 超弹性材料 (Polynomial / ReducedPolynomial / Ogden*)
                // 解析顺序：
                //  1) 直接参数 -> PolynomialParams / ReducedPolynomialParams / OgdenParams
                //  2) 实验曲线 TestCurve-* -> HyperelasticMode::*_funcs (存 Curve 实体)
                // -----------------------------------------------------------------
                auto map_curve_names_to_entities = [&](const char* json_key, std::vector<entt::entity>& out_entities) {
                    if (!cons.contains(json_key) || !cons[json_key].is_array()) return;
                    for (const auto& name_val : cons[json_key]) {
                        if (!name_val.is_string()) continue;
                        const std::string name = name_val.get<std::string>();
                        entt::entity curve_e = get_or_create_curve_entity_by_name(name);
                        out_entities.push_back(curve_e);
                    }
                };

                Component::HyperelasticMode hyper{};
                bool has_hyperelastic = false;

                // --- Polynomial (full) ---
                if (mat_type == "Polynomial") {
                    // 直接参数模式
                    if (cons.contains("Order") && cons.contains("Const") && cons["Const"].is_array()) {
                        const int order = cons["Order"].get<int>();
                        std::vector<double> all;
                        all.reserve(cons["Const"].size());
                        for (const auto& v : cons["Const"]) {
                            if (v.is_number()) all.push_back(v.get<double>());
                        }

                        if (order > 0 && static_cast<int>(all.size()) >= order) {
                            const int total = static_cast<int>(all.size());
                            const int d_count = order;
                            const int c_count = total - d_count;

                            Component::PolynomialParams poly{};
                            poly.c_ij.assign(all.begin(), all.begin() + c_count);
                            poly.d_i.assign(all.begin() + c_count, all.end());

                            registry.emplace<Component::PolynomialParams>(mat_e, std::move(poly));

                            hyper.order = order;
                            hyper.fit_from_data = false;
                            hyper.nu = params.nu;
                            has_hyperelastic = true;
                        }
                    }
                }

                // --- ReducedPolynomial ---
                if (mat_type == "ReducedPolynomial") {
                    if (cons.contains("Order") && cons.contains("Const") && cons["Const"].is_array()) {
                        const int order = cons["Order"].get<int>();
                        std::vector<double> all;
                        all.reserve(cons["Const"].size());
                        for (const auto& v : cons["Const"]) {
                            if (v.is_number()) all.push_back(v.get<double>());
                        }

                        if (order > 0 && static_cast<int>(all.size()) == 2 * order) {
                            Component::ReducedPolynomialParams rpoly{};
                            rpoly.c_i0.assign(all.begin(), all.begin() + order);
                            rpoly.d_i.assign(all.begin() + order, all.end());

                            registry.emplace<Component::ReducedPolynomialParams>(mat_e, std::move(rpoly));

                            hyper.order = order;
                            hyper.fit_from_data = false;
                            // ReducedPolynomial 的泊松比可能来自上层字段
                            hyper.nu = params.nu;
                            has_hyperelastic = true;
                        }
                    }
                }

                // --- Ogden family (OgdenRubber / Ogden2 / Ogden) ---
                if (mat_type == "OgdenRubber" || mat_type == "Ogden2" || mat_type == "Ogden") {
                    Component::OgdenParams og{};

                    if (cons.contains("Mu") && cons["Mu"].is_array()) {
                        for (const auto& v : cons["Mu"]) {
                            if (v.is_number()) og.mu_i.push_back(v.get<double>());
                        }
                    }
                    if (cons.contains("Alpha") && cons["Alpha"].is_array()) {
                        for (const auto& v : cons["Alpha"]) {
                            if (v.is_number()) og.alpha_i.push_back(v.get<double>());
                        }
                    }
                    if (cons.contains("D") && cons["D"].is_array()) {
                        for (const auto& v : cons["D"]) {
                            if (v.is_number()) og.d_i.push_back(v.get<double>());
                        }
                    }

                    if (!og.mu_i.empty() && og.mu_i.size() == og.alpha_i.size()) {
                        registry.emplace<Component::OgdenParams>(mat_e, std::move(og));

                        hyper.order = static_cast<int>(og.mu_i.size());
                        hyper.fit_from_data = false;
                        // OgdenRubber 与 Ogden2/HyperFoam2 一般都使用 PoissonRatio
                        hyper.nu = params.nu;
                        has_hyperelastic = true;
                    } else if (!og.mu_i.empty() || !og.alpha_i.empty()) {
                        spdlog::warn("Ogden material '{}' has inconsistent Mu/Alpha lengths. Ignoring direct parameters.", key);
                    }
                }

                // --- 实验曲线模式 (TestCurve-*)，适用于 Polynomial / ReducedPolynomial / Ogden* / MooneyRivlin / Yeoh 等 ---
                bool has_curve = false;
                if (cons.contains("TestCurve-Uniaxial") || cons.contains("TestCurve-Biaxial") ||
                    cons.contains("TestCurve-Planar")   || cons.contains("TestCurve-Volumetric")) {
                    has_curve = true;
                }

                if (has_curve) {
                    hyper.fit_from_data = true;

                    // 阶数: Ogden2/Polynomial/HyperFoam 系的 CurveFit_n，缺省为 1
                    if (cons.contains("CurveFit_n") && cons["CurveFit_n"].is_number_integer()) {
                        hyper.order = cons["CurveFit_n"].get<int>();
                    } else if (hyper.order <= 0) {
                        hyper.order = 1;
                    }

                    // 泊松比：HyperFoam2 等有 CurveFit_Nu，其它退化为 PoissonRatio / 线弹性 nu
                    if (cons.contains("CurveFit_Nu") && cons["CurveFit_Nu"].is_number()) {
                        hyper.nu = cons["CurveFit_Nu"].get<double>();
                    } else if (cons.contains("PoissonRatio") && cons["PoissonRatio"].is_number()) {
                        hyper.nu = cons["PoissonRatio"].get<double>();
                    } else {
                        hyper.nu = params.nu;
                    }

                    map_curve_names_to_entities("TestCurve-Uniaxial",   hyper.uniaxial_funcs);
                    map_curve_names_to_entities("TestCurve-Biaxial",    hyper.biaxial_funcs);
                    map_curve_names_to_entities("TestCurve-Planar",     hyper.planar_funcs);
                    map_curve_names_to_entities("TestCurve-Volumetric", hyper.volumetric_funcs);

                    has_hyperelastic = true;
                }

                if (has_hyperelastic) {
                    registry.emplace<Component::HyperelasticMode>(mat_e, std::move(hyper));
                }
            } else {
                // Still store density-only materials to keep linkage stable.
                if (val.contains("Density")) {
                    registry.emplace<Component::LinearElasticParams>(mat_e, params);
                }
            }
            registry.emplace<Component::SetName>(mat_e, key);
        }
    }

    if (j.contains("PartProperty") && j["PartProperty"].is_object()) {
        for (auto it = j["PartProperty"].begin(); it != j["PartProperty"].end(); ++it) {
            const std::string part_key = it.key();
            const auto& part_info = it.value();

            std::string title = part_info.value("Title", part_key);
            std::string ele_set_name = part_info.value("EleSet", "");
            std::string mat_name = part_info.value("Material", "");
            std::string cs_name  = part_info.value("CrossSection", "");

            entt::entity ele_set_entity = entt::null;
            if (!ele_set_name.empty()) {
                ele_set_entity = get_or_create_set_entity(registry, ele_set_name);
                registry.get_or_emplace<Component::ElementSetMembers>(ele_set_entity);
            }

            entt::entity mat_entity = entt::null;
            if (!mat_name.empty()) {
                auto view = registry.view<Component::SetName, Component::LinearElasticParams>();
                for(auto e : view) {
                    if(view.get<Component::SetName>(e).value == mat_name) {
                        mat_entity = e;
                        break;
                    }
                }
            }

            entt::entity section_entity = entt::null;
            if (!cs_name.empty()) {
                auto cs_it = cross_section_map.find(cs_name);
                if (cs_it != cross_section_map.end()) {
                    section_entity = cs_it->second;
                } else {
                    // 退化情况：CrossSection 块缺失，但 PartProperty 引用了名称
                    section_entity = registry.create();
                    registry.emplace<Component::SetName>(section_entity, cs_name);
                    cross_section_map.emplace(cs_name, section_entity);
                }
            }

            const entt::entity part_entity = registry.create();
            Component::SimdroidPart part;
            part.name = std::move(title);
            part.element_set = ele_set_entity;
            part.material = mat_entity;
            part.section = section_entity;
            registry.emplace<Component::SimdroidPart>(part_entity, std::move(part));
        }
    }

    if (j.contains("Contact") && j["Contact"].is_object()) {
        for (auto& [contact_name, ci] : j["Contact"].items()) {
            const entt::entity e = registry.create();

            // --- ContactTypeTag ---
            const std::string type_str = ci.value("Type", "");
            const auto inter_type = lookup_enum(kContactTypeMap, type_str,
                                                Component::ContactInterType::Unknown);
            registry.emplace<Component::ContactTypeTag>(e, inter_type);

            // --- ContactBase (master / slave entity handles) ---
            Component::ContactBase base;
            base.name = contact_name;
            std::string master_name, slave_name;
            if (inter_type == Component::ContactInterType::General) {
                master_name = ci.value("Surf1", "");
                slave_name  = ci.value("Surf2", "");
            } else {
                master_name = ci.value("MasterFaces", "");
                slave_name  = ci.value("SlaveNodes", ci.value("SlaveFaces", ""));
            }
            if (!master_name.empty()) base.master_entity = get_or_create_set_entity(registry, master_name);
            if (!slave_name.empty())  base.slave_entity  = get_or_create_set_entity(registry, slave_name);
            registry.emplace<Component::ContactBase>(e, std::move(base));
            registry.emplace<Component::SetName>(e, contact_name);

            // --- ContactFormulation ---
            {
                Component::ContactFormulation form{};
                bool has_form = false;
                if (ci.contains("Formulation")) {
                    form.formulation = lookup_enum(kFormulationMap,
                        ci["Formulation"].get<std::string>(),
                        Component::ContactFormulationType::Standard);
                    has_form = true;
                }
                if (ci.contains("StiffnessFactor")) {
                    form.stiffness_factor = ci["StiffnessFactor"].get<double>();
                    has_form = true;
                }
                if (ci.contains("DampingCoefficient")) {
                    form.damping_coefficient = ci["DampingCoefficient"].get<double>();
                    has_form = true;
                }
                if (ci.contains("Damping")) {
                    form.damping_coefficient = ci["Damping"].get<double>();
                    has_form = true;
                }
                if (ci.contains("InterfaceStiffness")) {
                    form.interface_stiffness = lookup_enum(kInterfaceStiffnessMap,
                        ci["InterfaceStiffness"].get<std::string>(),
                        Component::InterfaceStiffnessType::Default);
                    has_form = true;
                }
                if (ci.contains("StiffnessForm")) {
                    if (ci["StiffnessForm"].is_number_integer()) {
                        form.stiffness_form = ci["StiffnessForm"].get<int>();
                    } else if (ci["StiffnessForm"].is_string()) {
                        auto it2 = kStiffnessFormMap.find(to_lower_copy(ci["StiffnessForm"].get<std::string>()));
                        if (it2 != kStiffnessFormMap.end()) form.stiffness_form = it2->second;
                    }
                    has_form = true;
                }
                if (ci.contains("MinStiffness")) { form.min_stiffness = ci["MinStiffness"].get<double>(); has_form = true; }
                if (ci.contains("MaxStiffness")) { form.max_stiffness = ci["MaxStiffness"].get<double>(); has_form = true; }
                if (ci.contains("SearchMethod")) {
                    form.search_method = lookup_enum(kSearchMethodMap,
                        ci["SearchMethod"].get<std::string>(),
                        Component::SearchMethodType::Segment);
                    has_form = true;
                }
                if (ci.contains("SearchDistance")) { form.search_distance = ci["SearchDistance"].get<double>(); has_form = true; }
                if (has_form) registry.emplace<Component::ContactFormulation>(e, form);
            }

            // --- ContactFriction ---
            {
                Component::ContactFriction fric{};
                bool has_fric = false;
                if (ci.contains("FrictionLaw")) {
                    fric.friction_law = lookup_enum(kFrictionLawMap,
                        ci["FrictionLaw"].get<std::string>(),
                        Component::FrictionLawType::Coulomb);
                    has_fric = true;
                }
                if (ci.contains("FrictionCoef")) { fric.friction_coef = ci["FrictionCoef"].get<double>(); has_fric = true; }
                if (ci.contains("Friction"))     { fric.friction_coef = ci["Friction"].get<double>();     has_fric = true; }
                if (ci.contains("FrictionCoefs") && ci["FrictionCoefs"].is_array()) {
                    const auto& arr = ci["FrictionCoefs"];
                    for (size_t k = 0; k < std::min(arr.size(), size_t(6)); ++k)
                        fric.friction_coefs[k] = arr[k].get<double>();
                    has_fric = true;
                }
                if (ci.contains("DampingInterStiff")) { fric.damping_inter_stiff = ci["DampingInterStiff"].get<double>(); has_fric = true; }
                if (ci.contains("DampingInterFric"))  { fric.damping_inter_fric  = ci["DampingInterFric"].get<double>();  has_fric = true; }
                if (has_fric) registry.emplace<Component::ContactFriction>(e, fric);
            }

            // --- ContactGapControl ---
            {
                Component::ContactGapControl gap{};
                bool has_gap = false;
                if (ci.contains("GapFlag")) {
                    gap.gap_flag = lookup_enum(kGapFlagMap,
                        ci["GapFlag"].get<std::string>(),
                        Component::GapFlagType::None);
                    has_gap = true;
                }
                if (ci.contains("GapScale"))      { gap.gap_scale       = ci["GapScale"].get<double>();      has_gap = true; }
                if (ci.contains("GapMin"))         { gap.gap_min         = ci["GapMin"].get<double>();         has_gap = true; }
                if (ci.contains("GapMax"))         { gap.gap_max         = ci["GapMax"].get<double>();         has_gap = true; }
                if (ci.contains("SlaveGapMax"))    { gap.slave_gap_max   = ci["SlaveGapMax"].get<double>();    has_gap = true; }
                if (ci.contains("MasterGapMax"))   { gap.master_gap_max  = ci["MasterGapMax"].get<double>();   has_gap = true; }
                if (has_gap) registry.emplace<Component::ContactGapControl>(e, gap);
            }

            // --- ContactTieData (TYPE2-specific: Ignore, fail mode, curve refs) ---
            if (inter_type == Component::ContactInterType::Tie) {
                Component::ContactTieData tie{};
                if (ci.contains("Ignore")) {
                    tie.ignore = lookup_enum(kIgnoreMap,
                        ci["Ignore"].get<std::string>(),
                        Component::IgnoreType::MainThick);
                }
                if (ci.contains("FailMode")) {
                    tie.fail_mode = lookup_enum(kFailModeMap,
                        ci["FailMode"].get<std::string>(),
                        Component::FailModeType::None);
                }
                if (ci.contains("RuptureFlag")) {
                    tie.rupture_flag = lookup_enum(kRuptureFlagMap,
                        ci["RuptureFlag"].get<std::string>(),
                        Component::RuptureFlagType::TractionCompress);
                }
                if (ci.contains("Max_N_Dist")) tie.max_n_dist = ci["Max_N_Dist"].get<double>();
                if (ci.contains("Max_T_Dist")) tie.max_t_dist = ci["Max_T_Dist"].get<double>();

                // Curve references (via DofMap)
                auto resolve_curve = [&](const char* key) -> entt::entity {
                    if (!ci.contains(key) || !ci[key].is_string()) return entt::null;
                    std::string cname = ci[key].get<std::string>();
                    trim(cname);
                    if (cname.empty() || !dof_map) return entt::null;
                    auto cit = dof_map->curve_name_to_entity.find(cname);
                    if (cit != dof_map->curve_name_to_entity.end()) return cit->second;
                    return get_or_create_curve_entity_by_name(cname);
                };
                tie.stress_vs_stress_rate = resolve_curve("StressVsStressRate");
                tie.nor_stress_vs_disp    = resolve_curve("NorStressVsDisp");
                tie.tang_stress_vs_disp   = resolve_curve("TangStressVsDisp");

                registry.emplace<Component::ContactTieData>(e, tie);
            }

            spdlog::info("  -> Contact '{}' (Type={}) created.", contact_name, type_str);
        }
    }

    if (j.contains("Constraint") && j["Constraint"].is_object()) {
        spdlog::info("Parsing Constraints from Simdroid Control...");
        const auto& j_cons = j["Constraint"];

        // 1) Boundary Conditions (SPC)
        if (j_cons.contains("Boundary")) {
            parse_boundary_conditions(j_cons["Boundary"], registry);
        }

        // 2) Rigid Bodies (MPC)
        if (j_cons.contains("NodalRigidBody")) {
            parse_rigid_bodies(j_cons["NodalRigidBody"], registry);
        }

        // 3) Distributing Couplings (RBE3-like) - treated similar to rigid bodies for graph connectivity
        if (j_cons.contains("DistributingCoupling")) {
            parse_rigid_bodies(j_cons["DistributingCoupling"], registry);
        }

        // 4) [新增] Rigid Walls
        if (j_cons.contains("RigidWall")) {
            spdlog::info("Parsing RigidWalls...");
            parse_rigid_walls(j_cons["RigidWall"], registry);
        }
    }

    // [新增] Radioss /RBODY rigid bodies (top-level)
    if (j.contains("RigidBody") && j["RigidBody"].is_object()) {
        spdlog::info("Parsing Radioss RigidBodies...");
        parse_radioss_rigid_bodies(j["RigidBody"], registry);
    }

    if (j.contains("Load") && j["Load"].is_object()) {
        spdlog::info("Parsing Loads from Simdroid Control...");
        parse_loads(j["Load"], registry);
    }

    // [新增] Initial Conditions
    if (j.contains("InitialCondition") && j["InitialCondition"].is_object()) {
        spdlog::info("Parsing Initial Conditions...");
        parse_initial_conditions(j["InitialCondition"], registry);
    }

    // [新增] Analysis Settings (Step)
    if (j.contains("Step") && j["Step"].is_object()) {
        spdlog::info("Parsing Analysis Settings...");
        parse_analysis_settings(j["Step"], registry, ctx);
    }

    // Post-parse: validate entity references & cross-constraint overlap
    validate_contacts(registry);
    validate_rigid_bodies(registry);
    validate_cross_constraints(registry);
}

void SimdroidParser::validate_constraints(entt::registry& registry) {
    validate_contacts(registry);
    validate_rigid_bodies(registry);
    validate_cross_constraints(registry);
}

void SimdroidParser::list_constraint_warnings(entt::registry& registry) {
    if (!registry.ctx().contains<ConstraintWarningReport>()) {
        spdlog::info("Constraint warnings: no cached report (run import_simdroid or validate_constraints first).");
        return;
    }

    const auto& report = registry.ctx().get<ConstraintWarningReport>();
    if (report.cross_conflicts.empty()) {
        spdlog::info("Constraint warnings: no cross-constraint master/slave conflicts cached.");
        return;
    }

    spdlog::warn("Cross-constraint conflict details ({} node(s)):", report.cross_conflicts.size());
    const std::size_t max_show = 50;
    for (std::size_t i = 0; i < report.cross_conflicts.size() && i < max_show; ++i) {
        const auto& c = report.cross_conflicts[i];

        auto join = [](const std::vector<std::string>& v) {
            std::string out;
            for (std::size_t k = 0; k < v.size(); ++k) {
                if (k) out += ", ";
                out += v[k];
            }
            return out;
        };

        spdlog::warn("  - Node {}: master=[{}], slave=[{}]", c.node_id, join(c.as_master), join(c.as_slave));
    }
    if (report.cross_conflicts.size() > max_show) {
        spdlog::warn("  ... {} more node(s) not shown.", report.cross_conflicts.size() - max_show);
    }
}

entt::entity SimdroidParser::find_set_by_name(entt::registry& registry, const std::string& name) {
    auto view = registry.view<const Component::SetName>();
    for (auto entity : view) {
        if (view.get<const Component::SetName>(entity).value == name) {
            return entity;
        }
    }
    return entt::null;
}

void SimdroidParser::parse_boundary_conditions(const json& j_bcs, entt::registry& registry) {
    int next_boundary_id = 1;
    for (auto& [key, val] : j_bcs.items()) {
        std::string set_name = val.value("NodeSet", "");
        if (set_name.empty()) set_name = val.value("Set", ""); // fallback for older files
        if (set_name.empty()) continue;

        entt::entity set_entity = find_set_by_name(registry, set_name);
        if (set_entity == entt::null || !registry.all_of<Component::NodeSetMembers>(set_entity)) {
            spdlog::warn("Boundary '{}' refers to unknown Node Set '{}'", key, set_name);
            continue;
        }

        struct DofConfig { std::string axis; double val; };
        std::vector<DofConfig> target_dofs;

        // Simdroid format variant: "Dofs": [1,1,1,1,1,1] indicates constrained DOFs
        // Order convention: [Ux, Uy, Uz, Rx, Ry, Rz]
        if (val.contains("Dofs") && val["Dofs"].is_array()) {
            const auto& arr = val["Dofs"];
            const auto is_on = [&](size_t idx) -> bool {
                if (idx >= arr.size()) return false;
                if (arr[idx].is_boolean()) return arr[idx].get<bool>();
                if (arr[idx].is_number_integer()) return arr[idx].get<int>() != 0;
                if (arr[idx].is_number()) return std::abs(arr[idx].get<double>()) > 1e-12;
                return false;
            };

            if (is_on(0)) target_dofs.push_back({"x", 0.0});
            if (is_on(1)) target_dofs.push_back({"y", 0.0});
            if (is_on(2)) target_dofs.push_back({"z", 0.0});
            if (is_on(3)) target_dofs.push_back({"rx", 0.0});
            if (is_on(4)) target_dofs.push_back({"ry", 0.0});
            if (is_on(5)) target_dofs.push_back({"rz", 0.0});
        } else {
            const std::string type = val.value("Type", "Displacement");

            if (type == "Fixed" || type == "Encastre") {
                target_dofs = {{"x", 0.0}, {"y", 0.0}, {"z", 0.0}, {"rx", 0.0}, {"ry", 0.0}, {"rz", 0.0}};
            } else if (type == "Pinned") {
                target_dofs = {{"x", 0.0}, {"y", 0.0}, {"z", 0.0}};
            } else {
                // "Displacement" or specific type
                if (val.contains("U1") || val.contains("X")) target_dofs.push_back({"x", val.value("U1", val.value("X", 0.0))});
                if (val.contains("U2") || val.contains("Y")) target_dofs.push_back({"y", val.value("U2", val.value("Y", 0.0))});
                if (val.contains("U3") || val.contains("Z")) target_dofs.push_back({"z", val.value("U3", val.value("Z", 0.0))});
            }
        }

        if (target_dofs.empty()) continue;

        const auto& node_members = registry.get<Component::NodeSetMembers>(set_entity).members;

        for (const auto& cfg : target_dofs) {
            auto bc_entity = registry.create();
            registry.emplace<Component::BoundaryID>(bc_entity, next_boundary_id++);
            registry.emplace<Component::BoundarySPC>(bc_entity, 1, cfg.axis, cfg.val);
            registry.emplace<Component::SetName>(bc_entity, key); // Debug info

            for (auto node_e : node_members) {
                if (!registry.valid(node_e)) continue;
                auto& applied = registry.get_or_emplace<Component::AppliedBoundaryRef>(node_e);
                applied.boundary_entities.push_back(bc_entity);
            }
        }
        spdlog::info("  -> Applied Boundary '{}' to {} nodes.", key, node_members.size());
    }
}

void SimdroidParser::parse_rigid_bodies(const json& j_rbs, entt::registry& registry) {
    for (auto& [key, val] : j_rbs.items()) {
        const std::string m_set_name = val.value("MasterNodeSet", "");
        const std::string s_set_name = val.value("SlaveNodeSet", "");

        entt::entity m_set = find_set_by_name(registry, m_set_name);
        entt::entity s_set = find_set_by_name(registry, s_set_name);

        const bool m_ok = (m_set != entt::null) && registry.all_of<Component::NodeSetMembers>(m_set);
        const bool s_ok = (s_set != entt::null) && registry.all_of<Component::NodeSetMembers>(s_set);

        if (m_ok && s_ok) {
            auto constraint_entity = registry.create();
            Component::RigidBodyConstraint rbc;
            rbc.master_node_set = m_set;
            rbc.slave_node_set = s_set;

            registry.emplace<Component::RigidBodyConstraint>(constraint_entity, rbc);
            registry.emplace<Component::SetName>(constraint_entity, key);

            spdlog::info("  -> Created RigidBody '{}' between '{}' and '{}'", key, m_set_name, s_set_name);
        } else {
            spdlog::warn("RigidBody '{}' missing sets: Master='{}', Slave='{}'", key, m_set_name, s_set_name);
        }
    }
}

void SimdroidParser::parse_loads(const json& j_loads, entt::registry& registry) {
    DofMap* dof_map = registry.ctx().contains<DofMap>() ? &registry.ctx().get<DofMap>() : nullptr;

    int next_load_id = 1;
    for (auto& [key, val] : j_loads.items()) {
        std::string type = val.value("Type", "");

        // 1. Handle Nodal Loads (Force, Moment)
        if (type == "Force" || type == "Moment" || type == "ConcentratedForce") {
            std::string set_name = val.value("NodeSet", "");
            if (set_name.empty()) set_name = val.value("Set", ""); // fallback for older files
            if (set_name.empty()) continue;

            entt::entity set_entity = find_set_by_name(registry, set_name);
            if (set_entity == entt::null || !registry.all_of<Component::NodeSetMembers>(set_entity)) {
                spdlog::warn("Load '{}' refers to unknown NodeSet '{}'", key, set_name);
                continue;
            }

            const bool is_moment = (type == "Moment");
            struct LoadComp { std::string axis; double val; };
            std::vector<LoadComp> components;

            // Simdroid 格式：Dof + Value（单分量），与 Material 的 Function 类似可带 TimeCurve
            if (val.contains("Dof")) {
                std::string dof_str = to_lower_copy(val["Dof"].get<std::string>());
                double value = val.value("Value", val.value("Magnitude", 0.0));
                if (val.contains("Mag")) value = val["Mag"].get<double>();
                if (is_moment && dof_str.size() == 1) {
                    if (dof_str == "x") dof_str = "rx";
                    else if (dof_str == "y") dof_str = "ry";
                    else if (dof_str == "z") dof_str = "rz";
                }
                if (std::abs(value) > 1e-12) components.push_back({dof_str, value});
            } else {
                double mag = val.value("Magnitude", 0.0);
                if (val.contains("Mag")) mag = val["Mag"].get<double>();
                const bool has_mag = val.contains("Magnitude") || val.contains("Mag");

                double fx = 0.0, fy = 0.0, fz = 0.0;

                if (val.contains("Direction") && val["Direction"].is_array()) {
                    const auto& d = val["Direction"];
                    if (d.size() >= 3) {
                        double dx = d[0].get<double>();
                        double dy = d[1].get<double>();
                        double dz = d[2].get<double>();
                        std::tie(dx, dy, dz) = normalize(dx, dy, dz);
                        fx = mag * dx;
                        fy = mag * dy;
                        fz = mag * dz;
                    }
                } else {
                    const double x = val.value("X", 0.0);
                    const double y = val.value("Y", 0.0);
                    const double z = val.value("Z", 0.0);

                    if (has_mag && std::abs(mag) > 0.0) {
                        double dx = x, dy = y, dz = z;
                        std::tie(dx, dy, dz) = normalize(dx, dy, dz);
                        fx = mag * dx;
                        fy = mag * dy;
                        fz = mag * dz;
                    } else {
                        fx = x;
                        fy = y;
                        fz = z;
                    }
                }

                const char* ax_x = is_moment ? "rx" : "x";
                const char* ax_y = is_moment ? "ry" : "y";
                const char* ax_z = is_moment ? "rz" : "z";
                if (std::abs(fx) > 1e-12) components.push_back({ax_x, fx});
                if (std::abs(fy) > 1e-12) components.push_back({ax_y, fy});
                if (std::abs(fz) > 1e-12) components.push_back({ax_z, fz});
            }

            if (components.empty()) continue;

            const auto& node_members = registry.get<Component::NodeSetMembers>(set_entity).members;

            // TimeCurve：从 ctx 中的 curve 名->实体映射解析（与 Material 的 Function 预解析一致）
            entt::entity curve_entity = entt::null;
            if (dof_map && val.contains("TimeCurve") && val["TimeCurve"].is_string()) {
                std::string curve_name = val["TimeCurve"].get<std::string>();
                trim(curve_name);
                if (!curve_name.empty()) {
                    auto it = dof_map->curve_name_to_entity.find(curve_name);
                    if (it != dof_map->curve_name_to_entity.end() && registry.valid(it->second)) {
                        curve_entity = it->second;
                    } else {
                        spdlog::warn("Load '{}' TimeCurve '{}' not found in Function.", key, curve_name);
                    }
                }
            }

            for (const auto& comp : components) {
                auto load_def_entity = registry.create();
                registry.emplace<Component::LoadID>(load_def_entity, next_load_id++);
                registry.emplace<Component::NodalLoad>(load_def_entity, is_moment ? 2 : 1, comp.axis, comp.val, curve_entity);
                registry.emplace<Component::SetName>(load_def_entity, key);

                for (auto node_entity : node_members) {
                    if (!registry.valid(node_entity)) continue;
                    auto& applied = registry.get_or_emplace<Component::AppliedLoadRef>(node_entity);
                    applied.load_entities.push_back(load_def_entity);
                }
            }
            spdlog::info("  -> Applied {} '{}' to {} nodes.", type, key, node_members.size());
        }
        // 2. Handle Base Acceleration (Gravity / Ground Acceleration) - /GRAV
        else if (type == "BaseAcceleration") {
            std::string set_name = val.value("NodeSet", "");
            if (set_name.empty()) set_name = val.value("Set", "");

            std::vector<entt::entity> target_nodes;
            if (!set_name.empty()) {
                entt::entity set_entity = find_set_by_name(registry, set_name);
                if (set_entity == entt::null || !registry.all_of<Component::NodeSetMembers>(set_entity)) {
                    spdlog::warn("Load '{}' refers to unknown NodeSet '{}'", key, set_name);
                    continue;
                }
                target_nodes = registry.get<Component::NodeSetMembers>(set_entity).members;
            } else {
                // NodeSet not provided -> apply to all nodes
                auto view_nodes = registry.view<const Component::NodeID>();
                for (auto node_entity : view_nodes) target_nodes.push_back(node_entity);
            }

            Component::BaseAccelerationLoad grav{};
            grav.ax = val.value("XValue", 0.0);
            grav.ay = val.value("YValue", 0.0);
            grav.az = val.value("ZValue", 0.0);
            grav.coord_sys = val.value("CoordSys", "");
            trim(grav.coord_sys);

            auto resolve_curve = [&](const char* field) -> entt::entity {
                if (!dof_map || !val.contains(field) || !val[field].is_string()) return entt::null;
                std::string curve_name = val[field].get<std::string>();
                trim(curve_name);
                if (curve_name.empty()) return entt::null;
                auto it = dof_map->curve_name_to_entity.find(curve_name);
                if (it != dof_map->curve_name_to_entity.end() && registry.valid(it->second)) return it->second;
                spdlog::warn("Load '{}' {} '{}' not found in Function.", key, field, curve_name);
                return entt::null;
            };

            grav.x_curve_entity = resolve_curve("XTimeCurve");
            grav.y_curve_entity = resolve_curve("YTimeCurve");
            grav.z_curve_entity = resolve_curve("ZTimeCurve");

            // Skip if nothing to apply
            const bool has_any =
                (std::abs(grav.ax) > 1e-12) || (std::abs(grav.ay) > 1e-12) || (std::abs(grav.az) > 1e-12) ||
                (grav.x_curve_entity != entt::null) || (grav.y_curve_entity != entt::null) || (grav.z_curve_entity != entt::null);
            if (!has_any || target_nodes.empty()) continue;

            auto load_def_entity = registry.create();
            registry.emplace<Component::LoadID>(load_def_entity, next_load_id++);
            registry.emplace<Component::BaseAccelerationLoad>(load_def_entity, grav);
            registry.emplace<Component::SetName>(load_def_entity, key);

            int applied_count = 0;
            for (auto node_entity : target_nodes) {
                if (!registry.valid(node_entity)) continue;
                auto& applied = registry.get_or_emplace<Component::AppliedLoadRef>(node_entity);
                applied.load_entities.push_back(load_def_entity);
                applied_count++;
            }
            spdlog::info("  -> Applied BaseAcceleration '{}' to {} nodes.", key, applied_count);
        }
        // 3. Handle Pressure (Element Load) - Placeholder
        else if (type == "Pressure") {
            std::string set_name = val.value("EleSet", "");
            spdlog::info("  -> Found Pressure Load '{}' on EleSet '{}'. (Solver conversion pending)", key, set_name);
        }
    }
}

// =========================================================
// 实现：初始条件 (Initial Velocity)
// =========================================================
void SimdroidParser::parse_initial_conditions(const json& j_ics, entt::registry& registry) {
    for (auto& [key, val] : j_ics.items()) {
        if (!val.is_object()) continue;

        std::string type = val.value("Type", "");
        const std::string type_l = to_lower_copy(type);

        // Only handle initial velocity for now
        if (type_l != "velocity" && type_l != "initialvelocity" && type_l != "initial_velocity") continue;

        std::string set_name = val.value("NodeSet", "");
        if (set_name.empty()) set_name = val.value("Set", "");
        if (set_name.empty()) {
            spdlog::warn("InitialCondition '{}' missing NodeSet/Set field.", key);
            continue;
        }

        entt::entity set_entity = find_set_by_name(registry, set_name);
        if (set_entity == entt::null || !registry.all_of<Component::NodeSetMembers>(set_entity)) {
            spdlog::warn("InitialCondition '{}' refers to unknown NodeSet '{}'", key, set_name);
            continue;
        }

        const auto& node_members = registry.get<Component::NodeSetMembers>(set_entity).members;

        double vx = val.value("X", 0.0);
        double vy = val.value("Y", 0.0);
        double vz = val.value("Z", 0.0);

        // If Magnitude + Direction are provided, they override X/Y/Z
        if (val.contains("Magnitude") && val.contains("Direction") && val["Direction"].is_array()) {
            const double mag = val["Magnitude"].get<double>();
            const auto& d = val["Direction"];
            if (d.size() >= 3) {
                double nx = d[0].get<double>();
                double ny = d[1].get<double>();
                double nz = d[2].get<double>();
                std::tie(nx, ny, nz) = normalize(nx, ny, nz);
                vx = mag * nx;
                vy = mag * ny;
                vz = mag * nz;
            }
        }

        int count = 0;
        for (auto node_e : node_members) {
            if (!registry.valid(node_e)) continue;
            registry.emplace_or_replace<Component::Velocity>(node_e, vx, vy, vz);
            count++;
        }
        spdlog::info("  -> Applied Initial Velocity ({}, {}, {}) to {} nodes.", vx, vy, vz, count);
    }
}

// =========================================================
// 实现：刚性墙 (Rigid Wall)
// =========================================================
void SimdroidParser::parse_rigid_walls(const json& j_rw, entt::registry& registry) {
    int next_wall_id = 1;
    for (auto& [key, val] : j_rw.items()) {
        if (!val.is_object()) continue;

        const auto rw_entity = registry.create();

        Component::RigidWall rw{};
        rw.id = next_wall_id++;
        rw.type = val.value("Type", "Planar");
        rw.secondary_node_set = entt::null;

        // Direct parameters (optional)
        if (val.contains("Parameters") && val["Parameters"].is_array()) {
            rw.parameters = val["Parameters"].get<std::vector<double>>();
        } else {
            // Planar: Normal + Point -> ax+by+cz+d=0
            std::vector<double> normal = {0.0, 0.0, 1.0};
            std::vector<double> point = {0.0, 0.0, 0.0};

            if (val.contains("Normal") && val["Normal"].is_array()) normal = val["Normal"].get<std::vector<double>>();
            if (val.contains("Point") && val["Point"].is_array()) point = val["Point"].get<std::vector<double>>();

            if (normal.size() >= 3 && point.size() >= 3) {
                double nx = normal[0], ny = normal[1], nz = normal[2];
                std::tie(nx, ny, nz) = normalize(nx, ny, nz);
                const double d = -(nx * point[0] + ny * point[1] + nz * point[2]);
                rw.parameters = {nx, ny, nz, d};
            }
        }

        // Secondary/slave node set (optional)
        std::string slave_set_name = val.value("SecondaryNodes", "");
        if (slave_set_name.empty()) slave_set_name = val.value("SlaveNodes", "");
        if (!slave_set_name.empty()) {
            const entt::entity slave_set = find_set_by_name(registry, slave_set_name);
            if (slave_set != entt::null) {
                rw.secondary_node_set = slave_set;
            } else {
                spdlog::warn("RigidWall '{}' refers to unknown NodeSet '{}'", key, slave_set_name);
            }
        }

        registry.emplace<Component::RigidWall>(rw_entity, rw);
        registry.emplace<Component::SetName>(rw_entity, key);

        spdlog::info("  -> Created RigidWall '{}' ({})", key, rw.type);
    }
}

// =========================================================
// 实现：分析设置 (Analysis Settings)
// =========================================================
void SimdroidParser::parse_analysis_settings(const json& j_step, entt::registry& registry, DataContext& ctx) {
    if (!j_step.is_object() || j_step.empty()) return;

    const json* step_cfg = &j_step;
    const bool looks_like_step_cfg =
        j_step.contains("Type") || j_step.contains("EndTime") || j_step.contains("Duration") ||
        j_step.contains("TimeStep") || j_step.contains("Output");

    if (!looks_like_step_cfg) {
        auto it = j_step.begin();
        if (it != j_step.end() && it.value().is_object()) {
            step_cfg = &it.value();
        }
    }

    // Analysis entity (singleton)
    entt::entity analysis_entity = ctx.analysis_entity;
    if (analysis_entity == entt::null || !registry.valid(analysis_entity)) {
        analysis_entity = registry.create();
        ctx.analysis_entity = analysis_entity;
    }

    const std::string type = step_cfg->value("Type", "Explicit");
    registry.emplace_or_replace<Component::AnalysisType>(analysis_entity, type);

    double end_time = step_cfg->value("EndTime", 1.0);
    if (step_cfg->contains("Duration") && (*step_cfg)["Duration"].is_number()) {
        end_time = (*step_cfg)["Duration"].get<double>();
    }
    registry.emplace_or_replace<Component::EndTime>(analysis_entity, end_time);

    if (step_cfg->contains("TimeStep") && (*step_cfg)["TimeStep"].is_number()) {
        const double dt = (*step_cfg)["TimeStep"].get<double>();
        registry.emplace_or_replace<Component::FixedTimeStep>(analysis_entity, dt);
    }

    // Output entity (singleton) - used by explicit solver
    entt::entity output_entity = ctx.output_entity;
    if (output_entity == entt::null || !registry.valid(output_entity)) {
        output_entity = registry.create();
        ctx.output_entity = output_entity;
    }

    double interval = (end_time > 0.0 ? end_time / 20.0 : 0.0);
    if (step_cfg->contains("Output") && (*step_cfg)["Output"].is_object()) {
        const auto& out = (*step_cfg)["Output"];
        interval = (end_time > 0.0 ? end_time / 100.0 : 0.0);
        interval = out.value("Interval", interval);
        if (out.contains("Frequency") && out["Frequency"].is_number()) {
            const double freq = out["Frequency"].get<double>();
            if (freq > 0.0) interval = 1.0 / freq;
        }
    }

    registry.emplace_or_replace<Component::OutputControl>(output_entity, interval);
    registry.emplace_or_replace<Component::OutputIntervalTime>(output_entity, interval);

    spdlog::info("  -> Analysis Configured: Type={}, EndTime={}, OutputInterval={}", type, end_time, interval);
}

void SimdroidParser::parse_mesh_dat(const std::string& path, DataContext& ctx) {
    MeshSetDefs defs;
    collect_set_definitions_from_file(path, defs);

    auto& registry = ctx.registry;
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("cannot open mesh file");

    node_lookup.clear();
    element_lookup.clear();
    surface_lookup.clear();

    std::string line;
    std::string pending_element_line;
    std::string pending_surface_line;
    bool in_node_section = false;
    bool in_element_section = false;
    bool in_element_type_section = false;
    bool in_surface_section = false;
    bool in_surface_type_section = false;
    bool in_skip_block = false;     // Skip "Set { ... }" and "Part { ... }" definitions during geometry parsing
    int  skip_brace_level = 0;
    int current_element_type_id = 0;
    std::string current_element_block_name;
    std::string current_surface_block_name;
    
    // Safety reserve
    node_lookup.resize(10000, entt::null);
    element_lookup.resize(10000, entt::null);
    surface_lookup.resize(10000, entt::null);

    while (std::getline(file, line)) {
        preprocess_line(line);
        if (line.empty()) continue;

        // Skip definition blocks (Set/Part) — members are already collected by collect_set_definitions_from_file().
        if (in_skip_block) {
            if (ends_with(line, "{")) ++skip_brace_level;
            if (line == "}") {
                --skip_brace_level;
                if (skip_brace_level <= 0) {
                    in_skip_block = false;
                    skip_brace_level = 0;
                }
            }
            continue;
        } else {
            // Only start skipping when we're not inside core geometry blocks
            if (!in_node_section && !in_element_section && !in_surface_section) {
                if (line == "Set {") {
                    in_skip_block = true;
                    skip_brace_level = 1;
                    continue;
                }
                if (line == "Part {") {
                    in_skip_block = true;
                    skip_brace_level = 1;
                    continue;
                }
            }
        }

        if (line == "Node {") {
            in_node_section = true;
            continue;
        }
        if (line == "Element {") {
            in_element_section = true;
            current_element_block_name.clear();
            continue;
        }
        if (line == "Surface {") {
            in_surface_section = true;
            current_surface_block_name.clear();
            continue;
        }
        if (ends_with(line, "{")) {
            if (in_element_section) {
                in_element_type_section = true;
                // Extract element block name (e.g., "Hex8" from "Hex8 {")
                const size_t end_pos = line.find('{');
                std::string raw_name = (end_pos == std::string::npos) ? line : line.substr(0, end_pos);
                trim(raw_name);
                current_element_block_name = to_lower_copy(raw_name);
            } else if (in_surface_section) {
                in_surface_type_section = true;
                const size_t end_pos = line.find('{');
                std::string raw_name = (end_pos == std::string::npos) ? line : line.substr(0, end_pos);
                trim(raw_name);
                current_surface_block_name = to_lower_copy(raw_name);
            }
            continue;
        }
        if (line == "}") {
            if (in_element_type_section) {
                in_element_type_section = false;
                current_element_block_name.clear();
            } else if (in_surface_type_section) {
                in_surface_type_section = false;
                current_surface_block_name.clear();
            } else {
                in_node_section = false;
                in_element_section = false;
                in_surface_section = false;
            }
            continue;
        }

        // --- Node Parsing ---
        // --- Node Parsing ---
        if (in_node_section) {
            // [删除] 之前的暴力跳过逻辑，因为现在的坐标定义里包含了 '['
            // if (line.find('[') != std::string::npos) continue;
            
            // 1. 数据清洗：将 [, ], , 替换为空格，统一格式
            std::string clean_line = line;
            std::replace(clean_line.begin(), clean_line.end(), '[', ' ');
            std::replace(clean_line.begin(), clean_line.end(), ']', ' ');
            std::replace(clean_line.begin(), clean_line.end(), ',', ' ');

            std::stringstream ss(clean_line);
            int nid;
            double x, y, z;
            
            // 2. 尝试提取 ID, X, Y, Z
            if (!(ss >> nid >> x >> y >> z)) {
                // 如果提取失败，说明这行可能不是坐标定义（可能是 Node_Set 定义，如 "123_Set [list]"）
                // 我们在 debug 模式下记录一下，但不视为错误，直接跳过
                // spdlog::debug("Skipping non-coordinate line in Node block: {}", line);
                continue;
            }
            
            if (nid < 0) continue;

            // Expand lookup
            if (static_cast<size_t>(nid) >= node_lookup.size()) {
                // 稍微多分配一点，防止频繁扩容
                size_t new_size = std::max(static_cast<size_t>(nid) * 2, node_lookup.size() + 10000); 
                node_lookup.resize(new_size, entt::null);
            }

            auto e = registry.create();
            registry.emplace<Component::Position>(e, x, y, z);
            registry.emplace<Component::NodeID>(e, nid);
            registry.emplace<Component::OriginalID>(e, nid); 
            
            node_lookup[nid] = e;
        }
        
        // --- Element Parsing ---
        if (in_element_section) {
            // Support wrapped lists: element connectivity may span multiple lines inside '[ ... ]'
            std::string elem_line = line;
            if (!pending_element_line.empty()) {
                pending_element_line += " " + elem_line;
                if (pending_element_line.find(']') == std::string::npos) continue;
                elem_line = pending_element_line;
                pending_element_line.clear();
            } else {
                const size_t lb0 = elem_line.find('[');
                if (lb0 != std::string::npos && elem_line.find(']') == std::string::npos) {
                    pending_element_line = elem_line;
                    continue;
                }
            }

            size_t lb = elem_line.find('[');
            size_t rb = elem_line.rfind(']');
            if (lb == std::string::npos || rb == std::string::npos) continue;

            // 1. 解析 Element ID
            int eid = 0;
            try {
                // 处理 ID 后面的潜在逗号 (e.g. "357298, [...")
                std::string id_str = elem_line.substr(0, lb);
                std::replace(id_str.begin(), id_str.end(), ',', ' '); 
                eid = std::stoi(id_str);
            } catch (...) { continue; }

            if (eid < 0) continue;

            // 2. 解析 Node IDs (关键修复)
            std::string content = elem_line.substr(lb + 1, rb - lb - 1);
            
            // [修复] 暴力替换所有逗号为空格，解决 "1,2" 和 "1 2" 的兼容性问题
            std::replace(content.begin(), content.end(), ',', ' ');
            
            std::vector<int> node_ids;
            std::stringstream ss(content);
            int nid;
            while (ss >> nid) {
                node_ids.push_back(nid);
            }

            // 3. 验证节点有效性
            // 即使是退化单元（重复节点），只要 ID 存在于 lookup 表中就是有效的
            std::vector<entt::entity> valid_node_entities;
            valid_node_entities.reserve(node_ids.size());
            bool is_element_broken = false;

            for (int id : node_ids) {
                if (id >= 0 && static_cast<size_t>(id) < node_lookup.size() && node_lookup[id] != entt::null) {
                    valid_node_entities.push_back(node_lookup[id]);
                } else {
                    spdlog::warn("Element {} refers to undefined Node ID: {}", eid, id);
                    is_element_broken = true;
                    break;
                }
            }

            if (is_element_broken) continue;

            // 4. 类型推断 (包含你列出的所有类型)
            int type_id = 0;
            size_t count = node_ids.size();
            
            // 优先信任 count，因为 Simdroid 的块结构可能仅仅是分组
            if (count == 8) type_id = 308;      // Hex8 (包括退化为 Wedge 的情况)
            else if (count == 4) type_id = 304; // Tet4 (Simdroid Tet4) 或 Quad4 (Shell)
            else if (count == 10) type_id = 310;// Tet10
            else if (count == 20) type_id = 320;// Hex20
            else if (count == 3) type_id = 203; // Tri3
            else if (count == 2) type_id = 102; // Line2
            
            // 如果推断出的类型和 block 类型不一致，记录一下（但在 Simdroid 中通常以节点数为准）
            // 例如 Quad4 也是 4节点，Tet4 也是 4节点。
            // 这里可以通过 in_element_type_section 上下文来修正
            // 但为了通用性，先按节点数处理。
            // *特殊处理*: 如果是 2D 单元 Quad4 (4节点)，但被误判为 Tet4 (304)
            // 通常 3D 求解器里 304 是体单元。如果是壳单元，通常 ID 不同。
            // 鉴于目前是 Simdroid 结构求解，4节点大概率是 Tet4。
            // 如果你的 Quad4 是壳单元，可能需要根据 Element ID 范围或额外的 Property 来区分。
            // 暂时映射：4节点 -> 304 (Tet4) 或者是 204 (Quad4)? 
            // 如果 Simdroid 输入中有 Quad4，我们需要看看它的 Block Name。
            
            // 简单修正逻辑：
            if (count == 4) {
                if (current_element_block_name.find("quad") != std::string::npos) type_id = 204; // Quad4
                else type_id = 304; // Default to Tet4
            }

            // Expand lookup
            if (static_cast<size_t>(eid) >= element_lookup.size()) {
                element_lookup.resize(static_cast<size_t>(eid) * 2, entt::null);
            }

            auto e = registry.create();
            registry.emplace<Component::ElementID>(e, eid);
            registry.emplace<Component::OriginalID>(e, eid);
            registry.emplace<Component::ElementType>(e, type_id);
            
            auto& conn = registry.emplace<Component::Connectivity>(e);
            conn.nodes = std::move(valid_node_entities);

            element_lookup[eid] = e;
        }

        // --- Surface Parsing (Simdroid boundary faces/edges) ---
        // Format inside Surface/{Type} blocks:
        //   sid [n1,n2,...,parent_eid]
        // where (n1..nk) are node IDs on the face/edge, and the last entry is parent element ID.
        if (in_surface_section && in_surface_type_section) {
            // Support wrapped lists: surface definition may span multiple lines inside '[ ... ]'
            std::string surf_line = line;
            if (!pending_surface_line.empty()) {
                pending_surface_line += " " + surf_line;
                if (pending_surface_line.find(']') == std::string::npos) continue;
                surf_line = pending_surface_line;
                pending_surface_line.clear();
            } else {
                const size_t lb0 = surf_line.find('[');
                if (lb0 != std::string::npos && surf_line.find(']') == std::string::npos) {
                    pending_surface_line = surf_line;
                    continue;
                }
            }

            const size_t lb = surf_line.find('[');
            const size_t rb = surf_line.rfind(']');
            if (lb == std::string::npos || rb == std::string::npos) continue;

            int sid = 0;
            try {
                std::string id_str = surf_line.substr(0, lb);
                std::replace(id_str.begin(), id_str.end(), ',', ' ');
                sid = std::stoi(id_str);
            } catch (...) { continue; }

            if (sid < 0) continue;

            std::string content = surf_line.substr(lb + 1, rb - lb - 1);
            std::replace(content.begin(), content.end(), ',', ' ');

            std::vector<int> ids;
            ids.reserve(8);
            {
                std::stringstream ss(content);
                int v = 0;
                while (ss >> v) ids.push_back(v);
            }
            if (ids.size() < 2) continue; // must have at least 1 node + parent_eid

            const int parent_eid = ids.back();
            ids.pop_back();

            // Validate parent element
            entt::entity parent_elem_entity = entt::null;
            if (parent_eid >= 0 && static_cast<size_t>(parent_eid) < element_lookup.size()) {
                parent_elem_entity = element_lookup[parent_eid];
            }
            if (parent_elem_entity == entt::null) {
                spdlog::warn("Surface {} refers to undefined parent Element ID: {}", sid, parent_eid);
                continue;
            }

            // Validate nodes
            std::vector<entt::entity> valid_node_entities;
            valid_node_entities.reserve(ids.size());
            bool is_broken = false;
            for (int nid : ids) {
                if (nid >= 0 && static_cast<size_t>(nid) < node_lookup.size() && node_lookup[nid] != entt::null) {
                    valid_node_entities.push_back(node_lookup[nid]);
                } else {
                    spdlog::warn("Surface {} refers to undefined Node ID: {}", sid, nid);
                    is_broken = true;
                    break;
                }
            }
            if (is_broken) continue;

            // Expand lookup
            if (static_cast<size_t>(sid) >= surface_lookup.size()) {
                size_t new_size = std::max(static_cast<size_t>(sid) * 2, surface_lookup.size() + 10000);
                surface_lookup.resize(new_size, entt::null);
            }

            auto se = registry.create();
            registry.emplace<Component::SurfaceID>(se, sid);
            registry.emplace<Component::OriginalID>(se, sid); // compatibility
            auto& sc = registry.emplace<Component::SurfaceConnectivity>(se);
            sc.nodes = std::move(valid_node_entities);
            registry.emplace<Component::SurfaceParentElement>(se, parent_elem_entity);

            surface_lookup[sid] = se;
        }
    }

    // ---------------------------------------------------------------------
    // Apply Sets Definition (Post-Parsing)
    // ---------------------------------------------------------------------
    
    // Helper to add entities to sets
    auto add_to_set = [&](auto& member_list, const std::vector<IdRange>& ranges, const std::vector<entt::entity>& lookup) {
        for (const auto& r : ranges) {
            if (r.step <= 0) continue;
            for (int id = r.start; id <= r.end; id += r.step) {
                if (id >= 0 && static_cast<size_t>(id) < lookup.size()) {
                    entt::entity e = lookup[id];
                    if (e != entt::null) {
                        member_list.push_back(e);
                    }
                }
            }
        }
    };

    // 1. Node Sets
    for (const auto& [name, ranges] : defs.node_sets) {
        entt::entity e = get_or_create_set_entity(registry, name);
        auto& members = registry.get_or_emplace<Component::NodeSetMembers>(e);
        add_to_set(members.members, ranges, node_lookup);
    }

    // 2. Element Sets
    for (const auto& [name, ranges] : defs.element_sets) {
        entt::entity e = get_or_create_set_entity(registry, name);
        auto& members = registry.get_or_emplace<Component::ElementSetMembers>(e);
        add_to_set(members.members, ranges, element_lookup);
    }

    // 3. Part Sets (treat as element sets)
    for (const auto& [name, ranges] : defs.parts_ranges) {
        entt::entity e = get_or_create_set_entity(registry, name);
        auto& members = registry.get_or_emplace<Component::ElementSetMembers>(e);
        add_to_set(members.members, ranges, element_lookup);
    }

    // 4. Surface Sets
    for (const auto& [name, ranges] : defs.surface_sets) {
        entt::entity e = get_or_create_set_entity(registry, name);
        auto& members = registry.get_or_emplace<Component::SurfaceSetMembers>(e);
        add_to_set(members.members, ranges, surface_lookup);
    }
}

// =========================================================
// 实现：Radioss /RBODY 刚体解析
// =========================================================
void SimdroidParser::parse_radioss_rigid_bodies(const json& j_rb, entt::registry& registry) {
    for (auto& [name, val] : j_rb.items()) {
        if (!val.is_object()) continue;

        Component::RigidBody rb{};

        // 1. Master node: 先按集合名查找，再按 node ID 回退
        std::string master_name = val.value("MasterNodeSet", "");
        if (!master_name.empty()) {
            rb.master_node = find_set_by_name(registry, master_name);
            if (rb.master_node == entt::null) {
                try {
                    int nid = std::stoi(master_name);
                    if (nid >= 0 && static_cast<size_t>(nid) < node_lookup.size())
                        rb.master_node = node_lookup[nid];
                } catch (...) {}
            }
            if (rb.master_node == entt::null)
                spdlog::warn("RigidBody '{}': MasterNodeSet '{}' not found.", name, master_name);
        }

        // 2. Slave node set
        std::string slave_name = val.value("SlaveNodeSet", "");
        if (!slave_name.empty()) {
            rb.slave_node_set = find_set_by_name(registry, slave_name);
            if (rb.slave_node_set == entt::null)
                spdlog::warn("RigidBody '{}': SlaveNodeSet '{}' not found.", name, slave_name);
        }

        // 3. 物理属性
        rb.coord_sys = val.value("CoordSys", "");

        std::string i_cal = val.value("InertiaCal", "automatic");
        rb.inertia_cal = (to_lower_copy(i_cal) == "input")
                             ? Component::InertiaMode::Input
                             : Component::InertiaMode::Automatic;

        rb.mass = val.value("mass", 0.0);
        rb.cog_mode = val.value("CoG", 1);

        // 4. 转动惯量数组 [I11, I22, I33, I12, I23, I13]
        if (val.contains("InertiaInput") && val["InertiaInput"].is_array()) {
            const auto& arr = val["InertiaInput"];
            for (size_t i = 0; i < std::min<size_t>(6, arr.size()); ++i)
                rb.inertia_tensor[i] = arr[i].get<double>();
        }

        // 5. 创建实体并挂载
        const auto rb_entity = registry.create();
        registry.emplace<Component::RigidBody>(rb_entity, rb);
        registry.emplace<Component::SetName>(rb_entity, name);

        spdlog::info("  -> RigidBody '{}' added (Master: {}, Slave: {})", name, master_name, slave_name);
    }
}

// =========================================================
// Post-parse: validate rigid body master/slave overlap
// =========================================================
void SimdroidParser::validate_rigid_bodies(entt::registry& registry) {
    auto view = registry.view<const Component::RigidBody, const Component::SetName>();
    int warned = 0;
    int total = 0;

    for (auto entity : view) {
        ++total;
        const auto& rb = view.get<const Component::RigidBody>(entity);
        const auto& sn = view.get<const Component::SetName>(entity);

        // Collect master nodes
        std::unordered_set<entt::entity> master_nodes;
        if (rb.master_node != entt::null && registry.valid(rb.master_node)) {
            if (registry.all_of<Component::NodeSetMembers>(rb.master_node)) {
                for (auto n : registry.get<Component::NodeSetMembers>(rb.master_node).members)
                    master_nodes.insert(n);
            } else if (registry.all_of<Component::NodeID>(rb.master_node)) {
                master_nodes.insert(rb.master_node);
            }
        }

        // Collect slave nodes
        std::unordered_set<entt::entity> slave_nodes;
        if (rb.slave_node_set != entt::null && registry.valid(rb.slave_node_set)) {
            if (registry.all_of<Component::NodeSetMembers>(rb.slave_node_set)) {
                for (auto n : registry.get<Component::NodeSetMembers>(rb.slave_node_set).members)
                    slave_nodes.insert(n);
            }
        }

        // Detect overlap
        std::vector<int> overlap_ids;
        for (auto n : master_nodes) {
            if (slave_nodes.count(n)) {
                if (registry.valid(n) && registry.all_of<Component::NodeID>(n))
                    overlap_ids.push_back(registry.get<Component::NodeID>(n).value);
            }
        }
        if (!overlap_ids.empty()) {
            std::sort(overlap_ids.begin(), overlap_ids.end());
            std::string ids_str;
            for (size_t k = 0; k < std::min(overlap_ids.size(), size_t(10)); ++k) {
                if (k) ids_str += ", ";
                ids_str += std::to_string(overlap_ids[k]);
            }
            if (overlap_ids.size() > 10) ids_str += ", ...";
            spdlog::warn("RigidBody '{}': {} node(s) belong to BOTH master and slave [{}]",
                         sn.value, overlap_ids.size(), ids_str);
            ++warned;
        }
    }

    if (warned > 0)
        spdlog::warn("RigidBody validation: {} issue(s) found.", warned);
    else
        spdlog::info("RigidBody validation: all {} rigid body(ies) OK.", total);
}

// =========================================================
// Post-parse: cross-constraint validation
//   检测同一节点在不同定义中同时扮演 master 和 slave
//   (e.g. Contact/Tie master 节点同时是 RigidBody slave)
// =========================================================
void SimdroidParser::validate_cross_constraints(entt::registry& registry) {
    auto& report = get_constraint_warning_report(registry);
    report.cross_conflicts.clear();

    // role -> { node_entity, ...}
    // 收集所有 Contact 和 RigidBody 定义中的 master/slave 节点
    struct NodeRole {
        std::vector<std::string> as_master; // 作为 master 的定义名
        std::vector<std::string> as_slave;  // 作为 slave 的定义名
    };
    std::unordered_map<entt::entity, NodeRole> node_roles;

    auto collect_nodes_from_set = [&](entt::entity se, std::vector<entt::entity>& out) {
        if (se == entt::null || !registry.valid(se)) return;
        if (registry.all_of<Component::NodeSetMembers>(se)) {
            for (auto n : registry.get<Component::NodeSetMembers>(se).members)
                out.push_back(n);
        }
        if (registry.all_of<Component::SurfaceSetMembers>(se)) {
            for (auto surf : registry.get<Component::SurfaceSetMembers>(se).members) {
                if (!registry.valid(surf)) continue;
                if (registry.all_of<Component::SurfaceConnectivity>(surf))
                    for (auto n : registry.get<Component::SurfaceConnectivity>(surf).nodes)
                        out.push_back(n);
            }
        }
        // 如果 se 本身就是一个节点
        if (registry.all_of<Component::NodeID>(se))
            out.push_back(se);
    };

    // 1) Contact 定义
    {
        auto view = registry.view<const Component::ContactBase, const Component::ContactTypeTag>();
        for (auto entity : view) {
            const auto& cb = view.get<const Component::ContactBase>(entity);
            const auto& ct = view.get<const Component::ContactTypeTag>(entity);
            const std::string& def_name = cb.name;

            // GeneralContact 自接触：Surf2 留空 (slave_entity == null) 时不参与 master/slave 冲突检测
            if (ct.type == Component::ContactInterType::General && cb.slave_entity == entt::null)
                continue;

            std::vector<entt::entity> m_nodes, s_nodes;
            collect_nodes_from_set(cb.master_entity, m_nodes);
            collect_nodes_from_set(cb.slave_entity, s_nodes);

            for (auto n : m_nodes) node_roles[n].as_master.push_back("Contact:" + def_name);
            for (auto n : s_nodes) node_roles[n].as_slave.push_back("Contact:" + def_name);
        }
    }

    // 2) RigidBody 定义
    {
        auto view = registry.view<const Component::RigidBody, const Component::SetName>();
        for (auto entity : view) {
            const auto& rb = view.get<const Component::RigidBody>(entity);
            const std::string& def_name = view.get<const Component::SetName>(entity).value;

            std::vector<entt::entity> m_nodes, s_nodes;
            collect_nodes_from_set(rb.master_node, m_nodes);
            collect_nodes_from_set(rb.slave_node_set, s_nodes);

            for (auto n : m_nodes) node_roles[n].as_master.push_back("RigidBody:" + def_name);
            for (auto n : s_nodes) node_roles[n].as_slave.push_back("RigidBody:" + def_name);
        }
    }

    // 3) RigidBodyConstraint (NodalRigidBody / DistributingCoupling)
    {
        auto view = registry.view<const Component::RigidBodyConstraint, const Component::SetName>();
        for (auto entity : view) {
            const auto& rbc = view.get<const Component::RigidBodyConstraint>(entity);
            const std::string& def_name = view.get<const Component::SetName>(entity).value;

            std::vector<entt::entity> m_nodes, s_nodes;
            collect_nodes_from_set(rbc.master_node_set, m_nodes);
            collect_nodes_from_set(rbc.slave_node_set, s_nodes);

            for (auto n : m_nodes) node_roles[n].as_master.push_back("MPC:" + def_name);
            for (auto n : s_nodes) node_roles[n].as_slave.push_back("MPC:" + def_name);
        }
    }

    // 4) 检测冲突：同一节点既是某定义的 master 又是另一定义的 slave
    int conflict_count = 0;
    std::unordered_set<std::string> conflict_defs;
    for (auto& [node_e, role] : node_roles) {
        if (role.as_master.empty() || role.as_slave.empty()) continue;
        if (!registry.valid(node_e)) continue;

        ++conflict_count;
        for (const auto& name : role.as_master) conflict_defs.insert(name);
        for (const auto& name : role.as_slave)  conflict_defs.insert(name);

        int nid = -1;
        if (registry.all_of<Component::NodeID>(node_e)) {
            nid = registry.get<Component::NodeID>(node_e).value;
        }

        CrossConstraintConflictDetail detail;
        detail.node_id = nid;
        detail.as_master = role.as_master;
        detail.as_slave = role.as_slave;
        report.cross_conflicts.push_back(std::move(detail));
    }

    std::sort(report.cross_conflicts.begin(), report.cross_conflicts.end(),
              [](const CrossConstraintConflictDetail& a, const CrossConstraintConflictDetail& b) {
                  return a.node_id < b.node_id;
              });

    if (conflict_count > 0) {
        // 汇总级别的告警，避免对每个节点逐条输出
        std::vector<std::string> def_list;
        def_list.reserve(conflict_defs.size());
        for (const auto& name : conflict_defs) def_list.push_back(name);
        std::sort(def_list.begin(), def_list.end());

        std::string defs_str;
        const size_t max_show = 10;
        for (size_t i = 0; i < std::min(def_list.size(), max_show); ++i) {
            if (i) defs_str += ", ";
            defs_str += def_list[i];
        }
        if (def_list.size() > max_show) defs_str += ", ...";

        spdlog::warn(
            "Cross-constraint validation: {} node(s) with conflicting master/slave roles; affected definitions include [{}].",
            conflict_count, defs_str);
    } else {
        spdlog::info("Cross-constraint validation: no master/slave conflicts detected.");
    }
}

// =========================================================
// Post-parse: validate contact master/slave set references
// =========================================================
void SimdroidParser::validate_contacts(entt::registry& registry) {
    auto view = registry.view<const Component::ContactBase, const Component::ContactTypeTag>();
    int warned = 0;
    int total = 0;
    for (auto entity : view) {
        ++total;
        const auto& cb = view.get<const Component::ContactBase>(entity);
        const auto& ct = view.get<const Component::ContactTypeTag>(entity);

        // GeneralContact 自接触：Surf2 留空 (slave_entity == null) 时跳过 master/slave 校验
        if (ct.type == Component::ContactInterType::General && cb.slave_entity == entt::null)
            continue;

        auto check_set = [&](entt::entity se, const char* role) {
            if (se == entt::null) return;
            if (!registry.valid(se)) {
                spdlog::warn("Contact '{}': {} entity is invalid.", cb.name, role);
                ++warned;
                return;
            }
            const bool has_members =
                registry.all_of<Component::NodeSetMembers>(se) ||
                registry.all_of<Component::SurfaceSetMembers>(se) ||
                registry.all_of<Component::ElementSetMembers>(se);
            if (!has_members) {
                spdlog::warn("Contact '{}': {} set '{}' has no member component yet (may be populated later by mesh).",
                             cb.name, role,
                             registry.all_of<Component::SetName>(se)
                                 ? registry.get<Component::SetName>(se).value : "<unnamed>");
            }
        };

        check_set(cb.master_entity, "master");
        check_set(cb.slave_entity,  "slave");

        // Detect nodes that appear in both master and slave sets
        auto collect_nodes = [&](entt::entity se, std::unordered_set<entt::entity>& out) {
            if (se == entt::null || !registry.valid(se)) return;
            if (registry.all_of<Component::NodeSetMembers>(se)) {
                for (auto n : registry.get<Component::NodeSetMembers>(se).members)
                    out.insert(n);
            }
            if (registry.all_of<Component::SurfaceSetMembers>(se)) {
                for (auto surf : registry.get<Component::SurfaceSetMembers>(se).members) {
                    if (!registry.valid(surf)) continue;
                    if (registry.all_of<Component::SurfaceConnectivity>(surf)) {
                        for (auto n : registry.get<Component::SurfaceConnectivity>(surf).nodes)
                            out.insert(n);
                    }
                }
            }
        };

        std::unordered_set<entt::entity> master_nodes, slave_nodes;
        collect_nodes(cb.master_entity, master_nodes);
        collect_nodes(cb.slave_entity,  slave_nodes);

        std::vector<int> overlap_ids;
        for (auto n : master_nodes) {
            if (slave_nodes.count(n)) {
                if (registry.valid(n) && registry.all_of<Component::NodeID>(n))
                    overlap_ids.push_back(registry.get<Component::NodeID>(n).value);
            }
        }
        if (!overlap_ids.empty()) {
            std::sort(overlap_ids.begin(), overlap_ids.end());
            std::string ids_str;
            for (size_t k = 0; k < std::min(overlap_ids.size(), size_t(10)); ++k) {
                if (k) ids_str += ", ";
                ids_str += std::to_string(overlap_ids[k]);
            }
            if (overlap_ids.size() > 10) ids_str += ", ...";
            spdlog::warn("Contact '{}': {} node(s) belong to BOTH master and slave sets [{}]",
                         cb.name, overlap_ids.size(), ids_str);
            ++warned;
        }
    }
    if (warned > 0)
        spdlog::warn("Contact validation: {} issue(s) found.", warned);
    else
        spdlog::info("Contact validation: all {} contact(s) OK.", total);
}

