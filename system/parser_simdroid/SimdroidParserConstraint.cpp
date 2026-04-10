// ============================================================
// SimdroidParserConstraint.cpp
//
// Constraint/Load parsing helpers:
//   find_set_by_name, parse_boundary_conditions, parse_rigid_bodies,
//   parse_loads, parse_initial_conditions, parse_rigid_walls,
//   parse_analysis_settings, parse_radioss_rigid_bodies
// ============================================================

#include "SimdroidParser.h"
#include "SimdroidParserDetail.h"

#include "../../data_center/DofMap.h"

#include "nlohmann/json.hpp"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace SimdroidParserDetail;
using json = nlohmann::json;

// ---------------------------------------------------------------
// Helper: find a set entity by name
// ---------------------------------------------------------------

entt::entity SimdroidParser::find_set_by_name(entt::registry& registry, const std::string& name) {
    auto view = registry.view<const Component::SetName>();
    for (auto entity : view) {
        if (view.get<const Component::SetName>(entity).value == name) {
            return entity;
        }
    }
    return entt::null;
}

// ---------------------------------------------------------------
// Parse: Boundary Conditions (SPC)
// ---------------------------------------------------------------

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

// ---------------------------------------------------------------
// Parse: Rigid Bodies (MPC)
// ---------------------------------------------------------------

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

// ---------------------------------------------------------------
// Parse: Loads (Force/Moment/BaseAcceleration/Pressure)
// ---------------------------------------------------------------

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

            // TimeCurve
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
        // 2. Handle Base Acceleration (Gravity / Ground Acceleration)
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

// ---------------------------------------------------------------
// Parse: Initial Conditions (Initial Velocity)
// ---------------------------------------------------------------

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

// ---------------------------------------------------------------
// Parse: Rigid Wall
// ---------------------------------------------------------------

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

// ---------------------------------------------------------------
// Parse: Analysis Settings (Step)
// ---------------------------------------------------------------

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

// ---------------------------------------------------------------
// Parse: Radioss /RBODY rigid bodies
// ---------------------------------------------------------------

void SimdroidParser::parse_radioss_rigid_bodies(const json& j_rb, entt::registry& registry) {
    for (auto& [name, val] : j_rb.items()) {
        if (!val.is_object()) continue;

        Component::RigidBody rb{};

        // 1. Master node: first try set name lookup, then fall back to node ID
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

        // 3. Physical properties
        rb.coord_sys = val.value("CoordSys", "");

        std::string i_cal = val.value("InertiaCal", "automatic");
        rb.inertia_cal = (to_lower_copy(i_cal) == "input")
                             ? Component::InertiaMode::Input
                             : Component::InertiaMode::Automatic;

        rb.mass = val.value("mass", 0.0);
        rb.cog_mode = val.value("CoG", 1);

        // 4. Inertia tensor [I11, I22, I33, I12, I23, I13]
        if (val.contains("InertiaInput") && val["InertiaInput"].is_array()) {
            const auto& arr = val["InertiaInput"];
            for (size_t i = 0; i < std::min<size_t>(6, arr.size()); ++i)
                rb.inertia_tensor[i] = arr[i].get<double>();
        }

        // 5. Create entity and attach
        const auto rb_entity = registry.create();
        registry.emplace<Component::RigidBody>(rb_entity, rb);
        registry.emplace<Component::SetName>(rb_entity, name);

        spdlog::info("  -> RigidBody '{}' added (Master: {}, Slave: {})", name, master_name, slave_name);
    }
}
