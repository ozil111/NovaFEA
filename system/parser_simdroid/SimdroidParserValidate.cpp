// ============================================================
// SimdroidParserValidate.cpp
//
// Post-parse validation logic:
//   validate_constraints, list_constraint_warnings,
//   validate_rigid_bodies, validate_cross_constraints,
//   validate_contacts
// ============================================================

#include "SimdroidParser.h"
#include "SimdroidParserDetail.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace SimdroidParserDetail;

// ============================================================
// Internal validation report structures (anonymous namespace)
// ============================================================

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

// ---------------------------------------------------------------
// Re-run all validations
// ---------------------------------------------------------------

void SimdroidParser::validate_constraints(entt::registry& registry) {
    validate_contacts(registry);
    validate_rigid_bodies(registry);
    validate_cross_constraints(registry);
}

// ---------------------------------------------------------------
// Print cached cross-constraint conflict details
// ---------------------------------------------------------------

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

// ---------------------------------------------------------------
// Post-validate: rigid body master/slave overlap
// ---------------------------------------------------------------

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

// ---------------------------------------------------------------
// Post-validate: cross-constraint conflicts
// ---------------------------------------------------------------

void SimdroidParser::validate_cross_constraints(entt::registry& registry) {
    auto& report = get_constraint_warning_report(registry);
    report.cross_conflicts.clear();

    struct NodeRole {
        std::vector<std::string> as_master;
        std::vector<std::string> as_slave;
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
        if (registry.all_of<Component::NodeID>(se))
            out.push_back(se);
    };

    // 1) Contact definitions
    {
        auto view = registry.view<const Component::ContactBase, const Component::ContactTypeTag>();
        for (auto entity : view) {
            const auto& cb = view.get<const Component::ContactBase>(entity);
            const auto& ct = view.get<const Component::ContactTypeTag>(entity);
            const std::string& def_name = cb.name;

            // GeneralContact self-contact: Surf2 empty (slave_entity == null) -> skip
            if (ct.type == Component::ContactInterType::General && cb.slave_entity == entt::null)
                continue;

            std::vector<entt::entity> m_nodes, s_nodes;
            collect_nodes_from_set(cb.master_entity, m_nodes);
            collect_nodes_from_set(cb.slave_entity, s_nodes);

            for (auto n : m_nodes) node_roles[n].as_master.push_back("Contact:" + def_name);
            for (auto n : s_nodes) node_roles[n].as_slave.push_back("Contact:" + def_name);
        }
    }

    // 2) RigidBody definitions
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

    // 4) Detect conflicts
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

// ---------------------------------------------------------------
// Post-validate: contact master/slave set references
// ---------------------------------------------------------------

void SimdroidParser::validate_contacts(entt::registry& registry) {
    auto view = registry.view<const Component::ContactBase, const Component::ContactTypeTag>();
    int warned = 0;
    int total = 0;
    for (auto entity : view) {
        ++total;
        const auto& cb = view.get<Component::ContactBase>(entity);
        const auto& ct = view.get<Component::ContactTypeTag>(entity);

        // GeneralContact self-contact: Surf2 empty (slave_entity == null) -> skip check
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
