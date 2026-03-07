/**
 * TUI Universal Inspection Panel - Registry implementation and component printers.
 */
#include "tui/ComponentTUI.h"
#include "simdroid/SimdroidInspector.h"
#include "components/mesh_components.h"
#include "components/load_components.h"
#include "components/material_components.h"
#include "components/property_components.h"
#include "components/simdroid_components.h"
#include "PartGraph.h"
#include "analysis/GraphBuilder.h"
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace tui {

namespace {

entt::entity find_set_by_name(entt::registry& reg, const std::string& name) {
    auto view = reg.view<const Component::SetName>();
    for (auto e : view) {
        if (view.get<const Component::SetName>(e).value == name)
            return e;
    }
    return entt::null;
}

entt::entity find_part_entity_by_name(entt::registry& reg, const std::string& name) {
    auto view = reg.view<const Component::SimdroidPart>();
    for (auto e : view) {
        if (view.get<const Component::SimdroidPart>(e).name == name)
            return e;
    }
    return entt::null;
}

void print_matrix_6x6(std::ostream& out, const Eigen::Matrix<double, 6, 6>& D) {
    const int w = 12;
    for (int i = 0; i < 6; ++i) {
        out << "    ";
        for (int j = 0; j < 6; ++j)
            out << std::setw(w) << std::fixed << std::setprecision(4) << D(i, j);
        out << "\n";
    }
}

void init_registry() {
    auto& r = ComponentTUIRegistry::instance();

    r.register_component<Component::Position>("Position",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            const auto& p = reg.get<Component::Position>(e);
            out << "  " << std::fixed << std::setprecision(6)
                << "(" << p.x << ", " << p.y << ", " << p.z << ")\n";
        });

    r.register_component<Component::NodeID>("NodeID",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            out << "  " << reg.get<Component::NodeID>(e).value << "\n";
        });

    r.register_component<Component::ElementID>("ElementID",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            out << "  " << reg.get<Component::ElementID>(e).value << "\n";
        });

    r.register_component<Component::Connectivity>("Connectivity",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            const auto& c = reg.get<Component::Connectivity>(e);
            out << "  Nodes (" << c.nodes.size() << "): ";
            for (size_t i = 0; i < c.nodes.size(); ++i) {
                if (i > 0) out << ", ";
                entt::entity ne = c.nodes[i];
                if (reg.valid(ne) && reg.all_of<Component::NodeID>(ne))
                    out << reg.get<Component::NodeID>(ne).value;
                else
                    out << "?";
            }
            out << "\n";
        });

    r.register_component<Component::ElementType>("ElementType",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            out << "  type_id = " << reg.get<Component::ElementType>(e).type_id << "\n";
        });

    r.register_component<Component::SimdroidPart>("SimdroidPart",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            const auto& p = reg.get<Component::SimdroidPart>(e);
            out << "  name = " << p.name << "\n";
            if (reg.valid(p.material) && reg.all_of<Component::SetName>(p.material))
                out << "  material = " << reg.get<Component::SetName>(p.material).value << "\n";
            else
                out << "  material = (entity)\n";
            if (reg.valid(p.section))
                out << "  section = (entity)\n";
        });

    r.register_component<Component::MaterialModel>("MaterialModel",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            out << "  " << reg.get<Component::MaterialModel>(e).value << "\n";
        });

    r.register_component<Component::LinearElasticParams>("LinearElasticParams",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            const auto& le = reg.get<Component::LinearElasticParams>(e);
            out << "  rho = " << le.rho << ", E = " << le.E << ", nu = " << le.nu << "\n";
        });

    r.register_component<Component::LinearElasticMatrix>("LinearElasticMatrix",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            const auto& lem = reg.get<Component::LinearElasticMatrix>(e);
            out << "  is_initialized = " << (lem.is_initialized ? "true" : "false") << "\n";
            if (lem.is_initialized) {
                out << "  D (6x6):\n";
                print_matrix_6x6(out, lem.D);
            }
        });

    r.register_component<Component::AppliedLoadRef>("AppliedLoadRef",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            const auto& ref = reg.get<Component::AppliedLoadRef>(e);
            out << "  loads: " << ref.load_entities.size() << " ref(s)\n";
            for (size_t i = 0; i < ref.load_entities.size(); ++i) {
                entt::entity le = ref.load_entities[i];
                if (!reg.valid(le)) { out << "    [" << i << "] (invalid)\n"; continue; }
                if (reg.all_of<Component::NodalLoad>(le)) {
                    const auto& nl = reg.get<Component::NodalLoad>(le);
                    out << "    [" << i << "] NodalLoad dof=" << nl.dof << " value=" << nl.value << "\n";
                } else if (reg.all_of<Component::BaseAccelerationLoad>(le)) {
                    const auto& ba = reg.get<Component::BaseAccelerationLoad>(le);
                    out << "    [" << i << "] BaseAcceleration ax=" << ba.ax << " ay=" << ba.ay << " az=" << ba.az << "\n";
                } else
                    out << "    [" << i << "] (other)\n";
            }
        });

    r.register_component<Component::AppliedBoundaryRef>("AppliedBoundaryRef",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            const auto& ref = reg.get<Component::AppliedBoundaryRef>(e);
            out << "  boundaries: " << ref.boundary_entities.size() << " ref(s)\n";
            for (size_t i = 0; i < ref.boundary_entities.size(); ++i) {
                entt::entity be = ref.boundary_entities[i];
                if (!reg.valid(be)) { out << "    [" << i << "] (invalid)\n"; continue; }
                if (reg.all_of<Component::BoundarySPC>(be)) {
                    const auto& spc = reg.get<Component::BoundarySPC>(be);
                    out << "    [" << i << "] SPC dof=" << spc.dof << " value=" << spc.value << "\n";
                } else
                    out << "    [" << i << "] (other)\n";
            }
        });

    r.register_component<Component::ForcePathNode>("ForcePathNode",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            const auto& fp = reg.get<Component::ForcePathNode>(e);
            out << "  weight = " << fp.weight
                << ", is_load_point = " << (fp.is_load_point ? "true" : "false")
                << ", is_constraint_point = " << (fp.is_constraint_point ? "true" : "false") << "\n";
        });

    r.register_component<Component::SetName>("SetName",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*, std::ostream& out) {
            out << "  " << reg.get<Component::SetName>(e).value << "\n";
        });

    r.register_component<Component::ElementSetMembers>("ElementSetMembers",
        [](entt::registry& reg, entt::entity e, SimdroidInspector* insp, std::ostream& out) {
            const auto& m = reg.get<Component::ElementSetMembers>(e);
            out << "  count = " << m.members.size() << "\n  ";
            const size_t show = (std::min)(m.members.size(), size_t(20));
            for (size_t i = 0; i < show; ++i) {
                if (i > 0) out << ", ";
                entt::entity me = m.members[i];
                if (reg.valid(me) && reg.all_of<Component::ElementID>(me))
                    out << reg.get<Component::ElementID>(me).value;
                else
                    out << "?";
            }
            if (m.members.size() > show) out << " ...";
            out << "\n";
        });

    r.register_component<Component::NodeSetMembers>("NodeSetMembers",
        [](entt::registry& reg, entt::entity e, SimdroidInspector* insp, std::ostream& out) {
            const auto& m = reg.get<Component::NodeSetMembers>(e);
            out << "  count = " << m.members.size() << "\n  ";
            const size_t show = (std::min)(m.members.size(), size_t(20));
            for (size_t i = 0; i < show; ++i) {
                if (i > 0) out << ", ";
                entt::entity me = m.members[i];
                if (reg.valid(me) && reg.all_of<Component::NodeID>(me))
                    out << reg.get<Component::NodeID>(me).value;
                else
                    out << "?";
            }
            if (m.members.size() > show) out << " ...";
            out << "\n";
        });
}

} // anonymous namespace

void ComponentTUIRegistry::for_each_applicable(entt::registry& reg, entt::entity e, SimdroidInspector* insp, std::ostream& out) const {
    for (const auto& entry : entries_) {
        if (entry.has_component(reg, e)) {
            out << color::dim << entry.display_name << ":" << color::reset << "\n";
            entry.print(reg, e, insp, out);
        }
    }
}

ComponentTUIRegistry& ComponentTUIRegistry::instance() {
    static ComponentTUIRegistry inst;
    static bool once = false;
    if (!once) {
        once = true;
        init_registry();
    }
    return inst;
}

entt::entity resolve_panel_entity(entt::registry& reg, SimdroidInspector* insp,
    const std::string& type, const std::string& id_or_name, PanelEntityKind* out_kind, std::string* out_display_id)
{
    if (out_kind) *out_kind = PanelEntityKind::Unknown;
    if (out_display_id) *out_display_id = id_or_name;

    if (type == "node") {
        int nid = 0;
        try { nid = std::stoi(id_or_name); } catch (...) { return entt::null; }
        if (!insp || !insp->is_built) return entt::null;
        auto it = insp->nid_to_entity.find(nid);
        if (it == insp->nid_to_entity.end()) return entt::null;
        if (out_kind) *out_kind = PanelEntityKind::Node;
        if (out_display_id) *out_display_id = std::to_string(nid);
        return it->second;
    }

    if (type == "elem" || type == "element") {
        int eid = 0;
        try { eid = std::stoi(id_or_name); } catch (...) { return entt::null; }
        if (!insp || !insp->is_built) return entt::null;
        auto it = insp->eid_to_entity.find(eid);
        if (it == insp->eid_to_entity.end()) return entt::null;
        if (out_kind) *out_kind = PanelEntityKind::Element;
        if (out_display_id) *out_display_id = std::to_string(eid);
        return it->second;
    }

    if (type == "part") {
        entt::entity pe = find_part_entity_by_name(reg, id_or_name);
        if (pe == entt::null) return entt::null;
        if (out_kind) *out_kind = PanelEntityKind::Part;
        if (out_display_id) *out_display_id = id_or_name;
        return pe;
    }

    if (type == "set") {
        entt::entity se = find_set_by_name(reg, id_or_name);
        if (se == entt::null) return entt::null;
        if (out_kind) *out_kind = PanelEntityKind::Set;
        if (out_display_id) *out_display_id = id_or_name;
        return se;
    }

    return entt::null;
}

void render_panel(entt::registry& reg, entt::entity e, SimdroidInspector* insp,
    PanelEntityKind kind, const std::string& display_id, std::ostream& out)
{
    const char* kind_str = "Entity";
    switch (kind) {
        case PanelEntityKind::Node:   kind_str = "Node";   break;
        case PanelEntityKind::Element: kind_str = "Element"; break;
        case PanelEntityKind::Part:   kind_str = "Part";   break;
        case PanelEntityKind::Set:   kind_str = "Set";   break;
        default: break;
    }
    out << color::title << "\n=== TUI Panel [" << kind_str << " " << display_id << "] ===\n"
        << "  entity = " << entt::to_integral(e) << color::reset << "\n\n";

    ComponentTUIRegistry::instance().for_each_applicable(reg, e, insp, out);

    if (kind == PanelEntityKind::Node) {
        append_force_path_info(reg, e, insp, out);
    } else if (kind == PanelEntityKind::Element) {
        if (insp && insp->is_built && reg.all_of<Component::ElementID>(e)) {
            int eid = reg.get<Component::ElementID>(e).value;
            if (reg.all_of<Component::Connectivity>(e)) {
                const auto& c = reg.get<Component::Connectivity>(e);
                out << color::dim << "Contains Nodes: ";
                for (size_t i = 0; i < c.nodes.size(); ++i) {
                    if (i > 0) out << ", ";
                    if (reg.valid(c.nodes[i]) && reg.all_of<Component::NodeID>(c.nodes[i]))
                        out << reg.get<Component::NodeID>(c.nodes[i]).value;
                }
                out << " (Use panel node <id> to inspect)" << color::reset << "\n";
            }
        }
    }
}

void append_force_path_info(entt::registry& reg, entt::entity node_entity, SimdroidInspector* insp, std::ostream& out) {
    if (!insp || !insp->is_built) return;
    if (!reg.all_of<Component::NodeID>(node_entity)) return;
    int nid = reg.get<Component::NodeID>(node_entity).value;

    bool is_load = false, is_constraint = false;
    if (reg.all_of<Component::ForcePathNode>(node_entity)) {
        const auto& fp = reg.get<Component::ForcePathNode>(node_entity);
        is_load = fp.is_load_point;
        is_constraint = fp.is_constraint_point;
    } else {
        if (reg.all_of<Component::AppliedLoadRef>(node_entity)) {
            const auto& ref = reg.get<Component::AppliedLoadRef>(node_entity);
            if (!ref.load_entities.empty()) is_load = true;
        }
        if (reg.all_of<Component::AppliedBoundaryRef>(node_entity)) {
            const auto& ref = reg.get<Component::AppliedBoundaryRef>(node_entity);
            if (!ref.boundary_entities.empty()) is_constraint = true;
        }
    }

    if (is_load || is_constraint) {
        out << color::dim << "Force path: ";
        if (is_load) out << color::green << "load point" << color::reset << color::dim;
        if (is_load && is_constraint) out << ", ";
        if (is_constraint) out << color::yellow << "constraint point" << color::reset << color::dim;
        out << color::reset << "\n";
    }

    auto it_elems = insp->nid_to_elems.find(nid);
    if (it_elems == insp->nid_to_elems.end()) return;
    std::vector<std::string> parts;
    for (int eid : it_elems->second) {
        auto itp = insp->eid_to_part.find(eid);
        if (itp != insp->eid_to_part.end()) {
            if (std::find(parts.begin(), parts.end(), itp->second) == parts.end())
                parts.push_back(itp->second);
        }
    }
    if (parts.empty()) return;

    PartGraph graph = GraphBuilder::build(reg, *insp);
    out << color::dim << "Parts: ";
    for (const auto& p : parts) out << "[" << p << "] ";
    out << "\n";
    for (const auto& part_name : parts) {
        auto itn = graph.nodes.find(part_name);
        if (itn == graph.nodes.end()) continue;
        for (const auto& edge : itn->second.edges) {
            const char* ct = "?";
            switch (edge.type) {
                case ConnectionType::Contact:    ct = "Contact"; break;
                case ConnectionType::SharedNode: ct = "SharedNode"; break;
                case ConnectionType::MPC:        ct = "MPC"; break;
            }
            out << "  " << part_name << " --" << ct;
            if (!edge.sub_type.empty()) out << " (" << edge.sub_type << ")";
            out << "--> " << edge.target_part << "\n";
        }
    }
    out << color::reset;
}

} // namespace tui
