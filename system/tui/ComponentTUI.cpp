/**
 * TUI Universal Inspection Panel - Registry implementation and FTXUI component renderers.
 */
#include "tui/ComponentTUI.h"
#include "simdroid/SimdroidInspector.h"
#include "components/mesh_components.h"
#include "components/load_components.h"
#include "components/material_components.h"
#include "components/simdroid_components.h"
#include "PartGraph.h"
#include "analysis/GraphBuilder.h"
#include "AppSession.h"
#include "CommandProcessor.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/base_sink.h"
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>
#include <optional>
#include <algorithm>
#include <deque>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>

namespace tui {

namespace {

entt::entity find_set_by_name(entt::registry& reg, const std::string& name) {
    auto view = reg.view<const ::Component::SetName>();
    for (auto e : view) {
        if (view.get<const ::Component::SetName>(e).value == name)
            return e;
    }
    return entt::null;
}

entt::entity find_part_entity_by_name(entt::registry& reg, const std::string& name) {
    auto view = reg.view<const ::Component::SimdroidPart>();
    for (auto e : view) {
        if (view.get<const ::Component::SimdroidPart>(e).name == name)
            return e;
    }
    return entt::null;
}

Element matrix_6x6_element(const Eigen::Matrix<double, 6, 6>& D) {
    Elements rows;
    for (int i = 0; i < 6; ++i) {
        std::ostringstream line;
        for (int j = 0; j < 6; ++j)
            line << std::setw(12) << std::fixed << std::setprecision(4) << D(i, j);
        rows.push_back(text("    " + line.str()));
    }
    return vbox(std::move(rows));
}

void init_registry() {
    auto& r = ComponentTUIRegistry::instance();

    r.register_component<::Component::Position>("Position",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            const auto& p = reg.get<::Component::Position>(e);
            std::ostringstream sx, sy, sz;
            sx << std::fixed << std::setprecision(6) << p.x;
            sy << std::fixed << std::setprecision(6) << p.y;
            sz << std::fixed << std::setprecision(6) << p.z;
            return hbox({
                text(" X: ") | color(Color::Red),   text(sx.str()),
                text(" Y: ") | color(Color::Green), text(sy.str()),
                text(" Z: ") | color(Color::Blue),  text(sz.str())
            });
        });

    r.register_component<::Component::NodeID>("NodeID",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            return text("  " + std::to_string(reg.get<::Component::NodeID>(e).value));
        });

    r.register_component<::Component::ElementID>("ElementID",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            return text("  " + std::to_string(reg.get<::Component::ElementID>(e).value));
        });

    r.register_component<::Component::Connectivity>("Connectivity",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            const auto& c = reg.get<::Component::Connectivity>(e);
            Elements node_list;
            for (entt::entity ne : c.nodes) {
                if (reg.valid(ne) && reg.all_of<::Component::NodeID>(ne))
                    node_list.push_back(text(std::to_string(reg.get<::Component::NodeID>(ne).value)) | border);
                else
                    node_list.push_back(text("?") | border);
            }
            return hbox(std::move(node_list)) | flex;
        });

    r.register_component<::Component::ElementType>("ElementType",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            return text("  type_id = " + std::to_string(reg.get<::Component::ElementType>(e).type_id));
        });

    r.register_component<::Component::SimdroidPart>("SimdroidPart",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            const auto& p = reg.get<::Component::SimdroidPart>(e);
            Elements lines = { text("  name = " + p.name) };
            if (reg.valid(p.material) && reg.all_of<::Component::SetName>(p.material))
                lines.push_back(text("  material = " + reg.get<::Component::SetName>(p.material).value));
            else
                lines.push_back(text("  material = (entity)"));
            if (reg.valid(p.section))
                lines.push_back(text("  section = (entity)"));
            return vbox(std::move(lines));
        });

    r.register_component<::Component::MaterialModel>("MaterialModel",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            return text("  " + reg.get<::Component::MaterialModel>(e).value);
        });

    r.register_component<::Component::LinearElasticParams>("LinearElasticParams",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            const auto& le = reg.get<::Component::LinearElasticParams>(e);
            std::ostringstream ss;
            ss << "  rho = " << le.rho << ", E = " << le.E << ", nu = " << le.nu;
            return text(ss.str());
        });

    r.register_component<::Component::LinearElasticMatrix>("LinearElasticMatrix",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            const auto& lem = reg.get<::Component::LinearElasticMatrix>(e);
            Elements parts = { text("  is_initialized = " + std::string(lem.is_initialized ? "true" : "false")) };
            if (lem.is_initialized)
                parts.push_back(vbox({ text("  D (6x6):"), matrix_6x6_element(lem.D) }));
            return vbox(std::move(parts));
        });

    r.register_component<::Component::AppliedLoadRef>("AppliedLoadRef",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            const auto& ref = reg.get<::Component::AppliedLoadRef>(e);
            Elements lines = { text("  loads: " + std::to_string(ref.load_entities.size()) + " ref(s)") };
            for (size_t i = 0; i < ref.load_entities.size(); ++i) {
                entt::entity le = ref.load_entities[i];
                if (!reg.valid(le)) { lines.push_back(text("    [" + std::to_string(i) + "] (invalid)")); continue; }
                if (reg.all_of<::Component::NodalLoad>(le)) {
                    const auto& nl = reg.get<::Component::NodalLoad>(le);
                    lines.push_back(text("    [" + std::to_string(i) + "] NodalLoad dof=" + nl.dof + " value=" + std::to_string(nl.value)));
                } else if (reg.all_of<::Component::BaseAccelerationLoad>(le)) {
                    const auto& ba = reg.get<::Component::BaseAccelerationLoad>(le);
                    std::ostringstream ss;
                    ss << "    [" << i << "] BaseAcceleration ax=" << ba.ax << " ay=" << ba.ay << " az=" << ba.az;
                    lines.push_back(text(ss.str()));
                } else
                    lines.push_back(text("    [" + std::to_string(i) + "] (other)"));
            }
            return vbox(std::move(lines));
        });

    r.register_component<::Component::AppliedBoundaryRef>("AppliedBoundaryRef",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            const auto& ref = reg.get<::Component::AppliedBoundaryRef>(e);
            Elements lines = { text("  boundaries: " + std::to_string(ref.boundary_entities.size()) + " ref(s)") };
            for (size_t i = 0; i < ref.boundary_entities.size(); ++i) {
                entt::entity be = ref.boundary_entities[i];
                if (!reg.valid(be)) { lines.push_back(text("    [" + std::to_string(i) + "] (invalid)")); continue; }
                if (reg.all_of<::Component::BoundarySPC>(be)) {
                    const auto& spc = reg.get<::Component::BoundarySPC>(be);
                    lines.push_back(text("    [" + std::to_string(i) + "] SPC dof=" + spc.dof + " value=" + std::to_string(spc.value)));
                } else
                    lines.push_back(text("    [" + std::to_string(i) + "] (other)"));
            }
            return vbox(std::move(lines));
        });

    r.register_component<::Component::ForcePathNode>("ForcePathNode",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            const auto& fp = reg.get<::Component::ForcePathNode>(e);
            std::ostringstream ss;
            ss << "  weight = " << fp.weight
               << ", is_load_point = " << (fp.is_load_point ? "true" : "false")
               << ", is_constraint_point = " << (fp.is_constraint_point ? "true" : "false");
            return text(ss.str());
        });

    r.register_component<::Component::SetName>("SetName",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            return text("  " + reg.get<::Component::SetName>(e).value);
        });

    r.register_component<::Component::ElementSetMembers>("ElementSetMembers",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            const auto& m = reg.get<::Component::ElementSetMembers>(e);
            std::string line = "  count = " + std::to_string(m.members.size()) + "\n  ";
            const size_t show = (std::min)(m.members.size(), size_t(20));
            for (size_t i = 0; i < show; ++i) {
                if (i > 0) line += ", ";
                entt::entity me = m.members[i];
                if (reg.valid(me) && reg.all_of<::Component::ElementID>(me))
                    line += std::to_string(reg.get<::Component::ElementID>(me).value);
                else
                    line += "?";
            }
            if (m.members.size() > show) line += " ...";
            return text(line);
        });

    r.register_component<::Component::NodeSetMembers>("NodeSetMembers",
        [](entt::registry& reg, entt::entity e, SimdroidInspector*) -> Element {
            const auto& m = reg.get<::Component::NodeSetMembers>(e);
            std::string line = "  count = " + std::to_string(m.members.size()) + "\n  ";
            const size_t show = (std::min)(m.members.size(), size_t(20));
            for (size_t i = 0; i < show; ++i) {
                if (i > 0) line += ", ";
                entt::entity me = m.members[i];
                if (reg.valid(me) && reg.all_of<::Component::NodeID>(me))
                    line += std::to_string(reg.get<::Component::NodeID>(me).value);
                else
                    line += "?";
            }
            if (m.members.size() > show) line += " ...";
            return text(line);
        });
}

} // anonymous namespace

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
    PanelEntityKind kind, const std::string& display_id)
{
    const char* kind_str = "Entity";
    switch (kind) {
        case PanelEntityKind::Node:    kind_str = "Node";    break;
        case PanelEntityKind::Element: kind_str = "Element"; break;
        case PanelEntityKind::Part:    kind_str = "Part";    break;
        case PanelEntityKind::Set:     kind_str = "Set";     break;
        default: break;
    }

    Elements component_views;
    for (const auto& entry : ComponentTUIRegistry::instance().entries()) {
        if (entry.has_component(reg, e)) {
            component_views.push_back(
                window(text(entry.display_name) | bold, entry.render(reg, e, insp)));
        }
    }

    if (kind == PanelEntityKind::Node)
        component_views.push_back(window(text("Force path") | bold, force_path_element(reg, e, insp)));
    else if (kind == PanelEntityKind::Element && insp && insp->is_built && reg.all_of<::Component::ElementID>(e) && reg.all_of<::Component::Connectivity>(e)) {
        const auto& c = reg.get<::Component::Connectivity>(e);
        Elements node_texts;
        for (size_t i = 0; i < c.nodes.size(); ++i) {
            if (i > 0) node_texts.push_back(text(", "));
            if (reg.valid(c.nodes[i]) && reg.all_of<::Component::NodeID>(c.nodes[i]))
                node_texts.push_back(text(std::to_string(reg.get<::Component::NodeID>(c.nodes[i]).value)));
        }
        node_texts.push_back(text(" (Use panel node <id> to inspect)") | dim);
        component_views.push_back(window(text("Contains Nodes") | bold, hbox(std::move(node_texts))));

        // Show part of this element if available in inspector
        int eid = reg.get<::Component::ElementID>(e).value;
        auto itp = insp->eid_to_part.find(eid);
        if (itp != insp->eid_to_part.end()) {
            std::string part_line = "Part: [" + itp->second + "]";
            component_views.push_back(window(text("Part") | bold, text(part_line) | dim));
        }
    }

    Element document = vbox({
        hbox({
            text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
            text(" Universal Inspector ") | color(Color::Cyan),
            filler(),
            text("Entity ID: " + std::to_string(entt::to_integral(e))) | dim
        }) | border,
        vbox({
            text("") | dim,
            text(std::string("TUI Panel [") + kind_str + " " + display_id + "]") | bold,
            text("") | dim
        }),
        vbox(std::move(component_views)) | flex,
        hbox(text(" [P] Panel  [G] Graph  [Q] Quit ") | dim)
    });

    auto screen = Screen::Create(Dimension::Full(), Dimension::Fit(document));
    Render(screen, document);
    std::cout << screen.ToString() << std::endl;
}

Element force_path_element(entt::registry& reg, entt::entity node_entity, SimdroidInspector* insp) {
    Element none = text("(none)") | dim;
    if (!insp || !insp->is_built) return none;
    if (!reg.all_of<::Component::NodeID>(node_entity)) return none;
    int nid = reg.get<::Component::NodeID>(node_entity).value;

    bool is_load = false, is_constraint = false;
    if (reg.all_of<::Component::ForcePathNode>(node_entity)) {
        const auto& fp = reg.get<::Component::ForcePathNode>(node_entity);
        is_load = fp.is_load_point;
        is_constraint = fp.is_constraint_point;
    } else {
        if (reg.all_of<::Component::AppliedLoadRef>(node_entity)) {
            const auto& ref = reg.get<::Component::AppliedLoadRef>(node_entity);
            if (!ref.load_entities.empty()) is_load = true;
        }
        if (reg.all_of<::Component::AppliedBoundaryRef>(node_entity)) {
            const auto& ref = reg.get<::Component::AppliedBoundaryRef>(node_entity);
            if (!ref.boundary_entities.empty()) is_constraint = true;
        }
    }

    Elements parts_el;
    if (is_load || is_constraint) {
        Elements labels;
        if (is_load) labels.push_back(text("load point") | color(Color::Green));
        if (is_load && is_constraint) labels.push_back(text(", "));
        if (is_constraint) labels.push_back(text("constraint point") | color(Color::Yellow));
        parts_el.push_back(hbox({ text("Force path: ") | dim, hbox(std::move(labels)) }));
    }

    auto it_elems = insp->nid_to_elems.find(nid);
    if (it_elems == insp->nid_to_elems.end()) return parts_el.empty() ? none : vbox(std::move(parts_el));

    const auto& elem_ids = it_elems->second;
    if (!elem_ids.empty()) {
        std::string elems_line = "Elements: ";
        const size_t show_e = (std::min)(elem_ids.size(), size_t(20));
        for (size_t i = 0; i < show_e; ++i) {
            if (i > 0) elems_line += ", ";
            elems_line += std::to_string(elem_ids[i]);
        }
        if (elem_ids.size() > show_e) elems_line += " ...";
        parts_el.push_back(text(elems_line) | dim);
    }

    std::vector<std::string> parts;
    for (int eid : elem_ids) {
        auto itp = insp->eid_to_part.find(eid);
        if (itp != insp->eid_to_part.end()) {
            if (std::find(parts.begin(), parts.end(), itp->second) == parts.end())
                parts.push_back(itp->second);
        }
    }
    if (parts.empty()) return parts_el.empty() ? none : vbox(std::move(parts_el));

    PartGraph graph = GraphBuilder::build(reg, *insp);
    std::string parts_line = "Parts: ";
    for (const auto& p : parts) parts_line += "[" + p + "] ";
    parts_el.push_back(text(parts_line) | dim);
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
            std::string line = "  " + part_name + " --" + ct;
            if (!edge.sub_type.empty()) line += " (" + edge.sub_type + ")";
            line += "--> " + edge.target_part;
            parts_el.push_back(text(line) | dim);
        }
    }
    return vbox(std::move(parts_el));
}

void render_nodes_list(entt::registry& reg) {
    struct Row {
        int nid;
        double x, y, z;
    };
    std::vector<Row> rows;
    auto view = reg.view<const ::Component::NodeID, const ::Component::Position>();
    rows.reserve(view.size_hint());
    for (auto e : view) {
        const auto& id = view.get<const ::Component::NodeID>(e);
        const auto& p = view.get<const ::Component::Position>(e);
        rows.push_back(Row{ id.value, p.x, p.y, p.z });
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.nid < b.nid; });

    int selected_row = rows.empty() ? -1 : 0;

    auto screen = ScreenInteractive::Fullscreen();

    ftxui::Component ui = ftxui::Renderer([&] {
        Elements lines;
        lines.push_back(
            hbox({
                text(" NodeID ") | bold,
                text(" | "),
                text(" X ") | bold,
                text(" | "),
                text(" Y ") | bold,
                text(" | "),
                text(" Z ") | bold,
            }) | border);

        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];
            std::ostringstream sx, sy, sz;
            sx.setf(std::ios::fixed); sy.setf(std::ios::fixed); sz.setf(std::ios::fixed);
            sx << std::setprecision(6) << r.x;
            sy << std::setprecision(6) << r.y;
            sz << std::setprecision(6) << r.z;

            Element row = hbox({
                text(" " + std::to_string(r.nid) + " ") | color(Color::Cyan),
                text(" | "),
                text(" " + sx.str() + " "),
                text(" | "),
                text(" " + sy.str() + " "),
                text(" | "),
                text(" " + sz.str() + " "),
            }) | border;

            if (static_cast<int>(i) == selected_row)
                row = row | inverted;
            lines.push_back(std::move(row));
        }

        Element header = hbox({
            text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
            text(" Nodes ") | color(Color::Cyan),
            filler(),
            text("Count: " + std::to_string(rows.size())) | dim
        }) | border;

        const float focus_y =
            rows.empty() ? 0.0f : static_cast<float>(selected_row + 1) / static_cast<float>(rows.size() + 1);
        Element body = vbox(std::move(lines))
            | focusPositionRelative(0.0f, focus_y)
            | yframe
            | vscroll_indicator
            | flex;
        Element footer = text("Scroll: wheel / ↑↓ / PgUp PgDn   Quit: Enter or Q") | dim;

        return vbox({ header, body, footer }) | border;
    });

    ui = ftxui::CatchEvent(ui, [&](Event event) {
        if (!rows.empty()) {
            const int max_idx = static_cast<int>(rows.size()) - 1;
            if (event == Event::ArrowUp) {
                selected_row = (std::max)(0, selected_row - 1);
                return true;
            }
            if (event == Event::ArrowDown) {
                selected_row = (std::min)(max_idx, selected_row + 1);
                return true;
            }
            if (event == Event::PageUp) {
                selected_row = (std::max)(0, selected_row - 10);
                return true;
            }
            if (event == Event::PageDown) {
                selected_row = (std::min)(max_idx, selected_row + 10);
                return true;
            }
            if (event.is_mouse()) {
                const auto& m = event.mouse();
                if (m.button == Mouse::WheelUp) {
                    selected_row = (std::max)(0, selected_row - 3);
                    return true;
                }
                if (m.button == Mouse::WheelDown) {
                    selected_row = (std::min)(max_idx, selected_row + 3);
                    return true;
                }
            }
        }
        if (event == Event::Return || event == Event::Character('q') || event == Event::Character('Q')) {
            screen.Exit();
            return true;
        }
        return false;
    });
    screen.Loop(ui);
}

void render_elements_list(entt::registry& reg) {
    (void)render_elements_list_select(reg);
}

int render_elements_list_select(entt::registry& reg) {
    struct Row {
        int eid;
        int type_id;
        std::string nodes;
    };
    std::vector<Row> rows;
    auto view = reg.view<const ::Component::ElementID, const ::Component::ElementType, const ::Component::Connectivity>();
    rows.reserve(view.size_hint());
    for (auto e : view) {
        const int eid = view.get<const ::Component::ElementID>(e).value;
        const int type_id = view.get<const ::Component::ElementType>(e).type_id;
        const auto& conn = view.get<const ::Component::Connectivity>(e);

        std::vector<int> nids;
        nids.reserve(conn.nodes.size());
        for (auto ne : conn.nodes) {
            if (!reg.valid(ne) || !reg.all_of<::Component::NodeID>(ne)) continue;
            nids.push_back(reg.get<::Component::NodeID>(ne).value);
        }

        std::string nodes_str;
        const std::size_t show_n = (std::min<std::size_t>)(nids.size(), 8);
        for (std::size_t i = 0; i < show_n; ++i) {
            if (i > 0) nodes_str += ", ";
            nodes_str += std::to_string(nids[i]);
        }
        if (nids.size() > show_n) nodes_str += " ...";

        rows.push_back(Row{ eid, type_id, std::move(nodes_str) });
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.eid < b.eid; });

    int selected_row = rows.empty() ? -1 : 0;
    int selected_eid = -1;
    auto screen = ScreenInteractive::Fullscreen();

    ftxui::Component ui = ftxui::Renderer([&] {
        Elements lines;
        lines.push_back(
            hbox({
                text(" ElementID ") | bold,
                text(" | "),
                text(" TypeID ") | bold,
                text(" | "),
                text(" Nodes ") | bold,
            }) | border);

        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];

            Element row = hbox({
                text(" " + std::to_string(r.eid) + " ") | color(Color::Cyan),
                text(" | "),
                text(" " + std::to_string(r.type_id) + " ") | color(Color::YellowLight),
                text(" | "),
                text(" " + r.nodes + " "),
            }) | border;

            if (static_cast<int>(i) == selected_row)
                row = row | inverted;
            lines.push_back(std::move(row));
        }

        Element header = hbox({
            text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
            text(" Elements ") | color(Color::Cyan),
            filler(),
            text("Count: " + std::to_string(rows.size())) | dim
        }) | border;

        const float focus_y =
            rows.empty() ? 0.0f : static_cast<float>(selected_row + 1) / static_cast<float>(rows.size() + 1);
        Element body = vbox(std::move(lines))
            | focusPositionRelative(0.0f, focus_y)
            | yframe
            | vscroll_indicator
            | flex;
        Element footer = text("Scroll: wheel / ↑↓ / PgUp PgDn   Select: Enter   Quit: Q / Esc") | dim;

        return vbox({ header, body, footer }) | border;
    });

    ui = ftxui::CatchEvent(ui, [&](Event event) {
        if (!rows.empty()) {
            const int max_idx = static_cast<int>(rows.size()) - 1;
            if (event == Event::ArrowUp) {
                selected_row = (std::max)(0, selected_row - 1);
                return true;
            }
            if (event == Event::ArrowDown) {
                selected_row = (std::min)(max_idx, selected_row + 1);
                return true;
            }
            if (event == Event::PageUp) {
                selected_row = (std::max)(0, selected_row - 10);
                return true;
            }
            if (event == Event::PageDown) {
                selected_row = (std::min)(max_idx, selected_row + 10);
                return true;
            }
            if (event.is_mouse()) {
                const auto& m = event.mouse();
                if (m.button == Mouse::WheelUp) {
                    selected_row = (std::max)(0, selected_row - 3);
                    return true;
                }
                if (m.button == Mouse::WheelDown) {
                    selected_row = (std::min)(max_idx, selected_row + 3);
                    return true;
                }
            }
        }
        if (event == Event::Return) {
            if (!rows.empty() && selected_row >= 0 && selected_row < static_cast<int>(rows.size()))
                selected_eid = rows[static_cast<std::size_t>(selected_row)].eid;
            screen.Exit();
            return true;
        }
        if (event == Event::Escape || event == Event::Character('q') || event == Event::Character('Q')) {
            screen.Exit();
            return true;
        }
        return false;
    });
    screen.Loop(ui);
    return selected_eid;
}

namespace {

Element status_lines_element(const std::vector<std::string>& lines) {
    Elements els;
    els.reserve(lines.size());
    for (const auto& s : lines) {
        els.push_back(paragraph(s));
    }
    return vbox(std::move(els));
}

struct TuiLogLine {
    spdlog::level::level_enum level;
    std::string text;
};

Decorator level_decorator(spdlog::level::level_enum lvl) {
    using spdlog::level::level_enum;
    switch (lvl) {
        case level_enum::trace:    return dim;
        case level_enum::debug:    return color(Color::GrayDark);
        case level_enum::info:     return color(Color::GreenLight);
        case level_enum::warn:     return color(Color::YellowLight);
        case level_enum::err:      return color(Color::RedLight);
        case level_enum::critical: return color(Color::RedLight) | bold;
        default:                   return nothing;
    }
}

Element status_lines_element(const std::vector<TuiLogLine>& lines) {
    Elements els;
    els.reserve(lines.size());
    for (const auto& s : lines) {
        els.push_back(paragraph(s.text) | level_decorator(s.level));
    }
    return vbox(std::move(els));
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

std::mutex g_tui_log_mutex;
std::deque<TuiLogLine> g_tui_log_lines;
std::shared_ptr<spdlog::sinks::sink> g_tui_log_sink;

class TuiLogSink final : public spdlog::sinks::base_sink<std::mutex> {
protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        spdlog::memory_buf_t formatted;
        formatter_->format(msg, formatted);
        std::string line = fmt::to_string(formatted);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (line.empty()) return;

        std::lock_guard<std::mutex> lock(g_tui_log_mutex);
        g_tui_log_lines.push_back(TuiLogLine{ msg.level, std::move(line) });
        if (g_tui_log_lines.size() > 1000) {
            g_tui_log_lines.erase(g_tui_log_lines.begin(), g_tui_log_lines.begin() + 200);
        }
    }

    void flush_() override {}
};

std::vector<TuiLogLine> tui_log_lines_snapshot() {
    std::lock_guard<std::mutex> lock(g_tui_log_mutex);
    return std::vector<TuiLogLine>(g_tui_log_lines.begin(), g_tui_log_lines.end());
}

enum class FocusRegion {
    TopLeftView = 0,
    TopRightOutput = 1,
    BottomCommand = 2,
};

} // namespace

void install_tui_log_sink() {
    if (g_tui_log_sink) return;
    auto logger = spdlog::default_logger();
    if (!logger) return;

    auto sink = std::make_shared<TuiLogSink>();
    logger->sinks().push_back(sink);
    g_tui_log_sink = std::move(sink);
}

void run_app_tui(AppSession& session) {
    using namespace ftxui;

    auto screen = ScreenInteractive::Fullscreen();

    std::string input;
    // top_view: if set, overrides status line view (e.g. panel/list render).
    std::optional<Element> top_view;
    float left_focus = 0.0f;
    float right_focus = 0.0f;
    FocusRegion focus_region = FocusRegion::BottomCommand;
    enum class LeftViewMode { None, StaticElement, NodesList, ElementsList };
    LeftViewMode left_view_mode = LeftViewMode::None;
    struct NodeListRow {
        int nid;
        double x, y, z;
    };
    std::vector<NodeListRow> node_rows;
    int node_selected_row = -1;

    struct ElemListRow {
        int eid;
        int type_id;
        std::string nodes;
    };
    std::vector<ElemListRow> elem_rows;
    int elem_selected_row = -1;

    auto open_panel_in_top_view = [&](const std::string& type, const std::string& id_or_name) -> bool {
        PanelEntityKind kind = PanelEntityKind::Unknown;
        std::string display_id;
        entt::entity e = resolve_panel_entity(
            session.data.registry, &session.inspector, type, id_or_name, &kind, &display_id);
        if (e == entt::null) {
            spdlog::error("Panel: entity not found. Ensure mesh is loaded and index built.");
            return false;
        }

        // Build the same document structure as render_panel(), but keep it inside the TUI top area.
        const char* kind_str = "Entity";
        switch (kind) {
            case PanelEntityKind::Node:    kind_str = "Node";    break;
            case PanelEntityKind::Element: kind_str = "Element"; break;
            case PanelEntityKind::Part:    kind_str = "Part";    break;
            case PanelEntityKind::Set:     kind_str = "Set";     break;
            default: break;
        }

        Elements component_views;
        for (const auto& entry : ComponentTUIRegistry::instance().entries()) {
            if (entry.has_component(session.data.registry, e)) {
                component_views.push_back(
                    window(text(entry.display_name) | bold, entry.render(session.data.registry, e, &session.inspector)));
            }
        }
        if (kind == PanelEntityKind::Node) {
            component_views.push_back(window(text("Force path") | bold,
                force_path_element(session.data.registry, e, &session.inspector)));
        }

        top_view = vbox({
            hbox({
                text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
                text(" Universal Inspector ") | color(Color::Cyan),
                filler(),
                text(std::string(kind_str) + " " + display_id) | dim,
            }) | border,
            vbox(std::move(component_views)),
            text("Tip: type another command below. Use 'help' for quick help.") | dim,
        });
        left_view_mode = LeftViewMode::StaticElement;
        left_focus = 0.0f;
        focus_region = FocusRegion::TopLeftView;
        return true;
    };

    auto sync_nodes_focus = [&]() {
        if (node_rows.empty() || node_selected_row < 0) {
            left_focus = 0.0f;
            return;
        }
        left_focus = clamp01(
            static_cast<float>(node_selected_row + 1) /
            static_cast<float>(node_rows.size() + 1));
    };

    auto sync_elems_focus = [&]() {
        if (elem_rows.empty() || elem_selected_row < 0) {
            left_focus = 0.0f;
            return;
        }
        left_focus = clamp01(
            static_cast<float>(elem_selected_row + 1) /
            static_cast<float>(elem_rows.size() + 1));
    };

    auto render_elements_list_element = [&]() -> Element {
        Elements lines;
        lines.push_back(
            hbox({
                text(" ElementID ") | bold, text(" | "),
                text(" TypeID ") | bold,    text(" | "),
                text(" Nodes ") | bold,
            }) | border);

        for (size_t i = 0; i < elem_rows.size(); ++i) {
            const auto& r = elem_rows[i];
            Element row = hbox({
                text(" " + std::to_string(r.eid) + " ") | color(Color::Cyan),
                text(" | "),
                text(" " + std::to_string(r.type_id) + " ") | color(Color::YellowLight),
                text(" | "),
                text(" " + r.nodes + " "),
            }) | border;
            if (static_cast<int>(i) == elem_selected_row)
                row = row | inverted;
            lines.push_back(std::move(row));
        }

        return vbox({
            hbox({
                text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
                text(" Elements ") | color(Color::Cyan),
                filler(),
                text("Count: " + std::to_string(elem_rows.size())) | dim
            }) | border,
            vbox(std::move(lines)),
            text("Scroll: wheel / ↑↓ / PgUp PgDn   Enter: panel   Tip: use 'panel elem <eid>' for details.") | dim,
        });
    };

    auto render_nodes_list_element = [&]() -> Element {
        Elements lines;
        lines.push_back(
            hbox({
                text(" NodeID ") | bold, text(" | "),
                text(" X ") | bold,     text(" | "),
                text(" Y ") | bold,     text(" | "),
                text(" Z ") | bold,
            }) | border);

        for (size_t i = 0; i < node_rows.size(); ++i) {
            const auto& r = node_rows[i];
            std::ostringstream sx, sy, sz;
            sx.setf(std::ios::fixed); sy.setf(std::ios::fixed); sz.setf(std::ios::fixed);
            sx << std::setprecision(6) << r.x;
            sy << std::setprecision(6) << r.y;
            sz << std::setprecision(6) << r.z;
            Element row = hbox({
                text(" " + std::to_string(r.nid) + " ") | color(Color::Cyan),
                text(" | "),
                text(" " + sx.str() + " "),
                text(" | "),
                text(" " + sy.str() + " "),
                text(" | "),
                text(" " + sz.str() + " "),
            }) | border;
            if (static_cast<int>(i) == node_selected_row) {
                row = row | inverted;
            }
            lines.push_back(std::move(row));
        }

        return vbox({
            hbox({
                text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
                text(" Nodes ") | color(Color::Cyan),
                filler(),
                text("Count: " + std::to_string(node_rows.size())) | dim
            }) | border,
            vbox(std::move(lines)),
            text("Scroll: wheel / ↑↓ / PgUp PgDn   Enter: panel   Tip: use 'panel node <nid>' for details.") | dim,
        });
    };

    auto input_component = Input(&input, "command...");

    auto ui = Renderer(input_component, [&] {
        Element top_left;
        if (left_view_mode == LeftViewMode::NodesList) {
            top_left = render_nodes_list_element();
        } else if (left_view_mode == LeftViewMode::ElementsList) {
            top_left = render_elements_list_element();
        } else if (top_view.has_value()) {
            top_left = top_view.value();
        } else {
            top_left = text("No active TUI view. Try: list_nodes, list_elements, panel ...") | dim;
        }

        const bool left_focused = focus_region == FocusRegion::TopLeftView;
        const bool right_focused = focus_region == FocusRegion::TopRightOutput;
        const bool command_focused = focus_region == FocusRegion::BottomCommand;
        const std::vector<TuiLogLine> log_lines = tui_log_lines_snapshot();

        Element left_box =
            window(
                text(left_focused ? " TUI View [TAB] " : " TUI View ") | bold,
                top_left
                    | focusPositionRelative(0.0f, left_focus)
                    | yframe
                    | vscroll_indicator
                    | flex);

        Element right_box =
            window(
                text(right_focused ? " Output [TAB] " : " Output ") | bold,
                status_lines_element(log_lines)
                    | focusPositionRelative(0.0f, right_focus)
                    | yframe
                    | vscroll_indicator
                    | flex);

        Element top_row = hbox({
            left_box | flex,
            right_box | size(WIDTH, EQUAL, 56),
        }) | flex;

        Element bottom_box =
            window(
                text(command_focused ? " Command [TAB] " : " Command ") | bold,
                hbox({
                    text("NovaFEA> ") | color(Color::Cyan),
                    input_component->Render() | flex,
                }));

        return vbox({
            top_row | flex,
            bottom_box,
        }) | border;
    });

    ui = CatchEvent(ui, [&](Event event) {
        if (event == Event::Tab) {
            const int idx = (static_cast<int>(focus_region) + 1) % 3;
            focus_region = static_cast<FocusRegion>(idx);
            return true;
        }

        if (focus_region != FocusRegion::BottomCommand && event.is_character()) {
            return true;
        }

        if (event == Event::Return) {
            if (focus_region == FocusRegion::TopLeftView) {
                if (left_view_mode == LeftViewMode::NodesList && !node_rows.empty() &&
                    node_selected_row >= 0 && node_selected_row < static_cast<int>(node_rows.size())) {
                    (void)open_panel_in_top_view("node", std::to_string(node_rows[static_cast<std::size_t>(node_selected_row)].nid));
                } else if (left_view_mode == LeftViewMode::ElementsList && !elem_rows.empty() &&
                    elem_selected_row >= 0 && elem_selected_row < static_cast<int>(elem_rows.size())) {
                    (void)open_panel_in_top_view("elem", std::to_string(elem_rows[static_cast<std::size_t>(elem_selected_row)].eid));
                }
                return true;
            }
            if (focus_region != FocusRegion::BottomCommand) {
                return true;
            }
            const std::string cmd = input;
            input.clear();

            if (cmd.empty()) return true;

            // Clear the view when a normal command is entered; view commands will set it again.
            top_view.reset();
            left_view_mode = LeftViewMode::None;
            node_rows.clear();
            node_selected_row = -1;
            elem_rows.clear();
            elem_selected_row = -1;

            if (cmd == "quit" || cmd == "exit") {
                session.is_running = false;
                screen.Exit();
                return true;
            }

            // Minimal in-TUI help. Full logs are still in spdlog.
            if (cmd == "help") {
                std::vector<std::string> help = {
                    "Commands:",
                    "  import <file>",
                    "  info",
                    "  list_nodes",
                    "  list_elements",
                    "  panel node <nid>",
                    "  panel elem <eid>",
                    "  panel part <name>",
                    "  panel set <name>",
                    "  quit / exit",
                    "",
                    "Note: other commands are supported; they will execute and log to the normal logger.",
                };
                top_view = window(text(" Help ") | bold, status_lines_element(help));
                left_view_mode = LeftViewMode::StaticElement;
                left_focus = 0.0f;
                focus_region = FocusRegion::TopLeftView;
                return true;
            }

            // list_*: render into top.
            if (cmd == "list_nodes") {
                auto& reg = session.data.registry;
                auto view = reg.view<const ::Component::NodeID, const ::Component::Position>();
                node_rows.clear();
                node_rows.reserve(view.size_hint());
                for (auto e : view) {
                    const auto& id = view.get<const ::Component::NodeID>(e);
                    const auto& p = view.get<const ::Component::Position>(e);
                    node_rows.push_back(NodeListRow{ id.value, p.x, p.y, p.z });
                }
                std::sort(node_rows.begin(), node_rows.end(), [](const NodeListRow& a, const NodeListRow& b) { return a.nid < b.nid; });
                node_selected_row = node_rows.empty() ? -1 : 0;
                left_view_mode = LeftViewMode::NodesList;
                sync_nodes_focus();
                focus_region = FocusRegion::TopLeftView;
                return true;
            }

            if (cmd == "list_elements") {
                auto& reg = session.data.registry;
                auto view = reg.view<const ::Component::ElementID, const ::Component::ElementType, const ::Component::Connectivity>();
                elem_rows.clear();
                elem_rows.reserve(view.size_hint());
                for (auto e : view) {
                    const int eid = view.get<const ::Component::ElementID>(e).value;
                    const int type_id = view.get<const ::Component::ElementType>(e).type_id;
                    const auto& conn = view.get<const ::Component::Connectivity>(e);
                    std::vector<int> nids;
                    nids.reserve(conn.nodes.size());
                    for (auto ne : conn.nodes) {
                        if (!reg.valid(ne) || !reg.all_of<::Component::NodeID>(ne)) continue;
                        nids.push_back(reg.get<::Component::NodeID>(ne).value);
                    }
                    std::string nodes_str;
                    const std::size_t show_n = (std::min<std::size_t>)(nids.size(), 8);
                    for (std::size_t i = 0; i < show_n; ++i) {
                        if (i > 0) nodes_str += ", ";
                        nodes_str += std::to_string(nids[i]);
                    }
                    if (nids.size() > show_n) nodes_str += " ...";
                    elem_rows.push_back(ElemListRow{ eid, type_id, std::move(nodes_str) });
                }
                std::sort(elem_rows.begin(), elem_rows.end(), [](const ElemListRow& a, const ElemListRow& b) { return a.eid < b.eid; });
                elem_selected_row = elem_rows.empty() ? -1 : 0;
                left_view_mode = LeftViewMode::ElementsList;
                sync_elems_focus();
                focus_region = FocusRegion::TopLeftView;
                return true;
            }

            // panel: render into top using existing universal inspector panel element composition.
            if (starts_with(cmd, "panel ")) {
                std::stringstream ss(cmd);
                std::string keyword, type, id_or_name;
                ss >> keyword >> type >> id_or_name;
                if (type.empty() || id_or_name.empty()) {
                    spdlog::error("Usage: panel <type> <id_or_name>  (type: node|elem|element|part|set)");
                    return true;
                }
                (void)open_panel_in_top_view(type, id_or_name);
                return true;
            }

            // Fallback: execute existing command processor (logs go to spdlog sinks).
            process_command(cmd, session);
            return true;
        }

        // Quick escape for clearing a view.
        if (event == Event::Escape) {
            top_view.reset();
            left_view_mode = LeftViewMode::None;
            node_rows.clear();
            node_selected_row = -1;
            elem_rows.clear();
            elem_selected_row = -1;
            left_focus = 0.0f;
            return true;
        }

        if (focus_region == FocusRegion::TopLeftView &&
            left_view_mode == LeftViewMode::NodesList &&
            !node_rows.empty()) {
            const int max_idx = static_cast<int>(node_rows.size()) - 1;
            if (event == Event::ArrowUp) {
                node_selected_row = (std::max)(0, node_selected_row - 1);
                sync_nodes_focus();
                return true;
            }
            if (event == Event::ArrowDown) {
                node_selected_row = (std::min)(max_idx, node_selected_row + 1);
                sync_nodes_focus();
                return true;
            }
            if (event == Event::PageUp) {
                node_selected_row = (std::max)(0, node_selected_row - 10);
                sync_nodes_focus();
                return true;
            }
            if (event == Event::PageDown) {
                node_selected_row = (std::min)(max_idx, node_selected_row + 10);
                sync_nodes_focus();
                return true;
            }
        }

        if (focus_region == FocusRegion::TopLeftView &&
            left_view_mode == LeftViewMode::ElementsList &&
            !elem_rows.empty()) {
            const int max_idx = static_cast<int>(elem_rows.size()) - 1;
            if (event == Event::ArrowUp) {
                elem_selected_row = (std::max)(0, elem_selected_row - 1);
                sync_elems_focus();
                return true;
            }
            if (event == Event::ArrowDown) {
                elem_selected_row = (std::min)(max_idx, elem_selected_row + 1);
                sync_elems_focus();
                return true;
            }
            if (event == Event::PageUp) {
                elem_selected_row = (std::max)(0, elem_selected_row - 10);
                sync_elems_focus();
                return true;
            }
            if (event == Event::PageDown) {
                elem_selected_row = (std::min)(max_idx, elem_selected_row + 10);
                sync_elems_focus();
                return true;
            }
        }

        if (focus_region == FocusRegion::TopLeftView && event == Event::ArrowUp) {
            left_focus = clamp01(left_focus - 0.03f);
            return true;
        }
        if (focus_region == FocusRegion::TopLeftView && event == Event::ArrowDown) {
            left_focus = clamp01(left_focus + 0.03f);
            return true;
        }
        if (focus_region == FocusRegion::TopLeftView && event == Event::PageUp) {
            left_focus = clamp01(left_focus - 0.12f);
            return true;
        }
        if (focus_region == FocusRegion::TopLeftView && event == Event::PageDown) {
            left_focus = clamp01(left_focus + 0.12f);
            return true;
        }
        if (focus_region == FocusRegion::TopRightOutput && event == Event::ArrowUp) {
            right_focus = clamp01(right_focus - 0.03f);
            return true;
        }
        if (focus_region == FocusRegion::TopRightOutput && event == Event::ArrowDown) {
            right_focus = clamp01(right_focus + 0.03f);
            return true;
        }
        if (focus_region == FocusRegion::TopRightOutput && event == Event::PageUp) {
            right_focus = clamp01(right_focus - 0.12f);
            return true;
        }
        if (focus_region == FocusRegion::TopRightOutput && event == Event::PageDown) {
            right_focus = clamp01(right_focus + 0.12f);
            return true;
        }
        if (event.is_mouse()) {
            const auto& m = event.mouse();
            if (focus_region == FocusRegion::TopLeftView &&
                left_view_mode == LeftViewMode::NodesList &&
                !node_rows.empty()) {
                const int max_idx = static_cast<int>(node_rows.size()) - 1;
                if (m.button == Mouse::WheelUp) {
                    node_selected_row = (std::max)(0, node_selected_row - 3);
                    sync_nodes_focus();
                    return true;
                }
                if (m.button == Mouse::WheelDown) {
                    node_selected_row = (std::min)(max_idx, node_selected_row + 3);
                    sync_nodes_focus();
                    return true;
                }
            }
            if (focus_region == FocusRegion::TopLeftView &&
                left_view_mode == LeftViewMode::ElementsList &&
                !elem_rows.empty()) {
                const int max_idx = static_cast<int>(elem_rows.size()) - 1;
                if (m.button == Mouse::WheelUp) {
                    elem_selected_row = (std::max)(0, elem_selected_row - 3);
                    sync_elems_focus();
                    return true;
                }
                if (m.button == Mouse::WheelDown) {
                    elem_selected_row = (std::min)(max_idx, elem_selected_row + 3);
                    sync_elems_focus();
                    return true;
                }
            }
            if (focus_region == FocusRegion::TopLeftView && m.button == Mouse::WheelUp) {
                left_focus = clamp01(left_focus - 0.06f);
                return true;
            }
            if (focus_region == FocusRegion::TopLeftView && m.button == Mouse::WheelDown) {
                left_focus = clamp01(left_focus + 0.06f);
                return true;
            }
            if (focus_region == FocusRegion::TopRightOutput && m.button == Mouse::WheelUp) {
                right_focus = clamp01(right_focus - 0.06f);
                return true;
            }
            if (focus_region == FocusRegion::TopRightOutput && m.button == Mouse::WheelDown) {
                right_focus = clamp01(right_focus + 0.06f);
                return true;
            }

            // Mouse-wheel scroll for Output pane when hovering it (no TAB required).
            if (m.button == Mouse::WheelUp || m.button == Mouse::WheelDown) {
                const int dimx = screen.dimx();
                const int dimy = screen.dimy();
                const int output_width = 56;
                const int output_left = (std::max)(0, dimx - output_width - 1);
                const int top_area_bottom = (std::max)(0, dimy - 3); // command box is ~3 lines tall
                const bool over_output = (m.x >= output_left) && (m.y >= 0) && (m.y < top_area_bottom);
                if (over_output) {
                    right_focus = clamp01(right_focus + (m.button == Mouse::WheelUp ? -0.06f : 0.06f));
                    return true;
                }
            }
        }
        return false;
    });

    session.is_running = true;
    screen.Loop(ui);
}

} // namespace tui
