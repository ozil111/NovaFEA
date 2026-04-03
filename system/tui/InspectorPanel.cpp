/**
 * TUI Universal Inspection Panel - Entity inspection logic.
 */
#include "tui/ComponentTUI.h"
#include "simdroid/SimdroidInspector.h"
#include "components/mesh_components.h"
#include "components/simdroid_components.h"
#include "components/material_components.h"
#include "components/property_components.h"
#include "PartGraph.h"
#include "analysis/GraphBuilder.h"
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>
#include <iostream>
#include <algorithm>

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

template <typename TargetComponent>
entt::entity find_named_entity_with_component(entt::registry& reg, const std::string& name) {
    auto view = reg.view<const ::Component::SetName, const TargetComponent>();
    for (auto e : view) {
        if (view.template get<const ::Component::SetName>(e).value == name)
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

} // anonymous namespace

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

    if (type == "material" || type == "mat") {
        // Try to find by MaterialID first
        int mid = 0;
        try { mid = std::stoi(id_or_name); } catch (...) {
            entt::entity me = find_named_entity_with_component<::Component::MaterialID>(reg, id_or_name);
            if (me != entt::null) {
                if (out_kind) *out_kind = PanelEntityKind::Material;
                if (out_display_id) *out_display_id = id_or_name;
                return me;
            }
            return entt::null;
        }
        // Find by MaterialID value
        auto view = reg.view<const ::Component::MaterialID>();
        for (auto e : view) {
            if (view.get<const ::Component::MaterialID>(e).value == mid) {
                if (out_kind) *out_kind = PanelEntityKind::Material;
                if (out_display_id) *out_display_id = std::to_string(mid);
                return e;
            }
        }
        return entt::null;
    }

    if (type == "section" || type == "prop" || type == "property") {
        // Try to find by PropertyID first
        int pid = 0;
        try { pid = std::stoi(id_or_name); } catch (...) {
            entt::entity pe = find_named_entity_with_component<::Component::PropertyID>(reg, id_or_name);
            if (pe != entt::null) {
                if (out_kind) *out_kind = PanelEntityKind::Section;
                if (out_display_id) *out_display_id = id_or_name;
                return pe;
            }
            return entt::null;
        }
        // Find by PropertyID value
        auto view = reg.view<const ::Component::PropertyID>();
        for (auto e : view) {
            if (view.get<const ::Component::PropertyID>(e).value == pid) {
                if (out_kind) *out_kind = PanelEntityKind::Section;
                if (out_display_id) *out_display_id = std::to_string(pid);
                return e;
            }
        }
        return entt::null;
    }

    return entt::null;
}

void render_panel(entt::registry& reg, entt::entity e, SimdroidInspector* insp,
    PanelEntityKind kind, const std::string& display_id)
{
    const char* kind_str = "Entity";
    switch (kind) {
        case PanelEntityKind::Node:     kind_str = "Node";     break;
        case PanelEntityKind::Element:  kind_str = "Element";  break;
        case PanelEntityKind::Part:     kind_str = "Part";     break;
        case PanelEntityKind::Set:      kind_str = "Set";      break;
        case PanelEntityKind::Material: kind_str = "Material"; break;
        case PanelEntityKind::Section:  kind_str = "Section";  break;
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

} // namespace tui
