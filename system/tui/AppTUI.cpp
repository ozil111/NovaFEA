/**
 * TUI Universal Inspection Panel - Interactive application TUI.
 */
#include "tui/ComponentTUI.h"
#include "simdroid/SimdroidInspector.h"
#include "components/mesh_components.h"
#include "components/simdroid_components.h"
#include "components/material_components.h"
#include "components/property_components.h"
#include "AppSession.h"
#include "CommandProcessor.h"
#include <spdlog/spdlog.h>
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
    std::vector<std::string> history;
    int history_index = -1;
    std::string input_backup;
    float left_focus = 0.0f;
    float right_focus = 0.0f;
    FocusRegion focus_region = FocusRegion::BottomCommand;
    enum class LeftViewMode { None, StaticElement, NodesList, ElementsList, PartsList, SetList };
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

    struct PartListRow {
        std::string name;
        std::string element_set;
        std::string material_id;
        std::string section_id;
        std::size_t elem_count;
    };
    std::vector<PartListRow> part_rows;
    int part_selected_row = -1;

    // top_panel: interactive component shown in the left-top view.
    std::optional<ftxui::Component> top_panel;

    struct ViewState {
        LeftViewMode mode = LeftViewMode::None;
        std::string entity_type;
        std::string entity_id;
        int node_idx = -1;
        int elem_idx = -1;
        int part_idx = -1;
        float left_focus = 0.0f;
        FocusRegion focus_region = FocusRegion::BottomCommand;
        std::string label;
    };

    std::vector<ViewState> view_history;
    constexpr std::size_t kMaxViewHistoryDepth = 20;

    std::string current_panel_type;
    std::string current_panel_id;

    auto save_view_state = [&](std::string label) {
        if (view_history.size() >= kMaxViewHistoryDepth) {
            view_history.erase(view_history.begin());
        }
        ViewState st;
        st.mode = left_view_mode;
        st.node_idx = node_selected_row;
        st.elem_idx = elem_selected_row;
        st.part_idx = part_selected_row;
        st.left_focus = left_focus;
        st.focus_region = focus_region;
        if (left_view_mode == LeftViewMode::StaticElement) {
            st.entity_type = current_panel_type;
            st.entity_id = current_panel_id;
        }
        st.label = std::move(label);
        view_history.push_back(std::move(st));
    };

    auto clear_left_view = [&]() {
        top_panel.reset();
        left_view_mode = LeftViewMode::None;
        node_rows.clear();
        node_selected_row = -1;
        elem_rows.clear();
        elem_selected_row = -1;
        part_rows.clear();
        part_selected_row = -1;
        left_focus = 0.0f;
        current_panel_type.clear();
        current_panel_id.clear();
    };

    std::function<bool(const std::string&, const std::string&, bool)> open_panel_in_top_view;
    open_panel_in_top_view = [&](const std::string& type, const std::string& id_or_name, bool push_history) -> bool {
        if (push_history) {
            save_view_state("panel " + type + " " + id_or_name);
        }
        PanelEntityKind kind = PanelEntityKind::Unknown;
        std::string display_id;
        entt::entity e = resolve_panel_entity(
            session.data.registry, &session.inspector, type, id_or_name, &kind, &display_id);
        if (e == entt::null) {
            spdlog::error("Panel: entity not found. Ensure mesh is loaded and index built.");
            return false;
        }

        // Build the same document structure as render_panel(), but keep it inside the TUI top area.
        const char* kind_str_c = "Entity";
        switch (kind) {
            case PanelEntityKind::Node:     kind_str_c = "Node";     break;
            case PanelEntityKind::Element:  kind_str_c = "Element";  break;
            case PanelEntityKind::Part:     kind_str_c = "Part";     break;
            case PanelEntityKind::Set:      kind_str_c = "Set";      break;
            case PanelEntityKind::Material: kind_str_c = "Material"; break;
            case PanelEntityKind::Section:  kind_str_c = "Section";  break;
            default: break;
        }
        const std::string kind_str = kind_str_c;

        // Set: build a scrollable list panel inside top_panel (StaticElement mode)
        if (kind == PanelEntityKind::Set) {
            struct SetMemberRow {
                int id;
                std::string type;  // "node" or "elem"
                std::string extra;
            };
            std::vector<SetMemberRow> rows;
            std::string member_type;
            auto& reg = session.data.registry;

            // NodeSetMembers
            if (reg.all_of<::Component::NodeSetMembers>(e)) {
                const auto& members = reg.get<::Component::NodeSetMembers>(e).members;
                member_type = "node";
                for (auto me : members) {
                    if (!reg.valid(me) || !reg.all_of<::Component::NodeID>(me)) continue;
                    const int nid = reg.get<::Component::NodeID>(me).value;
                    std::string extra;
                    if (reg.all_of<::Component::Position>(me)) {
                        const auto& p = reg.get<::Component::Position>(me);
                        std::ostringstream sx, sy, sz;
                        sx.setf(std::ios::fixed); sy.setf(std::ios::fixed); sz.setf(std::ios::fixed);
                        sx << std::setprecision(6) << p.x;
                        sy << std::setprecision(6) << p.y;
                        sz << std::setprecision(6) << p.z;
                        extra = sx.str() + ", " + sy.str() + ", " + sz.str();
                    }
                    rows.push_back(SetMemberRow{ nid, "node", std::move(extra) });
                }
            }
            // ElementSetMembers
            if (reg.all_of<::Component::ElementSetMembers>(e)) {
                const auto& members = reg.get<::Component::ElementSetMembers>(e).members;
                member_type = "elem";
                for (auto me : members) {
                    if (!reg.valid(me) || !reg.all_of<::Component::ElementID>(me)) continue;
                    const int eid = reg.get<::Component::ElementID>(me).value;
                    std::string extra;
                    if (reg.all_of<::Component::ElementType>(me)) {
                        extra = "type=" + std::to_string(reg.get<::Component::ElementType>(me).type_id);
                    }
                    rows.push_back(SetMemberRow{ eid, "elem", std::move(extra) });
                }
            }
            std::sort(rows.begin(), rows.end(),
                [](const SetMemberRow& a, const SetMemberRow& b) { return a.id < b.id; });

            const bool is_node_set = (member_type == "node");
            const std::string set_title = is_node_set ? "NodeSet: " + id_or_name : "ElementSet: " + id_or_name;

            auto shared_rows = std::make_shared<std::vector<SetMemberRow>>(std::move(rows));
            auto shared_selected = std::make_shared<int>(shared_rows->empty() ? -1 : 0);

            ftxui::Component dummy = Container::Vertical({});
            top_panel = Renderer(dummy, [&, shared_rows, shared_selected, is_node_set, set_title]() {
                const int selected = *shared_selected;
                const int total_count = static_cast<int>(shared_rows->size());
                const int margin = 30;
                const int anchor_idx = selected >= 0 ? selected : 0;
                const int start_idx = (std::max)(0, anchor_idx - margin);
                const int end_idx = (std::min)(total_count, anchor_idx + margin + 1);

                Elements lines;
                if (is_node_set) {
                    lines.push_back(hbox({
                        text(" NodeID ") | bold, text(" | "),
                        text(" Position (X, Y, Z) ") | bold,
                    }) | color(Color::Cyan));
                } else {
                    lines.push_back(hbox({
                        text(" ElementID ") | bold, text(" | "),
                        text(" Type ") | bold,
                    }) | color(Color::YellowLight));
                }
                lines.push_back(separatorLight());

                for (int i = start_idx; i < end_idx; ++i) {
                    const auto& r = (*shared_rows)[static_cast<std::size_t>(i)];
                    Element row;
                    if (is_node_set) {
                        row = hbox({
                            text(" " + std::to_string(r.id) + " ") | color(Color::Cyan),
                            text(" | "),
                            text(" " + r.extra + " "),
                        });
                    } else {
                        row = hbox({
                            text(" " + std::to_string(r.id) + " ") | color(Color::Cyan),
                            text(" | "),
                            text(" " + r.extra + " ") | color(Color::YellowLight),
                        });
                    }
                    if (i == selected)
                        row = row | inverted | focus;
                    lines.push_back(std::move(row));
                }

                return vbox({
                    hbox({
                        text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
                        text(" Set ") | color(Color::Cyan),
                        filler(),
                        text(set_title) | dim,
                    }) | border,
                    hbox({
                        text(" " + set_title + " "),
                        filler(),
                        text(
                            "Viewing: " + std::to_string(total_count == 0 ? 0 : start_idx + 1) +
                            "-" + std::to_string(end_idx) +
                            " / " + std::to_string(total_count)
                        ) | dim
                    }),
                    separator(),
                    vbox(std::move(lines)) | flex,
                    separator(),
                    text("Scroll: ArrowUp ArrowDown / PgUp PgDn   Enter: jump to item   Esc: back") | dim,
                });
            });

            const std::string set_id_capture = id_or_name;
            top_panel = CatchEvent(*top_panel, [&, shared_rows, shared_selected, set_id_capture](Event ev) -> bool {
                const int total = static_cast<int>(shared_rows->size());
                if (total == 0) return false;
                const int max_idx = total - 1;
                int& sel = *shared_selected;
                if (ev == Event::ArrowUp) {
                    sel = (std::max)(0, sel - 1);
                    return true;
                }
                if (ev == Event::ArrowDown) {
                    sel = (std::min)(max_idx, sel + 1);
                    return true;
                }
                if (ev == Event::PageUp) {
                    sel = (std::max)(0, sel - 10);
                    return true;
                }
                if (ev == Event::PageDown) {
                    sel = (std::min)(max_idx, sel + 10);
                    return true;
                }
                if (ev == Event::Return && sel >= 0 && sel < total) {
                    const auto& r = (*shared_rows)[static_cast<std::size_t>(sel)];
                    save_view_state("panel set " + set_id_capture);
                    (void)open_panel_in_top_view(r.type, std::to_string(r.id), false);
                    return true;
                }
                return false;
            });

            left_view_mode = LeftViewMode::StaticElement;
            left_focus = 0.0f;
            focus_region = FocusRegion::TopLeftView;
            current_panel_type = type;
            current_panel_id = id_or_name;
            return true;
        }

        std::vector<ftxui::Component> link_buttons;

        if (kind == PanelEntityKind::Node && session.inspector.is_built && session.data.registry.all_of<::Component::NodeID>(e)) {
            const int nid = session.data.registry.get<::Component::NodeID>(e).value;
            auto it = session.inspector.nid_to_elems.find(nid);
            if (it != session.inspector.nid_to_elems.end()) {
                const auto& elem_ids = it->second;
                const std::size_t show = (std::min<std::size_t>)(elem_ids.size(), 40);
                for (std::size_t i = 0; i < show; ++i) {
                    const int eid = elem_ids[i];
                    link_buttons.push_back(Button(std::to_string(eid), [&, eid]() {
                        (void)open_panel_in_top_view("elem", std::to_string(eid), true);
                    }));
                }
            }
        }

        if (kind == PanelEntityKind::Element && session.inspector.is_built && session.data.registry.all_of<::Component::Connectivity>(e)) {
            const auto& c = session.data.registry.get<::Component::Connectivity>(e);
            std::vector<int> nids;
            nids.reserve(c.nodes.size());
            for (auto ne : c.nodes) {
                if (!session.data.registry.valid(ne) || !session.data.registry.all_of<::Component::NodeID>(ne)) continue;
                nids.push_back(session.data.registry.get<::Component::NodeID>(ne).value);
            }
            const std::size_t show = (std::min<std::size_t>)(nids.size(), 40);
            for (std::size_t i = 0; i < show; ++i) {
                const int nid = nids[i];
                link_buttons.push_back(Button(std::to_string(nid), [&, nid]() {
                    (void)open_panel_in_top_view("node", std::to_string(nid), true);
                }));
            }
        }

        if (kind == PanelEntityKind::Part && session.data.registry.all_of<::Component::SimdroidPart>(e)) {
            const auto& part = session.data.registry.get<::Component::SimdroidPart>(e);
            
            // Add Material jump button
            if (session.data.registry.valid(part.material)) {
                std::string mat_id;
                if (session.data.registry.all_of<::Component::MaterialID>(part.material)) {
                    mat_id = std::to_string(session.data.registry.get<::Component::MaterialID>(part.material).value);
                }
                if (!mat_id.empty()) {
                    link_buttons.push_back(Button("Material ID: " + mat_id, [&, part]() {
                        if (session.data.registry.valid(part.material)) {
                            std::string mid;
                            if (session.data.registry.all_of<::Component::MaterialID>(part.material)) {
                                mid = std::to_string(session.data.registry.get<::Component::MaterialID>(part.material).value);
                            }
                            if (!mid.empty())
                                (void)open_panel_in_top_view("material", mid, true);
                        }
                    }));
                }
            }
            
            // Add Section jump button
            if (session.data.registry.valid(part.section)) {
                std::string sec_id;
                if (session.data.registry.all_of<::Component::PropertyID>(part.section)) {
                    sec_id = std::to_string(session.data.registry.get<::Component::PropertyID>(part.section).value);
                }
                if (!sec_id.empty()) {
                    link_buttons.push_back(Button("Section ID: " + sec_id, [&, part]() {
                        if (session.data.registry.valid(part.section)) {
                            std::string pid;
                            if (session.data.registry.all_of<::Component::PropertyID>(part.section)) {
                                pid = std::to_string(session.data.registry.get<::Component::PropertyID>(part.section).value);
                            }
                            if (!pid.empty())
                                (void)open_panel_in_top_view("section", pid, true);
                        }
                    }));
                }
            }
        }

        const bool has_links = !link_buttons.empty();
        ftxui::Component links =
            has_links ? Container::Vertical(std::move(link_buttons)) : Container::Vertical({});

        ftxui::Component panel_root = links;
        top_panel = Renderer(panel_root, [&, e, kind, kind_str, display_id, links, has_links] {
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

            Elements body;
            body.push_back(
                hbox({
                    text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
                    text(" Universal Inspector ") | color(Color::Cyan),
                    filler(),
                    text(std::string(kind_str) + " " + display_id) | dim,
                }) | border);
            body.push_back(vbox(component_views));
            if (has_links) {
                const std::string title =
                    (kind == PanelEntityKind::Node) ? "Elements (jump)" :
                    (kind == PanelEntityKind::Element) ? "Contains Nodes (jump)" : "Related (jump)";
                body.push_back(window(text(title) | bold, links->Render() | flex));
            }
            body.push_back(text("Tip: type another command below. Use 'help' for quick help.") | dim);
            return vbox(std::move(body));
        });
        left_view_mode = LeftViewMode::StaticElement;
        left_focus = 0.0f;
        focus_region = FocusRegion::TopLeftView;
        current_panel_type = type;
        current_panel_id = id_or_name;
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

    auto sync_parts_focus = [&]() {
        if (part_rows.empty() || part_selected_row < 0) {
            left_focus = 0.0f;
            return;
        }
        left_focus = clamp01(
            static_cast<float>(part_selected_row + 1) /
            static_cast<float>(part_rows.size() + 1));
    };

    auto build_nodes_list_view = [&]() {
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
        left_view_mode = LeftViewMode::NodesList;
        top_panel.reset();
        current_panel_type.clear();
        current_panel_id.clear();
        focus_region = FocusRegion::TopLeftView;
    };

    auto build_elements_list_view = [&]() {
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
        left_view_mode = LeftViewMode::ElementsList;
        top_panel.reset();
        current_panel_type.clear();
        current_panel_id.clear();
        focus_region = FocusRegion::TopLeftView;
    };

    auto build_parts_list_view = [&]() {
        auto& reg = session.data.registry;
        auto view = reg.view<const ::Component::SimdroidPart>();
        part_rows.clear();
        part_rows.reserve(static_cast<std::size_t>(view.size()));
        for (auto e : view) {
            const auto& part = view.get<const ::Component::SimdroidPart>(e);
            std::size_t count = 0;
            if (reg.valid(part.element_set) && reg.all_of<::Component::ElementSetMembers>(part.element_set)) {
                count = reg.get<::Component::ElementSetMembers>(part.element_set).members.size();
            }
            std::string element_set_name = "-";
            if (reg.valid(part.element_set) && reg.all_of<::Component::SetName>(part.element_set)) {
                element_set_name = reg.get<::Component::SetName>(part.element_set).value;
            }
            std::string material_id = "-";
            if (reg.valid(part.material) && reg.all_of<::Component::MaterialID>(part.material)) {
                material_id = std::to_string(reg.get<::Component::MaterialID>(part.material).value);
            }
            std::string section_id = "-";
            if (reg.valid(part.section) && reg.all_of<::Component::PropertyID>(part.section)) {
                section_id = std::to_string(reg.get<::Component::PropertyID>(part.section).value);
            }
            part_rows.push_back(PartListRow{
                part.name,
                std::move(element_set_name),
                std::move(material_id),
                std::move(section_id),
                count
            });
        }
        std::sort(part_rows.begin(), part_rows.end(),
            [](const PartListRow& a, const PartListRow& b) { return a.name < b.name; });
        left_view_mode = LeftViewMode::PartsList;
        top_panel.reset();
        current_panel_type.clear();
        current_panel_id.clear();
        focus_region = FocusRegion::TopLeftView;
    };

    auto restore_view_state = [&]() -> bool {
        if (view_history.empty())
            return false;
        ViewState st = std::move(view_history.back());
        view_history.pop_back();

        if (st.mode == LeftViewMode::NodesList) {
            build_nodes_list_view();
            if (!node_rows.empty()) {
                const int max_idx = static_cast<int>(node_rows.size()) - 1;
                node_selected_row = (std::max)(0, (std::min)(max_idx, st.node_idx));
            } else {
                node_selected_row = -1;
            }
            sync_nodes_focus();
        } else if (st.mode == LeftViewMode::ElementsList) {
            build_elements_list_view();
            if (!elem_rows.empty()) {
                const int max_idx = static_cast<int>(elem_rows.size()) - 1;
                elem_selected_row = (std::max)(0, (std::min)(max_idx, st.elem_idx));
            } else {
                elem_selected_row = -1;
            }
            sync_elems_focus();
        } else if (st.mode == LeftViewMode::PartsList) {
            build_parts_list_view();
            if (!part_rows.empty()) {
                const int max_idx = static_cast<int>(part_rows.size()) - 1;
                part_selected_row = (std::max)(0, (std::min)(max_idx, st.part_idx));
            } else {
                part_selected_row = -1;
            }
            sync_parts_focus();
        } else if (st.mode == LeftViewMode::StaticElement && !st.entity_type.empty() && !st.entity_id.empty()) {
            (void)open_panel_in_top_view(st.entity_type, st.entity_id, false);
        } else {
            clear_left_view();
        }

        left_focus = st.left_focus;
        focus_region = st.focus_region;
        return true;
    };

    auto render_elements_list_element = [&]() -> Element {
        Elements lines;
        const int total_count = static_cast<int>(elem_rows.size());
        const int margin = 30;
        const int anchor_idx = elem_selected_row >= 0 ? elem_selected_row : 0;
        const int start_idx = (std::max)(0, anchor_idx - margin);
        const int end_idx = (std::min)(total_count, anchor_idx + margin + 1);
        Element header_row = hbox({
            text(" ElementID ") | bold, text(" | "),
            text(" TypeID ") | bold,    text(" | "),
            text(" Nodes ") | bold,
        }) | color(Color::YellowLight);

        for (int i = start_idx; i < end_idx; ++i) {
            const auto& r = elem_rows[static_cast<std::size_t>(i)];
            Element row = hbox({
                text(" " + std::to_string(r.eid) + " ") | color(Color::Cyan),
                text(" | "),
                text(" " + std::to_string(r.type_id) + " ") | color(Color::YellowLight),
                text(" | "),
                text(" " + r.nodes + " "),
            });
            if (i == elem_selected_row)
                row = row | inverted | focus;
            lines.push_back(std::move(row));
        }

        return vbox({
            hbox({
                text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
                text(" Elements ") | color(Color::Cyan),
                filler(),
                text(
                    "Viewing: " + std::to_string(total_count == 0 ? 0 : start_idx + 1) +
                    "-" + std::to_string(end_idx) +
                    " / " + std::to_string(total_count)
                ) | dim
            }),
            separator(),
            header_row,
            separatorLight(),
            vbox(std::move(lines)) | flex,
            separator(),
            text("Scroll: wheel / ArrowUp ArrowDown / PgUp PgDn   Enter: panel") | dim,
        }) | border;
    };

    auto render_nodes_list_element = [&]() -> Element {
        Elements lines;
        const int total_count = static_cast<int>(node_rows.size());
        const int margin = 30;
        const int anchor_idx = node_selected_row >= 0 ? node_selected_row : 0;
        const int start_idx = (std::max)(0, anchor_idx - margin);
        const int end_idx = (std::min)(total_count, anchor_idx + margin + 1);
        Element header_row = hbox({
            text(" NodeID ") | bold, text(" | "),
            text(" X ") | bold,     text(" | "),
            text(" Y ") | bold,     text(" | "),
            text(" Z ") | bold,
        }) | color(Color::Cyan);

        for (int i = start_idx; i < end_idx; ++i) {
            const auto& r = node_rows[static_cast<std::size_t>(i)];
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
            });
            if (i == node_selected_row) {
                row = row | inverted | focus;
            }
            lines.push_back(std::move(row));
        }

        return vbox({
            hbox({
                text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
                text(" Nodes ") | color(Color::Cyan),
                filler(),
                text(
                    "Viewing: " + std::to_string(total_count == 0 ? 0 : start_idx + 1) +
                    "-" + std::to_string(end_idx) +
                    " / " + std::to_string(total_count)
                ) | dim
            }),
            separator(),
            header_row,
            separatorLight(),
            vbox(std::move(lines)) | flex,
            separator(),
            text("Scroll: wheel / ArrowUp ArrowDown / PgUp PgDn   Enter: panel") | dim,
        }) | border;
    };

    auto render_parts_list_element = [&]() -> Element {
        Elements lines;
        const int total_count = static_cast<int>(part_rows.size());
        const int margin = 30;
        const int anchor_idx = part_selected_row >= 0 ? part_selected_row : 0;
        const int start_idx = (std::max)(0, anchor_idx - margin);
        const int end_idx = (std::min)(total_count, anchor_idx + margin + 1);
        Element header_row = hbox({
            text(" Part ") | bold, text(" | "),
            text(" Element Set ") | bold, text(" | "),
            text(" Material ID ") | bold, text(" | "),
            text(" Section ID ") | bold, text(" | "),
            text(" Elements ") | bold,
        }) | color(Color::Cyan);

        for (int i = start_idx; i < end_idx; ++i) {
            const auto& r = part_rows[static_cast<std::size_t>(i)];
            Element row = hbox({
                text(" " + r.name + " ") | color(Color::Cyan),
                text(" | "),
                text(" " + r.element_set + " ") | color(Color::White),
                text(" | "),
                text(" " + r.material_id + " ") | color(Color::YellowLight),
                text(" | "),
                text(" " + r.section_id + " ") | color(Color::GreenLight),
                text(" | "),
                text(" " + std::to_string(r.elem_count) + " "),
            });
            if (i == part_selected_row)
                row = row | inverted | focus;
            lines.push_back(std::move(row));
        }

        return vbox({
            hbox({
                text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
                text(" Parts ") | color(Color::Cyan),
                filler(),
                text(
                    "Viewing: " + std::to_string(total_count == 0 ? 0 : start_idx + 1) +
                    "-" + std::to_string(end_idx) +
                    " / " + std::to_string(total_count)
                ) | dim
            }),
            separator(),
            header_row,
            separatorLight(),
            vbox(std::move(lines)) | flex,
            separator(),
            text("Scroll: wheel / ArrowUp ArrowDown / PgUp PgDn   Enter: panel") | dim,
        }) | border;
    };

    auto input_component = Input(&input, "command...");

    auto ui = Renderer(input_component, [&] {
        Element top_left;
        if (left_view_mode == LeftViewMode::NodesList) {
            top_left = render_nodes_list_element();
        } else if (left_view_mode == LeftViewMode::ElementsList) {
            top_left = render_elements_list_element();
        } else if (left_view_mode == LeftViewMode::PartsList) {
            top_left = render_parts_list_element();
        } else if (top_panel.has_value()) {
            top_left = top_panel.value()->Render();
        } else {
            top_left = text("No active TUI view. Try: list_nodes, list_elements, list_parts, panel ...") | dim;
        }

        const bool left_focused = focus_region == FocusRegion::TopLeftView;
        const bool right_focused = focus_region == FocusRegion::TopRightOutput;
        const bool command_focused = focus_region == FocusRegion::BottomCommand;
        const std::vector<TuiLogLine> log_lines = tui_log_lines_snapshot();

        Element left_box =
            window(
                text(left_focused ? " TUI View [TAB] " : " TUI View ") | bold,
                top_left
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

        if (focus_region == FocusRegion::TopLeftView &&
            left_view_mode == LeftViewMode::StaticElement &&
            top_panel.has_value()) {
            if (top_panel.value()->OnEvent(event))
                return true;
        }

        if (focus_region == FocusRegion::BottomCommand) {
            const int history_size = static_cast<int>(history.size());
            if (event == Event::ArrowUp) {
                if (history_size <= 0) return true;
                if (history_index == -1) {
                    input_backup = input;
                    history_index = history_size - 1;
                } else {
                    history_index = (std::max)(0, history_index - 1);
                }
                input = history[static_cast<std::size_t>(history_index)];
                return true;
            }
            if (event == Event::ArrowDown) {
                if (history_size <= 0) return true;
                if (history_index == -1) return true;
                if (history_index < history_size - 1) {
                    history_index = history_index + 1;
                    input = history[static_cast<std::size_t>(history_index)];
                } else {
                    input = input_backup;
                    history_index = -1;
                }
                return true;
            }
        }

        if (focus_region != FocusRegion::BottomCommand && event.is_character()) {
            return true;
        }

        if (event == Event::Return) {
            if (focus_region == FocusRegion::TopLeftView) {
                if (left_view_mode == LeftViewMode::NodesList && !node_rows.empty() &&
                    node_selected_row >= 0 && node_selected_row < static_cast<int>(node_rows.size())) {
                    save_view_state("list_nodes");
                    (void)open_panel_in_top_view("node",
                        std::to_string(node_rows[static_cast<std::size_t>(node_selected_row)].nid), false);
                } else if (left_view_mode == LeftViewMode::ElementsList && !elem_rows.empty() &&
                    elem_selected_row >= 0 && elem_selected_row < static_cast<int>(elem_rows.size())) {
                    save_view_state("list_elements");
                    (void)open_panel_in_top_view("elem",
                        std::to_string(elem_rows[static_cast<std::size_t>(elem_selected_row)].eid), false);
                } else if (left_view_mode == LeftViewMode::PartsList && !part_rows.empty() &&
                    part_selected_row >= 0 && part_selected_row < static_cast<int>(part_rows.size())) {
                    save_view_state("list_parts");
                    (void)open_panel_in_top_view("part",
                        part_rows[static_cast<std::size_t>(part_selected_row)].name, false);
                }
                return true;
            }
            if (focus_region != FocusRegion::BottomCommand) {
                return true;
            }
            const std::string cmd = input;
            input.clear();
            if (!cmd.empty()) {
                if (history.empty() || cmd != history.back()) {
                    history.push_back(cmd);
                }
            }
            history_index = -1;

            if (cmd.empty()) return true;

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
                    "  list_parts",
                    "  panel node <nid>",
                    "  panel elem <eid>",
                    "  panel part <name>",
                    "  panel set <name>",
                    "  panel material <id_or_name>",
                    "  panel section <id_or_name>",
                    "  quit / exit",
                    "",
                    "Note: other commands are supported; they will execute and log to the normal logger.",
                };
                ftxui::Component dummy = Container::Vertical({});
                top_panel = Renderer(dummy, [help = std::move(help)]() { return window(text(" Help ") | bold, status_lines_element(help)); });
                left_view_mode = LeftViewMode::StaticElement;
                left_focus = 0.0f;
                focus_region = FocusRegion::TopLeftView;
                return true;
            }

            // list_*: render into top.
            if (cmd == "list_nodes") {
                save_view_state("list_nodes");
                build_nodes_list_view();
                node_selected_row = node_rows.empty() ? -1 : 0;
                sync_nodes_focus();
                focus_region = FocusRegion::TopLeftView;
                return true;
            }

            if (cmd == "list_elements") {
                save_view_state("list_elements");
                build_elements_list_view();
                elem_selected_row = elem_rows.empty() ? -1 : 0;
                sync_elems_focus();
                focus_region = FocusRegion::TopLeftView;
                return true;
            }

            if (cmd == "list_parts") {
                save_view_state("list_parts");
                build_parts_list_view();
                part_selected_row = part_rows.empty() ? -1 : 0;
                sync_parts_focus();
                focus_region = FocusRegion::TopLeftView;
                return true;
            }

            // panel: render into top using existing universal inspector panel element composition.
            if (starts_with(cmd, "panel ")) {
                std::stringstream ss(cmd);
                std::string keyword, type, id_or_name;
                ss >> keyword >> type;
                std::getline(ss >> std::ws, id_or_name);
                if (type.empty() || id_or_name.empty()) {
                    spdlog::error("Usage: panel <type> <id_or_name>  (type: node|elem|element|part|set|material|section)");
                    return true;
                }
                (void)open_panel_in_top_view(type, id_or_name, true);
                return true;
            }

            // Fallback: execute existing command processor (logs go to spdlog sinks).
            view_history.clear();
            clear_left_view();
            process_command(cmd, session);
            return true;
        }

        // Quick escape for clearing a view.
        if (event == Event::Escape) {
            if (!restore_view_state()) {
                clear_left_view();
            }
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

        if (focus_region == FocusRegion::TopLeftView &&
            left_view_mode == LeftViewMode::PartsList &&
            !part_rows.empty()) {
            const int max_idx = static_cast<int>(part_rows.size()) - 1;
            if (event == Event::ArrowUp) {
                part_selected_row = (std::max)(0, part_selected_row - 1);
                sync_parts_focus();
                return true;
            }
            if (event == Event::ArrowDown) {
                part_selected_row = (std::min)(max_idx, part_selected_row + 1);
                sync_parts_focus();
                return true;
            }
            if (event == Event::PageUp) {
                part_selected_row = (std::max)(0, part_selected_row - 10);
                sync_parts_focus();
                return true;
            }
            if (event == Event::PageDown) {
                part_selected_row = (std::min)(max_idx, part_selected_row + 10);
                sync_parts_focus();
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
            if (focus_region == FocusRegion::TopLeftView &&
                left_view_mode == LeftViewMode::PartsList &&
                !part_rows.empty()) {
                const int max_idx = static_cast<int>(part_rows.size()) - 1;
                if (m.button == Mouse::WheelUp) {
                    part_selected_row = (std::max)(0, part_selected_row - 3);
                    sync_parts_focus();
                    return true;
                }
                if (m.button == Mouse::WheelDown) {
                    part_selected_row = (std::min)(max_idx, part_selected_row + 3);
                    sync_parts_focus();
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
