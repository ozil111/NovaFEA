/**
 * TUI Universal Inspection Panel - List views for nodes and elements.
 */
#include "tui/ComponentTUI.h"
#include "components/mesh_components.h"
#include "components/simdroid_components.h"
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <vector>

namespace tui {

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

void render_parts_list(entt::registry& reg) {
    struct Row {
        std::string name;
        std::string material;
        std::size_t elem_count;
    };
    std::vector<Row> rows;
    auto view = reg.view<const ::Component::SimdroidPart>();
    rows.reserve(static_cast<std::size_t>(view.size()));
    for (auto e : view) {
        const auto& part = view.get<const ::Component::SimdroidPart>(e);
        std::size_t count = 0;
        if (reg.valid(part.element_set) && reg.all_of<::Component::ElementSetMembers>(part.element_set)) {
            count = reg.get<::Component::ElementSetMembers>(part.element_set).members.size();
        }
        std::string mat_name = "-";
        if (reg.valid(part.material) && reg.all_of<::Component::SetName>(part.material)) {
            mat_name = reg.get<::Component::SetName>(part.material).value;
        }
        rows.push_back(Row{ part.name, std::move(mat_name), count });
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.name < b.name; });

    int selected_row = rows.empty() ? -1 : 0;

    auto screen = ScreenInteractive::Fullscreen();

    ftxui::Component ui = ftxui::Renderer([&] {
        Elements lines;
        lines.push_back(
            hbox({
                text(" Part ") | bold,
                text(" | "),
                text(" Material ") | bold,
                text(" | "),
                text(" Elements ") | bold,
            }) | border);

        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];
            Element row = hbox({
                text(" " + r.name + " ") | color(Color::Cyan),
                text(" | "),
                text(" " + r.material + " ") | color(Color::YellowLight),
                text(" | "),
                text(" " + std::to_string(r.elem_count) + " "),
            }) | border;

            if (static_cast<int>(i) == selected_row)
                row = row | inverted;
            lines.push_back(std::move(row));
        }

        Element header = hbox({
            text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
            text(" Parts ") | color(Color::Cyan),
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

} // namespace tui
