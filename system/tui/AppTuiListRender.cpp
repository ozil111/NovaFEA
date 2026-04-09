/**
 * TUI List Rendering - Renders node/element/part list views as FTXUI Elements.
 */
#include "tui/AppTuiState.h"
#include <ftxui/dom/node.hpp>
#include <iomanip>
#include <sstream>

namespace tui {
using namespace ftxui;

Element render_nodes_list_element(const TuiAppState& state) {
    Elements lines;
    const int total_count = static_cast<int>(state.node_rows.size());
    const int margin = 30;
    const int anchor_idx = state.node_selected_row >= 0 ? state.node_selected_row : 0;
    const int start_idx = (std::max)(0, anchor_idx - margin);
    const int end_idx = (std::min)(total_count, anchor_idx + margin + 1);
    Element header_row = hbox({
        text(" NodeID ") | bold, text(" | "),
        text(" X ") | bold,     text(" | "),
        text(" Y ") | bold,     text(" | "),
        text(" Z ") | bold,
    }) | color(Color::Cyan);

    for (int i = start_idx; i < end_idx; ++i) {
        const auto& r = state.node_rows[static_cast<std::size_t>(i)];
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
        if (i == state.node_selected_row) {
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
}

Element render_elements_list_element(const TuiAppState& state) {
    using namespace ftxui;
    Elements lines;
    const int total_count = static_cast<int>(state.elem_rows.size());
    const int margin = 30;
    const int anchor_idx = state.elem_selected_row >= 0 ? state.elem_selected_row : 0;
    const int start_idx = (std::max)(0, anchor_idx - margin);
    const int end_idx = (std::min)(total_count, anchor_idx + margin + 1);
    Element header_row = hbox({
        text(" ElementID ") | bold, text(" | "),
        text(" TypeID ") | bold,    text(" | "),
        text(" Nodes ") | bold,
    }) | color(Color::YellowLight);

    for (int i = start_idx; i < end_idx; ++i) {
        const auto& r = state.elem_rows[static_cast<std::size_t>(i)];
        Element row = hbox({
            text(" " + std::to_string(r.eid) + " ") | color(Color::Cyan),
            text(" | "),
            text(" " + std::to_string(r.type_id) + " ") | color(Color::YellowLight),
            text(" | "),
            text(" " + r.nodes + " " ),
        });
        if (i == state.elem_selected_row)
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
}

Element render_parts_list_element(const TuiAppState& state) {
    Elements lines;
    const int total_count = static_cast<int>(state.part_rows.size());
    const int margin = 30;
    const int anchor_idx = state.part_selected_row >= 0 ? state.part_selected_row : 0;
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
        const auto& r = state.part_rows[static_cast<std::size_t>(i)];
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
        if (i == state.part_selected_row)
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
}

Element render_sets_list_element(const TuiAppState& state) {
    Elements lines;
    const int total_count = static_cast<int>(state.set_rows.size());
    const int margin = 30;
    const int anchor_idx = state.set_selected_row >= 0 ? state.set_selected_row : 0;
    const int start_idx = (std::max)(0, anchor_idx - margin);
    const int end_idx = (std::min)(total_count, anchor_idx + margin + 1);
    Element header_row = hbox({
        text(" Set Name ") | bold, text(" | "),
        text(" Type ") | bold, text(" | "),
        text(" Members ") | bold,
    }) | color(Color::Cyan);

    for (int i = start_idx; i < end_idx; ++i) {
        const auto& r = state.set_rows[static_cast<std::size_t>(i)];
        const bool is_node = (r.type == "node");
        Element row = hbox({
            text(" " + r.name + " ") | color(Color::Cyan),
            text(" | "),
            text(" " + r.type + " ") | (is_node ? color(Color::GreenLight) : color(Color::YellowLight)),
            text(" | "),
            text(" " + std::to_string(r.member_count) + " "),
        });
        if (i == state.set_selected_row)
            row = row | inverted | focus;
        lines.push_back(std::move(row));
    }

    return vbox({
        hbox({
            text(" NovaFEA ") | bgcolor(Color::Blue) | color(Color::White) | bold,
            text(" Sets ") | color(Color::Cyan),
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
}

} // namespace tui
