/**
 * TUI App State - Shared state, enumerations, and function declarations
 * for the interactive application TUI (AppTUI).
 */
#pragma once

#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>
#include <spdlog/common.h>
#include <optional>
#include <string>
#include <vector>

class AppSession;

namespace tui {

// ── Enumerations ──────────────────────────────────────────────────────

enum class FocusRegion {
    TopLeftView = 0,
    TopRightOutput = 1,
    BottomCommand = 2,
};

enum class LeftViewMode {
    None,
    StaticElement,
    NodesList,
    ElementsList,
    PartsList,
    SetList
};

// ── Row data structures for list views ────────────────────────────────

struct NodeListRow {
    int nid;
    double x, y, z;
};

struct ElemListRow {
    int eid;
    int type_id;
    std::string nodes;
};

struct PartListRow {
    std::string name;
    std::string element_set;
    std::string material_id;
    std::string section_id;
    std::size_t elem_count;
};

// ── View history ──────────────────────────────────────────────────────

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

// ── Log line ──────────────────────────────────────────────────────────

struct TuiLogLine {
    spdlog::level::level_enum level;
    std::string text;
};

// ── Shared TUI application state ──────────────────────────────────────

struct TuiAppState {
    // List data
    std::vector<NodeListRow> node_rows;
    int node_selected_row = -1;
    std::vector<ElemListRow> elem_rows;
    int elem_selected_row = -1;
    std::vector<PartListRow> part_rows;
    int part_selected_row = -1;

    // Panel state
    std::optional<ftxui::Component> top_panel;
    LeftViewMode left_view_mode = LeftViewMode::None;
    std::string current_panel_type;
    std::string current_panel_id;

    // Focus / scroll
    float left_focus = 0.0f;
    float right_focus = 0.0f;
    FocusRegion focus_region = FocusRegion::BottomCommand;

    // View history (back-navigation)
    std::vector<ViewState> view_history;

    // Input history (command-line up/down)
    std::vector<std::string> input_history;
    int history_index = -1;
    std::string input_backup;
};

// ── Utility functions (TuiLogSink.cpp) ────────────────────────────────

float clamp01(float v);
bool starts_with(const std::string& s, const std::string& prefix);

ftxui::Decorator level_decorator(spdlog::level::level_enum lvl);
ftxui::Element status_lines_element(const std::vector<std::string>& lines);
ftxui::Element status_lines_element(const std::vector<TuiLogLine>& lines);

std::vector<TuiLogLine> tui_log_lines_snapshot();
void install_tui_log_sink();

// ── Panel / view management (AppTuiPanels.cpp) ────────────────────────

void save_view_state(TuiAppState& state, std::string label);
void clear_left_view(TuiAppState& state);
bool open_panel_in_top_view(AppSession& session, TuiAppState& state,
                            const std::string& type, const std::string& id_or_name,
                            bool push_history);

void build_nodes_list_view(AppSession& session, TuiAppState& state);
void build_elements_list_view(AppSession& session, TuiAppState& state);
void build_parts_list_view(AppSession& session, TuiAppState& state);

bool restore_view_state(AppSession& session, TuiAppState& state);

void sync_nodes_focus(TuiAppState& state);
void sync_elems_focus(TuiAppState& state);
void sync_parts_focus(TuiAppState& state);

// ── List rendering (AppTuiListRender.cpp) ─────────────────────────────

ftxui::Element render_nodes_list_element(const TuiAppState& state);
ftxui::Element render_elements_list_element(const TuiAppState& state);
ftxui::Element render_parts_list_element(const TuiAppState& state);

} // namespace tui
