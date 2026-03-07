/**
 * TUI Universal Inspection Panel - Component metadata registry and render pipeline.
 * Phase 1: Metadata registry; Phase 2: Render engine (header, dispatch, output).
 */
#pragma once

#include "entt/entt.hpp"
#include <functional>
#include <string>
#include <vector>
#include <ostream>

struct SimdroidInspector;

namespace tui {

/** ANSI color helpers for terminal output */
namespace color {
    constexpr const char* title   = "\033[1;36m";
    constexpr const char* reset   = "\033[0m";
    constexpr const char* green   = "\033[1;32m";
    constexpr const char* yellow  = "\033[1;33m";
    constexpr const char* dim     = "\033[2m";
}

/** One registered component: display name + type-erased has/print. */
struct ComponentTUIEntry {
    std::string display_name;
    std::function<bool(entt::registry&, entt::entity)> has_component;
    std::function<void(entt::registry&, entt::entity, SimdroidInspector*, std::ostream&)> print;
};

/** Registry of component types to TUI printers. */
class ComponentTUIRegistry {
public:
    using PrinterFn = std::function<void(entt::registry&, entt::entity, SimdroidInspector*, std::ostream&)>;

    /** Register a component type T with a display name and printer. */
    template<typename T>
    void register_component(const std::string& display_name, PrinterFn printer) {
        entries_.push_back({
            display_name,
            [](entt::registry& r, entt::entity e) { return r.all_of<T>(e); },
            std::move(printer)
        });
    }

    /** For a given entity, call every applicable printer to out. */
    void for_each_applicable(entt::registry& reg, entt::entity e, SimdroidInspector* insp, std::ostream& out) const;

    /** Access registered entries (e.g. for ordered iteration). */
    const std::vector<ComponentTUIEntry>& entries() const { return entries_; }

    /** Singleton access for registration and render. */
    static ComponentTUIRegistry& instance();

private:
    std::vector<ComponentTUIEntry> entries_;
};

/** Entity kind for header display */
enum class PanelEntityKind { Node, Element, Part, Set, Unknown };

/** Resolve panel type + id/name to entity. Returns entt::null if not found. */
entt::entity resolve_panel_entity(entt::registry& reg, SimdroidInspector* insp,
    const std::string& type, const std::string& id_or_name, PanelEntityKind* out_kind, std::string* out_display_id);

/** Full TUI render: header + all registered component blocks + optional footer hints. */
void render_panel(entt::registry& reg, entt::entity e, SimdroidInspector* insp,
    PanelEntityKind kind, const std::string& display_id, std::ostream& out);

/** Append force-path insight for a node (PartGraph edges, load/constraint highlight). Call after component blocks for Node. */
void append_force_path_info(entt::registry& reg, entt::entity node_entity, SimdroidInspector* insp,
    std::ostream& out);

} // namespace tui
