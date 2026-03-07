/**
 * TUI Universal Inspection Panel - Component metadata registry and render pipeline.
 * Uses FTXUI for componentized rendering (Flexbox-style layout).
 */
#pragma once

#include "entt/entt.hpp"
#include <ftxui/dom/elements.hpp>
#include <functional>
#include <string>
#include <vector>

struct SimdroidInspector;

namespace tui {

using namespace ftxui;

/** Component render function: returns an FTXUI Element instead of writing to ostream. */
using PrinterFn = std::function<Element(entt::registry&, entt::entity, SimdroidInspector*)>;

/** One registered component: display name + type-erased has/render. */
struct ComponentTUIEntry {
    std::string display_name;
    std::function<bool(entt::registry&, entt::entity)> has_component;
    PrinterFn render;
};

/** Registry of component types to TUI renderers. */
class ComponentTUIRegistry {
public:
    /** Register a component type T with a display name and renderer. */
    template<typename T>
    void register_component(const std::string& display_name, PrinterFn renderer) {
        entries_.push_back({
            display_name,
            [](entt::registry& r, entt::entity e) { return r.all_of<T>(e); },
            std::move(renderer)
        });
    }

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

/** Full TUI render: header + all registered component blocks + footer. Renders via FTXUI to stdout. */
void render_panel(entt::registry& reg, entt::entity e, SimdroidInspector* insp,
    PanelEntityKind kind, const std::string& display_id);

/** Force-path insight element for a node (PartGraph edges, load/constraint). Returns empty element if none. */
Element force_path_element(entt::registry& reg, entt::entity node_entity, SimdroidInspector* insp);

} // namespace tui
