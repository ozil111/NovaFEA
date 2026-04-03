/**
 * TUI Universal Inspection Panel - Registry implementation.
 */
#include "tui/ComponentTUI.h"
#include "components/mesh_components.h"
#include "components/load_components.h"
#include "components/material_components.h"
#include "components/simdroid_components.h"
#include "components/property_components.h"
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace tui {

namespace {

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
            
            // Improved section display logic
            if (reg.valid(p.section)) {
                std::string section_info;
                if (reg.all_of<::Component::SetName>(p.section)) {
                    section_info = reg.get<::Component::SetName>(p.section).value;
                } else if (reg.all_of<::Component::PropertyID>(p.section)) {
                    section_info = "ID:" + std::to_string(reg.get<::Component::PropertyID>(p.section).value);
                } else if (reg.all_of<::Component::Formulation>(p.section)) {
                    section_info = "[" + reg.get<::Component::Formulation>(p.section).value + "]";
                } else {
                    section_info = "(entity)";
                }
                lines.push_back(text("  section = " + section_info));
            }
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

} // namespace tui
