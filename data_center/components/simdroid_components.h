#pragma once
#include "entt/entt.hpp"
#include <string>
#include <vector>

namespace Component {

    // 1. Simdroid-specific Part definition (binder: geometry + section + material)
    // In Python, Part here is an Entity, connecting ElementSet, Property, Material
    struct SimdroidPart {
        std::string name;
        entt::entity element_set; // Geometric scope (element set)
        entt::entity material;    // Points to Material entity
        entt::entity section;     // Points to Property (section properties) entity
    };

    // ===================================================================
    // 2. Contact definition — split into multiple fine-grained ECS components
    // ===================================================================

    // --- Enums ---

    // /INTER/TYPE2 (Tie), /INTER/TYPE7 (N-S), /INTER/TYPE24 (General)
    enum class ContactInterType { Tie, NodeToSurface, General, Unknown };

    enum class ContactFormulationType {
        Standard, Optimized,
        FailShBrick, FailSh, FailBrick,
        Penalty, StandSwitchPenal, OptimSwitchPenal,
        Unknown
    };

    enum class InterfaceStiffnessType { Default, Main, Maximum, Minimum };

    enum class SearchMethodType { Box, Segment };

    enum class FrictionLawType { Coulomb, Viscous, Darmstad, Renard, ExpDecay };

    enum class GapFlagType { None, Variable, VariableScale, VariableScalePen, Gapmin };

    enum class IgnoreType { Length, MainThick, MainThickZero };

    enum class FailModeType { None, Max, Quad };

    enum class RuptureFlagType { TractionCompress, Traction };

    // --- Core components ---

    struct ContactBase {
        std::string name;
        entt::entity master_entity = entt::null;
        entt::entity slave_entity  = entt::null;
    };

    struct ContactTypeTag {
        ContactInterType type = ContactInterType::Unknown;
    };

    // --- Specialized parameter components ---

    struct ContactFormulation {
        ContactFormulationType formulation = ContactFormulationType::Standard;
        double stiffness_factor    = 1.0;
        double damping_coefficient = 0.0;
        InterfaceStiffnessType interface_stiffness = InterfaceStiffnessType::Default;
        int    stiffness_form      = 1000;   // Istf (TYPE7/TYPE24), raw int for wide range
        double min_stiffness       = 0.0;
        double max_stiffness       = 1e30;
        SearchMethodType search_method = SearchMethodType::Segment;
        double search_distance     = 0.0;
    };

    struct ContactFriction {
        FrictionLawType friction_law = FrictionLawType::Coulomb;
        double friction_coef = 0.0;
        double friction_coefs[6] = {};
        double damping_inter_stiff = 0.0;  // VISs (TYPE7)
        double damping_inter_fric  = 0.0;  // VISf (TYPE7)
    };

    struct ContactGapControl {
        GapFlagType gap_flag = GapFlagType::None;
        double gap_scale     = 1.0;
        double gap_min       = 0.0;
        double gap_max       = 0.0;
        double slave_gap_max  = 0.0;   // TYPE24
        double master_gap_max = 0.0;   // TYPE24
    };

    struct ContactTieData {
        IgnoreType ignore = IgnoreType::MainThick;
        FailModeType fail_mode = FailModeType::None;
        entt::entity stress_vs_stress_rate = entt::null;  // fac_IDsr curve
        entt::entity nor_stress_vs_disp    = entt::null;  // fac_IDsn curve
        entt::entity tang_stress_vs_disp   = entt::null;  // fac_IDst curve
        RuptureFlagType rupture_flag = RuptureFlagType::TractionCompress;
        double max_n_dist = 0.0;
        double max_t_dist = 0.0;
    };

    // 3. Rigid body/MPC definition (crucial for force transmission paths)
    struct RigidBodyConstraint {
        entt::entity master_node_set; // or master_node entity
        entt::entity slave_node_set;
    };

    // 3b. Radioss /RBODY rigid body definition
    enum class InertiaMode { Automatic = 2, Input = 3 };

    struct RigidBody {
        entt::entity master_node = entt::null;      // Master node entity (node_ID)
        entt::entity slave_node_set = entt::null;    // Slave node set entity (grnd_ID)

        std::string coord_sys;                       // Local coordinate system ID/name (Skew_ID)
        InertiaMode inertia_cal = InertiaMode::Automatic; // Ispher
        
        double mass = 0.0;
        int cog_mode = 1;                            // ICoG: 1/2/3/4
        double inertia_tensor[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // I11, I22, I33, I12, I23, I13
    };
    
    // 4. Analysis result components (for storing force path analysis results, avoiding redundant computation)
    struct ForcePathNode {
        double weight;
        bool is_load_point;
        bool is_constraint_point;
    };

    // 5. [New] Rigid wall definition
    struct RigidWall {
        int id;
        std::string type; // "Planar", "Cylindrical", "Spherical"
        std::vector<double> parameters; // Plane equation ax+by+cz+d=0 or cylinder parameters, etc.
        entt::entity secondary_node_set; // Associated slave node set (optional)
    };
}