#pragma once
#include "entt/entt.hpp"
#include <string>
#include <vector>

namespace Component {

    // 1. Simdroid 特有的 Part 定义（绑定器：几何 + 截面 + 材料）
    // Python 中的 Part 在这里是一个 Entity，连接 ElementSet、Property、Material
    struct SimdroidPart {
        std::string name;
        entt::entity element_set; // 几何范围（单元集）
        entt::entity material;    // 指向 Material 实体
        entt::entity section;     // 指向 Property（截面属性）实体
    };

    // ===================================================================
    // 2. 接触定义 (Contact) — 拆分为多个细粒度 ECS 组件
    // ===================================================================

    // --- 枚举 ---

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

    // --- 核心组件 ---

    struct ContactBase {
        std::string name;
        entt::entity master_entity = entt::null;
        entt::entity slave_entity  = entt::null;
    };

    struct ContactTypeTag {
        ContactInterType type = ContactInterType::Unknown;
    };

    // --- 特化参数组件 ---

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

    // 3. 刚体/MPC 定义 (对传力路径至关重要)
    struct RigidBodyConstraint {
        entt::entity master_node_set; // 或者 master_node entity
        entt::entity slave_node_set;
    };

    // 3b. Radioss /RBODY 刚体定义
    enum class InertiaMode { Automatic = 2, Input = 3 };

    struct RigidBody {
        entt::entity master_node = entt::null;      // 主节点实体 (node_ID)
        entt::entity slave_node_set = entt::null;    // 从节点集实体 (grnd_ID)

        std::string coord_sys;                       // 局部坐标系 ID/名称 (Skew_ID)
        InertiaMode inertia_cal = InertiaMode::Automatic; // Ispher
        
        double mass = 0.0;
        int cog_mode = 1;                            // ICoG: 1/2/3/4
        double inertia_tensor[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // I11, I22, I33, I12, I23, I13
    };
    
    // 4. 分析结果组件 (用于存储传力路径分析结果，避免重复计算)
    struct ForcePathNode {
        double weight;
        bool is_load_point;
        bool is_constraint_point;
    };

    // 5. [新增] 刚性墙定义
    struct RigidWall {
        int id;
        std::string type; // "Planar", "Cylindrical", "Spherical"
        std::vector<double> parameters; // 平面方程 ax+by+cz+d=0 或圆柱参数等
        entt::entity secondary_node_set; // 关联的从节点集（可选）
    };
}