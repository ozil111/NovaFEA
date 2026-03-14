// data_center/components/property_components.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include <string>
#include <vector>
#include <array>
#include "entt/entt.hpp"

/**
 * @namespace Component
 * @brief ECS组件 - Property（截�?属性）部分
 * @details Property �?Material 解耦，仅存储截面与积分相关参数（积分方案、沙漏控制等）�?
 * 材料通过 SimdroidPart 绑定，见 simdroid_components.h�?
 */
namespace Component {

    /**
     * @brief [新] 附加�?Property 实体，存储其用户定义的ID (pid)
     * @details 用于标识Property实体，避免与其他类型实体的ID冲突
     */
    struct PropertyID {
        int value;
    };

    /**
     * @brief [新] 附加�?Property 实体，存储固体单元的截面/积分属�?
     * @details 对应 JSON 中的 "property" 对象，仅存放与截面、积分相关的参数
     */
    struct SolidProperty {
        int type_id;                    // 来自 JSON �?"typeid"
        int integration_network;        // 积分网络参数，如 "integration_network": 2
        std::string hourglass_control;  // 沙漏控制方法，如 "hourglass_control": "eas"
    };

    // ------------------------------------------------------------------
    //  通用小组件（可被多种 Property 复用�?
    // ------------------------------------------------------------------

    // 单元算法/配方（例�?"Shell4", "Hex8R" 等）
    struct Formulation {
        std::string value;
    };

    // 小应变选项 ("Auto", "T0", "Tnone" �?
    struct SmallStrain {
        std::string value;
    };

    // 体积/沙漏粘性参�?(qa, qb)
    struct ViscosityParams {
        double quadratic = 0.1; // qa
        double linear    = 0.05; // qb
    };

    // 通用局部坐标系标记
    struct CoordSys {
        std::string value;
    };

    // ------------------------------------------------------------------
    //  典型截面属性组件（映射�?Simdroid CrossSection / Radioss PROP�?
    // ------------------------------------------------------------------

    // Type == Truss (桁架)
    struct TrussProperty {
        double area = 0.0; // Area
    };

    // Type == Shell / SandwichShell
    struct ShellProperty {
        int   type_id = 0;                         // 备用：Shell / Sandwich / Sh3n ...
        std::array<double, 4> thickness{};        // Thick[4]
        bool  thickness_change = false;           // Ithick
        bool  drill_dof        = false;           // Idrill
        double shear_factor    = 0.83;            // Ashear
        int   integration_points = 2;             // InpNum / N
        std::string inp_rule;                     // InpRule (e.g. "Gauss")
        double fail_thick = 1.0;                  // P_thickfail
        std::array<double, 3> hourglass_coefs{};  // hm, hf, hr
        std::string plastic_plane_stress_return;  // "Default" / "Iteration" / "Newton"
        std::string mid_shell_flag;               // "NoOffset"/"Upper"/"Lower"
    };

    // Type == Solid / SolidOrthotropic（在原有 SolidProperty 基础上做轻量扩展�?
    struct SolidAdvancedProperty {
        // 直接复用/补充 SolidProperty 的高阶选项，保持解耦：
        std::string formulation;      // Hex8R / Tet4Q / ...
        std::string small_strain;     // Ismstr
        std::string const_pressure;   // Icpre
        std::string co_rotation_flag; // Iframe
        double visco_hourglass_k = 0.1; // h
        ViscosityParams bulk_viscosity;  // qa / qb
        double dtmin = 0.0;              // Δtmin
        double numeric_damping = 0.0;    // dn
        bool distortion_control = false; // DistortionControl
        std::array<double, 3> distortion_coeffs{}; // DistortionControlCoeffs[3]
        double disp_hourglass_factor = 1.0;        // DispHourglassFactor
        std::string hourglass_type;                // "Stiffness" / "RelaxStiffness"
        bool ele_charac_length = false;            // EleCharacLength
    };

    // Type == SolidShell (PROP TYPE20)
    struct SolidShellProperty {
        Formulation formulation;              // TShell / TShellRPH
        SmallStrain small_strain;            // Ismstr
        std::array<int, 3> integration_points{}; // Inpts (r,s,t)
        double visco_hourglass_k = 0.0;      // h
        ViscosityParams bulk_viscosity;      // qa / qb
        double dtmin = 0.0;                  // Δtmin
        double thickness_penalty = 10.0;     // ThicknessPenaltyFactor
        std::array<double, 3> distortion_coeffs{}; // DistortionControlCoeffs
    };

    // Type == SolidShComp (复合 SolidShell, PROP TYPE22)
    struct SolidShCompProperty {
        Formulation formulation;              // TShell / TShellRPH
        SmallStrain small_strain;            // Ismstr
        std::array<int, 3> integration_points{}; // Inpts
        double numeric_damping = 0.0;        // dn
        ViscosityParams bulk_viscosity;      // qa / qb
        double thickness_penalty = 10.0;     // ThicknessPenaltyFactor
        CoordSys coord_sys;                  // skew_ID

        // 复合层数�?
        std::vector<double> layer_angles;    // Angles[]
        std::vector<double> layer_thicks;    // Thicks[]
        std::vector<double> layer_positions; // Positions[]
        std::vector<std::string> layer_materials; // Materials[]
        std::vector<std::string> position_flags;  // Ipos[]
    };

    // Type == GeneralBeam (TYPE3)
    struct BeamProperty {
        SmallStrain small_strain;  // Ismstr
        double area = 0.0;         // Area
        double ixx  = 0.0;         // IXX
        double iyy  = 0.0;         // IYY
        double izz  = 0.0;         // IZZ
        bool shear_flag = true;    // Ishear (true/false)
    };

    // Type == FiberBeam (TYPE18)
    struct FiberBeamProperty {
        std::string pattern;             // Pattern
        SmallStrain small_strain;        // Ismstr
        int integration_points = 0;      // InpNum
        std::vector<double> yi;          // Yi[]
        std::vector<double> zi;          // Zi[]
        std::vector<double> areai;       // Areai[]
        std::vector<double> dj;          // Dj[...] 截面尺寸 D*
    };

    // Type == Cohesive (TYPE43)
    struct CohesiveProperty {
        SmallStrain small_strain; // Ismstr
        double thickness = 0.0;   // True_thickness
    };

    // Type == AxialSpringDamper (TYPE4)
    struct AxialSpringDamperProperty {
        double mass       = 0.0;  // Mass
        double stiffness  = 0.0;  // K1
        double damping    = 0.0;  // C1 (DampingCoefficient)

        bool nonlinear_spring = false;   // NonlinearSpring
        bool nonlinear_damper = false;   // NonlinearDamper
        std::string hardening_flag;      // HardeningFlag

        // 曲线引用（通过 DofMap �?curve_name_to_entity 解析�?
        entt::entity stiffness_curve = entt::null; // Load_DeflectionCurve
        entt::entity damping_curve   = entt::null; // DampingCurve
    };

    // Type == BeamSpring (TYPE13)
    struct BeamSpringProperty {
        double mass    = 0.0;  // Mass
        double inertia = 0.0;  // Inertia
        CoordSys coord_sys;    // skew_ID

        std::string failure_criteria; // FailureCriteria
        std::string length_flag;      // LengthFlag
        std::string failure_model;    // FailureModel

        std::array<double, 6> linear_stiffness{};   // K
        std::array<double, 6> linear_damping{};     // C
        std::array<double, 6> non_stiff_fac_a{};    // A
        std::array<double, 6> non_stiff_fac_b{};    // B
        std::array<double, 6> non_stiff_fac_d{};    // D

        std::array<std::string, 6> hardening_flag{}; // HardenFlag[6]

        // 6 向曲线引�?
        std::array<entt::entity, 6> nonlinear_stiffness{};      // f(δ)
        std::array<entt::entity, 6> for_or_mom_with_vel{};      // g(δ)
        std::array<entt::entity, 6> harden_related_curve{};     // according to HardenFlag
        std::array<entt::entity, 6> nonlinear_damping{};        // h(δ)

        std::array<double, 6> upper_failure_limit{}; // δmax / θmax
        std::array<double, 6> lower_failure_limit{}; // δmin / θmin

        std::array<double, 6> absc_scale_damp{};     // F
        std::array<double, 6> ordina_scale_damp{};   // H
        std::array<double, 6> absc_scale_stiff{};    // Ascale
        std::array<double, 6> ordina_scale_stiff{};  // （无名系数）

        double ref_tran_vel = 0.0; // RefTranVel
        double ref_rot_vel  = 0.0; // RefRotVel

        bool smooth_strain_rate = false; // SmoothStrRate
        std::array<double, 6> rela_vec_coeff{};      // RelaVecCoeff
        std::array<double, 6> rela_vec_exp{};        // RelaVecExp
        std::array<double, 6> failure_scale{};       // FailureScale
        std::array<double, 6> failure_exp{};         // FailureExp
    };

} // namespace Component

