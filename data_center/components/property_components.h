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
#include <entt/entt.hpp>

/**
 * @namespace Component
 * @brief ECS components - Property (section properties) section
 * @details Property is decoupled from Material, storing only section and integration related parameters (integration scheme, hourglass control, etc.).
 * Material is bound through SimdroidPart, see simdroid_components.h.
 */
namespace Component {

    /**
     * @brief [New] Attached to Property entity, stores its user-defined ID (pid)
     * @details Used to identify Property entity, avoiding ID conflicts with other types of entities
     */
    struct PropertyID {
        int value;
    };

    /**
     * @brief [New] Attached to Property entity, stores section/integration properties for solid elements
     * @details Corresponds to "property" object in JSON, stores only parameters related to section and integration
     */
    struct SolidProperty {
        int type_id;                    // From JSON "typeid"
        int integration_network;        // Integration network parameter, e.g. "integration_network": 2
        std::string hourglass_control;  // Hourglass control method, e.g. "hourglass_control": "eas"
    };

    // ------------------------------------------------------------------
    //  Common small components (reusable by multiple Property types)
    // ------------------------------------------------------------------

    // Element formulation/recipe (e.g. "Shell4", "Hex8R", etc.)
    struct Type{
        std::string value;
    };
    struct Formulation {
        std::string value;
    };

    // Small strain options ("Auto", "T0", "Tnone", etc.)
    struct SmallStrain {
        std::string value;
    };

    // Bulk/hourglass viscosity parameters (qa, qb)
    struct ViscosityParams {
        double quadratic = 0.1; // qa
        double linear    = 0.05; // qb
    };

    // General local coordinate system marker
    struct CoordSys {
        std::string value;
    };

    // ------------------------------------------------------------------
    //  Typical section property components (mapped to Simdroid CrossSection / Radioss PROP)
    // ------------------------------------------------------------------

    // Type == Truss (truss)
    struct TrussProperty {
        double area = 0.0; // Area
    };

    // Type == Shell / SandwichShell
    struct ShellProperty {
        int   type_id = 0;                         // Spare: Shell / Sandwich / Sh3n ...
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

    // Type == Solid / SolidOrthotropic (lightweight extension based on original SolidProperty)
    struct SolidAdvancedProperty {
        // Directly reuse/supplement SolidProperty's advanced options, maintaining decoupling:
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

    // Type == SolidShComp (composite SolidShell, PROP TYPE22)
    struct SolidShCompProperty {
        Formulation formulation;              // TShell / TShellRPH
        SmallStrain small_strain;            // Ismstr
        std::array<int, 3> integration_points{}; // Inpts
        double numeric_damping = 0.0;        // dn
        ViscosityParams bulk_viscosity;      // qa / qb
        double thickness_penalty = 10.0;     // ThicknessPenaltyFactor
        CoordSys coord_sys;                  // skew_ID

        // Composite layer data
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
        std::vector<double> dj;          // Dj[...] section dimensions D*
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

        // Curve references (resolved via DofMap's curve_name_to_entity)
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

        // 6-direction curve references
        std::array<entt::entity, 6> nonlinear_stiffness{};      // f(δ)
        std::array<entt::entity, 6> for_or_mom_with_vel{};      // g(δ)
        std::array<entt::entity, 6> harden_related_curve{};     // according to HardenFlag
        std::array<entt::entity, 6> nonlinear_damping{};        // h(δ)

        std::array<double, 6> upper_failure_limit{}; // δmax / θmax
        std::array<double, 6> lower_failure_limit{}; // δmin / θmin

        std::array<double, 6> absc_scale_damp{};     // F
        std::array<double, 6> ordina_scale_damp{};   // H
        std::array<double, 6> absc_scale_stiff{};    // Ascale
        std::array<double, 6> ordina_scale_stiff{};  // (unnamed coefficient)

        double ref_tran_vel = 0.0; // RefTranVel
        double ref_rot_vel  = 0.0; // RefRotVel

        bool smooth_strain_rate = false; // SmoothStrRate
        std::array<double, 6> rela_vec_coeff{};      // RelaVecCoeff
        std::array<double, 6> rela_vec_exp{};        // RelaVecExp
        std::array<double, 6> failure_scale{};       // FailureScale
        std::array<double, 6> failure_exp{};         // FailureExp
    };

} // namespace Component

