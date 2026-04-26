// data_center/components/material_component.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include <vector>
#include <string>
#include <entt/entt.hpp>

/**
 * @namespace Component
 * @brief Contains all ECS components - material section
 */
namespace Component {

    /**
     * @brief [General] Identifies this as a material entity
     * @details Attached to all material entities, stores its user-defined ID (mid).
     * This is the "primary key" connecting elements and materials.
     */
    struct MaterialID {
        int value;
    };

    /**
     * @brief [Type 1] Isotropic linear elastic parameters
     * @details Corresponds to typeid = 1
     */
    struct LinearElasticParams {
        double rho; // Density
        double E;   // Elastic modulus
        double nu;  // Poisson's ratio
    };

    /**
     * @brief [General] Material type string (from Simdroid MaterialType/Type)
     * @details For example: "IsotropicElastic"(LAW1), "IsotropicPlasticJC"(LAW2), "RateDependentPlastic"(LAW36)
     */
    struct MaterialModel {
        std::string value;
    };

    /**
     * @brief [Derived data] Linear elastic constitutive matrix (D Matrix)
     * @details This is a runtime-generated component, attached to Material entity.
     * For 3D isotropic materials, this is a 6x6 matrix.
     * Uses Abaqus/Fortran order: xx, yy, zz, xy, yz, xz
     */
    struct LinearElasticMatrix {
        // Row-major 6x6 matrix stored as flat array (Voigt notation: xx, yy, zz, xy, yz, xz - Abaqus/Fortran order)
        // Access: D[row * 6 + col]
        double D[36] = {};
        
        // Flag whether initialized, prevent redundant calculation
        bool is_initialized = false;
    };

    /**
     * @brief [General-Hyperelastic] General data for hyperelastic materials
     * @details Stores mode=0 (direct input) or mode=1 (curve fitting)
     * This component can be attached to all hyperelastic material entities.
     */
    struct HyperelasticMode {
        int order;
        bool fit_from_data; // Corresponds to mode 0 (false) or 1 (true)
        
        // Only meaningful when fit_from_data == true
        std::vector<entt::entity> uniaxial_funcs;
        std::vector<entt::entity> biaxial_funcs;
        std::vector<entt::entity> planar_funcs;
        std::vector<entt::entity> volumetric_funcs;
        double nu; // Poisson's ratio, only used for curve fitting
    };

    /**
     * @brief [Type 101] Polynomial (N=order)
     * @details Only used when HyperelasticMode::fit_from_data == false
     */
    struct PolynomialParams {
        // Cij, storage order [C10, C01, C20, C02, ..., CN0, C0N]
        std::vector<double> c_ij; 
        // Di, storage order [D1, D2, ..., DN]
        std::vector<double> d_i;  
    };

    /**
     * @brief [Type 102] Reduced Polynomial (N=order)
     * @details Only used when HyperelasticMode::fit_from_data == false
     */
    struct ReducedPolynomialParams {
        // Ci0, storage order [C10, C20, ..., CN0]
        std::vector<double> c_i0; 
        // Di, storage order [D1, D2, ..., DN]
        std::vector<double> d_i;  
    };

    /**
     * @brief [Type 103] Ogden (N=order)
     * @details Only used when HyperelasticMode::fit_from_data == false
     */
    struct OgdenParams {
        // Storage order [mu1, mu2, ..., muN]
        std::vector<double> mu_i;
        // Storage order [alpha1, alpha2, ..., alphaN]
        std::vector<double> alpha_i;
        // Storage order [D1, D2, ..., DN]
        std::vector<double> d_i;
    };

    /**
     * @brief [LAW2] IsotropicPlasticJC (Johnson-Cook) parameters
     * @details Corresponds to MaterialType == "IsotropicPlasticJC"
     * Field names aligned with docs/simdroid_ex_material.md MaterialConstants as much as possible.
     */
    struct IsotropicPlasticParams {
        // A, B, n, C
        double yield_stress_A = 0.0;     // YieldStress
        double hardening_coef_B = 0.0;   // HardeningCoefB
        double hardening_exp_n = 0.0;    // HardeningExpN
        double rate_coef_C = 0.0;        // RateCoef

        // Hardening / temperature
        double hardening_mode = 0.0;     // HardeningMode
        double temperature_exp_m = 0.0;  // TemperatureExp
        double melt_temperature = 0.0;   // MeltTemperature
        double env_temperature = 0.0;    // EnvTemperature

        // Misc
        double ref_strain_rate = 1.0;    // RefStrainRate
        double specific_heat = 0.0;      // SpecificHeat
    };

    /**
     * @brief [LAW36] RateDependentPlastic 参数
     * @details 对应 MaterialType == "RateDependentPlastic"
     */
    struct RateDependentPlasticParams {
        double hardening_mode = 0.0;               // HardeningMode
        double failure_plastic_strain = 0.0;       // FailurePlasticStrain
        double fail_begin_tensile_strain = 0.0;    // FailBeginTensileStrain
        double fail_end_tensile_strain = 0.0;      // FailEndTensileStrain
        double elem_del_tensile_strain = 0.0;      // ElemDelTensileStrain
        std::string strain_rate_type;              // StrainRateType
        std::vector<std::string> yield_curves;     // StrainAndStrainRateYieldCurve
        std::vector<double> strain_rates;          // StrainRate
    };

    // ---------------------------------------------------------------------
    // MaterialType <-> typeid mapping (centralized maintenance, avoid scattered magic numbers in assembly systems)
    // ---------------------------------------------------------------------

    inline int material_typeid_from_model(const std::string& model) {
        if (model == "IsotropicElastic")      return 1;
        if (model == "Polynomial")            return 101;
        if (model == "ReducedPolynomial")     return 102;
        if (model == "OgdenRubber"
            || model == "Ogden2"
            || model == "Ogden")             return 103;

        // Reserved: plastic 3xx
        if (model == "IsotropicPlasticJC")    return 301; // LAW2
        if (model == "RateDependentPlastic")  return 302; // LAW36

        return 0; // Unknown / default
    }

    inline std::string material_model_from_typeid(int typeid_value) {
        switch (typeid_value) {
        case 1:   return "IsotropicElastic";
        case 101: return "Polynomial";
        case 102: return "ReducedPolynomial";
        case 103: return "OgdenRubber";
        case 301: return "IsotropicPlasticJC";
        case 302: return "RateDependentPlastic";
        default:  return {};
        }
    }

} // namespace Component
