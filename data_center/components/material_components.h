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
#include <Eigen/Dense>
#include "entt/entt.hpp"

/**
 * @namespace Component
 * @brief 包含所有ECS组件 - 材料部分
 */
namespace Component {

    /**
     * @brief [通用] 标识这是一个材料实�?
     * @details 附加到所有材料实体上，存储其用户定义的ID (mid)�?
     * 这是连接单元和材料的"主键"�?
     */
    struct MaterialID {
        int value;
    };

    /**
     * @brief [Type 1] 各向同性线弹性参�?
     * @details 对应 typeid = 1
     */
    struct LinearElasticParams {
        double rho; // 密度
        double E;   // 弹性模�?
        double nu;  // 泊松�?
    };

    /**
     * @brief [通用] 材料类型字符串（来自 Simdroid �?MaterialType/Type�?
     * @details 例如: "IsotropicElastic"(LAW1), "IsotropicPlasticJC"(LAW2), "RateDependentPlastic"(LAW36)
     */
    struct MaterialModel {
        std::string value;
    };

    /**
     * @brief [派生数据] 线性弹性本构矩�?(D Matrix)
     * @details 这是一个运行时生成的组件，挂载�?Material 实体上�?
     * 对于 3D 各向同性材料，这是一�?6x6 矩阵�?
     * 使用 Abaqus/Fortran 顺序: xx, yy, zz, xy, yz, xz
     */
    struct LinearElasticMatrix {
        // 使用 Eigen 存储 6x6 矩阵 (Voigt notation: xx, yy, zz, xy, yz, xz - Abaqus/Fortran 顺序)
        Eigen::Matrix<double, 6, 6> D;
        
        // 标记是否已初始化，防止重复计�?
        bool is_initialized = false;
    };

    /**
     * @brief [通用-超弹性] 超弹性材料的通用数据
     * @details 存储 mode=0 (直接输入) �?mode=1 (曲线拟合)
     * 这个组件可以附加到所有超弹性材料实体上�?
     */
    struct HyperelasticMode {
        int order;
        bool fit_from_data; // 对应 mode 0 (false) �?1 (true)
        
        // 仅在 fit_from_data == true 时有意义
        std::vector<entt::entity> uniaxial_funcs;
        std::vector<entt::entity> biaxial_funcs;
        std::vector<entt::entity> planar_funcs;
        std::vector<entt::entity> volumetric_funcs;
        double nu; // 泊松比，仅用于曲线拟�?
    };

    /**
     * @brief [Type 101] Polynomial (N=order)
     * @details 仅在 HyperelasticMode::fit_from_data == false 时使�?
     */
    struct PolynomialParams {
        // Cij, 存储顺序 [C10, C01, C20, C02, ..., CN0, C0N]
        std::vector<double> c_ij; 
        // Di, 存储顺序 [D1, D2, ..., DN]
        std::vector<double> d_i;  
    };

    /**
     * @brief [Type 102] Reduced Polynomial (N=order)
     * @details 仅在 HyperelasticMode::fit_from_data == false 时使�?
     */
    struct ReducedPolynomialParams {
        // Ci0, 存储顺序 [C10, C20, ..., CN0]
        std::vector<double> c_i0; 
        // Di, 存储顺序 [D1, D2, ..., DN]
        std::vector<double> d_i;  
    };

    /**
     * @brief [Type 103] Ogden (N=order)
     * @details 仅在 HyperelasticMode::fit_from_data == false 时使�?
     */
    struct OgdenParams {
        // 存储顺序 [mu1, mu2, ..., muN]
        std::vector<double> mu_i;
        // 存储顺序 [alpha1, alpha2, ..., alphaN]
        std::vector<double> alpha_i;
        // 存储顺序 [D1, D2, ..., DN]
        std::vector<double> d_i;
    };

    /**
     * @brief [LAW2] IsotropicPlasticJC (Johnson-Cook) 参数
     * @details 对应 MaterialType == "IsotropicPlasticJC"
     * 字段名尽量与 docs/simdroid_ex_material.md �?MaterialConstants 对齐�?
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
    // MaterialType <-> typeid 映射（集中维护，避免在组�?系统里散落魔法数�?
    // ---------------------------------------------------------------------

    inline int material_typeid_from_model(const std::string& model) {
        if (model == "IsotropicElastic")      return 1;
        if (model == "Polynomial")            return 101;
        if (model == "ReducedPolynomial")     return 102;
        if (model == "OgdenRubber"
            || model == "Ogden2"
            || model == "Ogden")             return 103;

        // 预留：弹塑�?3xx �?
        if (model == "IsotropicPlasticJC")    return 301; // LAW2
        if (model == "RateDependentPlastic")  return 302; // LAW36

        return 0; // 未知 / 默认
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
