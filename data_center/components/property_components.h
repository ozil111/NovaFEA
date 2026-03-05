// data_center/components/property_components.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 hyperFEM. All rights reserved.
 * Author: Xiaotong Wang (or hyperFEM Team)
 */
#pragma once

#include <string>

/**
 * @namespace Component
 * @brief ECS组件 - Property（截面/属性）部分
 * @details Property 与 Material 解耦，仅存储截面与积分相关参数（积分方案、沙漏控制等）。
 * 材料通过 SimdroidPart 绑定，见 simdroid_components.h。
 */
namespace Component {

    /**
     * @brief [新] 附加到 Property 实体，存储其用户定义的ID (pid)
     * @details 用于标识Property实体，避免与其他类型实体的ID冲突
     */
    struct PropertyID {
        int value;
    };

    /**
     * @brief [新] 附加到 Property 实体，存储固体单元的截面/积分属性
     * @details 对应 JSON 中的 "property" 对象，仅存放与截面、积分相关的参数
     */
    struct SolidProperty {
        int type_id;                    // 来自 JSON 的 "typeid"
        int integration_network;        // 积分网络参数，如 "integration_network": 2
        std::string hourglass_control;  // 沙漏控制方法，如 "hourglass_control": "eas"
    };

    // 未来扩展: 可以添加其他类型的 Property
    // struct ShellProperty { ... };
    // struct BeamProperty { ... };

} // namespace Component

