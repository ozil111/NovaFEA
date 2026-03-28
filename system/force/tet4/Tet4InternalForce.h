// Tet4InternalForce.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2026 NovaFEA. All rights reserved.
 */
#pragma once

#include <entt/entt.hpp>

/**
 * @brief Compute and scatter internal forces for a single TET4 element
 * @param registry EnTT registry containing element and node data
 * @param element_entity Element entity to process
 * @return true if computed successfully, false otherwise
 */
bool compute_tet4_internal_forces(entt::registry& registry, entt::entity element_entity);

