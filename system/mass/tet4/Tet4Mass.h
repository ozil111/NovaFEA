// Tet4Mass.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include "entt/entt.hpp"

/**
 * @brief Compute lumped mass for linear tetrahedron (Tet4) elements
 * @param registry EnTT registry containing elements and nodes
 * @param element_entity Element entity to process
 * @return true if mass was successfully computed and distributed, false otherwise
 */
bool compute_tet4_mass(entt::registry& registry, entt::entity element_entity);
