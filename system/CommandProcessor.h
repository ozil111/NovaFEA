/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 hyperFEM. All rights reserved.
 * Author: Xiaotong Wang (or hyperFEM Team)
 */
#pragma once
#include "AppSession.h"
#include <string>

/**
 * @brief Process interactive mode commands
 * @param command_line The command line string to process
 * @param session The application session state
 */
void process_command(const std::string& command_line, AppSession& session);
