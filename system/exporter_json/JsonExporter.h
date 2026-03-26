/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include "DataContext.h"
#include <string>

/**
 * @class JsonExporter
 * @brief JSON format FEM input file exporter
 * @details Performs the inverse logic of JsonParser to export DataContext to a JSON file.
 *          This format is intended to completely replace the old .xfem format as the native format.
 */
class JsonExporter {
public:
    /**
     * @brief Saves the mesh data from DataContext's registry to a specified JSON file.
     * @param filepath The path to the file to be saved.
     * @param data_context [in] The DataContext object containing the registry with mesh data.
     * @return true if saving is successful, false if the file cannot be opened or a critical error occurs.
     */
    static bool save(const std::string& filepath, const DataContext& data_context);
};
