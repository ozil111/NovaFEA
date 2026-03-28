// system/exporter_base/exporterBase.cpp
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#include "exporter_base/exporterBase.h"
#include "exporter_json/JsonExporter.h"
#include <spdlog/spdlog.h>

bool FemExporter::save(const std::string& filepath, const DataContext& data_context) {
    spdlog::info("FemExporter: Saving model to JSON format: {}", filepath);
    return JsonExporter::save(filepath, data_context);
}
