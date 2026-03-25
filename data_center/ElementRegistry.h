// ElementRegistry.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include <string>
#include <unordered_map>
#include <stdexcept>

// Fixed properties of element types
struct ElementProperties {
    int numNodes;
    int dimension;
    std::string name;
};

// Use class and static members to implement singleton pattern, ensuring the registry is globally unique
class ElementRegistry {
public:
    // Get the globally unique registry instance
    static ElementRegistry& getInstance() {
        static ElementRegistry instance; // C++11 guarantees thread-safe initialization
        return instance;
    }

    // Get element properties by type ID
    const ElementProperties& getProperties(int typeId) const {
        auto it = propertiesMap.find(typeId);
        if (it == propertiesMap.end()) {
            throw std::runtime_error("Unknown element type ID: " + std::to_string(typeId));
        }
        return it->second;
    }

private:
    // Private constructor to prevent external instantiation
    ElementRegistry() {
        initialize();
    }

    // Initialization function to populate all supported element types
    void initialize() {
        propertiesMap[102] = {2, 1, "Line2"};
        propertiesMap[103] = {3, 1, "Line3"};
        propertiesMap[203] = {3, 2, "Triangle3"};
        propertiesMap[204] = {4, 2, "Quad4"};
        propertiesMap[208] = {8, 2, "Quad8"};
        propertiesMap[304] = {4, 3, "Tetra4"};
        propertiesMap[306] = {6, 3, "Penta6"}; // Note: 306 is typically a wedge/penta, not a pyramid
        propertiesMap[308] = {8, 3, "Hexa8"};
        propertiesMap[310] = {10, 3, "Tetra10"};
        propertiesMap[320] = {20, 3, "Hexa20"};
    }

    // Prohibit copying and assignment
    ElementRegistry(const ElementRegistry&) = delete;
    ElementRegistry& operator=(const ElementRegistry&) = delete;

    std::unordered_map<int, ElementProperties> propertiesMap;
};
