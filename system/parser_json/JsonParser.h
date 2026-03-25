// system/parser_json/JsonParser.h
/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 NovaFEA. All rights reserved.
 * Author: Xiaotong Wang (or NovaFEA Team)
 */
#pragma once

#include "DataContext.h"
#include "nlohmann/json_fwd.hpp"
#include <string>
#include <unordered_map>

/**
 * @class JsonParser
 * @brief JSON format FEM input file parser
 * @details Uses N-Step parsing strategy, strictly parsing according to entity dependency order:
 * 
 * Parsing order (critical!):
 *   1. Material  (no dependencies)
 *   2. Property  (depends on Material)
 *   3. Node      (no dependencies)
 *   4. Element   (depends on Node, Property)
 *   5. NodeSet   (depends on Node)
 *   6. EleSet    (depends on Element)
 *   7. Load      (no dependencies, but needs to be after NodeSet)
 *   8. Boundary  (no dependencies, but needs to be after NodeSet)
 *   9. Apply Load/Boundary (depends on Load, Boundary, NodeSet)
 * 
 * Architecture features:
 *   - Uses nlohmann::json library for parsing
 *   - Uses Plan B reference mode: entities are associated via entt::entity handles
 *   - Temporarily stores user ID to entity mappings through multiple std::unordered_map<int, entt::entity>
 *   - Each parsing step is independently encapsulated as a private method, easy to maintain and test
 */
class JsonParser {
public:
    /**
     * @brief Parse JSON format input file and populate DataContext
     * @param filepath JSON file path (recommended extension .jsonc supports comments)
     * @param data_context [out] DataContext object to be populated
     * @return true if parsing succeeds, false if file cannot be opened or error occurs
     */
    static bool parse(const std::string& filepath, DataContext& data_context);

    /**
     * @brief Merge JSON fragment into an existing DataContext (does not clear the registry).
     * @details Reuses the same top-level keys and per-entity schema as full-file import (@ref parse).
     *          ID maps are seeded from entities already in the registry so new entries can reference
     *          existing materials, nodesets, curves, etc. Duplicate numeric IDs in the fragment are
     *          skipped with the same warnings as in parse().
     * @param filepath Path to .json / .jsonc (comments allowed).
     * @param data_context Existing model to extend.
     * @return true on success.
     */
    static bool apply_fragment(const std::string& filepath, DataContext& data_context);

private:
    static bool run_parse_pipeline(const nlohmann::json& j, DataContext& data_context, bool replace_context);

    // ====================================================================
    // N-Step 解析方法（按依赖顺序调用�?
    // ====================================================================

    /**
     * @brief Step 1: Parse Material entities
     * @param j JSON root object
     * @param registry EnTT registry
     * @param material_id_map [out] mid -> entity mapping
     */
    static void parse_materials(
        const nlohmann::json& j,
        entt::registry& registry,
        std::unordered_map<int, entt::entity>& material_id_map
    );

    /**
     * @brief Step 2: Parse Property entities
     * @param j JSON root object
     * @param registry EnTT registry
     * @param material_id_map [in] mid -> entity mapping
     * @param property_id_map [out] pid -> entity mapping
     * @param property_id_to_material [out] pid -> material entity, for use by build_parts_from_properties
     */
    static void parse_properties(
        const nlohmann::json& j,
        entt::registry& registry,
        const std::unordered_map<int, entt::entity>& material_id_map,
        std::unordered_map<int, entt::entity>& property_id_map,
        std::unordered_map<int, entt::entity>& property_id_to_material
    );

    /**
     * @brief Step 4.5: Build SimdroidPart and element sets from Property, materials bound through Part
     */
    static void build_parts_from_properties(
        entt::registry& registry,
        const std::unordered_map<int, entt::entity>& property_id_map,
        const std::unordered_map<int, entt::entity>& property_id_to_material,
        const std::unordered_map<int, entt::entity>& element_id_map
    );

    /**
     * @brief Step 3: Parse Node entities
     * @param j JSON root object
     * @param registry EnTT registry
     * @param node_id_map [out] nid -> entity mapping
     */
    static void parse_nodes(
        const nlohmann::json& j,
        entt::registry& registry,
        std::unordered_map<int, entt::entity>& node_id_map
    );

    /**
     * @brief Step 4: Parse Element entities
     * @param j JSON root object
     * @param registry EnTT registry
     * @param node_id_map [in] nid -> entity mapping
     * @param property_id_map [in] pid -> entity mapping
     * @param element_id_map [out] eid -> entity mapping
     */
    static void parse_elements(
        const nlohmann::json& j,
        entt::registry& registry,
        const std::unordered_map<int, entt::entity>& node_id_map,
        const std::unordered_map<int, entt::entity>& property_id_map,
        std::unordered_map<int, entt::entity>& element_id_map
    );

    /**
     * @brief Step 5: Parse NodeSet entities
     * @param j JSON root object
     * @param registry EnTT registry
     * @param node_id_map [in] nid -> entity mapping
     * @param nodeset_id_map [out] nsid -> entity mapping
     */
    static void parse_nodesets(
        const nlohmann::json& j,
        entt::registry& registry,
        const std::unordered_map<int, entt::entity>& node_id_map,
        std::unordered_map<int, entt::entity>& nodeset_id_map
    );

    /**
     * @brief Step 6: Parse EleSet entities
     * @param j JSON root object
     * @param registry EnTT registry
     * @param element_id_map [in] eid -> entity mapping
     * @param eleset_id_map [out] esid -> entity mapping
     */
    static void parse_elesets(
        const nlohmann::json& j,
        entt::registry& registry,
        const std::unordered_map<int, entt::entity>& element_id_map,
        std::unordered_map<int, entt::entity>& eleset_id_map
    );

    /**
     * @brief Step 6.5: Parse Curve entities (curve definitions)
     * @param j JSON root object
     * @param registry EnTT registry
     * @param curve_id_map [out] cid -> entity mapping
     */
    static void parse_curves(
        const nlohmann::json& j,
        entt::registry& registry,
        std::unordered_map<int, entt::entity>& curve_id_map
    );

    /**
     * @brief Step 7: Parse Load entities (abstract definitions)
     * @param j JSON root object
     * @param registry EnTT registry
     * @param load_id_map [out] lid -> entity mapping
     * @param curve_id_map [inout] cid -> entity mapping table (may be modified to add default curve)
     */
    static void parse_loads(
        const nlohmann::json& j,
        entt::registry& registry,
        std::unordered_map<int, entt::entity>& load_id_map,
        std::unordered_map<int, entt::entity>& curve_id_map
    );

    /**
     * @brief Step 8: Parse Boundary entities (abstract definitions)
     * @param j JSON root object
     * @param registry EnTT registry
     * @param boundary_id_map [out] bid -> entity mapping
     */
    static void parse_boundaries(
        const nlohmann::json& j,
        entt::registry& registry,
        std::unordered_map<int, entt::entity>& boundary_id_map
    );

    /**
     * @brief Step 9: "Apply" Load to Node (establish reference relationships)
     * @param j JSON root object
     * @param registry EnTT registry
     * @param load_id_map [in] lid -> entity mapping
     * @param nodeset_id_map [in] nsid -> entity mapping
     */
    static void apply_loads(
        const nlohmann::json& j,
        entt::registry& registry,
        const std::unordered_map<int, entt::entity>& load_id_map,
        const std::unordered_map<int, entt::entity>& nodeset_id_map
    );

    /**
     * @brief Step 10: "Apply" Boundary to Node (establish reference relationships)
     * @param j JSON root object
     * @param registry EnTT registry
     * @param boundary_id_map [in] bid -> entity mapping
     * @param nodeset_id_map [in] nsid -> entity mapping
     */
    static void apply_boundaries(
        const nlohmann::json& j,
        entt::registry& registry,
        const std::unordered_map<int, entt::entity>& boundary_id_map,
        const std::unordered_map<int, entt::entity>& nodeset_id_map
    );

    /**
     * @brief Step 11: Parse Analysis entities
     * @param j JSON root object
     * @param registry EnTT registry
     * @param analysis_id_map [out] aid -> entity mapping
     */
    static void parse_analysis(
        const nlohmann::json& j,
        entt::registry& registry,
        std::unordered_map<int, entt::entity>& analysis_id_map
    );

    /**
     * @brief Step 12: Parse Output entities (currently only supports a single global output, no oid needed)
     * @param j JSON root object, j["output"] is a single object, e.g. {"node_output":["displacement"],"interval_time":0.01}
     * @param registry EnTT registry
     * @param output_id_map [out] unique output entity stored in output_id_map[0]
     */
    static void parse_output(
        const nlohmann::json& j,
        entt::registry& registry,
        std::unordered_map<int, entt::entity>& output_id_map
    );
};

