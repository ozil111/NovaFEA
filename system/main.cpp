/**
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
 * If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright (c) 2025 hyperFEM. All rights reserved.
 * Author: Xiaotong Wang (or hyperFEM Team)
 */
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "parser_base/parserBase.h"       // 旧的 .xfem 解析器（向后兼容）
#include "parser_json/JsonParser.h"       // 新的 JSON 解析器
#include "exporter_base/exporterBase.h"
#include "DataContext.h"                  // 引入ECS数据中心
#include "components/mesh_components.h"   // 引入组件定义
#include "components/analysis_component.h"
#include "TopologyData.h"                 // 引入拓扑数据结构
#include "mesh/TopologySystems.h"         // 引入拓扑逻辑系统
#include "AppSession.h"                   // 引入会话状态机
#include "dof/DofNumberingSystem.h"      // DOF 映射系统
#include "mass/MassSystem.h"             // 质量系统
#include "force/InternalForceSystem.h"   // 内力系统
#include "load/LoadSystem.h"             // 载荷系统
#include "explicit/ExplicitSolver.h"     // 显式求解器
#include "material/mat1/LinearElasticMatrixSystem.h"  // 材料矩阵系统
#include "parser_simdroid/SimdroidParser.h" // Simdroid 解析器
#include "exporter_simdroid/SimdroidExporter.h" // Simdroid 导出器
#include "analysis/GraphBuilder.h"
#include "analysis/MermaidReporter.h"
#include "main0_explicit.h"              // 显式求解器逻辑
#include "main0_linearstatic.h"          // 线性静力求解器逻辑
#include "CommandProcessor.h"            // 命令处理器
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <filesystem>
#include <sstream>
#include <cstdlib>

// Function to print the startup banner
void print_banner() {
    // You can use an online ASCII art generator to create your own style
    std::cout << R"(
    .__                              ______________________   _____   
    |  |__ ___.__.______   __________\_   _____/\_   _____/  /     \
    |  |  <   |  |\____ \_/ __ \_  __ \    __)   |    __)_  /  \ /  \
    |   Y  \___  ||  |_> >  ___/|  | \/     \    |        \/    Y    \
    |___|  / ____||   __/ \___  >__|  \___  /   /_______  /\____|__  /
        \/\/     |__|        \/          \/            \/         \/ 

)" << std::endl;
    std::cout << "  hyperFEM Version: 0.0.1" << std::endl;
    std::cout << "  Author: xiaotong wang" << std::endl;
    std::cout << "  Email:  xiaotongwang98@gmail.com" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl << std::endl;
}

// Function to print help information
void print_help() {
    std::cout << "Usage: hyperFEM_app [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --input-file, -i <file>    Specify input file (.xfem or .json/.jsonc)" << std::endl;
    std::cout << "  --output-file, -o <file>   Specify output file (.xfem)" << std::endl;
    std::cout << "  --log-level, -l <level>    Set log level (trace, debug, info, warn, error, critical)" << std::endl;
    std::cout << "  --log-directory, -d <path> Set log file path" << std::endl;
    std::cout << "  --help, -h                 Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Supported Input Formats:" << std::endl;
    std::cout << "  .xfem  - Legacy text format (backward compatible)" << std::endl;
    std::cout << "  .json  - JSON format (recommended)" << std::endl;
    std::cout << "  .jsonc - JSON with comments (recommended)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  hyperFEM_app --input-file case/model.jsonc --output-file case/output.xfem" << std::endl;
    std::cout << "  hyperFEM_app --input-file case/node.xfem --output-file case/output.xfem" << std::endl;
}

// --- 引入交互模式的命令处理器 ---

int main(int argc, char* argv[]) {
    // --- Step 1: Print the banner first ---
    print_banner();

    // --- Step 2: Proceed with your original argument parsing and logger setup ---
    
    // 默认日志级别为info
    spdlog::level::level_enum log_level = spdlog::level::info;
    
    // 默认日志文件路径
    std::string log_file_path = "logs/hyperFEM.log";
    
    // 输入文件路径
    std::string input_file_path;
    
    // 输出文件路径
    std::string output_file_path;
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        } else if (arg == "--input-file" || arg == "-i") {
            if (i + 1 < argc) {
                input_file_path = argv[++i];
                
                // 验证文件扩展名（支持 .xfem, .json, .jsonc）
                std::filesystem::path file_path(input_file_path);
                std::string extension = file_path.extension().string();
                if (extension != ".xfem" && extension != ".json" && extension != ".jsonc") {
                    std::cerr << "Error: Input file must have .xfem, .json, or .jsonc extension" << std::endl;
                    std::cerr << "Provided file: " << input_file_path << std::endl;
                    return 1;
                }
                
                // 验证文件是否存在
                if (!std::filesystem::exists(file_path)) {
                    std::cerr << "Error: Input file does not exist: " << input_file_path << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: --input-file requires a file path argument" << std::endl;
                return 1;
            }
        } else if (arg == "--log-level" || arg == "-l") {
            if (i + 1 < argc) {
                std::string level_str = argv[++i];
                if (level_str == "trace") {
                    log_level = spdlog::level::trace;
                } else if (level_str == "debug") {
                    log_level = spdlog::level::debug;
                } else if (level_str == "info") {
                    log_level = spdlog::level::info;
                } else if (level_str == "warn" || level_str == "warning") {
                    log_level = spdlog::level::warn;
                } else if (level_str == "error") {
                    log_level = spdlog::level::err;
                } else if (level_str == "critical") {
                    log_level = spdlog::level::critical;
                } else {
                    std::cerr << "Unknown log level: " << level_str << std::endl;
                    std::cerr << "Valid levels: trace, debug, info, warn, error, critical" << std::endl;
                    return 1;
                }
            }
        } else if (arg == "--output-file" || arg == "-o") {
            if (i + 1 < argc) {
                output_file_path = argv[++i];
                
                // 验证文件扩展名
                std::filesystem::path file_path(output_file_path);
                if (file_path.extension() != ".xfem") {
                    std::cerr << "Error: Output file must have .xfem extension" << std::endl;
                    std::cerr << "Provided file: " << output_file_path << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: --output-file requires a file path argument" << std::endl;
                return 1;
            }
        } else if (arg == "--log-directory" || arg == "-d") {
            if (i + 1 < argc) {
                log_file_path = argv[++i];
            } else {
                std::cerr << "Error: --log-directory requires a path argument" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::cerr << "Use --help or -h for usage information" << std::endl;
            return 1;
        }
    }
    
    // 创建多个sink：文件和控制台
    std::vector<spdlog::sink_ptr> sinks;
    
    // 文件输出sink - 输出到用户指定的日志文件
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_path, true);
    sinks.push_back(file_sink);
    
    // 控制台输出sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.push_back(console_sink);
    
    // 创建logger并注册
    auto logger = std::make_shared<spdlog::logger>("hyperFEM", begin(sinks), end(sinks));
    logger->set_level(log_level);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_default_logger(logger);
    
    // --- Step 3: Now use the logger for actual logging ---
    spdlog::info("Logger initialized. Application starting...");
    spdlog::info("Log level set to: {}", spdlog::level::to_string_view(log_level));
    
    // --- Step 4: 模式决策 ---
    // 根据是否提供了 --input-file 来决定进入哪种模式
    if (!input_file_path.empty()) {
        // --- BATCH MODE EXECUTION ---
        spdlog::info("Running in Batch Mode.");
        spdlog::info("Processing input file: {}", input_file_path);
        
        // 创建DataContext对象来存储解析的数据
        DataContext data_context;
        
        // 根据文件扩展名自动选择解析器
        std::filesystem::path path(input_file_path);
        std::string extension = path.extension().string();
        bool parse_success = false;
        
        if (extension == ".json" || extension == ".jsonc") {
            spdlog::info("Detected JSON format, using JsonParser...");
            parse_success = JsonParser::parse(input_file_path, data_context);
        } else if (extension == ".xfem") {
            spdlog::info("Detected XFEM format, using FemParser (legacy)...");
            parse_success = FemParser::parse(input_file_path, data_context);
        }
        
        if (parse_success) {
            spdlog::info("Successfully parsed input file: {}", input_file_path);
            
            // Count entities using views
            auto node_count = data_context.registry.view<Component::Position>().size();
            auto element_count = data_context.registry.view<Component::Connectivity>().size();
            auto set_count = data_context.registry.view<Component::SetName>().size();
            
            spdlog::info("Total nodes loaded: {}", node_count);
            spdlog::info("Total elements loaded: {}", element_count);
            spdlog::info("Total sets loaded: {}", set_count);
            
            // --- Step 5: Run solver if analysis type is specified ---
            if (data_context.analysis_entity != entt::null
                && data_context.registry.valid(data_context.analysis_entity)
                && data_context.registry.all_of<Component::AnalysisType>(data_context.analysis_entity)
                && data_context.registry.get<Component::AnalysisType>(data_context.analysis_entity).value == "explicit") {
                run_explicit_solver(data_context);
            } else if (data_context.analysis_entity != entt::null
                && data_context.registry.valid(data_context.analysis_entity)
                && data_context.registry.all_of<Component::AnalysisType>(data_context.analysis_entity)
                && data_context.registry.get<Component::AnalysisType>(data_context.analysis_entity).value == "static") {
                run_linearstatic_solver(data_context);
            }
            
            // --- Step 6: Export the mesh if an output file is specified ---
            if (!output_file_path.empty()) {
                spdlog::info("Exporting mesh data to: {}", output_file_path);
                if (FemExporter::save(output_file_path, data_context)) {
                    spdlog::info("Successfully exported mesh data.");
                } else {
                    spdlog::error("Failed to export mesh data to: {}", output_file_path);
                    return 1;
                }
            }
            
        } else {
            spdlog::error("Failed to parse input file: {}", input_file_path);
            return 1;
        }
    } else {
        // --- INTERACTIVE MODE EXECUTION ---
        spdlog::info("No input file specified. Running in Interactive Mode.");
        spdlog::info("Type 'help' for a list of commands, 'quit' or 'exit' to leave.");
        
        AppSession session;
        std::string command_line;
        
        while (session.is_running) {
            std::cout << "hyperFEM> " << std::flush;
            if (std::getline(std::cin, command_line)) {
                if (!command_line.empty()) {
                    process_command(command_line, session);
                }
            } else {
                // 处理 Ctrl+D (Unix) 或 Ctrl+Z (Windows) 结束输入
                session.is_running = false;
                std::cout << std::endl; // 换行以保持终端整洁
            }
        }
    }
    
    spdlog::info("Application finished successfully.");
    return 0;
}