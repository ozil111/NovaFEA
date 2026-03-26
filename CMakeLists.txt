cmake_minimum_required(VERSION 3.19)

# Set CMake policy
cmake_policy(SET CMP0167 NEW)

# Disable automatic vcpkg install
set(VCPKG_MANIFEST_INSTALL OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set vcpkg toolchain file path
set(CMAKE_TOOLCHAIN_FILE "D:/Document/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Set vcpkg installation path - dynamically based on triplet
if(DEFINED VCPKG_TARGET_TRIPLET)
    set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}" "${CMAKE_PREFIX_PATH}")
else()
    # Default to x64-mingw-dynamic
    set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-mingw-dynamic" "${CMAKE_PREFIX_PATH}")
endif()

# Project name and C++ standard
project(NovaFEA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
if(MSVC)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/msvc)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/msvc)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/msvc)
    set(TEST_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/msvc/tests)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
    set(TEST_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}/tests)
endif()

# MSVC specific settings
if(MSVC)
    add_compile_options(/utf-8 /bigobj)
    add_definitions(-D_USE_MATH_DEFINES)
endif()

# MinGW specific settings: support for large object files (resolves 'too many sections' error)
# When using MinGW compiler, add -Wa,-mbig-obj flag to support large object files
# This resolves compilation errors caused by large header libraries like nlohmann/json.hpp
if(MINGW OR (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    add_compile_options(-Wa,-mbig-obj)
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Include header directories
include_directories(${CMAKE_SOURCE_DIR}/data_center)
include_directories(${CMAKE_SOURCE_DIR}/system)

# Find dependencies
find_package(spdlog CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(eigen3 CONFIG REQUIRED)
find_package(gtest CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(entt CONFIG REQUIRED)
find_package(tinyxml2 CONFIG REQUIRED)
find_package(ftxui CONFIG REQUIRED)

# Add compile definitions to use header-only mode
add_compile_definitions(
    SPDLOG_HEADER_ONLY 
    FMT_HEADER_ONLY
)

# Configure EnTT to use 64-bit entity IDs to support large-scale meshes
# Default 32-bit IDs support up to about 1 million entities (2^20 = 1,048,576)
# 64-bit IDs support virtually unlimited entities, suitable for large FEM models
add_compile_definitions(
    ENTT_ID_TYPE=std::uint64_t
)

# Collect source files
file(GLOB_RECURSE NovaFEA_SOURCES
    system/*.cpp
    system/*.c
)

# Exclude convenience implementation file, as its content has been separated into individual modules
list(REMOVE_ITEM NovaFEA_SOURCES "${CMAKE_SOURCE_DIR}/src/NovaFEA_implementations.cpp")

# Main executable
add_executable(NovaFEA_app ${NovaFEA_SOURCES})

# Link dependencies
target_link_libraries(NovaFEA_app 
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    EnTT::EnTT
    tinyxml2::tinyxml2
    ftxui::dom
    ftxui::component
    ftxui::screen
)

# Set working directory for main program
set_target_properties(NovaFEA_app PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ========== Automatically copy dependent DLLs to output directory ==========
if(WIN32)
    if(MSVC)
        # MSVC: Copy vcpkg DLLs
        add_custom_command(TARGET NovaFEA_app POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Copying MSVC DLLs..."
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/$<IF:$<CONFIG:Debug>,debug/,>bin/*.dll"
                "$<TARGET_FILE_DIR:NovaFEA_app>"
            COMMENT "Copying runtime DLLs for MSVC build"
        )
    else()
        # MinGW: Copy vcpkg DLLs and MinGW runtime DLLs
        # Find MinGW installation path
        get_filename_component(MINGW_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
        
        # Set vcpkg DLL directory based on build type
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(VCPKG_BIN_DIR "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-mingw-dynamic/debug/bin")
            set(SPDLOG_DLL "libspdlogd.dll")
            set(FMT_DLL "libfmtd.dll")
            set(TINYXML2_DLL "libtinyxml2.dll")
        else()
            set(VCPKG_BIN_DIR "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-mingw-dynamic/bin")
            set(SPDLOG_DLL "libspdlog.dll")
            set(FMT_DLL "libfmt.dll")
            set(TINYXML2_DLL "libtinyxml2.dll")
        endif()
        
        add_custom_command(TARGET NovaFEA_app POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Copying MinGW DLLs for ${CMAKE_BUILD_TYPE} build..."
            
            # Copy vcpkg DLLs
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_BIN_DIR}/${SPDLOG_DLL}"
                "$<TARGET_FILE_DIR:NovaFEA_app>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_BIN_DIR}/${FMT_DLL}"
                "$<TARGET_FILE_DIR:NovaFEA_app>"
            
            # Copy MinGW runtime DLLs
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${MINGW_BIN_DIR}/libgcc_s_seh-1.dll"
                "$<TARGET_FILE_DIR:NovaFEA_app>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${MINGW_BIN_DIR}/libstdc++-6.dll"
                "$<TARGET_FILE_DIR:NovaFEA_app>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${MINGW_BIN_DIR}/libwinpthread-1.dll"
                "$<TARGET_FILE_DIR:NovaFEA_app>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_BIN_DIR}/${TINYXML2_DLL}"
                "$<TARGET_FILE_DIR:NovaFEA_app>"
            COMMENT "Copying runtime DLLs for MinGW build"
        )
    endif()
endif()

# Include test module
option(BUILD_TESTING "Build tests" OFF)
if(BUILD_TESTING)
    add_subdirectory(test)
endif()