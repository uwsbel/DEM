# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------- #
# Version information
# ---------------------------------------------------------------------------- #
set(DEME_VERSION_MAJOR 2)
set(DEME_VERSION_MINOR 1)
set(DEME_VERSION_PATCH 0)

# ---------------------------------------------------------------------------- #
# Backend selection API
# ---------------------------------------------------------------------------- #
# Standalone use:
#   -DDEME_GPU_BACKEND=AUTO|CUDA|HIP
# Embedded/submodule use:
#   parent project may preset DEME_GPU_BACKEND, DEME_ROCM_ROOT, DEME_HIP_ARCH.
#
# Compatibility:
#   USE_HIP is still accepted. If USE_HIP is defined and DEME_GPU_BACKEND was
#   not set explicitly, USE_HIP=ON maps to HIP, otherwise CUDA.

if(DEFINED USE_HIP AND NOT DEFINED DEME_GPU_BACKEND)
    if(USE_HIP)
        set(DEME_GPU_BACKEND "HIP")
    else()
        set(DEME_GPU_BACKEND "CUDA")
    endif()
endif()

set(DEME_GPU_BACKEND "${DEME_GPU_BACKEND}" CACHE STRING "GPU backend for DEME: AUTO, CUDA, HIP")
if(NOT DEME_GPU_BACKEND)
    set(DEME_GPU_BACKEND "AUTO" CACHE STRING "GPU backend for DEME: AUTO, CUDA, HIP" FORCE)
endif()
set_property(CACHE DEME_GPU_BACKEND PROPERTY STRINGS AUTO CUDA HIP)

set(DEME_MIN_HIP_VERSION "7.2.0" CACHE STRING "Minimum HIP version required by DEME")
set(DEME_ROCM_ROOT "" CACHE PATH "Explicit ROCm root for DEME")
set(DEME_HIP_ARCH "" CACHE STRING "AMD GPU architecture for DEME, e.g. gfx1151")

# Backward-compatible mirror for older subdirectories/tests.
if(DEME_GPU_BACKEND STREQUAL "HIP")
    set(USE_HIP ON CACHE BOOL "Use HIP (AMD GPU) instead of CUDA (NVIDIA GPU)" FORCE)
elseif(DEME_GPU_BACKEND STREQUAL "CUDA")
    set(USE_HIP OFF CACHE BOOL "Use HIP (AMD GPU) instead of CUDA (NVIDIA GPU)" FORCE)
else()
    option(USE_HIP "Use HIP (AMD GPU) instead of CUDA (NVIDIA GPU)" OFF)
endif()

# ---------------------------------------------------------------------------- #
# Helper functions for HIP detection
# ---------------------------------------------------------------------------- #
function(deme_read_hip_version hip_header out_var)
    if(NOT EXISTS "${hip_header}")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    file(STRINGS "${hip_header}" _major REGEX "^#define[ \t]+HIP_VERSION_MAJOR[ \t]+[0-9]+")
    file(STRINGS "${hip_header}" _minor REGEX "^#define[ \t]+HIP_VERSION_MINOR[ \t]+[0-9]+")
    file(STRINGS "${hip_header}" _patch REGEX "^#define[ \t]+HIP_VERSION_PATCH[ \t]+[0-9]+")

    if(NOT _major OR NOT _minor OR NOT _patch)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    string(REGEX REPLACE ".*HIP_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" _maj "${_major}")
    string(REGEX REPLACE ".*HIP_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" _min "${_minor}")
    string(REGEX REPLACE ".*HIP_VERSION_PATCH[ \t]+([0-9]+).*" "\\1" _pat "${_patch}")

    set(${out_var} "${_maj}.${_min}.${_pat}" PARENT_SCOPE)
endfunction()

function(deme_find_rocm out_root out_version)
    set(_roots)

    if(DEME_ROCM_ROOT)
        list(APPEND _roots "${DEME_ROCM_ROOT}")
    endif()
    if(DEFINED ROCM_PATH AND NOT ROCM_PATH STREQUAL "")
        list(APPEND _roots "${ROCM_PATH}")
    endif()
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        list(APPEND _roots "$ENV{ROCM_PATH}")
    endif()
    if(DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
        list(APPEND _roots "$ENV{HIP_PATH}")
    endif()

    list(APPEND _roots
        /opt/rocm
        /usr/lib/rocm
        /usr/local/rocm
    )

    foreach(_root IN LISTS _roots)
        if(NOT _root)
            continue()
        endif()

        set(_header "")
        if(EXISTS "${_root}/include/hip/hip_version.h")
            set(_header "${_root}/include/hip/hip_version.h")
        elseif(EXISTS "${_root}/hip/include/hip/hip_version.h")
            set(_header "${_root}/hip/include/hip/hip_version.h")
        endif()

        if(_header)
            deme_read_hip_version("${_header}" _ver)
            if(_ver)
                set(${out_root} "${_root}" PARENT_SCOPE)
                set(${out_version} "${_ver}" PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()

    set(${out_root} "" PARENT_SCOPE)
    set(${out_version} "" PARENT_SCOPE)
endfunction()

function(deme_resolve_gpu_backend)
    if(DEME_GPU_BACKEND STREQUAL "HIP")
        set(_want_hip TRUE)
        set(_want_cuda FALSE)
    elseif(DEME_GPU_BACKEND STREQUAL "CUDA")
        set(_want_hip FALSE)
        set(_want_cuda TRUE)
    elseif(DEME_GPU_BACKEND STREQUAL "AUTO")
        deme_find_rocm(_detected_rocm_root _detected_rocm_ver)
        if(_detected_rocm_root AND _detected_rocm_ver VERSION_GREATER_EQUAL DEME_MIN_HIP_VERSION)
            set(_want_hip TRUE)
            set(_want_cuda FALSE)
            set(DEME_GPU_BACKEND "HIP" CACHE STRING "GPU backend for DEME: AUTO, CUDA, HIP" FORCE)
            set(DEME_ROCM_ROOT "${_detected_rocm_root}" CACHE PATH "Explicit ROCm root for DEME" FORCE)
        else()
            set(_want_hip FALSE)
            set(_want_cuda TRUE)
            set(DEME_GPU_BACKEND "CUDA" CACHE STRING "GPU backend for DEME: AUTO, CUDA, HIP" FORCE)
        endif()
    else()
        message(FATAL_ERROR "Invalid DEME_GPU_BACKEND='${DEME_GPU_BACKEND}'. Expected AUTO, CUDA, or HIP.")
    endif()

    if(_want_hip)
        deme_find_rocm(_resolved_rocm_root _resolved_rocm_ver)
        if(NOT _resolved_rocm_root)
            message(FATAL_ERROR
                "DEME GPU backend is HIP, but no ROCm installation was found. "
                "Set DEME_ROCM_ROOT, ROCM_PATH, or HIP_PATH.")
        endif()
        if(_resolved_rocm_ver VERSION_LESS DEME_MIN_HIP_VERSION)
            message(FATAL_ERROR
                "Found ROCm ${_resolved_rocm_ver} at ${_resolved_rocm_root}, but DEME requires at least ${DEME_MIN_HIP_VERSION}.")
        endif()

        set(DEME_USE_HIP TRUE PARENT_SCOPE)
        set(DEME_USE_CUDA FALSE PARENT_SCOPE)
        set(DEME_RESOLVED_ROCM_ROOT "${_resolved_rocm_root}" PARENT_SCOPE)
        set(DEME_RESOLVED_HIP_VERSION "${_resolved_rocm_ver}" PARENT_SCOPE)
        set(DEME_ROCM_ROOT "${_resolved_rocm_root}" CACHE PATH "Explicit ROCm root for DEME" FORCE)

        if(NOT DEME_HIP_ARCH)
            set(DEME_HIP_ARCH "gfx1151" CACHE STRING "AMD GPU architecture for DEME, e.g. gfx1151" FORCE)
        endif()
        set(DEME_HIP_ARCH "${DEME_HIP_ARCH}" PARENT_SCOPE)

        set(USE_HIP ON CACHE BOOL "Use HIP (AMD GPU) instead of CUDA (NVIDIA GPU)" FORCE)
        message(STATUS "DEME: using HIP backend with ROCm ${_resolved_rocm_ver} at ${_resolved_rocm_root}")
    else()
        set(DEME_USE_HIP FALSE PARENT_SCOPE)
        set(DEME_USE_CUDA TRUE PARENT_SCOPE)
        set(DEME_RESOLVED_ROCM_ROOT "" PARENT_SCOPE)
        set(DEME_RESOLVED_HIP_VERSION "" PARENT_SCOPE)
        set(USE_HIP OFF CACHE BOOL "Use HIP (AMD GPU) instead of CUDA (NVIDIA GPU)" FORCE)
        message(STATUS "DEME: using CUDA backend")
    endif()
endfunction()

deme_resolve_gpu_backend()

# ---------------------------------------------------------------------------- #
# Project settings
# ---------------------------------------------------------------------------- #
if(DEME_USE_HIP)
    set(PROJECT_LANGUAGES CXX HIP)
else()
    set(PROJECT_LANGUAGES CXX CUDA)
endif()

# Help CMake locate ROCm in standalone HIP builds before project().
if(DEME_USE_HIP)
    list(PREPEND CMAKE_PREFIX_PATH "${DEME_RESOLVED_ROCM_ROOT}")
    if(NOT DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT)
        set(CMAKE_HIP_COMPILER_ROCM_ROOT "${DEME_RESOLVED_ROCM_ROOT}" CACHE PATH "ROCm root path")
    endif()
    if(NOT DEFINED CMAKE_HIP_PLATFORM)
        set(CMAKE_HIP_PLATFORM "amd" CACHE STRING "HIP platform")
    endif()
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES AND DEME_HIP_ARCH)
        set(CMAKE_HIP_ARCHITECTURES "${DEME_HIP_ARCH}" CACHE STRING "HIP architectures")
    endif()
endif()

project(
    Chrono-DEM-Engine
    VERSION ${DEME_VERSION_MAJOR}.${DEME_VERSION_MINOR}.${DEME_VERSION_PATCH}
    LANGUAGES ${PROJECT_LANGUAGES}
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting CMAKE_BUILD_TYPE to 'Release' (default)")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

add_compile_definitions(DEME_BEING_CMAKE_COMPILED)
add_compile_definitions(DEME_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# ---------------------------------------------------------------------------- #
# Additional packages
# ---------------------------------------------------------------------------- #
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)

include(cmake/FixNinjaColors.cmake)
fix_ninja_colors()

if(DEME_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    find_package(
        CUB REQUIRED
        HINTS ${CUDAToolkit_ROOT}/lib64/cmake/cub
    )

    find_path(
        NVIDIAJitifyPath
        NAMES jitify.hpp
        PATHS "${CMAKE_CURRENT_LIST_DIR}/thirdparty/jitify"
    )
endif()

option(USE_CHPF "Toggle the use of ChPF for outputting" OFF)

if(USE_CHPF)
    find_package(ChPF 3.0 QUIET)
    if(NOT ChPF_FOUND)
        FetchContent_Declare(
            ChPF
            GIT_REPOSITORY https://gitlab.com/uwsbel/chpf.git
            GIT_TAG 0699f01e720cecf58719a1651ab797c6a62a1897
        )
        FetchContent_MakeAvailable(ChPF)
    endif()

    if(ChPF_FOUND)
        set(ChPF_IMPORTED_NAME "ChPF::ChPF")
    elseif(chpf_POPULATED)
        set(ChPF_IMPORTED_NAME "ChPF")
    endif()
endif()

set(USE_MANAGED_ARRAYS_DESC
    "Use managed CUDA arrays (instead of manual CUDA host and device memory) to store data"
    "Switching it on makes the project operable on a Linux platform (excluding WSL) only."
)
option(USE_MANAGED_ARRAYS "${USE_MANAGED_ARRAYS_DESC}" OFF)

# ---------------------------------------------------------------------------- #
# HIP-specific dependency discovery
# ---------------------------------------------------------------------------- #
if(DEME_USE_HIP)
    find_library(HIP_HIPRTC_LIBRARY
        NAMES hiprtc libhiprtc
        HINTS "${DEME_RESOLVED_ROCM_ROOT}/lib" "${DEME_RESOLVED_ROCM_ROOT}/lib64"
        REQUIRED
    )

    find_library(HIP_HSA_LIBRARY
        NAMES hsa-runtime64 libhsa-runtime64
        HINTS "${DEME_RESOLVED_ROCM_ROOT}/lib" "${DEME_RESOLVED_ROCM_ROOT}/lib64"
        REQUIRED
    )

    find_library(HIP_AMDHIP64_LIBRARY
        NAMES amdhip64 libamdhip64
        HINTS "${DEME_RESOLVED_ROCM_ROOT}/lib" "${DEME_RESOLVED_ROCM_ROOT}/lib64"
        REQUIRED
    )

    find_path(
        hipCUB_PATH
        NAMES hipcub/hipcub.hpp
        HINTS
            "${DEME_RESOLVED_ROCM_ROOT}/include"
            /opt/rocm/include
            /usr/include
        CACHE PATH "Path to hipCUB headers" FORCE
    )

    if(NOT hipCUB_PATH)
        message(WARNING "hipCUB not found. Some features may not be available.")
    else()
        message(STATUS "hipCUB found at: ${hipCUB_PATH}")
    endif()
endif()

# ---------------------------------------------------------------------------- #
# Global configuration
# ---------------------------------------------------------------------------- #
include(cmake/CxxStdAutodetect.cmake)

set(TargetCXXStandard "STD_AUTODETECT" CACHE STRING "The C++ standard used by the compiler")
set_property(CACHE TargetCXXStandard PROPERTY STRINGS STD_AUTODETECT STD_CXX11 STD_CXX14 STD_CXX17 STD_CXX20)

if(TargetCXXStandard STREQUAL STD_CXX11)
    set(CXXSTD_MAX 11)
elseif(TargetCXXStandard STREQUAL STD_CXX14)
    set(CXXSTD_MAX 14)
elseif(TargetCXXStandard STREQUAL STD_CXX17)
    set(CXXSTD_MAX 17)
elseif(TargetCXXStandard STREQUAL STD_CXX20)
    set(CXXSTD_MAX 20)
else()
    set(CXXSTD_MAX 17)
endif()

cxx_std_autodetect()

set(ProjectIncludeSource "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(ProjectIncludeGenerated "${CMAKE_BINARY_DIR}/src")

if(DEME_USE_CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

set(DEME_INSTALL_DEMO "bin")

# ---------------------------------------------------------------------------- #
# Source-level configuration
# ---------------------------------------------------------------------------- #
add_subdirectory(src/core)
add_subdirectory(src/DEM)
add_subdirectory(src/algorithms)

# ---------------------------------------------------------------------------- #
# Final library generation
# ---------------------------------------------------------------------------- #
add_library(
    simulator_multi_gpu
    STATIC
    $<TARGET_OBJECTS:core>
    $<TARGET_OBJECTS:DEM>
    $<TARGET_OBJECTS:algorithms>
)

target_include_directories(simulator_multi_gpu
    PUBLIC $<BUILD_INTERFACE:${ProjectIncludeSource}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PUBLIC $<BUILD_INTERFACE:${ProjectIncludeGenerated}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(DEME_USE_HIP)
    target_compile_definitions(simulator_multi_gpu PUBLIC DEME_USE_HIP __HIP_PLATFORM_AMD__)
    target_include_directories(simulator_multi_gpu SYSTEM PUBLIC
        "${DEME_RESOLVED_ROCM_ROOT}/include"
    )
endif()

if(USE_CHPF)
    target_compile_definitions(simulator_multi_gpu PUBLIC DEME_USE_CHPF)
    set(USE_CHPF_STR "ON")

    if(DEME_USE_HIP)
        target_link_libraries(simulator_multi_gpu
            PUBLIC ${ChPF_IMPORTED_NAME}
            PUBLIC ${HIP_HIPRTC_LIBRARY}
            PUBLIC ${HIP_HSA_LIBRARY}
            PUBLIC ${HIP_AMDHIP64_LIBRARY}
            PUBLIC DEMERuntimeDataHelper
        )
    else()
        target_link_libraries(simulator_multi_gpu
            PUBLIC CUDA::cudart
            PUBLIC CUDA::nvrtc
            PUBLIC CUDA::cuda_driver
            PUBLIC ${ChPF_IMPORTED_NAME}
            PUBLIC DEMERuntimeDataHelper
        )
    endif()
else()
    set(USE_CHPF_STR "OFF")

    if(DEME_USE_HIP)
        target_link_libraries(simulator_multi_gpu
            PUBLIC ${HIP_HIPRTC_LIBRARY}
            PUBLIC ${HIP_HSA_LIBRARY}
            PUBLIC ${HIP_AMDHIP64_LIBRARY}
            PUBLIC DEMERuntimeDataHelper
        )
    else()
        target_link_libraries(simulator_multi_gpu
            PUBLIC CUDA::cudart
            PUBLIC CUDA::nvrtc
            PUBLIC CUDA::cuda_driver
            PUBLIC DEMERuntimeDataHelper
        )
    endif()
endif()

if(USE_MANAGED_ARRAYS)
    target_compile_definitions(simulator_multi_gpu PUBLIC DEME_USE_MANAGED_ARRAYS)
endif()

if(WIN32)
    target_link_libraries(simulator_multi_gpu PUBLIC dbghelp)
endif()

if(DEME_USE_CUDA)
    set_target_properties(simulator_multi_gpu PROPERTIES LINKER_LANGUAGE CUDA)
endif()

# ---------------------------------------------------------------------------- #
# Export and install the generated targets
# ---------------------------------------------------------------------------- #
set(config_directory ${CMAKE_INSTALL_LIBDIR}/cmake/DEME)

install(
    TARGETS simulator_multi_gpu DEMERuntimeDataHelper
    EXPORT DEMETargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(CODE "message(\"Removing build-tree libraries from installation target...\")")
install(CODE "file(REMOVE \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libDEMERuntimeDataHelper.so\")")

install(
    TARGETS DEMERuntimeDataHelper_install
    EXPORT DEMETargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

export(
    EXPORT DEMETargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/DEME/DEMETargets.cmake"
    NAMESPACE DEME::
)

install(
    EXPORT DEMETargets
    DESTINATION ${config_directory}
    NAMESPACE DEME::
)

set(CONF_INCLUDE_DIRS "${ProjectIncludeSource}" "${ProjectIncludeGenerated}")
configure_file(
    cmake/DEMEConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/DEME/DEMEConfig.cmake"
    @ONLY
)

file(RELATIVE_PATH REL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/${config_directory} ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
set(CONF_INCLUDE_DIRS "\${DEMECMakeDir}/${REL_INCLUDE_DIR}")
set(CONF_SHARE_DATA_DIRS "\${DEMECMakeDir}/${REL_INCLUDE_DIR}/../share/DEME/data")

configure_file(
    cmake/DEMEConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DEMEConfig.cmake"
    @ONLY
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/DEME/DEMEConfigVersion.cmake
    VERSION ${DEME_VERSION_MAJOR}.${DEME_VERSION_MINOR}.${DEME_VERSION_PATCH}
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DEMEConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/DEME/DEMEConfigVersion.cmake"
    DESTINATION ${config_directory}
)

# ---------------------------------------------------------------------------- #
# Copy data directory over
# ---------------------------------------------------------------------------- #
file(COPY ${CMAKE_CURRENT_LIST_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/data/)
add_custom_target(sync_data ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_LIST_DIR}/data
            ${CMAKE_BINARY_DIR}/data
    COMMENT "Syncing data directory into build tree")
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/data/ DESTINATION ${CMAKE_INSTALL_DATADIR}/DEME/data/)

# ---------------------------------------------------------------------------- #
# Copy kernels over
# ---------------------------------------------------------------------------- #
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/kernel/ DESTINATION ${CMAKE_BINARY_DIR}/kernel/)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/DEM/Defines.h DESTINATION ${CMAKE_BINARY_DIR}/DEM)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/DEM/VariableTypes.h DESTINATION ${CMAKE_BINARY_DIR}/DEM)
file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/core/utils/GpuRuntime.hpp DESTINATION ${CMAKE_BINARY_DIR}/core/utils/)
add_custom_target(sync_kernels ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_LIST_DIR}/src/kernel
            ${CMAKE_BINARY_DIR}/kernel
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_LIST_DIR}/src/DEM/Defines.h
            ${CMAKE_BINARY_DIR}/DEM/Defines.h
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_LIST_DIR}/src/DEM/VariableTypes.h
            ${CMAKE_BINARY_DIR}/DEM/VariableTypes.h
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_LIST_DIR}/src/core/utils/GpuRuntime.hpp
            ${CMAKE_BINARY_DIR}/core/utils/GpuRuntime.hpp
    COMMENT "Syncing kernel sources into build tree")
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/kernel/ DESTINATION ${CMAKE_INSTALL_DATADIR}/DEME/kernel/)

# ---------------------------------------------------------------------------- #
# Build embedded demos
# ---------------------------------------------------------------------------- #
add_subdirectory(src/demo)