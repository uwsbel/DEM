# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# 
#	SPDX-License-Identifier: BSD-3-Clause

message(STATUS "==== Configuring Core Components ====")

set (core_message "[core]")

# Capture CUDA/HIP include dirs before generating ApiVersion.h so the runtime JIT path
# can reuse the same include search list on both backends.
set(_DEME_CUDA_INC_DIRS "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
if(NOT _DEME_CUDA_INC_DIRS AND DEFINED CUDAToolkit_INCLUDE_DIRS)
    set(_DEME_CUDA_INC_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
endif()
if(NOT _DEME_CUDA_INC_DIRS AND DEFINED CUDA_INCLUDE_DIRS)
    set(_DEME_CUDA_INC_DIRS "${CUDA_INCLUDE_DIRS}")
endif()

set(_DEME_HIP_INC_DIRS "")
if(USE_HIP)
    if(NOT ROCM_PATH)
        foreach(_de_candidate "$ENV{ROCM_PATH}" "$ENV{ROCM_HOME}" "${CMAKE_HIP_COMPILER_ROCM_ROOT}" "/opt/rocm" "/usr/lib/rocm")
            if(_de_candidate AND EXISTS "${_de_candidate}")
                set(ROCM_PATH "${_de_candidate}")
                break()
            endif()
        endforeach()
    endif()

    if(ROCM_PATH AND EXISTS "${ROCM_PATH}/include")
        list(APPEND _DEME_HIP_INC_DIRS "${ROCM_PATH}/include")
    endif()

    set(_DEME_HIP_RESOURCE_DIR "")
    if(CMAKE_HIP_COMPILER)
        execute_process(
            COMMAND "${CMAKE_HIP_COMPILER}" --print-resource-dir
            OUTPUT_VARIABLE _DEME_HIP_RESOURCE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
    if(NOT _DEME_HIP_RESOURCE_DIR)
        find_program(_DEME_CLANG_FOR_HIP NAMES amdclang++ clang++ clang HINTS "${ROCM_PATH}/llvm/bin" "${ROCM_PATH}/bin")
        if(_DEME_CLANG_FOR_HIP)
            execute_process(
                COMMAND "${_DEME_CLANG_FOR_HIP}" --print-resource-dir
                OUTPUT_VARIABLE _DEME_HIP_RESOURCE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
        endif()
    endif()
    if(_DEME_HIP_RESOURCE_DIR AND EXISTS "${_DEME_HIP_RESOURCE_DIR}/include")
        list(APPEND _DEME_HIP_INC_DIRS "${_DEME_HIP_RESOURCE_DIR}/include")
    endif()

    if(UNIX AND EXISTS "/usr/include")
        list(APPEND _DEME_HIP_INC_DIRS "/usr/include")
    endif()
endif()

set(_DEME_GPU_JIT_INC_DIRS ${_DEME_CUDA_INC_DIRS} ${_DEME_HIP_INC_DIRS})
list(REMOVE_DUPLICATES _DEME_GPU_JIT_INC_DIRS)

# Only expose host-compiler include paths that are actually needed to compile DEME sources.
# In HIP mode, the Clang resource directory is for hipRTC/JIT, not for regular C++ files compiled by g++.
# Feeding that directory to host compilation breaks stddef/wchar processing on GCC.
set(_DEME_GPU_HOST_INC_DIRS ${_DEME_CUDA_INC_DIRS})
if(USE_HIP)
    if(ROCM_PATH AND EXISTS "${ROCM_PATH}/include")
        list(APPEND _DEME_GPU_HOST_INC_DIRS "${ROCM_PATH}/include")
    endif()
endif()
list(REMOVE_DUPLICATES _DEME_GPU_HOST_INC_DIRS)

set(DEME_GPU_TOOLKIT_HEADERS "")
foreach(_de_inc_dir IN LISTS _DEME_GPU_JIT_INC_DIRS)
    if(_de_inc_dir)
        if(DEME_GPU_TOOLKIT_HEADERS)
            string(APPEND DEME_GPU_TOOLKIT_HEADERS ";")
        endif()
        string(APPEND DEME_GPU_TOOLKIT_HEADERS "${_de_inc_dir}")
    endif()
endforeach()
set(DEME_PROJECT_SOURCE_DIR "${CMAKE_SOURCE_DIR}")

message(STATUS "${core_message} Generating API version header...")
configure_file(
	ApiVersion.h.in ApiVersion.h
	#	USE_SOURCE_PERMISSIONS ## This doesn't work in CMake <3.20
	@ONLY
)

if(USE_HIP)
    message(STATUS "${core_message} HIP mode: Skipping Jitify (will use hipRTC)")
    # Create empty jitify directory for HIP mode
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/src/jitify)
else()
    message(STATUS "${core_message} Extracting NVIDIA Jitify header...") 
    configure_file(
        ${NVIDIAJitifyPath}/jitify.hpp
        ${CMAKE_BINARY_DIR}/src/jitify/jitify.hpp
        COPYONLY
    )
endif()

add_library(core OBJECT)

target_include_directories(
	core
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeSource}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeGenerated}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	PUBLIC $<BUILD_INTERFACE:${_DEME_GPU_HOST_INC_DIRS}>
)

# Define DEME_USE_HIP for HIP builds to enable HIP code paths
if(USE_HIP)
    target_compile_definitions(core PUBLIC DEME_USE_HIP)
    target_compile_options(core PRIVATE -Wno-attributes)
endif()


if(USE_HIP)
    # hipCUB is header-only in our usage path; only ChPF needs an actual link dependency here.
    if(USE_CHPF)
        target_link_libraries(
            core
            INTERFACE ${ChPF_IMPORTED_NAME}
        )
    endif()
else()
    # CUDA mode
    if(USE_CHPF)
        target_link_libraries(
            core
            PUBLIC CUB::CUB
            INTERFACE ${ChPF_IMPORTED_NAME}
        )
    else()
        target_link_libraries(
            core
            PUBLIC CUB::CUB
        )
    endif()
endif()


set(core_headers
	${CMAKE_BINARY_DIR}/src/core/ApiVersion.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/BaseClasses.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/CudaAllocator.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ManagedMemory.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/JitHelper.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ThreadManager.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/Logger.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuManager.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/WavefrontMeshLoader.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/csv.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/Timer.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/DataMigrationHelper.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/DEMEPaths.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuRuntime.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/RuntimeData.h
)

set(core_sources
	${CMAKE_CURRENT_SOURCE_DIR}/DebugInfo.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/GpuManager.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/JitHelper.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/DEMEPaths.cpp
)

target_sources(
	core
	PUBLIC ${core_headers} ${core_utils_headers}
	PRIVATE ${core_sources}
)

set_target_properties(
	core PROPERTIES 
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${CXXSTD_SUPPORTED}
	PUBLIC_HEADER "${core_headers}"
)

# Install Core Headers
install(
	DIRECTORY ${ProjectIncludeSource}/core
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
		FILES_MATCHING 
			PATTERN "*.h"
			PATTERN "*.hpp"
)

# Install Generated Headers 
install(
	FILES 
		"${CMAKE_BINARY_DIR}/src/core/ApiVersion.h"
	DESTINATION
		${CMAKE_INSTALL_INCLUDEDIR}/core
)

# Install Third-party Headers 
if(NOT USE_HIP AND EXISTS "${NVIDIAJitifyPath}/jitify.hpp")
    install(
    	FILES 
    		"${NVIDIAJitifyPath}/jitify.hpp"
    	DESTINATION 
    		${CMAKE_INSTALL_INCLUDEDIR}/jitify
    )
endif()

# --------------------------------------------------------- #
# Runtime data path helper library 
# --------------------------------------------------------- #


message(STATUS "${core_message} Generating Runtime Data helper (build)...")
set(DEME_RUNTIME_DATA_DIRECTORY "${CMAKE_BINARY_DIR}")
set(DEME_RUNTIME_INCLUDE_DIRECTORY "${CMAKE_BINARY_DIR}")
configure_file(
	utils/RuntimeData.cpp.in utils/RuntimeDataBuild.cpp
	@ONLY
)

message(STATUS "${core_message} Generating Runtime Data helper (install)...")
set(DEME_RUNTIME_DATA_DIRECTORY "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/DEME")
set(DEME_RUNTIME_INCLUDE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
configure_file(
	utils/RuntimeData.cpp.in utils/RuntimeDataInstall.cpp
	@ONLY
)


# Build tree version

add_library(DEMERuntimeDataHelper SHARED)

# Windows specific: Define DLL export macros
if(WIN32)
    target_compile_definitions(DEMERuntimeDataHelper PRIVATE DEMERuntimeDataHelper_EXPORTS)
endif()


target_sources(DEMERuntimeDataHelper
	PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils/RuntimeData.h> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/core/utils/RuntimeData.h>
	PRIVATE "${CMAKE_BINARY_DIR}/src/core/utils/RuntimeDataBuild.cpp" 
)

target_include_directories(
	DEMERuntimeDataHelper
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeSource}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(
	DEMERuntimeDataHelper PROPERTIES
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD "${CXXSTD_SUPPORTED}"
)

# Install tree version

##
## This MUST be installed after the Build tree version as it overwrites the .so file 
##

add_library(DEMERuntimeDataHelper_install SHARED)

if(WIN32)
    target_compile_definitions(DEMERuntimeDataHelper_install PRIVATE DEMERuntimeDataHelper_EXPORTS)
endif()

target_sources(DEMERuntimeDataHelper_install
	PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils/RuntimeData.h> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/core/utils/RuntimeData.h>
	PRIVATE "${CMAKE_BINARY_DIR}/src/core/utils/RuntimeDataInstall.cpp" 
)


if(WIN32)
    # Windows-specific linker options
    # You can add options here if needed. Often, you don't need to specify the DLL name as in Unix.
    target_link_libraries(DEMERuntimeDataHelper_install PRIVATE DEMERuntimeDataHelper)
else()
    # Unix-like OS specific linker options
    target_link_options(DEMERuntimeDataHelper_install
        PUBLIC "LINKER:-soname,libDEMERuntimeDataHelper.so"
    )
endif()



target_include_directories(
	DEMERuntimeDataHelper_install
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeSource}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(
	DEMERuntimeDataHelper_install PROPERTIES
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${CXXSTD_SUPPORTED}
	LIBRARY_OUTPUT_NAME "DEMERuntimeDataHelper"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib_install"
)

# Make sure this target is generated **AFTER** the build-tree version 
add_dependencies(DEMERuntimeDataHelper_install DEMERuntimeDataHelper)

