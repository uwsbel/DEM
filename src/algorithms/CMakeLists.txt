# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# 
#	SPDX-License-Identifier: BSD-3-Clause

message(STATUS "==== Configuring GPU Algorithm Components ====")
message(STATUS "ProjectIncludeSource: ${ProjectIncludeSource}")
message(STATUS "ProjectIncludeGenerated: ${ProjectIncludeGenerated}")

set (algorithms_message "[algorithms]")

add_library(algorithms OBJECT)

### HOST HEADERS ONLY (.h, .hpp) ###
set(algorithms_interface
	${CMAKE_CURRENT_SOURCE_DIR}/DEMStaticDeviceSubroutines.h
)

### INTERNAL HEADERS ONLY (.h, .hpp, or .cuh) ###
set(algorithms_headers
	${CMAKE_CURRENT_SOURCE_DIR}/DEMStaticDeviceUtilities.cuh
	${CMAKE_CURRENT_SOURCE_DIR}/DEMContactDetectionKernels.cuh
)

set(algorithms_sources
	${CMAKE_CURRENT_SOURCE_DIR}/DEMCubInstantiations.cu
	${CMAKE_CURRENT_SOURCE_DIR}/DEMContactDetection.cu
	${CMAKE_CURRENT_SOURCE_DIR}/DEMCubWrappers.cu 
	${CMAKE_CURRENT_SOURCE_DIR}/DEMDynamicMisc.cu
	${CMAKE_CURRENT_SOURCE_DIR}/DEMMiscKernels.cu
)

set(_DEME_ALGO_BUILD_INCLUDES
    ${ProjectIncludeSource}
    ${ProjectIncludeGenerated}
)
if(USE_HIP AND ROCM_PATH AND EXISTS "${ROCM_PATH}/include")
    list(APPEND _DEME_ALGO_BUILD_INCLUDES "${ROCM_PATH}/include")
endif()
if(USE_HIP AND hipCUB_PATH)
    list(APPEND _DEME_ALGO_BUILD_INCLUDES "${hipCUB_PATH}")
endif()

# Use include_directories directly instead of generator expression to avoid path normalization issues
include_directories(${_DEME_ALGO_BUILD_INCLUDES})

if(USE_HIP)
    target_compile_definitions(algorithms PUBLIC DEME_USE_HIP)
    target_compile_options(algorithms PRIVATE -Wno-attributes)
    if(hipCUB_PATH)
        target_compile_definitions(algorithms PUBLIC DEME_USE_HIPCUB)
    endif()
else()
    target_link_libraries(
        algorithms
        PUBLIC CUB::CUB
    )
endif()

set(_DEME_HIP_ARCH "")
if(USE_HIP)
    if(DEFINED CMAKE_HIP_ARCHITECTURES AND NOT "${CMAKE_HIP_ARCHITECTURES}" STREQUAL "")
        list(GET CMAKE_HIP_ARCHITECTURES 0 _DEME_HIP_ARCH)
    elseif(DEFINED HIP_ARCH AND NOT "${HIP_ARCH}" STREQUAL "")
        set(_DEME_HIP_ARCH "${HIP_ARCH}")
    elseif(DEFINED ENV{GPU_TARGETS} AND NOT "$ENV{GPU_TARGETS}" STREQUAL "")
        string(REPLACE "," ";" _de_gpu_targets "$ENV{GPU_TARGETS}")
        list(GET _de_gpu_targets 0 _DEME_HIP_ARCH)
    endif()
    if(_DEME_HIP_ARCH)
        message(STATUS "${algorithms_message} HIP architecture hint: ${_DEME_HIP_ARCH}")
    else()
        message(STATUS "${algorithms_message} HIP architecture hint: auto")
    endif()
endif()

set(_DEME_USE_CMAKE_HIP FALSE)
if(USE_HIP)
    if(CMAKE_HIP_COMPILER)
        set(_DEME_USE_CMAKE_HIP TRUE)
        message(STATUS "${algorithms_message} Using CMake HIP compiler: ${CMAKE_HIP_COMPILER}")
    endif()
endif()

if(USE_HIP AND _DEME_USE_CMAKE_HIP)
    target_sources(
        algorithms
        PUBLIC ${algorithms_interface}
        INTERFACE ${algorithms_headers}
        PRIVATE ${algorithms_sources}
    )
    set_source_files_properties(${algorithms_sources} PROPERTIES LANGUAGE HIP)
else()
    if(USE_HIP)
        if(NOT HIPCC_EXECUTABLE)
            find_program(HIPCC_EXECUTABLE hipcc HINTS "${ROCM_PATH}/bin" "$ENV{ROCM_PATH}/bin" "/opt/rocm/bin" "/usr/bin")
        endif()
        if(NOT HIPCC_EXECUTABLE)
            message(FATAL_ERROR "hipcc not found for HIP builds")
        endif()
        message(STATUS "${algorithms_message} Using hipcc fallback: ${HIPCC_EXECUTABLE}")

        set(algorithms_objects "")
        set(_DEME_HIP_FLAGS "-fPIC" "-std=c++${CXXSTD_SUPPORTED}" "-DDEME_USE_HIP" "-D__HIP_PLATFORM_AMD__")
        if(_DEME_HIP_ARCH)
            list(APPEND _DEME_HIP_FLAGS "--offload-arch=${_DEME_HIP_ARCH}")
        endif()
        foreach(_de_inc IN LISTS _DEME_ALGO_BUILD_INCLUDES)
            list(APPEND _DEME_HIP_FLAGS "-I${_de_inc}")
        endforeach()

        foreach(src_file ${algorithms_sources})
            get_filename_component(src_basename ${src_file} NAME_WE)
            set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/${src_basename}.o")
            add_custom_command(
                OUTPUT ${obj_file}
                COMMAND ${HIPCC_EXECUTABLE} -c ${src_file} -o ${obj_file} ${_DEME_HIP_FLAGS}
                DEPENDS ${src_file} ${algorithms_interface} ${algorithms_headers}
                COMMENT "Compiling ${src_file} with hipcc"
                VERBATIM
            )
            list(APPEND algorithms_objects ${obj_file})
        endforeach()

        target_sources(
            algorithms
            PUBLIC ${algorithms_interface}
            INTERFACE ${algorithms_headers}
            PRIVATE ${algorithms_objects}
        )
    else()
        target_sources(
            algorithms
            PUBLIC ${algorithms_interface}
            INTERFACE ${algorithms_headers}
            PRIVATE ${algorithms_sources}
        )
    endif()
endif()

if(USE_HIP)
    set_target_properties(
        algorithms PROPERTIES 
        POSITION_INDEPENDENT_CODE True
        CXX_STANDARD ${CXXSTD_SUPPORTED}
        PUBLIC_HEADER "${algorithms_interface}"
        LINKER_LANGUAGE CXX
    )
    if(_DEME_USE_CMAKE_HIP AND _DEME_HIP_ARCH)
        set_target_properties(algorithms PROPERTIES HIP_ARCHITECTURES "${_DEME_HIP_ARCH}")
    endif()
else()
    set(virtual_architectures "")
    if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/CudaSupportedArchitectures.cmake")
        include(${CMAKE_SOURCE_DIR}/cmake/CudaSupportedArchitectures.cmake)
        cuda_supported_architectures()
        foreach(arch ${CUDASUP_ARCHITECTURES})
            list(APPEND virtual_architectures "${arch}-virtual")
        endforeach()
    endif()

    set_target_properties(
        algorithms PROPERTIES 
        POSITION_INDEPENDENT_CODE True
        CXX_STANDARD ${CXXSTD_SUPPORTED}
        PUBLIC_HEADER "${algorithms_interface}"
        CUDA_ARCHITECTURES "${virtual_architectures}"
        LINKER_LANGUAGE CUDA
    )
endif()

install(
	TARGETS algorithms
	PUBLIC_HEADER
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/algorithms
)
