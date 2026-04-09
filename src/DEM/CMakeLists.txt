# Copyright (c) 2021, SBEL GPU Development Team
# Copyright (c) 2021, University of Wisconsin - Madison
# 
#	SPDX-License-Identifier: BSD-3-Clause

message(STATUS "==== Building DEM Components ====")

add_library(DEM OBJECT)

target_include_directories(
	DEM
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeSource}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	PUBLIC $<BUILD_INTERFACE:${ProjectIncludeGenerated}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Define DEME_USE_HIP for HIP builds to enable HIP code paths
if(USE_HIP)
    target_compile_definitions(DEM PUBLIC DEME_USE_HIP)
    target_compile_options(DEM PRIVATE -Wno-attributes)
endif()

# For HIP builds, add ROCm include paths
if(USE_HIP)
    if(ROCM_PATH)
        target_include_directories(DEM PUBLIC ${ROCM_PATH}/include)
    endif()
endif()

if(DEME_ENABLE_NVTX)
	target_compile_definitions(DEM PUBLIC DEME_ENABLE_NVTX)
endif()

if(USE_HIP)
    # hipCUB is header-only in this target; only ChPF contributes a link dependency.
    if(USE_CHPF)
        target_link_libraries(
            DEM
            PUBLIC ${ChPF_IMPORTED_NAME}
        )
    endif()
else()
    # CUDA mode
    if(USE_CHPF)
        target_link_libraries(
            DEM
            PUBLIC CUB::CUB
            PUBLIC ${ChPF_IMPORTED_NAME}
        )
    else()
        target_link_libraries(
            DEM
            PUBLIC CUB::CUB
        )
    endif()
endif()

# if(WIN32)
# target_compile_options(DEM PRIVATE /GR)
# endif()

set(DEM_headers
	${CMAKE_CURRENT_SOURCE_DIR}/kT.h
	${CMAKE_CURRENT_SOURCE_DIR}/dT.h
	${CMAKE_CURRENT_SOURCE_DIR}/API.h
	${CMAKE_CURRENT_SOURCE_DIR}/Defines.h
	${CMAKE_CURRENT_SOURCE_DIR}/Structs.h
	${CMAKE_CURRENT_SOURCE_DIR}/Models.h
	${CMAKE_CURRENT_SOURCE_DIR}/VariableTypes.h
	${CMAKE_CURRENT_SOURCE_DIR}/BdrsAndObjs.h
	${CMAKE_CURRENT_SOURCE_DIR}/utils/HostSideHelpers.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/utils/Samplers.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/AuxClasses.h
)

set(DEM_sources
	${CMAKE_CURRENT_SOURCE_DIR}/kT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/dT.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/APIPublic.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/APIPrivate.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/MeshUtils.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/AuxClasses.cpp
)

target_sources(
	DEM
	PUBLIC ${DEM_headers}
	PRIVATE ${DEM_sources}
)

set_target_properties(
	DEM PROPERTIES
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${CXXSTD_SUPPORTED}
	PUBLIC_HEADER "${DEM_headers}"
)

#foreach(header ${DEM_headers})
#	set_target_properties(DEM PROPERTIES PUBLIC_HEADER ${header})
#endforeach()

install(
	TARGETS DEM
	PUBLIC_HEADER
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/DEM 
)

install(
	DIRECTORY utils
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/DEM
	FILES_MATCHING
		PATTERN "utils/*.h"
		PATTERN "utils/*.hpp"
)

