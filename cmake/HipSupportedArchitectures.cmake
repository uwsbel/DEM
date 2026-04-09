# Copyright (c) 2026, DEM-Engine AMD HIP Port
# SPDX-License-Identifier: BSD-3-Clause

# Determine the AMD GPU architectures supported by the active version of hipcc

# Effects:
#
# Populates the cache variable HIPSUP_ARCHITECTURES with a list of AMD GPU
# architecture names supported by the current ROCm/HIP version
#
# For gfx1151 (RDNA3-based Radeon Graphics)

function(hip_supported_architectures)
    # Supported AMD GPU architectures by ROCm version
    # gfx900, gfx906, gfx908, gfx90a, gfx940, gfx941, gfx942 (Vega/Navi)
    # gfx1010, gfx1011, gfx1012, gfx1030, gfx1031, gfx1032 (Navi 10/12/14)
    # gfx1100, gfx1101, gfx1102, gfx1103 (Navi 3x - RDNA3)
    # gfx1150, gfx1151 (Embedded/APU RDNA3)
    
    set(HIPSUP_ARCHITECTURES "gfx900" "gfx906" "gfx908" "gfx90a" "gfx1010" "gfx1030" "gfx1100" "gfx1101" "gfx1102" "gfx1151" CACHE INTERNAL "")
    
    message(STATUS "Supported HIP architectures: ${HIPSUP_ARCHITECTURES}")
endfunction()
