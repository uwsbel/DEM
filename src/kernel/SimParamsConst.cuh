//  Copyright (c) 2024, SBEL GPU Development Team
//
//  SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_SIM_PARAMS_CONST_CUH
#define DEME_SIM_PARAMS_CONST_CUH

#include <DEM/Defines.h>

// Constant-memory copy of frequently used static sim params (device only).
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    #if defined(__CUDACC_RTC__)
        #define DEME_SIM_PARAMS_CONST_DECL __device__ __constant__ __attribute__((used))
    #else
        #define DEME_SIM_PARAMS_CONST_DECL __device__ __constant__ __attribute__((used, visibility("default")))
    #endif
    #if defined(__CUDACC__) && !defined(__CUDACC_RTC__)
        #pragma diag_suppress 20044
    #endif
    extern "C" {
    #if defined(DEME_DEFINE_SIM_PARAMS_CONST)
    DEME_SIM_PARAMS_CONST_DECL deme::DEMSimParamsConst DEME_SimParamsConst;
    #else
    extern DEME_SIM_PARAMS_CONST_DECL deme::DEMSimParamsConst DEME_SimParamsConst;
    #endif
    }
    #if defined(__CUDACC__) && !defined(__CUDACC_RTC__)
        #pragma diag_default 20044
    #endif
    #undef DEME_SIM_PARAMS_CONST_DECL
#endif

#if defined(__cplusplus) && !defined(__CUDACC_RTC__)
extern "C" void DEME_touchSimParamsConst();
#endif

#endif
