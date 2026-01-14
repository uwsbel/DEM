// Definition for static compilation of sim param constants.
#include "DEM/Defines.h"

extern "C" {
__device__ __constant__ __attribute__((used, visibility("default"))) deme::DEMSimParamsConst DEME_SimParamsConst;
}

extern "C" void DEME_touchSimParamsConst() {}
