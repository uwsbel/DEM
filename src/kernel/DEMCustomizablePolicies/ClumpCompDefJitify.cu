// Fixed-capacity clump component constants; values are uploaded at runtime.
#include <DEM/Defines.h>
extern "C" __device__ __constant__ __attribute__((used)) float Radii[deme::DEME_CLUMP_COMPONENT_CONST_CAPACITY];
extern "C" __device__ __constant__ __attribute__((used)) float CDRelPosX[deme::DEME_CLUMP_COMPONENT_CONST_CAPACITY];
extern "C" __device__ __constant__ __attribute__((used)) float CDRelPosY[deme::DEME_CLUMP_COMPONENT_CONST_CAPACITY];
extern "C" __device__ __constant__ __attribute__((used)) float CDRelPosZ[deme::DEME_CLUMP_COMPONENT_CONST_CAPACITY];
