// Fixed-capacity analytical constants; values are uploaded at runtime.
extern "C" __device__ __constant__ __attribute__((used)) deme::objType_t objType[512];
extern "C" __device__ __constant__ __attribute__((used)) deme::bodyID_t objOwner[512];
extern "C" __device__ __constant__ __attribute__((used)) float objNormal[512];
extern "C" __device__ __constant__ __attribute__((used)) deme::materialsOffset_t objMaterial[512];
extern "C" __device__ __constant__ __attribute__((used)) float objRelPosX[512];
extern "C" __device__ __constant__ __attribute__((used)) float objRelPosY[512];
extern "C" __device__ __constant__ __attribute__((used)) float objRelPosZ[512];
extern "C" __device__ __constant__ __attribute__((used)) float objRotX[512];
extern "C" __device__ __constant__ __attribute__((used)) float objRotY[512];
extern "C" __device__ __constant__ __attribute__((used)) float objRotZ[512];
extern "C" __device__ __constant__ __attribute__((used)) float objSize1[512];
extern "C" __device__ __constant__ __attribute__((used)) float objSize2[512];
extern "C" __device__ __constant__ __attribute__((used)) float objSize3[512];
extern "C" __device__ __constant__ __attribute__((used)) float objMass[512];
