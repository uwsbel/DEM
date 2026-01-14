// Mass/MOI data are uploaded at runtime (one copy per program).
extern "C" __device__ __constant__ __attribute__((used)) float moiX[1024];
extern "C" __device__ __constant__ __attribute__((used)) float moiY[1024];
extern "C" __device__ __constant__ __attribute__((used)) float moiZ[1024];
