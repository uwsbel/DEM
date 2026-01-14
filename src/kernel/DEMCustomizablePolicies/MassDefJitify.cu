// Mass/MOI data are uploaded at runtime (one copy per program).
extern "C" __device__ __constant__ __attribute__((used)) float MassProperties[1024];
