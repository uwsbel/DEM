#pragma once

#include <limits>

#include <core/utils/GpuRuntime.hpp>
#include <core/utils/Logger.hpp>

#if defined(DEME_GPU_BACKEND_HIP) || defined(DEME_USE_HIP)
    #include <hipcub/hipcub.hpp>
    namespace demecub = hipcub;
#else
    #include <cub/cub.cuh>
    namespace demecub = cub;
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
    #define DEME_CUB_HD __host__ __device__
    #define DEME_CUB_HDINLINE __host__ __device__ __forceinline__
#else
    #define DEME_CUB_HD
    #define DEME_CUB_HDINLINE inline
#endif

namespace deme {

inline int cubCount(size_t n) {
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        DEME_ERROR("CUB/hipCUB item count exceeds int range: %zu", n);
    }
    return static_cast<int>(n);
}

template <class T>
struct CubOpAdd {
    DEME_CUB_HDINLINE T operator()(const T& a, const T& b) const { return a + b; }
};

template <class T>
struct CubOpMin {
    DEME_CUB_HDINLINE T operator()(const T& a, const T& b) const { return (b < a) ? b : a; }
};

template <class T>
struct CubOpMax {
    DEME_CUB_HDINLINE T operator()(const T& a, const T& b) const { return (b > a) ? b : a; }
};

}  // namespace deme
