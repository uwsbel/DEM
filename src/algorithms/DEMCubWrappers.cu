//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include "DEMCubCompact.hpp"
#include <DEM/Defines.h>
#include <DEM/Structs.h>

#if defined(DEME_USE_HIP) || !defined(CUDART_VERSION) || CUDART_VERSION < 13000
    #define CUB_SUM_OP(T) demecub::Sum {}
#else
    #define CUB_SUM_OP(T) cuda::std::plus<T> {}
#endif

namespace deme {

// Functor type for selecting values equal to some criterion.
template <typename T>
struct CubEqualTo {
    T compare;
    DEME_CUB_HDINLINE explicit CubEqualTo(T compare) : compare(compare) {}
    DEME_CUB_HDINLINE bool operator()(const T& a) const { return (a == compare); }
};

struct CubFloat3Add {
    DEME_CUB_HDINLINE float3 operator()(const float3& a, const float3& b) const {
        return ::make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
};

// Custom functor for finding the "max negative" value:
// - Among negative values, find the one closest to zero (largest negative = smallest absolute value)
// - Positive values are treated as worse candidates than negative values
template <typename T>
struct CubOpMaxNegative {
    DEME_CUB_HDINLINE T operator()(const T& a, const T& b) const {
        if (a < 0 && b < 0) {
            return (b > a) ? b : a;
        }
        if (a < 0) {
            return a;
        }
        if (b < 0) {
            return b;
        }
        return (a < b) ? a : b;
    }
};

template <typename T1, typename T2>
inline void cubDEMSelectFlagged(T1* d_in,
                                T1* d_out,
                                T2* d_flags,
                                size_t* d_num_out,
                                size_t n,
                                cudaStream_t& this_stream,
                                DEMSolverScratchData& scratchPad) {
    const int num_items = cubCount(n);
    size_t cub_scratch_bytes = 0;
    DEME_GPU_CALL(demecub::DeviceSelect::Flagged(nullptr, cub_scratch_bytes, d_in, d_flags, d_out, d_num_out,
                                                 num_items, this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    DEME_GPU_CALL(demecub::DeviceSelect::Flagged(d_scratch_space, cub_scratch_bytes, d_in, d_flags, d_out,
                                                 d_num_out, num_items, this_stream));
}

template <typename T1, typename T2>
inline void cubDEMPrefixScan(T1* d_in,
                             T2* d_out,
                             size_t n,
                             cudaStream_t& this_stream,
                             DEMSolverScratchData& scratchPad) {
    const int num_items = cubCount(n);
    size_t cub_scratch_bytes = 0;
    DEME_GPU_CALL(demecub::DeviceScan::ExclusiveScan(nullptr, cub_scratch_bytes, d_in, d_out, CUB_SUM_OP(T2),
                                                     (T2)0, num_items, this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    DEME_GPU_CALL(demecub::DeviceScan::ExclusiveScan(d_scratch_space, cub_scratch_bytes, d_in, d_out,
                                                     CUB_SUM_OP(T2), (T2)0, num_items, this_stream));
}

template <typename T1, typename T2>
inline void cubDEMInclusiveScan(T1* d_in,
                                T2* d_out,
                                size_t n,
                                cudaStream_t& this_stream,
                                DEMSolverScratchData& scratchPad) {
    const int num_items = cubCount(n);
    size_t cub_scratch_bytes = 0;
    DEME_GPU_CALL(demecub::DeviceScan::InclusiveScan(nullptr, cub_scratch_bytes, d_in, d_out, CUB_SUM_OP(T2),
                                                     num_items, this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    DEME_GPU_CALL(demecub::DeviceScan::InclusiveScan(d_scratch_space, cub_scratch_bytes, d_in, d_out,
                                                     CUB_SUM_OP(T2), num_items, this_stream));
}

template <typename T1>
inline void cubDEMSortKeys(T1* d_keys_in,
                           T1* d_keys_out,
                           size_t n,
                           cudaStream_t& this_stream,
                           DEMSolverScratchData& scratchPad) {
    const int num_items = cubCount(n);
    size_t cub_scratch_bytes = 0;
    DEME_GPU_CALL(demecub::DeviceRadixSort::SortKeys(nullptr, cub_scratch_bytes, d_keys_in, d_keys_out,
                                                     num_items, 0,
                                                     static_cast<int>(sizeof(T1) * DEME_BITS_PER_BYTE),
                                                     this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    DEME_GPU_CALL(demecub::DeviceRadixSort::SortKeys(d_scratch_space, cub_scratch_bytes, d_keys_in, d_keys_out,
                                                     num_items, 0,
                                                     static_cast<int>(sizeof(T1) * DEME_BITS_PER_BYTE),
                                                     this_stream));
}

template <typename T1, typename T2>
inline void cubDEMSortByKeys(T1* d_keys_in,
                             T1* d_keys_out,
                             T2* d_vals_in,
                             T2* d_vals_out,
                             size_t n,
                             cudaStream_t& this_stream,
                             DEMSolverScratchData& scratchPad) {
    const int num_items = cubCount(n);
    size_t cub_scratch_bytes = 0;
    DEME_GPU_CALL(demecub::DeviceRadixSort::SortPairs(nullptr, cub_scratch_bytes, d_keys_in, d_keys_out,
                                                      d_vals_in, d_vals_out, num_items, 0,
                                                      static_cast<int>(sizeof(T1) * DEME_BITS_PER_BYTE),
                                                      this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    DEME_GPU_CALL(demecub::DeviceRadixSort::SortPairs(d_scratch_space, cub_scratch_bytes, d_keys_in, d_keys_out,
                                                      d_vals_in, d_vals_out, num_items, 0,
                                                      static_cast<int>(sizeof(T1) * DEME_BITS_PER_BYTE),
                                                      this_stream));
}

template <typename T1>
inline void cubDEMUnique(T1* d_in,
                         T1* d_out,
                         size_t* d_num_out,
                         size_t n,
                         cudaStream_t& this_stream,
                         DEMSolverScratchData& scratchPad) {
    const int num_items = cubCount(n);
    size_t cub_scratch_bytes = 0;
    DEME_GPU_CALL(demecub::DeviceSelect::Unique(nullptr, cub_scratch_bytes, d_in, d_out, d_num_out, num_items,
                                                this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    DEME_GPU_CALL(demecub::DeviceSelect::Unique(d_scratch_space, cub_scratch_bytes, d_in, d_out, d_num_out,
                                                num_items, this_stream));
}

template <typename T1, typename T2>
inline void cubDEMRunLengthEncode(T1* d_in,
                                  T1* d_unique_out,
                                  T2* d_counts_out,
                                  size_t* d_num_out,
                                  size_t n,
                                  cudaStream_t& this_stream,
                                  DEMSolverScratchData& scratchPad) {
    const int num_items = cubCount(n);
    size_t cub_scratch_bytes = 0;
    DEME_GPU_CALL(demecub::DeviceRunLengthEncode::Encode(nullptr, cub_scratch_bytes, d_in, d_unique_out,
                                                         d_counts_out, d_num_out, num_items, this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    DEME_GPU_CALL(demecub::DeviceRunLengthEncode::Encode(d_scratch_space, cub_scratch_bytes, d_in,
                                                         d_unique_out, d_counts_out, d_num_out, num_items,
                                                         this_stream));
}

template <typename T1, typename T2, typename T3>
inline void cubDEMReduceByKeys(T1* d_keys_in,
                               T1* d_unique_out,
                               T2* d_vals_in,
                               T2* d_aggregates_out,
                               size_t* d_num_out,
                               T3& reduce_op,
                               size_t n,
                               cudaStream_t& this_stream,
                               DEMSolverScratchData& scratchPad) {
    const int num_items = cubCount(n);
    size_t cub_scratch_bytes = 0;
    DEME_GPU_CALL(demecub::DeviceReduce::ReduceByKey(nullptr, cub_scratch_bytes, d_keys_in, d_unique_out,
                                                     d_vals_in, d_aggregates_out, d_num_out, reduce_op,
                                                     num_items, this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    DEME_GPU_CALL(demecub::DeviceReduce::ReduceByKey(d_scratch_space, cub_scratch_bytes, d_keys_in,
                                                     d_unique_out, d_vals_in, d_aggregates_out, d_num_out,
                                                     reduce_op, num_items, this_stream));
}

template <typename T1, typename T2>
void cubDEMSum(T1* d_in, T2* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad) {
    const int num_items = cubCount(n);
    size_t cub_scratch_bytes = 0;
    DEME_GPU_CALL(demecub::DeviceReduce::Reduce(nullptr, cub_scratch_bytes, d_in, d_out, num_items,
                                                CUB_SUM_OP(T2), (T2)0, this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    DEME_GPU_CALL(demecub::DeviceReduce::Reduce(d_scratch_space, cub_scratch_bytes, d_in, d_out, num_items,
                                                CUB_SUM_OP(T2), (T2)0, this_stream));
}

template <typename T1>
void cubDEMMax(T1* d_in, T1* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad) {
    const int num_items = cubCount(n);
    size_t cub_scratch_bytes = 0;
    DEME_GPU_CALL(demecub::DeviceReduce::Max(nullptr, cub_scratch_bytes, d_in, d_out, num_items, this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    DEME_GPU_CALL(demecub::DeviceReduce::Max(d_scratch_space, cub_scratch_bytes, d_in, d_out, num_items,
                                             this_stream));
}

template <typename T1>
void cubDEMMin(T1* d_in, T1* d_out, size_t n, cudaStream_t& this_stream, DEMSolverScratchData& scratchPad) {
    const int num_items = cubCount(n);
    size_t cub_scratch_bytes = 0;
    DEME_GPU_CALL(demecub::DeviceReduce::Min(nullptr, cub_scratch_bytes, d_in, d_out, num_items, this_stream));
    void* d_scratch_space = (void*)scratchPad.allocateScratchSpace(cub_scratch_bytes);
    DEME_GPU_CALL(demecub::DeviceReduce::Min(d_scratch_space, cub_scratch_bytes, d_in, d_out, num_items,
                                             this_stream));
}

}  // namespace deme
