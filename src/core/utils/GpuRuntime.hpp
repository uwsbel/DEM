// Copyright (c) 2026, DEM-Engine AMD HIP Port
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_GPU_RUNTIME_HPP
#define DEME_GPU_RUNTIME_HPP

#include <cstddef>

#if defined(DEME_USE_HIP)
    #ifndef __HIP_PLATFORM_AMD__
        #define __HIP_PLATFORM_AMD__
    #endif
    #if defined(__HIPCC_RTC__)
        // hipRTC device compilation exposes device builtins, but not the full host runtime API.
        // Provide lightweight CUDA-compatibility aliases so shared headers still parse.
        using cudaStream_t = void*;
        using cudaEvent_t = void*;
        using cudaError_t = int;
        struct cudaDeviceProp {
            char gcnArchName[256];
        };
        struct cudaPointerAttributes {
            const void* hostPointer;
            const void* devicePointer;
            int type;
            int device;
            int isManaged;
        };
        using cudaMemcpyKind = int;
        using cudaMemoryAdvise = int;
        struct cudaMemLocation {
            int type;
            int id;
        };
        using cudaMemLocationType = int;
        using cudaMemoryType = int;
        using cudaStreamCallback_t = void (*)(cudaStream_t, cudaError_t, void*);

        static constexpr cudaError_t cudaSuccess = 0;
        static constexpr cudaError_t cudaErrorMemoryAllocation = 2;
        static constexpr cudaError_t cudaErrorNotReady = 34;
        static constexpr cudaError_t cudaErrorNotSupported = 801;
        static constexpr cudaError_t cudaErrorPeerAccessAlreadyEnabled = 704;
        static constexpr unsigned int cudaEventDefault = 0u;
        static constexpr unsigned int cudaEventDisableTiming = 2u;
        static constexpr unsigned int cudaHostAllocDefault = 0u;
        static constexpr unsigned int cudaMemAttachGlobal = 1u;
        static constexpr cudaMemcpyKind cudaMemcpyHostToDevice = 1;
        static constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = 2;
        static constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice = 3;
        static constexpr cudaMemLocationType cudaMemLocationTypeDevice = 0;
        static constexpr cudaMemLocationType cudaMemLocationTypeHost = 1;
        static constexpr cudaMemoryAdvise cudaMemAdviseSetReadMostly = 1;
        static constexpr cudaMemoryAdvise cudaMemAdviseSetPreferredLocation = 3;
        static constexpr cudaMemoryAdvise cudaMemAdviseSetAccessedBy = 5;
        static constexpr cudaMemoryAdvise cudaMemAdviseUnsetReadMostly = 2;
        static constexpr cudaMemoryAdvise cudaMemAdviseUnsetPreferredLocation = 4;
        static constexpr cudaMemoryAdvise cudaMemAdviseUnsetAccessedBy = 6;
        static constexpr cudaMemoryType cudaMemoryTypeHost = 1;
        static constexpr cudaMemoryType cudaMemoryTypeDevice = 2;
        static constexpr cudaMemoryType cudaMemoryTypeManaged = 3;
        static constexpr cudaMemoryType cudaMemoryTypeUnregistered = 0;
    #else
        #if defined(__clang__)
            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wattributes"
        #endif
        #include <hip/hip_runtime.h>
        #include <hip/hip_runtime_api.h>
        #if defined(__clang__)
            #pragma clang diagnostic pop
        #endif

using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaError_t = hipError_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaPointerAttributes = hipPointerAttribute_t;
using cudaMemcpyKind = hipMemcpyKind;
using cudaMemoryAdvise = hipMemoryAdvise;
using cudaMemLocation = hipMemLocation;
using cudaMemLocationType = hipMemLocationType;
using cudaMemoryType = hipMemoryType;
using cudaStreamCallback_t = hipStreamCallback_t;

static constexpr cudaError_t cudaSuccess = hipSuccess;
static constexpr cudaError_t cudaErrorMemoryAllocation = hipErrorMemoryAllocation;
static constexpr cudaError_t cudaErrorNotReady = hipErrorNotReady;
static constexpr cudaError_t cudaErrorNotSupported = hipErrorNotSupported;
static constexpr cudaError_t cudaErrorPeerAccessAlreadyEnabled = hipErrorPeerAccessAlreadyEnabled;
static constexpr unsigned int cudaEventDefault = hipEventDefault;
static constexpr unsigned int cudaEventDisableTiming = hipEventDisableTiming;
static constexpr unsigned int cudaHostAllocDefault = hipHostMallocDefault;
static constexpr unsigned int cudaMemAttachGlobal = hipMemAttachGlobal;
static constexpr cudaMemcpyKind cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
static constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
static constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
static constexpr cudaMemLocationType cudaMemLocationTypeDevice = hipMemLocationTypeDevice;
static constexpr cudaMemLocationType cudaMemLocationTypeHost = hipMemLocationTypeHost;
static constexpr cudaMemoryAdvise cudaMemAdviseSetReadMostly = hipMemAdviseSetReadMostly;
static constexpr cudaMemoryAdvise cudaMemAdviseSetPreferredLocation = hipMemAdviseSetPreferredLocation;
static constexpr cudaMemoryAdvise cudaMemAdviseSetAccessedBy = hipMemAdviseSetAccessedBy;
static constexpr cudaMemoryAdvise cudaMemAdviseUnsetReadMostly = hipMemAdviseUnsetReadMostly;
static constexpr cudaMemoryAdvise cudaMemAdviseUnsetPreferredLocation = hipMemAdviseUnsetPreferredLocation;
static constexpr cudaMemoryAdvise cudaMemAdviseUnsetAccessedBy = hipMemAdviseUnsetAccessedBy;
static constexpr cudaMemoryType cudaMemoryTypeHost = hipMemoryTypeHost;
static constexpr cudaMemoryType cudaMemoryTypeDevice = hipMemoryTypeDevice;
static constexpr cudaMemoryType cudaMemoryTypeManaged = hipMemoryTypeManaged;
static constexpr cudaMemoryType cudaMemoryTypeUnregistered = hipMemoryTypeUnregistered;

inline const char* cudaGetErrorString(cudaError_t err) { return hipGetErrorString(err); }
inline cudaError_t cudaGetLastError() { return hipGetLastError(); }
inline cudaError_t cudaPeekAtLastError() { return hipPeekAtLastError(); }
inline cudaError_t cudaMalloc(void** ptr, size_t bytes) { return hipMalloc(ptr, bytes); }
inline cudaError_t cudaFree(void* ptr) { return hipFree(ptr); }
inline cudaError_t cudaMallocHost(void** ptr, size_t bytes) {
    return hipHostMalloc(ptr, bytes, hipHostMallocDefault);
}
inline cudaError_t cudaFreeHost(void* ptr) { return hipHostFree(ptr); }
inline cudaError_t cudaHostAlloc(void** ptr, size_t bytes, unsigned int flags) {
    return hipHostMalloc(ptr, bytes, flags);
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    return hipMemcpy(dst, src, bytes, kind);
}
inline cudaError_t cudaMemcpyAsync(void* dst,
                                   const void* src,
                                   size_t bytes,
                                   cudaMemcpyKind kind,
                                   cudaStream_t stream = nullptr) {
    return hipMemcpyAsync(dst, src, bytes, kind, stream);
}
inline cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t bytes) {
    return hipMemcpyPeer(dst, dstDevice, src, srcDevice, bytes);
}
inline cudaError_t cudaMemcpyPeerAsync(void* dst,
                                       int dstDevice,
                                       const void* src,
                                       int srcDevice,
                                       size_t bytes,
                                       cudaStream_t stream = nullptr) {
    return hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, bytes, stream);
}
inline cudaError_t cudaMemset(void* dst, int value, size_t bytes) { return hipMemset(dst, value, bytes); }
inline cudaError_t cudaMemsetAsync(void* dst,
                                   int value,
                                   size_t bytes,
                                   cudaStream_t stream = nullptr) {
    return hipMemsetAsync(dst, value, bytes, stream);
}
inline cudaError_t cudaGetDevice(int* device) { return hipGetDevice(device); }
inline cudaError_t cudaSetDevice(int device) { return hipSetDevice(device); }
inline cudaError_t cudaGetDeviceCount(int* count) { return hipGetDeviceCount(count); }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* props, int device) {
    return hipGetDeviceProperties(props, device);
}
inline cudaError_t cudaDeviceSynchronize() { return hipDeviceSynchronize(); }
inline cudaError_t cudaDeviceReset() { return hipDeviceReset(); }
inline cudaError_t cudaDeviceEnablePeerAccess(int peer_device, unsigned int flags = 0) {
    return hipDeviceEnablePeerAccess(peer_device, flags);
}
inline cudaError_t cudaDeviceCanAccessPeer(int* can_access_peer, int device, int peer_device) {
    return hipDeviceCanAccessPeer(can_access_peer, device, peer_device);
}
inline cudaError_t cudaStreamCreate(cudaStream_t* stream) { return hipStreamCreate(stream); }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    return hipStreamCreateWithFlags(stream, flags);
}
inline cudaError_t cudaStreamDestroy(cudaStream_t stream) { return hipStreamDestroy(stream); }
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) { return hipStreamSynchronize(stream); }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    return hipStreamWaitEvent(stream, event, flags);
}
inline cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                         cudaStreamCallback_t callback,
                                         void* user_data,
                                         unsigned int flags) {
    return hipStreamAddCallback(stream, callback, user_data, flags);
}
inline cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void*), void* user_data) {
    return hipLaunchHostFunc(stream, fn, user_data);
}
inline cudaError_t cudaEventCreate(cudaEvent_t* event) { return hipEventCreate(event); }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    return hipEventCreateWithFlags(event, flags);
}
inline cudaError_t cudaEventDestroy(cudaEvent_t event) { return hipEventDestroy(event); }
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr) {
    return hipEventRecord(event, stream);
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t event) { return hipEventSynchronize(event); }
inline cudaError_t cudaEventQuery(cudaEvent_t event) { return hipEventQuery(event); }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t stop) {
    return hipEventElapsedTime(ms, start, stop);
}
inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) {
    return hipPointerGetAttributes(attributes, ptr);
}
inline cudaError_t cudaMallocManaged(void** ptr, size_t bytes, unsigned int flags = cudaMemAttachGlobal) {
    return hipMallocManaged(ptr, bytes, flags);
}
inline cudaError_t cudaMemPrefetchAsync(const void* ptr,
                                        size_t bytes,
                                        int device,
                                        cudaStream_t stream = nullptr) {
    return hipMemPrefetchAsync(ptr, bytes, device, stream);
}
inline cudaError_t cudaMemPrefetchAsync(const void* ptr,
                                        size_t bytes,
                                        cudaMemLocation location,
                                        unsigned int flags,
                                        cudaStream_t stream = nullptr) {
    return hipMemPrefetchAsync_v2(ptr, bytes, location, flags, stream);
}
inline cudaError_t cudaMemAdvise(const void* ptr, size_t bytes, cudaMemoryAdvise advice, int device) {
    return hipMemAdvise(ptr, bytes, advice, device);
}
inline cudaError_t cudaMemAdvise(const void* ptr,
                                 size_t bytes,
                                 cudaMemoryAdvise advice,
                                 cudaMemLocation location) {
    return hipMemAdvise_v2(ptr, bytes, advice, location);
}
    #endif
#else
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
#endif

namespace deme {

enum class GpuPointerKind {
    Unregistered,
    Device,
    Host,
    Managed,
    Unknown
};

inline GpuPointerKind gpuPointerKind(const cudaPointerAttributes& a) {
#if defined(DEME_GPU_BACKEND_HIP) || defined(DEME_USE_HIP)
    if (a.isManaged)
        return GpuPointerKind::Managed;
    if (a.type == cudaMemoryTypeDevice)
        return GpuPointerKind::Device;
    if (a.type == cudaMemoryTypeHost)
        return GpuPointerKind::Host;
    if (a.type == cudaMemoryTypeUnregistered)
        return GpuPointerKind::Unregistered;
    return GpuPointerKind::Unknown;
#else
    if (a.type == cudaMemoryTypeDevice)
        return GpuPointerKind::Device;
    if (a.type == cudaMemoryTypeHost)
        return GpuPointerKind::Host;
    if (a.type == cudaMemoryTypeManaged)
        return GpuPointerKind::Managed;
    if (a.type == cudaMemoryTypeUnregistered)
        return GpuPointerKind::Unregistered;
    return GpuPointerKind::Unknown;
#endif
}

}  // namespace deme

#endif
