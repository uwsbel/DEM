//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <utility>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <core/ApiVersion.h>
#include <core/utils/JitHelper.h>
#include <DEM/kT.h>
#include "dT.h"
#include "utils/HostSideHelpers.hpp"
#include "Defines.h"
#include "algorithms/DEMStaticDeviceSubroutines.h"
#include "kernel/DEMHelperKernels.cuh"

#ifdef DEME_ENABLE_NVTX
    #include <nvtx3/nvtx3.hpp>
    #include <nvtx3/nvToolsExtCudaRt.h>
    #define DEME_NVTX_CONCAT_IMPL(x, y) x##y
    #define DEME_NVTX_CONCAT(x, y) DEME_NVTX_CONCAT_IMPL(x, y)
    #define DEME_NVTX_RANGE(name)                                                   \
        auto DEME_NVTX_CONCAT(__deme_nvtx_range_, __LINE__) = nvtx3::scoped_range { \
            name                                                                    \
        }
    #define DEME_NVTX_NAME_STREAM(stream, label) nvtxNameCudaStreamA((stream), (label))
#else
    #define DEME_NVTX_RANGE(name) \
        do {                      \
        } while (0)
    #define DEME_NVTX_NAME_STREAM(stream, label) \
        do {                                     \
        } while (0)
#endif

namespace deme {

namespace {
struct DynamicProduceReadyPayload {
    ThreadManager* sched = nullptr;
};

inline bool triangle_scene(const DEMSimParams* simParams) {
    return simParams && simParams->nTriGM > 0;
}

inline bool triangle_scene(const DualStruct<DEMSimParams>& simParams) {
    return triangle_scene(&(*simParams));
}

inline size_t quantized_contact_capacity(size_t n) {
    const size_t floor_cap = 1024;
    n = std::max(n, floor_cap);
    const size_t padded = n + n / 8 + 64;
    const size_t quantum = (padded < 8192) ? 256 : ((padded < 65536) ? 1024 : 4096);
    return ((padded + quantum - 1) / quantum) * quantum;
}

inline uint32_t expandBits10(uint32_t v) {
    v &= 0x000003ffu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

inline uint32_t morton3D10(uint32_t x, uint32_t y, uint32_t z) {
    return expandBits10(x) | (expandBits10(y) << 1) | (expandBits10(z) << 2);
}

inline uint32_t quantizeUnitTo10Bits(float v) {
    v = std::max(0.0f, std::min(1.0f, v));
    return static_cast<uint32_t>(v * 1023.0f + 0.5f);
}

struct MortonTriEntry {
    bodyID_t triID;
    uint32_t key;
};

inline double hostDot3(const float3& a, const float3& b) {
    return (double)a.x * (double)b.x + (double)a.y * (double)b.y + (double)a.z * (double)b.z;
}

inline double hostLength3(const float3& v) {
    return std::sqrt(hostDot3(v, v));
}

inline float3 hostSub3(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline float3 hostTriangleCentroid(const float3& p1, const float3& p2, const float3& p3) {
    return make_float3((p1.x + p2.x + p3.x) / 3.f, (p1.y + p2.y + p3.y) / 3.f, (p1.z + p2.z + p3.z) / 3.f);
}

inline float hostTriangleExpandCoeff(const float3& p1, const float3& p2, const float3& p3) {
    const double a = hostLength3(hostSub3(p2, p3));
    const double b = hostLength3(hostSub3(p3, p1));
    const double c = hostLength3(hostSub3(p1, p2));
    const double perimeter = a + b + c;
    if (perimeter <= 1e-12) {
        return 1024.f;
    }

    const float3 incenter = make_float3((float)((a * (double)p1.x + b * (double)p2.x + c * (double)p3.x) / perimeter),
                                        (float)((a * (double)p1.y + b * (double)p2.y + c * (double)p3.y) / perimeter),
                                        (float)((a * (double)p1.z + b * (double)p2.z + c * (double)p3.z) / perimeter));

    auto coeff_for_vertex = [&](const float3& vertex, const float3& side_end) {
        const float3 expand = hostSub3(vertex, incenter);
        const float3 side = hostSub3(side_end, vertex);
        const double expand_len = hostLength3(expand);
        const double side_len = hostLength3(side);
        if (expand_len <= 1e-12 || side_len <= 1e-12) {
            return 1024.0;
        }
        const double cos_half = std::max(-1.0, std::min(1.0, -hostDot3(expand, side) / (expand_len * side_len)));
        const double sin_sq = std::max(1e-12, 1.0 - cos_half * cos_half);
        return 1.0 + 1.0 / std::sqrt(sin_sq);
    };

    const double coeff = std::max({coeff_for_vertex(p1, p2), coeff_for_vertex(p2, p3), coeff_for_vertex(p3, p1)});
    return (float)coeff;
}

void CUDART_CB NotifyDynamicProduceReady(void* userData) {
    auto* payload = static_cast<DynamicProduceReadyPayload*>(userData);
    if (payload && payload->sched) {
        payload->sched->dynamicOwned_Prod2ConsBuffer_isFresh.store(true, std::memory_order_release);
        payload->sched->cv_DynamicCanProceed.notify_all();
    }
    delete payload;
}
}  // namespace

inline void DEMKinematicThread::transferPrimitivesArraysResize(int buffer_idx, size_t nContactPairs) {
    // These buffers are on dT
    DEME_GPU_CALL(cudaSetDevice(dT->streamInfo.device));
    DEME_DEVICE_ARRAY_RESIZE(dT->idPrimitiveA_buffer[buffer_idx], nContactPairs);
    DEME_DEVICE_ARRAY_RESIZE(dT->idPrimitiveB_buffer[buffer_idx], nContactPairs);
    DEME_DEVICE_ARRAY_RESIZE(dT->contactTypePrimitive_buffer[buffer_idx], nContactPairs);
    DEME_DEVICE_ARRAY_RESIZE(dT->geomToPatchMap_buffer[buffer_idx], nContactPairs);
    granData->pDTOwnedBuffer_idPrimitiveA = dT->idPrimitiveA_buffer[buffer_idx].data();
    granData->pDTOwnedBuffer_idPrimitiveB = dT->idPrimitiveB_buffer[buffer_idx].data();
    granData->pDTOwnedBuffer_contactType = dT->contactTypePrimitive_buffer[buffer_idx].data();
    granData->pDTOwnedBuffer_geomToPatchMap = dT->geomToPatchMap_buffer[buffer_idx].data();

    // Unset the device change we just made
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));

    // But we don't have to toDevice granData or dT->granData, and this is because all _buffer arrays don't
    // particupate kernel computations, so even if their pointers are fresh only on host, it's fine
}

inline void DEMKinematicThread::transferPatchArrayResize(int buffer_idx, size_t nContactPairs) {
    // These buffers are on dT
    DEME_GPU_CALL(cudaSetDevice(dT->streamInfo.device));
    DEME_DEVICE_ARRAY_RESIZE(dT->idPatchA_buffer[buffer_idx], nContactPairs);
    DEME_DEVICE_ARRAY_RESIZE(dT->idPatchB_buffer[buffer_idx], nContactPairs);
    DEME_DEVICE_ARRAY_RESIZE(dT->contactTypePatch_buffer[buffer_idx], nContactPairs);
    DEME_DEVICE_ARRAY_RESIZE(dT->contactPatchIsland_buffer[buffer_idx], nContactPairs);
    if (!solverFlags.isHistoryless) {
        DEME_DEVICE_ARRAY_RESIZE(dT->contactMapping_buffer[buffer_idx], nContactPairs);
        granData->pDTOwnedBuffer_contactMapping = dT->contactMapping_buffer[buffer_idx].data();
    }
    granData->pDTOwnedBuffer_idPatchA = dT->idPatchA_buffer[buffer_idx].data();
    granData->pDTOwnedBuffer_idPatchB = dT->idPatchB_buffer[buffer_idx].data();
    granData->pDTOwnedBuffer_contactTypePatch = dT->contactTypePatch_buffer[buffer_idx].data();
    granData->pDTOwnedBuffer_contactPatchIsland = dT->contactPatchIsland_buffer[buffer_idx].data();

    // Unset the device change we just made
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));
}

void DEMKinematicThread::recordAndSyncEvent() {
    DEME_GPU_CALL(cudaEventRecord(streamSyncEvent, streamInfo.stream));
    DEME_GPU_CALL(cudaEventSynchronize(streamSyncEvent));
}

void DEMKinematicThread::recordEventOnly() {
    DEME_GPU_CALL(cudaEventRecord(streamSyncEvent, streamInfo.stream));
}

void DEMKinematicThread::syncRecordedEvent() {
    DEME_GPU_CALL(cudaEventSynchronize(streamSyncEvent));
}

void DEMKinematicThread::calibrateParams() {
    double prev_time, curr_time;
    // If it is true, then it's the AccumTimer telling us it is the right time to decide how to change bin size
    if (CDAccumTimer.QueryOn(prev_time, curr_time, stateParams.binChangeObserveSteps)) {
        // Auto-adjust bin size
        if (solverFlags.autoBinSize) {
            const bool tri_scene = triangle_scene(simParams);
            const float top_rate = tri_scene ? std::min(stateParams.binTopChangeRate, 0.03f)
                                             : stateParams.binTopChangeRate;
            const bool strong_prescribed_motion =
                (solverFlags.prescribedAngVelMagnitudeHint > 20.f);
            if (tri_scene && strong_prescribed_motion) {
                // In strong prescribed-rotation tri scenes, timing-noise-driven bin adaptation causes
                // large run-to-run variability and occasional candidate spikes. Keep bin size fixed.
                stateParams.binCurrentChangeRate = 0.f;
                DEME_DEBUG_PRINTF("Strong-motion tri scene: freezing adaptive bin-size updates.");
                DEME_DEBUG_PRINTF("kT runtime per step: %.7gs", CDAccumTimer.GetPrevTime());
                simParams.toDeviceAsync(streamInfo.stream);
                return;
            }
            const float sph_bin_pressure =
                (simParams->errOutBinSphNum > 0)
                    ? std::clamp((float)stateParams.maxSphFoundInBin / (float)simParams->errOutBinSphNum, 0.f, 1.f)
                    : 0.f;
            const float tri_bin_pressure =
                (simParams->errOutBinTriNum > 0)
                    ? std::clamp((float)stateParams.maxTriFoundInBin / (float)simParams->errOutBinTriNum, 0.f, 1.f)
                    : 0.f;
            const float bin_pressure = std::max(sph_bin_pressure, tri_bin_pressure);
            const float avg_contacts = std::max(0.f, stateParams.avgCntsPerPrimitive);
            int speed_dir = sign_func(stateParams.binCurrentChangeRate);
            // Keep the first direction deterministic to reduce run-to-run bin-size jitter.
            if (speed_dir == 0) {
                const bool high_bin_pressure =
                    (stateParams.maxSphFoundInBin > 0.50 * simParams->errOutBinSphNum) ||
                    (stateParams.maxTriFoundInBin > 0.50 * simParams->errOutBinTriNum);
                speed_dir = high_bin_pressure ? -1 : 1;
            }
            const bool over_upper_safety =
                (stateParams.maxSphFoundInBin > stateParams.binChangeUpperSafety * simParams->errOutBinSphNum) ||
                (stateParams.maxTriFoundInBin > stateParams.binChangeUpperSafety * simParams->errOutBinTriNum);
            const bool over_lower_safety =
                (stateParams.numBins > stateParams.binChangeLowerSafety * (double)(std::numeric_limits<binID_t>::max()));
            if (tri_scene) {
                // Pressure/contact-band control for tri scenes keeps behavior deterministic across runs.
                const float prev_rate = stateParams.binCurrentChangeRate;
                float target_rate = 0.f;

                if (over_upper_safety) {
                    target_rate = -top_rate;
                } else if (over_lower_safety) {
                    target_rate = (bin_pressure < 0.38f && avg_contacts < 0.8f) ? (0.35f * top_rate) : 0.f;
                } else {
                    const float high_band = (avg_contacts > 2.5f) ? 0.58f : 0.64f;
                    const float low_band = (avg_contacts < 0.9f) ? 0.42f : 0.34f;
                    if (bin_pressure > high_band) {
                        const float sev = std::clamp((bin_pressure - high_band) / std::max(1e-6f, 1.f - high_band), 0.f, 1.f);
                        target_rate = -top_rate * (0.30f + 0.70f * sev);
                    } else if (bin_pressure < low_band && avg_contacts < 1.8f) {
                        const float sev = std::clamp((low_band - bin_pressure) / std::max(1e-6f, low_band), 0.f, 1.f);
                        target_rate = top_rate * (0.08f + 0.32f * sev);
                    } else {
                        target_rate = 0.f;
                    }
                }

                // If contacts are already dense, don't let bin growth push false candidates.
                if (avg_contacts > 3.5f) {
                    target_rate = std::min(target_rate, 0.f);
                }

                // Slew-limit updates to avoid one-observation jumps.
                const float max_delta = std::max(0.0005f, stateParams.binChangeRateAcc * top_rate);
                const float lo = prev_rate - max_delta;
                const float hi = prev_rate + max_delta;
                stateParams.binCurrentChangeRate = std::clamp(target_rate, lo, hi);
                stateParams.binCurrentChangeRate = clampBetween(stateParams.binCurrentChangeRate, -top_rate, top_rate);
            } else {
                float speed_update;
                if (curr_time < prev_time) {
                    // If there is improvement, then we accelerate the current change direction
                    speed_update = speed_dir * stateParams.binChangeRateAcc * top_rate;
                } else {
                    // If no improvement, revert the direction
                    speed_update = -speed_dir * stateParams.binChangeRateAcc * top_rate;
                }
                // But, if the bin size is going to get too big or too small, a penalty is enforced
                if (over_upper_safety) {
                    // Then the size must start to decrease
                    speed_update = -1.0f * stateParams.binChangeRateAcc * top_rate;
                }
                if (over_lower_safety) {
                    // Then size must start to increase
                    speed_update = 1.0f * stateParams.binChangeRateAcc * top_rate;
                }
                // Acc is done. Now apply it to bin size change speed
                stateParams.binCurrentChangeRate += speed_update;
                // But, the speed must fall in range
                stateParams.binCurrentChangeRate = clampBetween(stateParams.binCurrentChangeRate, -top_rate, top_rate);
            }

            // Change bin size
            if (stateParams.binCurrentChangeRate > 0) {
                simParams->dyn.binSize *= (1. + stateParams.binCurrentChangeRate);
            } else {
                simParams->dyn.binSize /= (1. - stateParams.binCurrentChangeRate);
            }
            // Register the new bin size
            simParams->dyn.inv_binSize = 1. / simParams->dyn.binSize;
            stateParams.numBins =
                hostCalcBinNum(simParams->nbX, simParams->nbY, simParams->nbZ, simParams->voxelSize,
                               simParams->dyn.binSize, simParams->nvXp2, simParams->nvYp2, simParams->nvZp2);

            DEME_DEBUG_PRINTF("Bin size is now: %.7g", simParams->dyn.binSize);
            DEME_DEBUG_PRINTF("Total num of bins is now: %zu", stateParams.numBins);
        }
        DEME_DEBUG_PRINTF("kT runtime per step: %.7gs", CDAccumTimer.GetPrevTime());
    }
    // binSize is now calculated, we need to migrate that to device
    simParams.toDeviceAsync(streamInfo.stream);
}

inline void DEMKinematicThread::computeMarginFromAbsv(float* absVel_owner, float* absAngVel_owner) {
    size_t blocks_needed;
    blocks_needed = (simParams->nSpheresGM + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        misc_kernels->kernel("computeMarginFromAbsv_implSph")
            .instantiate()
            .configure(dim3(blocks_needed), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData, absVel_owner, absAngVel_owner, &(stateParams.ts), &(stateParams.maxDrift),
                    (size_t)simParams->nSpheresGM);
    }
    const bool useBigMeshBVHLeafPath =
        (simParams->nBigMeshOwners > 0 && simParams->nBigMeshBVHLeaves > 0 &&
            !(simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f));

    blocks_needed = (simParams->nTriGM + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        misc_kernels->kernel("computeMarginFromAbsv_implTri")
            .instantiate()
            .configure(dim3(blocks_needed), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData, absVel_owner, absAngVel_owner, &(stateParams.ts), &(stateParams.maxDrift),
                    &(stateParams.maxTriTriPenetration), solverFlags.meshUniversalContact,
                    false, (size_t)simParams->nTriGM);
    }
    if (useBigMeshBVHLeafPath && simParams->nBigMeshBVHLeaves > 0) {
        blocks_needed = (simParams->nBigMeshBVHLeaves + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
        misc_kernels->kernel("computeMarginFromAbsv_implBigMeshLeaf")
            .instantiate()
            .configure(dim3(blocks_needed), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData, absVel_owner, absAngVel_owner, &(stateParams.ts), &(stateParams.maxDrift),
                    (size_t)simParams->nBigMeshBVHLeaves);
    }
    blocks_needed = (simParams->nAnalGM + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        misc_kernels->kernel("computeMarginFromAbsv_implAnal")
            .instantiate()
            .configure(dim3(blocks_needed), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
            .launch(&simParams, &granData, absVel_owner, absAngVel_owner, &(stateParams.ts), &(stateParams.maxDrift),
                    (size_t)simParams->nAnalGM);
    }
}

inline void DEMKinematicThread::unpackMyBuffer() {
    const int dev = streamInfo.device;
    const bool same_dev = (streamInfo.device == dT->streamInfo.device);
    bool swapped = false;
    bool needGranDataDeviceSync = false;

    if (same_dev && dT->dT_to_kT_BufferReadyEvent) {
        DEME_GPU_CALL(cudaStreamWaitEvent(streamInfo.stream, dT->dT_to_kT_BufferReadyEvent, 0));
    }
#ifndef DEME_USE_MANAGED_ARRAYS
    if (same_dev) {
        swapped = swap_device_buffer(voxelID, voxelID_buffer);
        swapped = swap_device_buffer(locX, locX_buffer) && swapped;
        swapped = swap_device_buffer(locY, locY_buffer) && swapped;
        swapped = swap_device_buffer(locZ, locZ_buffer) && swapped;
        swapped = swap_device_buffer(oriQw, oriQ0_buffer) && swapped;
        swapped = swap_device_buffer(oriQx, oriQ1_buffer) && swapped;
        swapped = swap_device_buffer(oriQy, oriQ2_buffer) && swapped;
        swapped = swap_device_buffer(oriQz, oriQ3_buffer) && swapped;
    }
#endif

    if (swapped) {
        xfer::XferList scalars;
        scalars.add(&(stateParams.ts), &(stateParams.ts_buffer), sizeof(float));
        scalars.add(&(stateParams.maxDrift), &(stateParams.maxDrift_buffer), sizeof(unsigned int));
        scalars.add(&(stateParams.maxTriTriPenetration), &(stateParams.maxTriTriPenetration_buffer), sizeof(double));
        scalars.run(dev, dev, streamInfo.stream);
        // swap_device_buffer updates pointer fields in host-side granData via DualArray binding; sync them to device
        // immediately because kernels below use the device-side copy of granData.
        granData.toDeviceAsync(streamInfo.stream);
    } else {
        xfer::XferList xl;
        xl.add(granData->voxelID, voxelID_buffer.data(), simParams->nOwnerBodies * sizeof(voxelID_t));
        xl.add(granData->locX, locX_buffer.data(), simParams->nOwnerBodies * sizeof(subVoxelPos_t));
        xl.add(granData->locY, locY_buffer.data(), simParams->nOwnerBodies * sizeof(subVoxelPos_t));
        xl.add(granData->locZ, locZ_buffer.data(), simParams->nOwnerBodies * sizeof(subVoxelPos_t));
        xl.add(granData->oriQw, oriQ0_buffer.data(), simParams->nOwnerBodies * sizeof(oriQ_t));
        xl.add(granData->oriQx, oriQ1_buffer.data(), simParams->nOwnerBodies * sizeof(oriQ_t));
        xl.add(granData->oriQy, oriQ2_buffer.data(), simParams->nOwnerBodies * sizeof(oriQ_t));
        xl.add(granData->oriQz, oriQ3_buffer.data(), simParams->nOwnerBodies * sizeof(oriQ_t));
        xl.add(&(stateParams.ts), &(stateParams.ts_buffer), sizeof(float));
        xl.add(&(stateParams.maxDrift), &(stateParams.maxDrift_buffer), sizeof(unsigned int));
        xl.add(&(stateParams.maxTriTriPenetration), &(stateParams.maxTriTriPenetration_buffer), sizeof(double));
        xl.run(dev, dev, streamInfo.stream);
    }
    if (simParams->useCylPeriodicDiagCounters) {
        DEME_GPU_CALL(cudaMemcpyAsync(granData->ownerCylGhostActive, ownerCylGhostActive_buffer.data(),
                                      simParams->nOwnerBodies * sizeof(unsigned int), cudaMemcpyDeviceToDevice,
                                      streamInfo.stream));
    }

    // Make sure we don't have velocity that is too high
    cubMaxReduce<float>(absVel_buffer.data(), &(stateParams.maxVel), simParams->nOwnerBodies, streamInfo.stream,
                        solverScratchSpace);
    cubMaxReduce<float>(absAngVel_buffer.data(), &(stateParams.maxAngVel), simParams->nOwnerBodies, streamInfo.stream,
                        solverScratchSpace);
    stateParams.maxVel.toHost();
    stateParams.maxAngVel.toHost();
    stateParams.maxTriTriPenetration.toHost();
    if (*stateParams.maxVel > simParams->errOutVel || !std::isfinite(*stateParams.maxVel) ||
        *stateParams.maxAngVel > simParams->errOutAngVel || !std::isfinite(*stateParams.maxAngVel)) {
        DEME_ERROR(
            "System max velocity/angular velocity is %.7g/%.7g, exceeded max allowance (%.7g/%.7g).\nIf this velocity "
            "is not abnormal and you "
            "want to increase this allowance, use SetErrorOutVelocity before initializing simulation.\nOtherwise, the "
            "simulation may have diverged and relaxing the physics may help, such as decreasing the step size and "
            "modifying material properties.\nIf this happens at the start of simulation, check if there are initial "
            "penetrations, a.k.a. elements initialized inside walls.",
            *(stateParams.maxVel), *(stateParams.maxAngVel), simParams->errOutVel, simParams->errOutAngVel);
    }
    if (*stateParams.maxVel >
        simParams->dyn.approxMaxVel) {  // If maxVel is larger than the user estimation, that is an anomaly
        // This prints when verbosity higher than METRIC
        DEME_STATUS("OVER_MAX_VEL", "Simulation entity velocity reached %.6g, over the user-estimated max (%.6g)",
                    *stateParams.maxVel, simParams->dyn.approxMaxVel);
    }
    if (*stateParams.maxTriTriPenetration > simParams->capTriTriPenetration) {
        DEME_STATUS("OVER_MAX_MESH_PENETRATION",
                    "Mesh--mesh contact penetration reached %.6g, over the user-estimated max (%.6g)",
                    *stateParams.maxTriTriPenetration, simParams->capTriTriPenetration);
    }
    DEME_DEBUG_PRINTF("kT received an update, max vel: %.6g", *stateParams.maxVel);
    DEME_DEBUG_PRINTF("kT received an update, max ang vel: %.6g", *stateParams.maxAngVel);
    DEME_DEBUG_PRINTF("kT received an update, max tri--tri penetration: %.6g", *stateParams.maxTriTriPenetration);

    // Whatever drift value dT says, kT listens; unless kinematicMaxFutureDrift is negative in which case the user
    // explicitly said not caring the future drift.
    stateParams.maxDrift.toHost();
    stateParams.ts.toHost();
    pSchedSupport->kinematicMaxFutureDrift = (pSchedSupport->kinematicMaxFutureDrift.load() < 0.)
                                                 ? pSchedSupport->kinematicMaxFutureDrift.load()
                                                 : *(stateParams.maxDrift);

    if (solverFlags.canFamilyChangeOnDevice) {
        bool swapped_family = false;
#ifndef DEME_USE_MANAGED_ARRAYS
        if (same_dev) {
            swapped_family = swap_device_buffer(familyID, familyID_buffer);
        }
#endif
        if (!swapped_family) {
            DEME_GPU_CALL(cudaMemcpyAsync(granData->familyID, familyID_buffer.data(),
                                          simParams->nOwnerBodies * sizeof(family_t), cudaMemcpyDeviceToDevice,
                                          streamInfo.stream));
        }
        needGranDataDeviceSync = needGranDataDeviceSync || swapped_family;
    }

    if (solverFlags.willMeshDeform) {
        bool swapped_mesh = false;
#ifndef DEME_USE_MANAGED_ARRAYS
        if (same_dev) {
            swapped_mesh = swap_device_buffer(relPosNode1, relPosNode1_buffer);
            swapped_mesh = swap_device_buffer(relPosNode2, relPosNode2_buffer) && swapped_mesh;
            swapped_mesh = swap_device_buffer(relPosNode3, relPosNode3_buffer) && swapped_mesh;
        }
#endif
        if (!swapped_mesh) {
            DEME_GPU_CALL(cudaMemcpyAsync(granData->relPosNode1, relPosNode1_buffer.data(),
                                          simParams->nTriGM * sizeof(float3), cudaMemcpyDeviceToDevice,
                                          streamInfo.stream));
            DEME_GPU_CALL(cudaMemcpyAsync(granData->relPosNode2, relPosNode2_buffer.data(),
                                          simParams->nTriGM * sizeof(float3), cudaMemcpyDeviceToDevice,
                                          streamInfo.stream));
            DEME_GPU_CALL(cudaMemcpyAsync(granData->relPosNode3, relPosNode3_buffer.data(),
                                          simParams->nTriGM * sizeof(float3), cudaMemcpyDeviceToDevice,
                                          streamInfo.stream));
        }
        solverFlags.willMeshDeform = false;
        needGranDataDeviceSync = needGranDataDeviceSync || swapped_mesh;
    }

    if (needGranDataDeviceSync) {
        granData.toDeviceAsync(streamInfo.stream);
    }

    // kT will need to derive the thickness of the CD margin, based on dT's info on system vel.
    if (!solverFlags.isExpandFactorFixed) {
        // This kernel will turn absv to marginSize, and if a vel is over max, it will clamp it.
        // Converting to size_t is SUPER important... CUDA kernel call basically does not have type conversion.
        computeMarginFromAbsv(absVel_buffer.data(), absAngVel_buffer.data());
    } else {  // If isExpandFactorFixed, then just fill in that constant array.
        // This one is statically compiled, unlike the other branch
        fillMarginValues(&simParams, &granData, (size_t)(simParams->nSpheresGM), (size_t)(simParams->nTriGM),
                         (size_t)(simParams->nAnalGM), streamInfo.stream);
        const bool useBigMeshBVHLeafPath =
            (simParams->nBigMeshOwners > 0 && simParams->nBigMeshBVHLeaves > 0 && !solverFlags.meshUniversalContact &&
             !(simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f));
        if (useBigMeshBVHLeafPath && simParams->nBigMeshBVHLeaves > 0) {
            size_t blocks_needed = (simParams->nBigMeshBVHLeaves + DEME_MAX_THREADS_PER_BLOCK - 1) /
                                   DEME_MAX_THREADS_PER_BLOCK;
            misc_kernels->kernel("fillFixedMarginBigMeshLeaf")
                .instantiate()
                .configure(dim3(blocks_needed), dim3(DEME_MAX_THREADS_PER_BLOCK), 0, streamInfo.stream)
                .launch(&simParams, &granData, (size_t)simParams->nBigMeshBVHLeaves);
        }
    }

    // Keep ghosting/wrapping margins consistent with kT's dynamic margin size.
    if (pSchedSupport) {
        float ghost_margin = simParams->dyn.beta;
        if (!solverFlags.isExpandFactorFixed) {
            float max_margin = 0.f;
            float tmp_sph = -DEME_HUGE_FLOAT;
            float tmp_tri = -DEME_HUGE_FLOAT;
            float tmp_big_leaf = -DEME_HUGE_FLOAT;
            float tmp_anal = -DEME_HUGE_FLOAT;
            const bool useBigMeshBVHLeafPath =
                (simParams->nBigMeshOwners > 0 && simParams->nBigMeshBVHLeaves > 0 && !solverFlags.meshUniversalContact &&
                 !(simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f));
            const bool has_margins =
                (simParams->nSpheresGM > 0) || (simParams->nTriGM > 0) || (simParams->nAnalGM > 0) ||
                (useBigMeshBVHLeafPath && simParams->nBigMeshBVHLeaves > 0);
            if (has_margins) {
                float* max_margin_dev =
                    (float*)solverScratchSpace.allocateTempVector("maxMarginTmp", sizeof(float));
                if (simParams->nSpheresGM > 0) {
                    cubMaxReduce<float>(granData->marginSizeSphere, max_margin_dev, simParams->nSpheresGM,
                                        streamInfo.stream, solverScratchSpace);
                    DEME_GPU_CALL(cudaMemcpyAsync(&tmp_sph, max_margin_dev, sizeof(float), cudaMemcpyDeviceToHost,
                                                  streamInfo.stream));
                }
                if (simParams->nTriGM > 0) {
                    cubMaxReduce<float>(granData->marginSizeTriangle, max_margin_dev, simParams->nTriGM,
                                        streamInfo.stream, solverScratchSpace);
                    DEME_GPU_CALL(cudaMemcpyAsync(&tmp_tri, max_margin_dev, sizeof(float), cudaMemcpyDeviceToHost,
                                                  streamInfo.stream));
                }
                if (useBigMeshBVHLeafPath && simParams->nBigMeshBVHLeaves > 0) {
                    cubMaxReduce<float>(granData->bigMeshLeafMargin, max_margin_dev, simParams->nBigMeshBVHLeaves,
                                        streamInfo.stream, solverScratchSpace);
                    DEME_GPU_CALL(cudaMemcpyAsync(&tmp_big_leaf, max_margin_dev, sizeof(float), cudaMemcpyDeviceToHost,
                                                  streamInfo.stream));
                }
                if (simParams->nAnalGM > 0) {
                    cubMaxReduce<float>(granData->marginSizeAnalytical, max_margin_dev, simParams->nAnalGM,
                                        streamInfo.stream, solverScratchSpace);
                    DEME_GPU_CALL(cudaMemcpyAsync(&tmp_anal, max_margin_dev, sizeof(float), cudaMemcpyDeviceToHost,
                                                  streamInfo.stream));
                }
                DEME_GPU_CALL(cudaStreamSynchronize(streamInfo.stream));
                if (std::isfinite(tmp_sph) && tmp_sph > max_margin) {
                    max_margin = tmp_sph;
                }
                if (std::isfinite(tmp_tri) && tmp_tri > max_margin) {
                    max_margin = tmp_tri;
                }
                if (std::isfinite(tmp_big_leaf) && tmp_big_leaf > max_margin) {
                    max_margin = tmp_big_leaf;
                }
                if (std::isfinite(tmp_anal) && tmp_anal > max_margin) {
                    max_margin = tmp_anal;
                }
                solverScratchSpace.finishUsingTempVector("maxMarginTmp");
            }

            ghost_margin = max_margin - simParams->maxFamilyExtraMargin;
            if (!std::isfinite(ghost_margin) || ghost_margin < 0.f) {
                ghost_margin = 0.f;
            }
        }
        if (!std::isfinite(ghost_margin) || ghost_margin < 0.f) {
            ghost_margin = 0.f;
        }
        if (fabsf(simParams->dyn.beta - ghost_margin) > 1e-8f) {
            simParams->dyn.beta = ghost_margin;
            simParams.toDeviceAsync(streamInfo.stream);
        }
        pSchedSupport->kinematicGhostMargin.store(ghost_margin, std::memory_order_relaxed);
    }

    // Update dT's write pointers (buffer and DualArray ping-pong via swap_device_buffer)
    dT->granData->pKTOwnedBuffer_absVel = absVel_buffer.data();
    dT->granData->pKTOwnedBuffer_absAngVel = absAngVel_buffer.data();
    dT->granData->pKTOwnedBuffer_voxelID = voxelID_buffer.data();
    dT->granData->pKTOwnedBuffer_locX = locX_buffer.data();
    dT->granData->pKTOwnedBuffer_locY = locY_buffer.data();
    dT->granData->pKTOwnedBuffer_locZ = locZ_buffer.data();
    dT->granData->pKTOwnedBuffer_oriQ0 = oriQ0_buffer.data();
    dT->granData->pKTOwnedBuffer_oriQ1 = oriQ1_buffer.data();
    dT->granData->pKTOwnedBuffer_oriQ2 = oriQ2_buffer.data();
    dT->granData->pKTOwnedBuffer_oriQ3 = oriQ3_buffer.data();
    dT->granData->pKTOwnedBuffer_familyID = familyID_buffer.data();
    dT->granData->pKTOwnedBuffer_relPosNode1 = relPosNode1_buffer.data();
    dT->granData->pKTOwnedBuffer_relPosNode2 = relPosNode2_buffer.data();
    dT->granData->pKTOwnedBuffer_relPosNode3 = relPosNode3_buffer.data();
}

inline void DEMKinematicThread::sendToTheirBuffer() {
    const int write_idx = dT->kt_write_buf;
    const size_t nPrimitive = *solverScratchSpace.numPrimitiveContacts;
    const size_t nPatch = *solverScratchSpace.numContacts;

    const int srcDev = streamInfo.device;      // kT GPU
    const int dstDev = dT->streamInfo.device;  // dT GPU
    const bool same_dev = (srcDev == dstDev);
    const cudaStream_t xfer_stream = same_dev ? streamInfo.stream : 0;

    static const bool allow_output_swap = []() {
        const char* env = std::getenv("DEME_KT_SEND_SWAP");
        if (!env || !*env) {
            return true;
        }
        return !(env[0] == '0' && env[1] == '\0');
    }();

    const bool tri_scene_flag = triangle_scene(simParams);

    size_t resize_prim = tri_scene_flag ? quantized_contact_capacity(std::max<size_t>(nPrimitive, 1)) : nPrimitive;
    size_t resize_patch = tri_scene_flag ? quantized_contact_capacity(std::max<size_t>(nPatch, 1)) : nPatch;
    if (same_dev && allow_output_swap && !tri_scene_flag) {
        resize_prim = DEME_MAX(resize_prim, idPrimitiveA.size());
        resize_prim = DEME_MAX(resize_prim, idPrimitiveB.size());
        resize_prim = DEME_MAX(resize_prim, contactTypePrimitive.size());
        resize_prim = DEME_MAX(resize_prim, geomToPatchMap.size());
        resize_patch = DEME_MAX(resize_patch, idPatchA.size());
        resize_patch = DEME_MAX(resize_patch, idPatchB.size());
        resize_patch = DEME_MAX(resize_patch, contactTypePatch.size());
        // Keep patch-side buffers in lockstep; missing one can corrupt swap/copies and crash kernels.
        resize_patch = DEME_MAX(resize_patch, contactPatchIsland.size());
        if (!solverFlags.isHistoryless) {
            resize_patch = DEME_MAX(resize_patch, contactMapping.size());
        }
    }

    bool need_resize_prim = resize_prim > dT->idPrimitiveA_buffer[write_idx].size() ||
                            resize_prim > dT->idPrimitiveB_buffer[write_idx].size() ||
                            resize_prim > dT->contactTypePrimitive_buffer[write_idx].size() ||
                            resize_prim > dT->geomToPatchMap_buffer[write_idx].size();
    bool need_resize_patch = resize_patch > dT->idPatchA_buffer[write_idx].size() ||
                             resize_patch > dT->idPatchB_buffer[write_idx].size() ||
                             resize_patch > dT->contactTypePatch_buffer[write_idx].size() ||
                             resize_patch > dT->contactPatchIsland_buffer[write_idx].size();
    if (!solverFlags.isHistoryless) {
        need_resize_patch = need_resize_patch || (resize_patch > dT->contactMapping_buffer[write_idx].size());
    }
    if (need_resize_prim) {
        transferPrimitivesArraysResize(write_idx, resize_prim);
    }
    if (need_resize_patch) {
        transferPatchArrayResize(write_idx, resize_patch);
    }

    bool output_swapped = false;
#ifndef DEME_USE_MANAGED_ARRAYS
    if (same_dev && allow_output_swap && !tri_scene_flag) {
        output_swapped = swap_device_buffer(idPrimitiveA, dT->idPrimitiveA_buffer[write_idx]);
        output_swapped = swap_device_buffer(idPrimitiveB, dT->idPrimitiveB_buffer[write_idx]) && output_swapped;
        output_swapped =
            swap_device_buffer(contactTypePrimitive, dT->contactTypePrimitive_buffer[write_idx]) && output_swapped;
        output_swapped = swap_device_buffer(geomToPatchMap, dT->geomToPatchMap_buffer[write_idx]) && output_swapped;
        output_swapped = swap_device_buffer(idPatchA, dT->idPatchA_buffer[write_idx]) && output_swapped;
        output_swapped = swap_device_buffer(idPatchB, dT->idPatchB_buffer[write_idx]) && output_swapped;
        output_swapped = swap_device_buffer(contactTypePatch, dT->contactTypePatch_buffer[write_idx]) && output_swapped;
        output_swapped =
            swap_device_buffer(contactPatchIsland, dT->contactPatchIsland_buffer[write_idx]) && output_swapped;
        if (!solverFlags.isHistoryless) {
            output_swapped = swap_device_buffer(contactMapping, dT->contactMapping_buffer[write_idx]) && output_swapped;
        }
    }
#endif

    granData->pDTOwnedBuffer_idPrimitiveA = dT->idPrimitiveA_buffer[write_idx].data();
    granData->pDTOwnedBuffer_idPrimitiveB = dT->idPrimitiveB_buffer[write_idx].data();
    granData->pDTOwnedBuffer_contactType = dT->contactTypePrimitive_buffer[write_idx].data();
    granData->pDTOwnedBuffer_geomToPatchMap = dT->geomToPatchMap_buffer[write_idx].data();
    granData->pDTOwnedBuffer_idPatchA = dT->idPatchA_buffer[write_idx].data();
    granData->pDTOwnedBuffer_idPatchB = dT->idPatchB_buffer[write_idx].data();
    granData->pDTOwnedBuffer_contactTypePatch = dT->contactTypePatch_buffer[write_idx].data();
    granData->pDTOwnedBuffer_contactPatchIsland = dT->contactPatchIsland_buffer[write_idx].data();
    if (!solverFlags.isHistoryless) {
        granData->pDTOwnedBuffer_contactMapping = dT->contactMapping_buffer[write_idx].data();
    }

    if (output_swapped) {
        // swap_device_buffer updates pointer fields in host-side granData via DualArray binding; sync them to device
        // before the next round of kernels.
        granData.toDeviceAsync(streamInfo.stream);
    }

    if (same_dev) {
        DEME_GPU_CALL(cudaMemcpyAsync(granData->pDTOwnedBuffer_nPrimitiveContacts,
                                      &(solverScratchSpace.numPrimitiveContacts), sizeof(size_t),
                                      cudaMemcpyDeviceToDevice, streamInfo.stream));
        DEME_GPU_CALL(cudaMemcpyAsync(granData->pDTOwnedBuffer_nPatchContacts, &(solverScratchSpace.numContacts),
                                      sizeof(size_t), cudaMemcpyDeviceToDevice, streamInfo.stream));
    } else {
        DEME_GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_nPrimitiveContacts,
                                 &(solverScratchSpace.numPrimitiveContacts), sizeof(size_t), cudaMemcpyDeviceToDevice));
        DEME_GPU_CALL(cudaMemcpy(granData->pDTOwnedBuffer_nPatchContacts, &(solverScratchSpace.numContacts),
                                 sizeof(size_t), cudaMemcpyDeviceToDevice));
    }

    if (!output_swapped) {
        xfer::XferList xs;
        xs.add(dT->idPrimitiveA_buffer[write_idx].data(), granData->idPrimitiveA, nPrimitive * sizeof(bodyID_t));
        xs.add(dT->idPrimitiveB_buffer[write_idx].data(), granData->idPrimitiveB, nPrimitive * sizeof(bodyID_t));
        xs.add(dT->contactTypePrimitive_buffer[write_idx].data(), granData->contactTypePrimitive,
               nPrimitive * sizeof(contact_t));
        xs.add(dT->geomToPatchMap_buffer[write_idx].data(), granData->geomToPatchMap,
               nPrimitive * sizeof(contactPairs_t));
        xs.add(dT->idPatchA_buffer[write_idx].data(), granData->idPatchA, nPatch * sizeof(bodyID_t));
        xs.add(dT->idPatchB_buffer[write_idx].data(), granData->idPatchB, nPatch * sizeof(bodyID_t));
        xs.add(dT->contactTypePatch_buffer[write_idx].data(), granData->contactTypePatch, nPatch * sizeof(contact_t));
        xs.add(dT->contactPatchIsland_buffer[write_idx].data(), granData->contactPatchIsland,
               nPatch * sizeof(bodyID_t));
        if (!solverFlags.isHistoryless) {
            xs.add(dT->contactMapping_buffer[write_idx].data(), granData->contactMapping,
                   nPatch * sizeof(contactPairs_t));
        }
        xs.run(dstDev, srcDev, xfer_stream);
    }
}

void DEMKinematicThread::workerThread() {
    // Set the device for this thread
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));
    DEME_NVTX_NAME_STREAM(streamInfo.stream, "kT_stream");
    // Allocate arrays whose length does not depend on user inputs
    initAllocation();

    while (!pSchedSupport->kinematicShouldJoin) {
        {
            std::unique_lock<std::mutex> lock(pSchedSupport->kinematicStartLock);
            while (!pSchedSupport->kinematicStarted) {
                pSchedSupport->cv_KinematicStartLock.wait(lock);
            }
            // Ensure that we wait for start signal on next iteration
            pSchedSupport->kinematicStarted = false;
            // The following is executed when kT and dT are being destroyed
            if (pSchedSupport->kinematicShouldJoin) {
                break;
            }
        }

        // Run a while loop producing stuff in each iteration; once produced, it should be made available to the dynamic
        // via memcpy
        while (!pSchedSupport->dynamicDone) {
            // Before producing something, a new work order should be in place. Wait on it.
            if (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh.load(std::memory_order_acquire)) {
                timers.GetTimer("Wait for dT update").start();
                pSchedSupport->schedulingStats.nTimesKinematicHeldBack++;
                std::unique_lock<std::mutex> lock(pSchedSupport->kinematicCanProceed);

                // kT never got locked in here indefinitely because, dT will always send a cv_KinematicCanProceed signal
                // AFTER setting dynamicDone to true, if dT is about to finish
                while (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh.load(std::memory_order_acquire)) {
                    // Loop to avoid spurious wakeups
                    pSchedSupport->cv_KinematicCanProceed.wait(lock);
                }
                timers.GetTimer("Wait for dT update").stop();

                // In the case where this weak-up call is at the destructor (dT has been executing without notifying the
                // end of user calls, aka running DoDynamics), we don't have to do CD one more time, just break
                if (kTShouldReset) {
                    break;
                }
            }

            {
                std::lock_guard<std::mutex> order_lock(pSchedSupport->kinematicOrderStateLock);
                if (!pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh.load(std::memory_order_acquire)) {
                    continue;
                }
                pSchedSupport->kinematicOrderClaimed.store(true, std::memory_order_release);
                pSchedSupport->kinematicProduceSourceStamp.store(
                    pSchedSupport->kinematicOrderIssuedStamp.load(std::memory_order_acquire), std::memory_order_release);
                pSchedSupport->kinematicProduceUsableDrift.store(
                    pSchedSupport->kinematicOrderUsableDrift.load(std::memory_order_acquire), std::memory_order_release);
            }

            DEME_NVTX_RANGE("kT::cycle");  // Do net consider waiting
            timers.GetTimer("Unpack updates from dT").start();
            // Getting here means that new `work order' data has been provided
            {
                DEME_NVTX_RANGE("kT::unpackFromDT");
                unpackMyBuffer();
                // pSchedSupport->schedulingStats.nKinematicReceives++;
            }
            timers.GetTimer("Unpack updates from dT").stop();

            // Make it clear that the data for most recent work order has been used, in case there is interest in
            // updating it
            {
                std::lock_guard<std::mutex> order_lock(pSchedSupport->kinematicOrderStateLock);
            pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh.store(false, std::memory_order_release);
                pSchedSupport->kinematicOrderClaimed.store(false, std::memory_order_release);
            }

            // figure out the amount of shared mem
            // cudaDeviceGetAttribute.cudaDevAttrMaxSharedMemoryPerBlock

            // kT's main task, contact detection.
            // For auto-adjusting bin size, this part of code is encapsuled in an accumulative timer.
            CDAccumTimer.Begin();
            DEME_NVTX_RANGE("kT::contactDetection");
            contactDetection(bin_sphere_kernels, bin_triangle_kernels, sphere_contact_kernels, sphTri_contact_kernels,
                             granData, simParams, solverFlags, verbosity, idPrimitiveA, idPrimitiveB,
                             contactTypePrimitive, previous_idPrimitiveA, previous_idPrimitiveB,
                             previous_contactTypePrimitive, contactPersistency, contactMapping, idPatchA, idPatchB,
                             previous_idPatchA, previous_idPatchB, contactTypePatch, previous_contactTypePatch,
                             contactPatchIsland, previous_contactPatchIsland, typeStartCountPatchMap, geomToPatchMap,
                             streamInfo.stream, solverScratchSpace, timers, stateParams);
            CDAccumTimer.End();
            pSchedSupport->kinematicMaxSphInBin.store((uint64_t)stateParams.maxSphFoundInBin, std::memory_order_relaxed);
            pSchedSupport->kinematicMaxTriInBin.store((uint64_t)stateParams.maxTriFoundInBin, std::memory_order_relaxed);
            pSchedSupport->kinematicAvgPrimitiveContacts.store(stateParams.avgCntsPerPrimitive,
                                                               std::memory_order_relaxed);

            timers.GetTimer("Send to dT buffer").start();
            {
                DEME_NVTX_RANGE("kT::sendToDT");
                // kT will reflect on how good the choice of parameters is
                calibrateParams();
                sendToTheirBuffer();
            }
            const bool same_dev = (streamInfo.device == dT->streamInfo.device);
            if (same_dev && kT_to_dT_BufferReadyEvent) {
                DEME_GPU_CALL(cudaEventRecord(kT_to_dT_BufferReadyEvent, streamInfo.stream));
                auto* payload = new DynamicProduceReadyPayload{pSchedSupport};
                DEME_GPU_CALL(cudaLaunchHostFunc(streamInfo.stream, NotifyDynamicProduceReady, payload));
            } else {
                pSchedSupport->dynamicOwned_Prod2ConsBuffer_isFresh.store(true, std::memory_order_release);
                pSchedSupport->cv_DynamicCanProceed.notify_all();
            }
            pSchedSupport->schedulingStats.nDynamicUpdates++;
            timers.GetTimer("Send to dT buffer").stop();

            // std::cout << "kT host mem usage: " << pretty_format_bytes(estimateHostMemUsage()) << std::endl;
            // std::cout << "kT device mem usage: " << pretty_format_bytes(estimateDeviceMemUsage()) << std::endl;
            // solverScratchSpace.printVectorUsage();
        }

        // In case the dynamic is hanging in there...
        pSchedSupport->cv_DynamicCanProceed.notify_all();

        // When getting here, kT has finished one user call (although perhaps not at the end of the user script)
        {
            std::lock_guard<std::mutex> lock(pPagerToMain->mainCanProceed);
            pPagerToMain->userCallDone = true;
            pPagerToMain->cv_mainCanProceed.notify_all();
        }
    }
}

void DEMKinematicThread::getTiming(std::vector<std::string>& names, std::vector<double>& vals) {
    names = timer_names;
    for (const auto& name : timer_names) {
        vals.push_back(timers.GetTimer(name).GetTimeSeconds());
    }
}

void DEMKinematicThread::changeFamily(unsigned int ID_from, unsigned int ID_to) {
    family_t ID_from_impl = ID_from;
    family_t ID_to_impl = ID_to;

    migrateFamilyToHost();
    std::replace_if(
        familyID.getHostVector().begin(), familyID.getHostVector().end(),
        [ID_from_impl](family_t& i) { return i == ID_from_impl; }, ID_to_impl);
    familyID.toDevice();
}

void DEMKinematicThread::changeOwnerSizes(const std::vector<bodyID_t>& IDs, const std::vector<float>& factors) {
    // Set the gpu for this thread
    cudaSetDevice(streamInfo.device);
    // cudaStream_t new_stream;
    // cudaStreamCreate(&new_stream);

    // First get IDs and factors to device side
    size_t IDSize = IDs.size() * sizeof(bodyID_t);
    bodyID_t* dIDs = (bodyID_t*)solverScratchSpace.allocateTempVector("dIDs", IDSize);
    DEME_GPU_CALL(cudaMemcpy(dIDs, IDs.data(), IDSize, cudaMemcpyHostToDevice));
    size_t factorSize = factors.size() * sizeof(float);
    float* dFactors = (float*)solverScratchSpace.allocateTempVector("dFactors", factorSize);
    DEME_GPU_CALL(cudaMemcpy(dFactors, factors.data(), factorSize, cudaMemcpyHostToDevice));

    size_t idBoolSize = (size_t)simParams->nOwnerBodies * sizeof(notStupidBool_t);
    size_t ownerFactorSize = (size_t)simParams->nOwnerBodies * sizeof(float);
    // Bool table for whether this owner should change
    notStupidBool_t* idBool = (notStupidBool_t*)solverScratchSpace.allocateTempVector("idBool", idBoolSize);
    DEME_GPU_CALL(cudaMemset(idBool, 0, idBoolSize));
    float* ownerFactors = (float*)solverScratchSpace.allocateTempVector("ownerFactors", ownerFactorSize);

    // Mark on the bool array those owners that need a change
    markOwnerToChange(idBool, ownerFactors, dIDs, dFactors, (size_t)IDs.size(), streamInfo.stream);

    // Change the size of the sphere components in question
    modifyComponents<DEMDataKT>(&granData, idBool, ownerFactors, (size_t)simParams->nSpheresGM, streamInfo.stream);

    solverScratchSpace.finishUsingTempVector("dIDs");
    solverScratchSpace.finishUsingTempVector("dFactors");
    solverScratchSpace.finishUsingTempVector("idBool");
    solverScratchSpace.finishUsingTempVector("ownerFactors");
    // cudaStreamDestroy(new_stream);

    // Update them back to host
    relPosSphereX.toHost();
    relPosSphereY.toHost();
    relPosSphereZ.toHost();
    radiiSphere.toHost();
}

void DEMKinematicThread::startThread() {
    std::lock_guard<std::mutex> lock(pSchedSupport->kinematicStartLock);
    pSchedSupport->kinematicStarted = true;
    pSchedSupport->cv_KinematicStartLock.notify_one();
}

void DEMKinematicThread::breakWaitingStatus() {
    // dynamicDone == true and cv_KinematicCanProceed should ensure kT breaks to the outer loop
    pSchedSupport->dynamicDone = true;
    // We distrubed kinematicOwned_Cons2ProdBuffer_isFresh and kTShouldReset here, but it matters not, as when
    // breakWaitingStatus is called, they will always be reset to default soon
    {
        std::lock_guard<std::mutex> order_lock(pSchedSupport->kinematicOrderStateLock);
    pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh.store(true, std::memory_order_release);
        pSchedSupport->kinematicOrderClaimed.store(false, std::memory_order_release);
    }
    kTShouldReset = true;

    std::lock_guard<std::mutex> lock(pSchedSupport->kinematicCanProceed);
    pSchedSupport->cv_KinematicCanProceed.notify_one();
}

void DEMKinematicThread::resetUserCallStat() {
    // Reset kT stats variables, making ready for next user call
    {
        std::lock_guard<std::mutex> order_lock(pSchedSupport->kinematicOrderStateLock);
    pSchedSupport->kinematicOwned_Cons2ProdBuffer_isFresh.store(false, std::memory_order_release);
        pSchedSupport->kinematicOrderClaimed.store(false, std::memory_order_release);
    }
    kTShouldReset = false;
    // My ingredient production date is... unknown now
    pSchedSupport->kinematicIngredProdDateStamp = -1;
    pSchedSupport->kinematicOrderIssuedStamp = -1;
    pSchedSupport->kinematicProduceSourceStamp = -1;
    pSchedSupport->kinematicOrderUsableDrift = -1;
    pSchedSupport->kinematicProduceUsableDrift = -1;

    // We also reset the CD timer (for adjusting bin size)
    CDAccumTimer.Clear();
    // Reset bin size change speed
    stateParams.binCurrentChangeRate = 0.;
}

size_t DEMKinematicThread::estimateDeviceMemUsage() const {
    return m_approxDeviceBytesUsed;
}

size_t DEMKinematicThread::estimateHostMemUsage() const {
    return m_approxHostBytesUsed;
}

// Put sim data array pointers in place
void DEMKinematicThread::packDataPointers() {
    familyID.bindDevicePointer(&(granData->familyID));
    voxelID.bindDevicePointer(&(granData->voxelID));
    locX.bindDevicePointer(&(granData->locX));
    locY.bindDevicePointer(&(granData->locY));
    locZ.bindDevicePointer(&(granData->locZ));
    oriQw.bindDevicePointer(&(granData->oriQw));
    oriQx.bindDevicePointer(&(granData->oriQx));
    oriQy.bindDevicePointer(&(granData->oriQy));
    oriQz.bindDevicePointer(&(granData->oriQz));
    granData->ownerBoundRadius = nullptr;
    granData->ownerMeshShellHalfThickness = nullptr;
    ownerCylGhostActive.bindDevicePointer(&(granData->ownerCylGhostActive));
    marginSizeSphere.bindDevicePointer(&(granData->marginSizeSphere));
    marginSizeTriangle.bindDevicePointer(&(granData->marginSizeTriangle));
    marginSizeAnalytical.bindDevicePointer(&(granData->marginSizeAnalytical));
    idPrimitiveA.bindDevicePointer(&(granData->idPrimitiveA));
    idPrimitiveB.bindDevicePointer(&(granData->idPrimitiveB));
    contactTypePrimitive.bindDevicePointer(&(granData->contactTypePrimitive));
    contactPersistency.bindDevicePointer(&(granData->contactPersistency));
    previous_idPrimitiveA.bindDevicePointer(&(granData->previous_idPrimitiveA));
    previous_idPrimitiveB.bindDevicePointer(&(granData->previous_idPrimitiveB));
    previous_contactTypePrimitive.bindDevicePointer(&(granData->previous_contactTypePrimitive));
    contactMapping.bindDevicePointer(&(granData->contactMapping));

    // NEW: Bind separate patch ID and mapping array pointers
    idPatchA.bindDevicePointer(&(granData->idPatchA));
    idPatchB.bindDevicePointer(&(granData->idPatchB));
    previous_idPatchA.bindDevicePointer(&(granData->previous_idPatchA));
    previous_idPatchB.bindDevicePointer(&(granData->previous_idPatchB));
    contactTypePatch.bindDevicePointer(&(granData->contactTypePatch));
    previous_contactTypePatch.bindDevicePointer(&(granData->previous_contactTypePatch));
    contactPatchIsland.bindDevicePointer(&(granData->contactPatchIsland));
    previous_contactPatchIsland.bindDevicePointer(&(granData->previous_contactPatchIsland));
    geomToPatchMap.bindDevicePointer(&(granData->geomToPatchMap));

    familyMaskMatrix.bindDevicePointer(&(granData->familyMasks));
    familyExtraMarginSize.bindDevicePointer(&(granData->familyExtraMarginSize));

    // The offset info that indexes into the template arrays
    ownerClumpBody.bindDevicePointer(&(granData->ownerClumpBody));
    clumpComponentOffset.bindDevicePointer(&(granData->clumpComponentOffset));
    clumpComponentOffsetExt.bindDevicePointer(&(granData->clumpComponentOffsetExt));
    ownerAnalBody.bindDevicePointer(&(granData->ownerAnalBody));

    // Mesh-related
    ownerTriMesh.bindDevicePointer(&(granData->ownerTriMesh));
    ownerTriStart.bindDevicePointer(&(granData->ownerTriStart));
    ownerTriCount.bindDevicePointer(&(granData->ownerTriCount));
    ownerIsBigMesh.bindDevicePointer(&(granData->ownerIsBigMesh));
    bigMeshOwners.bindDevicePointer(&(granData->bigMeshOwners));
    ownerBigMeshLeafStart.bindDevicePointer(&(granData->ownerBigMeshLeafStart));
    ownerBigMeshLeafCount.bindDevicePointer(&(granData->ownerBigMeshLeafCount));
    bigMeshLeafOwner.bindDevicePointer(&(granData->bigMeshLeafOwner));
    bigMeshLeafTriStart.bindDevicePointer(&(granData->bigMeshLeafTriStart));
    bigMeshLeafTriCount.bindDevicePointer(&(granData->bigMeshLeafTriCount));
    bigMeshLeafTriIDs.bindDevicePointer(&(granData->bigMeshLeafTriIDs));
    bigMeshLeafLocalMin.bindDevicePointer(&(granData->bigMeshLeafLocalMin));
    bigMeshLeafLocalMax.bindDevicePointer(&(granData->bigMeshLeafLocalMax));
    bigMeshLeafMaxCentroidRadius.bindDevicePointer(&(granData->bigMeshLeafMaxCentroidRadius));
    bigMeshLeafMaxExpandCoeff.bindDevicePointer(&(granData->bigMeshLeafMaxExpandCoeff));
    bigMeshLeafMargin.bindDevicePointer(&(granData->bigMeshLeafMargin));
    ownerMeshConvex.bindDevicePointer(&(granData->ownerMeshConvex));
    ownerMeshNeverWinner.bindDevicePointer(&(granData->ownerMeshNeverWinner));
    triPatchID.bindDevicePointer(&(granData->triPatchID));
    triNeighborIndex.bindDevicePointer(&(granData->triNeighborIndex));
    triNeighbor1.bindDevicePointer(&(granData->triNeighbor1));
    triNeighbor2.bindDevicePointer(&(granData->triNeighbor2));
    triNeighbor3.bindDevicePointer(&(granData->triNeighbor3));
    relPosNode1.bindDevicePointer(&(granData->relPosNode1));
    relPosNode2.bindDevicePointer(&(granData->relPosNode2));
    relPosNode3.bindDevicePointer(&(granData->relPosNode3));

    // Template array pointers
    radiiSphere.bindDevicePointer(&(granData->radiiSphere));
    relPosSphereX.bindDevicePointer(&(granData->relPosSphereX));
    relPosSphereY.bindDevicePointer(&(granData->relPosSphereY));
    relPosSphereZ.bindDevicePointer(&(granData->relPosSphereZ));

    granData->ts = stateParams.ts.getDevicePointer();
    granData->maxDrift = stateParams.maxDrift.getDevicePointer();
    granData->useFixedMargin = stateParams.useFixedMargin.getDevicePointer();
}

void DEMKinematicThread::migrateDataToDevice() {
    familyID.toDeviceAsync(streamInfo.stream);
    voxelID.toDeviceAsync(streamInfo.stream);
    locX.toDeviceAsync(streamInfo.stream);
    locY.toDeviceAsync(streamInfo.stream);
    locZ.toDeviceAsync(streamInfo.stream);
    oriQw.toDeviceAsync(streamInfo.stream);
    oriQx.toDeviceAsync(streamInfo.stream);
    oriQy.toDeviceAsync(streamInfo.stream);
    oriQz.toDeviceAsync(streamInfo.stream);
    ownerCylGhostActive.toDeviceAsync(streamInfo.stream);
    idPrimitiveA.toDeviceAsync(streamInfo.stream);
    idPrimitiveB.toDeviceAsync(streamInfo.stream);
    contactTypePrimitive.toDeviceAsync(streamInfo.stream);
    contactPersistency.toDeviceAsync(streamInfo.stream);
    previous_idPrimitiveA.toDeviceAsync(streamInfo.stream);
    previous_idPrimitiveB.toDeviceAsync(streamInfo.stream);
    previous_contactTypePrimitive.toDeviceAsync(streamInfo.stream);
    contactMapping.toDeviceAsync(streamInfo.stream);
    previous_idPatchA.toDeviceAsync(streamInfo.stream);
    previous_idPatchB.toDeviceAsync(streamInfo.stream);
    contactTypePatch.toDeviceAsync(streamInfo.stream);
    previous_contactTypePatch.toDeviceAsync(streamInfo.stream);
    contactPatchIsland.toDeviceAsync(streamInfo.stream);
    previous_contactPatchIsland.toDeviceAsync(streamInfo.stream);
    familyMaskMatrix.toDeviceAsync(streamInfo.stream);
    familyExtraMarginSize.toDeviceAsync(streamInfo.stream);

    ownerClumpBody.toDeviceAsync(streamInfo.stream);
    clumpComponentOffset.toDeviceAsync(streamInfo.stream);
    clumpComponentOffsetExt.toDeviceAsync(streamInfo.stream);
    ownerAnalBody.toDeviceAsync(streamInfo.stream);

    ownerTriMesh.toDeviceAsync(streamInfo.stream);
    ownerTriStart.toDeviceAsync(streamInfo.stream);
    ownerTriCount.toDeviceAsync(streamInfo.stream);
    ownerIsBigMesh.toDeviceAsync(streamInfo.stream);
    bigMeshOwners.toDeviceAsync(streamInfo.stream);
    ownerBigMeshLeafStart.toDeviceAsync(streamInfo.stream);
    ownerBigMeshLeafCount.toDeviceAsync(streamInfo.stream);
    bigMeshLeafOwner.toDeviceAsync(streamInfo.stream);
    bigMeshLeafTriStart.toDeviceAsync(streamInfo.stream);
    bigMeshLeafTriCount.toDeviceAsync(streamInfo.stream);
    bigMeshLeafTriIDs.toDeviceAsync(streamInfo.stream);
    bigMeshLeafLocalMin.toDeviceAsync(streamInfo.stream);
    bigMeshLeafLocalMax.toDeviceAsync(streamInfo.stream);
    bigMeshLeafMaxCentroidRadius.toDeviceAsync(streamInfo.stream);
    bigMeshLeafMaxExpandCoeff.toDeviceAsync(streamInfo.stream);
    bigMeshLeafMargin.toDeviceAsync(streamInfo.stream);
    ownerMeshConvex.toDeviceAsync(streamInfo.stream);
    ownerMeshNeverWinner.toDeviceAsync(streamInfo.stream);
    triPatchID.toDeviceAsync(streamInfo.stream);
    triNeighborIndex.toDeviceAsync(streamInfo.stream);
    triNeighbor1.toDeviceAsync(streamInfo.stream);
    triNeighbor2.toDeviceAsync(streamInfo.stream);
    triNeighbor3.toDeviceAsync(streamInfo.stream);
    relPosNode1.toDeviceAsync(streamInfo.stream);
    relPosNode2.toDeviceAsync(streamInfo.stream);
    relPosNode3.toDeviceAsync(streamInfo.stream);

    radiiSphere.toDeviceAsync(streamInfo.stream);
    relPosSphereX.toDeviceAsync(streamInfo.stream);
    relPosSphereY.toDeviceAsync(streamInfo.stream);
    relPosSphereZ.toDeviceAsync(streamInfo.stream);

    // Might not be necessary... but it's a big call anyway, let's sync
    syncMemoryTransfer();
}

void DEMKinematicThread::migrateFamilyToHost() {
    if (solverFlags.canFamilyChangeOnDevice) {
        familyID.toHost();
    }
}

void DEMKinematicThread::migrateDeviceModifiableInfoToHost() {
    migrateFamilyToHost();
}

void DEMKinematicThread::packTransferPointers(DEMDynamicThread*& dT) {
    // Set the pointers to dT owned buffers
    granData->pDTOwnedBuffer_nPrimitiveContacts = &(dT->nPrimitiveContactPairs_buffer);
    granData->pDTOwnedBuffer_nPatchContacts = &(dT->nPatchContactPairs_buffer);
    granData->ownerBoundRadius = dT->ownerBoundRadius.data();
    granData->ownerMeshShellHalfThickness = nullptr;
    for (size_t owner = 0; owner < dT->simParams->nOwnerBodies; owner++) {
        if (dT->ownerMeshShellHalfThickness[owner] > DEME_TINY_FLOAT) {
            granData->ownerMeshShellHalfThickness = dT->ownerMeshShellHalfThickness.data();
            break;
        }
    }
    const int write_idx = dT->kt_write_buf;
    granData->pDTOwnedBuffer_idPrimitiveA = dT->idPrimitiveA_buffer[write_idx].data();
    granData->pDTOwnedBuffer_idPrimitiveB = dT->idPrimitiveB_buffer[write_idx].data();
    granData->pDTOwnedBuffer_contactType = dT->contactTypePrimitive_buffer[write_idx].data();
    granData->pDTOwnedBuffer_geomToPatchMap = dT->geomToPatchMap_buffer[write_idx].data();

    // NEW: Set pointers for separate patch arrays
    granData->pDTOwnedBuffer_idPatchA = dT->idPatchA_buffer[write_idx].data();
    granData->pDTOwnedBuffer_idPatchB = dT->idPatchB_buffer[write_idx].data();
    granData->pDTOwnedBuffer_contactTypePatch = dT->contactTypePatch_buffer[write_idx].data();
    granData->pDTOwnedBuffer_contactPatchIsland = dT->contactPatchIsland_buffer[write_idx].data();
    granData->pDTOwnedBuffer_contactMapping = dT->contactMapping_buffer[write_idx].data();
}

void DEMKinematicThread::setSimParams(unsigned char nvXp2,
                                      unsigned char nvYp2,
                                      unsigned char nvZp2,
                                      float l,
                                      double voxelSize,
                                      double binSize,
                                      binID_t nbX,
                                      binID_t nbY,
                                      binID_t nbZ,
                                      float3 LBFPoint,
                                      float3 user_box_min,
                                      float3 user_box_max,
                                      float3 G,
                                      double ts_size,
                                      float expand_factor,
                                      float approx_max_vel,
                                      double max_tritri_penetration,
                                      float expand_safety_param,
                                      float expand_safety_adder,
                                      bool use_angvel_margin,
                                      const std::set<std::string>& contact_wildcards,
                                      const std::set<std::string>& owner_wildcards,
                                      const std::set<std::string>& geo_wildcards) {
    simParams->nvXp2 = nvXp2;
    simParams->nvYp2 = nvYp2;
    simParams->nvZp2 = nvZp2;
    simParams->l = l;
    simParams->voxelSize = voxelSize;
    simParams->LBFX = LBFPoint.x;
    simParams->LBFY = LBFPoint.y;
    simParams->LBFZ = LBFPoint.z;
    simParams->Gx = G.x;
    simParams->Gy = G.y;
    simParams->Gz = G.z;
    simParams->capTriTriPenetration = max_tritri_penetration;
    simParams->nbX = nbX;
    simParams->nbY = nbY;
    simParams->nbZ = nbZ;
    simParams->userBoxMin = user_box_min;
    simParams->userBoxMax = user_box_max;

    simParams->dyn.binSize = binSize;
    simParams->dyn.inv_binSize = 1. / binSize;
    simParams->dyn.h = ts_size;
    simParams->dyn.beta = expand_factor;  // If beta is auto-adapting, this assignment has no effect
    simParams->dyn.approxMaxVel = approx_max_vel;
    simParams->dyn.expSafetyMulti = expand_safety_param;
    simParams->dyn.expSafetyAdder = expand_safety_adder;
    simParams->useAngVelMargin = use_angvel_margin ? 1 : 0;

    simParams->nContactWildcards = contact_wildcards.size();
    simParams->nOwnerWildcards = owner_wildcards.size();
    simParams->nGeoWildcards = geo_wildcards.size();
}

void DEMKinematicThread::allocateGPUArrays(size_t nOwnerBodies,
                                           size_t nOwnerClumps,
                                           unsigned int nExtObj,
                                           size_t nTriMeshes,
                                           size_t nSpheresGM,
                                           size_t nTriGM,
                                           size_t nTriNeighbors,
                                           unsigned int nAnalGM,
                                           size_t nExtraContacts,
                                           unsigned int nMassProperties,
                                           unsigned int nClumpTopo,
                                           unsigned int nClumpComponents,
                                           unsigned int nJitifiableClumpComponents,
                                           unsigned int nMatTuples) {
    DEME_GPU_CALL(cudaSetDevice(streamInfo.device));

    // Sizes of these arrays
    simParams->nSpheresGM = nSpheresGM;
    simParams->nTriGM = nTriGM;
    simParams->nAnalGM = nAnalGM;
    simParams->nOwnerBodies = nOwnerBodies;
    simParams->nOwnerClumps = nOwnerClumps;
    simParams->nExtObj = nExtObj;
    simParams->nTriMeshes = nTriMeshes;
    simParams->nBigMeshOwners = 0;
    simParams->maxBigMeshOwnerTriCount = 0;
    simParams->bigMeshOwnerTriThreshold = 65536;
    simParams->nBigMeshBVHLeaves = 0;
    simParams->bigMeshBVHLeafTriCap = 64;
    simParams->nDistinctMassProperties = nMassProperties;
    simParams->nDistinctClumpBodyTopologies = nClumpTopo;
    simParams->nJitifiableClumpComponents = nJitifiableClumpComponents;
    simParams->nDistinctClumpComponents = nClumpComponents;
    simParams->nMatTuples = nMatTuples;

    // Resize the family mask `matrix' (in fact it is flattened)
    DEME_DUAL_ARRAY_RESIZE(familyMaskMatrix, (NUM_AVAL_FAMILIES + 1) * NUM_AVAL_FAMILIES / 2, DONT_PREVENT_CONTACT);

    // Resize to the number of clumps
    DEME_DUAL_ARRAY_RESIZE(familyID, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(voxelID, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(locX, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(locY, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(locZ, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(oriQw, nOwnerBodies, 1);
    DEME_DUAL_ARRAY_RESIZE(oriQx, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(oriQy, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(oriQz, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(ownerCylGhostActive, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(ownerMeshConvex, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(ownerMeshNeverWinner, nOwnerBodies, 0);
    DEME_DEVICE_ARRAY_RESIZE(marginSizeSphere, nSpheresGM);
    DEME_DEVICE_ARRAY_RESIZE(marginSizeAnalytical, nAnalGM);
    DEME_DEVICE_ARRAY_RESIZE(marginSizeTriangle, nTriGM);

    // Transfer buffer arrays
    // It is cudaMalloc-ed memory, not on host, because we want explicit locality control of buffers
    {
        // These buffers should be on dT, to save dT access time
        DEME_GPU_CALL(cudaSetDevice(dT->streamInfo.device));
        DEME_DEVICE_ARRAY_RESIZE(voxelID_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(locX_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(locY_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(locZ_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(oriQ0_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(oriQ1_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(oriQ2_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(oriQ3_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(ownerCylGhostActive_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(absVel_buffer, nOwnerBodies);
        DEME_DEVICE_ARRAY_RESIZE(absAngVel_buffer, nOwnerBodies);
        // DEME_ADVISE_DEVICE(voxelID_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(locX_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(locY_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(locZ_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(oriQ0_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(oriQ1_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(oriQ2_buffer, dT->streamInfo.device);
        // DEME_ADVISE_DEVICE(oriQ3_buffer, dT->streamInfo.device);

        if (solverFlags.canFamilyChangeOnDevice) {
            // DEME_ADVISE_DEVICE(familyID_buffer, dT->streamInfo.device);
            DEME_DEVICE_ARRAY_RESIZE(familyID_buffer, nOwnerBodies);
        }

        DEME_DEVICE_ARRAY_RESIZE(relPosNode1_buffer, nTriGM);
        DEME_DEVICE_ARRAY_RESIZE(relPosNode2_buffer, nTriGM);
        DEME_DEVICE_ARRAY_RESIZE(relPosNode3_buffer, nTriGM);

        // Unset the device change we just did
        DEME_GPU_CALL(cudaSetDevice(streamInfo.device));
    }

    // Resize to the number of spheres (or plus num of triangle facets)
    DEME_DUAL_ARRAY_RESIZE(ownerClumpBody, nSpheresGM, 0);

    // Resize to the number of triangle facets
    DEME_DUAL_ARRAY_RESIZE(ownerTriMesh, nTriGM, 0);
    DEME_DUAL_ARRAY_RESIZE(ownerTriStart, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(ownerTriCount, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(ownerIsBigMesh, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(bigMeshOwners, nTriMeshes > 0 ? nTriMeshes : 1, 0);
    DEME_DUAL_ARRAY_RESIZE(ownerBigMeshLeafStart, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(ownerBigMeshLeafCount, nOwnerBodies, 0);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafOwner, 1, 0);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafTriStart, 1, 0);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafTriCount, 1, 0);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafTriIDs, 1, 0);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafLocalMin, 1, make_float3(0.f, 0.f, 0.f));
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafLocalMax, 1, make_float3(0.f, 0.f, 0.f));
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafMaxCentroidRadius, 1, 0.f);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafMaxExpandCoeff, 1, 0.f);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafMargin, 1, 0.f);
    DEME_DUAL_ARRAY_RESIZE(triPatchID, nTriGM, 0);
    DEME_DUAL_ARRAY_RESIZE(triNeighborIndex, nTriGM, NULL_BODYID);
    DEME_DUAL_ARRAY_RESIZE(triNeighbor1, nTriNeighbors, NULL_BODYID);
    DEME_DUAL_ARRAY_RESIZE(triNeighbor2, nTriNeighbors, NULL_BODYID);
    DEME_DUAL_ARRAY_RESIZE(triNeighbor3, nTriNeighbors, NULL_BODYID);
    DEME_DUAL_ARRAY_RESIZE(relPosNode1, nTriGM, make_float3(0));
    DEME_DUAL_ARRAY_RESIZE(relPosNode2, nTriGM, make_float3(0));
    DEME_DUAL_ARRAY_RESIZE(relPosNode3, nTriGM, make_float3(0));

    // And analytical geometry owner array
    DEME_DUAL_ARRAY_RESIZE(ownerAnalBody, nAnalGM, 0);

    if (solverFlags.useClumpJitify) {
        DEME_DUAL_ARRAY_RESIZE(clumpComponentOffset, nSpheresGM, 0);
        // This extended component offset array can hold offset numbers even for big clumps (whereas
        // clumpComponentOffset is typically uint_8, so it may not). If a sphere's component offset index falls in this
        // range then it is not jitified, and the kernel needs to look for it in the global memory.
        DEME_DUAL_ARRAY_RESIZE(clumpComponentOffsetExt, nSpheresGM, 0);
        // Resize to the length of the clump templates
        DEME_DUAL_ARRAY_RESIZE(radiiSphere, nClumpComponents, 0);
        DEME_DUAL_ARRAY_RESIZE(relPosSphereX, nClumpComponents, 0);
        DEME_DUAL_ARRAY_RESIZE(relPosSphereY, nClumpComponents, 0);
        DEME_DUAL_ARRAY_RESIZE(relPosSphereZ, nClumpComponents, 0);
    } else {
        DEME_DUAL_ARRAY_RESIZE(radiiSphere, nSpheresGM, 0);
        DEME_DUAL_ARRAY_RESIZE(relPosSphereX, nSpheresGM, 0);
        DEME_DUAL_ARRAY_RESIZE(relPosSphereY, nSpheresGM, 0);
        DEME_DUAL_ARRAY_RESIZE(relPosSphereZ, nSpheresGM, 0);
    }

    // Arrays for kT produced contact info
    // The following several arrays will have variable sizes, so here we only used a good initial value. My estimate of
    // total contact pairs is ~n, and I think the max is 6n.
    {
        size_t cnt_arr_size = DEME_MAX(*solverScratchSpace.numPrevContacts, INITIAL_CONTACT_ARRAY_SIZE);
        DEME_DUAL_ARRAY_RESIZE(idPrimitiveA, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(idPrimitiveB, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(contactTypePrimitive, cnt_arr_size, NOT_A_CONTACT);
        DEME_DUAL_ARRAY_RESIZE(contactPersistency, cnt_arr_size, CONTACT_NOT_PERSISTENT);

        DEME_DUAL_ARRAY_RESIZE(idPatchA, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(idPatchB, cnt_arr_size, 0);
        DEME_DUAL_ARRAY_RESIZE(contactTypePatch, cnt_arr_size, NOT_A_CONTACT);
        DEME_DUAL_ARRAY_RESIZE(contactPatchIsland, cnt_arr_size, NULL_BODYID);
        DEME_DUAL_ARRAY_RESIZE(geomToPatchMap, cnt_arr_size, 0);

        if (!solverFlags.isHistoryless) {
            // No need to resize prev_primitive ID arrays: used only when persistency is enabled and that is rare
            // DEME_DUAL_ARRAY_RESIZE(previous_idPrimitiveA, cnt_arr_size, 0);
            // DEME_DUAL_ARRAY_RESIZE(previous_idPrimitiveB, cnt_arr_size, 0);
            // DEME_DUAL_ARRAY_RESIZE(previous_contactTypePrimitive, cnt_arr_size, NOT_A_CONTACT);
            DEME_DUAL_ARRAY_RESIZE(contactMapping, cnt_arr_size, NULL_MAPPING_PARTNER);
            DEME_DUAL_ARRAY_RESIZE(previous_idPatchA, cnt_arr_size, 0);
            DEME_DUAL_ARRAY_RESIZE(previous_idPatchB, cnt_arr_size, 0);
            DEME_DUAL_ARRAY_RESIZE(previous_contactTypePatch, cnt_arr_size, NOT_A_CONTACT);
            DEME_DUAL_ARRAY_RESIZE(previous_contactPatchIsland, cnt_arr_size, NULL_BODYID);
        }
    }
}

void DEMKinematicThread::registerPolicies(const std::vector<notStupidBool_t>& family_mask_matrix) {
    // Store family mask
    for (size_t i = 0; i < family_mask_matrix.size(); i++)
        familyMaskMatrix[i] = family_mask_matrix.at(i);
}

void DEMKinematicThread::populateEntityArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                              const std::vector<unsigned int>& input_ext_obj_family,
                                              const std::vector<unsigned int>& input_mesh_obj_family,
                                              const std::vector<notStupidBool_t>& input_mesh_obj_convex,
                                              const std::vector<notStupidBool_t>& input_mesh_obj_never_winner,
                                              const std::vector<unsigned int>& input_mesh_facet_owner,
                                              const std::vector<bodyID_t>& input_mesh_facet_patch,
                                              const std::vector<bodyID_t>& input_mesh_facet_neighbor1,
                                              const std::vector<bodyID_t>& input_mesh_facet_neighbor2,
                                              const std::vector<bodyID_t>& input_mesh_facet_neighbor3,
                                              const std::vector<DEMTriangle>& input_mesh_facets,
                                              const ClumpTemplateFlatten& clump_templates,
                                              const std::vector<unsigned int>& ext_obj_comp_num,
                                              size_t nExistOwners,
                                              size_t nExistSpheres,
                                              size_t nExistingFacets,
                                              size_t nExistingMeshPatches,
                                              size_t nExistingTriNeighbors) {
    // All the input vectors should have the same length, nClumpTopo
    size_t k = 0;
    std::vector<unsigned int> prescans_comp;

    if (solverFlags.useClumpJitify) {
        prescans_comp.push_back(0);
        for (auto elem : clump_templates.spRadii) {
            for (auto radius : elem) {
                radiiSphere[k] = radius;
                k++;
            }
            prescans_comp.push_back(k);
        }
        prescans_comp.pop_back();
        k = 0;

        for (auto elem : clump_templates.spRelPos) {
            for (auto loc : elem) {
                relPosSphereX[k] = loc.x;
                relPosSphereY[k] = loc.y;
                relPosSphereZ[k] = loc.z;
                k++;
            }
        }
    }

    k = 0;
    // float3 LBF;
    // LBF.x = simParams->LBFX;
    // LBF.y = simParams->LBFY;
    // LBF.z = simParams->LBFZ;
    // Now load clump init info
    std::vector<unsigned int> input_clump_types;
    {
        std::vector<unsigned int> input_clump_family;
        // Flatten the input clump batches (because by design we transfer flatten clump info to GPU)
        for (const auto& a_batch : input_clump_batches) {
            // Decode type number and flatten
            std::vector<unsigned int> type_marks(a_batch->GetNumClumps());
            for (size_t i = 0; i < a_batch->GetNumClumps(); i++) {
                type_marks.at(i) = a_batch->types.at(i)->mark;
            }
            input_clump_types.insert(input_clump_types.end(), type_marks.begin(), type_marks.end());
            input_clump_family.insert(input_clump_family.end(), a_batch->families.begin(), a_batch->families.end());
        }

        for (size_t i = 0; i < input_clump_types.size(); i++) {
            auto type_of_this_clump = input_clump_types.at(i);

            // auto this_CoM_coord = input_clump_xyz.at(i) - LBF; // kT don't have to init owner xyz
            auto this_clump_no_sp_radii = clump_templates.spRadii.at(type_of_this_clump);
            auto this_clump_no_sp_relPos = clump_templates.spRelPos.at(type_of_this_clump);

            for (size_t j = 0; j < this_clump_no_sp_radii.size(); j++) {
                ownerClumpBody[nExistSpheres + k] = nExistOwners + i;

                // Depending on whether we jitify or flatten
                if (solverFlags.useClumpJitify) {
                    // This component offset, is it too large that can't live in the jitified array?
                    unsigned int this_comp_offset = prescans_comp.at(type_of_this_clump) + j;
                    clumpComponentOffsetExt[nExistSpheres + k] = this_comp_offset;
                    if (this_comp_offset < simParams->nJitifiableClumpComponents) {
                        clumpComponentOffset[nExistSpheres + k] = this_comp_offset;
                    } else {
                        // If not, an indicator will be put there
                        clumpComponentOffset[nExistSpheres + k] = RESERVED_CLUMP_COMPONENT_OFFSET;
                    }
                } else {
                    radiiSphere[nExistSpheres + k] = this_clump_no_sp_radii.at(j);
                    const float3 relPos = this_clump_no_sp_relPos.at(j);
                    relPosSphereX[nExistSpheres + k] = relPos.x;
                    relPosSphereY[nExistSpheres + k] = relPos.y;
                    relPosSphereZ[nExistSpheres + k] = relPos.z;
                }

                k++;
            }

            family_t this_family_num = input_clump_family.at(i);
            familyID[nExistOwners + i] = this_family_num;
        }
    }

    // Analytical objs
    k = 0;
    size_t owner_offset_for_ext_obj = nExistOwners + input_clump_types.size();
    for (size_t i = 0; i < input_ext_obj_family.size(); i++) {
        // For each analytical geometry component of this obj, it needs to know its owner number
        for (size_t j = 0; j < ext_obj_comp_num.at(i); j++) {
            ownerAnalBody[k] = i + owner_offset_for_ext_obj;
            k++;
        }

        family_t this_family_num = input_ext_obj_family.at(i);
        familyID[i + owner_offset_for_ext_obj] = this_family_num;
    }

    // Mesh objs
    size_t owner_offset_for_mesh_obj = owner_offset_for_ext_obj + input_ext_obj_family.size();
    // k for indexing the triangle facets
    k = 0;
    size_t neighbor_write = nExistingTriNeighbors;
    for (size_t i = 0; i < input_mesh_obj_family.size(); i++) {
        // Per-facet info
        const size_t local_tri_begin = k;
        size_t this_facet_owner = input_mesh_facet_owner.at(k);
        const bool mesh_needs_neighbors =
            !(input_mesh_obj_convex.at(this_facet_owner) != 0 && input_mesh_obj_never_winner.at(this_facet_owner) != 0);
        for (; k < input_mesh_facet_owner.size(); k++) {
            // input_mesh_facet_owner run length is the num of facets in this mesh entity
            if (input_mesh_facet_owner.at(k) != this_facet_owner)
                break;
            const size_t global_tri = nExistingFacets + k;
            ownerTriMesh[global_tri] = owner_offset_for_mesh_obj + this_facet_owner;
            triPatchID[global_tri] = nExistingMeshPatches + input_mesh_facet_patch.at(k);
            if (mesh_needs_neighbors) {
                triNeighborIndex[global_tri] = neighbor_write;
                triNeighbor1[neighbor_write] = input_mesh_facet_neighbor1.at(k);
                triNeighbor2[neighbor_write] = input_mesh_facet_neighbor2.at(k);
                triNeighbor3[neighbor_write] = input_mesh_facet_neighbor3.at(k);
                neighbor_write++;
            } else {
                triNeighborIndex[global_tri] = NULL_BODYID;
            }
            DEMTriangle this_tri = input_mesh_facets.at(k);
            relPosNode1[global_tri] = this_tri.p1;
            relPosNode2[global_tri] = this_tri.p2;
            relPosNode3[global_tri] = this_tri.p3;
        }

        const bodyID_t owner_id = owner_offset_for_mesh_obj + i;
        ownerTriStart[owner_id] = nExistingFacets + local_tri_begin;
        ownerTriCount[owner_id] = k - local_tri_begin;
        family_t this_family_num = input_mesh_obj_family.at(i);
        familyID[owner_id] = this_family_num;
        ownerMeshConvex[owner_id] = input_mesh_obj_convex.at(i);
        ownerMeshNeverWinner[owner_id] = input_mesh_obj_never_winner.at(i);
        // DEME_DEBUG_PRINTF("kT just loaded a mesh in family %u", +(this_family_num));
        // DEME_DEBUG_PRINTF("Number of triangle facets loaded thus far: %zu", k);
    }
}

void DEMKinematicThread::initGPUArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                       const std::vector<unsigned int>& input_ext_obj_family,
                                       const std::vector<unsigned int>& input_mesh_obj_family,
                                       const std::vector<notStupidBool_t>& input_mesh_obj_convex,
                                       const std::vector<notStupidBool_t>& input_mesh_obj_never_winner,
                                       const std::vector<unsigned int>& input_mesh_facet_owner,
                                       const std::vector<bodyID_t>& input_mesh_facet_patch,
                                       const std::vector<bodyID_t>& input_mesh_facet_neighbor1,
                                       const std::vector<bodyID_t>& input_mesh_facet_neighbor2,
                                       const std::vector<bodyID_t>& input_mesh_facet_neighbor3,
                                       const std::vector<DEMTriangle>& input_mesh_facets,
                                       const std::vector<unsigned int>& ext_obj_comp_num,
                                       const std::vector<notStupidBool_t>& family_mask_matrix,
                                       const ClumpTemplateFlatten& clump_templates) {
    // Get the info into the GPU memory from the host side. Can this process be more efficient? Maybe, but it's
    // initialization anyway.

    registerPolicies(family_mask_matrix);

    populateEntityArrays(input_clump_batches, input_ext_obj_family, input_mesh_obj_family, input_mesh_obj_convex,
                         input_mesh_obj_never_winner, input_mesh_facet_owner, input_mesh_facet_patch,
                         input_mesh_facet_neighbor1, input_mesh_facet_neighbor2, input_mesh_facet_neighbor3,
                         input_mesh_facets, clump_templates, ext_obj_comp_num, 0, 0, 0, 0, 0);
    rebuildBigMeshOwnerMetadata();
}

void DEMKinematicThread::updateClumpMeshArrays(const std::vector<std::shared_ptr<DEMClumpBatch>>& input_clump_batches,
                                               const std::vector<unsigned int>& input_ext_obj_family,
                                               const std::vector<unsigned int>& input_mesh_obj_family,
                                               const std::vector<notStupidBool_t>& input_mesh_obj_convex,
                                               const std::vector<notStupidBool_t>& input_mesh_obj_never_winner,
                                               const std::vector<unsigned int>& input_mesh_facet_owner,
                                               const std::vector<bodyID_t>& input_mesh_facet_patch,
                                               const std::vector<bodyID_t>& input_mesh_facet_neighbor1,
                                               const std::vector<bodyID_t>& input_mesh_facet_neighbor2,
                                               const std::vector<bodyID_t>& input_mesh_facet_neighbor3,
                                               const std::vector<DEMTriangle>& input_mesh_facets,
                                               const std::vector<unsigned int>& ext_obj_comp_num,
                                               const std::vector<notStupidBool_t>& family_mask_matrix,
                                               const ClumpTemplateFlatten& clump_templates,
                                               size_t nExistingOwners,
                                               size_t nExistingClumps,
                                               size_t nExistingSpheres,
                                               size_t nExistingTriMesh,
                                               size_t nExistingFacets,
                                               size_t nExistingTriNeighbors,
                                               size_t nExistingPatches,
                                               unsigned int nExistingObj,
                                               unsigned int nExistingAnalGM) {
    populateEntityArrays(input_clump_batches, input_ext_obj_family, input_mesh_obj_family, input_mesh_obj_convex,
                         input_mesh_obj_never_winner, input_mesh_facet_owner, input_mesh_facet_patch,
                         input_mesh_facet_neighbor1, input_mesh_facet_neighbor2, input_mesh_facet_neighbor3,
                         input_mesh_facets, clump_templates, ext_obj_comp_num, nExistingOwners, nExistingSpheres,
                         nExistingFacets, nExistingPatches, nExistingTriNeighbors);
    rebuildBigMeshOwnerMetadata();
}

void DEMKinematicThread::rebuildBigMeshOwnerMetadata() {
    const bodyID_t mesh_owner_start = simParams->nOwnerBodies - simParams->nTriMeshes;
    bodyID_t write_idx = 0;
    bodyID_t write_leaf = 0;
    bodyID_t max_tri_count = 0;
    bodyID_t total_leaf_tri_ids = 0;
    const bodyID_t leaf_cap = DEME_MAX((bodyID_t)1, simParams->bigMeshBVHLeafTriCap);

    for (bodyID_t owner = 0; owner < simParams->nOwnerBodies; owner++) {
        ownerIsBigMesh[owner] = 0;
        ownerBigMeshLeafStart[owner] = 0;
        ownerBigMeshLeafCount[owner] = 0;
        if (owner >= mesh_owner_start && ownerTriCount[owner] >= simParams->bigMeshOwnerTriThreshold) {
            ownerIsBigMesh[owner] = 1;
            write_idx++;
            max_tri_count = DEME_MAX(max_tri_count, ownerTriCount[owner]);
            write_leaf += (ownerTriCount[owner] + leaf_cap - 1) / leaf_cap;
            total_leaf_tri_ids += ownerTriCount[owner];
        }
    }

    DEME_DUAL_ARRAY_RESIZE(bigMeshOwners, write_idx > 0 ? write_idx : 1, 0);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafOwner, write_leaf > 0 ? write_leaf : 1, 0);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafTriStart, write_leaf > 0 ? write_leaf : 1, 0);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafTriCount, write_leaf > 0 ? write_leaf : 1, 0);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafTriIDs, total_leaf_tri_ids > 0 ? total_leaf_tri_ids : 1, 0);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafLocalMin, write_leaf > 0 ? write_leaf : 1, make_float3(0.f, 0.f, 0.f));
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafLocalMax, write_leaf > 0 ? write_leaf : 1, make_float3(0.f, 0.f, 0.f));
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafMaxCentroidRadius, write_leaf > 0 ? write_leaf : 1, 0.f);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafMaxExpandCoeff, write_leaf > 0 ? write_leaf : 1, 0.f);
    DEME_DUAL_ARRAY_RESIZE(bigMeshLeafMargin, write_leaf > 0 ? write_leaf : 1, 0.f);

    write_idx = 0;
    write_leaf = 0;
    bodyID_t tri_id_write = 0;
    std::vector<MortonTriEntry> morton_sorted_tris;
    morton_sorted_tris.reserve((size_t)DEME_MAX(max_tri_count, (bodyID_t)1));

    for (bodyID_t owner = 0; owner < simParams->nOwnerBodies; owner++) {
        if (!(owner >= mesh_owner_start && ownerIsBigMesh[owner])) {
            continue;
        }

        bigMeshOwners[write_idx++] = owner;
        ownerBigMeshLeafStart[owner] = write_leaf;

        const bodyID_t tri_start = ownerTriStart[owner];
        const bodyID_t tri_count = ownerTriCount[owner];
        morton_sorted_tris.clear();
        morton_sorted_tris.reserve((size_t)tri_count);

        float3 bbox_min = make_float3(DEME_HUGE_FLOAT, DEME_HUGE_FLOAT, DEME_HUGE_FLOAT);
        float3 bbox_max = make_float3(-DEME_HUGE_FLOAT, -DEME_HUGE_FLOAT, -DEME_HUGE_FLOAT);
        for (bodyID_t local = 0; local < tri_count; local++) {
            const bodyID_t triID = tri_start + local;
            const float3 p1 = relPosNode1[triID];
            const float3 p2 = relPosNode2[triID];
            const float3 p3 = relPosNode3[triID];
            const float3 c = hostTriangleCentroid(p1, p2, p3);
            bbox_min.x = std::min(bbox_min.x, c.x);
            bbox_min.y = std::min(bbox_min.y, c.y);
            bbox_min.z = std::min(bbox_min.z, c.z);
            bbox_max.x = std::max(bbox_max.x, c.x);
            bbox_max.y = std::max(bbox_max.y, c.y);
            bbox_max.z = std::max(bbox_max.z, c.z);
            morton_sorted_tris.push_back({triID, 0u});
        }

        const float span_x = std::max(bbox_max.x - bbox_min.x, 1e-9f);
        const float span_y = std::max(bbox_max.y - bbox_min.y, 1e-9f);
        const float span_z = std::max(bbox_max.z - bbox_min.z, 1e-9f);
        for (auto& entry : morton_sorted_tris) {
            const float3 p1 = relPosNode1[entry.triID];
            const float3 p2 = relPosNode2[entry.triID];
            const float3 p3 = relPosNode3[entry.triID];
            const float3 c = hostTriangleCentroid(p1, p2, p3);
            const uint32_t qx = quantizeUnitTo10Bits((c.x - bbox_min.x) / span_x);
            const uint32_t qy = quantizeUnitTo10Bits((c.y - bbox_min.y) / span_y);
            const uint32_t qz = quantizeUnitTo10Bits((c.z - bbox_min.z) / span_z);
            entry.key = morton3D10(qx, qy, qz);
        }
        std::stable_sort(morton_sorted_tris.begin(), morton_sorted_tris.end(),
                         [](const MortonTriEntry& a, const MortonTriEntry& b) {
                             if (a.key != b.key) {
                                 return a.key < b.key;
                             }
                             return a.triID < b.triID;
                         });

        const bodyID_t n_leaves = (tri_count + leaf_cap - 1) / leaf_cap;
        ownerBigMeshLeafCount[owner] = n_leaves;
        for (bodyID_t li = 0; li < n_leaves; li++) {
            const bodyID_t leaf_tri_offset = li * leaf_cap;
            const bodyID_t remaining = tri_count - leaf_tri_offset;
            const bodyID_t leaf_tri_count = DEME_MIN(leaf_cap, remaining);
            bigMeshLeafOwner[write_leaf] = owner;
            bigMeshLeafTriStart[write_leaf] = tri_id_write;
            bigMeshLeafTriCount[write_leaf] = leaf_tri_count;

            float3 leaf_min = make_float3(DEME_HUGE_FLOAT, DEME_HUGE_FLOAT, DEME_HUGE_FLOAT);
            float3 leaf_max = make_float3(-DEME_HUGE_FLOAT, -DEME_HUGE_FLOAT, -DEME_HUGE_FLOAT);
            float max_centroid_radius = 0.f;
            float max_expand_coeff = 0.f;

            for (bodyID_t local = 0; local < leaf_tri_count; local++) {
                const bodyID_t triID = morton_sorted_tris[leaf_tri_offset + local].triID;
                bigMeshLeafTriIDs[tri_id_write++] = triID;

                const float3 p1 = relPosNode1[triID];
                const float3 p2 = relPosNode2[triID];
                const float3 p3 = relPosNode3[triID];
                leaf_min.x = std::min(leaf_min.x, std::min(p1.x, std::min(p2.x, p3.x)));
                leaf_min.y = std::min(leaf_min.y, std::min(p1.y, std::min(p2.y, p3.y)));
                leaf_min.z = std::min(leaf_min.z, std::min(p1.z, std::min(p2.z, p3.z)));
                leaf_max.x = std::max(leaf_max.x, std::max(p1.x, std::max(p2.x, p3.x)));
                leaf_max.y = std::max(leaf_max.y, std::max(p1.y, std::max(p2.y, p3.y)));
                leaf_max.z = std::max(leaf_max.z, std::max(p1.z, std::max(p2.z, p3.z)));

                const float3 centroid = hostTriangleCentroid(p1, p2, p3);
                max_centroid_radius = std::max(max_centroid_radius, (float)hostLength3(centroid));
                max_expand_coeff = std::max(max_expand_coeff, hostTriangleExpandCoeff(p1, p2, p3));
            }

            bigMeshLeafLocalMin[write_leaf] = leaf_min;
            bigMeshLeafLocalMax[write_leaf] = leaf_max;
            bigMeshLeafMaxCentroidRadius[write_leaf] = max_centroid_radius;
            bigMeshLeafMaxExpandCoeff[write_leaf] = max_expand_coeff;
            bigMeshLeafMargin[write_leaf] = 0.f;
            write_leaf++;
        }
    }

    simParams->nBigMeshOwners = write_idx;
    simParams->maxBigMeshOwnerTriCount = max_tri_count;
    simParams->nBigMeshBVHLeaves = write_leaf;
    packDataPointers();
}

void DEMKinematicThread::updatePrevContactArrays(DualStruct<DEMDataDT>& dT_data, size_t nContacts) {
    // Store the incoming info in kT's arrays
    // Note kT never had the responsibility to migrate contact info to host, even at Update, as even in this case
    // its host-side update comes from dT
    overwritePrevContactArrays(granData, dT_data, previous_idPatchA, previous_idPatchB, previous_contactTypePatch,
                               previous_contactPatchIsland, typeStartCountPatchMap, simParams, solverScratchSpace,
                               streamInfo.stream, nContacts);
    DEME_DEBUG_PRINTF("Number of contacts after a user-manual contact load: %zu", nContacts);
    DEME_DEBUG_PRINTF("Number of spheres after a user-manual contact load: %zu", (size_t)simParams->nSpheresGM);
}

void DEMKinematicThread::jitifyKernels(const std::unordered_map<std::string, std::string>& Subs,
                                       const std::vector<std::string>& JitifyOptions) {
    // First one is bin_sphere_kernels kernels, which figure out the bin--sphere touch pairs
    {
        bin_sphere_kernels = std::make_shared<JitHelper::CachedProgram>(JitHelper::buildProgram(
            "DEMBinSphereKernels", JitHelper::KERNEL_DIR / "DEMBinSphereKernels.cu", Subs, JitifyOptions));
    }
    // Then CD kernels
    {
        sphere_contact_kernels = std::make_shared<JitHelper::CachedProgram>(
            JitHelper::buildProgram("DEMContactKernels_SphereSphere",
                                    JitHelper::KERNEL_DIR / "DEMContactKernels_SphereSphere.cu", Subs, JitifyOptions));
    }
    // Then triangle--bin intersection-related kernels
    {
        bin_triangle_kernels = std::make_shared<JitHelper::CachedProgram>(JitHelper::buildProgram(
            "DEMBinTriangleKernels", JitHelper::KERNEL_DIR / "DEMBinTriangleKernels.cu", Subs, JitifyOptions));
    }
    // Then sphere--triangle/tri--tri contact detection-related kernels
    {
        sphTri_contact_kernels = std::make_shared<JitHelper::CachedProgram>(std::move(JitHelper::buildProgram(
            "DEMContactKernels_SphTri_TriTri", JitHelper::KERNEL_DIR / "DEMContactKernels_SphTri_TriTri.cu", Subs,
            JitifyOptions)));
    }
    // Then misc.
    {
        misc_kernels = std::make_shared<JitHelper::CachedProgram>(std::move(JitHelper::buildProgram(
            "DEMKinematicMisc", JitHelper::KERNEL_DIR / "DEMKinematicMisc.cu", Subs, JitifyOptions)));
    }
    prewarmKernels();
}

void DEMKinematicThread::initAllocation() {
    DEME_DUAL_ARRAY_RESIZE(familyExtraMarginSize, NUM_AVAL_FAMILIES, 0);
}

void DEMKinematicThread::deallocateEverything() {
    // Device and dual array will have their destructor called once kT is gone, so this is not needed
}

void DEMKinematicThread::setOwnerFamily(bodyID_t ownerID, family_t fam, bodyID_t n) {
    familyID.setVal(std::vector<family_t>(n, fam), ownerID);
}

void DEMKinematicThread::setTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& triangles) {
    for (size_t i = 0; i < triangles.size(); i++) {
        relPosNode1[start + i] = triangles[i].p1;
        relPosNode2[start + i] = triangles[i].p2;
        relPosNode3[start + i] = triangles[i].p3;
    }
    relPosNode1.toDeviceAsync(streamInfo.stream, start, triangles.size());
    relPosNode2.toDeviceAsync(streamInfo.stream, start, triangles.size());
    relPosNode3.toDeviceAsync(streamInfo.stream, start, triangles.size());
    syncMemoryTransfer();
}

// It's true that this method is never used in either kT or dT
void DEMKinematicThread::updateTriNodeRelPos(size_t start, const std::vector<DEMTriangle>& updates) {
    for (size_t i = 0; i < updates.size(); i++) {
        relPosNode1[start + i] += updates[i].p1;
        relPosNode2[start + i] += updates[i].p2;
        relPosNode3[start + i] += updates[i].p3;
    }
    relPosNode1.toDeviceAsync(streamInfo.stream, start, updates.size());
    relPosNode2.toDeviceAsync(streamInfo.stream, start, updates.size());
    relPosNode3.toDeviceAsync(streamInfo.stream, start, updates.size());
    syncMemoryTransfer();
}

void DEMKinematicThread::prewarmKernels() {
    if (bin_sphere_kernels) {
        bin_sphere_kernels->kernel("populateBinSphereTouchingPairs").instantiate();
        bin_sphere_kernels->kernel("getNumberOfBinsEachSphereTouches").instantiate();
    }
    if (sphere_contact_kernels) {
        sphere_contact_kernels->kernel("populateSphereContactPairsEachBin").instantiate();
        sphere_contact_kernels->kernel("getNumberOfSphereContactsEachBin").instantiate();
    }
    if (bin_triangle_kernels) {
        bin_triangle_kernels->kernel("precomputeTriangleSandwichData").instantiate();
        bin_triangle_kernels->kernel("precomputeTriangleSandwichDataBigMeshOwners").instantiate();
        bin_triangle_kernels->kernel("computeBigMeshLeafWorldBounds").instantiate();
        bin_triangle_kernels->kernel("countBinsEachBigMeshLeafTouches").instantiate();
        bin_triangle_kernels->kernel("countActiveBinsEachBigMeshLeafTouches").instantiate();
        bin_triangle_kernels->kernel("populateBigMeshLeafActiveBinPairs").instantiate();
        bin_triangle_kernels->kernel("populateBigMeshLeafBinPairs").instantiate();
        bin_triangle_kernels->kernel("countTriPairsEachBigMeshLeafBinTouch").instantiate();
        bin_triangle_kernels->kernel("populateTriPairsFromBigMeshLeafBinPairs").instantiate();
        bin_triangle_kernels->kernel("markCylPeriodicOwnerGhosts").instantiate();
        bin_triangle_kernels->kernel("getNumberOfBinsEachTriangleTouches").instantiate();
        bin_triangle_kernels->kernel("populateBinTriangleTouchingPairs").instantiate();
    }
    if (sphTri_contact_kernels) {
        sphTri_contact_kernels->kernel("getNumberOfTriangleContactsEachBin").instantiate();
        sphTri_contact_kernels->kernel("populateTriangleContactsEachBin").instantiate();
    }
    if (history_kernels) {
        history_kernels->kernel("buildPersistentMap").instantiate();
        history_kernels->kernel("rearrangeMapping").instantiate();
        history_kernels->kernel("lineNumbers").instantiate();
        history_kernels->kernel("fillRunLengthArray").instantiate();
        history_kernels->kernel("convertToAndFrom").instantiate();
    }
    if (misc_kernels) {
        misc_kernels->kernel("computeMarginFromAbsv_implSph").instantiate();
        misc_kernels->kernel("computeMarginFromAbsv_implTri").instantiate();
        misc_kernels->kernel("computeMarginFromAbsv_implBigMeshLeaf").instantiate();
        misc_kernels->kernel("fillFixedMarginBigMeshLeaf").instantiate();
        misc_kernels->kernel("computeMarginFromAbsv_implAnal").instantiate();
    }
}

}  // namespace deme
