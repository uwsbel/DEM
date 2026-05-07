//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#include <algorithms/DEMStaticDeviceSubroutines.h>
#include <algorithms/DEMStaticDeviceUtilities.cuh>

#include <kernel/DEMHelperKernels.cuh>
#include <kernel/DEMCollisionKernels_SphTri_TriTri.cuh>

// Reject insane local contact points that are actually packed-double storage (overlap depth / area).
// This prevents catastrophic torque explosions when a slot is misclassified as patch-contact.
__device__ inline bool saneLocalCP(const float3& p) {
    // packed-double storage often yields absurd magnitudes (1e10+), while real local CP is on mm–cm scale.
    // Also reject NaN/inf.
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) return false;
    const float m2 = p.x*p.x + p.y*p.y + p.z*p.z;
    // 1 meter in local space is already absurd for your cube sizes; threshold can be tuned.
    return (m2 < 1.0f);
}

__device__ inline bool saneLocalCPWithBound(const float3& p, float max_norm) {
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) return false;
    max_norm = fmaxf(max_norm, 1e-6f);
    const float m2 = p.x * p.x + p.y * p.y + p.z * p.z;
    return (m2 <= max_norm * max_norm);
}

namespace deme {

__device__ __forceinline__ void fetchTriangleWorldNodesForTriTri(const deme::DEMSimParams* simParams,
                                                                 const deme::DEMDataDT* granData,
                                                                 const deme::bodyID_t triID,
                                                                 double3& a,
                                                                 double3& b,
                                                                 double3& c) {
    const deme::bodyID_t owner = granData->ownerTriMesh[triID];
    double3 ownerPos;
    voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(ownerPos.x, ownerPos.y, ownerPos.z, granData->voxelID[owner],
                                                         granData->locX[owner], granData->locY[owner], granData->locZ[owner],
                                                         simParams->nvXp2, simParams->nvYp2, simParams->voxelSize, simParams->l);
    ownerPos.x += simParams->LBFX;
    ownerPos.y += simParams->LBFY;
    ownerPos.z += simParams->LBFZ;
    const float4 q = make_float4(granData->oriQx[owner], granData->oriQy[owner], granData->oriQz[owner], granData->oriQw[owner]);

    a = to_double3(granData->relPosNode1[triID]);
    b = to_double3(granData->relPosNode2[triID]);
    c = to_double3(granData->relPosNode3[triID]);
    applyOriQToVector3<double, float>(a.x, a.y, a.z, q.w, q.x, q.y, q.z);
    applyOriQToVector3<double, float>(b.x, b.y, b.z, q.w, q.x, q.y, q.z);
    applyOriQToVector3<double, float>(c.x, c.y, c.z, q.w, q.x, q.y, q.z);
    a += ownerPos;
    b += ownerPos;
    c += ownerPos;
}

// Rescue path for seam slivers: retry the same mutually-submerged overlap reconstruction with a progressively
// relaxed clipping tolerance. This preserves the geometric model (submerged polygons) and avoids the large-area
// artifacts of full-triangle projection.
__device__ __forceinline__ double projectedTriangleOverlapAreaCpAlongNormal(const double3& A0,
                                                                            const double3& A1,
                                                                            const double3& A2,
                                                                            const double3& B0,
                                                                            const double3& B1,
                                                                            const double3& B2,
                                                                            double3 nCommon,
                                                                            double3& contactPoint) {
    double projArea = 0.0;
    contactPoint = make_zero3<double3>();

    const double geomScale = local_length_scale6<double3, double>(A0, A1, A2, B0, B1, B2);
    const double baseEps = rel_len_tol<double>() * geomScale;

    const double nLen2 = dot(nCommon, nCommon);
    if (nLen2 <= (double)(DEME_TINY_FLOAT * DEME_TINY_FLOAT)) {
        return 0.0;
    }
    nCommon = nCommon * (1.0 / sqrt(nLen2));

    const double3 nA = cross(A1 - A0, A2 - A0);
    const double3 nB = cross(B1 - B0, B2 - B0);
    const double nALen2 = dot(nA, nA);
    const double nBLen2 = dot(nB, nB);
    if (nALen2 <= (double)(DEME_TINY_FLOAT * DEME_TINY_FLOAT) ||
        nBLen2 <= (double)(DEME_TINY_FLOAT * DEME_TINY_FLOAT)) {
        return 0.0;
    }

    const double denA = dot(nA, nCommon);
    const double denB = dot(nB, nCommon);
    if (absT(denA) <= (double)DEME_TINY_FLOAT || absT(denB) <= (double)DEME_TINY_FLOAT) {
        return 0.0;
    }

    double3 u, v;
    build_plane_basis_from_normal<double3, double>(nCommon, u, v);
    const double3 O = (A0 + B0) * 0.5;
    const double cA = dot(nA, A0 - O);
    const double cB = dot(nB, B0 - O);
    const double alphaA = dot(nA, u), betaA = dot(nA, v);
    const double alphaB = dot(nB, u), betaB = dot(nB, v);

    const double epsMults[4] = {1.0, 4.0, 16.0, 64.0};
    for (int attempt = 0; attempt < 4; attempt++) {
        const double planeEps = baseEps * epsMults[attempt];
        const double eps2d = planeEps;

        double3 aPen3[4], bPen3[4];
        int nAPen = clip_triangle_by_plane_keep_negative<double3, double>(A0, A1, A2, B0, nB, planeEps, aPen3);
        int nBPen = clip_triangle_by_plane_keep_negative<double3, double>(B0, B1, B2, A0, nA, planeEps, bPen3);
        if (nAPen < 3 || nBPen < 3) {
            continue;
        }

        double aX[8], aY[8], bX[8], bY[8];
        nAPen = project_poly_3d_to_2d<double3, double>(aPen3, nAPen, O, u, v, aX, aY, eps2d);
        nBPen = project_poly_3d_to_2d<double3, double>(bPen3, nBPen, O, u, v, bX, bY, eps2d);
        if (nAPen < 3 || nBPen < 3) {
            continue;
        }

        double outX[8], outY[8];
        const int nPoly = clip_convex_poly_2d<double>(bX, bY, nBPen, aX, aY, nAPen, outX, outY, eps2d);
        if (nPoly < 3) {
            continue;
        }

        double cx = 0.0, cy = 0.0;
        if (!polygon_area_centroid_2d<double>(outX, outY, nPoly, projArea, cx, cy)) {
            continue;
        }
        if (projArea <= 0.0) {
            continue;
        }

        const double wA = (cA - alphaA * cx - betaA * cy) / denA;
        const double wB = (cB - alphaB * cx - betaB * cy) / denB;
        contactPoint = O + u * cx + v * cy + nCommon * ((wA + wB) * 0.5);
        return projArea;
    }

    return 0.0;
}

__global__ void getContactForcesConcerningOwners_impl(float3* d_points,
                                                      float3* d_forces,
                                                      float3* d_torques,
                                                      unsigned long long* d_numUsefulCnt,
                                                      bodyID_t* d_ownerIDs,
                                                      size_t IDListSize,
                                                      DEMSimParams* simParams,
                                                      DEMDataDT* granData,
                                                      size_t numCnt,
                                                      bool need_torque,
                                                      bool torque_in_local) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numCnt) {
        
        const bool patch_space = (granData->contactTypePatch && granData->idPatchA && granData->idPatchB);
        if (!patch_space) {
            // IMPORTANT: Patch and primitive contact arrays do not share the same index space.
            // This kernel is intended for PATCH contacts only. Handling primitive contacts must use a separate kernel launch.
            return;
        }
        const contact_t typeContact = granData->contactTypePatch[i];
        if (typeContact == NOT_A_CONTACT) {
            return;
        }
        bodyID_t geoA_raw = granData->idPatchA[i];
        bodyID_t geoB_raw = granData->idPatchB[i];
        bool ghostA = false;
        bool ghostB = false;
        bool ghostA_neg = false;
        bool ghostB_neg = false;
        bodyID_t geoA = cylPeriodicDecodeID(geoA_raw, ghostA, ghostA_neg);
        bodyID_t geoB = cylPeriodicDecodeID(geoB_raw, ghostB, ghostB_neg);
        bodyID_t ownerA = DEME_GET_PATCH_OWNER_ID(geoA, decodeTypeA(typeContact));
        bodyID_t ownerB = DEME_GET_PATCH_OWNER_ID(geoB, decodeTypeB(typeContact));
        bool AorB;  // true for A, false for B
        if (cuda_binary_search<bodyID_t, ssize_t>(d_ownerIDs, ownerA, 0, IDListSize - 1)) {
            AorB = true;
        } else if (cuda_binary_search<bodyID_t, ssize_t>(d_ownerIDs, ownerB, 0, IDListSize - 1)) {
            AorB = false;
        } else {
            return;
        }

        float3 force = granData->contactForces[i];
        float3 torque_only_force = make_float3(0.f, 0.f, 0.f);
        if (need_torque) {
            torque_only_force = granData->contactTorque_convToForce[i];
        }
        {
            float mag = length(force) + length(torque_only_force);
            if (mag < DEME_TINY_FLOAT)
                return;
        }

        // It's a contact we need to output...
        unsigned long long writeIndex = atomicAdd(d_numUsefulCnt, 1);
        float3 cntPnt = granData->contactPointGeometryA[i];
        double3 CoM;
        float4 oriQ;
        bodyID_t ownerID;
        int wrapShiftA = ghostA ? (ghostA_neg ? -1 : 1) : 0;
        int wrapShiftB = ghostB ? (ghostB_neg ? -1 : 1) : 0;
        if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f && granData->ownerCylWrapOffset) {
            wrapShiftA += granData->ownerCylWrapOffset[ownerA];
            wrapShiftB += granData->ownerCylWrapOffset[ownerB];
        }
        if (AorB) {
            ownerID = ownerA;
            if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f && wrapShiftA != 0) {
                float cos_theta = 1.f, sin_theta = 0.f, cos_half = 1.f, sin_half = 0.f;
                cylPeriodicShiftTrig(-wrapShiftA, simParams, cos_theta, sin_theta, cos_half, sin_half);
                force = cylPeriodicRotate(force, make_float3(0.f, 0.f, 0.f), simParams->cylPeriodicAxisVec,
                                          simParams->cylPeriodicU, simParams->cylPeriodicV, cos_theta, sin_theta);
                if (need_torque) {
                    torque_only_force = cylPeriodicRotate(torque_only_force, make_float3(0.f, 0.f, 0.f),
                                                          simParams->cylPeriodicAxisVec, simParams->cylPeriodicU,
                                                          simParams->cylPeriodicV, cos_theta, sin_theta);
                }
            }
        } else {
            ownerID = ownerB;
            force = -force;
            if (need_torque)
                torque_only_force = -torque_only_force;
            if (simParams->useCylPeriodic && simParams->cylPeriodicSpan > 0.f) {
                const int cpShift = wrapShiftA - wrapShiftB;
                if (cpShift != 0) {
                    float cos_theta = 1.f, sin_theta = 0.f, cos_half = 1.f, sin_half = 0.f;
                    cylPeriodicShiftTrig(cpShift, simParams, cos_theta, sin_theta, cos_half, sin_half);
                    cntPnt = cylPeriodicRotate(cntPnt, simParams->cylPeriodicOrigin, simParams->cylPeriodicAxisVec,
                                               simParams->cylPeriodicU, simParams->cylPeriodicV, cos_theta, sin_theta);
                }
                if (wrapShiftB != 0) {
                    float cos_theta = 1.f, sin_theta = 0.f, cos_half = 1.f, sin_half = 0.f;
                    cylPeriodicShiftTrig(-wrapShiftB, simParams, cos_theta, sin_theta, cos_half, sin_half);
                    force = cylPeriodicRotate(force, make_float3(0.f, 0.f, 0.f), simParams->cylPeriodicAxisVec,
                                              simParams->cylPeriodicU, simParams->cylPeriodicV, cos_theta, sin_theta);
                    if (need_torque) {
                        torque_only_force = cylPeriodicRotate(torque_only_force, make_float3(0.f, 0.f, 0.f),
                                                              simParams->cylPeriodicAxisVec, simParams->cylPeriodicU,
                                                              simParams->cylPeriodicV, cos_theta, sin_theta);
                    }
                }
            }
        }
        oriQ.w = granData->oriQw[ownerID];
        oriQ.x = granData->oriQx[ownerID];
        oriQ.y = granData->oriQy[ownerID];
        oriQ.z = granData->oriQz[ownerID];
        voxelID_t voxel = granData->voxelID[ownerID];
        subVoxelPos_t subVoxX = granData->locX[ownerID];
        subVoxelPos_t subVoxY = granData->locY[ownerID];
        subVoxelPos_t subVoxZ = granData->locZ[ownerID];
        voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(CoM.x, CoM.y, CoM.z, voxel, subVoxX, subVoxY, subVoxZ,
                                                            simParams->nvXp2, simParams->nvYp2, simParams->voxelSize,
                                                            simParams->l);
        CoM.x += simParams->LBFX;
        CoM.y += simParams->LBFY;
        CoM.z += simParams->LBFZ;
        if (need_torque) {
            float3 cntPnt_local =
                make_float3(cntPnt.x - (float)CoM.x, cntPnt.y - (float)CoM.y, cntPnt.z - (float)CoM.z);
            applyOriQToVector3<float, oriQ_t>(cntPnt_local.x, cntPnt_local.y, cntPnt_local.z, oriQ.w, -oriQ.x,
                                              -oriQ.y, -oriQ.z);
            // Final guard: reject implausibly large local lever arms for this owner.
            float max_lever = 1.0f;
            if (granData->ownerBoundRadius && ownerID != NULL_BODYID && ownerID < simParams->nOwnerBodies) {
                const float bound_r = fmaxf(granData->ownerBoundRadius[ownerID], 0.f);
                // Keep some tolerance for non-spherical geometry and contact-point scatter.
                max_lever = fmaxf(4.0f * bound_r, 5e-2f);
            }
            if (!saneLocalCPWithBound(cntPnt_local, max_lever)) {
                d_torques[writeIndex] = make_float3(0.f, 0.f, 0.f);
            } else {
                float3 myF = force + torque_only_force;
                applyOriQToVector3<float, oriQ_t>(myF.x, myF.y, myF.z, oriQ.w, -oriQ.x, -oriQ.y, -oriQ.z);
                float3 torque = cross(cntPnt_local, myF);
                if (!torque_in_local) {
                    applyOriQToVector3<float, oriQ_t>(torque.x, torque.y, torque.z, oriQ.w, oriQ.x, oriQ.y, oriQ.z);
                }
                d_torques[writeIndex] = torque;
            }
        }
        d_points[writeIndex] = cntPnt;
        d_forces[writeIndex] = force;
    }
}

void getContactForcesConcerningOwners(float3* d_points,
                                      float3* d_forces,
                                      float3* d_torques,
                                      size_t* d_numUsefulCnt,
                                      bodyID_t* d_ownerIDs,
                                      size_t IDListSize,
                                      DEMSimParams* simParams,
                                      DEMDataDT* granData,
                                      size_t numCnt,
                                      bool need_torque,
                                      bool torque_in_local,
                                      cudaStream_t& this_stream) {
    size_t blocks_needed = (numCnt + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    getContactForcesConcerningOwners_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
        d_points, d_forces, d_torques, reinterpret_cast<unsigned long long*>(d_numUsefulCnt), d_ownerIDs, IDListSize,
        simParams, granData, numCnt, need_torque, torque_in_local);
}

////////////////////////////////////////////////////////////////////////////////
// Patch-based voting kernels for mesh contact correction
////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ bool primitiveMeshScratchAvailable(const DEMDataDT* granData,
                                                              contactPairs_t contactID) {
    return granData->contactPointGeometryB &&
           contactID >= granData->contactScalarOffset &&
           contactID < granData->contactScalarOffset + granData->contactScalarCount;
}

__device__ __forceinline__ float3 primitiveMeshScratchNormal(const DEMDataDT* granData,
                                                            contactPairs_t contactID) {
    if (primitiveMeshScratchAvailable(granData, contactID)) {
        const contactPairs_t scalarID = contactID - granData->contactScalarOffset;
        return granData->contactPointGeometryB[scalarID];
    }
    return granData->contactForces[contactID];
}

__device__ __forceinline__ float3 primitiveMeshScratchContactPoint(const DEMDataDT* granData,
                                                                   contactPairs_t contactID) {
    if (primitiveMeshScratchAvailable(granData, contactID)) {
        const contactPairs_t scalarID = contactID - granData->contactScalarOffset;
        return granData->contactPointGeometryB[granData->contactScalarCount + scalarID];
    }
    return granData->contactTorque_convToForce[contactID];
}

// Kernel to compute weighted normals (normal * area / penetration) for voting
// Also prepares the area values for reduction and extracts the keys (geomToPatchMap values)

// Optimized overload: prepare weighted normals only (no temporary areas/keys arrays).
__global__ void prepareWeightedNormalsForVoting_impl(DEMDataDT* granData,
                                                          float3* weightedNormals,
                                                          contactPairs_t startOffset,
                                                          contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffset + idx;

        // Normal and geometric quantities were produced by the primitive contact kernels.
        float3 normal = primitiveMeshScratchNormal(granData, myContactID);

        // TODO: is this block necessary?

        // For tri-tri contacts, orient primitive normals to the canonical patch-pair ordering.
        // Otherwise, primitive A/B role flips can make normals cancel in patch voting.
        if (granData->contactTypePrimitive[myContactID] == TRIANGLE_TRIANGLE_CONTACT) {
            const contactPairs_t patchContactID = granData->geomToPatchMap[myContactID];
            const bodyID_t patchA = granData->idPatchA[patchContactID];
            const bodyID_t patchB = granData->idPatchB[patchContactID];

            bool ghostA = false;
            bool ghostA_neg = false;
            bool ghostB = false;
            bool ghostB_neg = false;
            const bodyID_t triA = cylPeriodicDecodeID(granData->idPrimitiveA[myContactID], ghostA, ghostA_neg);
            const bodyID_t triB = cylPeriodicDecodeID(granData->idPrimitiveB[myContactID], ghostB, ghostB_neg);

            bodyID_t triPatchA = granData->triPatchID[triA];
            bodyID_t triPatchB = granData->triPatchID[triB];
            if (ghostA) {
                triPatchA = cylPeriodicEncodeGhostID(triPatchA, ghostA_neg);
            }
            if (ghostB) {
                triPatchB = cylPeriodicEncodeGhostID(triPatchB, ghostB_neg);
            }

            // Primitive kernel normal convention is B->A in primitive ordering.
            // If primitive ordering is opposite to canonical patch ordering, flip sign.
            const bool primitiveAlignedToPatch = (triPatchA == patchA && triPatchB == patchB);
            const bool primitiveOppositeToPatch = (triPatchA == patchB && triPatchB == patchA);
            if (!primitiveAlignedToPatch && primitiveOppositeToPatch) {
                normal = -normal;
            }
        }

        // End TODO Block

        weightedNormals[idx] = make_float3(normal.x, normal.y, normal.z);
    }
}

void prepareWeightedNormalsForVoting(DEMDataDT* granData,
                                     float3* weightedNormals,
                                     contactPairs_t startOffset,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        prepareWeightedNormalsForVoting_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, weightedNormals, startOffset, count);
    }
}

// Kernel to normalize the voted normals by dividing by total area and scatter to output
// If total area is 0, set result to (0,0,0)
// Assumes uniqueKeys are sorted (CUB's ReduceByKey maintains sort order)
// Uses contactPairs_t keys (geomToPatchMap values)
__global__ void normalizeAndScatterVotedNormals_impl(float3* votedWeightedNormals,
                                                     float3* output,
                                                     contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float3 votedNormal = votedWeightedNormals[idx];
        float len2 = length2(votedNormal);
        if (len2 > 0.f) {
            // Normalize votedNormal
            votedNormal *= rsqrtf(len2);
        } else {
            // If total area is 0, set to (0,0,0) to mark no real contact
            votedNormal = make_float3(0.0f, 0.0f, 0.0f);
        }

        // Write to output at the correct position
        output[idx] = votedNormal;
    }
}

void normalizeAndScatterVotedNormals(float3* votedWeightedNormals,
                                     float3* output,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        normalizeAndScatterVotedNormals_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            votedWeightedNormals, output, count);
    }
}

__device__ __forceinline__ bool finiteDouble3(const double3& a) {
    return isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
}

__device__ __forceinline__ float3 normalizeFloat3OrZero(const float3& v) {
    const float len2 = length2(v);
    if (!(len2 > 0.f) || !isfinite(len2)) {
        return make_float3(0.f, 0.f, 0.f);
    }
    return v * rsqrtf(len2);
}

__device__ __forceinline__ bool jacobiEigenSymmetric3x3Device(const double in_A[3][3],
                                                              double eigvals[3],
                                                              double eigvecs[3][3]) {
    double A[3][3];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            A[r][c] = in_A[r][c];
            eigvecs[r][c] = (r == c) ? 1.0 : 0.0;
        }
    }
    constexpr int max_iters = 24;
    constexpr double eps = 1e-24;
    for (int it = 0; it < max_iters; it++) {
        int p = 0;
        int q = 1;
        double max_off = fabs(A[0][1]);
        const double off_02 = fabs(A[0][2]);
        const double off_12 = fabs(A[1][2]);
        if (off_02 > max_off) { p = 0; q = 2; max_off = off_02; }
        if (off_12 > max_off) { p = 1; q = 2; max_off = off_12; }
        const double diag_scale = fabs(A[0][0]) + fabs(A[1][1]) + fabs(A[2][2]) + 1.0;
        if (max_off <= diag_scale * 1e-14) break;
        const double app = A[p][p];
        const double aqq = A[q][q];
        const double apq = A[p][q];
        if (fabs(apq) <= eps) continue;
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0) ? (1.0 / (tau + sqrt(1.0 + tau * tau)))
                                      : (-1.0 / (-tau + sqrt(1.0 + tau * tau)));
        const double c = 1.0 / sqrt(1.0 + t * t);
        const double s = t * c;
        for (int k = 0; k < 3; k++) {
            if (k == p || k == q) continue;
            const double aik = A[k][p];
            const double akq = A[k][q];
            A[k][p] = c * aik - s * akq;
            A[p][k] = A[k][p];
            A[k][q] = c * akq + s * aik;
            A[q][k] = A[k][q];
        }
        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p][q] = 0.0; A[q][p] = 0.0;
        for (int k = 0; k < 3; k++) {
            const double vkp = eigvecs[k][p];
            const double vkq = eigvecs[k][q];
            eigvecs[k][p] = c * vkp - s * vkq;
            eigvecs[k][q] = s * vkp + c * vkq;
        }
    }
    eigvals[0] = A[0][0]; eigvals[1] = A[1][1]; eigvals[2] = A[2][2];
    return isfinite(eigvals[0]) && isfinite(eigvals[1]) && isfinite(eigvals[2]);
}

__device__ __forceinline__ float3 deterministicPerpendicular(const float3& t) {
    const float3 tx = make_float3(fabsf(t.x), fabsf(t.y), fabsf(t.z));
    float3 axis = make_float3(1.f, 0.f, 0.f);
    if (tx.y <= tx.x && tx.y <= tx.z) {
        axis = make_float3(0.f, 1.f, 0.f);
    } else if (tx.z <= tx.x && tx.z <= tx.y) {
        axis = make_float3(0.f, 0.f, 1.f);
    }
    return normalizeFloat3OrZero(cross(t, axis));
}

__global__ void prepareTriTriNormalsForPatchVote_impl(const DEMSimParams* simParams,
                                                      DEMDataDT* granData,
                                                      float3* orientedNormals,
                                                      contactPairs_t startOffset,
                                                      contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    const contactPairs_t myContactID = startOffset + idx;
    float3 n_raw = primitiveMeshScratchNormal(granData, myContactID);
    if (granData->contactTypePrimitive[myContactID] != TRIANGLE_TRIANGLE_CONTACT) {
        orientedNormals[idx] = n_raw;
        return;
    }

    // TODO: Check if this block is necessary

    // Keep tri-tri primitive normals consistent with canonical patch ordering.
    // Primitive A/B can swap across frames, but patch pairs are canonicalized; align signs here
    // before patch voting to avoid normal cancellation and spurious patch-force drops.
    const contactPairs_t patchContactID = granData->geomToPatchMap[myContactID];
    const bodyID_t patchA = granData->idPatchA[patchContactID];
    const bodyID_t patchB = granData->idPatchB[patchContactID];

    bool ghostA = false;
    bool ghostA_neg = false;
    bool ghostB = false;
    bool ghostB_neg = false;
    const bodyID_t triA = cylPeriodicDecodeID(granData->idPrimitiveA[myContactID], ghostA, ghostA_neg);
    const bodyID_t triB = cylPeriodicDecodeID(granData->idPrimitiveB[myContactID], ghostB, ghostB_neg);

    bodyID_t triPatchA = granData->triPatchID[triA];
    bodyID_t triPatchB = granData->triPatchID[triB];
    if (ghostA) {
        triPatchA = cylPeriodicEncodeGhostID(triPatchA, ghostA_neg);
    }
    if (ghostB) {
        triPatchB = cylPeriodicEncodeGhostID(triPatchB, ghostB_neg);
    }

    const bool primitiveAlignedToPatch = (triPatchA == patchA && triPatchB == patchB);
    const bool primitiveOppositeToPatch = (triPatchA == patchB && triPatchB == patchA);
    if (!primitiveAlignedToPatch && primitiveOppositeToPatch) {
        n_raw = -n_raw;
    }

    // End block check

    // Weight the patch-normal vote by positive penetration. This keeps the stage-0 normal
    // common across the patch, but avoids letting near-degenerate primitives dominate the vote.
    const double rawPen = granData->contactPenetration[myContactID - granData->contactScalarOffset];
    const float w = (rawPen > 0.0) ? static_cast<float>(rawPen) : 0.0f;
    orientedNormals[idx] = make_float3(n_raw.x * w, n_raw.y * w, n_raw.z * w);
}

void prepareTriTriNormalsForPatchVote(const DEMSimParams* simParams,
                                      DEMDataDT* granData,
                                      float3* orientedNormals,
                                      contactPairs_t startOffset,
                                      contactPairs_t count,
                                      cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        prepareTriTriNormalsForPatchVote_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            simParams, granData, orientedNormals, startOffset, count);
    }
}

__global__ void prepareTriTriPlaneFitAccumulators_impl(DEMDataDT* granData,
                                                       TriTriPlaneFitAccum* accumulators,
                                                       contactPairs_t startOffset,
                                                       contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    TriTriPlaneFitAccum acc{};
    const contactPairs_t myContactID = startOffset + idx;
    if (granData->contactTypePrimitive[myContactID] == TRIANGLE_TRIANGLE_CONTACT) {
        const double rawPen = granData->contactPenetration[myContactID - granData->contactScalarOffset];
        if (rawPen >= 0.0) {
            const double3 cp = to_double3(primitiveMeshScratchContactPoint(granData, myContactID));
            if (finiteDouble3(cp)) {
                const float3 n_unit = normalizeFloat3OrZero(primitiveMeshScratchNormal(granData, myContactID));
                const double w = (rawPen > 0.0) ? rawPen : 1.0;
                acc.weightSum = w;
                acc.sumPx = w * cp.x;
                acc.sumPy = w * cp.y;
                acc.sumPz = w * cp.z;
                acc.sumPxx = w * cp.x * cp.x;
                acc.sumPxy = w * cp.x * cp.y;
                acc.sumPxz = w * cp.x * cp.z;
                acc.sumPyy = w * cp.y * cp.y;
                acc.sumPyz = w * cp.y * cp.z;
                acc.sumPzz = w * cp.z * cp.z;
                acc.sumNx = w * (double)n_unit.x;
                acc.sumNy = w * (double)n_unit.y;
                acc.sumNz = w * (double)n_unit.z;
                acc.count = 1u;
            }
        }
    }
    accumulators[idx] = acc;
}

void prepareTriTriPlaneFitAccumulators(DEMDataDT* granData,
                                       TriTriPlaneFitAccum* accumulators,
                                       contactPairs_t startOffset,
                                       contactPairs_t count,
                                       cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        prepareTriTriPlaneFitAccumulators_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, accumulators, startOffset, count);
    }
}

__global__ void finalizeTriTriPatchNormalsFromPlaneFit_impl(const TriTriPlaneFitAccum* patchAccumulators,
                                                            float3* patchNormals,
                                                            contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    const TriTriPlaneFitAccum acc = patchAccumulators[idx];
    const float3 meanNormal = normalizeFloat3OrZero(make_float3((float)acc.sumNx, (float)acc.sumNy, (float)acc.sumNz));
    if (!(acc.weightSum > 0.0) || acc.count == 0u) {
        patchNormals[idx] = make_float3(0.f, 0.f, 0.f);
        return;
    }
    // With fewer than 3 witness points there is no reliable plane fit.
    // Keep the primitive-voted normal directly and avoid PCA tie/degeneracy artifacts.
    if (acc.count < 3u) {
        patchNormals[idx] = meanNormal;
        return;
    }
    const double invW = 1.0 / acc.weightSum;
    const double cx = acc.sumPx * invW;
    const double cy = acc.sumPy * invW;
    const double cz = acc.sumPz * invW;
    double C[3][3];
    C[0][0] = acc.sumPxx * invW - cx * cx;
    C[0][1] = acc.sumPxy * invW - cx * cy;
    C[0][2] = acc.sumPxz * invW - cx * cz;
    C[1][0] = C[0][1];
    C[1][1] = acc.sumPyy * invW - cy * cy;
    C[1][2] = acc.sumPyz * invW - cy * cz;
    C[2][0] = C[0][2];
    C[2][1] = C[1][2];
    C[2][2] = acc.sumPzz * invW - cz * cz;
    double eigvals[3];
    double eigvecs[3][3];
    if (!jacobiEigenSymmetric3x3Device(C, eigvals, eigvecs)) {
        patchNormals[idx] = meanNormal;
        return;
    }
    // Robust eigen index ordering (ascending) with deterministic tie handling.
    int ord0 = 0, ord1 = 1, ord2 = 2;
    if (eigvals[ord0] > eigvals[ord1]) { const int t = ord0; ord0 = ord1; ord1 = t; }
    if (eigvals[ord1] > eigvals[ord2]) { const int t = ord1; ord1 = ord2; ord2 = t; }
    if (eigvals[ord0] > eigvals[ord1]) { const int t = ord0; ord0 = ord1; ord1 = t; }
    const int min_idx = ord0;
    const int mid_idx = ord1;
    const int max_idx = ord2;
    const double max_eval = eigvals[max_idx];
    const double mid_eval = eigvals[mid_idx];
    const double geom_scale = fabs(max_eval) + fabs(mid_eval) + fabs(eigvals[min_idx]) + 1.0;
    const bool planar = (acc.count >= 3u) && isfinite(mid_eval) && (mid_eval > geom_scale * 1e-8);
    float3 n;
    if (planar) {
        n = make_float3((float)eigvecs[0][min_idx], (float)eigvecs[1][min_idx], (float)eigvecs[2][min_idx]);
        n = normalizeFloat3OrZero(n);
    } else {
        // Non-planar/near-degenerate covariance: keep the primitive-voted mean normal.
        // This avoids unstable tangent-projection artifacts in sparse/noisy contact islands.
        n = meanNormal;
    }
    if (length2(n) <= 0.f) {
        patchNormals[idx] = meanNormal;
        return;
    }
    if (length2(meanNormal) > 0.f && dot(n, meanNormal) < 0.f) {
        n = -n;
    }
    patchNormals[idx] = n;
}

void finalizeTriTriPatchNormalsFromPlaneFit(const TriTriPlaneFitAccum* patchAccumulators,
                                            float3* patchNormals,
                                            contactPairs_t count,
                                            cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        finalizeTriTriPatchNormalsFromPlaneFit_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            patchAccumulators, patchNormals, count);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Fused patch aggregation kernels (projected area, penetration, contact point)
////////////////////////////////////////////////////////////////////////////////

__global__ void recomputeTriTriAreaAndPrepareLiteAccumulators_impl(const DEMSimParams* simParams,
                                                                   DEMDataDT* granData,
                                                                   const contactPairs_t* keys,
                                                                   const float3* patchNormals,
                                                                   TriTriLiteAccum* accumulators,
                                                                   contactPairs_t startOffsetPrimitive,
                                                                   contactPairs_t startOffsetPatch,
                                                                   contactPairs_t countPatch,
                                                                   contactPairs_t countPrimitive) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= countPrimitive) {
        return;
    }

    TriTriLiteAccum acc{};
    const contactPairs_t myContactID = startOffsetPrimitive + idx;
    if (granData->contactTypePrimitive[myContactID] != TRIANGLE_TRIANGLE_CONTACT) {
        accumulators[idx] = acc;
        return;
    }

    const float3 cpFallbackStorage = primitiveMeshScratchContactPoint(granData, myContactID);
    double3 cp = to_double3(cpFallbackStorage);
    double area = 0.0;

    const contactPairs_t patchID = keys[idx];
    const bool patchValid = (patchID >= startOffsetPatch && patchID < startOffsetPatch + countPatch);
    const double rawPen = granData->contactPenetration[myContactID - granData->contactScalarOffset];
    const double posPen = (rawPen > 0.0) ? rawPen : 0.0;

    if (patchValid) {
        const float3 n = patchNormals[patchID - startOffsetPatch];
        const double nlen2 = (double)length2(n);
        if (nlen2 > (double)(DEME_TINY_FLOAT * DEME_TINY_FLOAT)) {
            // Keep penetration aggregation independent from stage-1 area reconstruction.
            // A primitive can be a valid overlap witness (positive pass-1 penetration) even when
            // its reconstructed projected area collapses numerically at a seam/sliver. If we only
            // update maxPositivePen on area>0 contributors, patch penetration can dip although the
            // contact island is still present.
            acc.maxPositivePen = posPen;
            bool ghostA = false, ghostA_neg = false, ghostB = false, ghostB_neg = false;
            const bodyID_t triA = cylPeriodicDecodeID(granData->idPrimitiveA[myContactID], ghostA, ghostA_neg);
            const bodyID_t triB = cylPeriodicDecodeID(granData->idPrimitiveB[myContactID], ghostB, ghostB_neg);
            double3 a, b, c, d, e, f;
            fetchTriangleWorldNodesForTriTri(simParams, granData, triA, a, b, c);
            fetchTriangleWorldNodesForTriTri(simParams, granData, triB, d, e, f);

            // Once the patch normal is known, a primitive can contribute positive projected area even when the
            // zero-thickness SAT/Moller pass reported a slightly negative raw penetration. These "support-only"
            // contributors are essential near mesh-to-mesh seam transitions; dropping them causes the patch area to
            // lose entire cells although the reconstructed island geometry is still valid.
            const double3 n_patch = to_double3(n);
            const double3 n_patch_neg = make_double3(-n_patch.x, -n_patch.y, -n_patch.z);
            area = area_cp_along_normal<double3, double>(a, b, c, d, e, f, n_patch, cp);
            if (area <= 0.0) {
                double3 cp2 = cp;
                const double area2 = area_cp_along_normal<double3, double>(d, e, f, a, b, c, n_patch, cp2);
                if (area2 > 0.0) {
                    area = area2;
                    cp = cp2;
                }
            }
            if (area <= 0.0) {
                // Orientation-robust retry: if patch normal orientation is locally inconsistent,
                // the submerged-polygon clipping can fail for +n but succeed for -n.
                double3 cp3 = cp;
                const double area3 = area_cp_along_normal<double3, double>(a, b, c, d, e, f, n_patch_neg, cp3);
                if (area3 > 0.0) {
                    area = area3;
                    cp = cp3;
                }
            }
            if (area <= 0.0) {
                double3 cp4 = cp;
                const double area4 = area_cp_along_normal<double3, double>(d, e, f, a, b, c, n_patch_neg, cp4);
                if (area4 > 0.0) {
                    area = area4;
                    cp = cp4;
                }
            }
            // Keep area reconstruction conservative: extra rescue passes can over-inflate
            // patch area in grazing cases where normals/penetration are already stable.
            if (area > 0.0) {
                if (posPen > 0.0) {
                    const double invn = 1.0 / sqrt(nlen2);
                    const double3 nUnit = to_double3(n) * invn;
                    cp = cp - nUnit * (0.5 * posPen);
                }
                acc.sumProjArea = area;
                acc.sumAreaWeightedCPx = cp.x * area;
                acc.sumAreaWeightedCPy = cp.y * area;
                acc.sumAreaWeightedCPz = cp.z * area;
            }
        }
    }

    granData->contactArea[myContactID - granData->contactScalarOffset] = area > 0.0 ? area : 0.0;
    if (primitiveMeshScratchAvailable(granData, myContactID)) {
        const contactPairs_t scalarID = myContactID - granData->contactScalarOffset;
        granData->contactPointGeometryB[granData->contactScalarCount + scalarID] = to_float3(cp);
    }
    granData->contactTorque_convToForce[myContactID] = to_float3(cp);
    // Do not kill the primitive candidate here. This stage only reconstructs patch area; candidate lifetime belongs to
    // kT/pass-1. Zeroing the type here turns tiny numerical seam misses into visible patch-area flicker.
    accumulators[idx] = acc;
}

void recomputeTriTriAreaAndPrepareLiteAccumulators(const DEMSimParams* simParams,
                                                   DEMDataDT* granData,
                                                   const contactPairs_t* keys,
                                                   const float3* patchNormals,
                                                   TriTriLiteAccum* accumulators,
                                                   contactPairs_t startOffsetPrimitive,
                                                   contactPairs_t startOffsetPatch,
                                                   contactPairs_t countPatch,
                                                   contactPairs_t countPrimitive,
                                                   cudaStream_t& this_stream) {
    constexpr int RECOMPUTE_THREADS = 64;
    size_t blocks_needed = (countPrimitive + RECOMPUTE_THREADS - 1) / RECOMPUTE_THREADS;
    if (blocks_needed > 0) {
        recomputeTriTriAreaAndPrepareLiteAccumulators_impl<<<blocks_needed, RECOMPUTE_THREADS, 0, this_stream>>>(
            simParams, granData, keys, patchNormals, accumulators, startOffsetPrimitive, startOffsetPatch, countPatch,
            countPrimitive);
    }
}

__global__ void finalizeTriTriLitePatchResults_impl(const TriTriLiteAccum* patchAccumulators,
                                                    const float3* patchNormals,
                                                    const float3* zeroAreaNormals,
                                                    const double* zeroAreaPenetrations,
                                                    const double3* zeroAreaContactPoints,
                                                    double* finalAreas,
                                                    float3* finalNormals,
                                                    double* finalPenetrations,
                                                    double3* finalContactPoints,
                                                    contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    const TriTriLiteAccum acc = patchAccumulators[idx];
    if (acc.sumProjArea > 0.0) {
        float3 n = patchNormals[idx];
        const float nlen2 = length2(n);
        if (nlen2 > 0.f) {
            n *= rsqrtf(nlen2);
        } else {
            n = make_float3(0.f, 0.f, 0.f);
        }
        finalAreas[idx] = acc.sumProjArea;
        finalNormals[idx] = n;
        finalPenetrations[idx] = acc.maxPositivePen;
        const double invA = 1.0 / acc.sumProjArea;
        finalContactPoints[idx] = make_double3(acc.sumAreaWeightedCPx * invA,
                                               acc.sumAreaWeightedCPy * invA,
                                               acc.sumAreaWeightedCPz * invA);
    } else {
        finalAreas[idx] = 0.0;
        finalNormals[idx] = zeroAreaNormals[idx];
        finalPenetrations[idx] = zeroAreaPenetrations[idx];
        finalContactPoints[idx] = zeroAreaContactPoints[idx];
    }
}

void finalizeTriTriLitePatchResults(const TriTriLiteAccum* patchAccumulators,
                                    const float3* patchNormals,
                                    const float3* zeroAreaNormals,
                                    const double* zeroAreaPenetrations,
                                    const double3* zeroAreaContactPoints,
                                    double* finalAreas,
                                    float3* finalNormals,
                                    double* finalPenetrations,
                                    double3* finalContactPoints,
                                    contactPairs_t count,
                                    cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        finalizeTriTriLitePatchResults_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            patchAccumulators, patchNormals, zeroAreaNormals, zeroAreaPenetrations, zeroAreaContactPoints,
            finalAreas, finalNormals, finalPenetrations, finalContactPoints, count);
    }
}

__device__ __forceinline__ void makePatchTangentBasis(const float3& n, float3& t1, float3& t2) {
    const float3 ref = (fabsf(n.z) < 0.9f) ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(0.0f, 1.0f, 0.0f);
    t1 = cross(ref, n);
    const float t1_len2 = length2(t1);
    if (t1_len2 > 0.0f) {
        t1 *= rsqrtf(t1_len2);
    } else {
        t1 = make_float3(1.0f, 0.0f, 0.0f);
    }
    t2 = cross(n, t1);
    const float t2_len2 = length2(t2);
    if (t2_len2 > 0.0f) {
        t2 *= rsqrtf(t2_len2);
    } else {
        t2 = make_float3(0.0f, 1.0f, 0.0f);
    }
}

// Per-primitive accumulator generation.
//
// This replaces the former pipeline:
//   computeWeightedUsefulPenetration -> ReduceByKey(sum projArea)
//   ReduceByKey(max projPen)
//   computeWeightedContactPoints -> ReduceByKey(sum weightedCP) -> ReduceByKey(sum weight)
//
// It produces the same patch-level quantities, but materializes only one array
// (PatchContactAccum) and performs a single ReduceByKey.
__global__ void computePatchContactAccumulators_impl(DEMDataDT* granData,
                                                     const contactPairs_t* keys,
                                                     PatchContactAccum* accumulators,
                                                     contactPairs_t startOffsetPrimitive,
                                                     contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const contactPairs_t myContactID = startOffsetPrimitive + idx;

        double originalPenetration = granData->contactPenetration[myContactID - granData->contactScalarOffset];
        originalPenetration = (originalPenetration > 0.0) ? originalPenetration : 0.0;

        const double area = granData->contactArea[myContactID - granData->contactScalarOffset];

        const double projectedArea = area;
        const bool contributes = projectedArea > 0.0;
        // For patch normal/depth voting, only primitives with positive projected area are valid contributors.
        // Non-contact primitives can still carry positive geometric penetration placeholders (e.g., one-sided
        // sph-tri face rejects), and allowing them into max-penetration voting can inject invalid normals.
        const double projectedPenetration = contributes ? originalPenetration : 0.0;

        const double3 contactPointRaw = to_double3(primitiveMeshScratchContactPoint(granData, myContactID));
        const double3 contactPoint = contributes ? contactPointRaw : make_double3(0.0, 0.0, 0.0);
        const double3 areaWeightedCP =
            make_double3(contactPoint.x * projectedArea,
                         contactPoint.y * projectedArea,
                         contactPoint.z * projectedArea);
        const float3 primitiveNormal = contributes ? primitiveMeshScratchNormal(granData, myContactID) : make_float3(0.f, 0.f, 0.f);

        PatchContactAccum acc{};
        acc.sumProjArea = projectedArea;
        acc.maxProjPen = projectedPenetration;
        acc.sumAreaWeightedCP = areaWeightedCP;
        acc.cpAtMaxPen = contactPoint;
        acc.normalAtMaxPen = primitiveNormal;
        acc.minSpanU = DEME_HUGE_FLOAT;
        acc.maxSpanU = -DEME_HUGE_FLOAT;
        acc.minSpanV = DEME_HUGE_FLOAT;
        acc.maxSpanV = -DEME_HUGE_FLOAT;
        acc.triTriCount = 0u;
        accumulators[idx] = acc;
    }
}

void computePatchContactAccumulators(DEMDataDT* granData,
                                     const contactPairs_t* keys,
                                     PatchContactAccum* accumulators,
                                     contactPairs_t startOffsetPrimitive,
                                     contactPairs_t count,
                                     cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computePatchContactAccumulators_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, keys, accumulators, startOffsetPrimitive, count);
    }
}

__global__ void extractPrimitivePatchAccumFields_impl(const PatchContactAccum* primitiveAccumulators,
                                                      double* projectedAreas,
                                                      double* projectedPenetrations,
                                                      double3* areaWeightedContactPoints,
                                                      contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const PatchContactAccum acc = primitiveAccumulators[idx];
        projectedAreas[idx] = acc.sumProjArea;
        projectedPenetrations[idx] = acc.maxProjPen;
        areaWeightedContactPoints[idx] = acc.sumAreaWeightedCP;
    }
}

void extractPrimitivePatchAccumFields(const PatchContactAccum* primitiveAccumulators,
                                      double* projectedAreas,
                                      double* projectedPenetrations,
                                      double3* areaWeightedContactPoints,
                                      contactPairs_t count,
                                      cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        extractPrimitivePatchAccumFields_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            primitiveAccumulators, projectedAreas, projectedPenetrations, areaWeightedContactPoints, count);
    }
}

// Finalization from patch accumulators (no intermediate per-patch arrays).
__global__ void finalizePatchResultsFromAccumulators_impl(const PatchContactAccum* patchAccumulators,
                                                          const float3* votedNormals,
                                                          const float3* zeroAreaNormals,
                                                          const double* zeroAreaPenetrations,
                                                          const double3* zeroAreaContactPoints,
                                                          double* finalAreas,
                                                          float3* finalNormals,
                                                          double* finalPenetrations,
                                                          double3* finalContactPoints,
                                                          contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const PatchContactAccum acc = patchAccumulators[idx];
        const double patchArea = acc.sumProjArea;

        if (patchArea > 0.0) {
            finalAreas[idx] = patchArea;
            float3 n = acc.normalAtMaxPen;
            const float nlen2 = length2(n);
            if (nlen2 > 0.f) {
                n *= rsqrtf(nlen2);
                finalNormals[idx] = n;
            } else {
                finalNormals[idx] = zeroAreaNormals[idx];
            }
            finalPenetrations[idx] = acc.maxProjPen;
            // Use area-averaged contact point for patch contacts to avoid witness-jump
            // torque spikes when the max-penetration primitive switches between facets.
            const double invA = 1.0 / patchArea;
            finalContactPoints[idx] = make_double3(acc.sumAreaWeightedCP.x * invA,
                                                   acc.sumAreaWeightedCP.y * invA,
                                                   acc.sumAreaWeightedCP.z * invA);
        } else {
            finalAreas[idx] = 0.0;
            finalNormals[idx] = zeroAreaNormals[idx];
            finalPenetrations[idx] = zeroAreaPenetrations[idx];
            finalContactPoints[idx] = zeroAreaContactPoints[idx];
        }
    }
}

void finalizePatchResultsFromAccumulators(const PatchContactAccum* patchAccumulators,
                                          const float3* votedNormals,
                                          const float3* zeroAreaNormals,
                                          const double* zeroAreaPenetrations,
                                          const double3* zeroAreaContactPoints,
                                          double* finalAreas,
                                          float3* finalNormals,
                                          double* finalPenetrations,
                                          double3* finalContactPoints,
                                          contactPairs_t count,
                                          cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        finalizePatchResultsFromAccumulators_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            patchAccumulators, votedNormals, zeroAreaNormals, zeroAreaPenetrations, zeroAreaContactPoints, finalAreas,
            finalNormals, finalPenetrations, finalContactPoints, count);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Special case handling: zero-area patches (no positive-penetration primitives)
////////////////////////////////////////////////////////////////////////////////

// Kernel to extract primitive penetrations for max-reduce operation
// For zero-area case handling, we need the max (biggest/least-negative) penetration per patch
__global__ void extractPrimitivePenetrations_impl(DEMDataDT* granData,
                                                  double* penetrations,
                                                  contactPairs_t startOffset,
                                                  contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        contactPairs_t myContactID = startOffset + idx;

        // Extract primitive penetration from dedicated double storage.
        penetrations[idx] = granData->contactPenetration[myContactID - granData->contactScalarOffset];
    }
}

void extractPrimitivePenetrations(DEMDataDT* granData,
                                  double* penetrations,
                                  contactPairs_t startOffset,
                                  contactPairs_t count,
                                  cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        extractPrimitivePenetrations_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, penetrations, startOffset, count);
    }
}


__global__ void computeDirectContactPointLimit_impl(DEMDataDT* granData,
                                                    contactPairs_t* directPointLimit,
                                                    contactPairs_t startOffset,
                                                    contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const contactPairs_t myContactID = startOffset + idx;
        const contactPairs_t patchContactID = granData->geomToPatchMap[myContactID];
        // directPointLimit is one-past-the-largest patch slot that a direct primitive force kernel may write.
        atomicMax(directPointLimit, patchContactID + 1u);
    }
}

void computeDirectContactPointLimit(DEMDataDT* granData,
                                    contactPairs_t* directPointLimit,
                                    contactPairs_t sphSphStart,
                                    contactPairs_t sphSphCount,
                                    contactPairs_t sphAnalStart,
                                    contactPairs_t sphAnalCount,
                                    cudaStream_t& this_stream) {
    DEME_GPU_CALL(cudaMemsetAsync(directPointLimit, 0, sizeof(contactPairs_t), this_stream));
    const auto launch_range = [&](contactPairs_t startOffset, contactPairs_t count) {
        size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
        if (blocks_needed > 0) {
            computeDirectContactPointLimit_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
                granData, directPointLimit, startOffset, count);
        }
    };
    launch_range(sphSphStart, sphSphCount);
    launch_range(sphAnalStart, sphAnalCount);
}

// Kernel to handle zero-area patches by finding the primitive with max penetration
// and using its penetration, normal, and contact point for the patch result.
// For each primitive, check if it has the max penetration for its patch.
// Note: Race condition when multiple primitives have the same max penetration is acceptable
// since any of them produces a valid result.
__global__ void findMaxPenetrationPrimitiveForZeroAreaPatches_impl(DEMDataDT* granData,
                                                                   double* maxPenetrations,
                                                                   float3* zeroAreaNormals,
                                                                   double* zeroAreaPenetrations,
                                                                   double3* zeroAreaContactPoints,
                                                                   contactPairs_t* keys,
                                                                   contactPairs_t startOffsetPrimitive,
                                                                   contactPairs_t startOffsetPatch,
                                                                   contactPairs_t countPatch,
                                                                   contactPairs_t countPrimitive) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < countPrimitive) {
        contactPairs_t myContactID = startOffsetPrimitive + idx;
        contactPairs_t patchIdx = keys[idx];
        if (patchIdx < startOffsetPatch || patchIdx >= startOffsetPatch + countPatch) {
            return;
        }
        contactPairs_t localPatchIdx = patchIdx - startOffsetPatch;

        // In fact, we just need to proceed if area is zero. But these no-contact cases are so
        // common, that we don't do an early termination here.

        // Get this primitive's penetration from dedicated double storage.
        double myPenetration = granData->contactPenetration[myContactID - granData->contactScalarOffset];

        // Check if this primitive has the max penetration for its patch
        // Use a relative tolerance for floating-point comparison
        double maxPen = maxPenetrations[localPatchIdx];
        double absTol = 1e-15;  // Absolute tolerance for very small values
        double relTol = 1e-12;  // Relative tolerance for larger values
        double tolerance = fmax(absTol, fabs(maxPen) * relTol);
        if (fabs(myPenetration - maxPen) <= tolerance) {
            // This primitive has the max penetration - use its normal, penetration, and contact point
            // Note: if multiple primitives have the same max, any one of them is fine
            // The race condition is acceptable since all competing values are valid
            float3 myNormal = primitiveMeshScratchNormal(granData, myContactID);
            zeroAreaNormals[localPatchIdx] = myNormal;
            zeroAreaPenetrations[localPatchIdx] = myPenetration < 0.0 ? myPenetration : -DEME_HUGE_FLOAT;
            // This zeroAreaPenetrations should store a negative number, as when it is needed, it's usually the
            // separation case (all zero-area primitives). But for the no-SAT case, which can resemble cross-particle
            // erroneous detection, we could have a positive max here (search for CubOpMaxNegative to understand how
            // this max is derived). In that case, we give it a very negative number, so in the patch-based force
            // calculation, this one is considered a non-contact.

            // Also store the contact point from this max-penetration primitive
            double3 myContactPoint = to_double3(primitiveMeshScratchContactPoint(granData, myContactID));
            zeroAreaContactPoints[localPatchIdx] = myContactPoint;
        }
    }
}

void findMaxPenetrationPrimitiveForZeroAreaPatches(DEMDataDT* granData,
                                                   double* maxPenetrations,
                                                   float3* zeroAreaNormals,
                                                   double* zeroAreaPenetrations,
                                                   double3* zeroAreaContactPoints,
                                                   contactPairs_t* keys,
                                                   contactPairs_t startOffsetPrimitive,
                                                   contactPairs_t startOffsetPatch,
                                                   contactPairs_t countPatch,
                                                   contactPairs_t countPrimitive,
                                                   cudaStream_t& this_stream) {
    size_t blocks_needed = (countPrimitive + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        findMaxPenetrationPrimitiveForZeroAreaPatches_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0,
                                                             this_stream>>>(
            granData, maxPenetrations, zeroAreaNormals, zeroAreaPenetrations, zeroAreaContactPoints, keys,
            startOffsetPrimitive, startOffsetPatch, countPatch, countPrimitive);
    }
}

// Kernel to finalize patch results by combining normal voting results with zero-area case handling
__global__ void finalizePatchResults_impl(double* totalProjectedAreas,
                                          float3* votedNormals,
                                          double* votedPenetrations,
                                          double3* votedContactPoints,
                                          float3* zeroAreaNormals,
                                          double* zeroAreaPenetrations,
                                          double3* zeroAreaContactPoints,
                                          double* finalAreas,
                                          float3* finalNormals,
                                          double* finalPenetrations,
                                          double3* finalContactPoints,
                                          contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double projectedArea = totalProjectedAreas[idx];

        // Use voted results only if projectedArea > 0
        if (projectedArea > 0.0) {
            // Normal case: use voted results
            finalAreas[idx] = projectedArea;
            finalNormals[idx] = votedNormals[idx];
            finalPenetrations[idx] = votedPenetrations[idx];
            finalContactPoints[idx] = votedContactPoints[idx];
        } else {
            // Zero-area case: use max-penetration primitive's results (Step 8 fallback)
            // Set finalArea to 0 for these cases
            finalAreas[idx] = 0.0;
            finalNormals[idx] = zeroAreaNormals[idx];
            finalPenetrations[idx] = zeroAreaPenetrations[idx];
            finalContactPoints[idx] = zeroAreaContactPoints[idx];
        }
    }
}

void finalizePatchResults(double* totalProjectedAreas,
                          float3* votedNormals,
                          double* votedPenetrations,
                          double3* votedContactPoints,
                          float3* zeroAreaNormals,
                          double* zeroAreaPenetrations,
                          double3* zeroAreaContactPoints,
                          double* finalAreas,
                          float3* finalNormals,
                          double* finalPenetrations,
                          double3* finalContactPoints,
                          contactPairs_t count,
                          cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        finalizePatchResults_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            totalProjectedAreas, votedNormals, votedPenetrations, votedContactPoints, zeroAreaNormals,
            zeroAreaPenetrations, zeroAreaContactPoints, finalAreas, finalNormals, finalPenetrations,
            finalContactPoints, count);
    }
}

// Kernel to compute final contact points per patch by dividing by total weight
// If total weight is 0, contact point is set to (0,0,0)
__global__ void computeFinalContactPointsPerPatch_impl(double3* totalWeightedContactPoints,
                                                       double* totalWeights,
                                                       double3* finalContactPoints,
                                                       contactPairs_t count) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        double totalWeight = totalWeights[idx];
        if (totalWeight > 0.0) {
            // Normalize by dividing by total weight
            double invTotalWeight = (1.0 / totalWeight);
            finalContactPoints[idx] = totalWeightedContactPoints[idx] * invTotalWeight;
        } else {
            // No valid contact point, set to (0,0,0)
            finalContactPoints[idx] = make_double3(0, 0, 0);
        }
    }
}

void computeFinalContactPointsPerPatch(double3* totalWeightedContactPoints,
                                       double* totalWeights,
                                       double3* finalContactPoints,
                                       contactPairs_t count,
                                       cudaStream_t& this_stream) {
    size_t blocks_needed = (count + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computeFinalContactPointsPerPatch_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            totalWeightedContactPoints, totalWeights, finalContactPoints, count);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Per-triangle P / V / P*V accumulation from patch contacts
////////////////////////////////////////////////////////////////////////////////

__global__ void computePatchPVScalars_impl(const DEMSimParams* simParams,
                                           DEMDataDT* granData,
                                           const float3* finalNormals,
                                           const double3* finalContactPoints,
                                           contactPairs_t startOffsetPatch,
                                           contactPairs_t countPatch,
                                           float* patchNormalForce,
                                           float* patchSlipSpeed) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= countPatch || !patchNormalForce || !patchSlipSpeed || !finalContactPoints) {
        return;
    }

    patchNormalForce[idx] = 0.f;
    patchSlipSpeed[idx] = 0.f;

    const contactPairs_t patchContactID = startOffsetPatch + idx;
    const contact_t patchType = granData->contactTypePatch[patchContactID];
    if (patchType == NOT_A_CONTACT) {
        return;
    }

    float3 normal = finalNormals[idx];
    const float n2 = dot(normal, normal);
    if (!(n2 > DEME_TINY_FLOAT)) {
        return;
    }
    normal *= rsqrtf(n2);

    const float3 patchForce = granData->contactForces[patchContactID];
    const float normalForce = fabsf(dot(patchForce, normal));
    if (!(normalForce > DEME_TINY_FLOAT)) {
        return;
    }

    const bodyID_t geoA_raw = granData->idPatchA[patchContactID];
    const bodyID_t geoB_raw = granData->idPatchB[patchContactID];
    bool ghostA = false, ghostA_neg = false;
    bool ghostB = false, ghostB_neg = false;
    const bodyID_t geoA = cylPeriodicDecodeID(geoA_raw, ghostA, ghostA_neg);
    const bodyID_t geoB = cylPeriodicDecodeID(geoB_raw, ghostB, ghostB_neg);
    const bodyID_t ownerA = DEME_GET_PATCH_OWNER_ID(geoA, decodeTypeA(patchType));
    const bodyID_t ownerB = DEME_GET_PATCH_OWNER_ID(geoB, decodeTypeB(patchType));

    float3 velCPA = make_float3(0.f, 0.f, 0.f);
    float3 velCPB = make_float3(0.f, 0.f, 0.f);
    const double3 cp_d = finalContactPoints[idx];
    if (!isfinite(cp_d.x) || !isfinite(cp_d.y) || !isfinite(cp_d.z)) {
        return;
    }
    const float3 cp_global = make_float3((float)cp_d.x, (float)cp_d.y, (float)cp_d.z);

    if (ownerA != NULL_BODYID && ownerA < simParams->nOwnerBodies) {
        float3 linVelA = make_float3(granData->vX[ownerA], granData->vY[ownerA], granData->vZ[ownerA]);
        float3 angVelA_local =
            make_float3(granData->omgBarX[ownerA], granData->omgBarY[ownerA], granData->omgBarZ[ownerA]);
        const float4 oriA =
            make_float4(granData->oriQx[ownerA], granData->oriQy[ownerA], granData->oriQz[ownerA], granData->oriQw[ownerA]);
        if (isfinite(linVelA.x) && isfinite(linVelA.y) && isfinite(linVelA.z) && isfinite(angVelA_local.x) &&
            isfinite(angVelA_local.y) && isfinite(angVelA_local.z)) {
            float3 angVelA_global = angVelA_local;
            applyOriQToVector3<float, oriQ_t>(angVelA_global.x, angVelA_global.y, angVelA_global.z, oriA.w, oriA.x,
                                              oriA.y, oriA.z);

            double3 comA;
            voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                comA.x, comA.y, comA.z, granData->voxelID[ownerA], granData->locX[ownerA], granData->locY[ownerA],
                granData->locZ[ownerA], simParams->nvXp2, simParams->nvYp2, simParams->voxelSize, simParams->l);
            comA.x += simParams->LBFX;
            comA.y += simParams->LBFY;
            comA.z += simParams->LBFZ;

            float3 rA_global = make_float3(cp_global.x - (float)comA.x, cp_global.y - (float)comA.y,
                                           cp_global.z - (float)comA.z);
            if (granData->ownerBoundRadius) {
                const float bound_r = fmaxf(granData->ownerBoundRadius[ownerA], 0.f);
                if (isfinite(bound_r) && bound_r > DEME_TINY_FLOAT) {
                    const float geom_tol = fmaxf(simParams->dyn.beta + simParams->maxFamilyExtraMargin, 0.f) + 1e-4f;
                    const float max_lever = fmaxf(bound_r + geom_tol, 1e-3f);
                    const float r2 = dot(rA_global, rA_global);
                    const float max2 = max_lever * max_lever;
                    if (isfinite(r2) && r2 > max2) {
                        rA_global *= max_lever * rsqrtf(r2);
                    }
                }
            }
            velCPA = linVelA + cross(angVelA_global, rA_global);
        }
    }

    if (ownerB != NULL_BODYID && ownerB < simParams->nOwnerBodies) {
        float3 linVelB = make_float3(granData->vX[ownerB], granData->vY[ownerB], granData->vZ[ownerB]);
        float3 angVelB_local =
            make_float3(granData->omgBarX[ownerB], granData->omgBarY[ownerB], granData->omgBarZ[ownerB]);
        const float4 oriB =
            make_float4(granData->oriQx[ownerB], granData->oriQy[ownerB], granData->oriQz[ownerB], granData->oriQw[ownerB]);
        if (isfinite(linVelB.x) && isfinite(linVelB.y) && isfinite(linVelB.z) && isfinite(angVelB_local.x) &&
            isfinite(angVelB_local.y) && isfinite(angVelB_local.z)) {
            float3 angVelB_global = angVelB_local;
            applyOriQToVector3<float, oriQ_t>(angVelB_global.x, angVelB_global.y, angVelB_global.z, oriB.w, oriB.x,
                                              oriB.y, oriB.z);

            double3 comB;
            voxelIDToPosition<double, deme::voxelID_t, deme::subVoxelPos_t>(
                comB.x, comB.y, comB.z, granData->voxelID[ownerB], granData->locX[ownerB], granData->locY[ownerB],
                granData->locZ[ownerB], simParams->nvXp2, simParams->nvYp2, simParams->voxelSize, simParams->l);
            comB.x += simParams->LBFX;
            comB.y += simParams->LBFY;
            comB.z += simParams->LBFZ;

            float3 rB_global = make_float3(cp_global.x - (float)comB.x, cp_global.y - (float)comB.y,
                                           cp_global.z - (float)comB.z);
            if (granData->ownerBoundRadius) {
                const float bound_r = fmaxf(granData->ownerBoundRadius[ownerB], 0.f);
                if (isfinite(bound_r) && bound_r > DEME_TINY_FLOAT) {
                    const float geom_tol = fmaxf(simParams->dyn.beta + simParams->maxFamilyExtraMargin, 0.f) + 1e-4f;
                    const float max_lever = fmaxf(bound_r + geom_tol, 1e-3f);
                    const float r2 = dot(rB_global, rB_global);
                    const float max2 = max_lever * max_lever;
                    if (isfinite(r2) && r2 > max2) {
                        rB_global *= max_lever * rsqrtf(r2);
                    }
                }
            }
            velCPB = linVelB + cross(angVelB_global, rB_global);
        }
    }

    const float3 relVel = velCPA - velCPB;
    const float vRelN = dot(relVel, normal);
    const float3 vRelT = relVel - vRelN * normal;
    const float slipSpeed = length(vRelT);
    if (!isfinite(slipSpeed) || !(slipSpeed >= 0.f)) {
        return;
    }
    patchNormalForce[idx] = normalForce;
    patchSlipSpeed[idx] = slipSpeed;
}

void computePatchPVScalars(DEMSimParams* simParams,
                           DEMDataDT* granData,
                           const float3* finalNormals,
                           const double3* finalContactPoints,
                           contactPairs_t startOffsetPatch,
                           contactPairs_t countPatch,
                           float* patchNormalForce,
                           float* patchSlipSpeed,
                           cudaStream_t& this_stream) {
    size_t blocks_needed = (countPatch + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        computePatchPVScalars_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            simParams, granData, finalNormals, finalContactPoints, startOffsetPatch, countPatch, patchNormalForce,
            patchSlipSpeed);
    }
}

__device__ __forceinline__ float primitiveTriangleReferenceArea(const DEMDataDT* granData, bodyID_t triID) {
    const float3 a = granData->relPosNode1[triID];
    const float3 b = granData->relPosNode2[triID];
    const float3 c = granData->relPosNode3[triID];
    const float3 e1 = b - a;
    const float3 e2 = c - a;
    const float area = 0.5f * length(cross(e1, e2));
    return (isfinite(area) && area > DEME_TINY_FLOAT) ? area : 0.f;
}

__global__ void accumulateTrianglePVFromPatchContacts_impl(const DEMSimParams* simParams,
                                                           DEMDataDT* granData,
                                                           const contactPairs_t* keys,
                                                           const PatchContactAccum* primitiveAccumulators,
                                                           const PatchContactAccum* patchAccumulators,
                                                           const double* finalPatchAreas,
                                                           const float* patchNormalForce,
                                                           const float* patchSlipSpeed,
                                                           contactPairs_t startOffsetPrimitive,
                                                           contactPairs_t startOffsetPatch,
                                                           contactPairs_t countPatch,
                                                           contactPairs_t countPrimitive,
                                                           const int* triGlobalToLocal,
                                                           float* triAccumP,
                                                           float* triAccumPV,
                                                           float* triAccumV) {
    contactPairs_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= countPrimitive || !triGlobalToLocal || !triAccumP || !triAccumPV || !triAccumV ||
        !finalPatchAreas) {
        return;
    }

    const contactPairs_t primContactID = startOffsetPrimitive + idx;
    const contactPairs_t patchContactID = keys[idx];
    if (patchContactID < startOffsetPatch || patchContactID >= startOffsetPatch + countPatch) {
        return;
    }
    const contactPairs_t localPatchIdx = patchContactID - startOffsetPatch;

    const double patchWeight = finalPatchAreas[localPatchIdx];
    const double rawPatchWeight =
        patchAccumulators ? patchAccumulators[localPatchIdx].sumProjArea : patchWeight;
    const double primitiveWeight = primitiveAccumulators[idx].sumProjArea;
    if (patchWeight <= 0.0 || primitiveWeight <= 0.0) {
        return;
    }

    const double shareDenom = (rawPatchWeight > 0.0) ? rawPatchWeight : patchWeight;
    float share = static_cast<float>(primitiveWeight / shareDenom);
    if (!(share > 0.f)) {
        return;
    }
    share = fminf(share, 1.f);

    const float normalForce = patchNormalForce[localPatchIdx];
    if (!(normalForce > DEME_TINY_FLOAT)) {
        return;
    }
    const float pContribution = normalForce * share;
    const float slipSpeed = patchSlipSpeed[localPatchIdx];

    const contact_t primType = granData->contactTypePrimitive[primContactID];
    if (primType == NOT_A_CONTACT) {
        return;
    }

    const geoType_t typeA = decodeTypeA(primType);
    const geoType_t typeB = decodeTypeB(primType);

    // Track any contact contribution on triangle sides (sphere-triangle and triangle-triangle).
    if (typeA == GEO_T_TRIANGLE) {
        bool triGhost = false, triGhostNeg = false;
        const bodyID_t triA = cylPeriodicDecodeID(granData->idPrimitiveA[primContactID], triGhost, triGhostNeg);
        if (triA < simParams->nTriGM) {
            const int localIdx = triGlobalToLocal[triA];
            const float triArea = primitiveTriangleReferenceArea(granData, triA);
            if (localIdx >= 0 && triArea > 0.f) {
                const float pressure = pContribution / triArea;
                atomicAdd(triAccumP + localIdx, pressure);
                atomicAdd(triAccumPV + localIdx, pressure * slipSpeed);
                atomicAdd(triAccumV + localIdx, slipSpeed);
            }
        }
    }
    if (typeB == GEO_T_TRIANGLE) {
        bool triGhost = false, triGhostNeg = false;
        const bodyID_t triB = cylPeriodicDecodeID(granData->idPrimitiveB[primContactID], triGhost, triGhostNeg);
        if (triB < simParams->nTriGM) {
            const int localIdx = triGlobalToLocal[triB];
            const float triArea = primitiveTriangleReferenceArea(granData, triB);
            if (localIdx >= 0 && triArea > 0.f) {
                const float pressure = pContribution / triArea;
                atomicAdd(triAccumP + localIdx, pressure);
                atomicAdd(triAccumPV + localIdx, pressure * slipSpeed);
                atomicAdd(triAccumV + localIdx, slipSpeed);
            }
        }
    }
}

void accumulateTrianglePVFromPatchContacts(DEMSimParams* simParams,
                                           DEMDataDT* granData,
                                           const contactPairs_t* keys,
                                           const PatchContactAccum* primitiveAccumulators,
                                           const PatchContactAccum* patchAccumulators,
                                           const double* finalPatchAreas,
                                           const float* patchNormalForce,
                                           const float* patchSlipSpeed,
                                           contactPairs_t startOffsetPrimitive,
                                           contactPairs_t startOffsetPatch,
                                           contactPairs_t countPatch,
                                           contactPairs_t countPrimitive,
                                           const int* triGlobalToLocal,
                                           float* triAccumP,
                                           float* triAccumPV,
                                           float* triAccumV,
                                           cudaStream_t& this_stream) {
    size_t blocks_needed = (countPrimitive + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        accumulateTrianglePVFromPatchContacts_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            simParams, granData, keys, primitiveAccumulators, patchAccumulators, finalPatchAreas, patchNormalForce,
            patchSlipSpeed, startOffsetPrimitive, startOffsetPatch, countPatch, countPrimitive, triGlobalToLocal,
            triAccumP, triAccumPV, triAccumV);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Prep force kernels
////////////////////////////////////////////////////////////////////////////////

inline __device__ void cleanUpContactForces(size_t thisContact, DEMSimParams* simParams, DEMDataDT* granData) {
    const float3 zeros = make_float3(0, 0, 0);
    granData->contactForces[thisContact] = zeros;
    granData->contactTorque_convToForce[thisContact] = zeros;
}

inline __device__ void cleanUpAcc(size_t thisClump, DEMSimParams* simParams, DEMDataDT* granData) {
    // If should not clear acc arrays, then just mark it to be clear in the next ts
    if (granData->accSpecified[thisClump]) {
        granData->accSpecified[thisClump] = 0;
    } else {
        granData->aX[thisClump] = 0;
        granData->aY[thisClump] = 0;
        granData->aZ[thisClump] = 0;
    }
    if (granData->angAccSpecified[thisClump]) {
        granData->angAccSpecified[thisClump] = 0;
    } else {
        granData->alphaX[thisClump] = 0;
        granData->alphaY[thisClump] = 0;
        granData->alphaZ[thisClump] = 0;
    }
}

__global__ void prepareAccArrays_impl(DEMSimParams* simParams, DEMDataDT* granData) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < simParams->nOwnerBodies) {
        cleanUpAcc(myID, simParams, granData);
        if (simParams->useCylPeriodicDiagCounters) {
            if (granData->ownerCylSkipCount) {
                granData->ownerCylSkipCount[myID] = 0;
            }
            if (granData->ownerCylSkipPotentialCount) {
                granData->ownerCylSkipPotentialCount[myID] = 0;
            }
        }
    }
}

__global__ void prepareForceArrays_impl(DEMSimParams* simParams, DEMDataDT* granData, size_t nContactPairs) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        cleanUpContactForces(myID, simParams, granData);
    }
}

void prepareForceArrays(DEMSimParams* simParams,
                        DEMDataDT* granData,
                        size_t nPrimitiveContactPairs,
                        cudaStream_t& this_stream) {
    size_t blocks_needed_for_force_prep =
        (nPrimitiveContactPairs + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed_for_force_prep > 0) {
        prepareForceArrays_impl<<<blocks_needed_for_force_prep, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            simParams, granData, nPrimitiveContactPairs);
    }
}

void prepareAccArrays(DEMSimParams* simParams, DEMDataDT* granData, bodyID_t nOwnerBodies, cudaStream_t& this_stream) {
    size_t blocks_needed_for_acc_prep = (nOwnerBodies + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed_for_acc_prep > 0) {
        prepareAccArrays_impl<<<blocks_needed_for_acc_prep, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(simParams,
                                                                                                          granData);
    }
}

__global__ void rearrangeContactWildcards_impl(DEMDataDT* granData,
                                               float* newWildcards,
                                               notStupidBool_t* sentry,
                                               unsigned int nWildcards,
                                               size_t nPrevContactPairs,
                                               size_t nContactPairs) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        contactPairs_t map_from = granData->contactMapping[myID];
        if (map_from == NULL_MAPPING_PARTNER || map_from >= nPrevContactPairs) {
            // If it is a NULL ID then kT says this contact is new. Initialize all wildcard arrays.
            for (size_t i = 0; i < nWildcards; i++) {
                newWildcards[nContactPairs * i + myID] = 0;
            }
        } else {
            // Not a new contact, need to map it from somewhere in the old history array
            for (size_t i = 0; i < nWildcards; i++) {
                newWildcards[nContactPairs * i + myID] = granData->contactWildcards[i][map_from];
            }
            // This sentry trys to make sure that all `alive' contacts got mapped to some place
            sentry[map_from] = 0;
        }
    }
}

void rearrangeContactWildcards(DEMDataDT* granData,
                               float* wildcard,
                               notStupidBool_t* sentry,
                               unsigned int nWildcards,
                               size_t nPrevContactPairs,
                               size_t nContactPairs,
                               cudaStream_t& this_stream) {
    size_t blocks_needed = (nContactPairs + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        rearrangeContactWildcards_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(
            granData, wildcard, sentry, nWildcards, nPrevContactPairs, nContactPairs);
    }
}

__global__ void markAliveContacts_impl(float* wildcard, notStupidBool_t* sentry, size_t nContactPairs) {
    size_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContactPairs) {
        float myEntry = abs(wildcard[myID]);
        // If this is alive then mark it
        if (myEntry > DEME_TINY_FLOAT) {
            sentry[myID] = 1;
        } else {
            sentry[myID] = 0;
        }
    }
}

void markAliveContacts(float* wildcard, notStupidBool_t* sentry, size_t nContactPairs, cudaStream_t& this_stream) {
    size_t blocks_needed = (nContactPairs + DEME_MAX_THREADS_PER_BLOCK - 1) / DEME_MAX_THREADS_PER_BLOCK;
    if (blocks_needed > 0) {
        markAliveContacts_impl<<<blocks_needed, DEME_MAX_THREADS_PER_BLOCK, 0, this_stream>>>(wildcard, sentry,
                                                                                              nContactPairs);
    }
}

}  // namespace deme
