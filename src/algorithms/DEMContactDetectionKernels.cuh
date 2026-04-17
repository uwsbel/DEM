// DEM history mapping related custom kernels

#ifndef DEME_CD_KERNELS_CUH
#define DEME_CD_KERNELS_CUH

#include <DEM/Defines.h>
#include <kernel/DEMHelperKernels.cuh>

#include <stdio.h>
#include <math.h>

namespace deme {

struct CylGhostDebugRecord {
    int type;  // 1 = triangle
    bodyID_t geo_id;
    bodyID_t owner_id;
    unsigned char ghost_neg;
    float3 pos;
    float3 ghost_pos;
    float dist_start;
    float dist_end;
    float ghost_dist;
};

inline __device__ float distSquaredPoint(const float3& p, const float3& c) {
    const float dx = p.x - c.x;
    const float dy = p.y - c.y;
    const float dz = p.z - c.z;
    return dx * dx + dy * dy + dz * dz;
}

inline __device__ void updatePlaneMinMax(const float3& p, const float3& origin, const float3& n, float& min_d, float& max_d) {
    const float3 pg = p - origin;
    const float d = dot(pg, n);
    min_d = DEME_MIN(min_d, d);
    max_d = DEME_MAX(max_d, d);
}

// Kernel to add a constant offset to each element of an array
__global__ void addOffsetToArray(contactPairs_t* arr, contactPairs_t offset, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] += offset;
    }
}

// Kernel to fill a contact_t array with a constant value
__global__ void fillContactTypeArray(contact_t* arr, contact_t val, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] = val;
    }
}

__global__ void markBoolIf(notStupidBool_t* bool_arr, notStupidBool_t* value_arr, notStupidBool_t val, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        notStupidBool_t my_val = value_arr[myID];
        if (my_val == val) {
            bool_arr[myID] = 1;
        } else {
            bool_arr[myID] = 0;
        }
    }
}

__global__ void collectCylGhostDebugTriangles(DEMSimParams* simParams,
                                              DEMDataKT* granData,
                                              const float3* vA1,
                                              const float3* vB1,
                                              const float3* vC1,
                                              const float3* vA2,
                                              const float3* vB2,
                                              const float3* vC2,
                                              CylGhostDebugRecord* records,
                                              unsigned long long* count,
                                              unsigned long long max_records) {
    const bodyID_t triID = blockIdx.x * blockDim.x + threadIdx.x;
    if (triID >= simParams->nTriGM) {
        return;
    }
    const float3 A1 = vA1[triID];
    const float3 B1 = vB1[triID];
    const float3 C1 = vC1[triID];
    const float3 A2 = vA2[triID];
    const float3 B2 = vB2[triID];
    const float3 C2 = vC2[triID];

    const float3 centroid = make_float3((A1.x + B1.x + C1.x) / 3.f,
                                        (A1.y + B1.y + C1.y) / 3.f,
                                        (A1.z + B1.z + C1.z) / 3.f);
    float r2 = distSquaredPoint(A1, centroid);
    r2 = DEME_MAX(r2, distSquaredPoint(B1, centroid));
    r2 = DEME_MAX(r2, distSquaredPoint(C1, centroid));
    r2 = DEME_MAX(r2, distSquaredPoint(A2, centroid));
    r2 = DEME_MAX(r2, distSquaredPoint(B2, centroid));
    r2 = DEME_MAX(r2, distSquaredPoint(C2, centroid));
    const float myTriRadius = sqrtf(r2);

    const float max_other = (simParams->maxSphereRadius > simParams->maxTriRadius)
                                ? simParams->maxSphereRadius
                                : simParams->maxTriRadius;
    const float other_margin = simParams->dyn.beta + simParams->maxFamilyExtraMargin;
    const float ghost_dist = myTriRadius + max_other + other_margin;

    const float3 origin = simParams->cylPeriodicOrigin;
    const float3 n_start = simParams->cylPeriodicStartNormal;
    const float3 n_end = simParams->cylPeriodicEndNormal;
    const float3 pos_global = centroid - origin;
    const float dist_start = dot(pos_global, n_start);
    const float dist_end = dot(pos_global, n_end);

    float min_d = DEME_HUGE_FLOAT;
    float max_d = -DEME_HUGE_FLOAT;
    updatePlaneMinMax(A1, origin, n_start, min_d, max_d);
    updatePlaneMinMax(B1, origin, n_start, min_d, max_d);
    updatePlaneMinMax(C1, origin, n_start, min_d, max_d);
    updatePlaneMinMax(A2, origin, n_start, min_d, max_d);
    updatePlaneMinMax(B2, origin, n_start, min_d, max_d);
    updatePlaneMinMax(C2, origin, n_start, min_d, max_d);

    if (min_d <= ghost_dist && max_d >= -ghost_dist) {
        const float3 ghost_pos =
            cylPeriodicRotate(centroid, origin, simParams->cylPeriodicAxisVec, simParams->cylPeriodicU,
                              simParams->cylPeriodicV, simParams->cylPeriodicCosSpan, simParams->cylPeriodicSinSpan);
        const unsigned long long idx = atomicAdd(count, 1ULL);
        if (idx < max_records) {
            CylGhostDebugRecord rec;
            rec.type = 1;
            rec.geo_id = triID;
            rec.owner_id = granData->ownerTriMesh[triID];
            rec.ghost_neg = 0;
            rec.pos = centroid;
            rec.ghost_pos = ghost_pos;
            rec.dist_start = dist_start;
            rec.dist_end = dist_end;
            rec.ghost_dist = ghost_dist;
            records[idx] = rec;
        }
    }

    min_d = DEME_HUGE_FLOAT;
    max_d = -DEME_HUGE_FLOAT;
    updatePlaneMinMax(A1, origin, n_end, min_d, max_d);
    updatePlaneMinMax(B1, origin, n_end, min_d, max_d);
    updatePlaneMinMax(C1, origin, n_end, min_d, max_d);
    updatePlaneMinMax(A2, origin, n_end, min_d, max_d);
    updatePlaneMinMax(B2, origin, n_end, min_d, max_d);
    updatePlaneMinMax(C2, origin, n_end, min_d, max_d);

    if (min_d <= ghost_dist && max_d >= -ghost_dist) {
        const float3 ghost_pos =
            cylPeriodicRotate(centroid, origin, simParams->cylPeriodicAxisVec, simParams->cylPeriodicU,
                              simParams->cylPeriodicV, simParams->cylPeriodicCosSpan, -simParams->cylPeriodicSinSpan);
        const unsigned long long idx = atomicAdd(count, 1ULL);
        if (idx < max_records) {
            CylGhostDebugRecord rec;
            rec.type = 1;
            rec.geo_id = triID;
            rec.owner_id = granData->ownerTriMesh[triID];
            rec.ghost_neg = 1;
            rec.pos = centroid;
            rec.ghost_pos = ghost_pos;
            rec.dist_start = dist_start;
            rec.dist_end = dist_end;
            rec.ghost_dist = ghost_dist;
            records[idx] = rec;
        }
    }
}

__global__ void setArr(notStupidBool_t* arr, size_t n, notStupidBool_t val) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] = val;
    }
}

__global__ void markDuplicateContacts(primitivesPrimTouches_t* idA_runlength,
                                      contactPairs_t* idA_scanned_runlength,
                                      bodyID_t* idB,
                                      contact_t* contactTypePrimitive,
                                      notStupidBool_t* persistency,
                                      notStupidBool_t* retain_list,
                                      size_t n,
                                      bool persistency_affect) {
    bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        primitivesPrimTouches_t cnt_count = idA_runlength[myID];
        // If this idA has non-zero runlength in new: a potential removal needed
        if (cnt_count > 0) {
            // Where should I start looking? Grab the offset.
            contactPairs_t cnt_offset = idA_scanned_runlength[myID];
            for (primitivesPrimTouches_t i = 0; i < cnt_count - 1; i++) {
                for (primitivesPrimTouches_t j = i + 1; j < cnt_count; j++) {
                    // Current contact numbers we are inspecting
                    contactPairs_t contactA = cnt_offset + i;
                    contactPairs_t contactB = cnt_offset + j;
                    bodyID_t contactA_idB = idB[contactA];
                    contact_t contactA_cntType = contactTypePrimitive[contactA];
                    bodyID_t contactB_idB = idB[contactB];
                    contact_t contactB_cntType = contactTypePrimitive[contactB];
                    notStupidBool_t contactA_persistency, contactB_persistency;
                    if (persistency_affect) {
                        contactA_persistency = persistency[contactA];
                        contactB_persistency = persistency[contactB];
                    }

                    if (contactA_idB == contactB_idB && contactA_cntType == contactB_cntType) {
                        // If so, they are the same contact, we have to remove one
                        if (!persistency_affect) {
                            // If persistency does not matter, we remove the one with bigger index, in case there are
                            // 3-way duplicates
                            retain_list[contactB] = 0;
                        } else {
                            if (contactA_persistency == contactB_persistency) {
                                // If both are persistent or both are non-persistent, we remove the one with bigger
                                // index, in case there are 3-way duplicates
                                retain_list[contactB] = 0;
                            } else if (contactB_persistency == CONTACT_NOT_PERSISTENT) {
                                retain_list[contactB] = 0;
                            } else {
                                retain_list[contactA] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void extractPatchInvolvedContactPatchIDPairs(patchIDPair_t* contactPatchPairs,
                                                        contact_t* contactTypePrimitive,
                                                        bodyID_t* idPrimitiveA,
                                                        bodyID_t* idPrimitiveB,
                                                        bodyID_t* triPatchID,
                                                        size_t nContacts) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nContacts) {
        bodyID_t bodyA_raw = idPrimitiveA[myID];
        bodyID_t bodyB_raw = idPrimitiveB[myID];
        bool ghostA = false;
        bool ghostB = false;
        bool ghostA_neg = false;
        bool ghostB_neg = false;
        bodyID_t bodyA = cylPeriodicDecodeID(bodyA_raw, ghostA, ghostA_neg);
        bodyID_t bodyB = cylPeriodicDecodeID(bodyB_raw, ghostB, ghostB_neg);
        switch (contactTypePrimitive[myID]) {
            case SPHERE_TRIANGLE_CONTACT: {
                bodyID_t patchB = triPatchID[bodyB];
                if (ghostB) {
                    patchB = cylPeriodicEncodeGhostID(patchB, ghostB_neg);
                }
                bodyID_t sphA = ghostA ? cylPeriodicEncodeGhostID(bodyA, ghostA_neg) : bodyA;
                // For sphere-triangle contact: triangle has patch ID, sphere does not but its geoID serves the same
                // purpose. We ensure sphere is A.
                contactPatchPairs[myID] = encodeType<patchIDPair_t, bodyID_t>(
                    sphA, patchB);  // Input bodyID_t, return patchIDPair_t
                break;
            }
            case TRIANGLE_ANALYTICAL_CONTACT: {
                bodyID_t patchA = triPatchID[bodyA];
                if (ghostA) {
                    patchA = cylPeriodicEncodeGhostID(patchA, ghostA_neg);
                }
                bodyID_t objB = ghostB ? cylPeriodicEncodeGhostID(bodyB, ghostB_neg) : bodyB;
                // For mesh-analytical contact: mesh has patch ID, analytical object does not but its geoID serves the
                // same purpose. We ensure triangle is A.
                contactPatchPairs[myID] = encodeType<patchIDPair_t, bodyID_t>(patchA, objB);
                break;
            }
            case TRIANGLE_TRIANGLE_CONTACT: {
                bodyID_t patchA = triPatchID[bodyA];
                bodyID_t patchB = triPatchID[bodyB];
                if (ghostA) {
                    patchA = cylPeriodicEncodeGhostID(patchA, ghostA_neg);
                }
                if (ghostB) {
                    patchB = cylPeriodicEncodeGhostID(patchB, ghostB_neg);
                }
                // For triangle-triangle contact: both triangles have patch IDs. Keeps original order.
                contactPatchPairs[myID] = encodeType<patchIDPair_t, bodyID_t>(patchA, patchB);
                break;
            }
            default:
                // In other sph-sph and sph-anal, for now, we just use geoID as patchIDs. Keeps original order.
                contactPatchPairs[myID] = encodeType<patchIDPair_t, bodyID_t>(
                    ghostA ? cylPeriodicEncodeGhostID(bodyA, ghostA_neg) : bodyA,
                    ghostB ? cylPeriodicEncodeGhostID(bodyB, ghostB_neg) : bodyB);
                break;
        }
    }
}

// Decode unique patch pairs into separate idPatchA/idPatchB arrays
__global__ void decodePatchPairsToSeparateArrays(patchIDPair_t* uniquePatchPairs,
                                                 bodyID_t* idPatchA,
                                                 bodyID_t* idPatchB,
                                                 size_t numUnique) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < numUnique) {
        patchIDPair_t patchPair = uniquePatchPairs[myID];
        // decodeTypeA returns the value stored in high bits, and decodeTypeB returns the value stored in low bits. This
        // decoding does not change the order as they were encoded in encodeType.
        idPatchA[myID] = decodeTypeA<patchIDPair_t, bodyID_t>(patchPair);
        idPatchB[myID] = decodeTypeB<patchIDPair_t, bodyID_t>(patchPair);
    }
}

// Build geomToPatchMap by detecting boundaries of unique patch pairs in sorted array
// and using prefix scan on "is first of group" flags.
// For sorted patch pairs, mark 1 at each position where a new unique value starts, then prefix scan.
__global__ void markNewPatchPairGroups(patchIDPair_t* sortedPatchPairs, contactPairs_t* isNewGroup, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        if (myID == 0) {
            isNewGroup[myID] = 0;
        } else {
            // Compare with previous element - if different, it's a new group
            isNewGroup[myID] = (sortedPatchPairs[myID] != sortedPatchPairs[myID - 1]) ? 1 : 0;
        }
    }
}

// Build geomToPatchMap by detecting boundaries of unique (patchPair, contactType) groups in a sorted array.
// The array is expected to be sorted by contact type, then by patch pairs.
__global__ void markNewPatchPairGroupsByType(const patchIDPair_t* sortedPatchPairs,
                                             const contact_t* sortedContactTypes,
                                             contactPairs_t* isNewGroup,
                                             size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        if (myID == 0) {
            isNewGroup[myID] = 0;
        } else {
            const bool new_pair = sortedPatchPairs[myID] != sortedPatchPairs[myID - 1];
            const bool new_type = sortedContactTypes[myID] != sortedContactTypes[myID - 1];
            isNewGroup[myID] = (new_pair || new_type) ? 1 : 0;
        }
    }
}

__global__ void buildCompositeKeys(const patchIDPair_t* patchPairs,
                                   const contact_t* contactTypes,
                                   ulonglong2* keys,
                                   size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        keys[myID].x = static_cast<unsigned long long>(patchPairs[myID]);
        keys[myID].y = static_cast<unsigned long long>(contactTypes[myID]);
    }
}

__global__ void lineNumbers(contactPairs_t* arr, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] = myID;
    }
}

__global__ void markNewCompositeGroups(const ulonglong2* keys, contactPairs_t* isNewGroup, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        if (myID == 0) {
            isNewGroup[myID] = 0;
        } else {
            const ulonglong2 prev = keys[myID - 1];
            const ulonglong2 curr = keys[myID];
            isNewGroup[myID] = (prev.x != curr.x || prev.y != curr.y) ? 1 : 0;
        }
    }
}

__global__ void extractContactTypeFromCompositeKeys(const ulonglong2* keys, contact_t* types, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        types[myID] = static_cast<contact_t>(keys[myID].y);
    }
}

__global__ void decodeCompositeKeysToPatchContacts(const ulonglong2* keys,
                                                   bodyID_t* idPatchA,
                                                   bodyID_t* idPatchB,
                                                   contact_t* contactTypePatch,
                                                   size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        patchIDPair_t patchPair = static_cast<patchIDPair_t>(keys[myID].x);
        idPatchA[myID] = decodeTypeA<patchIDPair_t, bodyID_t>(patchPair);
        idPatchB[myID] = decodeTypeB<patchIDPair_t, bodyID_t>(patchPair);
        contactTypePatch[myID] = static_cast<contact_t>(keys[myID].y);
    }
}

template <typename T>
__global__ void gatherByIndex(const T* in, T* out, const contactPairs_t* idx, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        out[myID] = in[idx[myID]];
    }
}

// Build packed (groupIndex, primitiveID) keys for unique counting.
__global__ void buildGroupPrimitiveKeys(const contactPairs_t* groupIndex,
                                        const bodyID_t* primitiveIDs,
                                        uint64_t* keys,
                                        size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const bodyID_t phys_id = primitiveIDs[myID] & deme::CYL_PERIODIC_SPHERE_ID_MASK;
        keys[myID] = (static_cast<uint64_t>(groupIndex[myID]) << 32) | static_cast<uint64_t>(phys_id);
    }
}

// Extract group indices from packed keys (high 32 bits).
__global__ void extractGroupIndexFromKey(const uint64_t* keys, contactPairs_t* groupIndex, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        groupIndex[myID] = static_cast<contactPairs_t>(keys[myID] >> 32);
    }
}

// Scatter run-length counts into dense per-group counters.
__global__ void scatterGroupCounts(const contactPairs_t* groupIDs,
                                   const contactPairs_t* counts,
                                   contactPairs_t* groupCounts,
                                   size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        groupCounts[groupIDs[myID]] = counts[myID];
    }
}

// Determine winner side for each group.
__global__ void computeGroupWinners(const contact_t* groupTypes,
                                    const bodyID_t* groupPrimA,
                                    const bodyID_t* groupPrimB,
                                    const contactPairs_t* countA,
                                    const contactPairs_t* countB,
                                    const bodyID_t* ownerTriMesh,
                                    const notStupidBool_t* ownerMeshConvex,
                                    const notStupidBool_t* ownerMeshNeverWinner,
                                    notStupidBool_t* winnerIsA,
                                    notStupidBool_t* winnerIsTri,
                                    notStupidBool_t* forceSingleIsland,
                                    size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const contact_t ctype = groupTypes[myID];
        const geoType_t typeA = decodeTypeA<contact_t, geoType_t>(ctype);
        const geoType_t typeB = decodeTypeB<contact_t, geoType_t>(ctype);
        const contactPairs_t nA = countA[myID];
        const contactPairs_t nB = countB[myID];
        const bool A_is_tri = (typeA == GEO_T_TRIANGLE);
        const bool B_is_tri = (typeB == GEO_T_TRIANGLE);
        bool A_convex = false;
        bool B_convex = false;
        bool A_never = false;
        bool B_never = false;
        if (A_is_tri) {
            const bodyID_t triA = groupPrimA[myID] & deme::CYL_PERIODIC_SPHERE_ID_MASK;
            const bodyID_t ownerA = ownerTriMesh[triA];
            if (ownerA != NULL_BODYID) {
                A_convex = (ownerMeshConvex[ownerA] != 0);
                A_never = (ownerMeshNeverWinner[ownerA] != 0);
            }
        }
        if (B_is_tri) {
            const bodyID_t triB = groupPrimB[myID] & deme::CYL_PERIODIC_SPHERE_ID_MASK;
            const bodyID_t ownerB = ownerTriMesh[triB];
            if (ownerB != NULL_BODYID) {
                B_convex = (ownerMeshConvex[ownerB] != 0);
                B_never = (ownerMeshNeverWinner[ownerB] != 0);
            }
        }
        const bool single_island = (A_is_tri && B_is_tri && A_convex && B_convex);
        forceSingleIsland[myID] = single_island ? 1 : 0;

        notStupidBool_t pickA = 0;
        if (A_never && B_never) {
            pickA = 0;  // deterministic: prefer B when both are never-winner
        } else if (A_never && !B_never) {
            pickA = 0;
        } else if (B_never && !A_never) {
            pickA = 1;
        } else if (nA > nB) {
            pickA = 1;
        } else if (nA < nB) {
            pickA = 0;
        } else {
            if (A_is_tri && B_is_tri) {
                if (A_convex != B_convex) {
                    pickA = A_convex ? 0 : 1;  // prefer concave if tied
                } else {
                    pickA = 0;  // deterministic tie-break: prefer B
                }
            } else if (A_is_tri && !B_is_tri) {
                pickA = 1;
            } else if (B_is_tri && !A_is_tri) {
                pickA = 0;
            } else {
                pickA = 0;  // deterministic tie-break: prefer B
            }
        }
        winnerIsA[myID] = pickA;
        if (single_island) {
            winnerIsTri[myID] = 0;
        } else {
            const geoType_t winnerType = (pickA ? typeA : typeB);
            winnerIsTri[myID] = (winnerType == GEO_T_TRIANGLE) ? 1 : 0;
        }
    }
}

// Select winner primitive and flag if it is a triangle.
__global__ void selectWinnerPrimitive(const contactPairs_t* groupIndex,
                                      const bodyID_t* idA,
                                      const bodyID_t* idB,
                                      const notStupidBool_t* groupWinnerIsA,
                                      const notStupidBool_t* groupWinnerIsTri,
                                      const notStupidBool_t* groupForceSingleIsland,
                                      bodyID_t* winnerID,
                                      notStupidBool_t* winnerIsTri,
                                      size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const contactPairs_t grp = groupIndex[myID];
        if (groupForceSingleIsland[grp] != 0) {
            winnerID[myID] = 0;
            winnerIsTri[myID] = 0;
            return;
        }
        const bool pickA = (groupWinnerIsA[grp] != 0);
        const bodyID_t raw = pickA ? idA[myID] : idB[myID];
        winnerID[myID] = raw & deme::CYL_PERIODIC_SPHERE_ID_MASK;
        winnerIsTri[myID] = groupWinnerIsTri[grp];
    }
}



// Extract a specific side's primitive ID and whether that side is a triangle.
// sideA != 0 selects primitive A / typeA, otherwise primitive B / typeB.
__global__ void extractSidePrimitiveAndTriFlag(const bodyID_t* idA,
                                               const bodyID_t* idB,
                                               const contact_t* contactType,
                                               bodyID_t* sidePrimitive,
                                               notStupidBool_t* sideIsTri,
                                               size_t n,
                                               int sideA) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const contact_t ctype = contactType[myID];
        const geoType_t sideType = sideA ? decodeTypeA<contact_t, geoType_t>(ctype)
                                         : decodeTypeB<contact_t, geoType_t>(ctype);
        // Contact IDs may carry periodic ghost bits; strip them before any indexing/use as primitive IDs.
        sidePrimitive[myID] = (sideA ? idA[myID] : idB[myID]) & deme::CYL_PERIODIC_SPHERE_ID_MASK;
        sideIsTri[myID] = (sideType == GEO_T_TRIANGLE) ? 1 : 0;
    }
}

// Build active triangle keys for compacting (groupIndex, triID).
__global__ void buildActiveTriKeys(const contactPairs_t* groupIndex,
                                   const bodyID_t* winnerID,
                                   const notStupidBool_t* winnerIsTri,
                                   uint64_t* keys,
                                   notStupidBool_t* flags,
                                   size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const notStupidBool_t is_tri = winnerIsTri[myID];
        flags[myID] = is_tri;
        if (is_tri) {
            keys[myID] = (static_cast<uint64_t>(groupIndex[myID]) << 32) | static_cast<uint64_t>(winnerID[myID]);
        } else {
            keys[myID] = 0;
        }
    }
}

// Initialize labels from active triangle keys (label = triID).
__global__ void initActiveTriLabels(const uint64_t* keys, bodyID_t* labels, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        labels[myID] = static_cast<bodyID_t>(keys[myID] & 0xffffffffull);
    }
}

// Count active triangles per group (atomic add).
__global__ void countActiveTriPerGroup(const uint64_t* keys, contactPairs_t* groupCounts, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const contactPairs_t grp = static_cast<contactPairs_t>(keys[myID] >> 32);
        atomicAdd(&groupCounts[grp], (contactPairs_t)1);
    }
}

// Sentinel for "not present" in the active-triangle key list.
// NOTE: contactPairs_t is uint32_t.
constexpr contactPairs_t DEME_INVALID_ACTIVE_TRI_POS = 0xffffffffu;

// Precompute, for each active triangle (in keys[]), the positions of its 3 edge-neighbors in the same active list.
// The output is an array of length 3*n, laid out as [nb0, nb1, nb2] per active triangle.
// If a neighbor is not present (or boundary), the corresponding position is DEME_INVALID_ACTIVE_TRI_POS.
__global__ void buildActiveTriNeighborPos(const uint64_t* keys,
                                         const contactPairs_t* groupStart,
                                         const contactPairs_t* groupCount,
                                         const bodyID_t* triNeighborIndex,
                                         const bodyID_t* triNeighbor1,
                                         const bodyID_t* triNeighbor2,
                                         const bodyID_t* triNeighbor3,
                                         contactPairs_t* outNeighborPos,
                                         size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const uint64_t key = keys[myID];
        const contactPairs_t grp = static_cast<contactPairs_t>(key >> 32);
        const bodyID_t triID = static_cast<bodyID_t>(key & 0xffffffffull);
        const contactPairs_t start = groupStart[grp];
        const contactPairs_t count = groupCount[grp];

        // Default: no neighbors.
        outNeighborPos[3 * myID + 0] = DEME_INVALID_ACTIVE_TRI_POS;
        outNeighborPos[3 * myID + 1] = DEME_INVALID_ACTIVE_TRI_POS;
        outNeighborPos[3 * myID + 2] = DEME_INVALID_ACTIVE_TRI_POS;

        const bodyID_t nb_idx = triNeighborIndex[triID];
        if (nb_idx == NULL_BODYID || count == 0) {
            return;
        }

        const bodyID_t nbs[3] = {triNeighbor1[nb_idx], triNeighbor2[nb_idx], triNeighbor3[nb_idx]};
        for (int e = 0; e < 3; ++e) {
            const bodyID_t nb = nbs[e];
            if (nb == NULL_BODYID) {
                continue;
            }
            const uint64_t target = (static_cast<uint64_t>(grp) << 32) | static_cast<uint64_t>(nb);
            contactPairs_t left = 0;
            contactPairs_t right = count;
            while (left < right) {
                contactPairs_t mid = left + (right - left) / 2;
                const uint64_t mid_key = keys[start + mid];
                if (mid_key < target) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            if (left < count && keys[start + left] == target) {
                outNeighborPos[3 * myID + e] = start + left;
            }
        }
    }
}

// Label propagation for active triangles within each group.
__global__ void propagateActiveTriLabels(const uint64_t* keys,
                                         const bodyID_t* labelsIn,
                                         bodyID_t* labelsOut,
                                         const contactPairs_t* groupStart,
                                         const contactPairs_t* groupCount,
                                         const bodyID_t* triNeighborIndex,
                                         const bodyID_t* triNeighbor1,
                                         const bodyID_t* triNeighbor2,
                                         const bodyID_t* triNeighbor3,
                                         size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const uint64_t key = keys[myID];
        const contactPairs_t grp = static_cast<contactPairs_t>(key >> 32);
        const bodyID_t triID = static_cast<bodyID_t>(key & 0xffffffffull);
        const contactPairs_t start = groupStart[grp];
        const contactPairs_t count = groupCount[grp];
        bodyID_t label = labelsIn[myID];

        const bodyID_t nb_idx = triNeighborIndex[triID];
        if (nb_idx == NULL_BODYID) {
            labelsOut[myID] = label;
            return;
        }
        bodyID_t nbs[3] = {triNeighbor1[nb_idx], triNeighbor2[nb_idx], triNeighbor3[nb_idx]};
        for (int e = 0; e < 3; ++e) {
            const bodyID_t nb = nbs[e];
            if (nb == NULL_BODYID || count == 0) {
                continue;
            }
            const uint64_t target = (static_cast<uint64_t>(grp) << 32) | static_cast<uint64_t>(nb);
            contactPairs_t left = 0;
            contactPairs_t right = count;
            while (left < right) {
                contactPairs_t mid = left + (right - left) / 2;
                const uint64_t mid_key = keys[start + mid];
                if (mid_key < target) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            if (left < count) {
                const uint64_t found = keys[start + left];
                if (found == target) {
                    const bodyID_t nb_label = labelsIn[start + left];
                    if (nb_label < label) {
                        label = nb_label;
                    }
                }
            }
        }
        labelsOut[myID] = label;
    }
}

// Label propagation using precomputed neighbor positions (avoids per-iteration binary searches).
// If changedFlag is non-null, it will be set to 1 if any label changes in this iteration.
__global__ void propagateActiveTriLabelsFromNeighborPos(const bodyID_t* labelsIn,
                                                        bodyID_t* labelsOut,
                                                        const contactPairs_t* neighborPos,
                                                        contactPairs_t* changedFlag,
                                                        size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;

    // Block-local change flag to reduce global atomics.
    __shared__ int blockChanged;
    if (threadIdx.x == 0) {
        blockChanged = 0;
    }
    __syncthreads();

    if (myID < n) {
        const bodyID_t oldLabel = labelsIn[myID];
        bodyID_t label = oldLabel;
        const contactPairs_t p0 = neighborPos[3 * myID + 0];
        const contactPairs_t p1 = neighborPos[3 * myID + 1];
        const contactPairs_t p2 = neighborPos[3 * myID + 2];

        if (p0 != DEME_INVALID_ACTIVE_TRI_POS) {
            const bodyID_t nb_label = labelsIn[p0];
            if (nb_label < label) {
                label = nb_label;
            }
        }
        if (p1 != DEME_INVALID_ACTIVE_TRI_POS) {
            const bodyID_t nb_label = labelsIn[p1];
            if (nb_label < label) {
                label = nb_label;
            }
        }
        if (p2 != DEME_INVALID_ACTIVE_TRI_POS) {
            const bodyID_t nb_label = labelsIn[p2];
            if (nb_label < label) {
                label = nb_label;
            }
        }

        labelsOut[myID] = label;
        if (changedFlag != nullptr && label != oldLabel) {
            atomicExch(&blockChanged, 1);
        }
    }

    __syncthreads();
    if (changedFlag != nullptr && threadIdx.x == 0 && blockChanged) {
        atomicExch(changedFlag, (contactPairs_t)1);
    }
}

// Assign per-contact island labels using winner primitive and active triangle labels.
__global__ void assignContactIslandLabel(const contactPairs_t* groupIndex,
                                         const bodyID_t* winnerID,
                                         const notStupidBool_t* winnerIsTri,
                                         const uint64_t* activeKeys,
                                         const bodyID_t* activeLabels,
                                         const contactPairs_t* groupStart,
                                         const contactPairs_t* groupCount,
                                         bodyID_t* outLabels,
                                         size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const bodyID_t prim = winnerID[myID];
        if (winnerIsTri[myID] == 0) {
            outLabels[myID] = prim;
            return;
        }
        const contactPairs_t grp = groupIndex[myID];
        const contactPairs_t start = groupStart[grp];
        const contactPairs_t count = groupCount[grp];
        if (count == 0) {
            outLabels[myID] = prim;
            return;
        }
        const uint64_t target = (static_cast<uint64_t>(grp) << 32) | static_cast<uint64_t>(prim);
        contactPairs_t left = 0;
        contactPairs_t right = count;
        while (left < right) {
            contactPairs_t mid = left + (right - left) / 2;
            const uint64_t mid_key = activeKeys[start + mid];
            if (mid_key < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (left < count && activeKeys[start + left] == target) {
            outLabels[myID] = activeLabels[start + left];
        } else {
            outLabels[myID] = prim;
        }
    }
}

// Simple copy kernel for bodyID arrays.
__global__ void copyBodyIDArray(const bodyID_t* in, bodyID_t* out, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        out[myID] = in[myID];
    }
}

// Build a winner-invariant persistent island label.
// For tri-tri contacts we combine side-A and side-B labels deterministically so winner-side
// switches do not reset history. For other contact types we keep the winner-derived label.
__global__ void buildPersistentIslandLabel(const contact_t* contactTypes,
                                           const bodyID_t* winnerLabels,
                                           const bodyID_t* sideLabelsA,
                                           const bodyID_t* sideLabelsB,
                                           bodyID_t* outLabels,
                                           size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        if (contactTypes[myID] == TRIANGLE_TRIANGLE_CONTACT) {
            const bodyID_t a = sideLabelsA[myID];
            const bodyID_t b = sideLabelsB[myID];
            const uint64_t lo = static_cast<uint64_t>((a <= b) ? a : b);
            const uint64_t hi = static_cast<uint64_t>((a <= b) ? b : a);
            // 64->32 mix, deterministic and cheap on GPU.
            uint64_t x = (hi << 32) | lo;
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdULL;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53ULL;
            x ^= x >> 33;
            outLabels[myID] = static_cast<bodyID_t>(x & 0xffffffffULL);
        } else {
            outLabels[myID] = winnerLabels[myID];
        }
    }
}

// Build composite key parts (contactType + patchA, patchB + label) for island grouping.
__global__ void buildIslandCompositeKeyParts(const patchIDPair_t* patchPairs,
                                             const contact_t* contactTypes,
                                             const bodyID_t* labels,
                                             uint64_t* key_hi,
                                             uint64_t* key_lo,
                                             size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const patchIDPair_t pair = patchPairs[myID];
        const uint64_t hi = static_cast<uint64_t>(pair >> 32);
        const uint64_t lo = static_cast<uint64_t>(pair & 0xffffffffull);
        // key_hi: contactType + patchA (primary key)
        key_hi[myID] = (static_cast<uint64_t>(contactTypes[myID]) << 32) | hi;
        // key_lo: patchB + island label (secondary key)
        key_lo[myID] = (lo << 32) | static_cast<uint64_t>(labels[myID]);
    }
}

// Mark new composite groups for sorted (key_hi, key_lo) arrays.
__global__ void markNewCompositeGroups64(const uint64_t* key_hi,
                                         const uint64_t* key_lo,
                                         contactPairs_t* isNewGroup,
                                         size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        if (myID == 0) {
            isNewGroup[myID] = 0;
        } else {
            const bool new_hi = key_hi[myID] != key_hi[myID - 1];
            const bool new_lo = key_lo[myID] != key_lo[myID - 1];
            isNewGroup[myID] = (new_hi || new_lo) ? 1 : 0;
        }
    }
}

// Build a sortable 64-bit key from (idB, contactType, persistency_preference).
// - High 32 bits: idB (so contacts with the same idB group together)
// - Low bits: contactType then persistency (so within a duplicate group, the preferred contact comes first)
//
// If prefer_persistent is true, persistent contacts (CONTACT_IS_PERSISTENT) sort before non-persistent ones.
__global__ void buildKeyBTypePersist(const bodyID_t* idB,
                                     const contact_t* contactType,
                                     const notStupidBool_t* persistency,
                                     unsigned long long* keys,
                                     size_t n,
                                     bool prefer_persistent) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const unsigned long long high = (static_cast<unsigned long long>(idB[myID]) << 32);
        const unsigned long long type_part = (static_cast<unsigned long long>(contactType[myID]) << 1);
        const unsigned long long persist_part =
            prefer_persistent ? static_cast<unsigned long long>(persistency[myID] == CONTACT_NOT_PERSISTENT) : 0ull;
        keys[myID] = high | type_part | persist_part;
    }
}

// Mark (idA, idB, contactType) duplicates in a lexicographically sorted contact list.
// retain_flags[i] is set to 1 if this row should be kept, 0 if it's a duplicate of the previous row.
__global__ void markUniqueTriples(const bodyID_t* idA,
                                  const bodyID_t* idB,
                                  const contact_t* contactType,
                                  notStupidBool_t* retain_flags,
                                  size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        if (myID == 0) {
            retain_flags[myID] = (notStupidBool_t)1;
            return;
        }
        const bool is_dup = (idA[myID] == idA[myID - 1]) && (idB[myID] == idB[myID - 1]) &&
                            (contactType[myID] == contactType[myID - 1]);
        retain_flags[myID] = is_dup ? (notStupidBool_t)0 : (notStupidBool_t)1;
    }
}

__global__ void setFirstFlagToOne(contactPairs_t* flags, size_t n) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && n > 0) {
        flags[0] = 1;
    }
}

// Set NULL_MAPPING_PARTNER for contacts of a specific type segment.
// Used when current step has contacts of this type but previous step does not.
//
// Parameters:
//   contactMapping: Output mapping array (entire array, not segment)
//   curr_start: Starting index in current arrays for this contact type segment
//   curr_count: Number of contacts of this type in current step
__global__ void setNullMappingForType(contactPairs_t* contactMapping,
                                      contactPairs_t curr_start,
                                      contactPairs_t curr_count) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < curr_count) {
        contactPairs_t curr_idx = curr_start + myID;
        contactMapping[curr_idx] = NULL_MAPPING_PARTNER;
    }
}

// Build patch-based contact mapping for a single contact type segment.
// This kernel operates on a specific segment of the contact arrays identified by start offsets and counts.
// Each thread processes one current contact and searches for a match in the previous contacts of the same type.
//
// Parameters:
//   curr_idPatchA, curr_idPatchB: Current step's patch contact arrays (entire arrays, not segment)
//   prev_idPatchA, prev_idPatchB: Previous step's patch contact arrays (entire arrays, not segment)
//   contactMapping: Output mapping array (entire array, not segment)
//   curr_start: Starting index in current arrays for this contact type segment
//   curr_count: Number of contacts of this type in current step
//   prev_start: Starting index in previous arrays for this contact type segment
//   prev_count: Number of contacts of this type in previous step
__global__ void buildPatchContactMappingForType(bodyID_t* curr_idPatchA,
                                                bodyID_t* curr_idPatchB,
                                                bodyID_t* curr_patchIsland,
                                                bodyID_t* prev_idPatchA,
                                                bodyID_t* prev_idPatchB,
                                                bodyID_t* prev_patchIsland,
                                                contactPairs_t* contactMapping,
                                                bool allow_label_fallback,
                                                bool allow_island_fallback,
                                                bool allow_rank_clamp,
                                                contactPairs_t curr_start,
                                                contactPairs_t curr_count,
                                                contactPairs_t prev_start,
                                                contactPairs_t prev_count) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < curr_count) {
        // Absolute index in the full contact array
        contactPairs_t curr_idx = curr_start + myID;

        bodyID_t curr_A = curr_idPatchA[curr_idx];
        bodyID_t curr_B = curr_idPatchB[curr_idx];
        bodyID_t curr_L = curr_patchIsland[curr_idx];

        // Default: no match found
        contactPairs_t my_partner = NULL_MAPPING_PARTNER;

        // Binary search within the previous type segment for the matching A/B pair
        // The segment is sorted by the combined patch ID pair
        contactPairs_t left = 0;
        contactPairs_t right = prev_count;
        while (left < right) {
            contactPairs_t mid = left + (right - left) / 2;
            contactPairs_t prev_idx = prev_start + mid;
            bodyID_t prev_A = prev_idPatchA[prev_idx];
            bodyID_t prev_B = prev_idPatchB[prev_idx];

            // Compare (A, B, label) lexicographically
            if (prev_A < curr_A || (prev_A == curr_A && (prev_B < curr_B ||
                                                         (prev_B == curr_B && prev_patchIsland[prev_idx] < curr_L)))) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Check if we found a match at position left
        if (left < prev_count) {
            contactPairs_t prev_idx = prev_start + left;
            bodyID_t prev_A = prev_idPatchA[prev_idx];
            bodyID_t prev_B = prev_idPatchB[prev_idx];
            bodyID_t prev_L = prev_patchIsland[prev_idx];
            if (prev_A == curr_A && prev_B == curr_B && prev_L == curr_L) {
                my_partner = prev_idx;
            }
        }

        // If a patch-pair run collapses from many previous contacts to a single current contact,
        // preserve continuity by selecting a stable representative (rank 0 in the pair run),
        // instead of binding to a potentially oscillating island label branch.
        if (my_partner == NULL_MAPPING_PARTNER && allow_island_fallback) {
            contactPairs_t prev_pair_lo = 0;
            contactPairs_t prev_pair_hi = prev_count;
            while (prev_pair_lo < prev_pair_hi) {
                contactPairs_t mid = prev_pair_lo + (prev_pair_hi - prev_pair_lo) / 2;
                contactPairs_t prev_idx = prev_start + mid;
                const bodyID_t prev_A = prev_idPatchA[prev_idx];
                const bodyID_t prev_B = prev_idPatchB[prev_idx];
                if (prev_A < curr_A || (prev_A == curr_A && prev_B < curr_B)) {
                    prev_pair_lo = mid + 1;
                } else {
                    prev_pair_hi = mid;
                }
            }

            contactPairs_t prev_pair_end = prev_pair_lo;
            while (prev_pair_end < prev_count) {
                const contactPairs_t prev_idx = prev_start + prev_pair_end;
                if (prev_idPatchA[prev_idx] == curr_A && prev_idPatchB[prev_idx] == curr_B) {
                    prev_pair_end++;
                } else {
                    break;
                }
            }

            contactPairs_t curr_pair_lo = 0;
            contactPairs_t curr_pair_hi = curr_count;
            while (curr_pair_lo < curr_pair_hi) {
                contactPairs_t mid = curr_pair_lo + (curr_pair_hi - curr_pair_lo) / 2;
                contactPairs_t curr_mid_idx = curr_start + mid;
                const bodyID_t mid_A = curr_idPatchA[curr_mid_idx];
                const bodyID_t mid_B = curr_idPatchB[curr_mid_idx];
                if (mid_A < curr_A || (mid_A == curr_A && mid_B < curr_B)) {
                    curr_pair_lo = mid + 1;
                } else {
                    curr_pair_hi = mid;
                }
            }

            contactPairs_t curr_pair_end = curr_pair_lo;
            while (curr_pair_end < curr_count) {
                const contactPairs_t curr_end_idx = curr_start + curr_pair_end;
                if (curr_idPatchA[curr_end_idx] == curr_A && curr_idPatchB[curr_end_idx] == curr_B) {
                    curr_pair_end++;
                } else {
                    break;
                }
            }

            const contactPairs_t prev_pair_count = prev_pair_end - prev_pair_lo;
            const contactPairs_t curr_pair_count = curr_pair_end - curr_pair_lo;
            if (curr_pair_count == 1 && prev_pair_count > 0) {
                if (allow_rank_clamp) {
                    // For sph-tri, a deterministic representative avoids intermittent
                    // history resets from transient over-splitting.
                    my_partner = prev_start + prev_pair_lo;
                } else {
                    // For tri-tri, keep label-nearest representative to preserve
                    // smoother continuity in grazing transitions.
                    contactPairs_t best_local = prev_pair_lo;
                    bodyID_t best_dist = (bodyID_t)(-1);
                    for (contactPairs_t k = prev_pair_lo; k < prev_pair_end; k++) {
                        const contactPairs_t prev_idx = prev_start + k;
                        const bodyID_t prev_L = prev_patchIsland[prev_idx];
                        const bodyID_t d = (prev_L > curr_L) ? (prev_L - curr_L) : (curr_L - prev_L);
                        if (d < best_dist) {
                            best_dist = d;
                            best_local = k;
                        }
                    }
                    my_partner = prev_start + best_local;
                }
            }
        }

        // Optional fallback for tri-tri history continuity:
        // If island labels churn, preserve continuity by matching within the (A,B) run using stable ordinal rank.
        // This is more robust than requiring label equality and avoids history holes on seam/island relabeling.
        if (my_partner == NULL_MAPPING_PARTNER && allow_label_fallback) {
            // Locate [prev_pair_lo, prev_pair_hi) range for (curr_A, curr_B) in previous segment.
            contactPairs_t prev_pair_lo = 0;
            contactPairs_t prev_pair_hi = prev_count;
            while (prev_pair_lo < prev_pair_hi) {
                contactPairs_t mid = prev_pair_lo + (prev_pair_hi - prev_pair_lo) / 2;
                contactPairs_t prev_idx = prev_start + mid;
                const bodyID_t prev_A = prev_idPatchA[prev_idx];
                const bodyID_t prev_B = prev_idPatchB[prev_idx];
                if (prev_A < curr_A || (prev_A == curr_A && prev_B < curr_B)) {
                    prev_pair_lo = mid + 1;
                } else {
                    prev_pair_hi = mid;
                }
            }

            contactPairs_t prev_pair_end = prev_pair_lo;
            while (prev_pair_end < prev_count) {
                const contactPairs_t prev_idx = prev_start + prev_pair_end;
                if (prev_idPatchA[prev_idx] == curr_A && prev_idPatchB[prev_idx] == curr_B) {
                    prev_pair_end++;
                } else {
                    break;
                }
            }

            const contactPairs_t prev_pair_count = prev_pair_end - prev_pair_lo;
            if (prev_pair_count > 0) {
                // Prefer label-nearest partner in the previous (A,B) run.
                // This is more stable than rank mapping when run order jitters.
                contactPairs_t best_local = prev_pair_lo;
                bodyID_t best_dist = (bodyID_t)(-1);
                for (contactPairs_t k = prev_pair_lo; k < prev_pair_end; k++) {
                    const contactPairs_t prev_idx = prev_start + k;
                    const bodyID_t prev_L = prev_patchIsland[prev_idx];
                    const bodyID_t d = (prev_L > curr_L) ? (prev_L - curr_L) : (curr_L - prev_L);
                    if (d < best_dist) {
                        best_dist = d;
                        best_local = k;
                    }
                }
                my_partner = prev_start + best_local;
            }
        }

        // Secondary tri-tri/sph-tri continuity fallback:
        // if patch-pair mapping fails due support-triangle switches, preserve history by
        // island-label stable rank within the contact-type segment.
        if (my_partner == NULL_MAPPING_PARTNER && allow_island_fallback) {
            contactPairs_t curr_island_rank = 0;
            for (contactPairs_t k = 0; k < myID; k++) {
                const contactPairs_t curr_k_idx = curr_start + k;
                if (curr_patchIsland[curr_k_idx] == curr_L) {
                    curr_island_rank++;
                }
            }

            contactPairs_t prev_island_seen = 0;
            for (contactPairs_t k = 0; k < prev_count; k++) {
                const contactPairs_t prev_k_idx = prev_start + k;
                if (prev_patchIsland[prev_k_idx] == curr_L) {
                    if (prev_island_seen == curr_island_rank) {
                        my_partner = prev_k_idx;
                        break;
                    }
                    prev_island_seen++;
                }
            }
        }

        // Final continuity guard: if patch-pair based matching still fails (e.g., transient
        // pair-key jitter), keep tangential history by taking the label-nearest contact in
        // the same type segment.
        if (my_partner == NULL_MAPPING_PARTNER && allow_label_fallback && prev_count > 0) {
            contactPairs_t best_local = 0;
            bodyID_t best_dist = (bodyID_t)(-1);
            for (contactPairs_t k = 0; k < prev_count; k++) {
                const contactPairs_t prev_idx = prev_start + k;
                const bodyID_t prev_L = prev_patchIsland[prev_idx];
                const bodyID_t d = (prev_L > curr_L) ? (prev_L - curr_L) : (curr_L - prev_L);
                if (d < best_dist) {
                    best_dist = d;
                    best_local = k;
                }
            }
            my_partner = prev_start + best_local;
        }

        contactMapping[curr_idx] = my_partner;
    }
}

// If both sides are cylindrical-periodic ghosts, drop history.
// Single-sided ghost contacts can legitimately persist across steps and should keep history.
__global__ void resetMappingForGhostContacts(contactPairs_t* contactMapping,
                                             const bodyID_t* idPatchA,
                                             const bodyID_t* idPatchB,
                                             size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        const bool ghostA = (idPatchA[myID] & CYL_PERIODIC_GHOST_FLAG) != 0;
        const bool ghostB = (idPatchB[myID] & CYL_PERIODIC_GHOST_FLAG) != 0;
        if (ghostA && ghostB) {
            contactMapping[myID] = NULL_MAPPING_PARTNER;
        }
    }
}

// The rest are old kernels only used in DEME2.x for primitive-primitive contact mapping
/*
__global__ void fillRunLengthArray(primitivesPrimTouches_t* runlength_full,
                                   bodyID_t* unique_ids,
                                   primitivesPrimTouches_t* runlength,
                                   size_t numUnique) {
    bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < numUnique) {
        bodyID_t i = unique_ids[myID];
        runlength_full[i] = runlength[myID];
    }
}

__global__ void buildPersistentMap(primitivesPrimTouches_t* new_idA_runlength_full,
                                   primitivesPrimTouches_t* old_idA_runlength_full,
                                   contactPairs_t* new_idA_scanned_runlength,
                                   contactPairs_t* old_idA_scanned_runlength,
                                   contactPairs_t* mapping,
                                   DEMDataKT* granData,
                                   size_t nGeoSafe) {
    bodyID_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < nGeoSafe) {
        primitivesPrimTouches_t new_cnt_count = new_idA_runlength_full[myID];
        primitivesPrimTouches_t old_cnt_count = old_idA_runlength_full[myID];
        // If this idA has non-zero runlength in new: a potential enduring sphere
        if (new_cnt_count > 0) {
            // Where should I start looking? Grab the offset.
            contactPairs_t new_cnt_offset = new_idA_scanned_runlength[myID];
            contactPairs_t old_cnt_offset = old_idA_scanned_runlength[myID];
            for (primitivesPrimTouches_t i = 0; i < new_cnt_count; i++) {
                // Current contact number we are inspecting
                contactPairs_t this_contact = new_cnt_offset + i;
                bodyID_t new_idB = granData->idPrimitiveB[this_contact];
                contact_t new_cntType = granData->contactTypePrimitive[this_contact];
                // Mark it as no matching pair found, being a new contact; modify it later
                contactPairs_t my_partner = NULL_MAPPING_PARTNER;
                // If this is a fake contact, we can move on
                if (new_cntType == NOT_A_CONTACT) {
                    mapping[this_contact] = my_partner;
                    continue;
                }
                // Loop through the old idB to see if there is a match
                for (primitivesPrimTouches_t j = 0; j < old_cnt_count; j++) {
                    bodyID_t old_idB = granData->previous_idPrimitiveB[old_cnt_offset + j];
                    contact_t old_cntType = granData->previous_contactTypePrimitive[old_cnt_offset + j];
                    // If both idB and contact type match, then it is an enduring contact, write it to the mapping
                    // array
                    if (new_idB == old_idB && new_cntType == old_cntType) {
                        my_partner = old_cnt_offset + j;
                        break;
                    }
                }
                // If old_cnt_count == 0, it is automatically NULL_MAPPING_PARTNER
                mapping[this_contact] = my_partner;
            }
        }
    }
}

__global__ void lineNumbers(contactPairs_t* arr, size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        arr[myID] = myID;
    }
}

__global__ void convertToAndFrom(contactPairs_t* old_arr_unsort_to_sort_map,
                                 contactPairs_t* converted_map,
                                 size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        contactPairs_t map_from = old_arr_unsort_to_sort_map[myID];
        converted_map[map_from] = myID;
    }
}

__global__ void rearrangeMapping(contactPairs_t* map_sorted,
                                 contactPairs_t* old_arr_unsort_to_sort_map,
                                 size_t n) {
    contactPairs_t myID = blockIdx.x * blockDim.x + threadIdx.x;
    if (myID < n) {
        contactPairs_t map_to = map_sorted[myID];
        if (map_to != NULL_MAPPING_PARTNER)
            map_sorted[myID] = old_arr_unsort_to_sort_map[map_to];
    }
}
*/

}  // namespace deme

#endif
