// DEM collision-related kernel collection

#ifndef DEME_COLLI_KERNELS_ST_TT_CUH
#define DEME_COLLI_KERNELS_ST_TT_CUH

#include <DEM/Defines.h>
#include <DEMHelperKernels.cuh>

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------
__device__ __forceinline__ float3 make_zero3_float()  { return make_float3(0.f, 0.f, 0.f); }
__device__ __forceinline__ double3 make_zero3_double(){ return make_double3(0.0, 0.0, 0.0); }

template <typename T1>
__device__ __forceinline__ T1 make_zero3();
template <>
__device__ __forceinline__ float3 make_zero3<float3>() { return make_zero3_float(); }
template <>
__device__ __forceinline__ double3 make_zero3<double3>() { return make_zero3_double(); }

template <typename T1>
__device__ __forceinline__ T1 make_one3(int x, int y, int z);
template <>
__device__ __forceinline__ float3 make_one3<float3>(int x, int y, int z) {
    return make_float3((float)x, (float)y, (float)z);
}
template <>
__device__ __forceinline__ double3 make_one3<double3>(int x, int y, int z) {
    return make_double3((double)x, (double)y, (double)z);
}
template <typename T>
__device__ __forceinline__ T tmin2(T a, T b) { return a < b ? a : b; }
template <typename T>
__device__ __forceinline__ T tmax2(T a, T b) { return a > b ? a : b; }
template <typename T>
__device__ __forceinline__ T tmin3(T a, T b, T c) { return tmin2(a, tmin2(b, c)); }
template <typename T>
__device__ __forceinline__ T tmax3(T a, T b, T c) { return tmax2(a, tmax2(b, c)); }

template <typename S>
__device__ __forceinline__ S absT(S x) { return x >= (S)0 ? x : -x; }

template <typename S>
__device__ __forceinline__ S min2(S a, S b) { return a < b ? a : b; }

template <typename S>
__device__ __forceinline__ S max2(S a, S b) { return a > b ? a : b; }

template <typename S>
__device__ __forceinline__ void sort2(S& a, S& b) { if (a > b) { const S t = a; a = b; b = t; } }

// ------------------------------------------------------------------
// Triangle-analytical object collision detection utilities
// ------------------------------------------------------------------

template <typename T1, typename T2>
bool __device__ tri_plane_penetration(const T1** tri,
                                      const T1& entityLoc,
                                      const float3& entityDir,
                                      T2& overlapDepth,
                                      T2& overlapArea,
                                      T1& contactPnt) {
    // signed distances
    T2 d[3];
    // penetration depth: deepest point in triangle
    T2 dmin = DEME_HUGE_FLOAT;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        d[i] = planeSignedDistance<T2>(*tri[i], entityLoc, entityDir);
        if (d[i] < dmin)
            dmin = d[i];
    }
    // build clipped polygon
    T1 poly[4];
    int nNode = 0;                 // max 4 poly nodes
    bool hasIntersection = false;  // one edge intersecting the plane
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        int j = (i + 1) % 3;  // compare with next vertex
        bool in_i = (d[i] < 0.0);
        bool in_j = (d[j] < 0.0);

        // ^ means one is in, the other is out
        if (in_i ^ in_j) {
            T2 t = d[i] / (d[i] - d[j]);  // between 0 and 1
            T1 inter = *tri[i] + (*tri[j] - *tri[i]) * t;
            // Only register inside points once - project them onto the plane
            if (in_i) {
                // Project the submerging node onto the plane
                T1 projectedNode = *tri[i] - d[i] * to_real3<float3, T1>(entityDir);
                poly[nNode++] = projectedNode;
            }
            poly[nNode++] = inter;
            hasIntersection = true;
        }
    }

    // Handle the case where all three vertices are submerged (no edge crosses the plane)
    if (!hasIntersection) {
        // Check if all vertices are below the plane
        bool allBelow = true;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            if (d[i] >= 0.0) {
                allBelow = false;
                break;
            }
        }
        if (allBelow) {
            // All vertices are below the plane - project all three onto the plane
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                T1 projectedNode = *tri[i] - d[i] * to_real3<float3, T1>(entityDir);
                poly[nNode++] = projectedNode;
            }
            hasIntersection = true;  // We now have a valid polygon
        }
    }

    // centroid of contact
    T1 centroid;
    centroid.x = 0.;
    centroid.y = 0.;
    centroid.z = 0.;
    if (hasIntersection) {  // If has intersection, centroid of the (inside) polygon
        for (int i = 0; i < nNode; i++)
            centroid = centroid + poly[i];
        centroid = centroid / T2(nNode);
    } else {  // If no intersection, centroid is just average of all tri verts
#pragma unroll
        for (int i = 0; i < 3; i++)
            centroid = centroid + *tri[i];
        centroid = centroid / 3.0;
    }

    // We use the convention that if in contact, overlapDepth is positive
    overlapDepth = -dmin;
    bool in_contact = (overlapDepth >= 0.);
    // The centroid's projection to the plane
    T1 projection =
        centroid - planeSignedDistance<T2>(centroid, entityLoc, entityDir) * to_real3<float3, T1>(entityDir);

    // Calculate the area of the clipping polygon using fan triangulation from centroid
    float overlap_area_f = 0.0f;
    if (hasIntersection && nNode >= 3) {
        const float3 centroid_f = to_float3(centroid);
        for (int i = 0; i < nNode; ++i) {
            float3 v1 = to_float3(poly[i]) - centroid_f;
            float3 v2 = to_float3(poly[(i + 1) % nNode]) - centroid_f;
            float3 crossProd = cross(v1, v2);
            overlap_area_f += sqrtf(dot(crossProd, crossProd));
        }
        overlap_area_f *= 0.5f;
    }
    overlapArea = static_cast<T2>(overlap_area_f);

    // cntPnt is from the projection point, go half penetration depth.
    // Note this penetration depth is signed, so if no contact, we go positive plane normal; if in contact, we go
    // negative plane normal. As such, cntPnt always exists and this is important for the cases with extraMargin.
    contactPnt = projection - (overlapDepth * 0.5) * to_real3<float3, T1>(entityDir);
    return in_contact;
}

template <typename T1>
inline __host__ __device__ bool planar_cyl_plane_from_ref(const T1& ref,
                                                          const T1& entityLoc,
                                                          const float3& entityDir,
                                                          const float& radius,
                                                          const float& normal_sign,
                                                          T1& plane_point,
                                                          float3& plane_normal) {
    T1 radial_vec = cylRadialDistanceVec<T1>(ref, entityLoc, entityDir);
    const auto dist = length(radial_vec);
    if (dist <= (decltype(dist))DEME_TINY_FLOAT) {
        return false;
    }
    const T1 radial_dir = radial_vec / dist;
    const float dist_plane = normal_sign * (radius - (float)dist);
    if (dist_plane < 0) {
        return false;
    }
    plane_normal = to_real3<T1, float3>(-normal_sign * radial_dir);
    const T1 axis_point = ref - radial_vec;
    plane_point = axis_point + radial_dir * radius;
    return true;
}

template <typename T1, typename T2>
bool __device__ tri_cyl_penetration(const T1** tri,
                                    const T1& entityLoc,
                                    const float3& entityDir,
                                    const float& entitySize1,
                                    const float& entitySize2,
                                    const float& normal_sign,
                                    float3& contact_normal,
                                    T2& overlapDepth,
                                    T2& overlapArea,
                                    T1& contactPnt) {
    return false;
}

// Check only, no contact point, depth, area output
template <typename T1>
__host__ __device__ deme::contact_t checkTriEntityOverlap(const T1& A,
                                                          const T1& B,
                                                          const T1& C,
                                                          const deme::objType_t& typeB,
                                                          const T1& entityLoc,
                                                          const float3& entityDir,
                                                          const float& entitySize1,
                                                          const float& entitySize2,
                                                          const float& entitySize3,
                                                          const float& normal_sign,
                                                          const float& beta4Entity) {
    const T1* tri[] = {&A, &B, &C};
    switch (typeB) {
        case (deme::ANAL_OBJ_TYPE_PLANE): {
            for (const T1*& v : tri) {
                // Always cast to double
                double d = planeSignedDistance<double>(*v, entityLoc, entityDir);
                double overlapDepth = beta4Entity - d;
                // printf("v point %f %f %f, entityLoc %f %f %f\n", v->x, v->y, v->z, entityLoc.x, entityLoc.y,
                // entityLoc.z);
                if (overlapDepth >= 0.0)
                    return deme::TRIANGLE_ANALYTICAL_CONTACT;
            }
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_PLATE): {
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_CYL_INF): {
            for (const T1*& v : tri) {
                // Radial distance vector is from cylinder axis to a point
                double3 vec = cylRadialDistanceVec<double3>(*v, entityLoc, entityDir);
                // Also, inward normal is 1, outward is -1, so it's the signed dist from point to cylinder wall
                // (positive if same orientation, negative if opposite)
                double signed_dist = (entitySize1 - length(vec)) * normal_sign;
                if (signed_dist <= beta4Entity)
                    return deme::TRIANGLE_ANALYTICAL_CONTACT;
            }
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_PLANAR_CYL): {
            T1 centroid = (A + B + C) / 3.0;
            T1 plane_point;
            float3 plane_normal;
            if (!planar_cyl_plane_from_ref(centroid, entityLoc, entityDir, entitySize1, normal_sign, plane_point,
                                           plane_normal)) {
                return deme::NOT_A_CONTACT;
            }
            for (const T1*& v : tri) {
                double d = planeSignedDistance<double>(*v, plane_point, plane_normal);
                double overlapDepth = beta4Entity - d;
                if (overlapDepth >= 0.0)
                    return deme::TRIANGLE_ANALYTICAL_CONTACT;
            }
            return deme::NOT_A_CONTACT;
        }
        default:
            return deme::NOT_A_CONTACT;
    }
}

// Fast FP32-only overlap check for kT contact detection (no penetration/area outputs).
template <typename T1>
__host__ __device__ deme::contact_t checkTriEntityOverlapFP32(const T1& A,
                                                              const T1& B,
                                                              const T1& C,
                                                              const deme::objType_t& typeB,
                                                              const T1& entityLoc,
                                                              const float3& entityDir,
                                                              const float& entitySize1,
                                                              const float& entitySize2,
                                                              const float& entitySize3,
                                                              const float& normal_sign,
                                                              const float& beta4Entity) {
    const T1* tri[] = {&A, &B, &C};
    switch (typeB) {
        case (deme::ANAL_OBJ_TYPE_PLANE): {
            for (const T1*& v : tri) {
                const float d = planeSignedDistance<float>(*v, entityLoc, entityDir);
                const float overlapDepth = beta4Entity - d;
                if (overlapDepth >= 0.0f)
                    return deme::TRIANGLE_ANALYTICAL_CONTACT;
            }
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_PLATE): {
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_CYL_INF): {
            for (const T1*& v : tri) {
                float3 vec = cylRadialDistanceVec<float3>(*v, entityLoc, entityDir);
                const float signed_dist = (entitySize1 - length(vec)) * normal_sign;
                if (signed_dist <= beta4Entity)
                    return deme::TRIANGLE_ANALYTICAL_CONTACT;
            }
            return deme::NOT_A_CONTACT;
        }
        case (deme::ANAL_OBJ_TYPE_PLANAR_CYL): {
            T1 centroid = (A + B + C) / 3.0f;
            T1 plane_point;
            float3 plane_normal;
            if (!planar_cyl_plane_from_ref(centroid, entityLoc, entityDir, entitySize1, normal_sign, plane_point,
                                           plane_normal)) {
                return deme::NOT_A_CONTACT;
            }
            for (const T1*& v : tri) {
                const float d = planeSignedDistance<float>(*v, plane_point, plane_normal);
                const float overlapDepth = beta4Entity - d;
                if (overlapDepth >= 0.0f)
                    return deme::TRIANGLE_ANALYTICAL_CONTACT;
            }
            return deme::NOT_A_CONTACT;
        }
        default:
            return deme::NOT_A_CONTACT;
    }
}

// NOTE: Due to our algorithm needs a overlapDepth even in the case of no contact (because of extraMargin; and with
// extraMargin, negative overlapDepth can be considered in-contact), our tri-anal CD algorithm needs to always return a
// overlapDepth, even in the case of no contact. This is different from the usual CD algorithms which only return a
// overlapDepth. Also unlike tri-sph contact which has a unified util function, calcTriEntityOverlap is different from
// checkTriEntityOverlap.
template <typename T1, typename T2>
bool __device__ calcTriEntityOverlap(const T1& A,
                                     const T1& B,
                                     const T1& C,
                                     const deme::objType_t& entityType,
                                     const T1& entityLoc,
                                     const float3& entityDir,
                                     const float& entitySize1,
                                     const float& entitySize2,
                                     const float& entitySize3,
                                     const float& normal_sign,
                                     T1& contactPnt,
                                     float3& contact_normal,
                                     T2& overlapDepth,
                                     T2& overlapArea) {
    const T1* tri[] = {&A, &B, &C};
    bool in_contact;
    switch (entityType) {
        case deme::ANAL_OBJ_TYPE_PLANE:
            in_contact =
                tri_plane_penetration<T1, T2>(tri, entityLoc, entityDir, overlapDepth, overlapArea, contactPnt);
            // Plane contact's normal is always the plane's normal
            contact_normal = entityDir;
            return in_contact;
        case deme::ANAL_OBJ_TYPE_CYL_INF:
            in_contact = tri_cyl_penetration<T1, T2>(tri, entityLoc, entityDir, entitySize1, entitySize2, normal_sign,
                                                     contact_normal, overlapDepth, overlapArea, contactPnt);
            return in_contact;
        case deme::ANAL_OBJ_TYPE_PLANAR_CYL: {
            T1 centroid = (A + B + C) / 3.0;
            T1 plane_point;
            float3 plane_normal;
            if (!planar_cyl_plane_from_ref(centroid, entityLoc, entityDir, entitySize1, normal_sign, plane_point,
                                           plane_normal)) {
                return false;
            }
            in_contact =
                tri_plane_penetration<T1, T2>(tri, plane_point, plane_normal, overlapDepth, overlapArea, contactPnt);
            contact_normal = plane_normal;
            return in_contact;
        }
        default:
            return false;
    }
}

// -----------------------------------------------------------------
// Triangle-sphere collision detection utilities
// -----------------------------------------------------------------

/// This utility function takes the location 'P' and snaps it to the closest
/// point on the triangular face with given vertices (A, B, and C). The result
/// is returned in 'res'. Both 'P' and 'res' are assumed to be specified in
/// the same frame as the face vertices. This function returns 'true' if the
/// result is on an edge of this face and 'false' if the result is inside the
/// triangle.
/// Code from Ericson, "real-time collision detection", 2005, pp. 141
template <typename T1 = double3, typename T2 = double>
__device__ __forceinline__ bool snap_to_face(const T1& A, const T1& B, const T1& C, const T1& P, T1& res) {
    const T1 AB = B - A;
    const T1 AC = C - A;

    // Check if P in vertex region outside A
    const T1 AP = P - A;
    const T2 d1 = dot(AB, AP);
    const T2 d2 = dot(AC, AP);
    if (d1 <= (T2)0 && d2 <= (T2)0) {
        res = A;  // barycentric coordinates (1,0,0)
        return true;
    }

    // Check if P in vertex region outside B
    const T1 BP = P - B;
    const T2 d3 = dot(AB, BP);
    const T2 d4 = dot(AC, BP);
    if (d3 >= (T2)0 && d4 <= d3) {
        res = B;  // barycentric coordinates (0,1,0)
        return true;
    }

    // Check if P in edge region of AB
    const T2 vc = d1 * d4 - d3 * d2;
    if (vc <= (T2)0 && d1 >= (T2)0 && d3 <= (T2)0) {
        const T2 v = d1 / (d1 - d3);
        res = A + v * AB;  // barycentric coordinates (1-v,v,0)
        return true;
    }

    // Check if P in vertex region outside C
    const T1 CP = P - C;
    const T2 d5 = dot(AB, CP);
    const T2 d6 = dot(AC, CP);
    if (d6 >= (T2)0 && d5 <= d6) {
        res = C;  // barycentric coordinates (0,0,1)
        return true;
    }

    // Check if P in edge region of AC
    const T2 vb = d5 * d2 - d1 * d6;
    if (vb <= (T2)0 && d2 >= (T2)0 && d6 <= (T2)0) {
        const T2 w = d2 / (d2 - d6);
        res = A + w * AC;  // barycentric coordinates (1-w,0,w)
        return true;
    }

    // Check if P in edge region of BC
    const T2 va = d3 * d6 - d5 * d4;
    if (va <= (T2)0 && (d4 - d3) >= (T2)0 && (d5 - d6) >= (T2)0) {
        const T2 w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        res = B + w * (C - B);  // barycentric coordinates (0,1-w,w)
        return true;
    }

    // P inside face region. Return projection of P onto face
    const T2 inv = (T2)1 / (va + vb + vc);
    const T2 v = vb * inv;
    const T2 w = vc * inv;
    res = A + v * AB + w * AC;
    return false;
}

template <typename T2>
__device__ __forceinline__ T2 clamp01(const T2& x) {
    return x < (T2)0 ? (T2)0 : (x > (T2)1 ? (T2)1 : x);
}


template <typename T2>
__device__ __forceinline__ T2 clampRange(const T2& x, const T2& lo, const T2& hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

__device__ __forceinline__ float fast_atan2_area(float y, float x) {
    const float ax = fabsf(x);
    const float ay = fabsf(y);
    const float mx = fmaxf(ax, ay);
    const float mn = fminf(ax, ay);
    if (mx <= 0.0f) {
        return 0.0f;
    }

    const float a = mn / mx;
    const float s = a * a;
    float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;

    if (ay > ax) r = 1.57079632679f - r;
    if (x < 0.0f) r = 3.14159265359f - r;
    if (y < 0.0f) r = -r;
    return r;
}

__device__ __forceinline__ double fast_atan2_area(double y, double x) {
#ifdef DEME_FAST_APPROX_AREA_DOUBLE
    return (double)fast_atan2_area((float)y, (float)x);
#else
    return atan2(y, x);
#endif
}

template <typename T1, typename T2>
__device__ __forceinline__ T2 signed_sector_area3(const T1& a,
                                                  const T1& b,
                                                  const T1& unit_n,
                                                  const T2 r2) {
    const T2 cr = dot(unit_n, cross(a, b));
    const T2 dt = dot(a, b);
    return (T2)0.5 * r2 * fast_atan2_area(cr, dt);
}

template <typename T1, typename T2>
__device__ __forceinline__ int segment_circle_intersections3(const T1& a,
                                                             const T1& b,
                                                             const T2 r2,
                                                             T2& t0,
                                                             T2& t1) {
    const T1 d = b - a;
    const T2 A = dot(d, d);
    if (A <= (T2)DEME_TINY_FLOAT) {
        return 0;
    }

    const T2 B = (T2)2 * dot(a, d);
    const T2 C = dot(a, a) - r2;
    const T2 D = B * B - (T2)4 * A * C;
    if (D <= (T2)0) {
        return 0;
    }

    const T2 sD = sqrt(D);
    const T2 inv2A = (T2)0.5 / A;

    int n = 0;
    const T2 u0 = (-B - sD) * inv2A;
    const T2 u1 = (-B + sD) * inv2A;

    if (u0 > (T2)0 && u0 < (T2)1) {
        t0 = u0;
        ++n;
    }
    if (u1 > (T2)0 && u1 < (T2)1 && u1 != u0) {
        if (n == 0) {
            t0 = u1;
        } else {
            t1 = u1;
        }
        ++n;
    }

    return n;
}

template <typename T1, typename T2>
__device__ __forceinline__ T2 signed_edge_disk_area3(const T1& a,
                                                     const T1& b,
                                                     const T1& unit_n,
                                                     const T2 r2) {
    T2 ts[4];
    int n = 0;
    ts[n++] = (T2)0;

    T2 t0 = (T2)0;
    T2 t1 = (T2)0;
    const int ni = segment_circle_intersections3<T1, T2>(a, b, r2, t0, t1);
    if (ni >= 1) ts[n++] = t0;
    if (ni >= 2) ts[n++] = t1;
    ts[n++] = (T2)1;

    for (int i = 1; i < n; ++i) {
        const T2 key = ts[i];
        int j = i - 1;
        while (j >= 0 && ts[j] > key) {
            ts[j + 1] = ts[j];
            --j;
        }
        ts[j + 1] = key;
    }

    const T1 d = b - a;
    T2 area = (T2)0;
    for (int i = 0; i < n - 1; ++i) {
        const T2 ta = ts[i];
        const T2 tb = ts[i + 1];
        const T2 tm = (T2)0.5 * (ta + tb);

        const T1 p = a + ta * d;
        const T1 q = a + tb * d;
        const T1 m = a + tm * d;

        if (dot(m, m) <= r2) {
            area += (T2)0.5 * dot(unit_n, cross(p, q));
        } else {
            area += signed_sector_area3<T1, T2>(p, q, unit_n, r2);
        }
    }

    return area;
}

template <typename T1, typename T2>
__device__ __forceinline__ bool point_in_triangle_on_plane(const T1& P,
                                                           const T1& A,
                                                           const T1& B,
                                                           const T1& C,
                                                           const T1& unit_n) {
    const T2 s0 = dot(unit_n, cross(B - A, P - A));
    const T2 s1 = dot(unit_n, cross(C - B, P - B));
    const T2 s2 = dot(unit_n, cross(A - C, P - C));
    return ((s0 >= (T2)0) && (s1 >= (T2)0) && (s2 >= (T2)0)) ||
           ((s0 <= (T2)0) && (s1 <= (T2)0) && (s2 <= (T2)0));
}

template <typename T1, typename T2>
__device__ __forceinline__ T2 point_line_distance_on_plane_sq(const T1& P,
                                                              const T1& A,
                                                              const T1& B,
                                                              const T1& unit_n) {
    const T1 e = B - A;
    const T2 e2 = dot(e, e);
    if (e2 <= (T2)DEME_TINY_FLOAT) {
        const T1 d = P - A;
        return dot(d, d);
    }

    const T2 num = dot(unit_n, cross(e, P - A));
    return (num * num) / e2;
}

template <typename T1, typename T2>
__device__ __forceinline__ T2 additive_circle_triangle_area_fast(const T1& A,
                                                                 const T1& B,
                                                                 const T1& C,
                                                                 const T1& circle_center,
                                                                 const T1& unit_n,
                                                                 const T2 circle_r) {
    const T2 r2 = circle_r * circle_r;

    const T1 a = A - circle_center;
    const T1 b = B - circle_center;
    const T1 c = C - circle_center;

    const T2 da2 = dot(a, a);
    const T2 db2 = dot(b, b);
    const T2 dc2 = dot(c, c);
    const T2 triArea = (T2)0.5 * absT(dot(unit_n, cross(B - A, C - A)));

    // Fast path: whole triangle lies inside the circle.
    if (da2 <= r2 && db2 <= r2 && dc2 <= r2) {
        return triArea;
    }

    // Fast path: whole circle lies inside the triangle.
    if (point_in_triangle_on_plane<T1, T2>(circle_center, A, B, C, unit_n)) {
        const T2 d0_2 = point_line_distance_on_plane_sq<T1, T2>(circle_center, A, B, unit_n);
        const T2 d1_2 = point_line_distance_on_plane_sq<T1, T2>(circle_center, B, C, unit_n);
        const T2 d2_2 = point_line_distance_on_plane_sq<T1, T2>(circle_center, C, A, unit_n);
        const T2 dmin2 = tmin3(d0_2, d1_2, d2_2);
        if (dmin2 >= r2) {
            return (T2)deme::PI * r2;
        }
    }

    T2 area = (T2)0;
    area += signed_edge_disk_area3<T1, T2>(a, b, unit_n, r2);
    area += signed_edge_disk_area3<T1, T2>(b, c, unit_n, r2);
    area += signed_edge_disk_area3<T1, T2>(c, a, unit_n, r2);
    area = absT(area);

    if (area > triArea) {
        area = triArea;
    }
    return area;
}


template <typename T1, typename T2>
__device__ bool checkSphereTriPrismCandidate(const T1& outerA,
                                             const T1& outerB,
                                             const T1& outerC,
                                             const T1& innerA,
                                             const T1& sphere_pos,
                                             const T2 radius,
                                             T2& depth,
                                             T1& closest) {
    const T1 n_out = normalize(cross(outerB - outerA, outerC - outerA));
    const T1 shift = innerA - outerA;
    const T2 thick = fabs(dot(shift, n_out));

    T1 tri_closest;
    snap_to_face<T1, T2>(outerA, outerB, outerC, sphere_pos, tri_closest);

    const T2 z = dot(sphere_pos - outerA, n_out);
    // Candidate slab thickness must not be thinner than the sphere radius on the
    // back side; otherwise once penetration exceeds dynamic margin thickness, kT
    // can drop valid recovery contacts before dT resolves the overlap.
    const T2 backReach = fmax(thick, radius);
    const T2 zc = clampRange<T2>(z, (T2)(-backReach), (T2)0);
    closest = tri_closest + zc * n_out;

    const T1 d = sphere_pos - closest;
    const T2 dist = length(d);
    depth = radius - dist;
    return depth > (T2)0;
}

template <typename T1, typename T2>
__device__ bool checkSphereOuterTriPrismBarrier(const T1& A,
                                                const T1& B,
                                                const T1& C,
                                                const T1& sphere_pos,
                                                const T2 radius,
                                                const T2 shell_half,
                                                T1& normal,
                                                T2& depth,
                                                T2& overlapArea,
                                                T1& pt1) {
    const T1 n_out = normalize(cross(B - A, C - A));
    const T1 outerA = A + shell_half * n_out;
    const T1 outerB = B + shell_half * n_out;
    const T1 outerC = C + shell_half * n_out;
    const T1 inward = ((T2)-2 * shell_half) * n_out;
    const T2 inward_len2 = dot(inward, inward);

    bool have_candidate = false;
    bool best_is_barrier = false;
    T2 best_metric = DEME_HUGE_FLOAT;
    T2 best_depth = (T2)(-DEME_HUGE_FLOAT);
    T1 best_normal = n_out;
    T1 best_pt = outerA;

    // Outer-face barrier: one-sided, no normal flip. If the sphere center is already inside,
    // this remains active and pushes it back out.
    T1 face_loc;
    const bool face_is_edge = snap_to_face<T1, T2>(outerA, outerB, outerC, sphere_pos, face_loc);
    if (!face_is_edge) {
        const T2 h = dot(sphere_pos - outerA, n_out);
        // One-sided outer-face barrier: outside uses geometric depth (radius - h),
        // while inside keeps a saturated full-radius recovery depth.
        const T2 cand_depth = radius - max2(h, (T2)0);
        if (cand_depth > (T2)0) {
            have_candidate = true;
            best_is_barrier = true;
            best_metric = h;
            best_depth = cand_depth;
            best_normal = n_out;
            best_pt = face_loc;
        }
    }

    const T1 verts[3] = {outerA, outerB, outerC};
    for (int i = 0; i < 3; i++) {
        const T1 base = verts[i];
        const T1 next = verts[(i + 1) % 3];
        const T1 edge = next - base;
        const T2 edge_len2 = dot(edge, edge);
        if (edge_len2 <= (T2)DEME_TINY_FLOAT) {
            continue;
        }

        const T1 rel = sphere_pos - base;
        const T2 u_raw = dot(rel, edge) / edge_len2;
        const T2 w_raw = inward_len2 > (T2)DEME_TINY_FLOAT ? dot(rel, inward) / inward_len2 : (T2)0;

        // Side-wall barrier when the orthogonal projection lands inside the rectangle extents.
        if (u_raw >= (T2)0 && u_raw <= (T2)1 && w_raw >= (T2)0 && w_raw <= (T2)1) {
            T1 s_out = normalize(cross(edge, n_out));
            const T2 g = dot(rel, s_out);
            const T2 cand_depth = radius - max2(g, (T2)0);
            if (cand_depth > (T2)0) {
                if (!have_candidate || g < best_metric) {
                    have_candidate = true;
                    best_is_barrier = true;
                    best_metric = g;
                    best_depth = cand_depth;
                    best_normal = s_out;
                    best_pt = base + clampRange<T2>(u_raw, (T2)0, (T2)1) * edge + clampRange<T2>(w_raw, (T2)0, (T2)1) * inward;
                }
            }
        }

        // Rectangle closest-point candidate for edge/vertex grazing contacts.
        const T2 u = clampRange<T2>(u_raw, (T2)0, (T2)1);
        const T2 w = clampRange<T2>(w_raw, (T2)0, (T2)1);
        const T1 rect_q = base + u * edge + w * inward;
        const T1 d = sphere_pos - rect_q;
        const T2 dist = length(d);
        const T2 cand_depth = radius - dist;
        if (cand_depth > (T2)0) {
            const T2 metric = dist;
            if (!have_candidate || (!best_is_barrier && metric < best_metric)) {
                have_candidate = true;
                best_is_barrier = false;
                best_metric = metric;
                best_depth = cand_depth;
                if (dist > (T2)DEME_TINY_FLOAT) {
                    best_normal = ((T2)1 / dist) * d;
                } else {
                    best_normal = normalize(cross(edge, n_out));
                }
                best_pt = rect_q;
            }
        }
    }

    if (!have_candidate) {
        depth = (T2)(-DEME_HUGE_FLOAT);
        overlapArea = (T2)0;
        return false;
    }

    depth = best_depth;
    normal = best_normal;
    const T2 sphere_to_cp = radius - (T2)0.5 * depth;
    pt1 = sphere_pos - sphere_to_cp * normal;

    // Use a monotone geometric area proxy. For very deep recovery states, keep the
    // effective patch area at the hemisphere limit instead of letting it collapse.
    const T2 depth_for_area = depth > radius ? radius : depth;
    const float depth_f = static_cast<float>(depth_for_area);
    const float radius_f = static_cast<float>(radius);
    const float overlap_area_f = static_cast<float>(deme::PI) * (2.0f * radius_f * depth_f - depth_f * depth_f);
    overlapArea = overlap_area_f > 0.0f ? static_cast<T2>(overlap_area_f) : (T2)0;
    return true;
}

template <typename T1, typename T2>
__device__ __forceinline__ T2 closestPtSegmentSegment(const T1& p1,
                                                      const T1& q1,
                                                      const T1& p2,
                                                      const T1& q2,
                                                      T1& c1,
                                                      T1& c2) {
    constexpr T2 EPS = (T2)1e-20;
    const T1 d1 = q1 - p1;
    const T1 d2 = q2 - p2;
    const T1 r = p1 - p2;
    const T2 a = dot(d1, d1);
    const T2 e = dot(d2, d2);
    const T2 f = dot(d2, r);

    T2 s = (T2)0;
    T2 t = (T2)0;

    if (a <= EPS && e <= EPS) {
        c1 = p1;
        c2 = p2;
        return dot(c1 - c2, c1 - c2);
    }
    if (a <= EPS) {
        s = (T2)0;
        t = clamp01<T2>(f / e);
    } else {
        const T2 c = dot(d1, r);
        if (e <= EPS) {
            t = (T2)0;
            s = clamp01<T2>(-c / a);
        } else {
            const T2 b = dot(d1, d2);
            const T2 denom = a * e - b * b;
            if (denom > EPS) {
                s = clamp01<T2>((b * f - c * e) / denom);
            } else {
                s = (T2)0;
            }
            t = (b * s + f) / e;
            if (t < (T2)0) {
                t = (T2)0;
                s = clamp01<T2>(-c / a);
            } else if (t > (T2)1) {
                t = (T2)1;
                s = clamp01<T2>((b - c) / a);
            }
        }
    }

    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
    return dot(c1 - c2, c1 - c2);
}

template <typename T1, typename T2>
__device__ __forceinline__ T2 closestPtTriTriDistance(const T1& A1,
                                                       const T1& B1,
                                                       const T1& C1,
                                                       const T1& A2,
                                                       const T1& B2,
                                                       const T1& C2,
                                                       T1& outA,
                                                       T1& outB) {
    const T1 triA[3] = {A1, B1, C1};
    const T1 triB[3] = {A2, B2, C2};

    T2 best2 = DEME_HUGE_FLOAT;
    outA = A1;
    outB = A2;

    // Edge-edge pairs (9)
    for (int i = 0; i < 3; i++) {
        const T1 pA = triA[i];
        const T1 qA = triA[(i + 1) % 3];
        for (int j = 0; j < 3; j++) {
            const T1 pB = triB[j];
            const T1 qB = triB[(j + 1) % 3];
            T1 cA, cB;
            const T2 d2 = closestPtSegmentSegment<T1, T2>(pA, qA, pB, qB, cA, cB);
            if (d2 < best2) {
                best2 = d2;
                outA = cA;
                outB = cB;
            }
        }
    }

    // Vertices of A against face B
    for (int i = 0; i < 3; i++) {
        T1 q;
        snap_to_face<T1, T2>(A2, B2, C2, triA[i], q);
        const T2 d2 = dot(triA[i] - q, triA[i] - q);
        if (d2 < best2) {
            best2 = d2;
            outA = triA[i];
            outB = q;
        }
    }

    // Vertices of B against face A
    for (int i = 0; i < 3; i++) {
        T1 q;
        snap_to_face<T1, T2>(A1, B1, C1, triB[i], q);
        const T2 d2 = dot(q - triB[i], q - triB[i]);
        if (d2 < best2) {
            best2 = d2;
            outA = q;
            outB = triB[i];
        }
    }

    if (best2 < (T2)0) {
        best2 = (T2)0;
    }
    return sqrt(best2);
}

/**
/brief TRIANGLE FACE--SPHERE COLLISION DETECTION

The triangular face is defined by points A, B, C. The sequence is important as it defines the positive face via a
right-hand rule.
The sphere is centered at sphere_pos and has radius.
The index "1" is associated with the triangle. The index "2" is associated with the sphere.
The coordinates of the face and sphere are assumed to be provided in the same reference frame.

Output:
  - pt1:      contact point on triangle
  - depth:    penetration distance (a positive value means that overlap exists)
  - normal:     contact normal, from pt2 to pt1
A return value of "true" signals collision.
*/
template <typename T1, typename T2>
__device__ __forceinline__ bool checkTriSphereOverlap(const T1& A,           ///< First vertex of the triangle
                                                      const T1& B,           ///< Second vertex of the triangle
                                                      const T1& C,           ///< Third vertex of the triangle
                                                      const T1& sphere_pos,  ///< Location of the center of the sphere
                                                      const T2 radius,       ///< Sphere radius
                                                      T1& normal,            ///< contact normal
                                                      T2& depth,             ///< penetration (positive if in contact)
                                                      T2& overlapArea,       ///< overlap area
                                                      T1& pt1                ///< contact point on triangle
) {
    T1 faceLoc;
    snap_to_face<T1, T2>(A, B, C, sphere_pos, faceLoc);

    const T1 AB = B - A;
    const T1 AC = C - A;
    const T1 N = cross(AB, AC);
    const T2 N2 = dot(N, N);

    T1 face_n = make_one3<T1>(1, 0, 0);
    if (N2 > (T2)DEME_TINY_FLOAT * (T2)DEME_TINY_FLOAT) {
        face_n = ((T2)1 / sqrt(N2)) * N;
    }

    const T1 normal_d = sphere_pos - faceLoc;
    const T2 dist2 = dot(normal_d, normal_d);
    const T2 dist = sqrt(dist2);
    depth = radius - dist;

    if (dist > (T2)DEME_TINY_FLOAT) {
        normal = ((T2)1 / dist) * normal_d;
    } else {
        normal = face_n;
    }

    pt1 = faceLoc - (depth * (T2)0.5) * normal;

    if (depth <= (T2)0) {
        overlapArea = (T2)0;
        return false;
    }

    if (N2 <= (T2)DEME_TINY_FLOAT * (T2)DEME_TINY_FLOAT) {
        overlapArea = (T2)0;
        return true;
    }

    const T2 signed_h = dot(sphere_pos - A, face_n);
    const T2 abs_h = absT(signed_h);
    if (abs_h >= radius) {
        overlapArea = (T2)0;
        return true;
    }

    const T2 circle_r2 = radius * radius - signed_h * signed_h;
    if (circle_r2 <= (T2)0) {
        overlapArea = (T2)0;
        return true;
    }

    const T2 circle_r = sqrt(circle_r2);
    const T1 circle_center = sphere_pos - signed_h * face_n;
    overlapArea = additive_circle_triangle_area_fast<T1, T2>(A, B, C, circle_center, face_n, circle_r);
    return true;
}

/**
/brief TRIANGLE FACE--SPHERE COLLISION DETECTION (DIRECTIONAL)

The triangular face is defined by points A, B, C. The sequence is important as it defines the positive face via a
right-hand rule.
The sphere is centered at sphere_pos and has radius.
The index "1" is associated with the triangle. The index "2" is associated with the sphere.
The coordinates of the face and sphere are assumed to be provided in the same reference frame.
This flavor is directional, meaning if a sphere is not in (geometric) contact with a triangle, it may still be
considered in contact, if it is submerged in the (closed) object that this mesh is representing, or say the penetration
is too deep, large than 2 * sphere rad.

Output:
  - pt1:      contact point on triangle
  - depth:    penetration distance (a positive value means that overlap exists)
  - normal:     contact normal, from pt2 to pt1
A return value of "true" signals collision.
*/
template <typename T1, typename T2>
__device__ bool checkTriSphereOverlap_directional(const T1& A,           ///< First vertex of the triangle
                                                  const T1& B,           ///< Second vertex of the triangle
                                                  const T1& C,           ///< Third vertex of the triangle
                                                  const T1& sphere_pos,  ///< Location of the center of the sphere
                                                  const T2 radius,       ///< Sphere radius
                                                  T1& normal,            ///< contact normal
                                                  T2& depth,             ///< penetration (positive if in contact)
                                                  T1& pt1                ///< contact point on triangle
) {
    // Directional law:
    // - Face witness: oriented one-sided barrier depth (radius - signed height), normal fixed to face normal.
    // - Edge/vertex witness: geometric closest-point depth, but prefer oriented face support when active.
    // This maintains outward recovery when the center drifts behind the local face.
    const T1 face_n = normalize(cross(B - A, C - A));

    T1 faceLoc;
    const bool on_edge = snap_to_face<T1, T2>(A, B, C, sphere_pos, faceLoc);

    const T2 h = dot(sphere_pos - faceLoc, face_n);
    const T1 normal_d = sphere_pos - faceLoc;
    const T2 dist = length(normal_d);
    const T2 depth_geom = radius - dist;
    const T2 depth_face = radius - h;

    if (!on_edge) {
        depth = depth_face;
        normal = face_n;
    } else {
        // Edge/vertex witness:
        // - Outside the oriented face plane (h >= 0): use geometric closest-point contact.
        // - Behind the face plane (h < 0): switch to one-sided face recovery to avoid inward normals.
        // Using `depth_face > 0` here is too permissive (true for almost all contact states) and
        // can override edge geometry, creating seam-support discontinuities.
        if (h < (T2)0) {
            depth = depth_face;
            normal = face_n;
        } else {
            depth = depth_geom;
            if (dist > (T2)DEME_TINY_FLOAT) {
                normal = ((T2)1 / dist) * normal_d;
            } else {
                // Deterministic fallback when center is exactly on the witness point.
                normal = face_n;
            }
            // Keep edge/vertex normals in the outward hemisphere to avoid repulsive force inversion.
            if (dot(normal, face_n) < (T2)0) {
                normal = -normal;
            }
        }
    }

    pt1 = faceLoc - (depth * (T2)0.5) * normal;
    return depth > (T2)0;
}

// -----------------------------------------------------------------------------
// Triangle-triangle collision detection utilities
// -----------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
// Prism contact detection using the Separating Axis Theorem (SAT)
//
// For the extruded triangle "sandwich" prisms we only have 4 unique edge
// directions (3 base edges + extrusion). This yields:
// - 8 face normals (base + 3 side faces per prism)
// - 16 edge-edge axes
// Total: 24 axes, evaluated on the fly without normalization.
////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ float invSqrt(float x) {
    return rsqrtf(x);
}

__device__ __forceinline__ double invSqrt(double x) {
    return 1.0 / sqrt(x);
}

#ifndef DEME_SAT_ENABLE_MIXED_PRECISION
    #define DEME_SAT_ENABLE_MIXED_PRECISION 0
#endif

// minimal arithmetic for extruded-triangle prism projection.
// Key identity: prism projection = [minTri, maxTri] union [minTri+shift, maxTri+shift]
// => outMin = minTri + min(0,shift), outMax = maxTri + max(0,shift)
template <typename Vec, typename Scalar>
__device__ __forceinline__ void projectExtrudedTriPrism(const Vec& v0,
                                                        const Vec& v1,
                                                        const Vec& v2,
                                                        const Vec& d,
                                                        const Vec& axis,
                                                        Scalar& outMin,
                                                        Scalar& outMax) {
    const Scalar p0 = dot(v0, axis);
    const Scalar p1 = dot(v1, axis);
    const Scalar p2 = dot(v2, axis);

    const Scalar triMin = fmin(p0, fmin(p1, p2));
    const Scalar triMax = fmax(p0, fmax(p1, p2));

    const Scalar shift = dot(d, axis);
    const Scalar z = Scalar(0);

    outMin = triMin + fmin(z, shift);
    outMax = triMax + fmax(z, shift);
}

template <typename Vec, typename Scalar>
__device__ __forceinline__ Scalar satSeparationOnAxis(const Vec& axis,
                                                      const Vec& A0,
                                                      const Vec& A1,
                                                      const Vec& A2,
                                                      const Vec& dA,
                                                      const Vec& B0,
                                                      const Vec& B1,
                                                      const Vec& B2,
                                                      const Vec& dB) {
    Scalar len2 = dot(axis, axis);
    if (len2 < Scalar(DEME_TINY_FLOAT))
        return -Scalar(DEME_HUGE_FLOAT);

    Scalar minA, maxA, minB, maxB;
    projectExtrudedTriPrism<Vec, Scalar>(A0, A1, A2, dA, axis, minA, maxA);
    projectExtrudedTriPrism<Vec, Scalar>(B0, B1, B2, dB, axis, minB, maxB);

    Scalar sep1 = minB - maxA;
    Scalar sep2 = minA - maxB;
    Scalar sepProj = (sep1 > sep2) ? sep1 : sep2;
    Scalar invLen = invSqrt(len2);
    return sepProj * invLen;
}

/**
 * @brief Fast SAT contact check between two triangular prisms (triangle sandwiches).
 *
 * Evaluates 24 axes (8 face normals + 16 edge-edge) without normalization. Uses FP32 by
 * default with a narrow mixed-precision recheck near zero overlap to avoid false positives.
 * 
 * OPTIMIZED VERSION: 
 * - Uses fused operations to reduce register pressure
 * - Inline axis separation test to avoid lambda overhead
 * - Early termination structure optimized for GPU SIMT execution
 *
 * @return true if prisms are in contact (no separating axis found), false otherwise
 */
template <typename T1>
__device__ __forceinline__ bool calc_prism_contact(const T1& prismAFaceANode1,
                                                   const T1& prismAFaceANode2,
                                                   const T1& prismAFaceANode3,
                                                   const T1& prismAFaceBNode1,
                                                   const T1& prismAFaceBNode2,
                                                   const T1& prismAFaceBNode3,
                                                   const T1& prismBFaceANode1,
                                                   const T1& prismBFaceANode2,
                                                   const T1& prismBFaceANode3,
                                                   const T1& prismBFaceBNode1,
                                                   const T1& prismBFaceBNode2,
                                                   const T1& prismBFaceBNode3) {
    // Relative coordinates centered at prismAFaceANode1 reduce FP32 dynamic range issues.
    const float3 origin = prismAFaceANode1;

    // Prism A base triangle relative to origin (A0 is exactly zero)
    const float3 A0 = make_float3(0.0f, 0.0f, 0.0f);
    const float3 A1 = prismAFaceANode2 - origin;
    const float3 A2 = prismAFaceANode3 - origin;

    // Prism B base triangle relative to origin
    const float3 B0 = prismBFaceANode1 - origin;
    const float3 B1 = prismBFaceANode2 - origin;
    const float3 B2 = prismBFaceANode3 - origin;

    // Extrusion vectors (world differences; origin cancels)
    const float3 dA = prismAFaceBNode1 - prismAFaceANode1;
    const float3 dB = prismBFaceBNode1 - prismBFaceANode1;

    // ------------------------------------------------------------------
    // Cheap AABB overlap test on world axes (X/Y/Z) for the full prisms.
    // This rejects many non-contacts before the expensive SAT axes.
    // Prism extents: triMin + min(0,d), triMax + max(0,d).
    // ------------------------------------------------------------------
    {
        const float triMinAx = fminf(0.0f, fminf(A1.x, A2.x));
        const float triMaxAx = fmaxf(0.0f, fmaxf(A1.x, A2.x));
        const float triMinAy = fminf(0.0f, fminf(A1.y, A2.y));
        const float triMaxAy = fmaxf(0.0f, fmaxf(A1.y, A2.y));
        const float triMinAz = fminf(0.0f, fminf(A1.z, A2.z));
        const float triMaxAz = fmaxf(0.0f, fmaxf(A1.z, A2.z));

        const float minAx = triMinAx + fminf(0.0f, dA.x);
        const float maxAx = triMaxAx + fmaxf(0.0f, dA.x);
        const float minAy = triMinAy + fminf(0.0f, dA.y);
        const float maxAy = triMaxAy + fmaxf(0.0f, dA.y);
        const float minAz = triMinAz + fminf(0.0f, dA.z);
        const float maxAz = triMaxAz + fmaxf(0.0f, dA.z);

        const float triMinBx = fminf(B0.x, fminf(B1.x, B2.x));
        const float triMaxBx = fmaxf(B0.x, fmaxf(B1.x, B2.x));
        const float triMinBy = fminf(B0.y, fminf(B1.y, B2.y));
        const float triMaxBy = fmaxf(B0.y, fmaxf(B1.y, B2.y));
        const float triMinBz = fminf(B0.z, fminf(B1.z, B2.z));
        const float triMaxBz = fmaxf(B0.z, fmaxf(B1.z, B2.z));

        const float minBx = triMinBx + fminf(0.0f, dB.x);
        const float maxBx = triMaxBx + fmaxf(0.0f, dB.x);
        const float minBy = triMinBy + fminf(0.0f, dB.y);
        const float maxBy = triMaxBy + fmaxf(0.0f, dB.y);
        const float minBz = triMinBz + fminf(0.0f, dB.z);
        const float maxBz = triMaxBz + fmaxf(0.0f, dB.z);

        if (maxAx < minBx || maxBx < minAx) return false;
        if (maxAy < minBy || maxBy < minAy) return false;
        if (maxAz < minBz || maxBz < minAz) return false;
    }

    // Edge vectors (triangle edges only; extrusion edges handled via dA/dB)
    const float3 eA0 = A1 - A0;
    const float3 eA1 = A2 - A1;
    const float3 eA2 = A0 - A2;

    const float3 eB0 = B1 - B0;
    const float3 eB1 = B2 - B1;
    const float3 eB2 = B0 - B2;

    // Project an extruded triangle prism where the shift along axis is provided (can be forced to 0 cheaply).
    // NOTE: We avoid any rsqrt normalization. Separation sign is invariant to axis scale.
    #define PROJECT_PRISM_WITH_SHIFT(v0, v1, v2, shift, axis, outMin, outMax) do { \
        const float p0 = dot((v0), (axis)); \
        const float p1 = dot((v1), (axis)); \
        const float p2 = dot((v2), (axis)); \
        const float triMin = fminf(p0, fminf(p1, p2)); \
        const float triMax = fmaxf(p0, fmaxf(p1, p2)); \
        (outMin) = triMin + fminf(0.0f, (shift)); \
        (outMax) = triMax + fmaxf(0.0f, (shift)); \
    } while (0)

    // Test axis: caller can declare whether shiftA and/or shiftB are guaranteed zero for this axis.
    // This saves a dot(d,axis) on the relevant prism.
    #define TEST_AXIS(axis_expr, shiftA_zero, shiftB_zero) do { \
        const float3 axis = (axis_expr); \
        const float len2 = dot(axis, axis); \
        if (len2 > DEME_TINY_FLOAT) { \
            float minA, maxA, minB, maxB; \
            const float shiftA = (shiftA_zero) ? 0.0f : dot(dA, axis); \
            const float shiftB = (shiftB_zero) ? 0.0f : dot(dB, axis); \
            PROJECT_PRISM_WITH_SHIFT(A0, A1, A2, shiftA, axis, minA, maxA); \
            PROJECT_PRISM_WITH_SHIFT(B0, B1, B2, shiftB, axis, minB, maxB); \
            if (maxA < minB || maxB < minA) return false; \
        } \
    } while (0)

    // Face normals (2 axes): shifts generally non-zero for both prisms
    TEST_AXIS(cross(eA0, A2 - A0), false, false);
    TEST_AXIS(cross(eB0, B2 - B0), false, false);

    // Side normals (6 axes):
    // For axis = cross(eA?, dA): dot(dA, axis) == 0 in exact arithmetic -> shiftA_zero = true.
    // For axis = cross(eB?, dB): dot(dB, axis) == 0 -> shiftB_zero = true.
    TEST_AXIS(cross(eA0, dA), true,  false);
    TEST_AXIS(cross(eA1, dA), true,  false);
    TEST_AXIS(cross(eA2, dA), true,  false);
    TEST_AXIS(cross(eB0, dB), false, true);
    TEST_AXIS(cross(eB1, dB), false, true);
    TEST_AXIS(cross(eB2, dB), false, true);

    // Edge-edge axes (9 axes): shifts generally non-zero for both prisms
    TEST_AXIS(cross(eA0, eB0), false, false);
    TEST_AXIS(cross(eA0, eB1), false, false);
    TEST_AXIS(cross(eA0, eB2), false, false);
    TEST_AXIS(cross(eA1, eB0), false, false);
    TEST_AXIS(cross(eA1, eB1), false, false);
    TEST_AXIS(cross(eA1, eB2), false, false);
    TEST_AXIS(cross(eA2, eB0), false, false);
    TEST_AXIS(cross(eA2, eB1), false, false);
    TEST_AXIS(cross(eA2, eB2), false, false);

    // Edge-extrusion cross products (6 axes):
    // axis = cross(eA?, dB) => dot(dB,axis) == 0 -> shiftB_zero = true
    // axis = cross(dA, eB?) => dot(dA,axis) == 0 -> shiftA_zero = true
    TEST_AXIS(cross(eA0, dB), false, true);
    TEST_AXIS(cross(eA1, dB), false, true);
    TEST_AXIS(cross(eA2, dB), false, true);
    TEST_AXIS(cross(dA, eB0), true,  false);
    TEST_AXIS(cross(dA, eB1), true,  false);
    TEST_AXIS(cross(dA, eB2), true,  false);

    // Extrusion-extrusion (1 axis): dot(dA,cross(dA,dB)) == dot(dB,cross(dA,dB)) == 0
    TEST_AXIS(cross(dA, dB), true, true);

    #undef TEST_AXIS
    #undef PROJECT_PRISM_WITH_SHIFT

    return true;
}

/// Lightweight SAT check for triangle-triangle contact (physical contact only)
/// Returns true if triangles are in physical contact (no separating axis found), false otherwise
/// This is a simplified version that only performs the SAT test without computing contact details

// axis separation test (no normalization)
template <typename T1, typename T2>
__device__ __forceinline__ bool axis_separates_skin(
    const T1& axis,
    const T1& a0, const T1& a1, const T1& a2,
    const T1& b0, const T1& b1, const T1& b2,
    const T2 skin,
    const T2 tiny_axis2,
    const T2 num_eps) {

    const T2 len2 = (T2)dot(axis, axis);
    if (len2 <= tiny_axis2) return false; // ignore degenerate axis

    const T2 pa0 = (T2)dot(a0, axis);
    const T2 pa1 = (T2)dot(a1, axis);
    const T2 pa2 = (T2)dot(a2, axis);
    const T2 minA = tmin3(pa0, pa1, pa2);
    const T2 maxA = tmax3(pa0, pa1, pa2);

    const T2 pb0 = (T2)dot(b0, axis);
    const T2 pb1 = (T2)dot(b1, axis);
    const T2 pb2 = (T2)dot(b2, axis);
    const T2 minB = tmin3(pb0, pb1, pb2);
    const T2 maxB = tmax3(pb0, pb1, pb2);

    const T2 sep1 = minB - maxA;
    const T2 sep2 = minA - maxB;
    const T2 sep  = (sep1 > sep2) ? sep1 : sep2;

    // Separation only if gap is strictly larger than skin (+ numeric cushion)
    return sep > (skin + num_eps);
}

template <typename T1, typename T2>
__device__ __forceinline__ bool checkTriangleTriangleSAT(
    const T1& A1, const T1& B1, const T1& C1,
    const T1& A2, const T1& B2, const T1& C2) {

    // Contact skin in your length unit (e.g., mm)
    constexpr T2 CONTACT_SKIN = (T2)0.05;  // adjust

    // Degeneracy gate for axes (len^2). Keep very small.
    //// TODO: Beyond this threshol the presence of degenerated tris should be warned!
    constexpr T2 TINY_AXIS2   = (T2)1e-20;

    // Small numerical cushion (should be << skin)
    constexpr T2 NUM_EPS      = (T2)1e-12;

    // ---------------------------------------------------------
    // 1) Pair-local frame: shift by origin to shrink dynamic range
    // ---------------------------------------------------------
    const T1 O  = A1;
    const T1 a0 = make_zero3<T1>();
    const T1 a1 = B1 - O;
    const T1 a2 = C1 - O;

    const T1 b0 = A2 - O;
    const T1 b1 = B2 - O;
    const T1 b2 = C2 - O;

    // ---------------------------------------------------------
    // 2) AABB early-out on X/Y/Z with skin
    // ---------------------------------------------------------
    {
        const T2 minAx = tmin3((T2)a0.x, (T2)a1.x, (T2)a2.x);
        const T2 maxAx = tmax3((T2)a0.x, (T2)a1.x, (T2)a2.x);
        const T2 minBx = tmin3((T2)b0.x, (T2)b1.x, (T2)b2.x);
        const T2 maxBx = tmax3((T2)b0.x, (T2)b1.x, (T2)b2.x);
        if (maxAx < (minBx - CONTACT_SKIN) || maxBx < (minAx - CONTACT_SKIN)) return false;

        const T2 minAy = tmin3((T2)a0.y, (T2)a1.y, (T2)a2.y);
        const T2 maxAy = tmax3((T2)a0.y, (T2)a1.y, (T2)a2.y);
        const T2 minBy = tmin3((T2)b0.y, (T2)b1.y, (T2)b2.y);
        const T2 maxBy = tmax3((T2)b0.y, (T2)b1.y, (T2)b2.y);
        if (maxAy < (minBy - CONTACT_SKIN) || maxBy < (minAy - CONTACT_SKIN)) return false;

        const T2 minAz = tmin3((T2)a0.z, (T2)a1.z, (T2)a2.z);
        const T2 maxAz = tmax3((T2)a0.z, (T2)a1.z, (T2)a2.z);
        const T2 minBz = tmin3((T2)b0.z, (T2)b1.z, (T2)b2.z);
        const T2 maxBz = tmax3((T2)b0.z, (T2)b1.z, (T2)b2.z);
        if (maxAz < (minBz - CONTACT_SKIN) || maxBz < (minAz - CONTACT_SKIN)) return false;
    }

    // ---------------------------------------------------------
    // 3) SAT axes: 2 face normals + 9 edge×edge (no normalization)
    // ---------------------------------------------------------
    const T1 eA0 = a1 - a0;
    const T1 eA1 = a2 - a1;
    const T1 eA2 = a0 - a2;

    const T1 eB0 = b1 - b0;
    const T1 eB1 = b2 - b1;
    const T1 eB2 = b0 - b2;

    // Face normals (unnormalized)
    const T1 nA = cross(eA0, a2 - a0);
    if (axis_separates_skin<T1, T2>(nA, a0,a1,a2, b0,b1,b2, CONTACT_SKIN, TINY_AXIS2, NUM_EPS)) return false;

    const T1 nB = cross(eB0, b2 - b0);
    if (axis_separates_skin<T1, T2>(nB, a0,a1,a2, b0,b1,b2, CONTACT_SKIN, TINY_AXIS2, NUM_EPS)) return false;

    // Edge×Edge (9)
    if (axis_separates_skin<T1, T2>(cross(eA0,eB0), a0,a1,a2, b0,b1,b2, CONTACT_SKIN, TINY_AXIS2, NUM_EPS)) return false;
    if (axis_separates_skin<T1, T2>(cross(eA0,eB1), a0,a1,a2, b0,b1,b2, CONTACT_SKIN, TINY_AXIS2, NUM_EPS)) return false;
    if (axis_separates_skin<T1, T2>(cross(eA0,eB2), a0,a1,a2, b0,b1,b2, CONTACT_SKIN, TINY_AXIS2, NUM_EPS)) return false;

    if (axis_separates_skin<T1, T2>(cross(eA1,eB0), a0,a1,a2, b0,b1,b2, CONTACT_SKIN, TINY_AXIS2, NUM_EPS)) return false;
    if (axis_separates_skin<T1, T2>(cross(eA1,eB1), a0,a1,a2, b0,b1,b2, CONTACT_SKIN, TINY_AXIS2, NUM_EPS)) return false;
    if (axis_separates_skin<T1, T2>(cross(eA1,eB2), a0,a1,a2, b0,b1,b2, CONTACT_SKIN, TINY_AXIS2, NUM_EPS)) return false;

    if (axis_separates_skin<T1, T2>(cross(eA2,eB0), a0,a1,a2, b0,b1,b2, CONTACT_SKIN, TINY_AXIS2, NUM_EPS)) return false;
    if (axis_separates_skin<T1, T2>(cross(eA2,eB1), a0,a1,a2, b0,b1,b2, CONTACT_SKIN, TINY_AXIS2, NUM_EPS)) return false;
    if (axis_separates_skin<T1, T2>(cross(eA2,eB2), a0,a1,a2, b0,b1,b2, CONTACT_SKIN, TINY_AXIS2, NUM_EPS)) return false;

    return true; // no axis with gap > skin => treat as contact candidate
}


// ------------------------------------------------------------------
// Start of the final tri-tri overlap calculation functions
// ------------------------------------------------------------------
template <typename V, typename S>
__device__ __forceinline__ S local_length_scale6(const V& A0, const V& A1, const V& A2,
                                                 const V& B0, const V& B1, const V& B2) {
    const S minx = min2((S)A0.x, min2((S)A1.x, min2((S)A2.x, min2((S)B0.x, min2((S)B1.x, (S)B2.x)))));
    const S miny = min2((S)A0.y, min2((S)A1.y, min2((S)A2.y, min2((S)B0.y, min2((S)B1.y, (S)B2.y)))));
    const S minz = min2((S)A0.z, min2((S)A1.z, min2((S)A2.z, min2((S)B0.z, min2((S)B1.z, (S)B2.z)))));
    const S maxx = max2((S)A0.x, max2((S)A1.x, max2((S)A2.x, max2((S)B0.x, max2((S)B1.x, (S)B2.x)))));
    const S maxy = max2((S)A0.y, max2((S)A1.y, max2((S)A2.y, max2((S)B0.y, max2((S)B1.y, (S)B2.y)))));
    const S maxz = max2((S)A0.z, max2((S)A1.z, max2((S)A2.z, max2((S)B0.z, max2((S)B1.z, (S)B2.z)))));

    const S span = max2((S)0, max2(maxx - minx, max2(maxy - miny, maxz - minz)));

    const V eA0 = A1 - A0; const V eA1 = A2 - A1; const V eA2 = A0 - A2;
    const V eB0 = B1 - B0; const V eB1 = B2 - B1; const V eB2 = B0 - B2;

    const S maxEdge2 = max2((S)0,
        max2((S)dot(eA0, eA0), max2((S)dot(eA1, eA1), max2((S)dot(eA2, eA2),
        max2((S)dot(eB0, eB0), max2((S)dot(eB1, eB1), (S)dot(eB2, eB2)))))));

    return max2((S)1, max2(span, (S)sqrt(maxEdge2)));
}

template <typename S>
__device__ __forceinline__ S rel_len_tol() {
    return sizeof(S) == sizeof(double) ? (S)1.4210854715202004e-14 : (S)7.62939453125e-6;
}

template <typename S>
__device__ __forceinline__ S rel_ang_tol() {
    return sizeof(S) == sizeof(double) ? (S)1.4210854715202004e-14 : (S)7.62939453125e-6;
}

template <typename V, typename S>
__device__ __forceinline__ bool aabb_overlap6(const V& A0, const V& A1, const V& A2,
                                              const V& B0, const V& B1, const V& B2,
                                              const S eps) {
    const S aminx = min2((S)A0.x, min2((S)A1.x, (S)A2.x));
    const S aminy = min2((S)A0.y, min2((S)A1.y, (S)A2.y));
    const S aminz = min2((S)A0.z, min2((S)A1.z, (S)A2.z));
    const S amaxx = max2((S)A0.x, max2((S)A1.x, (S)A2.x));
    const S amaxy = max2((S)A0.y, max2((S)A1.y, (S)A2.y));
    const S amaxz = max2((S)A0.z, max2((S)A1.z, (S)A2.z));

    const S bminx = min2((S)B0.x, min2((S)B1.x, (S)B2.x));
    const S bminy = min2((S)B0.y, min2((S)B1.y, (S)B2.y));
    const S bminz = min2((S)B0.z, min2((S)B1.z, (S)B2.z));
    const S bmaxx = max2((S)B0.x, max2((S)B1.x, (S)B2.x));
    const S bmaxy = max2((S)B0.y, max2((S)B1.y, (S)B2.y));
    const S bmaxz = max2((S)B0.z, max2((S)B1.z, (S)B2.z));

    if (amaxx < bminx - eps || bmaxx < aminx - eps) return false;
    if (amaxy < bminy - eps || bmaxy < aminy - eps) return false;
    if (amaxz < bminz - eps || bmaxz < aminz - eps) return false;
    return true;
}

template <typename V, typename S>
__device__ __forceinline__ int dominant_axis(const V& v) {
    const S ax = absT((S)v.x), ay = absT((S)v.y), az = absT((S)v.z);
    if (ax >= ay && ax >= az) return 0;
    if (ay >= az) return 1;
    return 2;
}

template <typename V, typename S>
__device__ __forceinline__ S coord_axis(const V& p, int axis) {
    return axis == 0 ? (S)p.x : (axis == 1 ? (S)p.y : (S)p.z);
}

template <typename S>
__device__ __forceinline__ S orient2d(S ax, S ay, S bx, S by, S cx, S cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

template <typename S>
__device__ __forceinline__ bool on_segment_2d(S ax, S ay, S bx, S by, S px, S py,
                                              S orientTol, S coordTol) {
    if (absT(orient2d(ax, ay, bx, by, px, py)) > orientTol) return false;
    const S minx = min2(ax, bx) - coordTol, maxx = max2(ax, bx) + coordTol;
    const S miny = min2(ay, by) - coordTol, maxy = max2(ay, by) + coordTol;
    return (px >= minx && px <= maxx && py >= miny && py <= maxy);
}

template <typename S>
__device__ __forceinline__ bool seg_seg_2d(S ax, S ay, S bx, S by,
                                           S cx, S cy, S dx, S dy,
                                           S orientTol, S coordTol) {
    const S o1 = orient2d(ax, ay, bx, by, cx, cy);
    const S o2 = orient2d(ax, ay, bx, by, dx, dy);
    const S o3 = orient2d(cx, cy, dx, dy, ax, ay);
    const S o4 = orient2d(cx, cy, dx, dy, bx, by);

    const bool straddle1 = (o1 > orientTol && o2 < -orientTol) || (o1 < -orientTol && o2 > orientTol);
    const bool straddle2 = (o3 > orientTol && o4 < -orientTol) || (o3 < -orientTol && o4 > orientTol);
    if (straddle1 && straddle2) return true;

    if (on_segment_2d(ax, ay, bx, by, cx, cy, orientTol, coordTol)) return true;
    if (on_segment_2d(ax, ay, bx, by, dx, dy, orientTol, coordTol)) return true;
    if (on_segment_2d(cx, cy, dx, dy, ax, ay, orientTol, coordTol)) return true;
    if (on_segment_2d(cx, cy, dx, dy, bx, by, orientTol, coordTol)) return true;
    return false;
}

template <typename S>
__device__ __forceinline__ bool point_in_tri_2d(S px, S py,
                                                S ax, S ay,
                                                S bx, S by,
                                                S cx, S cy,
                                                S orientTol) {
    const S o0 = orient2d(ax, ay, bx, by, px, py);
    const S o1 = orient2d(bx, by, cx, cy, px, py);
    const S o2 = orient2d(cx, cy, ax, ay, px, py);
    const bool hasNeg = (o0 < -orientTol) || (o1 < -orientTol) || (o2 < -orientTol);
    const bool hasPos = (o0 > orientTol) || (o1 > orientTol) || (o2 > orientTol);
    return !(hasNeg && hasPos);
}

template <typename V, typename S>
__device__ __forceinline__ bool coplanar_tri_tri(const V& N,
                                                 const V& A0, const V& A1, const V& A2,
                                                 const V& B0, const V& B1, const V& B2,
                                                 S orientTol,
                                                 S coordTol) {
    const S nx = absT((S)N.x), ny = absT((S)N.y), nz = absT((S)N.z);
    int i0, i1;
    if (nx > ny) {
        if (nx > nz) { i0 = 1; i1 = 2; }
        else         { i0 = 0; i1 = 1; }
    } else {
        if (ny > nz) { i0 = 0; i1 = 2; }
        else         { i0 = 0; i1 = 1; }
    }

    const S a0x = coord_axis<V,S>(A0, i0), a0y = coord_axis<V,S>(A0, i1);
    const S a1x = coord_axis<V,S>(A1, i0), a1y = coord_axis<V,S>(A1, i1);
    const S a2x = coord_axis<V,S>(A2, i0), a2y = coord_axis<V,S>(A2, i1);
    const S b0x = coord_axis<V,S>(B0, i0), b0y = coord_axis<V,S>(B0, i1);
    const S b1x = coord_axis<V,S>(B1, i0), b1y = coord_axis<V,S>(B1, i1);
    const S b2x = coord_axis<V,S>(B2, i0), b2y = coord_axis<V,S>(B2, i1);

    if (seg_seg_2d(a0x,a0y,a1x,a1y,b0x,b0y,b1x,b1y,orientTol,coordTol)) return true;
    if (seg_seg_2d(a0x,a0y,a1x,a1y,b1x,b1y,b2x,b2y,orientTol,coordTol)) return true;
    if (seg_seg_2d(a0x,a0y,a1x,a1y,b2x,b2y,b0x,b0y,orientTol,coordTol)) return true;
    if (seg_seg_2d(a1x,a1y,a2x,a2y,b0x,b0y,b1x,b1y,orientTol,coordTol)) return true;
    if (seg_seg_2d(a1x,a1y,a2x,a2y,b1x,b1y,b2x,b2y,orientTol,coordTol)) return true;
    if (seg_seg_2d(a1x,a1y,a2x,a2y,b2x,b2y,b0x,b0y,orientTol,coordTol)) return true;
    if (seg_seg_2d(a2x,a2y,a0x,a0y,b0x,b0y,b1x,b1y,orientTol,coordTol)) return true;
    if (seg_seg_2d(a2x,a2y,a0x,a0y,b1x,b1y,b2x,b2y,orientTol,coordTol)) return true;
    if (seg_seg_2d(a2x,a2y,a0x,a0y,b2x,b2y,b0x,b0y,orientTol,coordTol)) return true;

    if (point_in_tri_2d(a0x,a0y,b0x,b0y,b1x,b1y,b2x,b2y,orientTol)) return true;
    if (point_in_tri_2d(b0x,b0y,a0x,a0y,a1x,a1y,a2x,a2y,orientTol)) return true;
    return false;
}

template <typename S>
__device__ __forceinline__ int interval_from_plane_hits(S p0, S p1, S p2,
                                                        S d0, S d1, S d2,
                                                        S hits[6], S tol) {
    int n = 0;
    if (absT(d0) <= tol) hits[n++] = p0;
    if (absT(d1) <= tol) {
        bool dup = false;
        for (int i = 0; i < n; ++i) dup = dup || (absT(hits[i] - p1) <= tol);
        if (!dup) hits[n++] = p1;
    }
    if (absT(d2) <= tol) {
        bool dup = false;
        for (int i = 0; i < n; ++i) dup = dup || (absT(hits[i] - p2) <= tol);
        if (!dup) hits[n++] = p2;
    }
    if ((d0 > tol && d1 < -tol) || (d0 < -tol && d1 > tol)) hits[n++] = p0 + (p1 - p0) * (d0 / (d0 - d1));
    if ((d1 > tol && d2 < -tol) || (d1 < -tol && d2 > tol)) hits[n++] = p1 + (p2 - p1) * (d1 / (d1 - d2));
    if ((d2 > tol && d0 < -tol) || (d2 < -tol && d0 > tol)) hits[n++] = p2 + (p0 - p2) * (d2 / (d2 - d0));
    return n;
}

template <typename T1, typename T2>
__device__ __forceinline__ T1 make_vec3(T2 x, T2 y, T2 z) {
    T1 out; out.x = x; out.y = y; out.z = z; return out;
}

template <typename T1, typename T2>
__device__ __forceinline__ void build_plane_basis_from_normal(const T1& n, T1& u, T1& v) {
    const T1 a = (absT((T2)n.x) > (T2)0.70710678) ? make_vec3<T1,T2>((T2)0, (T2)1, (T2)0)
                                                  : make_vec3<T1,T2>((T2)1, (T2)0, (T2)0);
    u = normalize(cross(a, n));
    v = cross(n, u);
}

template <typename T2>
__device__ __forceinline__ T2 cross2(T2 ax, T2 ay, T2 bx, T2 by) {
    return ax * by - ay * bx;
}

template <typename T1, typename T2>
__device__ __forceinline__ bool same_point3(const T1& a, const T1& b, T2 eps2) {
    const T1 d = a - b;
    return dot(d, d) <= eps2;
}

template <typename T1, typename T2>
__device__ __forceinline__ int cleanup_poly3d(T1 pts[4], int n, T2 eps) {
    if (n <= 1) return n;
    const T2 eps2 = eps * eps;
    T1 tmp[4];
    int m = 0;
    for (int i = 0; i < n; ++i) {
        if (m == 0 || !same_point3<T1,T2>(pts[i], tmp[m - 1], eps2)) {
            tmp[m++] = pts[i];
        }
    }
    if (m > 1 && same_point3<T1,T2>(tmp[0], tmp[m - 1], eps2)) --m;
    for (int i = 0; i < m; ++i) pts[i] = tmp[i];
    return m;
}

template <typename T1, typename T2>
__device__ __forceinline__ T1 interp_plane_hit(const T1& S0, const T1& S1, T2 d0, T2 d1) {
    const T2 t = d0 / (d0 - d1);
    return S0 + (S1 - S0) * t;
}

template <typename T1, typename T2>
__device__ __forceinline__ int clip_triangle_by_plane_keep_negative(const T1& T0, const T1& T1v, const T1& T2v,
                                                                    const T1& planeP, const T1& planeN,
                                                                    T2 eps,
                                                                    T1 outPoly[4]) {
    T1 inPoly[4];
    inPoly[0] = T0; inPoly[1] = T1v; inPoly[2] = T2v;
    int inCount = 3;
    int outCount = 0;

    for (int i = 0; i < inCount; ++i) {
        const T1 S0 = inPoly[(i + inCount - 1) % inCount];
        const T1 S1 = inPoly[i];
        const T2 d0 = dot(planeN, S0 - planeP);
        const T2 d1 = dot(planeN, S1 - planeP);
        const bool in0 = d0 <= eps;
        const bool in1 = d1 <= eps;

        if (in0 && in1) {
            outPoly[outCount++] = S1;
        } else if (in0 && !in1) {
            outPoly[outCount++] = interp_plane_hit<T1,T2>(S0, S1, d0, d1);
        } else if (!in0 && in1) {
            outPoly[outCount++] = interp_plane_hit<T1,T2>(S0, S1, d0, d1);
            outPoly[outCount++] = S1;
        }
    }

    return cleanup_poly3d<T1,T2>(outPoly, outCount, eps);
}

template <typename T2>
__device__ __forceinline__ bool same_point2(T2 ax, T2 ay, T2 bx, T2 by, T2 eps2) {
    const T2 dx = ax - bx;
    const T2 dy = ay - by;
    return dx * dx + dy * dy <= eps2;
}

template <typename T2>
__device__ __forceinline__ int cleanup_poly2d(T2 px[8], T2 py[8], int n, T2 eps) {
    if (n <= 1) return n;
    const T2 eps2 = eps * eps;
    T2 qx[8], qy[8];
    int m = 0;
    for (int i = 0; i < n; ++i) {
        if (m == 0 || !same_point2<T2>(px[i], py[i], qx[m - 1], qy[m - 1], eps2)) {
            qx[m] = px[i];
            qy[m] = py[i];
            ++m;
        }
    }
    if (m > 1 && same_point2<T2>(qx[0], qy[0], qx[m - 1], qy[m - 1], eps2)) --m;
    for (int i = 0; i < m; ++i) {
        px[i] = qx[i];
        py[i] = qy[i];
    }
    return m;
}

template <typename T2>
__device__ __forceinline__ T2 poly_twice_area_2d(const T2 px[8], const T2 py[8], int n) {
    T2 twiceArea = (T2)0;
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        twiceArea += px[i] * py[j] - py[i] * px[j];
    }
    return twiceArea;
}

template <typename T2>
__device__ __forceinline__ void reverse_poly2d(T2 px[8], T2 py[8], int n) {
    for (int i = 0; i < n / 2; ++i) {
        const int j = n - 1 - i;
        const T2 tx = px[i], ty = py[i];
        px[i] = px[j]; py[i] = py[j];
        px[j] = tx;    py[j] = ty;
    }
}

template <typename T2>
__device__ __forceinline__ bool inside_ccw_edge(T2 px, T2 py, T2 ax, T2 ay, T2 bx, T2 by, T2 eps) {
    return cross2(bx - ax, by - ay, px - ax, py - ay) >= -eps;
}

template <typename T2>
__device__ __forceinline__ void line_intersection_2d(T2 sx, T2 sy, T2 ex, T2 ey,
                                                     T2 ax, T2 ay, T2 bx, T2 by,
                                                     T2& ox, T2& oy) {
    const T2 rx = ex - sx, ry = ey - sy;
    const T2 qx = bx - ax, qy = by - ay;
    const T2 den = cross2(rx, ry, qx, qy);
    if (absT(den) <= (T2)DEME_TINY_FLOAT) { ox = ex; oy = ey; return; }
    const T2 t   = cross2(ax - sx, ay - sy, qx, qy) / den;
    ox = sx + t * rx;
    oy = sy + t * ry;
}

template <typename T1, typename T2>
__device__ __forceinline__ int project_poly_3d_to_2d(const T1 inPoly[4], int n,
                                                     const T1& O, const T1& u, const T1& v,
                                                     T2 outX[8], T2 outY[8],
                                                     T2 eps) {
    for (int i = 0; i < n; ++i) {
        const T1 P = inPoly[i] - O;
        outX[i] = dot(P, u);
        outY[i] = dot(P, v);
    }
    n = cleanup_poly2d<T2>(outX, outY, n, eps);
    if (n >= 3 && poly_twice_area_2d<T2>(outX, outY, n) < (T2)0) {
        reverse_poly2d<T2>(outX, outY, n);
    }
    return n;
}

template <typename T2>
__device__ __forceinline__ int clip_convex_poly_2d(const T2 subjX[8], const T2 subjY[8], int nSubj,
                                                   const T2 clipX[8], const T2 clipY[8], int nClip,
                                                   T2 outX[8], T2 outY[8], T2 eps) {
    T2 inX0[8], inY0[8], inX1[8], inY1[8];
    int inCount = nSubj;
    for (int i = 0; i < nSubj; ++i) { inX0[i] = subjX[i]; inY0[i] = subjY[i]; }

    for (int e = 0; e < nClip; ++e) {
        const T2 ax = clipX[e], ay = clipY[e];
        const T2 bx = clipX[(e + 1) % nClip], by = clipY[(e + 1) % nClip];
        int outCount = 0;
        if (inCount == 0) return 0;
        for (int i = 0; i < inCount; ++i) {
            const int j = (i + inCount - 1) % inCount;
            const T2 sx = inX0[j], sy = inY0[j];
            const T2 ex = inX0[i], ey = inY0[i];
            const bool sIn = inside_ccw_edge<T2>(sx, sy, ax, ay, bx, by, eps);
            const bool eIn = inside_ccw_edge<T2>(ex, ey, ax, ay, bx, by, eps);

            if (sIn && eIn) {
                inX1[outCount] = ex; inY1[outCount] = ey; ++outCount;
            } else if (sIn && !eIn) {
                line_intersection_2d<T2>(sx, sy, ex, ey, ax, ay, bx, by, inX1[outCount], inY1[outCount]);
                ++outCount;
            } else if (!sIn && eIn) {
                line_intersection_2d<T2>(sx, sy, ex, ey, ax, ay, bx, by, inX1[outCount], inY1[outCount]);
                ++outCount;
                inX1[outCount] = ex; inY1[outCount] = ey; ++outCount;
            }
        }
        inCount = cleanup_poly2d<T2>(inX1, inY1, outCount, eps);
        for (int i = 0; i < inCount; ++i) { inX0[i] = inX1[i]; inY0[i] = inY1[i]; }
    }

    for (int i = 0; i < inCount; ++i) { outX[i] = inX0[i]; outY[i] = inY0[i]; }
    return inCount;
}

template <typename T2>
__device__ __forceinline__ bool polygon_area_centroid_2d(const T2 px[8], const T2 py[8], int n,
                                                         T2& area, T2& cx, T2& cy) {
    area = (T2)0;
    cx = (T2)0;
    cy = (T2)0;
    if (n < 3) return false;

    T2 twiceArea = (T2)0;
    T2 mx = (T2)0;
    T2 my = (T2)0;
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        const T2 cr = px[i] * py[j] - py[i] * px[j];
        twiceArea += cr;
        mx += (px[i] + px[j]) * cr;
        my += (py[i] + py[j]) * cr;
    }
    if (absT(twiceArea) <= (T2)DEME_TINY_FLOAT) return false;
    area = absT(twiceArea) * (T2)0.5;
    const T2 inv = (T2)1 / ((T2)3 * twiceArea);
    cx = mx * inv;
    cy = my * inv;
    return true;
}

// ------------------------------------------------------------------
// Tri-Tri pair contact area and point alongside contact island normal
// clipped submerged polygon against submerged polygon
// ------------------------------------------------------------------
template <typename T1, typename T2>
__device__ __forceinline__ bool area_cp_along_normal(const T1& A0, const T1& A1, const T1& A2,
                                                     const T1& B0, const T1& B1, const T1& B2,
                                                     T1 nCommon,
                                                     T2& projArea,
                                                     T1& contactPoint) {
    projArea = (T2)0;
    contactPoint = make_zero3<T1>();

    const T2 geomScale = local_length_scale6<T1,T2>(A0,A1,A2,B0,B1,B2);
    const T2 planeEps = rel_len_tol<T2>() * geomScale;
    const T2 eps2d = planeEps;

    const T2 nLen2 = dot(nCommon, nCommon);
    if (nLen2 <= (T2)(DEME_TINY_FLOAT * DEME_TINY_FLOAT)) return false;
    nCommon = nCommon * ((T2)1 / sqrt(nLen2));

    const T1 nA = cross(A1 - A0, A2 - A0);
    const T1 nB = cross(B1 - B0, B2 - B0);
    const T2 nALen2 = dot(nA, nA);
    const T2 nBLen2 = dot(nB, nB);
    if (nALen2 <= (T2)(DEME_TINY_FLOAT * DEME_TINY_FLOAT) ||
        nBLen2 <= (T2)(DEME_TINY_FLOAT * DEME_TINY_FLOAT)) {
        return false;
    }

    T1 aPen3[4], bPen3[4];
    int nAPen = clip_triangle_by_plane_keep_negative<T1,T2>(A0, A1, A2, B0, nB, planeEps, aPen3);
    int nBPen = clip_triangle_by_plane_keep_negative<T1,T2>(B0, B1, B2, A0, nA, planeEps, bPen3);
    if (nAPen < 3 || nBPen < 3) return false;

    T1 u, v;
    build_plane_basis_from_normal<T1,T2>(nCommon, u, v);
    const T1 O = (A0 + B0) * (T2)0.5;

    T2 aX[8], aY[8], bX[8], bY[8];
    nAPen = project_poly_3d_to_2d<T1,T2>(aPen3, nAPen, O, u, v, aX, aY, eps2d);
    nBPen = project_poly_3d_to_2d<T1,T2>(bPen3, nBPen, O, u, v, bX, bY, eps2d);
    if (nAPen < 3 || nBPen < 3) return false;

    T2 outX[8], outY[8];
    const int nPoly = clip_convex_poly_2d<T2>(bX, bY, nBPen, aX, aY, nAPen, outX, outY, eps2d);
    if (nPoly < 3) return false;

    T2 cx, cy;
    if (!polygon_area_centroid_2d<T2>(outX, outY, nPoly, projArea, cx, cy)) return false;

    const T2 denA = dot(nA, nCommon);
    const T2 denB = dot(nB, nCommon);
    if (absT(denA) <= (T2)DEME_TINY_FLOAT || absT(denB) <= (T2)DEME_TINY_FLOAT) return false;

    const T2 cA = dot(nA, A0 - O);
    const T2 cB = dot(nB, B0 - O);
    const T2 alphaA = dot(nA, u), betaA = dot(nA, v);
    const T2 alphaB = dot(nB, u), betaB = dot(nB, v);

    const T2 wA = (cA - alphaA * cx - betaA * cy) / denA;
    const T2 wB = (cB - alphaB * cx - betaB * cy) / denB;

    contactPoint = O + u * cx + v * cy + nCommon * ((wA + wB) * (T2)0.5);
    return true;
}

// ------------------------------------------------------------------
// Moeller inspired tri-tri penetration test
// ------------------------------------------------------------------
template <typename T1, typename T2>
__device__ __forceinline__ bool checkTriangleTriangleOverlap(
    const T1& A0,
    const T1& A1,
    const T1& A2,
    const T1& B0,
    const T1& B1,
    const T1& B2,
    T2& depth) {

    const T2 geomScale    = local_length_scale6<T1,T2>(A0,A1,A2,B0,B1,B2);
    const T2 lenTol       = rel_len_tol<T2>() * geomScale;
    const T2 aabbEps      = lenTol;
    const T2 planeTol     = lenTol;
    const T2 lineTol      = lenTol;
    const T2 copOrientTol = lenTol * lenTol;
    const T2 copCoordTol  = lenTol;
    const T2 angTol       = rel_ang_tol<T2>();

    if (!aabb_overlap6<T1,T2>(A0,A1,A2,B0,B1,B2,aabbEps)) {
        const T2 minAx = tmin3((T2)A0.x, (T2)A1.x, (T2)A2.x), maxAx = tmax3((T2)A0.x, (T2)A1.x, (T2)A2.x);
        const T2 minAy = tmin3((T2)A0.y, (T2)A1.y, (T2)A2.y), maxAy = tmax3((T2)A0.y, (T2)A1.y, (T2)A2.y);
        const T2 minAz = tmin3((T2)A0.z, (T2)A1.z, (T2)A2.z), maxAz = tmax3((T2)A0.z, (T2)A1.z, (T2)A2.z);
        const T2 minBx = tmin3((T2)B0.x, (T2)B1.x, (T2)B2.x), maxBx = tmax3((T2)B0.x, (T2)B1.x, (T2)B2.x);
        const T2 minBy = tmin3((T2)B0.y, (T2)B1.y, (T2)B2.y), maxBy = tmax3((T2)B0.y, (T2)B1.y, (T2)B2.y);
        const T2 minBz = tmin3((T2)B0.z, (T2)B1.z, (T2)B2.z), maxBz = tmax3((T2)B0.z, (T2)B1.z, (T2)B2.z);
        const T2 sepX = max2((T2)0, max2(minBx - maxAx, minAx - maxBx));
        const T2 sepY = max2((T2)0, max2(minBy - maxAy, minAy - maxBy));
        const T2 sepZ = max2((T2)0, max2(minBz - maxAz, minAz - maxBz));
        depth = -max2(sepX, max2(sepY, sepZ));
        return false;
    }

    const T1 E1 = A1 - A0;
    const T1 E2 = A2 - A0;
    const T1 F1 = B1 - B0;
    const T1 F2 = B2 - B0;
    const T1 N1 = cross(E1, E2);
    const T1 N2 = cross(F1, F2);

    const T2 n1Len2 = dot(N1, N1);
    const T2 n2Len2 = dot(N2, N2);
    if (n1Len2 <= (T2)(DEME_TINY_FLOAT * DEME_TINY_FLOAT) ||
        n2Len2 <= (T2)(DEME_TINY_FLOAT * DEME_TINY_FLOAT)) {
        return false;
    }
    const T2 invN1 = (T2)1 / sqrt(n1Len2);
    const T2 invN2 = (T2)1 / sqrt(n2Len2);

    const T2 du0 = (T2)dot(N1, B0 - A0) * invN1;
    const T2 du1 = (T2)dot(N1, B1 - A0) * invN1;
    const T2 du2 = (T2)dot(N1, B2 - A0) * invN1;
    const bool sideRejectN1 =
        ((du0 > planeTol && du1 > planeTol && du2 > planeTol) ||
         (du0 < -planeTol && du1 < -planeTol && du2 < -planeTol));

    const T2 dv0 = (T2)dot(N2, A0 - B0) * invN2;
    const T2 dv1 = (T2)dot(N2, A1 - B0) * invN2;
    const T2 dv2 = (T2)dot(N2, A2 - B0) * invN2;
    const bool sideRejectN2 =
        ((dv0 > planeTol && dv1 > planeTol && dv2 > planeTol) ||
         (dv0 < -planeTol && dv1 < -planeTol && dv2 < -planeTol));

    if (sideRejectN1 || sideRejectN2) {
        T2 sep = (T2)0;
        if (sideRejectN1) {
            if (du0 > planeTol && du1 > planeTol && du2 > planeTol) {
                sep = max2(sep, tmin3(du0, du1, du2));
            } else if (du0 < -planeTol && du1 < -planeTol && du2 < -planeTol) {
                sep = max2(sep, tmin3(-du0, -du1, -du2));
            }
        }
        if (sideRejectN2) {
            if (dv0 > planeTol && dv1 > planeTol && dv2 > planeTol) {
                sep = max2(sep, tmin3(dv0, dv1, dv2));
            } else if (dv0 < -planeTol && dv1 < -planeTol && dv2 < -planeTol) {
                sep = max2(sep, tmin3(-dv0, -dv1, -dv2));
            }
        }
        depth = -sep;
        return false;
    }

    const T2 penA = max2((T2)0, max2(-du0, max2(-du1, -du2)));
    const T2 penB = max2((T2)0, max2(-dv0, max2(-dv1, -dv2)));
    const T1 n1u = N1 * invN1;
    const T1 n2u = N2 * invN2;

    const T1 D = cross(n1u, n2u);
    if (dot(D, D) <= (T2)(angTol * angTol)) {
        const bool hit = coplanar_tri_tri<T1,T2>(N1, A0,A1,A2, B0,B1,B2, copOrientTol, copCoordTol);
        if (hit) {
            depth = min2(penA, penB);
            if (depth > lineTol) {
                return true;
            }
        }
        if (depth > (T2)0) {
            depth = -depth;
        }
        return false;
    }

    const int axis = dominant_axis<T1,T2>(D);
    const T2 ap0 = coord_axis<T1,T2>(A0, axis);
    const T2 ap1 = coord_axis<T1,T2>(A1, axis);
    const T2 ap2 = coord_axis<T1,T2>(A2, axis);
    const T2 bp0 = coord_axis<T1,T2>(B0, axis);
    const T2 bp1 = coord_axis<T1,T2>(B1, axis);
    const T2 bp2 = coord_axis<T1,T2>(B2, axis);

    T2 ia[6], ib[6];
    const int na = interval_from_plane_hits<T2>(ap0, ap1, ap2, dv0, dv1, dv2, ia, lineTol);
    const int nb = interval_from_plane_hits<T2>(bp0, bp1, bp2, du0, du1, du2, ib, lineTol);

    if (na == 0 || nb == 0) {
        const bool hit = coplanar_tri_tri<T1,T2>(N1, A0,A1,A2, B0,B1,B2, copOrientTol, copCoordTol);
        if (hit) {
            depth = min2(penA, penB);
            if (depth > lineTol) {
                return true;
            }
        }
        if (depth > (T2)0) {
            depth = -depth;
        }
        return false;
    }

    T2 a0 = ia[0], a1 = ia[0];
    for (int i = 1; i < na; ++i) {
        a0 = min2(a0, ia[i]);
        a1 = max2(a1, ia[i]);
    }
    T2 b0 = ib[0], b1 = ib[0];
    for (int i = 1; i < nb; ++i) {
        b0 = min2(b0, ib[i]);
        b1 = max2(b1, ib[i]);
    }
    sort2(a0, a1);
    sort2(b0, b1);
    if (a1 < b0 - lineTol || b1 < a0 - lineTol) {
        const T2 sep = max2(b0 - a1, a0 - b1);
        depth = -sep;
        return false;
    }
    depth = min2(penA, penB);
    if (depth > lineTol) {
        return true;
    }
    if (depth > (T2)0) {
        depth = -depth;
    }
    return false;
}

/// Triangle-triangle contact detection with split pipeline:
/// 1. overlap + depth via replacement checkTriangleTriangleOverlap(..., depth)
/// 2. normal selection/orientation (B->A)
/// 3. separate area/contact-point reconstruction via area_cp_along_normal
template <typename T1, typename T2>
__device__ bool checkTriangleTriangleOverlap(
    const T1& A1,
    const T1& B1,
    const T1& C1,
    const T1& A2,
    const T1& B2,
    const T1& C2,
    T1& normal,
    T2& depth,
    T2& projectedArea,
    T1& point) {



    projectedArea = (T2)0;
    point = make_zero3<T1>();
    normal = make_zero3<T1>();
    depth = (T2)0;

    const bool hit = checkTriangleTriangleOverlap<T1, T2>(A1, B1, C1, A2, B2, C2, depth);

    if (!hit) {
        const T1 centA = (A1 + B1 + C1) / (T2)3;
        const T1 centB = (A2 + B2 + C2) / (T2)3;
        const T1 sep = centA - centB;
        const T2 sepLen2 = dot(sep, sep);

        if (sepLen2 > (T2)(DEME_TINY_FLOAT * DEME_TINY_FLOAT)) {
            const T2 sepLen = sqrt(sepLen2);
            normal = sep / sepLen;
            point = (centA + centB) * (T2)0.5;
            if (!(depth < (T2)0))
                depth = -sepLen;
        } else {
            const T1 nAraw = cross(B1 - A1, C1 - A1);
            const T2 nArawLen2 = dot(nAraw, nAraw);
            if (nArawLen2 > (T2)(DEME_TINY_FLOAT * DEME_TINY_FLOAT)) {
                normal = nAraw * ((T2)1 / sqrt(nArawLen2));
            }
            point = centA;
            if (!(depth < (T2)0))
                depth = -(T2)DEME_TINY_FLOAT;
        }
        projectedArea = (T2)0;
        return false;
    }

    const T1 nAraw = cross(B1 - A1, C1 - A1);
    const T1 nBraw = cross(B2 - A2, C2 - A2);
    const T2 nALen2 = dot(nAraw, nAraw);
    const T2 nBLen2 = dot(nBraw, nBraw);
    if (nALen2 <= (T2)(DEME_TINY_FLOAT * DEME_TINY_FLOAT) ||
        nBLen2 <= (T2)(DEME_TINY_FLOAT * DEME_TINY_FLOAT)) {
        return false;
    }

    const T1 nA = nAraw * ((T2)1 / sqrt(nALen2));
    const T1 nB = nBraw * ((T2)1 / sqrt(nBLen2));
    const T2 penA = max2((T2)0, max2(-(T2)dot(nA, A2 - A1), max2(-(T2)dot(nA, B2 - A1), -(T2)dot(nA, C2 - A1))));
    const T2 penB = max2((T2)0, max2(-(T2)dot(nB, A1 - A2), max2(-(T2)dot(nB, B1 - A2), -(T2)dot(nB, C1 - A2))));

    normal = (penA <= penB) ? ((T2)-1 * nA) : nB;

    depth = min2(penA, penB);

    const bool area_ok = area_cp_along_normal<T1, T2>(A1, B1, C1, A2, B2, C2, normal, projectedArea, point);
    if (!area_ok) {
        projectedArea = (T2)0;
        T1 closestA, closestB;
        const T2 sep_dist = closestPtTriTriDistance<T1, T2>(A1, B1, C1, A2, B2, C2, closestA, closestB);
        if (sep_dist >= (T2)0 && sep_dist < (T2)DEME_HUGE_FLOAT) {
            point = (closestA + closestB) * (T2)0.5;
        } else {
            const T1 centA = (A1 + B1 + C1) / (T2)3;
            const T1 centB = (A2 + B2 + C2) / (T2)3;
            point = (centA + centB) * (T2)0.5;
        }
    }

    return true;
}

#endif
