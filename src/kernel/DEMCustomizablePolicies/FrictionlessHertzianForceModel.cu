// DEM force calculation strategies, modifiable

if (overlapDepth > 0) {
    // Material properties
    float E_cnt, CoR_cnt;
    {
        matProxy2ContactParam<float>(E_cnt, E[bodyAMatType], nu[bodyAMatType], E[bodyBMatType], nu[bodyBMatType]);
        // CoR is pair-wise, so obtain it this way
        CoR_cnt = CoR[bodyAMatType][bodyBMatType];
    }

    float3 rotVelCPA = make_float3(0.f, 0.f, 0.f);
    float3 rotVelCPB = make_float3(0.f, 0.f, 0.f);
    if constexpr (AType != deme::GEO_T_SPHERE || BType != deme::GEO_T_SPHERE) {
        // We also need the relative velocity between A and B in global frame to use in the damping terms
        // To get that, we need contact points' rotational velocity in GLOBAL frame
        // This is local rotational velocity (the portion of linear vel contributed by rotation)
        rotVelCPA = cross(ARotVel, locCPA);
        rotVelCPB = cross(BRotVel, locCPB);
        // This is mapping from local rotational velocity to global
        applyOriQToVector3<float, deme::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AOriQ.w, AOriQ.x, AOriQ.y,
                                                AOriQ.z);
        applyOriQToVector3<float, deme::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BOriQ.w, BOriQ.x, BOriQ.y,
                                                BOriQ.z);
    } else {
        if (simParams->useAngVelMargin) {
            rotVelCPA = cross(ARotVel, locCPA);
            rotVelCPB = cross(BRotVel, locCPB);
            applyOriQToVector3<float, deme::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AOriQ.w, AOriQ.x, AOriQ.y,
                                                    AOriQ.z);
            applyOriQToVector3<float, deme::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BOriQ.w, BOriQ.x, BOriQ.y,
                                                    BOriQ.z);
        }
    }

    // The (total) relative linear velocity of A relative to B
    const float3 velB2A = (ALinVel + rotVelCPA) - (BLinVel + rotVelCPB);
    const float projection = dot(velB2A, B2A);

    const float mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);

    // Contact radius:
    // - sphere/triangle contacts: area-based partial-contact radius with ideal-sphere correction.
    //   Full patch recovers Hertz exactly, while edge/corner truncation still reduces force via overlapArea.
    //   Implemented as a_area * rsqrt(2 - d / R), i.e. no expensive division in the hot path if invR is known.
    // - triangle/triangle and other non-spherical triangle contacts: area-based proxy.
    // - everything else: classic Hertz with R_eff.-based proxy.
    const bool tri_involved = (AType == deme::GEO_T_TRIANGLE) || (BType == deme::GEO_T_TRIANGLE);
    float cnt_rad;
    if (tri_involved) {
        const float area_radius = sqrtf(fmaxf(overlapArea, 0.f) * deme::INV_PI);
        if constexpr (AType == deme::GEO_T_SPHERE && BType == deme::GEO_T_TRIANGLE) {
            const float effective_radius = ARadius;
            const float depth_over_R = fminf(fmaxf(overlapDepth / effective_radius, 0.f), 1.f);
            cnt_rad = area_radius * rsqrtf(2.f - depth_over_R);
        } else if constexpr (BType == deme::GEO_T_SPHERE && AType == deme::GEO_T_TRIANGLE) {
            const float effective_radius = BRadius;
            const float depth_over_R = fminf(fmaxf(overlapDepth / effective_radius, 0.f), 1.f);
            cnt_rad = area_radius * rsqrtf(2.f - depth_over_R);
        } else {
            cnt_rad = area_radius;
        }
    } else {
        const float effective_radius =
            (BType == deme::GEO_T_ANALYTICAL) ? ARadius : (ARadius * BRadius) / (ARadius + BRadius);
        cnt_rad = sqrtf(overlapDepth * effective_radius);
    }
    const float Sn = 2.f * E_cnt * cnt_rad;

    const float loge = (CoR_cnt < DEME_TINY_FLOAT) ? log(DEME_TINY_FLOAT) : log(CoR_cnt);
    const float beta = loge / sqrt(loge * loge + deme::PI_SQUARED);

    const float k_n = (2.f / 3.f) * Sn;
    const float gamma_n = deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrtf(Sn * mass_eff);

    force += (k_n * overlapDepth + gamma_n * projection) * B2A;
}
