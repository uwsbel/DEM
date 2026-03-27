// DEM force calculation strategies, modifiable
// Hertz-Mindlin baseline + effective dry adhesion (unloading hysteresis in penetration domain)
// + wet capillary bridge (gap-based, rupture-limited).
//
// Pair-wise material properties used by this model:
//   AdhesionDryPullOff  [-]  : dry pull-off ratio relative to a particle weight (1.0 = m*g)
//   AdhesionDryDistance [m]  : dry unloading width delta_c in penetration space
//   AdhesionWetCap      [-]  : wet capillary force ratio relative to a particle weight (1.0 = m*g)
//   AdhesionWetRupture  [m]  : wet bridge rupture gap distance
//
// Owner wildcards used by this model:
//   AdhesionWeight           : per-owner characteristic particle weight [N]; 0 for non-particles
//
// Contact wildcards used by this model:
//   delta_time, delta_tan_x/y/z : standard Hertz-Mindlin tangential history
//   delta_max                   : max physical penetration reached during current contact
//   bridge_on                   : wet bridge hysteresis state (0/1 encoded as float)

// Material properties
float E_cnt, G_cnt, CoR_cnt, mu_cnt, Crr_cnt;
float dry_pull_off_cnt = 0.f;
float dry_delta_cnt = 0.f;
float wet_cap_cnt = 0.f;
float wet_rupture_cnt = 0.f;
{
    // E and nu are associated with each material, so obtain them this way
    float E_A = E[bodyAMatType];
    float nu_A = nu[bodyAMatType];
    float E_B = E[bodyBMatType];
    float nu_B = nu[bodyBMatType];
    matProxy2ContactParam<float>(E_cnt, G_cnt, E_A, nu_A, E_B, nu_B);
    // Pair-wise properties
    CoR_cnt = CoR[bodyAMatType][bodyBMatType];
    mu_cnt = mu[bodyAMatType][bodyBMatType];
    Crr_cnt = Crr[bodyAMatType][bodyBMatType];
    dry_pull_off_cnt = AdhesionDryPullOff[bodyAMatType][bodyBMatType];
    dry_delta_cnt = AdhesionDryDistance[bodyAMatType][bodyBMatType];
    wet_cap_cnt = AdhesionWetCap[bodyAMatType][bodyBMatType];
    wet_rupture_cnt = AdhesionWetRupture[bodyAMatType][bodyBMatType];
}

const bool dry_enabled = (dry_pull_off_cnt > DEME_TINY_FLOAT) && (dry_delta_cnt > DEME_TINY_FLOAT);
const bool wet_enabled = (wet_cap_cnt > DEME_TINY_FLOAT) && (wet_rupture_cnt > DEME_TINY_FLOAT);
const float dry_delta_c = dry_enabled ? dry_delta_cnt : 0.f;

// Keep wet bridge alive while the pair is still inside rupture range.
const bool wet_bridge_live = wet_enabled && (bridge_on > 0.5f) && (overlapDepth > -wet_rupture_cnt);
const bool physical_contact = overlapDepth > 0.f;

const float adhesion_weight_A = fmaxf(AdhesionWeight[AOwner], 0.f);
const float adhesion_weight_B = fmaxf(AdhesionWeight[BOwner], 0.f);
float adhesion_weight_cnt = 0.f;
if (adhesion_weight_A > DEME_TINY_FLOAT && adhesion_weight_B > DEME_TINY_FLOAT) {
    adhesion_weight_cnt = fminf(adhesion_weight_A, adhesion_weight_B);
} else {
    adhesion_weight_cnt = fmaxf(adhesion_weight_A, adhesion_weight_B);
}
const float dry_pull_off_force = dry_pull_off_cnt * adhesion_weight_cnt;
const float wet_cap_force = wet_cap_cnt * adhesion_weight_cnt;

// No need to do any contact force calculation if no physical contact or active wet bridge.
if (physical_contact || wet_bridge_live) {
    float3 rotVelCPA = make_float3(0.f, 0.f, 0.f);
    float3 rotVelCPB = make_float3(0.f, 0.f, 0.f);
    // Contact relative velocity must always include rotational contributions at the contact point.
    rotVelCPA = cross(ARotVel, locCPA);
    rotVelCPB = cross(BRotVel, locCPB);
    applyOriQToVector3<float, deme::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AOriQ.w, AOriQ.x, AOriQ.y,
                                            AOriQ.z);
    applyOriQToVector3<float, deme::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BOriQ.w, BOriQ.x, BOriQ.y,
                                            BOriQ.z);

    const bool tri_involved = (AType == deme::GEO_T_TRIANGLE) || (BType == deme::GEO_T_TRIANGLE);
    float contact_radius = 0.f;    // area-based radius or sqrt(overlapDepth * R_eff)
    float effective_radius = 0.f;  // geometric effective radius when no triangle is involved

    float mass_eff;
    float beta = 0.f;
    float projection = 0.f;
    float3 vrel_tan = make_float3(0.f, 0.f, 0.f);
    float3 delta_tan = make_float3(delta_tan_x, delta_tan_y, delta_tan_z);

    // Relative kinematics are needed for both physical-contact Hertz and wet bridge damping direction.
    const float3 velB2A = (ALinVel + rotVelCPA) - (BLinVel + rotVelCPB);
    projection = dot(velB2A, B2A);
    vrel_tan = velB2A - projection * B2A;
    mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);

    // Wet bridge state update. Formation only on physical contact; once formed, it survives into the gap.
    if (wet_enabled) {
        if (physical_contact || wet_bridge_live) {
            bridge_on = 1.f;
        } else {
            bridge_on = 0.f;
        }
    } else {
        bridge_on = 0.f;
    }

    if (physical_contact) {
        // Standard tangential history update applies only in physical contact.
        delta_tan += ts * vrel_tan;
        const float disp_proj = dot(delta_tan, B2A);
        delta_tan -= disp_proj * B2A;
        delta_time += ts;
        delta_max = fmaxf(delta_max, overlapDepth);

        // Contact radius:
        // - sphere/triangle contacts: classic Hertz with R_eff = sphere radius (triangle treated as locally flat).
        // - triangle/triangle and other non-spherical triangle contacts: area-based proxy.
        // - everything else: classic Hertz with R_eff.
        if (tri_involved) {
            if constexpr (AType == deme::GEO_T_SPHERE && BType == deme::GEO_T_TRIANGLE) {
                effective_radius = ARadius;
                contact_radius = sqrtf(overlapDepth * effective_radius);
            } else if constexpr (BType == deme::GEO_T_SPHERE && AType == deme::GEO_T_TRIANGLE) {
                effective_radius = BRadius;
                contact_radius = sqrtf(overlapDepth * effective_radius);
            } else {
                contact_radius = sqrtf(overlapArea / deme::PI);
            }
        } else {
            effective_radius = (BType == deme::GEO_T_ANALYTICAL) ? ARadius : (ARadius * BRadius) / (ARadius + BRadius);
            contact_radius = sqrtf(overlapDepth * effective_radius);
        }

        const float Sn = 2.f * E_cnt * contact_radius;
        const float loge = (CoR_cnt < DEME_TINY_FLOAT) ? log(DEME_TINY_FLOAT) : log(CoR_cnt);
        beta = loge / sqrt(loge * loge + deme::PI_SQUARED);

        const float k_n = (2.f / 3.f) * Sn;
        const float gamma_n = (2.f * sqrtf(5.f / 6.f)) * beta * sqrtf(Sn * mass_eff);

        float normal_force_mag = k_n * overlapDepth + gamma_n * projection;

        // Dry adhesion is an unloading-only, penetration-domain correction. To keep it robust and cheap,
        // the active width is capped by both the configured delta_c and the actually reached delta_max.
        // When a wet bridge is active we suppress dry adhesion to avoid double counting.
        if (dry_enabled && !(wet_enabled && bridge_on > 0.5f)) {
            const float dry_width = fminf(delta_max, dry_delta_c);
            if ((projection > 0.f) && (dry_width > DEME_TINY_FLOAT) && (overlapDepth <= dry_width)) {
                const float u = fminf(fmaxf(overlapDepth / dry_width, 0.f), 1.f);
                normal_force_mag -= dry_pull_off_force * (1.f - u);
            }
        }

        // Wet bridge acts as an additional tensile normal load in contact as well.
        if (wet_enabled && bridge_on > 0.5f) {
            normal_force_mag -= wet_cap_force;
        }

        force += normal_force_mag * B2A;

        // Rolling resistance part
        if (Crr_cnt > 0.f) {
            bool should_add_rolling_resistance = true;
            {
                const float R_eff = tri_involved ? ((contact_radius * contact_radius) / overlapDepth) : effective_radius;
                const float kn_simple = deme::FOUR_OVER_THREE * E_cnt * sqrtf(R_eff);
                const float gn_simple = -2.f * sqrtf(deme::FIVE_OVER_THREE * mass_eff * E_cnt) * beta * powf(R_eff, 0.25f);
                const float d_coeff = gn_simple / (2.f * sqrtf(kn_simple * mass_eff));
                if (d_coeff < 1.f) {
                    float t_collision = deme::PI * sqrtf(mass_eff / (kn_simple * (1.f - d_coeff * d_coeff)));
                    if (delta_time <= t_collision) {
                        should_add_rolling_resistance = false;
                    }
                }
            }
            if (should_add_rolling_resistance) {
                const float3 v_rot = rotVelCPB - rotVelCPA;
                const float v_rot_mag = length(v_rot);
                if (v_rot_mag > DEME_TINY_FLOAT) {
                    torque_only_force = (v_rot / v_rot_mag) * (Crr_cnt * length(force));
                }
            }
        }

        // Tangential force part
        if (mu_cnt > 0.f) {
            float gt;
            const float kt = 8.f * G_cnt * contact_radius;
            if (tri_involved) {
                gt = -deme::TWO_TIMES_SQRT_FIVE_OVER_THREE * beta * sqrtf(mass_eff * kt);
            } else {
                gt = -deme::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrtf(mass_eff * kt);
            }
            float3 tangent_force = -kt * delta_tan - gt * vrel_tan;
            const float ft = length(tangent_force);
            if (ft > DEME_TINY_FLOAT) {
                const float ft_max = length(force) * mu_cnt;
                if (ft > ft_max) {
                    tangent_force = (ft_max / ft) * tangent_force;
                    delta_tan = (tangent_force + gt * vrel_tan) / (-kt);
                }
            } else {
                tangent_force = make_float3(0.f, 0.f, 0.f);
            }
            force += tangent_force;
        }
    } else {
        // Gap-only wet bridge: no tangential contact history and no dry memory should remain active.
        delta_time = 0.f;
        delta_tan = make_float3(0.f, 0.f, 0.f);
        delta_max = 0.f;

        if (wet_enabled && bridge_on > 0.5f) {
            const float gap = -overlapDepth;
            if (gap < wet_rupture_cnt) {
                const float wet_ratio = fmaxf(0.f, 1.f - gap / wet_rupture_cnt);
                force += (-wet_cap_force * wet_ratio) * B2A;
            } else {
                bridge_on = 0.f;
            }
        }
    }

    delta_tan_x = delta_tan.x;
    delta_tan_y = delta_tan.y;
    delta_tan_z = delta_tan.z;
} else {
    // If no physical contact / active wet bridge, clear history.
    delta_time = 0.f;
    delta_tan_x = 0.f;
    delta_tan_y = 0.f;
    delta_tan_z = 0.f;
    delta_max = 0.f;
    bridge_on = 0.f;
}
