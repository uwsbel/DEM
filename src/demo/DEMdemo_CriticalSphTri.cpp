//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//  SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// UI-equivalent sphere_test demo.
//
// Purpose:
//   Recreate the uploaded UI project `sphere_test` directly in the solver demo
//   layer, using only features that already exist in the solver.
//
// Scene mapping from sphere_test:
//   - 1000 analytical spheres, diameter 12 mm, density 2600 kg/m^3
//   - rotating analytic drum side wall via planar-contact cylinder
//   - top/bottom planes with the same cap clearance as the UI export
//   - one dynamic sphere mesh workpiece inside the drum
//   - gravity and material values copied from the UI project
//
// Intentionally omitted because the plain solver demo path does not expose the
// full UI/runtime plumbing for them in a simple standalone example:
//   - workpiece inspection elements
//   - triangle-metric handover stream
//   - mesh deform / wear streams
//   - GUI-only export options
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/HostSideHelpers.hpp>
#include <DEM/utils/Samplers.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstdlib>

using namespace deme;
using namespace std::filesystem;

namespace {

constexpr float PI_F = static_cast<float>(deme::PI);

struct DemoConfig {
    unsigned int target_particles = 1000;
    float sphere_radius = 0.006f;  // 12 mm diameter
    float sphere_density = 2600.0f;
    float step_size = 2.0e-5f;
    float time_end = 2.0f;
    unsigned int output_fps = 60;
    int cd_update_freq = 0;
    bool disable_adaptive_cd_update = true;
    float gravity = -9.81000042f;

    float drum_radius = 0.09f;
    float drum_height = 0.8f;
    float drum_center_x = 0.0f;
    float drum_center_y = 0.0f;
    float drum_rpm = 120.0f;
    float wall_clearance = 0.00200000009f;
    float fill_height_ratio = 0.600000024f;
    float fill_spacing_factor = 2.0f;

    path mesh_file = (GET_DATA_PATH() / "mesh/sphere_highres.obj").string();
    float mesh_scale = 0.025f;
    float mesh_mass = 5.0f;
    float3 mesh_init_pos = make_float3(0.0f, 0.05f, 0.15f);
    float3 mesh_init_euler_deg = make_float3(0.0f, 0.0f, 0.0f);
    float workpiece_sink_speed = 0.2f;  // m/s downward
    float workpiece_sink_duration = 0.5f;  // s

    float mat_particle_E = 15000000.0f;
    float mat_particle_nu = 0.300000012f;
    float mat_particle_CoR = 0.449999988f;
    float mat_particle_mu = 0.550000012f;
    float mat_particle_Crr = 0.00999999978f;

    float mat_boundary_E = 20000000.0f;
    float mat_boundary_nu = 0.300000012f;
    float mat_boundary_CoR = 0.349999994f;
    float mat_boundary_mu = 0.600000024f;
    float mat_boundary_Crr = 0.00999999978f;

    float mat_mesh_E = 20000000.0f;
    float mat_mesh_nu = 0.300000012f;
    float mat_mesh_CoR = 0.349999994f;
    float mat_mesh_mu = 0.600000024f;
    float mat_mesh_Crr = 0.00999999978f;

    float pair_particle_boundary_mu = 0.600000024f;
    float pair_particle_boundary_CoR = 0.349999994f;
    float pair_particle_boundary_Crr = 0.00999999978f;
    float pair_particle_mesh_mu = 0.550000012f;
    float pair_particle_mesh_CoR = 0.349999994f;
    float pair_particle_mesh_Crr = 0.00999999978f;
    float pair_boundary_mesh_mu = 0.600000024f;
    float pair_boundary_mesh_CoR = 0.349999994f;
    float pair_boundary_mesh_Crr = 0.00999999978f;

    unsigned int random_seed = 42;
    bool write_contacts = false;
    path out_dir = "DemoOutput_DEMdemo_CriticalSphTri";
};

float SphereMass(float radius, float density) {
    return density * (4.0f / 3.0f) * PI_F * radius * radius * radius;
}

bool TryGetEnvFloat(const char* name, float& out) {
    if (const char* v = std::getenv(name)) {
        try {
            out = std::stof(v);
            return true;
        } catch (...) {
        }
    }
    return false;
}

bool TryGetEnvUInt(const char* name, unsigned int& out) {
    if (const char* v = std::getenv(name)) {
        try {
            out = static_cast<unsigned int>(std::stoul(v));
            return true;
        } catch (...) {
        }
    }
    return false;
}

bool TryGetEnvInt(const char* name, int& out) {
    if (const char* v = std::getenv(name)) {
        try {
            out = std::stoi(v);
            return true;
        } catch (...) {
        }
    }
    return false;
}

bool GetEnvBool(const char* name, bool default_value) {
    if (const char* v = std::getenv(name)) {
        const std::string s(v);
        if (s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "on") {
            return true;
        }
        if (s == "0" || s == "false" || s == "FALSE" || s == "no" || s == "off") {
            return false;
        }
    }
    return default_value;
}

std::string GetEnvString(const char* name, const std::string& default_value) {
    if (const char* v = std::getenv(name)) {
        return std::string(v);
    }
    return default_value;
}

struct FrameMetrics {
    float max_v = 0.0f;
    int max_v_id = -1;
    float max_v_dist_to_workpiece = std::numeric_limits<float>::infinity();
    float max_v_contact_gap = std::numeric_limits<float>::infinity();
    float max_z = -std::numeric_limits<float>::infinity();
    unsigned int spheres_inside = 0;
    float max_penetration = 0.0f;
    int first_inside_id = -1;
    int max_penetration_id = -1;
};

std::vector<float3> BuildInitialSpherePositions(const DemoConfig& cfg) {
    const float clearance = std::max(0.0f, cfg.wall_clearance);
    const float min_usable_height = 2.0f * (cfg.sphere_radius + 1.0e-5f);
    const float requested_height = std::max(0.02f, cfg.fill_height_ratio * cfg.drum_height);
    const float max_height = std::max(min_usable_height, cfg.drum_height - 2.0f * clearance);
    const float usable_height = std::clamp(requested_height, min_usable_height, max_height);
    const float sample_half_height = std::max(1.0e-6f, 0.5f * usable_height - cfg.sphere_radius);
    const float sample_radius = std::max(1.0e-6f, cfg.drum_radius - clearance - cfg.sphere_radius);
    const float center_z = clearance + cfg.sphere_radius + sample_half_height;

    const float separation = std::max(2.0f * cfg.sphere_radius * 1.001f,
                                      cfg.fill_spacing_factor * cfg.sphere_radius);

    HCPSampler sampler(separation);
    auto pts = sampler.SampleCylinderZ(make_float3(cfg.drum_center_x, cfg.drum_center_y, center_z),
                                       sample_radius, sample_half_height);
    if (pts.empty()) {
        throw std::runtime_error("Failed to generate any initial particle seeds.");
    }

    std::stable_sort(pts.begin(), pts.end(), [](const float3& a, const float3& b) {
        if (a.z != b.z) {
            return a.z < b.z;
        }
        if (a.x != b.x) {
            return a.x < b.x;
        }
        return a.y < b.y;
    });

    const float workpiece_exclusion_r =
        0.5f * cfg.mesh_scale + cfg.sphere_radius + std::max(1.0e-4f, 0.5f * cfg.wall_clearance);

    std::vector<float3> filtered;
    filtered.reserve(std::min<std::size_t>(pts.size(), cfg.target_particles));
    for (const auto& p : pts) {
        const float3 d = p - cfg.mesh_init_pos;
        if (length(d) <= workpiece_exclusion_r) {
            continue;
        }
        filtered.push_back(p);
        if (filtered.size() >= cfg.target_particles) {
            break;
        }
    }

    if (filtered.size() < cfg.target_particles) {
        std::cerr << "Warning: after applying UI-like bottom-up inlet placement and workpiece exclusion, only "
                  << filtered.size() << " particle positions remain (requested "
                  << cfg.target_particles << ").\n";
    }

    return filtered;
}

std::vector<std::vector<uint32_t>> BuildTriangleAdjacency(const std::vector<float3>& vertices,
                                                          const std::vector<int3>& faces) {
    std::vector<std::vector<uint32_t>> adjacency(faces.size());
    if (faces.empty()) {
        return adjacency;
    }

    std::vector<size_t> canon;
    if (!vertices.empty()) {
        const double eps = computeVertexQuantEps(vertices);
        canon = buildCanonicalVertexMap(vertices, eps);
    }

    std::unordered_map<uint64_t, std::vector<uint32_t>> edge_to_faces;
    edge_to_faces.reserve(faces.size() * 3);

    auto edge_key = [](int a, int b) -> uint64_t {
        const uint32_t lo = static_cast<uint32_t>(std::min(a, b));
        const uint32_t hi = static_cast<uint32_t>(std::max(a, b));
        return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    };

    for (uint32_t i = 0; i < faces.size(); ++i) {
        const int3& f = faces[i];
        const int vx_raw = f.x;
        const int vy_raw = f.y;
        const int vz_raw = f.z;
        if (vx_raw < 0 || vy_raw < 0 || vz_raw < 0) {
            continue;
        }

        int vx = vx_raw;
        int vy = vy_raw;
        int vz = vz_raw;
        if (!canon.empty()) {
            if (static_cast<size_t>(vx_raw) >= canon.size() || static_cast<size_t>(vy_raw) >= canon.size() ||
                static_cast<size_t>(vz_raw) >= canon.size()) {
                continue;
            }
            vx = static_cast<int>(canon[static_cast<size_t>(vx_raw)]);
            vy = static_cast<int>(canon[static_cast<size_t>(vy_raw)]);
            vz = static_cast<int>(canon[static_cast<size_t>(vz_raw)]);
        }
        if (vx == vy || vy == vz || vz == vx) {
            continue;
        }

        edge_to_faces[edge_key(vx, vy)].push_back(i);
        edge_to_faces[edge_key(vy, vz)].push_back(i);
        edge_to_faces[edge_key(vz, vx)].push_back(i);
    }

    for (const auto& kv : edge_to_faces) {
        const auto& shared = kv.second;
        if (shared.size() < 2) {
            continue;
        }
        for (size_t i = 0; i < shared.size(); ++i) {
            for (size_t j = i + 1; j < shared.size(); ++j) {
                const uint32_t a = shared[i];
                const uint32_t b = shared[j];
                adjacency[a].push_back(b);
                adjacency[b].push_back(a);
            }
        }
    }

    for (auto& nbrs : adjacency) {
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
    return adjacency;
}

std::vector<patchID_t> BuildLocalTrianglePatchIDs(const std::vector<float3>& vertices,
                                                  const std::vector<int3>& faces,
                                                  size_t target_faces_per_patch) {
    const size_t n_tri = faces.size();
    std::vector<patchID_t> patch_ids(n_tri, 0);
    if (n_tri == 0) {
        return patch_ids;
    }

    const size_t target = std::max<size_t>(1, target_faces_per_patch);
    const auto adjacency = BuildTriangleAdjacency(vertices, faces);

    std::vector<unsigned char> assigned(n_tri, 0);
    patchID_t cur_patch = 0;
    std::queue<uint32_t> q;

    for (uint32_t seed = 0; seed < n_tri; ++seed) {
        if (assigned[seed]) {
            continue;
        }
        size_t in_patch = 0;
        q.push(seed);
        while (!q.empty()) {
            const uint32_t tri = q.front();
            q.pop();
            if (tri >= n_tri || assigned[tri]) {
                continue;
            }
            assigned[tri] = 1;
            patch_ids[tri] = cur_patch;
            ++in_patch;
            if (in_patch >= target) {
                continue;
            }
            for (uint32_t nb : adjacency[tri]) {
                if (nb < n_tri && !assigned[nb]) {
                    q.push(nb);
                }
            }
        }
        ++cur_patch;
    }

    return patch_ids;
}

struct IsolatedTriangleSummary {
    size_t active_count = 0;
    size_t isolated_count = 0;
    float max_iso_v = 0.f;
    size_t max_iso_tri = std::numeric_limits<size_t>::max();
    std::vector<std::pair<size_t, float>> top_iso;
};

IsolatedTriangleSummary AnalyzeIsolatedPositiveTriangles(const std::vector<float>& vals,
                                                         const std::vector<std::vector<uint32_t>>& adjacency,
                                                         float eps,
                                                         size_t topk) {
    IsolatedTriangleSummary out;
    if (vals.size() != adjacency.size()) {
        return out;
    }

    for (size_t tri = 0; tri < vals.size(); ++tri) {
        const float v = vals[tri];
        if (!(v > eps)) {
            continue;
        }
        out.active_count++;

        bool has_pos_neighbor = false;
        for (uint32_t nb : adjacency[tri]) {
            if (nb < vals.size() && vals[nb] > eps) {
                has_pos_neighbor = true;
                break;
            }
        }
        if (has_pos_neighbor) {
            continue;
        }

        out.isolated_count++;
        if (v > out.max_iso_v) {
            out.max_iso_v = v;
            out.max_iso_tri = tri;
        }

        if (topk > 0) {
            out.top_iso.emplace_back(tri, v);
            std::sort(out.top_iso.begin(), out.top_iso.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            if (out.top_iso.size() > topk) {
                out.top_iso.resize(topk);
            }
        }
    }
    return out;
}

std::string FormatIsoTopList(const std::vector<std::pair<size_t, float>>& top_iso) {
    std::ostringstream oss;
    for (size_t i = 0; i < top_iso.size(); ++i) {
        if (i) {
            oss << ",";
        }
        oss << top_iso[i].first << ":" << top_iso[i].second;
    }
    return oss.str();
}

void WriteTriangleFrameCsv(const path& out_file, const std::vector<std::vector<float>>& frame_columns) {
    std::ofstream csv(out_file);
    if (!csv.is_open()) {
        throw std::runtime_error("Failed to open CSV for writing: " + out_file.string());
    }
    csv << std::setprecision(9);
    csv << "triangle_id";
    for (size_t f = 0; f < frame_columns.size(); ++f) {
        csv << ",frame_" << f;
    }
    csv << "\n";

    if (frame_columns.empty()) {
        return;
    }

    const size_t n_triangles = frame_columns.front().size();
    for (size_t f = 1; f < frame_columns.size(); ++f) {
        if (frame_columns[f].size() != n_triangles) {
            throw std::runtime_error("CSV frame column size mismatch.");
        }
    }

    for (size_t tri = 0; tri < n_triangles; ++tri) {
        csv << tri;
        for (const auto& col : frame_columns) {
            csv << "," << col[tri];
        }
        csv << "\n";
    }
}

std::vector<float> AverageTriangleValuesOverFrames(const std::vector<std::vector<float>>& frame_columns,
                                                   size_t frame_begin,
                                                   size_t frame_end_exclusive) {
    if (frame_begin >= frame_end_exclusive || frame_end_exclusive > frame_columns.size()) {
        throw std::runtime_error("Invalid frame averaging range.");
    }
    const size_t n_triangles = frame_columns.front().size();
    std::vector<float> avg_vals(n_triangles, 0.f);
    for (size_t f = frame_begin; f < frame_end_exclusive; ++f) {
        if (frame_columns[f].size() != n_triangles) {
            throw std::runtime_error("Frame size mismatch while averaging triangle values.");
        }
        for (size_t tri = 0; tri < n_triangles; ++tri) {
            avg_vals[tri] += frame_columns[f][tri];
        }
    }
    const float inv_frames = 1.f / static_cast<float>(frame_end_exclusive - frame_begin);
    for (auto& v : avg_vals) {
        v *= inv_frames;
    }
    return avg_vals;
}

struct ColorScaleConfig {
    bool use_manual = false;
    float manual_min = 0.f;
    float manual_max = 1.f;
};

struct RGBColor {
    int r;
    int g;
    int b;
};

RGBColor LerpRGB(const RGBColor& a, const RGBColor& b, float t) {
    const float clamped_t = std::min(1.f, std::max(0.f, t));
    const int r = static_cast<int>(std::lround(a.r + (b.r - a.r) * clamped_t));
    const int g = static_cast<int>(std::lround(a.g + (b.g - a.g) * clamped_t));
    const int bch = static_cast<int>(std::lround(a.b + (b.b - a.b) * clamped_t));
    return {r, g, bch};
}

RGBColor EvaluateFullSpectrumColor(float t) {
    const float clamped_t = std::min(1.f, std::max(0.f, t));
    constexpr std::array<float, 6> kStops = {0.f, 0.2f, 0.4f, 0.6f, 0.8f, 1.f};
    constexpr std::array<RGBColor, 6> kColors = {
        RGBColor{0, 0, 255},
        RGBColor{0, 255, 255},
        RGBColor{0, 255, 0},
        RGBColor{255, 255, 0},
        RGBColor{255, 165, 0},
        RGBColor{255, 0, 0},
    };

    for (size_t i = 0; i + 1 < kStops.size(); ++i) {
        if (clamped_t <= kStops[i + 1]) {
            const float span = kStops[i + 1] - kStops[i];
            const float local_t = (span > 0.f) ? (clamped_t - kStops[i]) / span : 0.f;
            return LerpRGB(kColors[i], kColors[i + 1], local_t);
        }
    }
    return kColors.back();
}

std::pair<float, float> ResolveColorScale(const std::vector<float>& tri_values, const ColorScaleConfig& cfg) {
    if (cfg.use_manual) {
        if (!(cfg.manual_max > cfg.manual_min)) {
            throw std::runtime_error("Invalid manual color scale.");
        }
        return {cfg.manual_min, cfg.manual_max};
    }

    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    for (float v : tri_values) {
        if (!std::isfinite(v)) {
            continue;
        }
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
    }

    if (!std::isfinite(min_val) || !std::isfinite(max_val)) {
        return {0.f, 1.f};
    }
    if (!(max_val > min_val)) {
        return {min_val, min_val + 1.f};
    }
    return {min_val, max_val};
}

void WriteTriangleColorPly(const path& out_file,
                           const std::vector<float3>& vertices,
                           const std::vector<int3>& faces,
                           const std::vector<float>& tri_values,
                           const ColorScaleConfig& cfg) {
    if (faces.size() != tri_values.size()) {
        throw std::runtime_error("PLY export size mismatch.");
    }

    std::ofstream ply(out_file);
    if (!ply.is_open()) {
        throw std::runtime_error("Failed to open PLY for writing: " + out_file.string());
    }

    const auto [scale_min, scale_max] = ResolveColorScale(tri_values, cfg);
    const float inv_span = 1.f / (scale_max - scale_min);

    ply << "ply\n";
    ply << "format ascii 1.0\n";
    ply << "element vertex " << vertices.size() << "\n";
    ply << "property float x\n";
    ply << "property float y\n";
    ply << "property float z\n";
    ply << "element face " << faces.size() << "\n";
    ply << "property list uchar int vertex_indices\n";
    ply << "property uchar red\n";
    ply << "property uchar green\n";
    ply << "property uchar blue\n";
    ply << "end_header\n";

    ply << std::setprecision(9);
    for (const auto& v : vertices) {
        ply << v.x << " " << v.y << " " << v.z << "\n";
    }

    for (size_t i = 0; i < faces.size(); ++i) {
        float value = tri_values[i];
        if (!std::isfinite(value)) {
            value = scale_min;
        }
        const float t = std::min(1.f, std::max(0.f, (value - scale_min) * inv_span));
        const RGBColor c = EvaluateFullSpectrumColor(t);
        ply << "3 " << faces[i].x << " " << faces[i].y << " " << faces[i].z << " "
            << c.r << " " << c.g << " " << c.b << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    DemoConfig cfg;
    TryGetEnvFloat("DEME_CRITICAL_TIME_END", cfg.time_end);
    TryGetEnvFloat("DEME_CRITICAL_STEP_SIZE", cfg.step_size);
    TryGetEnvUInt("DEME_CRITICAL_OUTPUT_FPS", cfg.output_fps);
    TryGetEnvInt("DEME_CRITICAL_CD_UPDATE_FREQ", cfg.cd_update_freq);
    cfg.disable_adaptive_cd_update =
        GetEnvBool("DEME_CRITICAL_DISABLE_ADAPTIVE_CD_UPDATE", cfg.disable_adaptive_cd_update);
    const bool write_files = GetEnvBool("DEME_CRITICAL_WRITE_FILES", false);
    cfg.write_contacts = GetEnvBool("DEME_CRITICAL_WRITE_CONTACTS", cfg.write_contacts);

    // For this closed convex workpiece, keep sph-tri history continuity fallback enabled by default.
    // This preserves tangential-history continuity through benign island/label churn.
    if (!std::getenv("DEME_SPH_TRI_HISTORY_FALLBACK")) {
        setenv("DEME_SPH_TRI_HISTORY_FALLBACK", "1", 0);
    }

    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.SetOutputContent(OUTPUT_CONTENT::XYZ | OUTPUT_CONTENT::VEL | OUTPUT_CONTENT::ABSV | OUTPUT_CONTENT::FAMILY);
    DEMSim.SetMeshOutputFormat(MESH_FORMAT::VTK);
    DEMSim.SetMeshUniversalContact(true);
    DEMSim.SetGravitationalAcceleration(make_float3(0.0f, 0.0f, cfg.gravity));
    DEMSim.SetInitTimeStep(cfg.step_size);
    DEMSim.SetCDUpdateFreq(cfg.cd_update_freq);
    if (cfg.disable_adaptive_cd_update) {
        DEMSim.DisableAdaptiveUpdateFreq();
    }
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.SetExpandSafetyAdder(std::max(0.5f, std::abs(cfg.drum_rpm) * 2.0f * PI_F / 60.0f * cfg.drum_radius));
    DEMSim.SetUseAngularVelocityMargin(true);
    DEMSim.InstructBoxDomainDimension(0.6f, 0.6f, 1.2f);

    auto mat_boundary = DEMSim.LoadMaterial(
        {{"E", cfg.mat_boundary_E}, {"nu", cfg.mat_boundary_nu}, {"CoR", cfg.mat_boundary_CoR},
         {"mu", cfg.mat_boundary_mu}, {"Crr", cfg.mat_boundary_Crr}});
    auto mat_particle = DEMSim.LoadMaterial(
        {{"E", cfg.mat_particle_E}, {"nu", cfg.mat_particle_nu}, {"CoR", cfg.mat_particle_CoR},
         {"mu", cfg.mat_particle_mu}, {"Crr", cfg.mat_particle_Crr}});
    auto mat_mesh = DEMSim.LoadMaterial(
        {{"E", cfg.mat_mesh_E}, {"nu", cfg.mat_mesh_nu}, {"CoR", cfg.mat_mesh_CoR},
         {"mu", cfg.mat_mesh_mu}, {"Crr", cfg.mat_mesh_Crr}});

    DEMSim.SetMaterialPropertyPair("mu", mat_particle, mat_boundary, cfg.pair_particle_boundary_mu);
    DEMSim.SetMaterialPropertyPair("CoR", mat_particle, mat_boundary, cfg.pair_particle_boundary_CoR);
    DEMSim.SetMaterialPropertyPair("Crr", mat_particle, mat_boundary, cfg.pair_particle_boundary_Crr);
    DEMSim.SetMaterialPropertyPair("mu", mat_particle, mat_mesh, cfg.pair_particle_mesh_mu);
    DEMSim.SetMaterialPropertyPair("CoR", mat_particle, mat_mesh, cfg.pair_particle_mesh_CoR);
    DEMSim.SetMaterialPropertyPair("Crr", mat_particle, mat_mesh, cfg.pair_particle_mesh_Crr);
    DEMSim.SetMaterialPropertyPair("mu", mat_boundary, mat_mesh, cfg.pair_boundary_mesh_mu);
    DEMSim.SetMaterialPropertyPair("CoR", mat_boundary, mat_mesh, cfg.pair_boundary_mesh_CoR);
    DEMSim.SetMaterialPropertyPair("Crr", mat_boundary, mat_mesh, cfg.pair_boundary_mesh_Crr);

    const unsigned int drum_family = 100;
    const unsigned int workpiece_family = 200;

    auto drum_shell = DEMSim.AddExternalObject();
    drum_shell->AddPlanarContactCylinder(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f),
                                         cfg.drum_radius, mat_boundary, ENTITY_NORMAL_INWARD);
    drum_shell->SetMass(1.0f);
    drum_shell->SetMOI(make_float3(1.0f, 1.0f, 1.0f));
    drum_shell->SetInitPos(make_float3(cfg.drum_center_x, cfg.drum_center_y, 0.5f * cfg.drum_height));
    drum_shell->SetFamily(drum_family);

    const float half_h = 0.5f * cfg.drum_height;
    const float safe_delta = std::min(std::max(1.0e-6f, cfg.wall_clearance), std::max(1.0e-6f, half_h - 1.0e-6f));
    auto drum_caps = DEMSim.AddExternalObject();
    drum_caps->AddPlane(make_float3(0.0f, 0.0f, half_h - safe_delta), make_float3(0.0f, 0.0f, -1.0f), mat_boundary);
    drum_caps->AddPlane(make_float3(0.0f, 0.0f, -half_h + safe_delta), make_float3(0.0f, 0.0f, 1.0f), mat_boundary);
    drum_caps->SetMass(1.0f);
    drum_caps->SetMOI(make_float3(1.0f, 1.0f, 1.0f));
    drum_caps->SetInitPos(make_float3(cfg.drum_center_x, cfg.drum_center_y, 0.5f * cfg.drum_height));
    drum_caps->SetFamily(drum_family);
    DEMSim.DisableContactBetweenFamilies(drum_family, drum_family);

    const float drum_omega = cfg.drum_rpm * 2.0f * PI_F / 60.0f;
    DEMSim.SetFamilyPrescribedAngVel(drum_family, "0", "0", to_string_with_precision(drum_omega));

    auto workpiece = DEMSim.AddWavefrontMeshObject(cfg.mesh_file.string(), mat_mesh);
    // The loaded workpiece is a closed convex sphere mesh; use convex sph-tri contact handling.
    workpiece->SetConvex(true);
    bool force_unique_patches = false;
    bool force_single_patch = false;
    if (const char* legacy_patch_env = std::getenv("DEME_CRITICAL_UNIQUE_TRI_PATCHES")) {
        const std::string v(legacy_patch_env);
        if (v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on") {
            force_unique_patches = true;
        } else if (v == "0" || v == "false" || v == "FALSE" || v == "no" || v == "off") {
            force_single_patch = true;
        }
    }
    const std::string patch_mode = GetEnvString("DEME_CRITICAL_PATCH_MODE", "unique");
    unsigned int target_patch_faces = 32;
    TryGetEnvUInt("DEME_CRITICAL_PATCH_FACES", target_patch_faces);

    {
        const size_t n_tri = workpiece->GetNumTriangles();
        const auto& faces = workpiece->GetIndicesVertexes();
        const auto& vertices = workpiece->GetCoordsVertices();
        std::vector<patchID_t> tri_patch_ids(n_tri, 0);

        std::string resolved_mode = patch_mode;
        if (force_unique_patches) {
            resolved_mode = "unique";
        } else if (force_single_patch) {
            resolved_mode = "single";
        }

        if (resolved_mode == "unique") {
            for (size_t i = 0; i < n_tri; i++) {
                tri_patch_ids[i] = static_cast<patchID_t>(i);
            }
        } else if (resolved_mode == "single") {
            std::fill(tri_patch_ids.begin(), tri_patch_ids.end(), static_cast<patchID_t>(0));
        } else {
            tri_patch_ids = BuildLocalTrianglePatchIDs(vertices, faces, static_cast<size_t>(target_patch_faces));
        }

        patchID_t max_patch_id = 0;
        for (patchID_t pid : tri_patch_ids) {
            if (pid > max_patch_id) {
                max_patch_id = pid;
            }
        }
        const unsigned int n_patches = static_cast<unsigned int>(max_patch_id + 1);
        workpiece->m_patch_ids = tri_patch_ids;
        workpiece->nPatches = n_patches;
        workpiece->patches_explicitly_set = true;
        workpiece->patch_locations_explicitly_set = false;
        workpiece->m_patch_locations.clear();
        std::cout << "Using workpiece patch mode '" << resolved_mode << "' (" << n_patches
                  << " patches over " << n_tri << " triangles"
                  << ", target_faces_per_patch=" << target_patch_faces << ").\n";
    }

    workpiece->Scale(cfg.mesh_scale);
    workpiece->SetMass(cfg.mesh_mass);
    const float mesh_moi = 0.4f * cfg.mesh_mass * cfg.mesh_scale * cfg.mesh_scale;
    workpiece->SetMOI(make_float3(mesh_moi, mesh_moi, mesh_moi));
    workpiece->SetFamily(workpiece_family);
    workpiece->SetInitPos(cfg.mesh_init_pos);
    workpiece->SetInitQuat(make_float4(0.0f, 0.0f, 0.0f, 1.0f));

    const std::string sink_vel_expr =
        "(t < " + to_string_with_precision(cfg.workpiece_sink_duration) + ") ? " +
        to_string_with_precision(-cfg.workpiece_sink_speed) + " : 0";
    DEMSim.SetFamilyPrescribedLinVel(workpiece_family, "0", "0", sink_vel_expr);

    const float sphere_mass = SphereMass(cfg.sphere_radius, cfg.sphere_density);
    auto sphere_type = DEMSim.LoadSphereType(sphere_mass, cfg.sphere_radius, mat_particle);

    auto seeds = BuildInitialSpherePositions(cfg);
    if (seeds.empty()) {
        throw std::runtime_error("No valid particle seeds remain after workpiece exclusion.");
    }

    auto spheres = DEMSim.AddClumps(sphere_type, seeds);
    spheres->SetFamily(1);
    auto sphere_tracker = DEMSim.Track(spheres);
    auto workpiece_tracker = DEMSim.Track(workpiece);

    if (GetEnvBool("DEME_CRITICAL_DISABLE_SPH_TRI", false)) {
        DEMSim.DisableContactBetweenFamilies(1, workpiece_family);
        std::cout << "Diagnostic: sphere-workpiece contact disabled via DEME_CRITICAL_DISABLE_SPH_TRI=1.\n";
    }

    DEMSim.TryDisableRuntimeCompiler();

    DEMSim.Initialize();

    const bodyID_t wp_owner = workpiece->owner;
    if (wp_owner == NULL_BODYID) {
        throw std::runtime_error("Workpiece mesh owner ID was not assigned after initialization.");
    }

    auto& wp_cached_mesh_tracking = DEMSim.GetCachedMesh(wp_owner);
    const auto& wp_faces_tracking = wp_cached_mesh_tracking->GetIndicesVertexes();
    const size_t wp_track_tri_count = wp_faces_tracking.size();
    DEMSim.SetTrianglePVTrackingOwners({wp_owner});

    const bool enable_triangle_pv_debug_outputs = true;
    const bool enable_v_isolation_debug = true;
    const bool enable_p_isolation_debug = true;
    const float v_isolation_eps = 0.0f;
    const float p_isolation_eps = 0.0f;
    const size_t v_isolation_topk = 12;
    const size_t p_isolation_topk = 12;
    const ColorScaleConfig ply_scale_P{false, 0.f, 1.f};
    const ColorScaleConfig ply_scale_V{false, 0.f, 1.f};
    const ColorScaleConfig ply_scale_PV{false, 0.f, 1.f};

    create_directories(cfg.out_dir);

    const float frame_dt = 1.0f / static_cast<float>(std::max(1u, cfg.output_fps));
    const unsigned int n_frames =
        std::max(1u, static_cast<unsigned int>(std::ceil(cfg.time_end / frame_dt)));

    std::vector<std::vector<float>> frame_cols_P;
    std::vector<std::vector<float>> frame_cols_V;
    std::vector<std::vector<float>> frame_cols_PV;

    const auto wp_adjacency =
        BuildTriangleAdjacency(wp_cached_mesh_tracking->GetCoordsVertices(), wp_faces_tracking);

    const path csv_out_P = cfg.out_dir / "workpiece_tri_P_avg.csv";
    const path csv_out_V = cfg.out_dir / "workpiece_tri_V_avg.csv";
    const path csv_out_PV = cfg.out_dir / "workpiece_tri_PxV_avg.csv";
    const path v_isolation_out = cfg.out_dir / "workpiece_tri_V_isolated.tsv";
    const path p_isolation_out = cfg.out_dir / "workpiece_tri_P_isolated.tsv";

    std::ofstream v_iso_stream;
    std::ofstream p_iso_stream;
    size_t total_iso_frames = 0;
    size_t total_iso_count = 0;
    size_t total_p_iso_frames = 0;
    size_t total_p_iso_count = 0;

    if (enable_triangle_pv_debug_outputs) {
        frame_cols_P.reserve(n_frames);
        frame_cols_V.reserve(n_frames);
        frame_cols_PV.reserve(n_frames);

        if (enable_v_isolation_debug) {
            v_iso_stream.open(v_isolation_out);
            if (!v_iso_stream.is_open()) {
                throw std::runtime_error("Failed to open V-isolation TSV for writing: " + v_isolation_out.string());
            }
            v_iso_stream << "frame\ttime_end_s\tactive_v_count\tisolated_v_count\tisolated_fraction\tmax_isolated_v"
                         << "\tmax_isolated_tri\ttop_isolated_tri_v\n";
        }
        if (enable_p_isolation_debug) {
            p_iso_stream.open(p_isolation_out);
            if (!p_iso_stream.is_open()) {
                throw std::runtime_error("Failed to open P-isolation TSV for writing: " + p_isolation_out.string());
            }
            p_iso_stream << "frame\ttime_end_s\tactive_p_count\tisolated_p_count\tisolated_fraction\tmax_isolated_p"
                         << "\tmax_isolated_tri\ttop_isolated_tri_p\n";
        }
    }

    std::cout << "Running DEMdemo_DEMdemo_CriticalSphTri\n"
              << "  particles       : " << seeds.size() << " / requested " << cfg.target_particles << "\n"
              << "  sphere diameter : " << 2.0f * cfg.sphere_radius * 1000.0f << " mm\n"
              << "  drum radius     : " << cfg.drum_radius * 1000.0f << " mm\n"
              << "  drum height     : " << cfg.drum_height * 1000.0f << " mm\n"
              << "  drum rpm        : " << cfg.drum_rpm << "\n"
              << "  workpiece mesh  : " << cfg.mesh_file << "\n"
              << "  tri debug out   : CSV + PLY + V/P-isolation TSV" << std::endl;

    const auto sample_metrics = [&]() {
        FrameMetrics m;
        const auto sphere_positions = sphere_tracker->Positions();
        const auto sphere_velocities = sphere_tracker->Velocities();
        const float3 workpiece_center = workpiece_tracker->Pos();
        const float workpiece_radius = cfg.mesh_scale;
        const float contact_radius = workpiece_radius + cfg.sphere_radius;

        for (size_t i = 0; i < sphere_positions.size(); ++i) {
            const float3& pos = sphere_positions[i];
            const float3& vel = sphere_velocities[i];
            const float speed = length(vel);
            const float dist_to_center = length(pos - workpiece_center);

            if (speed > m.max_v) {
                m.max_v = speed;
                m.max_v_id = static_cast<int>(i);
                m.max_v_dist_to_workpiece = dist_to_center;
                m.max_v_contact_gap = contact_radius - dist_to_center;
            }
            if (pos.z > m.max_z) {
                m.max_z = pos.z;
            }
            if (dist_to_center < workpiece_radius) {
                ++m.spheres_inside;
                if (m.first_inside_id < 0) {
                    m.first_inside_id = static_cast<int>(i);
                }
            }

            const float penetration = contact_radius - dist_to_center;
            if (penetration > m.max_penetration) {
                m.max_penetration = penetration;
                m.max_penetration_id = static_cast<int>(i);
            }
        }
        return m;
    };

    double dynamics_wall = 0.0;
    const auto sim_wall_start = std::chrono::high_resolution_clock::now();

    for (unsigned int frame = 0; frame < n_frames; ++frame) {
        const double t = frame * frame_dt;
        const FrameMetrics metrics = sample_metrics();

        std::cout << "Frame " << std::setw(4) << frame
                  << "  t=" << std::fixed << std::setprecision(4) << t
                  << "  max|v|=" << metrics.max_v
                  << "  max_v_sid=" << metrics.max_v_id
                  << "  max_v_dist_wp=" << metrics.max_v_dist_to_workpiece
                  << "  max_v_gap_m=" << metrics.max_v_contact_gap
                  << "  max_z=" << metrics.max_z
                  << "  max_pen_mm=" << (1000.0f * metrics.max_penetration)
                  << "  max_pen_sid=" << metrics.max_penetration_id
                  << "  spheres_inside=" << metrics.spheres_inside
                  << "  first_inside_sid=" << metrics.first_inside_id << std::endl;

        if (enable_triangle_pv_debug_outputs) {
            std::vector<float> frameP;
            std::vector<float> frameV;
            std::vector<float> framePV;
            if (!DEMSim.GetTrackedOwnerTrianglePV(wp_owner, frameP, frameV, framePV, true)) {
                throw std::runtime_error("Failed to read tracked per-triangle P/V/PxV data for workpiece owner.");
            }
            if (frameP.size() != wp_track_tri_count || frameV.size() != wp_track_tri_count ||
                framePV.size() != wp_track_tri_count) {
                throw std::runtime_error("Tracked triangle output size mismatch.");
            }

            if (enable_v_isolation_debug) {
                const auto iso =
                    AnalyzeIsolatedPositiveTriangles(frameV, wp_adjacency, v_isolation_eps, v_isolation_topk);
                const double time_end_s = t + frame_dt;
                const double iso_frac =
                    (iso.active_count > 0)
                        ? static_cast<double>(iso.isolated_count) / static_cast<double>(iso.active_count)
                        : 0.0;

                v_iso_stream << frame << '\t' << time_end_s << '\t' << iso.active_count << '\t'
                             << iso.isolated_count << '\t' << iso_frac << '\t' << iso.max_iso_v << '\t';

                if (iso.max_iso_tri == std::numeric_limits<size_t>::max()) {
                    v_iso_stream << -1;
                } else {
                    v_iso_stream << iso.max_iso_tri;
                }
                v_iso_stream << '\t' << FormatIsoTopList(iso.top_iso) << '\n';

                if (iso.isolated_count > 0) {
                    total_iso_frames++;
                    total_iso_count += iso.isolated_count;
                    std::cout << "  V-isolation: isolated " << iso.isolated_count
                              << " / " << iso.active_count
                              << "  max_iso_tri=" << iso.max_iso_tri
                              << "  max_iso_V=" << iso.max_iso_v
                              << "  top=" << FormatIsoTopList(iso.top_iso) << std::endl;
                }
            }
            if (enable_p_isolation_debug) {
                const auto isoP =
                    AnalyzeIsolatedPositiveTriangles(frameP, wp_adjacency, p_isolation_eps, p_isolation_topk);
                const double time_end_s = t + frame_dt;
                const double iso_frac =
                    (isoP.active_count > 0)
                        ? static_cast<double>(isoP.isolated_count) / static_cast<double>(isoP.active_count)
                        : 0.0;

                p_iso_stream << frame << '\t' << time_end_s << '\t' << isoP.active_count << '\t'
                             << isoP.isolated_count << '\t' << iso_frac << '\t' << isoP.max_iso_v << '\t';

                if (isoP.max_iso_tri == std::numeric_limits<size_t>::max()) {
                    p_iso_stream << -1;
                } else {
                    p_iso_stream << isoP.max_iso_tri;
                }
                p_iso_stream << '\t' << FormatIsoTopList(isoP.top_iso) << '\n';

                if (isoP.isolated_count > 0) {
                    total_p_iso_frames++;
                    total_p_iso_count += isoP.isolated_count;
                    std::cout << "  P-isolation: isolated " << isoP.isolated_count
                              << " / " << isoP.active_count
                              << "  max_iso_tri=" << isoP.max_iso_tri
                              << "  max_iso_P=" << isoP.max_iso_v
                              << "  top=" << FormatIsoTopList(isoP.top_iso) << std::endl;
                }
            }

            frame_cols_P.push_back(std::move(frameP));
            frame_cols_V.push_back(std::move(frameV));
            frame_cols_PV.push_back(std::move(framePV));
        }

        if (write_files) {
            char filename[128];
            std::snprintf(filename, sizeof(filename), "sphere_test_ui_equiv_%04u.csv", frame);
            DEMSim.WriteClumpFile(cfg.out_dir / filename);

            std::snprintf(filename, sizeof(filename), "sphere_test_ui_equiv_mesh_%04u.vtk", frame);
            DEMSim.WriteMeshFile(cfg.out_dir / filename);

            if (cfg.write_contacts) {
                std::snprintf(filename, sizeof(filename), "sphere_test_ui_equiv_contacts_%04u.csv", frame);
                DEMSim.WriteContactFile(cfg.out_dir / filename);
            }
        }

        const auto dyn_start = std::chrono::high_resolution_clock::now();
        DEMSim.DoDynamicsThenSync(frame_dt);
        const auto dyn_end = std::chrono::high_resolution_clock::now();
        dynamics_wall += std::chrono::duration_cast<std::chrono::duration<double>>(dyn_end - dyn_start).count();
    }

    const auto sim_wall_end = std::chrono::high_resolution_clock::now();
    const double total_wall =
        std::chrono::duration_cast<std::chrono::duration<double>>(sim_wall_end - sim_wall_start).count();

    if (enable_triangle_pv_debug_outputs && !frame_cols_P.empty()) {
        WriteTriangleFrameCsv(csv_out_P, frame_cols_P);
        WriteTriangleFrameCsv(csv_out_V, frame_cols_V);
        WriteTriangleFrameCsv(csv_out_PV, frame_cols_PV);

        auto avg_p_for_ply = AverageTriangleValuesOverFrames(frame_cols_P, 0, frame_cols_P.size());
        auto avg_v_for_ply = AverageTriangleValuesOverFrames(frame_cols_V, 0, frame_cols_V.size());
        auto avg_pv_for_ply = AverageTriangleValuesOverFrames(frame_cols_PV, 0, frame_cols_PV.size());

        const auto wp_nodes_global_dbg = DEMSim.GetMeshNodesGlobal(wp_owner);
        const auto& wp_faces_dbg = wp_cached_mesh_tracking->GetIndicesVertexes();

        WriteTriangleColorPly(cfg.out_dir / "workpiece_tri_P_avg.ply",
                              wp_nodes_global_dbg, wp_faces_dbg, avg_p_for_ply, ply_scale_P);
        WriteTriangleColorPly(cfg.out_dir / "workpiece_tri_V_avg.ply",
                              wp_nodes_global_dbg, wp_faces_dbg, avg_v_for_ply, ply_scale_V);
        WriteTriangleColorPly(cfg.out_dir / "workpiece_tri_PxV_avg.ply",
                              wp_nodes_global_dbg, wp_faces_dbg, avg_pv_for_ply, ply_scale_PV);

        if (enable_v_isolation_debug) {
            v_iso_stream.flush();
            std::cout << "Wrote V-isolation TSV to " << v_isolation_out
                      << " (frames_with_isolated=" << total_iso_frames
                      << ", isolated_total=" << total_iso_count << ").\n";
        }
        if (enable_p_isolation_debug) {
            p_iso_stream.flush();
            std::cout << "Wrote P-isolation TSV to " << p_isolation_out
                      << " (frames_with_isolated=" << total_p_iso_frames
                      << ", isolated_total=" << total_p_iso_count << ").\n";
        }

        std::cout << "Wrote triangle CSVs and averaged PLY files.\n";
    }

    std::cout << "Done. Total wall time: " << total_wall << " s\n"
              << "Dynamics-only wall time: " << dynamics_wall << " s\n"
              << "Output written to: " << cfg.out_dir << std::endl;

    DEMSim.ShowTimingStats();
    return 0;
}
