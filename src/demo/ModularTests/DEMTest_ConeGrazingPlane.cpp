//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Grazing-contact reference demo with 5 parallel variants in one solver run:
//   cone, cube-edge, cube-tip, sphere, cross
//
// All 5 bodies graze on their own laterally shifted plane patch in the same run so
// their local relative position to the plane cells stays identical to the former
// single-variant setup, while the bodies themselves do not overlap.
//
// Supported study modes:
//   1) Parallel 5-variant run on a single plane resolution.
//   2) Parallel triangle-size study from 2 to 8192 triangles (factor 8 here via the
//      predefined TRIANGLE_STUDY_COUNTS list).
//   3) Parallel kT study over the graze-speed scaling sweep.
//
// Notes:
//   - sphere uses DEMSim.LoadSphereType(...), i.e. no mesh.
//   - cone, cube, and cross use meshes.
//   - min/max comparisons and OK flags are still evaluated from the 1000 fps force
//     samples, independent of the reduced visual/print frame rate.
// =============================================================================

#include <core/ApiVersion.h>
#include <core/utils/ThreadManager.h>
#include <DEM/API.h>
#include <DEM/utils/HostSideHelpers.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using namespace std::filesystem;
using namespace deme;

namespace {

constexpr double PI_D = 3.14159265358979323846;
constexpr std::array<int, 5> TRIANGLE_STUDY_COUNTS = {2, 16, 128, 1024, 8192};
constexpr std::array<double, 9> DEFAULT_KT_SWEEP = {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0};

enum class ShapeVariant { CONE, CUBE, SPHERE, CROSS };
enum class CubeContactMode { EDGE, TIP };
enum class DemoVariant { CONE, CONE_STUDY, CUBE, CUBE_STUDY, SPHERE, SPHERE_STUDY, CROSS, CROSS_STUDY };

struct DemoConfig {
    DemoVariant variant = DemoVariant::CONE;
    ShapeVariant shape = ShapeVariant::CONE;
    bool triangle_study = false;
    CubeContactMode cube_mode = CubeContactMode::EDGE;
    path cross_mesh_path = GET_DATA_PATH() / "mesh/cross.stl";

    float mu = 0.3f;
    float CoR = 0.3f;
    float E = 1e8f;
    float nu = 0.3f;
    float density = 2600.0f;

    float base_body_size = 0.2f;       // cone h/r, cube edge, sphere diameter
    float base_plane_halfwidth = 1.0f; // 2 m wide plane
    float base_penetration = 0.005f;
    float base_graze_speed = 1.0f;

    float global_scale = 1.0f;
    float body_size_scale = 1.0f;
    float plane_size_scale = 1.0f;
    float penetration_scale = 1.0f;
    float speed_scale = 1.0f;

    float step_size = 2e-5f;
    float frame_time = 0.02f;
    float total_time = 0.1f;
    float csv_fps = 1000.0f;

    int single_run_plane_triangles = 8192;

    bool kT_study = false;
    int kt_top_n = 3;
    std::vector<double> kt_sweep = std::vector<double>(DEFAULT_KT_SWEEP.begin(), DEFAULT_KT_SWEEP.end());
};

struct AnalyticalReference {
    bool available = false;
    std::string model;
    double effective_modulus = 0.0;
    double normal_force = 0.0;
    double tangential_force = 0.0;
    double total_force = 0.0;
    double half_angle_rad = 0.0;
};

struct RunSummary {
    int plane_triangles = 0;
    int plane_cells_per_side = 0;
    double mean_force = 0.0;
    double min_force = 0.0;
    double max_force = 0.0;
    double stddev_force = 0.0;
    double mean_fz = 0.0;
    double min_fz = 0.0;
    double max_fz = 0.0;
    double force_span = 0.0;
    double fz_span = 0.0;
    bool force_ok = false;
    bool fz_ok = false;
    AnalyticalReference reference;
    std::string label;
    path out_dir;
};

struct ParallelRunSummary {
    int plane_triangles = 0;
    int plane_cells_per_side = 0;
    path out_dir;
    std::vector<RunSummary> case_summaries;
};

struct MovingBodyState {
    std::shared_ptr<DEMTracker> tracker;
    float mass = 0.0f;
    float reference_above_lowest_point = 0.0f;
    float init_x = 0.0f;
    float init_z = 0.0f;
    unsigned int family = 1;
};


struct BinaryStlStats {
    std::vector<float3> vertices;
    double signed_volume = 0.0;
    float3 min_corner = make_float3(0.f, 0.f, 0.f);
    float3 max_corner = make_float3(0.f, 0.f, 0.f);
};

struct LowestSupportCluster {
    float3 centroid = make_float3(0.f, 0.f, 0.f);
    int count = 0;
};

struct LowestSupportInfo {
    float min_z = 0.f;
    float z_tol = 0.f;
    float xy_merge_tol = 0.f;
    int num_low_vertices = 0;
    std::vector<LowestSupportCluster> clusters;
};

float3 Min3(const float3& a, const float3& b) {
    return make_float3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

float3 Max3(const float3& a, const float3& b) {
    return make_float3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

BinaryStlStats LoadBinaryStlStats(const path& stl_path) {
    std::ifstream in(stl_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open STL file: " + stl_path.string());
    }

    char header[80];
    in.read(header, sizeof(header));
    if (!in) {
        throw std::runtime_error("Failed to read STL header: " + stl_path.string());
    }

    uint32_t tri_count = 0;
    in.read(reinterpret_cast<char*>(&tri_count), sizeof(tri_count));
    if (!in) {
        throw std::runtime_error("Failed to read STL triangle count: " + stl_path.string());
    }

    BinaryStlStats stats;
    stats.vertices.reserve(static_cast<size_t>(tri_count) * 3);
    bool first_vertex = true;

    for (uint32_t i = 0; i < tri_count; ++i) {
        float normal[3];
        float v[9];
        uint16_t attr = 0;
        in.read(reinterpret_cast<char*>(normal), sizeof(normal));
        in.read(reinterpret_cast<char*>(v), sizeof(v));
        in.read(reinterpret_cast<char*>(&attr), sizeof(attr));
        if (!in) {
            throw std::runtime_error("Failed while reading STL facets: " + stl_path.string());
        }

        const float3 a = make_float3(v[0], v[1], v[2]);
        const float3 b = make_float3(v[3], v[4], v[5]);
        const float3 c = make_float3(v[6], v[7], v[8]);
        stats.vertices.push_back(a);
        stats.vertices.push_back(b);
        stats.vertices.push_back(c);

        if (first_vertex) {
            stats.min_corner = a;
            stats.max_corner = a;
            first_vertex = false;
        }
        stats.min_corner = Min3(stats.min_corner, a);
        stats.min_corner = Min3(stats.min_corner, b);
        stats.min_corner = Min3(stats.min_corner, c);
        stats.max_corner = Max3(stats.max_corner, a);
        stats.max_corner = Max3(stats.max_corner, b);
        stats.max_corner = Max3(stats.max_corner, c);

        const float3 cross_bc = make_float3(b.y * c.z - b.z * c.y,
                                            b.z * c.x - b.x * c.z,
                                            b.x * c.y - b.y * c.x);
        stats.signed_volume += (static_cast<double>(a.x) * cross_bc.x +
                                static_cast<double>(a.y) * cross_bc.y +
                                static_cast<double>(a.z) * cross_bc.z) / 6.0;
    }

    return stats;
}

float3 RotateByQuat(const float4& q_raw, const float3& v);

float MeshLowestPointOffset(const std::vector<float3>& vertices, float mesh_scale, const float4& quat) {
    float min_z = std::numeric_limits<float>::infinity();
    for (const auto& v : vertices) {
        const float3 p = make_float3(mesh_scale * v.x, mesh_scale * v.y, mesh_scale * v.z);
        const float3 pr = RotateByQuat(quat, p);
        min_z = std::min(min_z, pr.z);
    }
    return -min_z;
}


float DistXY(const float3& a, const float3& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

LowestSupportInfo AnalyzeLowestSupport(const std::vector<float3>& vertices, float mesh_scale, const float4& quat) {
    LowestSupportInfo info;
    info.z_tol = std::max(1.0e-6f, 5.0e-4f * mesh_scale);
    info.xy_merge_tol = std::max(1.0e-6f, 5.0e-2f * mesh_scale);

    std::vector<float3> rotated;
    rotated.reserve(vertices.size());
    info.min_z = std::numeric_limits<float>::infinity();
    for (const auto& v : vertices) {
        const float3 p = make_float3(mesh_scale * v.x, mesh_scale * v.y, mesh_scale * v.z);
        const float3 pr = RotateByQuat(quat, p);
        rotated.push_back(pr);
        info.min_z = std::min(info.min_z, pr.z);
    }

    for (const auto& p : rotated) {
        if (p.z > info.min_z + info.z_tol)
            continue;
        ++info.num_low_vertices;
        bool merged = false;
        for (auto& cluster : info.clusters) {
            if (DistXY(cluster.centroid, p) <= info.xy_merge_tol) {
                const float inv = 1.0f / static_cast<float>(cluster.count + 1);
                cluster.centroid = make_float3((cluster.centroid.x * cluster.count + p.x) * inv,
                                               (cluster.centroid.y * cluster.count + p.y) * inv,
                                               (cluster.centroid.z * cluster.count + p.z) * inv);
                ++cluster.count;
                merged = true;
                break;
            }
        }
        if (!merged) {
            LowestSupportCluster cluster;
            cluster.centroid = p;
            cluster.count = 1;
            info.clusters.push_back(cluster);
        }
    }

    return info;
}

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string VariantName(const DemoConfig& cfg) {
    switch (cfg.variant) {
        case DemoVariant::CONE:
            return "cone";
        case DemoVariant::CONE_STUDY:
            return "cone-study";
        case DemoVariant::CUBE:
            return "cube";
        case DemoVariant::CUBE_STUDY:
            return "cube-study";
        case DemoVariant::SPHERE:
            return "sphere";
        case DemoVariant::SPHERE_STUDY:
            return "sphere-study";
        case DemoVariant::CROSS:
            return "cross";
        case DemoVariant::CROSS_STUDY:
            return "cross-study";
    }
    return "unknown";
}

std::string CubeModeName(CubeContactMode mode) {
    switch (mode) {
        case CubeContactMode::EDGE:
            return "edge";
        case CubeContactMode::TIP:
            return "tip";
    }
    return "edge";
}

std::string CaseName(const DemoConfig& cfg) {
    std::string name = VariantName(cfg);
    if (cfg.shape == ShapeVariant::CUBE) {
        name += "-" + CubeModeName(cfg.cube_mode);
    }
    return name;
}

DemoVariant ParseVariantName(const std::string& value) {
    const std::string v = ToLower(value);
    if (v == "cone")
        return DemoVariant::CONE;
    if (v == "cone-study" || v == "cone_study")
        return DemoVariant::CONE_STUDY;
    if (v == "cube")
        return DemoVariant::CUBE;
    if (v == "cube-study" || v == "cube_study")
        return DemoVariant::CUBE_STUDY;
    if (v == "sphere")
        return DemoVariant::SPHERE;
    if (v == "sphere-study" || v == "sphere_study")
        return DemoVariant::SPHERE_STUDY;
    if (v == "cross" || v == "cross" || v == "cross")
        return DemoVariant::CROSS;
    if (v == "cross-study" || v == "cross_study" || v == "cross-study")
        return DemoVariant::CROSS_STUDY;
    throw std::runtime_error("Unknown --variant option: " + value);
}

void ApplyVariantToConfig(DemoConfig& cfg, DemoVariant variant) {
    cfg.variant = variant;
    switch (variant) {
        case DemoVariant::CONE:
            cfg.shape = ShapeVariant::CONE;
            cfg.triangle_study = false;
            break;
        case DemoVariant::CONE_STUDY:
            cfg.shape = ShapeVariant::CONE;
            cfg.triangle_study = true;
            break;
        case DemoVariant::CUBE:
            cfg.shape = ShapeVariant::CUBE;
            cfg.triangle_study = false;
            break;
        case DemoVariant::CUBE_STUDY:
            cfg.shape = ShapeVariant::CUBE;
            cfg.triangle_study = true;
            break;
        case DemoVariant::SPHERE:
            cfg.shape = ShapeVariant::SPHERE;
            cfg.triangle_study = false;
            break;
        case DemoVariant::SPHERE_STUDY:
            cfg.shape = ShapeVariant::SPHERE;
            cfg.triangle_study = true;
            break;
        case DemoVariant::CROSS:
            cfg.shape = ShapeVariant::CROSS;
            cfg.triangle_study = false;
            break;
        case DemoVariant::CROSS_STUDY:
            cfg.shape = ShapeVariant::CROSS;
            cfg.triangle_study = true;
            break;
    }
}

void RefreshVariantFromFlags(DemoConfig& cfg) {
    if (cfg.shape == ShapeVariant::CONE)
        cfg.variant = cfg.triangle_study ? DemoVariant::CONE_STUDY : DemoVariant::CONE;
    else if (cfg.shape == ShapeVariant::CUBE)
        cfg.variant = cfg.triangle_study ? DemoVariant::CUBE_STUDY : DemoVariant::CUBE;
    else if (cfg.shape == ShapeVariant::SPHERE)
        cfg.variant = cfg.triangle_study ? DemoVariant::SPHERE_STUDY : DemoVariant::SPHERE;
    else
        cfg.variant = cfg.triangle_study ? DemoVariant::CROSS_STUDY : DemoVariant::CROSS;
}

CubeContactMode ParseCubeMode(const std::string& value) {
    const std::string v = ToLower(value);
    if (v == "edge" || v == "kante")
        return CubeContactMode::EDGE;
    if (v == "tip" || v == "vertex" || v == "corner" || v == "spitze")
        return CubeContactMode::TIP;
    throw std::runtime_error("Unknown --cube-mode option: " + value);
}

bool ParseBool(const std::string& value) {
    const std::string v = ToLower(value);
    if (v == "1" || v == "true" || v == "yes" || v == "on")
        return true;
    if (v == "0" || v == "false" || v == "no" || v == "off")
        return false;
    throw std::runtime_error("Expected boolean value, got: " + value);
}

float ParseFloatArg(const std::string& arg_name, const std::string& value) {
    try {
        return std::stof(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid numeric value for " + arg_name + ": " + value);
    }
}

int ParseIntArg(const std::string& arg_name, const std::string& value) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer value for " + arg_name + ": " + value);
    }
}


std::vector<double> ParseDoubleListArg(const std::string& arg_name, const std::string& value) {
    std::vector<double> out;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        try {
            out.push_back(std::stod(token));
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid numeric list value for " + arg_name + ": " + token);
        }
    }
    if (out.empty()) {
        throw std::runtime_error(arg_name + " requires at least one numeric value.");
    }
    return out;
}

void PrintUsage(const char* exe_name) {
    std::cout << "Usage:\n"
              << "  " << exe_name
              << " [--triangle-study true|false] [--kT_study [true|false]] [other scaling flags]\n"
              << "\nThis demo always runs 5 variants in parallel:\n"
              << "  cone, cube-edge, cube-tip, sphere, cross\n"
              << "\nScaling flags:\n"
              << "  --global-scale <f>\n"
              << "  --body-size-scale <f>\n"
              << "  --plane-size-scale <f>\n"
              << "  --penetration-scale <f>\n"
              << "  --speed-scale <f>\n"
              << "\nAnalysis flags:\n"
              << "  --triangle-study true|false  run all 5 variants for each plane resolution\n"
              << "  --kT_study [true|false]      run all 5 variants for each kT sweep value\n"
              << "  --kt-top-n <n>               report top-N kT/case results\n"
              << "  --kt-sweep <v1,v2,...>       override default kT sweep\n"
              << "\nOther flags:\n"
              << "  --single-run-plane-triangles <n>\n"
              << "  --step-size <f> --frame-time <f> --total-time <f>\n"
              << "  --csv-fps <f>\n"
              << std::endl;
}

DemoConfig ParseArguments(int argc, char* argv[]) {
    DemoConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        }

        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value after " + name);
            }
            return argv[++i];
        };

        if (arg == "--triangle-study") {
            cfg.triangle_study = ParseBool(require_value(arg));
        } else if (arg == "--global-scale") {
            cfg.global_scale = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--body-size-scale") {
            cfg.body_size_scale = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--plane-size-scale") {
            cfg.plane_size_scale = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--penetration-scale") {
            cfg.penetration_scale = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--speed-scale") {
            cfg.speed_scale = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--step-size") {
            cfg.step_size = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--frame-time") {
            cfg.frame_time = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--total-time") {
            cfg.total_time = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--csv-fps") {
            cfg.csv_fps = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--single-run-plane-triangles") {
            cfg.single_run_plane_triangles = ParseIntArg(arg, require_value(arg));
        } else if (arg == "--kT_study") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg.kT_study = ParseBool(require_value(arg));
            } else {
                cfg.kT_study = true;
            }
        } else if (arg == "--kt-top-n") {
            cfg.kt_top_n = ParseIntArg(arg, require_value(arg));
        } else if (arg == "--kt-sweep") {
            cfg.kt_sweep = ParseDoubleListArg(arg, require_value(arg));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.kt_top_n <= 0) {
        throw std::runtime_error("--kt-top-n must be > 0.");
    }

    return cfg;
}

int PlaneCellsFromTriangleCount(int tri_count) {
    if (tri_count <= 0 || tri_count % 2 != 0) {
        throw std::runtime_error("Plane triangle count must be positive and even.");
    }
    const double cells_real = std::sqrt(static_cast<double>(tri_count) / 2.0);
    const int cells = static_cast<int>(std::llround(cells_real));
    if (2 * cells * cells != tri_count) {
        throw std::runtime_error("Plane triangle count must satisfy triangles = 2 * N^2.");
    }
    return cells;
}

DEMMeshConnected BuildPlaneMesh(int tri_count, float halfwidth, const std::shared_ptr<DEMMaterial>& mat) {
    const int cells = PlaneCellsFromTriangleCount(tri_count);

    DEMMeshConnected plane;
    plane.Clear();
    plane.m_vertices.reserve(static_cast<size_t>(cells + 1) * static_cast<size_t>(cells + 1));
    plane.m_face_v_indices.reserve(static_cast<size_t>(tri_count));

    for (int j = 0; j <= cells; ++j) {
        const float y = -halfwidth + (2.0f * halfwidth * static_cast<float>(j) / static_cast<float>(cells));
        for (int i = 0; i <= cells; ++i) {
            const float x = -halfwidth + (2.0f * halfwidth * static_cast<float>(i) / static_cast<float>(cells));
            plane.m_vertices.push_back(make_float3(x, y, 0.0f));
        }
    }

    auto vertex_index = [cells](int i, int j) -> int { return j * (cells + 1) + i; };

    for (int j = 0; j < cells; ++j) {
        for (int i = 0; i < cells; ++i) {
            const int v00 = vertex_index(i, j);
            const int v10 = vertex_index(i + 1, j);
            const int v01 = vertex_index(i, j + 1);
            const int v11 = vertex_index(i + 1, j + 1);

            plane.m_face_v_indices.push_back(make_int3(v00, v10, v11));
            plane.m_face_v_indices.push_back(make_int3(v00, v11, v01));
        }
    }

    plane.nTri = plane.m_face_v_indices.size();
    plane.SetMaterial(mat);
    plane.SetMass(1.0f);
    plane.SetMOI(make_float3(1.0f, 1.0f, 1.0f));
    plane.SetInitPos(make_float3(0.0f, 0.0f, 0.0f));
    plane.SetFamily(100);
    return plane;
}

double DEMEEffectiveModulus(float E, float nu) {
    return static_cast<double>(E) / (2.0 * (1.0 - static_cast<double>(nu) * static_cast<double>(nu)));
}

float Dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 Cross3(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float Length3(const float3& v) {
    return std::sqrt(Dot3(v, v));
}

float3 Normalize3(const float3& v) {
    const float len = Length3(v);
    if (len < 1e-12f)
        return make_float3(0.f, 0.f, 0.f);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

float4 NormalizeQuat(const float4& q) {
    const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (n < 1e-12f)
        return make_float4(0.f, 0.f, 0.f, 1.f);
    return make_float4(q.x / n, q.y / n, q.z / n, q.w / n);
}

float4 QuatMul(const float4& a, const float4& b) {
    return make_float4(a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                       a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                       a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                       a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float4 QuatAxisAngle(const float3& axis_raw, float angle_rad) {
    const float3 axis = Normalize3(axis_raw);
    const float s = std::sin(0.5f * angle_rad);
    const float c = std::cos(0.5f * angle_rad);
    return NormalizeQuat(make_float4(axis.x * s, axis.y * s, axis.z * s, c));
}

float4 QuatFromTwoVectors(const float3& from_raw, const float3& to_raw) {
    const float3 from = Normalize3(from_raw);
    const float3 to = Normalize3(to_raw);
    const float dot_v = Dot3(from, to);

    if (dot_v > 1.f - 1e-6f) {
        return make_float4(0.f, 0.f, 0.f, 1.f);
    }

    if (dot_v < -1.f + 1e-6f) {
        float3 axis = Cross3(from, make_float3(1.f, 0.f, 0.f));
        if (Length3(axis) < 1e-6f) {
            axis = Cross3(from, make_float3(0.f, 1.f, 0.f));
        }
        return QuatAxisAngle(axis, static_cast<float>(PI_D));
    }

    const float3 axis = Cross3(from, to);
    const float s = std::sqrt((1.f + dot_v) * 2.f);
    const float inv_s = 1.f / s;
    return NormalizeQuat(make_float4(axis.x * inv_s, axis.y * inv_s, axis.z * inv_s, 0.5f * s));
}

float3 RotateByQuat(const float4& q_raw, const float3& v) {
    const float4 q = NormalizeQuat(q_raw);
    const float4 vq = make_float4(v.x, v.y, v.z, 0.f);
    const float4 q_conj = make_float4(-q.x, -q.y, -q.z, q.w);
    const float4 rq = QuatMul(QuatMul(q, vq), q_conj);
    return make_float3(rq.x, rq.y, rq.z);
}

float CubeLowestPointOffset(float cube_edge, const float4& quat) {
    float min_z = std::numeric_limits<float>::infinity();
    for (int sx : {-1, 1}) {
        for (int sy : {-1, 1}) {
            for (int sz : {-1, 1}) {
                const float3 p = make_float3(0.5f * cube_edge * static_cast<float>(sx),
                                             0.5f * cube_edge * static_cast<float>(sy),
                                             0.5f * cube_edge * static_cast<float>(sz));
                const float3 pr = RotateByQuat(quat, p);
                min_z = std::min(min_z, pr.z);
            }
        }
    }
    return -min_z;
}

AnalyticalReference BuildReferenceCone(float E, float nu, float mu, float radius, float height, float penetration) {
    AnalyticalReference ref;
    ref.available = true;
    ref.model = "DEME mesh area-proxy Hertz for conical mesh on plane + Coulomb friction";
    ref.half_angle_rad = std::atan(static_cast<double>(radius) / static_cast<double>(height));
    ref.effective_modulus = DEMEEffectiveModulus(E, nu);
    ref.normal_force = (4.0 / 3.0) * ref.effective_modulus * std::tan(ref.half_angle_rad) *
                       static_cast<double>(penetration) * static_cast<double>(penetration);
    ref.tangential_force = static_cast<double>(mu) * ref.normal_force;
    ref.total_force = std::sqrt(ref.normal_force * ref.normal_force + ref.tangential_force * ref.tangential_force);
    return ref;
}

AnalyticalReference BuildReferenceSphere(float E, float nu, float mu, float radius, float penetration) {
    AnalyticalReference ref;
    ref.available = true;
    ref.model = "DEME Hertz sphere on plane + Coulomb friction (identical materials)";
    ref.effective_modulus = DEMEEffectiveModulus(E, nu);
    ref.normal_force = (4.0 / 3.0) * ref.effective_modulus * std::sqrt(static_cast<double>(radius)) *
                       std::pow(static_cast<double>(penetration), 1.5);
    ref.tangential_force = static_cast<double>(mu) * ref.normal_force;
    ref.total_force = std::sqrt(ref.normal_force * ref.normal_force + ref.tangential_force * ref.tangential_force);
    return ref;
}

AnalyticalReference BuildReferenceCubeEdge(float E, float nu, float mu, float cube_edge, float penetration) {
    AnalyticalReference ref;
    ref.available = true;
    ref.model = "DEME mesh area-proxy Hertz for finite 90deg wedge edge + Coulomb friction";
    ref.half_angle_rad = PI_D / 4.0;
    ref.effective_modulus = DEMEEffectiveModulus(E, nu);

    const double overlap_area = 2.0 * static_cast<double>(cube_edge) * static_cast<double>(penetration);
    const double contact_radius = std::sqrt(overlap_area / PI_D);
    ref.normal_force = (4.0 / 3.0) * ref.effective_modulus * contact_radius * static_cast<double>(penetration);
    ref.tangential_force = static_cast<double>(mu) * ref.normal_force;
    ref.total_force = std::sqrt(ref.normal_force * ref.normal_force + ref.tangential_force * ref.tangential_force);
    return ref;
}

AnalyticalReference BuildReferenceCubeTip(float E, float nu, float mu, float penetration) {
    AnalyticalReference ref;
    ref.available = true;
    ref.model = "DEME mesh area-proxy Hertz for cube vertex (trihedral tip) + Coulomb friction";
    ref.effective_modulus = DEMEEffectiveModulus(E, nu);

    const double area_coeff = 1.5 * std::sqrt(3.0);  // A = (3*sqrt(3)/2) * delta^2
    const double contact_radius = std::sqrt(area_coeff / PI_D) * static_cast<double>(penetration);
    ref.half_angle_rad = std::atan(std::sqrt(area_coeff / PI_D));  // equivalent cone half-angle for same A(h)
    ref.normal_force = (4.0 / 3.0) * ref.effective_modulus * contact_radius * static_cast<double>(penetration);
    ref.tangential_force = static_cast<double>(mu) * ref.normal_force;
    ref.total_force = std::sqrt(ref.normal_force * ref.normal_force + ref.tangential_force * ref.tangential_force);
    return ref;
}

AnalyticalReference BuildReferenceCrossTwoTips(float E, float nu, float mu, float penetration) {
    AnalyticalReference single_tip = BuildReferenceCubeTip(E, nu, mu, penetration);
    AnalyticalReference ref;
    ref.available = true;
    ref.model = "2x DEME mesh area-proxy Hertz for cube vertex (trihedral tip), used as twin-tip proxy for cross STL";
    ref.effective_modulus = single_tip.effective_modulus;
    ref.half_angle_rad = single_tip.half_angle_rad;
    ref.normal_force = 2.0 * single_tip.normal_force;
    ref.tangential_force = 2.0 * single_tip.tangential_force;
    ref.total_force = 2.0 * single_tip.total_force;
    return ref;
}

AnalyticalReference BuildReferenceForShape(const DemoConfig& cfg, float body_size, float penetration) {
    switch (cfg.shape) {
        case ShapeVariant::CONE:
            return BuildReferenceCone(cfg.E, cfg.nu, cfg.mu, body_size, body_size, penetration);
        case ShapeVariant::SPHERE:
            return BuildReferenceSphere(cfg.E, cfg.nu, cfg.mu, 0.5f * body_size, penetration);
        case ShapeVariant::CUBE:
            if (cfg.cube_mode == CubeContactMode::EDGE)
                return BuildReferenceCubeEdge(cfg.E, cfg.nu, cfg.mu, body_size, penetration);
            return BuildReferenceCubeTip(cfg.E, cfg.nu, cfg.mu, penetration);
        case ShapeVariant::CROSS:
            return BuildReferenceCrossTwoTips(cfg.E, cfg.nu, cfg.mu, penetration);
    }
    AnalyticalReference ref;
    ref.available = false;
    ref.model = "Unsupported shape";
    return ref;
}

void PrintReference(const AnalyticalReference& ref) {
    if (!ref.available) {
        std::cout << "Analytical reference: not reported (" << ref.model << ")" << std::endl;
        return;
    }

    std::cout << "Analytical reference model: " << ref.model << std::endl;
    std::cout << "  E_eff:   " << ref.effective_modulus << " Pa" << std::endl;
    if (ref.half_angle_rad > 0.0) {
        std::cout << "  alpha:   " << ref.half_angle_rad * 180.0 / PI_D << " deg" << std::endl;
    }
    std::cout << "  F_n:     " << ref.normal_force << " N" << std::endl;
    std::cout << "  F_t:     " << ref.tangential_force << " N" << std::endl;
    std::cout << "  |F|:     " << ref.total_force << " N" << std::endl;
}

MovingBodyState AddMovingBody(DEMSolver& DEMSim,
                              const DemoConfig& cfg,
                              const std::shared_ptr<DEMMaterial>& mat,
                              float body_size,
                              float penetration,
                              float init_x,
                              float init_y,
                              unsigned int requested_family) {
    MovingBodyState state;

    if (cfg.shape == ShapeVariant::CONE) {
        const float cone_radius = body_size;
        const float cone_height = body_size;
        const float cone_volume = (1.0f / 3.0f) * static_cast<float>(PI_D) * cone_radius * cone_radius * cone_height;
        const float cone_mass = cfg.density * cone_volume;
        const float cone_Ixy = 3.0f * cone_mass / 20.0f * cone_radius * cone_radius +
                               3.0f * cone_mass / 80.0f * cone_height * cone_height;
        const float cone_Iz = 3.0f * cone_mass / 10.0f * cone_radius * cone_radius;
        const float centroid_above_tip = 0.75f * cone_height;
        const float init_z = -penetration + centroid_above_tip;

        auto cone = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cone.obj").string(), mat);
        cone->InformCentroidPrincipal(make_float3(0.f, 0.f, 0.75f), make_float4(0.f, 0.f, 0.f, 1.f));
        cone->Scale(body_size);
        cone->SetMass(cone_mass);
        cone->SetMOI(make_float3(cone_Ixy, cone_Ixy, cone_Iz));
        cone->SetInitPos(make_float3(init_x, init_y, init_z));
        cone->SetFamily(requested_family);

        state.tracker = DEMSim.Track(cone);
        state.mass = cone_mass;
        state.reference_above_lowest_point = centroid_above_tip;
        state.init_x = init_x;
        state.init_z = init_z;
        state.family = requested_family;
        return state;
    }

    if (cfg.shape == ShapeVariant::CUBE) {
        const float cube_edge = body_size;
        const float cube_mass = cfg.density * cube_edge * cube_edge * cube_edge;
        const float cube_I = cube_mass * cube_edge * cube_edge / 6.0f;

        float4 cube_quat = make_float4(0.f, 0.f, 0.f, 1.f);
        if (cfg.cube_mode == CubeContactMode::EDGE) {
            cube_quat = QuatAxisAngle(make_float3(0.f, 1.f, 0.f), static_cast<float>(PI_D / 4.0));
        } else {
            cube_quat = QuatFromTwoVectors(make_float3(-1.f, -1.f, -1.f), make_float3(0.f, 0.f, -1.f));
        }

        const float centroid_above_lowest = CubeLowestPointOffset(cube_edge, cube_quat);
        const float init_z = -penetration + centroid_above_lowest;

        auto cube = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cube.obj").string(), mat);
        cube->Scale(body_size);
        cube->SetMass(cube_mass);
        cube->SetMOI(make_float3(cube_I, cube_I, cube_I));
        cube->SetInitQuat(cube_quat);
        cube->SetInitPos(make_float3(init_x, init_y, init_z));
        cube->SetFamily(requested_family);

        state.tracker = DEMSim.Track(cube);
        state.mass = cube_mass;
        state.reference_above_lowest_point = centroid_above_lowest;
        state.init_x = init_x;
        state.init_z = init_z;
        state.family = requested_family;
        return state;
    }

    if (cfg.shape == ShapeVariant::CROSS) {
        const BinaryStlStats cross_stats = LoadBinaryStlStats(cfg.cross_mesh_path);
        const float mesh_scale = 0.005f * cfg.global_scale * cfg.body_size_scale;

        // 45 deg in-plane rotation plus 45 deg tilt about Y so two opposite tips become the lowest support points.
        const float4 cross_quat = NormalizeQuat(
            QuatMul(QuatAxisAngle(make_float3(0.f, 1.f, 0.f), static_cast<float>(PI_D / 4.0)),
                    QuatAxisAngle(make_float3(0.f, 0.f, 1.f), static_cast<float>(PI_D / 4.0))));
        const LowestSupportInfo support_info = AnalyzeLowestSupport(cross_stats.vertices, mesh_scale, cross_quat);
        const float centroid_above_lowest = MeshLowestPointOffset(cross_stats.vertices, mesh_scale, cross_quat);
        const float init_z = -penetration + centroid_above_lowest;

        const float volume_m3 = std::abs(static_cast<float>(cross_stats.signed_volume)) * mesh_scale * mesh_scale * mesh_scale;
        const float cross_mass = std::max(cfg.density * volume_m3, 1.0e-6f);
        const float x_extent = mesh_scale * (cross_stats.max_corner.x - cross_stats.min_corner.x);
        const float y_extent = mesh_scale * (cross_stats.max_corner.y - cross_stats.min_corner.y);
        const float z_extent = mesh_scale * (cross_stats.max_corner.z - cross_stats.min_corner.z);
        const float moi_x = cross_mass * (y_extent * y_extent + z_extent * z_extent) / 12.0f;
        const float moi_y = cross_mass * (x_extent * x_extent + z_extent * z_extent) / 12.0f;
        const float moi_z = cross_mass * (x_extent * x_extent + y_extent * y_extent) / 12.0f;

        auto cross = DEMSim.AddWavefrontMeshObject(cfg.cross_mesh_path.string(), mat);
        cross->Scale(mesh_scale);
        cross->SetMass(cross_mass);
        cross->SetMOI(make_float3(moi_x, moi_y, moi_z));
        cross->SetInitQuat(cross_quat);
        cross->SetInitPos(make_float3(init_x, init_y, init_z));
        cross->SetFamily(requested_family);
        // cross->SetConvex(true);

        std::cout << "Cross support diagnostic: lowest-z band has " << support_info.num_low_vertices
                  << " vertices grouped into " << support_info.clusters.size() << " XY-separated support cluster(s).\n";
        if (support_info.clusters.size() == 2) {
            const float sep = DistXY(support_info.clusters[0].centroid, support_info.clusters[1].centroid);
            std::cout << "Cross support diagnostic: OK, geometry resolves to 2 separated lowest tips."
                      << " tip-tip spacing = " << sep << " m\n";
        } else {
            std::cout << "Cross support diagnostic: WARNING, geometry does not resolve to exactly 2 separated lowest tips"
                      << " before the DEM solve.\n";
        }

        state.tracker = DEMSim.Track(cross);
        state.mass = cross_mass;
        state.reference_above_lowest_point = centroid_above_lowest;
        state.init_x = init_x;
        state.init_z = init_z;
        state.family = requested_family;
        return state;
    }

    const float sphere_radius = 0.5f * body_size;
    const float sphere_volume = 4.0f / 3.0f * static_cast<float>(PI_D) * sphere_radius * sphere_radius * sphere_radius;
    const float sphere_mass = cfg.density * sphere_volume;
    const float init_z = sphere_radius - penetration;

    auto sphere_type = DEMSim.LoadSphereType(sphere_mass, sphere_radius, mat);
    auto sphere_batch = DEMSim.AddClumps(sphere_type, make_float3(init_x, init_y, init_z));
    auto sphere_tracker = DEMSim.Track(sphere_batch);

    state.tracker = sphere_tracker;
    state.mass = sphere_mass;
    state.reference_above_lowest_point = sphere_radius;
    state.init_x = init_x;
    state.init_z = init_z;
    state.family = 0;
    return state;
}


constexpr double OK_TOL = 1.0e-10;

bool MinMaxOk(double min_val, double max_val, double tol = OK_TOL) {
    return std::abs(max_val - min_val) <= tol;
}

std::string OkString(bool value) {
    return value ? "OK" : "NOT_OK";
}

std::vector<DemoConfig> BuildParallelCaseConfigs(const DemoConfig& base_cfg) {
    std::vector<DemoConfig> cases;

    DemoConfig cone_cfg = base_cfg;
    ApplyVariantToConfig(cone_cfg, DemoVariant::CONE);
    cases.push_back(cone_cfg);

    DemoConfig cube_edge_cfg = base_cfg;
    ApplyVariantToConfig(cube_edge_cfg, DemoVariant::CUBE);
    cube_edge_cfg.cube_mode = CubeContactMode::EDGE;
    cases.push_back(cube_edge_cfg);

    DemoConfig cube_tip_cfg = base_cfg;
    ApplyVariantToConfig(cube_tip_cfg, DemoVariant::CUBE);
    cube_tip_cfg.cube_mode = CubeContactMode::TIP;
    cases.push_back(cube_tip_cfg);

    DemoConfig sphere_cfg = base_cfg;
    ApplyVariantToConfig(sphere_cfg, DemoVariant::SPHERE);
    cases.push_back(sphere_cfg);

    DemoConfig cross_cfg = base_cfg;
    ApplyVariantToConfig(cross_cfg, DemoVariant::CROSS);
    cases.push_back(cross_cfg);

    return cases;
}

void FinalizeSummaryFromTraces(RunSummary& summary,
                               const std::vector<double>& force_mags,
                               const std::vector<double>& force_z_components) {
    if (force_mags.empty() || force_z_components.empty()) {
        throw std::runtime_error("No force data collected for summary " + summary.label);
    }

    summary.mean_force = 0.0;
    summary.min_force = force_mags.front();
    summary.max_force = force_mags.front();
    summary.mean_fz = 0.0;
    summary.min_fz = force_z_components.front();
    summary.max_fz = force_z_components.front();

    for (size_t i = 0; i < force_mags.size(); ++i) {
        summary.mean_force += force_mags[i];
        summary.mean_fz += force_z_components[i];
        summary.min_force = std::min(summary.min_force, force_mags[i]);
        summary.max_force = std::max(summary.max_force, force_mags[i]);
        summary.min_fz = std::min(summary.min_fz, force_z_components[i]);
        summary.max_fz = std::max(summary.max_fz, force_z_components[i]);
    }

    summary.mean_force /= static_cast<double>(force_mags.size());
    summary.mean_fz /= static_cast<double>(force_z_components.size());

    double variance = 0.0;
    for (double f : force_mags) {
        const double d = f - summary.mean_force;
        variance += d * d;
    }
    summary.stddev_force = std::sqrt(variance / static_cast<double>(force_mags.size()));

    summary.force_span = summary.max_force - summary.min_force;
    summary.fz_span = summary.max_fz - summary.min_fz;
    summary.force_ok = MinMaxOk(summary.min_force, summary.max_force);
    summary.fz_ok = MinMaxOk(summary.min_fz, summary.max_fz);
}

void PrintCaseHeader(const DemoConfig& cfg, const AnalyticalReference& ref, float y_offset) {
    std::cout << "  Case:                " << CaseName(cfg) << "\n";
    std::cout << "    Lane y-offset:     " << y_offset << " m\n";
    PrintReference(ref);
}

void PrintCaseSummary(const RunSummary& summary) {
    std::cout << "\n=== Statistics for " << summary.label << " ===" << std::endl;
    std::cout << "  Mean |F|:   " << summary.mean_force << " N" << std::endl;
    std::cout << "  Min |F|:    " << summary.min_force << " N" << std::endl;
    std::cout << "  Max |F|:    " << summary.max_force << " N" << std::endl;
    std::cout << "  Span |F|:   " << summary.force_span << " N" << std::endl;
    std::cout << "  StdDev |F|: " << summary.stddev_force << " N" << std::endl;
    std::cout << "  Mean Fz:    " << summary.mean_fz << " N" << std::endl;
    std::cout << "  Min Fz:     " << summary.min_fz << " N" << std::endl;
    std::cout << "  Max Fz:     " << summary.max_fz << " N" << std::endl;
    std::cout << "  Span Fz:    " << summary.fz_span << " N" << std::endl;
    std::cout << "  F  = " << OkString(summary.force_ok) << " (|max-min| <= " << OK_TOL << ")" << std::endl;
    std::cout << "  Fz = " << OkString(summary.fz_ok) << " (|max-min| <= " << OK_TOL << ")" << std::endl;

    if (summary.reference.available) {
        const double rel_err_total =
            100.0 * (summary.mean_force - summary.reference.total_force) / summary.reference.total_force;
        const double rel_err_fz =
            100.0 * (summary.mean_fz - summary.reference.normal_force) / summary.reference.normal_force;
        std::cout << "  Rel. error |F| vs reference: " << rel_err_total << " %" << std::endl;
        std::cout << "  Rel. error Fz vs reference:  " << rel_err_fz << " %" << std::endl;
    }
}

struct ParallelCaseRuntime {
    DemoConfig cfg;
    MovingBodyState body;
    RunSummary summary;
    float y_offset = 0.0f;
    unsigned int plane_family = 0;
};

ParallelRunSummary RunParallelCaseSet(const DemoConfig& cfg, int plane_triangles, const path& root_out_dir) {
    const float body_size = cfg.base_body_size * cfg.global_scale * cfg.body_size_scale;
    const float plane_halfwidth = cfg.base_plane_halfwidth * cfg.global_scale * cfg.plane_size_scale;
    const float penetration = cfg.base_penetration * cfg.global_scale * cfg.penetration_scale;
    const float graze_speed = cfg.base_graze_speed * cfg.speed_scale;
    float expand_factor = 1.0f;
    if (const char* env_expand = std::getenv("DEM_GRAZE_EXPAND_FACTOR")) {
        try {
            expand_factor = std::stof(env_expand);
        } catch (const std::exception&) {
            expand_factor = 1.0f;
        }
    }

    if (cfg.csv_fps <= 0.0f) {
        throw std::runtime_error("csv_fps must be > 0.");
    }

    const std::vector<DemoConfig> case_cfgs = BuildParallelCaseConfigs(cfg);
    const int n_cases = static_cast<int>(case_cfgs.size());
    const int cells = PlaneCellsFromTriangleCount(plane_triangles);
    const float lane_margin = std::max(4.0f * body_size, 0.25f * plane_halfwidth);
    const float lane_pitch = 2.0f * plane_halfwidth + lane_margin;
    const float init_x = -body_size;
    const float max_abs_lane_y = 0.5f * static_cast<float>(n_cases - 1) * lane_pitch;
    const float domain_y = std::max(30.0f, 2.0f * (max_abs_lane_y + plane_halfwidth + body_size) + 2.0f);
    const float domain_x =
        std::max(30.0f, 4.0f * plane_halfwidth + graze_speed * cfg.total_time + 4.0f * body_size + 2.0f);

    ParallelRunSummary run_summary;
    run_summary.plane_triangles = plane_triangles;
    run_summary.plane_cells_per_side = cells;
    run_summary.out_dir = root_out_dir / ("parallel_T" + std::to_string(plane_triangles));

    std::error_code dir_ec;
    create_directories(run_summary.out_dir, dir_ec);
    if (dir_ec || !is_directory(run_summary.out_dir)) {
        throw std::runtime_error("Failed to create output directory: " + run_summary.out_dir.string());
    }

    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.InstructBoxDomainDimension(domain_x, domain_y, 5.0f);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, 0));
    DEMSim.SetMeshUniversalContact(true);
    DEMSim.SetExpandSafetyType("auto");
    DEMSim.SetExpandSafetyAdder(graze_speed * expand_factor);

    auto mat = DEMSim.LoadMaterial({{"E", cfg.E}, {"nu", cfg.nu}, {"CoR", cfg.CoR}, {"mu", cfg.mu}, {"Crr", 0.0f}});

    std::vector<ParallelCaseRuntime> runtimes;
    runtimes.reserve(case_cfgs.size());

    std::cout << "=====================================================\n";
    std::cout << "Parallel run with 5 variants in one solver\n";
    std::cout << "Plane triangles per lane: " << plane_triangles << " (" << cells << " x " << cells << " cells)\n";
    std::cout << "Body size:                " << body_size << " m\n";
    std::cout << "Plane halfwidth:          " << plane_halfwidth << " m\n";
    std::cout << "Penetration:              " << penetration << " m\n";
    std::cout << "Graze speed:              " << graze_speed << " m/s\n";
    std::cout << "Expand factor:            " << expand_factor << " (-)\n";
    std::cout << "Lane pitch:               " << lane_pitch << " m\n";
    std::cout << "Domain (x,y,z):           " << domain_x << ", " << domain_y << ", 5 m\n";
    std::cout << "Total graze dist:         " << graze_speed * cfg.total_time << " m\n";
    std::cout << "=====================================================\n";

    for (size_t i = 0; i < case_cfgs.size(); ++i) {
        const DemoConfig& case_cfg = case_cfgs[i];
        const float y_offset = (static_cast<float>(i) - 0.5f * static_cast<float>(n_cases - 1)) * lane_pitch;
        const unsigned int plane_family = 100u + static_cast<unsigned int>(i);
        const unsigned int requested_body_family =
            (case_cfg.shape == ShapeVariant::SPHERE) ? 0u : 1u + static_cast<unsigned int>(i);

        DEMMeshConnected plane_mesh = BuildPlaneMesh(plane_triangles, plane_halfwidth, mat);
        auto plane = DEMSim.AddWavefrontMeshObject(plane_mesh);
        plane->SetFamily(plane_family);
        plane->SetInitPos(make_float3(0.0f, y_offset, 0.0f));
        DEMSim.SetFamilyFixed(plane_family);

        ParallelCaseRuntime runtime;
        runtime.cfg = case_cfg;
        runtime.y_offset = y_offset;
        runtime.plane_family = plane_family;
        runtime.body =
            AddMovingBody(DEMSim, case_cfg, mat, body_size, penetration, init_x, y_offset, requested_body_family);
        runtime.summary.plane_triangles = plane_triangles;
        runtime.summary.plane_cells_per_side = cells;
        runtime.summary.label = CaseName(case_cfg) + "_T" + std::to_string(plane_triangles);
        runtime.summary.out_dir = run_summary.out_dir / runtime.summary.label;
        runtime.summary.reference = BuildReferenceForShape(case_cfg, body_size, penetration);

        create_directories(runtime.summary.out_dir, dir_ec);
        if (dir_ec || !is_directory(runtime.summary.out_dir)) {
            throw std::runtime_error("Failed to create output directory: " + runtime.summary.out_dir.string());
        }
        dir_ec.clear();

        DEMSim.SetFamilyPrescribedLinVel(runtime.body.family, to_string_with_precision(graze_speed), "0", "0");
        DEMSim.SetFamilyPrescribedAngVel(runtime.body.family, "0", "0", "0");

        PrintCaseHeader(case_cfg, runtime.summary.reference, y_offset);
        runtimes.push_back(std::move(runtime));
    }
    std::cout << "=====================================================\n";
    
    DEMSim.TryDisableRuntimeCompiler(true);

    DEMSim.SetInitTimeStep(cfg.step_size);
    DEMSim.Initialize();
    for (auto& runtime : runtimes) {
        runtime.body.tracker->SetVel(make_float3(graze_speed, 0.f, 0.f));
    }

    const double csv_dt = 1.0 / static_cast<double>(cfg.csv_fps);
    const int n_csv_samples = static_cast<int>(std::round(static_cast<double>(cfg.total_time) / csv_dt));
    const int n_frames = static_cast<int>(std::round(cfg.total_time / cfg.frame_time));
    const double visual_eps = 0.5 * csv_dt;

    std::vector<std::vector<double>> force_mags(runtimes.size());
    std::vector<std::vector<double>> force_z_components(runtimes.size());
    std::vector<std::ofstream> trace_csvs(runtimes.size());

    for (size_t i = 0; i < runtimes.size(); ++i) {
        trace_csvs[i].open(runtimes[i].summary.out_dir / "force_trace_1000fps.csv");
        trace_csvs[i] << std::setprecision(12);
        trace_csvs[i] << "sample,time_s,x_m,y_m,z_m,z_low_m,fx_N,fy_N,fz_N,fmag_N,plane_triangles,plane_cells_per_side";
        if (runtimes[i].summary.reference.available) {
            trace_csvs[i] << ",ref_total_N,ref_normal_N,rel_err_total_pct,rel_err_fz_pct";
        }
        trace_csvs[i] << "\n";
    }

    int visual_frame_idx = 0;
    double next_visual_time = static_cast<double>(cfg.frame_time);

    for (int sample_idx = 1; sample_idx <= n_csv_samples; ++sample_idx) {
        DEMSim.DoDynamics(csv_dt);
        const double time_s = static_cast<double>(sample_idx) * csv_dt;

        for (size_t i = 0; i < runtimes.size(); ++i) {
            auto& runtime = runtimes[i];
            const float3 cnt_acc = runtime.body.tracker->ContactAcc();
            const float3 cnt_force = cnt_acc * runtime.body.mass;
            const double force_mag = std::sqrt(static_cast<double>(cnt_force.x) * cnt_force.x +
                                               static_cast<double>(cnt_force.y) * cnt_force.y +
                                               static_cast<double>(cnt_force.z) * cnt_force.z);
            force_mags[i].push_back(force_mag);
            force_z_components[i].push_back(static_cast<double>(cnt_force.z));

            const float3 pos = runtime.body.tracker->Pos();
            const float body_lowest_z = pos.z - runtime.body.reference_above_lowest_point;

            trace_csvs[i] << sample_idx << ','
                          << time_s << ','
                          << pos.x << ','
                          << pos.y << ','
                          << pos.z << ','
                          << body_lowest_z << ','
                          << cnt_force.x << ','
                          << cnt_force.y << ','
                          << cnt_force.z << ','
                          << force_mag << ','
                          << plane_triangles << ','
                          << cells;

            if (runtime.summary.reference.available) {
                const double rel_err_total =
                    100.0 * (force_mag - runtime.summary.reference.total_force) / runtime.summary.reference.total_force;
                const double rel_err_fz =
                    100.0 * (static_cast<double>(cnt_force.z) - runtime.summary.reference.normal_force) /
                    runtime.summary.reference.normal_force;
                trace_csvs[i] << ',' << runtime.summary.reference.total_force
                              << ',' << runtime.summary.reference.normal_force
                              << ',' << rel_err_total
                              << ',' << rel_err_fz;
            }
            trace_csvs[i] << '\n';
        }

        if (visual_frame_idx < n_frames && time_s + visual_eps >= next_visual_time) {
            ++visual_frame_idx;

            char meshfilename[256];
            std::snprintf(meshfilename, sizeof(meshfilename), "mesh_%04d.vtk", visual_frame_idx);
            DEMSim.WriteMeshFile(run_summary.out_dir / meshfilename);

            char spherefilename[256];
            std::snprintf(spherefilename, sizeof(spherefilename), "sphere_%04d.csv", visual_frame_idx);
            DEMSim.WriteSphereFile(run_summary.out_dir / spherefilename);

            std::cout << "t=" << time_s << " s" << std::endl;
            for (size_t i = 0; i < runtimes.size(); ++i) {
                const auto& runtime = runtimes[i];
                const float3 pos = runtime.body.tracker->Pos();
                const float body_lowest_z = pos.z - runtime.body.reference_above_lowest_point;
                const float3 cnt_acc = runtime.body.tracker->ContactAcc();
                const float3 cnt_force = cnt_acc * runtime.body.mass;
                const double force_mag = std::sqrt(static_cast<double>(cnt_force.x) * cnt_force.x +
                                                   static_cast<double>(cnt_force.y) * cnt_force.y +
                                                   static_cast<double>(cnt_force.z) * cnt_force.z);
                std::cout << "  [" << runtime.summary.label << "]"
                          << " x=" << pos.x
                          << " y=" << pos.y
                          << " z_low=" << body_lowest_z
                          << " |F|=" << force_mag
                          << " Fz=" << cnt_force.z << std::endl;
            }

            next_visual_time += static_cast<double>(cfg.frame_time);
        }
    }

    run_summary.case_summaries.reserve(runtimes.size());
    for (size_t i = 0; i < runtimes.size(); ++i) {
        FinalizeSummaryFromTraces(runtimes[i].summary, force_mags[i], force_z_components[i]);
        PrintCaseSummary(runtimes[i].summary);
        run_summary.case_summaries.push_back(runtimes[i].summary);
    }

    DEMSim.ShowTimingStats();
    std::cout << "=====================================================\n";
    return run_summary;
}

void WriteParallelSummaryCsv(const path& filename, const std::vector<ParallelRunSummary>& results) {
    std::ofstream out(filename);
    out << "case_label,plane_triangles,plane_cells_per_side,mean_force_N,min_force_N,max_force_N,force_span_N,force_ok,mean_fz_N,min_fz_N,max_fz_N,fz_span_N,fz_ok,stddev_force_N,reference_total_N,reference_fz_N,delta_vs_finest_pct\n";

    for (const auto& result : results) {
        for (const auto& case_summary : result.case_summaries) {
            const std::string case_prefix = case_summary.label.substr(0, case_summary.label.rfind("_T"));
            double finest_force = std::numeric_limits<double>::quiet_NaN();
            for (auto it = results.rbegin(); it != results.rend(); ++it) {
                for (const auto& finest_case : it->case_summaries) {
                    if (finest_case.label.substr(0, finest_case.label.rfind("_T")) == case_prefix) {
                        finest_force = finest_case.mean_force;
                        break;
                    }
                }
                if (!std::isnan(finest_force)) {
                    break;
                }
            }

            double delta_vs_finest = std::numeric_limits<double>::quiet_NaN();
            if (!std::isnan(finest_force) && finest_force != 0.0) {
                delta_vs_finest = 100.0 * (case_summary.mean_force - finest_force) / finest_force;
            }

            out << case_summary.label << ','
                << case_summary.plane_triangles << ','
                << case_summary.plane_cells_per_side << ','
                << case_summary.mean_force << ','
                << case_summary.min_force << ','
                << case_summary.max_force << ','
                << case_summary.force_span << ','
                << (case_summary.force_ok ? 1 : 0) << ','
                << case_summary.mean_fz << ','
                << case_summary.min_fz << ','
                << case_summary.max_fz << ','
                << case_summary.fz_span << ','
                << (case_summary.fz_ok ? 1 : 0) << ','
                << case_summary.stddev_force << ','
                << (case_summary.reference.available ? case_summary.reference.total_force
                                                     : std::numeric_limits<double>::quiet_NaN())
                << ','
                << (case_summary.reference.available ? case_summary.reference.normal_force
                                                     : std::numeric_limits<double>::quiet_NaN())
                << ',' << delta_vs_finest << '\n';
        }
    }
}

void PrintTriangleStudyComparison(const std::vector<ParallelRunSummary>& results) {
    if (results.empty()) {
        return;
    }

    std::vector<std::string> case_prefixes;
    for (const auto& case_summary : results.front().case_summaries) {
        case_prefixes.push_back(case_summary.label.substr(0, case_summary.label.rfind("_T")));
    }

    for (const auto& case_prefix : case_prefixes) {
        double finest_force = std::numeric_limits<double>::quiet_NaN();
        for (const auto& case_summary : results.back().case_summaries) {
            if (case_summary.label.substr(0, case_summary.label.rfind("_T")) == case_prefix) {
                finest_force = case_summary.mean_force;
                break;
            }
        }

        std::cout << "\n================ Triangle-size comparison: " << case_prefix
                  << " ================\n";
        std::cout << std::setw(12) << "Triangles"
                  << std::setw(14) << "Cells/side"
                  << std::setw(18) << "Mean |F| [N]"
                  << std::setw(18) << "Mean Fz [N]"
                  << std::setw(18) << "StdDev [N]"
                  << std::setw(18) << "F OK"
                  << std::setw(18) << "Fz OK"
                  << std::setw(20) << "Delta vs finest [%]" << std::endl;

        for (const auto& result : results) {
            for (const auto& case_summary : result.case_summaries) {
                if (case_summary.label.substr(0, case_summary.label.rfind("_T")) != case_prefix) {
                    continue;
                }
                const double delta_vs_finest =
                    (finest_force != 0.0) ? 100.0 * (case_summary.mean_force - finest_force) / finest_force : 0.0;
                std::cout << std::setw(12) << case_summary.plane_triangles
                          << std::setw(14) << case_summary.plane_cells_per_side
                          << std::setw(18) << case_summary.mean_force
                          << std::setw(18) << case_summary.mean_fz
                          << std::setw(18) << case_summary.stddev_force
                          << std::setw(18) << OkString(case_summary.force_ok)
                          << std::setw(18) << OkString(case_summary.fz_ok)
                          << std::setw(20) << delta_vs_finest << std::endl;
            }
        }
        std::cout << "===============================================================\n";
    }
}

void WriteKTStudySummaryCsv(const path& filename,
                            const std::vector<std::pair<double, ParallelRunSummary>>& results) {
    std::ofstream out(filename);
    out << "kT_scale,case_label,plane_triangles,plane_cells_per_side,mean_force_N,min_force_N,max_force_N,force_span_N,force_ok,mean_fz_N,min_fz_N,max_fz_N,fz_span_N,fz_ok,stddev_force_N,reference_total_N,reference_fz_N\n";

    for (const auto& entry : results) {
        const double kT = entry.first;
        for (const auto& case_summary : entry.second.case_summaries) {
            out << kT << ','
                << case_summary.label << ','
                << case_summary.plane_triangles << ','
                << case_summary.plane_cells_per_side << ','
                << case_summary.mean_force << ','
                << case_summary.min_force << ','
                << case_summary.max_force << ','
                << case_summary.force_span << ','
                << (case_summary.force_ok ? 1 : 0) << ','
                << case_summary.mean_fz << ','
                << case_summary.min_fz << ','
                << case_summary.max_fz << ','
                << case_summary.fz_span << ','
                << (case_summary.fz_ok ? 1 : 0) << ','
                << case_summary.stddev_force << ','
                << (case_summary.reference.available ? case_summary.reference.total_force
                                                     : std::numeric_limits<double>::quiet_NaN())
                << ','
                << (case_summary.reference.available ? case_summary.reference.normal_force
                                                     : std::numeric_limits<double>::quiet_NaN())
                << '\n';
        }
    }
}

void PrintKTStudyTopN(const std::vector<std::pair<double, ParallelRunSummary>>& results, int top_n) {
    struct KTCaseRow {
        double kT = 0.0;
        RunSummary summary;
    };

    std::vector<KTCaseRow> rows;
    for (const auto& entry : results) {
        for (const auto& case_summary : entry.second.case_summaries) {
            KTCaseRow row;
            row.kT = entry.first;
            row.summary = case_summary;
            rows.push_back(row);
        }
    }

    if (rows.empty()) {
        return;
    }

    std::sort(rows.begin(), rows.end(), [](const KTCaseRow& a, const KTCaseRow& b) {
        return a.summary.mean_force > b.summary.mean_force;
    });

    const int n = std::min<int>(top_n, static_cast<int>(rows.size()));
    std::cout << "\n================ Top " << n << " kT/case results by mean |F| ================\n";
    std::cout << std::setw(12) << "kT"
              << std::setw(20) << "Case"
              << std::setw(18) << "Mean |F| [N]"
              << std::setw(18) << "Min |F| [N]"
              << std::setw(18) << "Max |F| [N]"
              << std::setw(18) << "Mean Fz [N]"
              << std::setw(18) << "F OK"
              << std::setw(18) << "Fz OK" << std::endl;

    for (int i = 0; i < n; ++i) {
        std::cout << std::setw(12) << rows[i].kT
                  << std::setw(20) << rows[i].summary.label
                  << std::setw(18) << rows[i].summary.mean_force
                  << std::setw(18) << rows[i].summary.min_force
                  << std::setw(18) << rows[i].summary.max_force
                  << std::setw(18) << rows[i].summary.mean_fz
                  << std::setw(18) << OkString(rows[i].summary.force_ok)
                  << std::setw(18) << OkString(rows[i].summary.fz_ok) << std::endl;
    }
    std::cout << "===============================================================\n";
}

void RunFullKTStudy(const DemoConfig& base_cfg, const path& root_out_dir) {
    std::vector<std::pair<double, ParallelRunSummary>> all_results;

    for (double kT : base_cfg.kt_sweep) {
        DemoConfig cfg = base_cfg;
        cfg.kT_study = false;
        cfg.triangle_study = false;
        cfg.speed_scale = base_cfg.speed_scale * static_cast<float>(kT);

        std::cout << "\n################ Parallel kT sweep case: " << kT << " ################\n";
        all_results.emplace_back(kT,
                                 RunParallelCaseSet(cfg, cfg.single_run_plane_triangles,
                                                    root_out_dir / "kT_study" / ("kT_" + std::to_string(kT))));
    }

    WriteKTStudySummaryCsv(root_out_dir / "kT_study_summary.csv", all_results);
    PrintKTStudyTopN(all_results, base_cfg.kt_top_n);
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        DemoConfig cfg = ParseArguments(argc, argv);

        const path root_out_dir = current_path() / "modular_test_output" / "DEMTest_GrazingPlaneVariants" /
                                  "all_variants_parallel";
        std::error_code dir_ec;
        create_directories(root_out_dir, dir_ec);
        if (dir_ec || !is_directory(root_out_dir)) {
            std::cerr << "Failed to create root output directory: " << root_out_dir << std::endl;
            return 1;
        }

        if (cfg.kT_study) {
            RunFullKTStudy(cfg, root_out_dir);
        } else {
            std::vector<ParallelRunSummary> results;
            if (cfg.triangle_study) {
                for (int tri_count : TRIANGLE_STUDY_COUNTS) {
                    results.push_back(RunParallelCaseSet(cfg, tri_count, root_out_dir / "triangle_study"));
                }
                PrintTriangleStudyComparison(results);
                WriteParallelSummaryCsv(root_out_dir / "triangle_study_summary.csv", results);
            } else {
                results.push_back(RunParallelCaseSet(cfg, cfg.single_run_plane_triangles, root_out_dir / "single_run"));
                WriteParallelSummaryCsv(root_out_dir / "single_run_summary.csv", results);
            }
        }

        std::cout << "DEMTest_GrazingPlaneVariants exiting..." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
