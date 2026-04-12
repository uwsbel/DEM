//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

// =============================================================================
// Grazing-contact reference demo on a fixed sphere target with these primary 6
// variants:
//   cone, cone-study, cube, cube-study, sphere, sphere-study
//
// The moving body performs a circular orbit around a fixed sphere mesh at the
// origin (no gravity), keeping the active contact feature grazing the sphere.
// Uses cone.obj, cube.obj, sphere.obj and sphere_highres.obj.
//
// Key extensions relative to the simple cone-only demo:
//   1) Variant flag for cone mesh, cube mesh, or sphere mesh.
//   2) Sphere target-mesh study over exactly two cases:
//        - sphere.obj
//        - sphere_highres.obj
//   3) Practical geometric scaling flags.
// Additional cube mode flag: --cube-mode edge|tip
//
// Notes:
//   - The target is always a fixed sphere mesh.
//   - The moving counterpart cases are cone, sphere, cube-edge, cube-tip.
//   - cone/cube references use the same DEME-style tangent-plane proxy as the
//     plane variant, now interpreted locally on the target sphere.
//   - sphere uses classic Hertz sphere-sphere with DEME's identical-material
//     effective modulus and reduced radius.
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
constexpr std::array<const char*, 2> TARGET_SPHERE_MESH_CASES = {"sphere.obj", "sphere_highres.obj"};
constexpr std::array<double, 9> DEFAULT_KT_SWEEP = {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0};

enum class ShapeVariant { CONE, CUBE, SPHERE };
enum class CubeContactMode { EDGE, TIP };
enum class DemoVariant { CONE, CONE_STUDY, CUBE, CUBE_STUDY, SPHERE, SPHERE_STUDY };

struct DemoConfig {
    DemoVariant variant = DemoVariant::CONE;
    ShapeVariant shape = ShapeVariant::CONE;
    bool target_mesh_study = false;
    CubeContactMode cube_mode = CubeContactMode::EDGE;

    float mu = 0.3f;
    float CoR = 0.3f;
    float E = 1e8f;
    float nu = 0.3f;
    float density = 2600.0f;

    float base_body_size = 0.2f;            // cone h/r, cube edge, moving sphere diameter
    float base_target_sphere_radius = 0.25f;
    float base_penetration = 0.005f;
    float base_orbit_omega = 2.0f;

    float global_scale = 1.0f;
    float body_size_scale = 1.0f;
    float target_sphere_scale = 1.0f;
    float penetration_scale = 1.0f;
    float speed_scale = 1.0f;

    float step_size = 2e-5f;
    float frame_time = 0.05f;
    float total_time = 0.2f;
    float csv_fps = 1000.0f;

    std::string single_run_target_sphere_mesh = "sphere.obj";

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
    std::string target_mesh;
    std::string label;
    path out_dir;

    double mean_force = 0.0;
    double min_force = 0.0;
    double max_force = 0.0;
    double stddev_force = 0.0;

    double mean_radial = 0.0;
    double min_radial = 0.0;
    double max_radial = 0.0;

    AnalyticalReference reference;
};

struct MovingBodyState {
    std::shared_ptr<DEMTracker> tracker;
    float mass = 0.0f;
    float reference_above_lowest_point = 0.0f;
    unsigned int family = 1;
};

struct ObjTriangleMesh {
    std::vector<float3> vertices;
    std::vector<std::array<int, 3>> triangles;
    std::vector<float3> face_normals;
    std::vector<float3> vertex_normals;
};

struct BodyMeshKinematics {
    std::string mesh_file;
    std::vector<float3> local_vertices;
    float4 plane_contact_quat = make_float4(0.f, 0.f, 0.f, 1.f);
};

struct BodyPose {
    float3 pos = make_float3(0.f, 0.f, 0.f);
    float3 vel = make_float3(0.f, 0.f, 0.f);
    float4 quat = make_float4(0.f, 0.f, 0.f, 1.f);
    float3 angvel_local = make_float3(0.f, 0.f, 0.f);
    float3 surface_point = make_float3(0.f, 0.f, 0.f);
    float3 surface_normal = make_float3(1.f, 0.f, 0.f);
    float3 tangent = make_float3(0.f, 1.f, 0.f);
    float support_distance = 0.f;
};

struct PosePath {
    std::vector<BodyPose> poses;
    float max_speed = 0.f;
};

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

std::string MeshLabel(const std::string& mesh_name) {
    return ToLower(mesh_name) == "sphere_highres.obj" ? "sphere_highres" : "sphere";
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
    throw std::runtime_error("Unknown --variant option: " + value);
}

void ApplyVariantToConfig(DemoConfig& cfg, DemoVariant variant) {
    cfg.variant = variant;
    switch (variant) {
        case DemoVariant::CONE:
            cfg.shape = ShapeVariant::CONE;
            cfg.target_mesh_study = false;
            break;
        case DemoVariant::CONE_STUDY:
            cfg.shape = ShapeVariant::CONE;
            cfg.target_mesh_study = true;
            break;
        case DemoVariant::CUBE:
            cfg.shape = ShapeVariant::CUBE;
            cfg.target_mesh_study = false;
            break;
        case DemoVariant::CUBE_STUDY:
            cfg.shape = ShapeVariant::CUBE;
            cfg.target_mesh_study = true;
            break;
        case DemoVariant::SPHERE:
            cfg.shape = ShapeVariant::SPHERE;
            cfg.target_mesh_study = false;
            break;
        case DemoVariant::SPHERE_STUDY:
            cfg.shape = ShapeVariant::SPHERE;
            cfg.target_mesh_study = true;
            break;
    }
}

void RefreshVariantFromFlags(DemoConfig& cfg) {
    if (cfg.shape == ShapeVariant::CONE)
        cfg.variant = cfg.target_mesh_study ? DemoVariant::CONE_STUDY : DemoVariant::CONE;
    else if (cfg.shape == ShapeVariant::CUBE)
        cfg.variant = cfg.target_mesh_study ? DemoVariant::CUBE_STUDY : DemoVariant::CUBE;
    else
        cfg.variant = cfg.target_mesh_study ? DemoVariant::SPHERE_STUDY : DemoVariant::SPHERE;
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
        if (token.empty())
            continue;
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

std::string NormalizeTargetSphereMeshName(const std::string& value) {
    const std::string v = ToLower(value);
    if (v == "sphere" || v == "default" || v == "sphere.obj")
        return "sphere.obj";
    if (v == "highres" || v == "sphere_highres" || v == "sphere_highres.obj")
        return "sphere_highres.obj";
    throw std::runtime_error("Unknown target sphere mesh option: " + value);
}

void PrintUsage(const char* exe_name) {
    std::cout << "Usage:\n"
              << "  " << exe_name << " --variant cone|cone-study|cube|cube-study|sphere|sphere-study\n"
              << "\nAlternative flags:\n"
              << "  --shape cone|cube|sphere   and   --target-mesh-study true|false\n"
              << "  --cube-mode edge|tip       (only used with cube)\n"
              << "\nTarget mesh flags:\n"
              << "  --target-sphere-mesh sphere|highres|sphere.obj|sphere_highres.obj\n"
              << "  --highres-sphere true|false\n"
              << "\nScaling flags:\n"
              << "  --global-scale <f>\n"
              << "  --body-size-scale <f>\n"
              << "  --target-sphere-scale <f>\n"
              << "  --penetration-scale <f>\n"
              << "  --speed-scale <f>\n"
              << "\nAnalysis flags:\n"
              << "  --kT_study [true|false]    trigger full analysis over the kT sweep\n"
              << "  --kt-top-n <n>             report top-N kT cases (default 3)\n"
              << "  --kt-sweep <v1,v2,...>     override default kT sweep\n"
              << "\nOther flags:\n"
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

        if (arg == "--variant") {
            ApplyVariantToConfig(cfg, ParseVariantName(require_value(arg)));
        } else if (arg == "--shape") {
            const std::string value = ToLower(require_value(arg));
            if (value == "cone")
                cfg.shape = ShapeVariant::CONE;
            else if (value == "cube")
                cfg.shape = ShapeVariant::CUBE;
            else if (value == "sphere")
                cfg.shape = ShapeVariant::SPHERE;
            else
                throw std::runtime_error("Unknown --shape option: " + value);
            RefreshVariantFromFlags(cfg);
        } else if (arg == "--target-mesh-study" || arg == "--target-sphere-mesh-study") {
            cfg.target_mesh_study = ParseBool(require_value(arg));
            RefreshVariantFromFlags(cfg);
        } else if (arg == "--cube-mode") {
            cfg.cube_mode = ParseCubeMode(require_value(arg));
        } else if (arg == "--target-sphere-mesh") {
            cfg.single_run_target_sphere_mesh = NormalizeTargetSphereMeshName(require_value(arg));
        } else if (arg == "--highres-sphere") {
            cfg.single_run_target_sphere_mesh = ParseBool(require_value(arg)) ? "sphere_highres.obj" : "sphere.obj";
        } else if (arg == "--global-scale") {
            cfg.global_scale = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--body-size-scale") {
            cfg.body_size_scale = ParseFloatArg(arg, require_value(arg));
        } else if (arg == "--target-sphere-scale") {
            cfg.target_sphere_scale = ParseFloatArg(arg, require_value(arg));
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
        } else if (arg == "--kT_study") {
            if (i + 1 < argc && argv[i + 1][0] != '-')
                cfg.kT_study = ParseBool(require_value(arg));
            else
                cfg.kT_study = true;
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
    if (cfg.step_size <= 0.0f || cfg.frame_time <= 0.0f || cfg.total_time <= 0.0f || cfg.csv_fps <= 0.0f) {
        throw std::runtime_error("step-size, frame-time, total-time and csv-fps must all be > 0.");
    }

    return cfg;
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

    if (dot_v > 1.f - 1e-6f)
        return make_float4(0.f, 0.f, 0.f, 1.f);

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

float4 CubeContactQuat(CubeContactMode mode) {
    if (mode == CubeContactMode::EDGE) {
        return QuatAxisAngle(make_float3(0.f, 1.f, 0.f), static_cast<float>(PI_D / 4.0));
    }
    return QuatFromTwoVectors(make_float3(-1.f, -1.f, -1.f), make_float3(0.f, 0.f, -1.f));
}

float4 ConvertPlaneContactQuatToSphereOrbitQuat(const float4& q_plane) {
    const float4 align = QuatFromTwoVectors(make_float3(0.f, 0.f, -1.f), make_float3(-1.f, 0.f, 0.f));
    return NormalizeQuat(QuatMul(align, q_plane));
}

float3 Add3(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 Sub3(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 Scale3(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

float4 ConjugateQuat(const float4& q) {
    return make_float4(-q.x, -q.y, -q.z, q.w);
}

float4 QuatFromBasis(const float3& x_raw, const float3& y_raw, const float3& z_raw) {
    const float3 x = Normalize3(x_raw);
    const float3 y = Normalize3(y_raw);
    const float3 z = Normalize3(z_raw);

    const float m00 = x.x, m01 = y.x, m02 = z.x;
    const float m10 = x.y, m11 = y.y, m12 = z.y;
    const float m20 = x.z, m21 = y.z, m22 = z.z;

    const float trace = m00 + m11 + m22;
    float4 q;
    if (trace > 0.f) {
        const float s = std::sqrt(trace + 1.f) * 2.f;
        q = make_float4((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25f * s);
    } else if (m00 > m11 && m00 > m22) {
        const float s = std::sqrt(1.f + m00 - m11 - m22) * 2.f;
        q = make_float4(0.25f * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s);
    } else if (m11 > m22) {
        const float s = std::sqrt(1.f + m11 - m00 - m22) * 2.f;
        q = make_float4((m01 + m10) / s, 0.25f * s, (m12 + m21) / s, (m02 - m20) / s);
    } else {
        const float s = std::sqrt(1.f + m22 - m00 - m11) * 2.f;
        q = make_float4((m02 + m20) / s, (m12 + m21) / s, 0.25f * s, (m10 - m01) / s);
    }
    return NormalizeQuat(q);
}

int ParseOBJVertexIndexToken(const std::string& token, int num_vertices) {
    const size_t slash = token.find('/');
    const std::string head = (slash == std::string::npos) ? token : token.substr(0, slash);
    if (head.empty()) {
        throw std::runtime_error("Malformed OBJ face index token: " + token);
    }
    const int raw = std::stoi(head);
    if (raw > 0)
        return raw - 1;
    if (raw < 0)
        return num_vertices + raw;
    throw std::runtime_error("OBJ indices are 1-based; got 0 in token: " + token);
}

ObjTriangleMesh LoadOBJTriangleMesh(const path& filename, float scale = 1.f, const float3& centroid_shift = make_float3(0.f, 0.f, 0.f)) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Failed to open OBJ file: " + filename.string());
    }

    ObjTriangleMesh mesh;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::string tag;
        iss >> tag;
        if (tag == "v") {
            float x = 0.f, y = 0.f, z = 0.f;
            iss >> x >> y >> z;
            mesh.vertices.push_back(make_float3((x - centroid_shift.x) * scale,
                                                (y - centroid_shift.y) * scale,
                                                (z - centroid_shift.z) * scale));
        } else if (tag == "f") {
            std::vector<int> face_indices;
            std::string tok;
            while (iss >> tok) {
                face_indices.push_back(ParseOBJVertexIndexToken(tok, static_cast<int>(mesh.vertices.size())));
            }
            if (face_indices.size() < 3)
                continue;
            for (size_t i = 1; i + 1 < face_indices.size(); ++i) {
                mesh.triangles.push_back({face_indices[0], face_indices[i], face_indices[i + 1]});
            }
        }
    }

    if (mesh.vertices.empty() || mesh.triangles.empty()) {
        throw std::runtime_error("OBJ file did not yield a usable triangle mesh: " + filename.string());
    }

    mesh.face_normals.reserve(mesh.triangles.size());
    mesh.vertex_normals.assign(mesh.vertices.size(), make_float3(0.f, 0.f, 0.f));
    for (const auto& tri : mesh.triangles) {
        const float3& v0 = mesh.vertices[tri[0]];
        const float3& v1 = mesh.vertices[tri[1]];
        const float3& v2 = mesh.vertices[tri[2]];
        float3 area_normal = Cross3(Sub3(v1, v0), Sub3(v2, v0));
        float3 n = Normalize3(area_normal);
        const float3 face_center = Scale3(Add3(Add3(v0, v1), v2), 1.f / 3.f);
        if (Dot3(n, face_center) < 0.f) {
            n = Scale3(n, -1.f);
            area_normal = Scale3(area_normal, -1.f);
        }
        mesh.face_normals.push_back(n);
        mesh.vertex_normals[tri[0]] = Add3(mesh.vertex_normals[tri[0]], area_normal);
        mesh.vertex_normals[tri[1]] = Add3(mesh.vertex_normals[tri[1]], area_normal);
        mesh.vertex_normals[tri[2]] = Add3(mesh.vertex_normals[tri[2]], area_normal);
    }
    for (size_t i = 0; i < mesh.vertex_normals.size(); ++i) {
        float3 n = Normalize3(mesh.vertex_normals[i]);
        if (Length3(n) < 1e-8f) {
            n = Normalize3(mesh.vertices[i]);
        }
        if (Dot3(n, mesh.vertices[i]) < 0.f) {
            n = Scale3(n, -1.f);
        }
        mesh.vertex_normals[i] = n;
    }

    return mesh;
}

float MaxVertexRadius(const std::vector<float3>& vertices) {
    float max_r = 0.f;
    for (const auto& v : vertices) {
        max_r = std::max(max_r, Length3(v));
    }
    return max_r;
}

bool RayIntersectTriangleFromOrigin(const float3& dir,
                                    const float3& v0,
                                    const float3& v1,
                                    const float3& v2,
                                    float& t_out,
                                    float& u_out,
                                    float& v_out) {
    constexpr float eps = 1e-8f;
    const float3 e1 = Sub3(v1, v0);
    const float3 e2 = Sub3(v2, v0);
    const float3 pvec = Cross3(dir, e2);
    const float det = Dot3(e1, pvec);
    if (std::fabs(det) < eps)
        return false;
    const float inv_det = 1.f / det;
    const float3 tvec = Scale3(v0, -1.f);
    const float u = Dot3(tvec, pvec) * inv_det;
    if (u < -eps || u > 1.f + eps)
        return false;
    const float3 qvec = Cross3(tvec, e1);
    const float v = Dot3(dir, qvec) * inv_det;
    if (v < -eps || u + v > 1.f + eps)
        return false;
    const float t = Dot3(e2, qvec) * inv_det;
    if (t <= eps)
        return false;
    t_out = t;
    u_out = u;
    v_out = v;
    return true;
}

float3 InterpolateTriangleNormal(const ObjTriangleMesh& mesh, int tri_idx, float bary_u, float bary_v) {
    const auto& tri = mesh.triangles[tri_idx];
    const float bary_w = 1.f - bary_u - bary_v;
    float3 n = Add3(Scale3(mesh.vertex_normals[tri[0]], bary_w),
                    Add3(Scale3(mesh.vertex_normals[tri[1]], bary_u),
                         Scale3(mesh.vertex_normals[tri[2]], bary_v)));
    n = Normalize3(n);
    if (Length3(n) < 1e-8f) {
        n = mesh.face_normals[tri_idx];
    }
    const float3 hit_point = Add3(Scale3(mesh.vertices[tri[0]], bary_w),
                                  Add3(Scale3(mesh.vertices[tri[1]], bary_u),
                                       Scale3(mesh.vertices[tri[2]], bary_v)));
    if (Dot3(n, hit_point) < 0.f) {
        n = Scale3(n, -1.f);
    }
    return n;
}

BodyMeshKinematics BuildBodyMeshKinematics(const DemoConfig& cfg, float body_size) {
    BodyMeshKinematics body;
    if (cfg.shape == ShapeVariant::CONE) {
        body.mesh_file = (GET_DATA_PATH() / "mesh/cone.obj").string();
        body.local_vertices = LoadOBJTriangleMesh(body.mesh_file, body_size, make_float3(0.f, 0.f, 0.75f)).vertices;
        body.plane_contact_quat = make_float4(0.f, 0.f, 0.f, 1.f);
    } else if (cfg.shape == ShapeVariant::CUBE) {
        body.mesh_file = (GET_DATA_PATH() / "mesh/cube.obj").string();
        body.local_vertices = LoadOBJTriangleMesh(body.mesh_file, body_size).vertices;
        body.plane_contact_quat = CubeContactQuat(cfg.cube_mode);
    } else {
        const float sphere_radius = 0.5f * body_size;
        body.mesh_file = (GET_DATA_PATH() / "mesh/sphere.obj").string();
        body.local_vertices = LoadOBJTriangleMesh(body.mesh_file, sphere_radius).vertices;
        body.plane_contact_quat = make_float4(0.f, 0.f, 0.f, 1.f);
    }
    return body;
}

float ComputeSupportDistance(const std::vector<float3>& local_vertices, const float4& q_body, const float3& normal) {
    float min_proj = std::numeric_limits<float>::infinity();
    for (const auto& v_local : local_vertices) {
        const float3 v_world = RotateByQuat(q_body, v_local);
        min_proj = std::min(min_proj, Dot3(v_world, normal));
    }
    return -min_proj;
}

float3 ComputeSurfaceTangent(const float3& normal, double phase_rad) {
    const float3 raw_orbit_tangent = make_float3(static_cast<float>(-std::sin(phase_rad)),
                                                 static_cast<float>(std::cos(phase_rad)),
                                                 0.f);
    float3 tangent = Sub3(raw_orbit_tangent, Scale3(normal, Dot3(raw_orbit_tangent, normal)));
    if (Length3(tangent) < 1e-8f) {
        tangent = Cross3(make_float3(0.f, 0.f, 1.f), normal);
        if (Length3(tangent) < 1e-8f) {
            tangent = Cross3(make_float3(1.f, 0.f, 0.f), normal);
        }
    }
    return Normalize3(tangent);
}

BodyPose EvaluateBodyPose(const DemoConfig& cfg,
                          const ObjTriangleMesh& target_mesh,
                          const BodyMeshKinematics& body_mesh,
                          float penetration,
                          double phase_rad,
                          int& preferred_tri_idx) {
    const float3 ray_dir = Normalize3(make_float3(static_cast<float>(std::cos(phase_rad)),
                                                  static_cast<float>(std::sin(phase_rad)),
                                                  0.f));

    float best_t = std::numeric_limits<float>::infinity();
    float best_u = 0.f;
    float best_v = 0.f;
    int best_tri = -1;
    auto try_tri = [&](int tri_idx) {
        float t = 0.f;
        float u = 0.f;
        float v = 0.f;
        const auto& tri = target_mesh.triangles[tri_idx];
        if (!RayIntersectTriangleFromOrigin(ray_dir,
                                            target_mesh.vertices[tri[0]],
                                            target_mesh.vertices[tri[1]],
                                            target_mesh.vertices[tri[2]],
                                            t,
                                            u,
                                            v)) {
            return;
        }
        if (t < best_t) {
            best_t = t;
            best_u = u;
            best_v = v;
            best_tri = tri_idx;
        }
    };

    if (preferred_tri_idx >= 0 && preferred_tri_idx < static_cast<int>(target_mesh.triangles.size())) {
        try_tri(preferred_tri_idx);
    }
    if (best_tri < 0) {
        for (int tri_idx = 0; tri_idx < static_cast<int>(target_mesh.triangles.size()); ++tri_idx) {
            try_tri(tri_idx);
        }
    }
    if (best_tri < 0) {
        throw std::runtime_error("Failed to ray-hit target sphere mesh along the prescribed orbit direction.");
    }
    preferred_tri_idx = best_tri;

    BodyPose pose;
    const auto& best_face = target_mesh.triangles[best_tri];
    const float best_w = 1.f - best_u - best_v;
    pose.surface_point = Add3(Scale3(target_mesh.vertices[best_face[0]], best_w),
                              Add3(Scale3(target_mesh.vertices[best_face[1]], best_u),
                                   Scale3(target_mesh.vertices[best_face[2]], best_v)));
    pose.surface_normal = InterpolateTriangleNormal(target_mesh, best_tri, best_u, best_v);
    pose.tangent = ComputeSurfaceTangent(pose.surface_normal, phase_rad);
    const float3 binormal = Normalize3(Cross3(pose.tangent, pose.surface_normal));
    const float4 frame_quat = QuatFromBasis(binormal, pose.tangent, pose.surface_normal);
    pose.quat = NormalizeQuat(QuatMul(frame_quat, body_mesh.plane_contact_quat));
    pose.support_distance = ComputeSupportDistance(body_mesh.local_vertices, pose.quat, pose.surface_normal);
    pose.pos = Add3(pose.surface_point, Scale3(pose.surface_normal, pose.support_distance - penetration));
    return pose;
}

PosePath BuildPosePath(const DemoConfig& cfg,
                       const ObjTriangleMesh& target_mesh,
                       const BodyMeshKinematics& body_mesh,
                       float penetration,
                       float orbit_omega) {
    PosePath path;
    const int n_steps = static_cast<int>(std::round(cfg.total_time / cfg.step_size));
    path.poses.resize(static_cast<size_t>(n_steps) + 1);

    int preferred_tri_idx = -1;
    for (int i = 0; i <= n_steps; ++i) {
        const double t = static_cast<double>(i) * static_cast<double>(cfg.step_size);
        const double phase = static_cast<double>(orbit_omega) * t;
        path.poses[static_cast<size_t>(i)] = EvaluateBodyPose(cfg, target_mesh, body_mesh, penetration, phase, preferred_tri_idx);
    }

    for (int i = 0; i < n_steps; ++i) {
        BodyPose& cur = path.poses[static_cast<size_t>(i)];
        const BodyPose& nxt = path.poses[static_cast<size_t>(i + 1)];
        cur.vel = Scale3(Sub3(nxt.pos, cur.pos), 1.f / cfg.step_size);

        float4 q_rel = NormalizeQuat(QuatMul(nxt.quat, ConjugateQuat(cur.quat)));
        if (q_rel.w < 0.f) {
            q_rel = make_float4(-q_rel.x, -q_rel.y, -q_rel.z, -q_rel.w);
        }
        const float sin_half = Length3(make_float3(q_rel.x, q_rel.y, q_rel.z));
        float3 omega_world = make_float3(0.f, 0.f, 0.f);
        if (sin_half > 1e-8f) {
            const float angle = 2.f * std::atan2(sin_half, q_rel.w);
            const float3 axis_world = Scale3(make_float3(q_rel.x, q_rel.y, q_rel.z), 1.f / sin_half);
            omega_world = Scale3(axis_world, angle / cfg.step_size);
        }
        cur.angvel_local = RotateByQuat(ConjugateQuat(cur.quat), omega_world);
        path.max_speed = std::max(path.max_speed, Length3(cur.vel));
    }

    if (!path.poses.empty()) {
        path.poses.back().vel = (path.poses.size() >= 2) ? path.poses[path.poses.size() - 2].vel : make_float3(0.f, 0.f, 0.f);
        path.poses.back().angvel_local =
            (path.poses.size() >= 2) ? path.poses[path.poses.size() - 2].angvel_local : make_float3(0.f, 0.f, 0.f);
    }

    return path;
}

AnalyticalReference BuildReferenceCone(float E, float nu, float mu, float radius, float height, float penetration) {
    AnalyticalReference ref;
    ref.available = true;
    ref.model = "DEME local tangent-plane proxy for conical mesh on sphere + Coulomb friction";
    ref.half_angle_rad = std::atan(static_cast<double>(radius) / static_cast<double>(height));
    ref.effective_modulus = DEMEEffectiveModulus(E, nu);
    ref.normal_force = (4.0 / 3.0) * ref.effective_modulus * std::tan(ref.half_angle_rad) *
                       static_cast<double>(penetration) * static_cast<double>(penetration);
    ref.tangential_force = static_cast<double>(mu) * ref.normal_force;
    ref.total_force = std::sqrt(ref.normal_force * ref.normal_force + ref.tangential_force * ref.tangential_force);
    return ref;
}

AnalyticalReference BuildReferenceSphereSphere(float E,
                                               float nu,
                                               float mu,
                                               float moving_radius,
                                               float target_radius,
                                               float penetration) {
    AnalyticalReference ref;
    ref.available = true;
    ref.model = "DEME Hertz sphere-sphere + Coulomb friction (identical materials)";
    ref.effective_modulus = DEMEEffectiveModulus(E, nu);
    const double reduced_radius = (static_cast<double>(moving_radius) * static_cast<double>(target_radius)) /
                                  (static_cast<double>(moving_radius) + static_cast<double>(target_radius));
    ref.normal_force = (4.0 / 3.0) * ref.effective_modulus * std::sqrt(reduced_radius) *
                       std::pow(static_cast<double>(penetration), 1.5);
    ref.tangential_force = static_cast<double>(mu) * ref.normal_force;
    ref.total_force = std::sqrt(ref.normal_force * ref.normal_force + ref.tangential_force * ref.tangential_force);
    return ref;
}

AnalyticalReference BuildReferenceCubeEdge(float E, float nu, float mu, float cube_edge, float penetration) {
    AnalyticalReference ref;
    ref.available = true;
    ref.model = "DEME local tangent-plane proxy for finite 90deg wedge edge on sphere + Coulomb friction";
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
    ref.model = "DEME local tangent-plane proxy for cube vertex on sphere + Coulomb friction";
    ref.effective_modulus = DEMEEffectiveModulus(E, nu);

    const double area_coeff = 1.5 * std::sqrt(3.0);
    const double contact_radius = std::sqrt(area_coeff / PI_D) * static_cast<double>(penetration);
    ref.half_angle_rad = std::atan(std::sqrt(area_coeff / PI_D));
    ref.normal_force = (4.0 / 3.0) * ref.effective_modulus * contact_radius * static_cast<double>(penetration);
    ref.tangential_force = static_cast<double>(mu) * ref.normal_force;
    ref.total_force = std::sqrt(ref.normal_force * ref.normal_force + ref.tangential_force * ref.tangential_force);
    return ref;
}

AnalyticalReference BuildReferenceForShape(const DemoConfig& cfg, float body_size, float target_radius, float penetration) {
    switch (cfg.shape) {
        case ShapeVariant::CONE:
            return BuildReferenceCone(cfg.E, cfg.nu, cfg.mu, body_size, body_size, penetration);
        case ShapeVariant::SPHERE:
            return BuildReferenceSphereSphere(cfg.E, cfg.nu, cfg.mu, 0.5f * body_size, target_radius, penetration);
        case ShapeVariant::CUBE:
            if (cfg.cube_mode == CubeContactMode::EDGE)
                return BuildReferenceCubeEdge(cfg.E, cfg.nu, cfg.mu, body_size, penetration);
            return BuildReferenceCubeTip(cfg.E, cfg.nu, cfg.mu, penetration);
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
                              const BodyPose& init_pose) {
    MovingBodyState state;

    if (cfg.shape == ShapeVariant::CONE) {
        const float cone_radius = body_size;
        const float cone_height = body_size;
        const float cone_volume = (1.0f / 3.0f) * static_cast<float>(PI_D) * cone_radius * cone_radius * cone_height;
        const float cone_mass = cfg.density * cone_volume;
        const float cone_Ixy = 3.0f * cone_mass / 20.0f * cone_radius * cone_radius +
                               3.0f * cone_mass / 80.0f * cone_height * cone_height;
        const float cone_Iz = 3.0f * cone_mass / 10.0f * cone_radius * cone_radius;

        auto cone = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cone.obj").string(), mat);
        cone->InformCentroidPrincipal(make_float3(0.f, 0.f, 0.75f), make_float4(0.f, 0.f, 0.f, 1.f));
        cone->Scale(body_size);
        cone->SetMass(cone_mass);
        cone->SetMOI(make_float3(cone_Ixy, cone_Ixy, cone_Iz));
        cone->SetInitPos(init_pose.pos);
        cone->SetInitQuat(init_pose.quat);
        cone->SetFamily(1);

        state.tracker = DEMSim.Track(cone);
        state.mass = cone_mass;
        state.reference_above_lowest_point = init_pose.support_distance;
        state.family = 1;
        return state;
    }

    if (cfg.shape == ShapeVariant::CUBE) {
        const float cube_edge = body_size;
        const float cube_mass = cfg.density * cube_edge * cube_edge * cube_edge;
        const float cube_I = cube_mass * cube_edge * cube_edge / 6.0f;

        auto cube = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/cube.obj").string(), mat);
        cube->Scale(body_size);
        cube->SetMass(cube_mass);
        cube->SetMOI(make_float3(cube_I, cube_I, cube_I));
        cube->SetInitQuat(init_pose.quat);
        cube->SetInitPos(init_pose.pos);
        cube->SetFamily(1);

        state.tracker = DEMSim.Track(cube);
        state.mass = cube_mass;
        state.reference_above_lowest_point = init_pose.support_distance;
        state.family = 1;
        return state;
    }

    const float sphere_radius = 0.5f * body_size;
    const float sphere_volume = 4.0f / 3.0f * static_cast<float>(PI_D) * sphere_radius * sphere_radius * sphere_radius;
    const float sphere_mass = cfg.density * sphere_volume;
    const float sphere_I = 2.0f / 5.0f * sphere_mass * sphere_radius * sphere_radius;

    auto sphere = DEMSim.AddWavefrontMeshObject((GET_DATA_PATH() / "mesh/sphere.obj").string(), mat);
    sphere->Scale(sphere_radius);
    sphere->SetMass(sphere_mass);
    sphere->SetMOI(make_float3(sphere_I, sphere_I, sphere_I));
    sphere->SetInitQuat(init_pose.quat);
    sphere->SetInitPos(init_pose.pos);
    sphere->SetFamily(1);

    state.tracker = DEMSim.Track(sphere);
    state.mass = sphere_mass;
    state.reference_above_lowest_point = init_pose.support_distance;
    state.family = 1;
    return state;
}

RunSummary RunSingleCase(const DemoConfig& cfg, const std::string& target_mesh_name, const path& root_out_dir) {
    const float body_size = cfg.base_body_size * cfg.global_scale * cfg.body_size_scale;
    const float target_radius = cfg.base_target_sphere_radius * cfg.global_scale * cfg.target_sphere_scale;
    const float penetration = cfg.base_penetration * cfg.global_scale * cfg.penetration_scale;
    const float orbit_omega = cfg.base_orbit_omega * cfg.speed_scale;

    float expand_factor = 2.0f;
    if (const char* env_expand = std::getenv("DEM_GRAZE_EXPAND_FACTOR")) {
        try {
            expand_factor = std::stof(env_expand);
        } catch (const std::exception&) {
            expand_factor = 2.0f;
        }
    }

    RunSummary summary;
    summary.target_mesh = target_mesh_name;
    summary.label = CaseName(cfg) + "_" + MeshLabel(target_mesh_name);
    summary.out_dir = root_out_dir / summary.label;

    const path target_mesh_path = GET_DATA_PATH() / "mesh" / target_mesh_name;
    const ObjTriangleMesh target_mesh_host = LoadOBJTriangleMesh(target_mesh_path, target_radius);
    const BodyMeshKinematics body_mesh = BuildBodyMeshKinematics(cfg, body_size);
    const PosePath pose_path = BuildPosePath(cfg, target_mesh_host, body_mesh, penetration, orbit_omega);
    if (pose_path.poses.empty()) {
        throw std::runtime_error("Failed to build prescribed pose path.");
    }

    DEMSolver DEMSim;
    DEMSim.SetVerbosity("INFO");
    DEMSim.SetOutputFormat(OUTPUT_FORMAT::CSV);
    DEMSim.InstructBoxDomainDimension(5, 5, 5);
    DEMSim.SetGravitationalAcceleration(make_float3(0, 0, 0));
    DEMSim.SetMeshUniversalContact(true);
    DEMSim.SetExpandSafetyType("auto");

    auto mat = DEMSim.LoadMaterial({{"E", cfg.E}, {"nu", cfg.nu}, {"CoR", cfg.CoR}, {"mu", cfg.mu}, {"Crr", 0.0f}});

    auto target = DEMSim.AddWavefrontMeshObject(target_mesh_path.string(), mat);
    target->Scale(target_radius);
    target->SetFamily(100);
    DEMSim.SetFamilyFixed(100);

    DEMSim.SetExpandSafetyAdder(std::max(1e-6f, pose_path.max_speed * expand_factor));

    MovingBodyState body = AddMovingBody(DEMSim, cfg, mat, body_size, pose_path.poses.front());
    DEMSim.SetFamilyPrescribedPosition(body.family);
    DEMSim.SetFamilyPrescribedQuaternion(body.family);
    DEMSim.SetFamilyPrescribedLinVel(body.family);
    DEMSim.SetFamilyPrescribedAngVel(body.family);

    DEMSim.TryDisableRuntimeCompiler(true);

    DEMSim.SetInitTimeStep(cfg.step_size);
    DEMSim.Initialize();

    body.tracker->SetPos(pose_path.poses.front().pos);
    body.tracker->SetOriQ(pose_path.poses.front().quat);
    body.tracker->SetVel(pose_path.poses.front().vel);
    body.tracker->SetAngVel(pose_path.poses.front().angvel_local);
    body.reference_above_lowest_point = pose_path.poses.front().support_distance;

    std::error_code dir_ec;
    create_directories(summary.out_dir, dir_ec);
    if (dir_ec || !is_directory(summary.out_dir)) {
        throw std::runtime_error("Failed to create output directory: " + summary.out_dir.string());
    }

    summary.reference = BuildReferenceForShape(cfg, body_size, target_radius, penetration);

    const double initial_centroid_radius = Length3(pose_path.poses.front().pos);

    std::cout << "=====================================================\n";
    std::cout << "Variant:             " << VariantName(cfg) << "\n";
    if (cfg.shape == ShapeVariant::CUBE)
        std::cout << "Cube mode:           " << CubeModeName(cfg.cube_mode) << "\n";
    std::cout << "Target sphere mesh:  " << target_mesh_name << "\n";
    std::cout << "Target sphere radius:" << " " << target_radius << " m\n";
    std::cout << "Body size:           " << body_size << " m\n";
    std::cout << "Penetration:         " << penetration << " m\n";
    std::cout << "Initial centroid r:  " << initial_centroid_radius << " m\n";
    std::cout << "Orbit omega:         " << orbit_omega << " rad/s\n";
    std::cout << "Max centroid speed:  " << pose_path.max_speed << " m/s\n";
    std::cout << "Expand factor:       " << expand_factor << " (-)\n";
    std::cout << "Total graze angle:   " << orbit_omega * cfg.total_time << " rad\n";
    std::cout << "Pose driver:         mesh-relative surface point + mesh-relative normal\n";
    PrintReference(summary.reference);
    std::cout << "=====================================================\n";

    std::vector<double> force_mags;
    std::vector<double> force_radial_components;

    const double csv_dt = 1.0 / static_cast<double>(cfg.csv_fps);
    const int n_steps = static_cast<int>(std::round(cfg.total_time / cfg.step_size));
    const int n_frames = static_cast<int>(std::round(cfg.total_time / cfg.frame_time));

    std::ofstream trace_csv(summary.out_dir / "force_trace.csv");
    trace_csv << std::setprecision(12);
    trace_csv << "sample,time_s,x_m,y_m,z_m,body_lowest_r_m,surface_x_m,surface_y_m,surface_z_m,normal_x,normal_y,normal_z,fx_N,fy_N,fz_N,fmag_N,fradial_N,target_mesh";
    if (summary.reference.available) {
        trace_csv << ",ref_total_N,ref_normal_N,rel_err_total_pct,rel_err_radial_pct";
    }
    trace_csv << "\n";

    int csv_sample_idx = 0;
    int visual_frame_idx = 0;
    double next_csv_time = csv_dt;
    double next_visual_time = static_cast<double>(cfg.frame_time);
    const double time_eps = 0.5 * static_cast<double>(cfg.step_size);

    for (int step_idx = 0; step_idx < n_steps; ++step_idx) {
        body.tracker->SetPos(pose_path.poses[static_cast<size_t>(step_idx)].pos);
        body.tracker->SetOriQ(pose_path.poses[static_cast<size_t>(step_idx)].quat);
        body.tracker->SetVel(pose_path.poses[static_cast<size_t>(step_idx)].vel);
        body.tracker->SetAngVel(pose_path.poses[static_cast<size_t>(step_idx)].angvel_local);
        body.reference_above_lowest_point = pose_path.poses[static_cast<size_t>(step_idx)].support_distance;

        DEMSim.DoDynamics(cfg.step_size);

        const size_t pose_out_idx = static_cast<size_t>(step_idx + 1);
        body.tracker->SetPos(pose_path.poses[pose_out_idx].pos);
        body.tracker->SetOriQ(pose_path.poses[pose_out_idx].quat);
        body.tracker->SetVel(pose_path.poses[pose_out_idx].vel);
        body.tracker->SetAngVel(pose_path.poses[pose_out_idx].angvel_local);
        body.reference_above_lowest_point = pose_path.poses[pose_out_idx].support_distance;

        const double time_s = static_cast<double>(step_idx + 1) * static_cast<double>(cfg.step_size);
        const float3 cnt_acc = body.tracker->ContactAcc();
        const float3 cnt_force = Scale3(cnt_acc, body.mass);
        const BodyPose& pose_out = pose_path.poses[pose_out_idx];
        const double force_mag = std::sqrt(static_cast<double>(cnt_force.x) * cnt_force.x +
                                           static_cast<double>(cnt_force.y) * cnt_force.y +
                                           static_cast<double>(cnt_force.z) * cnt_force.z);
        const double centroid_r = Length3(pose_out.pos);
        const double body_lowest_r = centroid_r - static_cast<double>(body.reference_above_lowest_point);
        const float3 radial_dir = (centroid_r > 1e-12) ? Scale3(pose_out.pos, static_cast<float>(1.0 / centroid_r))
                                                       : make_float3(1.f, 0.f, 0.f);
        const double force_radial = static_cast<double>(cnt_force.x) * radial_dir.x +
                                    static_cast<double>(cnt_force.y) * radial_dir.y +
                                    static_cast<double>(cnt_force.z) * radial_dir.z;

        if (time_s + time_eps >= next_csv_time) {
            ++csv_sample_idx;
            force_mags.push_back(force_mag);
            force_radial_components.push_back(force_radial);

            trace_csv << csv_sample_idx << ','
                      << time_s << ','
                      << pose_out.pos.x << ','
                      << pose_out.pos.y << ','
                      << pose_out.pos.z << ','
                      << body_lowest_r << ','
                      << pose_out.surface_point.x << ','
                      << pose_out.surface_point.y << ','
                      << pose_out.surface_point.z << ','
                      << pose_out.surface_normal.x << ','
                      << pose_out.surface_normal.y << ','
                      << pose_out.surface_normal.z << ','
                      << cnt_force.x << ','
                      << cnt_force.y << ','
                      << cnt_force.z << ','
                      << force_mag << ','
                      << force_radial << ','
                      << target_mesh_name;

            if (summary.reference.available) {
                const double rel_err_total = 100.0 * (force_mag - summary.reference.total_force) / summary.reference.total_force;
                const double rel_err_radial =
                    100.0 * (force_radial - summary.reference.normal_force) / summary.reference.normal_force;
                trace_csv << ',' << summary.reference.total_force
                          << ',' << summary.reference.normal_force
                          << ',' << rel_err_total
                          << ',' << rel_err_radial;
            }
            trace_csv << '\n';

            next_csv_time += csv_dt;
        }

        if (visual_frame_idx < n_frames && time_s + time_eps >= next_visual_time) {
            ++visual_frame_idx;

            char meshfilename[256];
            std::snprintf(meshfilename, sizeof(meshfilename), "mesh_%04d.vtk", visual_frame_idx);
            DEMSim.WriteMeshFile(summary.out_dir / meshfilename);

            std::cout << "t=" << time_s << " s"
                      << "  centroid_r=" << centroid_r
                      << "  body_lowest_r=" << body_lowest_r
                      << "  |F_cnt|=" << force_mag << " N"
                      << "  F_r=" << force_radial << " N"
                      << "  n=(" << pose_out.surface_normal.x << "," << pose_out.surface_normal.y << ","
                      << pose_out.surface_normal.z << ")";

            if (summary.reference.available && summary.reference.total_force > 0.0 && summary.reference.normal_force > 0.0) {
                const double rel_err_total = 100.0 * (force_mag - summary.reference.total_force) / summary.reference.total_force;
                const double rel_err_radial =
                    100.0 * (force_radial - summary.reference.normal_force) / summary.reference.normal_force;
                std::cout << "  ref|F|=" << summary.reference.total_force << " N"
                          << "  refF_n=" << summary.reference.normal_force << " N"
                          << "  err|F|=" << rel_err_total << " %"
                          << "  errFr=" << rel_err_radial << " %";
            }

            if (force_mag > 1e-12) {
                const double inv_f = 1.0 / force_mag;
                std::cout << "  F_dir=(" << cnt_force.x * inv_f << "," << cnt_force.y * inv_f << ","
                          << cnt_force.z * inv_f << ")";
            }
            std::cout << std::endl;

            next_visual_time += static_cast<double>(cfg.frame_time);
        }
    }

    if (force_mags.empty()) {
        throw std::runtime_error("No force data collected.");
    }

    summary.mean_force = 0.0;
    summary.min_force = force_mags.front();
    summary.max_force = force_mags.front();
    summary.mean_radial = 0.0;
    summary.min_radial = force_radial_components.front();
    summary.max_radial = force_radial_components.front();

    for (size_t i = 0; i < force_mags.size(); ++i) {
        summary.mean_force += force_mags[i];
        summary.mean_radial += force_radial_components[i];
        summary.min_force = std::min(summary.min_force, force_mags[i]);
        summary.max_force = std::max(summary.max_force, force_mags[i]);
        summary.min_radial = std::min(summary.min_radial, force_radial_components[i]);
        summary.max_radial = std::max(summary.max_radial, force_radial_components[i]);
    }

    summary.mean_force /= static_cast<double>(force_mags.size());
    summary.mean_radial /= static_cast<double>(force_radial_components.size());

    double variance = 0.0;
    for (double f : force_mags) {
        const double d = f - summary.mean_force;
        variance += d * d;
    }
    summary.stddev_force = std::sqrt(variance / static_cast<double>(force_mags.size()));

    std::cout << "\n=== Statistics for " << summary.label << " ===" << std::endl;
    std::cout << "  Mean |F|:   " << summary.mean_force << " N" << std::endl;
    std::cout << "  Min |F|:    " << summary.min_force << " N" << std::endl;
    std::cout << "  Max |F|:    " << summary.max_force << " N" << std::endl;
    std::cout << "  StdDev |F|: " << summary.stddev_force << " N" << std::endl;
    std::cout << "  Mean Fr:    " << summary.mean_radial << " N" << std::endl;
    std::cout << "  Min Fr:     " << summary.min_radial << " N" << std::endl;
    std::cout << "  Max Fr:     " << summary.max_radial << " N" << std::endl;

    if (summary.reference.available && summary.reference.total_force > 0.0 && summary.reference.normal_force > 0.0) {
        const double rel_err_total = 100.0 * (summary.mean_force - summary.reference.total_force) / summary.reference.total_force;
        const double rel_err_radial =
            100.0 * (summary.mean_radial - summary.reference.normal_force) / summary.reference.normal_force;
        std::cout << "  Ref |F|:    " << summary.reference.total_force << " N" << std::endl;
        std::cout << "  Ref F_n:    " << summary.reference.normal_force << " N" << std::endl;
        std::cout << "  Rel. error |F| vs reference: " << rel_err_total << " %" << std::endl;
        std::cout << "  Rel. error Fr vs reference:  " << rel_err_radial << " %" << std::endl;
    }

    DEMSim.ShowTimingStats();
    std::cout << "=====================================================\n";
    return summary;
}

void WriteStudySummaryCsv(const path& filename, const std::vector<RunSummary>& results) {
    std::ofstream out(filename);
    out << "case_label,target_mesh,mean_force_N,min_force_N,max_force_N,stddev_force_N,mean_radial_N,min_radial_N,max_radial_N,reference_total_N,reference_normal_N,delta_vs_highres_pct\n";

    double reference_highres_force = std::numeric_limits<double>::quiet_NaN();
    for (const auto& result : results) {
        if (result.target_mesh == "sphere_highres.obj") {
            reference_highres_force = result.mean_force;
        }
    }

    for (const auto& result : results) {
        double delta_vs_highres = std::numeric_limits<double>::quiet_NaN();
        if (!std::isnan(reference_highres_force) && reference_highres_force != 0.0) {
            delta_vs_highres = 100.0 * (result.mean_force - reference_highres_force) / reference_highres_force;
        }
        out << result.label << ','
            << result.target_mesh << ','
            << result.mean_force << ','
            << result.min_force << ','
            << result.max_force << ','
            << result.stddev_force << ','
            << result.mean_radial << ','
            << result.min_radial << ','
            << result.max_radial << ','
            << (result.reference.available ? result.reference.total_force : std::numeric_limits<double>::quiet_NaN()) << ','
            << (result.reference.available ? result.reference.normal_force : std::numeric_limits<double>::quiet_NaN()) << ','
            << delta_vs_highres << '\n';
    }
}

void PrintStudyComparison(const std::vector<RunSummary>& results) {
    if (results.empty())
        return;

    double highres_force = std::numeric_limits<double>::quiet_NaN();
    for (const auto& result : results) {
        if (result.target_mesh == "sphere_highres.obj") {
            highres_force = result.mean_force;
        }
    }

    std::cout << "\n================ Target sphere mesh comparison ================\n";
    std::cout << std::setw(20) << "Target mesh"
              << std::setw(18) << "Mean |F| [N]"
              << std::setw(18) << "Mean Fr [N]"
              << std::setw(18) << "StdDev [N]"
              << std::setw(22) << "Delta vs highres [%]" << std::endl;

    for (const auto& result : results) {
        double delta_vs_highres = std::numeric_limits<double>::quiet_NaN();
        if (!std::isnan(highres_force) && highres_force != 0.0) {
            delta_vs_highres = 100.0 * (result.mean_force - highres_force) / highres_force;
        }
        std::cout << std::setw(20) << result.target_mesh
                  << std::setw(18) << result.mean_force
                  << std::setw(18) << result.mean_radial
                  << std::setw(18) << result.stddev_force
                  << std::setw(22) << delta_vs_highres << std::endl;
    }
    std::cout << "================================================================\n";
}

void WriteKTSummaryCsv(const path& filename, const std::vector<std::pair<double, RunSummary>>& results) {
    std::ofstream out(filename);
    out << "kT_scale,case_label,target_mesh,mean_force_N,min_force_N,max_force_N,stddev_force_N,mean_radial_N,min_radial_N,max_radial_N,reference_total_N,reference_normal_N\n";
    for (const auto& entry : results) {
        const double kT = entry.first;
        const auto& result = entry.second;
        out << kT << ','
            << result.label << ','
            << result.target_mesh << ','
            << result.mean_force << ','
            << result.min_force << ','
            << result.max_force << ','
            << result.stddev_force << ','
            << result.mean_radial << ','
            << result.min_radial << ','
            << result.max_radial << ','
            << (result.reference.available ? result.reference.total_force : std::numeric_limits<double>::quiet_NaN()) << ','
            << (result.reference.available ? result.reference.normal_force : std::numeric_limits<double>::quiet_NaN()) << '\n';
    }
}

void PrintKTTopN(const std::vector<std::pair<double, RunSummary>>& results, int top_n) {
    if (results.empty())
        return;

    std::vector<std::pair<double, RunSummary>> sorted = results;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.second.mean_force > b.second.mean_force;
    });

    const int n = std::min<int>(top_n, static_cast<int>(sorted.size()));
    std::cout << "\n================ Top " << n << " kT cases by mean |F| ================\n";
    std::cout << std::setw(12) << "kT"
              << std::setw(18) << "Mean |F| [N]"
              << std::setw(18) << "Mean Fr [N]"
              << std::setw(18) << "StdDev [N]"
              << std::setw(18) << "Target mesh"
              << std::setw(18) << "Case" << std::endl;

    for (int i = 0; i < n; ++i) {
        const auto& [kT, result] = sorted[i];
        std::cout << std::setw(12) << kT
                  << std::setw(18) << result.mean_force
                  << std::setw(18) << result.mean_radial
                  << std::setw(18) << result.stddev_force
                  << std::setw(18) << result.target_mesh
                  << std::setw(18) << result.label << std::endl;
    }
    std::cout << "========================================================\n";
}

void RunFullKTStudy(const DemoConfig& base_cfg, const path& root_out_dir) {
    std::vector<std::pair<double, RunSummary>> all_results;

    std::vector<std::string> mesh_cases;
    if (base_cfg.target_mesh_study) {
        for (const char* mesh_name : TARGET_SPHERE_MESH_CASES) {
            mesh_cases.emplace_back(mesh_name);
        }
    } else {
        mesh_cases.push_back(base_cfg.single_run_target_sphere_mesh);
    }

    for (double kT : base_cfg.kt_sweep) {
        DemoConfig cfg = base_cfg;
        cfg.kT_study = false;
        cfg.speed_scale = base_cfg.speed_scale * static_cast<float>(kT);

        std::cout << "\n################ kT sweep case: " << kT << " ################\n";

        for (const auto& target_mesh_name : mesh_cases) {
            ApplyVariantToConfig(cfg, DemoVariant::CONE);
            cfg.target_mesh_study = false;
            all_results.emplace_back(kT, RunSingleCase(cfg, target_mesh_name, root_out_dir / "kT_study" / "cone" / MeshLabel(target_mesh_name)));

            ApplyVariantToConfig(cfg, DemoVariant::CUBE);
            cfg.cube_mode = CubeContactMode::EDGE;
            cfg.target_mesh_study = false;
            all_results.emplace_back(kT, RunSingleCase(cfg, target_mesh_name, root_out_dir / "kT_study" / "cube-edge" / MeshLabel(target_mesh_name)));

            ApplyVariantToConfig(cfg, DemoVariant::CUBE);
            cfg.cube_mode = CubeContactMode::TIP;
            cfg.target_mesh_study = false;
            all_results.emplace_back(kT, RunSingleCase(cfg, target_mesh_name, root_out_dir / "kT_study" / "cube-tip" / MeshLabel(target_mesh_name)));

            ApplyVariantToConfig(cfg, DemoVariant::SPHERE);
            cfg.target_mesh_study = false;
            all_results.emplace_back(kT, RunSingleCase(cfg, target_mesh_name, root_out_dir / "kT_study" / "sphere" / MeshLabel(target_mesh_name)));
        }
    }

    WriteKTSummaryCsv(root_out_dir / "kT_study_summary.csv", all_results);
    PrintKTTopN(all_results, base_cfg.kt_top_n);
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        DemoConfig cfg = ParseArguments(argc, argv);

        const path root_out_dir = current_path() / "modular_test_output" / "DEMTest_GrazingSphereVariants" / CaseName(cfg);
        std::error_code dir_ec;
        create_directories(root_out_dir, dir_ec);
        if (dir_ec || !is_directory(root_out_dir)) {
            std::cerr << "Failed to create root output directory: " << root_out_dir << std::endl;
            return 1;
        }

        if (cfg.kT_study) {
            RunFullKTStudy(cfg, root_out_dir);
        } else {
            std::vector<RunSummary> results;
            if (cfg.target_mesh_study) {
                for (const char* mesh_name : TARGET_SPHERE_MESH_CASES) {
                    results.push_back(RunSingleCase(cfg, mesh_name, root_out_dir));
                }
                PrintStudyComparison(results);
                WriteStudySummaryCsv(root_out_dir / "target_mesh_study_summary.csv", results);
            } else {
                results.push_back(RunSingleCase(cfg, cfg.single_run_target_sphere_mesh, root_out_dir));
                WriteStudySummaryCsv(root_out_dir / "single_run_summary.csv", results);
            }
        }

        std::cout << "DEMTest_GrazingSphereVariants exiting..." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
