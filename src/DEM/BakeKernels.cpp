// Copyright (c) 2024, SBEL GPU Development Team
// SPDX-License-Identifier: BSD-3-Clause

#include <DEM/API.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " [--cache-dir PATH] [--cache-tag TAG] [--materials N] [--require-cache] [--quiet]" << std::endl;
}

bool parse_uint(const std::string& text, size_t& out) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    unsigned long long val = std::strtoull(text.c_str(), &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    out = static_cast<size_t>(val);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string cache_dir;
    std::string cache_tag;
    size_t material_count = 2;
    bool require_cache = false;
    bool quiet = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--cache-dir" && i + 1 < argc) {
            cache_dir = argv[++i];
            continue;
        }
        if (arg.rfind("--cache-dir=", 0) == 0) {
            cache_dir = arg.substr(std::string("--cache-dir=").size());
            continue;
        }
        if (arg == "--cache-tag" && i + 1 < argc) {
            cache_tag = argv[++i];
            continue;
        }
        if (arg.rfind("--cache-tag=", 0) == 0) {
            cache_tag = arg.substr(std::string("--cache-tag=").size());
            continue;
        }
        if ((arg == "--materials" || arg == "--material-count") && i + 1 < argc) {
            size_t val = 0;
            if (!parse_uint(argv[++i], val)) {
                std::cerr << "Invalid material count." << std::endl;
                return 1;
            }
            material_count = val;
            continue;
        }
        if (arg.rfind("--materials=", 0) == 0) {
            size_t val = 0;
            if (!parse_uint(arg.substr(std::string("--materials=").size()), val)) {
                std::cerr << "Invalid material count." << std::endl;
                return 1;
            }
            material_count = val;
            continue;
        }
        if (arg == "--require-cache") {
            require_cache = true;
            continue;
        }
        if (arg == "--quiet") {
            quiet = true;
            continue;
        }

        std::cerr << "Unknown option: " << arg << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (material_count == 0) {
        material_count = 1;
    }

    deme::DEMSolver solver;
    if (quiet) {
        solver.SetVerbosity(deme::VERBOSITY_ERROR);
    }

    solver.UseFrictionalHertzianModel();
    solver.UseBakedKernelCache(true);
    if (!cache_dir.empty()) {
        solver.SetKernelCacheDir(cache_dir);
    }
    if (!cache_tag.empty()) {
        solver.SetKernelCacheTag(cache_tag);
    }
    solver.RequireKernelCache(require_cache);

    for (size_t i = 0; i < material_count; ++i) {
        solver.LoadMaterial({{"E", 1e6f}, {"nu", 0.3f}, {"CoR", 0.6f}, {"mu", 0.5f}, {"Crr", 0.01f}});
    }

    solver.Initialize(/*dry_run=*/false);
    return 0;
}
