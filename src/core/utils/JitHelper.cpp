//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//  SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <regex>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <mutex>
#include <memory>

#include <core/ApiVersion.h>
#include "RuntimeData.h"
#include "JitHelper.h"

namespace {

constexpr uint64_t kFNVOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFNVPrime = 0x100000001b3ULL;

constexpr int getCudaVersion() {
#ifdef CUDA_VERSION
    return CUDA_VERSION;
#else
    return CUDART_VERSION;
#endif
}

std::string sanitizeFilename(const std::string& name) {
    std::string sanitized = name;
    for (auto& c : sanitized) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) {
            c = '_';
        }
    }
    return sanitized;
}

void appendCudaIncludeFlags(std::vector<std::string>& flags) {
    std::vector<std::filesystem::path> include_paths;
    {
        std::string dirs = DEME_CUDA_TOOLKIT_INCLUDE_DIRS;  // "dir1;dir2;dir3"
        std::stringstream ss(dirs);
        std::string dir;
        while (std::getline(ss, dir, ';')) {
            if (!dir.empty()) {
                include_paths.emplace_back(dir);
            }
        }
    }
    auto add_inc = [&](const std::filesystem::path& p) {
        std::string inc_flag = "-I" + p.string();
        if (std::find(flags.begin(), flags.end(), inc_flag) == flags.end()) {
            flags.push_back(inc_flag);
        }
    };
    for (auto& p : include_paths) {
        add_inc(p);
        auto cccl = p / "cccl";
        add_inc(cccl);
    }
}

std::string applySubstitutions(std::string code,
                               const std::unordered_map<std::string, std::string>& substitutions) {
    std::vector<std::pair<std::string, std::string>> ordered_subs(substitutions.begin(), substitutions.end());
    std::sort(ordered_subs.begin(), ordered_subs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (auto& subst : ordered_subs) {
        code = std::regex_replace(code, std::regex(subst.first), subst.second);
    }
    return code;
}

std::mutex g_bake_mutex;

std::string quoteArg(const std::filesystem::path& path) {
    std::string out = path.string();
    std::string escaped;
    escaped.reserve(out.size());
    for (char c : out) {
        if (c == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(c);
    }
    return "\"" + escaped + "\"";
}

int runNvccCompile(const std::filesystem::path& source,
                   const std::filesystem::path& fatbin,
                   const std::vector<std::string>& flags,
                   const std::string& arch_tag,
                   const std::filesystem::path& log_path) {
    std::ostringstream cmd;
    cmd << "nvcc --fatbin -arch=" << arch_tag << " -DDEME_DEFINE_SIM_PARAMS_CONST";
    for (const auto& flag : flags) {
        if (flag.rfind("-diag-suppress=", 0) == 0 || flag.rfind("-diag_suppress=", 0) == 0) {
            continue;
        }
        cmd << " " << flag;
    }
    cmd << " -o " << quoteArg(fatbin) << " " << quoteArg(source);
    cmd << " > " << quoteArg(log_path) << " 2>&1";
    return std::system(cmd.str().c_str());
}

}  // namespace

const std::filesystem::path JitHelper::KERNEL_DIR = DEMERuntimeDataHelper::data_path / "kernel";
const std::filesystem::path JitHelper::KERNEL_INCLUDE_DIR = DEMERuntimeDataHelper::include_path;
std::filesystem::path JitHelper::s_cache_dir = JitHelper::resolveCacheDir();
std::string JitHelper::s_cache_tag;
bool JitHelper::s_require_cache = false;

void JitHelper::setCacheDir(const std::filesystem::path& dir) {
    if (dir.empty()) {
        s_cache_dir = resolveCacheDir();
    } else {
        s_cache_dir = dir;
    }
}

const std::filesystem::path& JitHelper::getCacheDir() {
    return s_cache_dir;
}

void JitHelper::setCacheTag(const std::string& tag) {
    s_cache_tag = sanitizeFilename(tag);
}

const std::string& JitHelper::getCacheTag() {
    return s_cache_tag;
}

void JitHelper::setRequireCache(bool require) {
    s_require_cache = require;
}

bool JitHelper::getRequireCache() {
    return s_require_cache;
}

JitHelper::Header::Header(const std::filesystem::path& sourcefile) {
    this->_source = JitHelper::loadSourceFile(sourcefile);
}

const std::string& JitHelper::Header::getSource() {
    return _source;
}

void JitHelper::Header::substitute(const std::string& symbol, const std::string& value) {
    // find occurrences of `symbol` until there are none left
    for (size_t p = this->_source.find(symbol); p != std::string::npos; p = this->_source.find(symbol)) {
        // Replace this occurrence with the new value
        this->_source.replace(p, symbol.length(), value);
    }
}

JitHelper::CachedProgram JitHelper::buildProgram(const std::string& name,
                                                 const std::filesystem::path& source,
                                                 std::unordered_map<std::string, std::string> substitutions,
                                                 std::vector<std::string> flags) {
    std::string source_code = JitHelper::loadSourceFile(source);
    source_code = applySubstitutions(std::move(source_code), substitutions);
    std::string code = name + "\n" + source_code;

    if (std::find(flags.begin(), flags.end(), "-std=c++17") == flags.end()) {
        flags.push_back("-std=c++17");
    }
    appendCudaIncludeFlags(flags);

    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess) {
        device = 0;
    }
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        prop.major = 0;
        prop.minor = 0;
    }
    const std::string arch_tag = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);

    std::vector<std::string> flags_sorted = flags;
    std::sort(flags_sorted.begin(), flags_sorted.end());
    std::string flags_sig;
    for (const auto& flag : flags_sorted) {
        flags_sig += flag;
        flags_sig += ";";
    }
    const std::string fingerprint = code + "|flags:" + flags_sig + "|api:" + std::to_string(DEME_API_VERSION) +
                                    "|cuda:" + std::to_string(getCudaVersion()) + "|arch:" + arch_tag;
    std::string program_hash = hashString(fingerprint);

    std::filesystem::path program_dir = getCacheDir();
    const std::string& tag = getCacheTag();
    if (!tag.empty()) {
        program_dir /= tag;
    }
    program_dir /= program_hash;
    std::error_code ec;
    std::filesystem::create_directories(program_dir, ec);
    std::ofstream fp_out(program_dir / "fingerprint.txt", std::ios::trunc);
    if (fp_out) {
        fp_out << program_hash << "\n" << fingerprint;
    }

    const std::string name_sanitized = sanitizeFilename(name);
    const std::filesystem::path source_out = program_dir / (name_sanitized + ".cu");
    const std::filesystem::path fatbin_out = program_dir / (name_sanitized + ".fatbin");
    const std::filesystem::path log_out = program_dir / (name_sanitized + ".log");

    if (!std::filesystem::exists(fatbin_out)) {
        if (getRequireCache()) {
            throw std::runtime_error("Baked kernel cache missing for program " + name);
        }
        std::lock_guard<std::mutex> lock(g_bake_mutex);
        if (!std::filesystem::exists(fatbin_out)) {
            std::ofstream src_out(source_out, std::ios::trunc);
            if (!src_out) {
                throw std::runtime_error("Failed to write baked kernel source for " + name);
            }
            src_out << source_code;
            src_out.close();

            std::cout << "bake-compiling for " << name << " ..." << std::endl;
            int ret = runNvccCompile(source_out, fatbin_out, flags, arch_tag, log_out);
            if (ret != 0 || !std::filesystem::exists(fatbin_out)) {
                std::ifstream log_in(log_out);
                if (log_in) {
                    std::cerr << log_in.rdbuf();
                }
                throw std::runtime_error("Baked kernel compilation failed for " + name);
            }
        }
    }

    CUresult init_res = cuInit(0);
    if (init_res != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to initialize CUDA driver API");
    }
    CUmodule module = nullptr;
    CUresult load_res = cuModuleLoad(&module, fatbin_out.string().c_str());
    if (load_res != CUDA_SUCCESS || module == nullptr) {
        throw std::runtime_error("Failed to load baked kernel module for " + name);
    }

    auto storage = std::make_shared<CachedProgram::ProgramStorage>(code, flags);
    storage->programHash = program_hash;
    storage->cacheDir = program_dir;
    storage->device = device;
    storage->archTag = arch_tag;
    storage->module = module;
    return CachedProgram(storage);
}

std::string JitHelper::hashString(const std::string& in) {
    uint64_t hash = kFNVOffset;
    for (unsigned char c : in) {
        hash ^= c;
        hash *= kFNVPrime;
    }
    return toHex(hash);
}

std::string JitHelper::toHex(uint64_t value) {
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << value;
    return oss.str();
}

std::filesystem::path JitHelper::resolveCacheDir() {
    if (const char* env = std::getenv("DEME_KERNEL_CACHE_DIR")) {
        return std::filesystem::path(env);
    }
    if (const char* env = std::getenv("DEME_JIT_CACHE_DIR")) {
        return std::filesystem::path(env);
    }
    // Prefer to keep cache alongside other runtime data in the build tree
    std::filesystem::path default_path = DEMERuntimeDataHelper::data_path / "baked";
    if (std::filesystem::exists(DEMERuntimeDataHelper::data_path)) {
        return default_path;
    }
    return std::filesystem::temp_directory_path() / "dem-baked";
}
JitHelper::CachedProgram::CachedProgram(std::shared_ptr<ProgramStorage> storage) : m_storage(std::move(storage)) {}
JitHelper::CachedProgram::Kernel JitHelper::CachedProgram::kernel(const std::string& name,
                                                                  std::vector<std::string> options) const {
    return Kernel(m_storage, name, std::move(options));
}
JitHelper::CachedProgram::ProgramStorage::ProgramStorage(std::string code_in, std::vector<std::string> flags_in)
    : code(std::move(code_in)), flags(std::move(flags_in)) {}
JitHelper::CachedProgram::ProgramStorage::~ProgramStorage() {
    if (module) {
        cuModuleUnload(module);
        module = nullptr;
    }
}
JitHelper::CachedProgram::Kernel::Kernel(std::shared_ptr<ProgramStorage> storage,
                                         std::string name,
                                         std::vector<std::string> options)
    : m_storage(std::move(storage)), m_name(std::move(name)), m_options(std::move(options)) {}

JitHelper::CachedProgram::Kernel::KernelInstantiation JitHelper::CachedProgram::Kernel::getKernelInstantiation(
    const std::vector<std::string>& template_args) const {
    if (!template_args.empty()) {
        throw std::runtime_error("Template arguments are not supported for baked kernels.");
    }
    std::lock_guard<std::mutex> storage_lock(m_storage->mutex);
    if (auto it = m_storage->bakedKernelCache.find(m_name); it != m_storage->bakedKernelCache.end()) {
        return KernelInstantiation(m_storage, it->second);
    }
    CUfunction func = nullptr;
    CUresult res = cuModuleGetFunction(&func, m_storage->module, m_name.c_str());
    if (res != CUDA_SUCCESS || func == nullptr) {
        throw std::runtime_error("Failed to find baked kernel " + m_name);
    }
    m_storage->bakedKernelCache.emplace(m_name, func);
    return KernelInstantiation(m_storage, func);
}

JitHelper::CachedProgram::Kernel::KernelInstantiation JitHelper::CachedProgram::Kernel::instantiate(
    std::vector<std::string> template_args) const {
    return getKernelInstantiation(template_args);
}

JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelInstantiation(
    std::shared_ptr<ProgramStorage> storage,
    CUfunction func)
    : m_storage(std::move(storage)), m_func(func) {}
