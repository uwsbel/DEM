//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//  SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <mutex>
#include <memory>

#include <core/ApiVersion.h>
#include "RuntimeData.h"
#include "JitHelper.h"

namespace {

constexpr uint64_t kFNVOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFNVPrime = 0x100000001b3ULL;

#ifdef DEME_USE_HIP
int getHipRuntimeVersion() {
    int version = 0;
    const hipError_t err = hipRuntimeGetVersion(&version);
    if (err != hipSuccess) {
        return 0;
    }
    return version;
}

void unloadHipModuleNoThrow(hipModule_t& module) {
    if (module == nullptr) {
        return;
    }
    const hipError_t err = hipModuleUnload(module);
    if (err != hipSuccess) {
        std::cerr << "hipModuleUnload failed: " << hipGetErrorString(err) << std::endl;
    }
    module = nullptr;
}

std::string getHipRtcVersionString() {
    int major = 0;
    int minor = 0;
    if (hiprtcVersion(&major, &minor) == HIPRTC_SUCCESS) {
        return std::to_string(major) + "." + std::to_string(minor);
    }
    return "unknown";
}
#else
constexpr int getCudaVersion() {
#ifdef CUDA_VERSION
    return CUDA_VERSION;
#else
    return CUDART_VERSION;
#endif
}
#endif

std::string sanitizeFilename(const std::string& name) {
    std::string sanitized = name;
    for (auto& c : sanitized) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) {
            c = '_';
        }
    }
    return sanitized;
}

std::vector<std::filesystem::path> extractIncludePaths(const std::vector<std::string>& flags) {
    std::vector<std::filesystem::path> include_paths;

    auto append_path = [&](const std::string& raw_path) {
        if (!raw_path.empty()) {
            include_paths.emplace_back(raw_path);
        }
    };

    for (size_t i = 0; i < flags.size(); ++i) {
        const auto& flag = flags[i];
        if (flag.rfind("-I", 0) == 0 && flag.size() > 2) {
            append_path(flag.substr(2));
            continue;
        }
        if ((flag == "-I" || flag == "-isystem") && i + 1 < flags.size()) {
            append_path(flags[++i]);
            continue;
        }
        if (flag.rfind("--include-path=", 0) == 0) {
            append_path(flag.substr(std::string("--include-path=").size()));
            continue;
        }
    }

    return include_paths;
}

void appendUniqueIncludeFlag(std::vector<std::string>& flags, const std::filesystem::path& path) {
    if (path.empty()) {
        return;
    }

    const std::string inc_flag = "-I" + path.string();
    if (std::find(flags.begin(), flags.end(), inc_flag) == flags.end()) {
        flags.push_back(inc_flag);
    }
}

std::filesystem::path normalizePath(const std::filesystem::path& path) {
    std::error_code ec;
    auto normalized = std::filesystem::weakly_canonical(path, ec);
    return ec ? path.lexically_normal() : normalized;
}

#ifdef DEME_USE_HIP
bool matchQuotedInclude(const std::string& line, std::string& include_name) {
    static const std::regex kQuotedInclude(R"___(^\s*#\s*include\s*"([^"]+)")___");
    std::smatch match;
    if (!std::regex_search(line, match, kQuotedInclude)) {
        return false;
    }
    include_name = match[1].str();
    return true;
}

std::filesystem::path resolveQuotedInclude(const std::string& include_name,
                                           const std::filesystem::path& current_dir,
                                           const std::vector<std::filesystem::path>& include_paths) {
    const auto direct_candidate = current_dir / include_name;
    if (std::filesystem::exists(direct_candidate)) {
        return normalizePath(direct_candidate);
    }
    for (const auto& include_path : include_paths) {
        const auto candidate = include_path / include_name;
        if (std::filesystem::exists(candidate)) {
            return normalizePath(candidate);
        }
    }
    return {};
}

std::string expandQuotedIncludes(const std::filesystem::path& file,
                                 const std::vector<std::filesystem::path>& include_paths,
                                 std::unordered_set<std::string>& active_stack) {
    const auto normalized = normalizePath(file);
    const auto normalized_key = normalized.string();
    if (!active_stack.insert(normalized_key).second) {
        std::ifstream nested_input(normalized);
        return std::string((std::istreambuf_iterator<char>(nested_input)), std::istreambuf_iterator<char>());
    }

    std::ifstream input(normalized);
    std::ostringstream expanded;
    std::string line;
    while (std::getline(input, line)) {
        std::string include_name;
        if (matchQuotedInclude(line, include_name)) {
            const auto resolved = resolveQuotedInclude(include_name, normalized.parent_path(), include_paths);
            if (!resolved.empty()) {
                expanded << "// begin include: " << resolved.string() << "\n";
                expanded << expandQuotedIncludes(resolved, include_paths, active_stack);
                if (expanded.tellp() > 0 && expanded.str().back() != '\n') {
                    expanded << "\n";
                }
                expanded << "// end include: " << resolved.string() << "\n";
                continue;
            }
        }
        expanded << line << "\n";
    }

    active_stack.erase(normalized_key);
    return expanded.str();
}

std::vector<std::string> mergeHipCompileOptions(const std::vector<std::string>& base_flags,
                                                const std::vector<std::string>& kernel_options,
                                                const std::string& arch_tag) {
    std::vector<std::string> merged = base_flags;
    for (const auto& option : kernel_options) {
        if (std::find(merged.begin(), merged.end(), option) == merged.end()) {
            merged.push_back(option);
        }
    }

    const bool has_gpu_arch = std::any_of(merged.begin(), merged.end(), [](const std::string& flag) {
        return flag.rfind("--gpu-architecture=", 0) == 0 || flag.rfind("--offload-arch=", 0) == 0;
    });
    if (!has_gpu_arch && !arch_tag.empty() && arch_tag != "unknown") {
        merged.push_back("--gpu-architecture=" + arch_tag);
    }

    return merged;
}

std::string makeHipNameExpression(const std::string& kernel_name, const std::vector<std::string>& template_args) {
    if (template_args.empty()) {
        return kernel_name;
    }

    std::ostringstream expr;
    expr << kernel_name << "<";
    for (size_t i = 0; i < template_args.size(); ++i) {
        if (i > 0) {
            expr << ", ";
        }
        expr << template_args[i];
    }
    expr << ">";
    return expr.str();
}

[[noreturn]] void throwHipError(const std::string& what, hipError_t status) {
    throw std::runtime_error(what + ": " + hipGetErrorString(status));
}

[[noreturn]] void throwHipRtcError(const std::string& what, hiprtcResult status, const std::string& log = {}) {
    std::string message = what + ": " + hiprtcGetErrorString(status);
    if (!log.empty()) {
        message += "\n";
        message += log;
    }
    throw std::runtime_error(message);
}

void checkHip(hipError_t status, const std::string& what) {
    if (status != hipSuccess) {
        throwHipError(what, status);
    }
}

void checkHipRtc(hiprtcResult status, const std::string& what, const std::string& log = {}) {
    if (status != HIPRTC_SUCCESS) {
        throwHipRtcError(what, status, log);
    }
}
#endif

}  // namespace

const std::filesystem::path JitHelper::KERNEL_DIR = DEMERuntimeDataHelper::data_path / "kernel";
const std::filesystem::path JitHelper::KERNEL_INCLUDE_DIR = DEMERuntimeDataHelper::include_path;
const std::filesystem::path JitHelper::CACHE_DIR = JitHelper::resolveCacheDir();

JitHelper::Header::Header(const std::filesystem::path& sourcefile) {
    this->_source = JitHelper::loadSourceFile(sourcefile);
}

const std::string& JitHelper::Header::getSource() {
    return _source;
}

void JitHelper::Header::substitute(const std::string& symbol, const std::string& value) {
    for (size_t p = this->_source.find(symbol); p != std::string::npos; p = this->_source.find(symbol)) {
        this->_source.replace(p, symbol.length(), value);
    }
}

JitHelper::CachedProgram JitHelper::buildProgram(const std::string& name,
                                                 const std::filesystem::path& source,
                                                 std::unordered_map<std::string, std::string> substitutions,
                                                 std::vector<std::string> flags) {
    std::string code = "// runtime unit: " + name + "\n";

#ifdef DEME_USE_HIP
    std::vector<std::filesystem::path> include_paths = extractIncludePaths(flags);
    include_paths.push_back(source.parent_path());
    include_paths.push_back(KERNEL_INCLUDE_DIR);
    appendUniqueIncludeFlag(flags, source.parent_path());
    appendUniqueIncludeFlag(flags, KERNEL_INCLUDE_DIR);
    std::unordered_set<std::string> active_stack;
    code.append(expandQuotedIncludes(source, include_paths, active_stack));
#else
    code.append(JitHelper::loadSourceFile(source));
#endif

    std::vector<std::pair<std::string, std::string>> ordered_subs(substitutions.begin(), substitutions.end());
    std::sort(ordered_subs.begin(), ordered_subs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (auto& subst : ordered_subs) {
        code = std::regex_replace(code, std::regex(subst.first), subst.second);
    }
#ifdef DEME_USE_HIP
    {
        const std::string cuda_trap = "asm volatile(\"trap;\");";
        const std::string hip_trap = "__builtin_trap();";
        for (size_t pos = code.find(cuda_trap); pos != std::string::npos; pos = code.find(cuda_trap, pos + hip_trap.size())) {
            code.replace(pos, cuda_trap.size(), hip_trap);
        }
    }
#endif

#ifndef DEME_USE_HIP
    {
        std::vector<std::filesystem::path> include_paths;
        {
            std::string dirs = DEME_CUDA_TOOLKIT_HEADERS;
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
#endif

#ifdef DEME_USE_HIP
    int device = 0;
    hipDeviceProp_t prop{};
    std::string arch_tag = "unknown";
    if (hipGetDevice(&device) != hipSuccess) {
        device = 0;
    }
    if (hipGetDeviceProperties(&prop, device) == hipSuccess && prop.gcnArchName[0] != '\0') {
        arch_tag = prop.gcnArchName;
    }

    std::vector<std::string> flags_sorted = flags;
    std::sort(flags_sorted.begin(), flags_sorted.end());
    std::ostringstream flags_sig_stream;
    for (const auto& flag : flags_sorted) {
        flags_sig_stream << flag << '\n';
    }
    const std::string flags_sig = flags_sig_stream.str();
    const std::string fingerprint = code + "|flags:" + flags_sig + "|api:" + std::to_string(DEME_API_VERSION) +
                                    "|hip_runtime:" + std::to_string(getHipRuntimeVersion()) +
                                    "|hiprtc:" + getHipRtcVersionString() + "|arch:" + arch_tag;
#else
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
    const std::string flags_sig = jitify::reflection::reflect_list(flags_sorted);
    const std::string fingerprint = code + "|flags:" + flags_sig + "|api:" + std::to_string(DEME_API_VERSION) +
                                    "|cuda:" + std::to_string(getCudaVersion()) + "|arch:" + arch_tag;
#endif

    std::string program_hash = hashString(fingerprint);

    const auto program_dir = CACHE_DIR / program_hash;
    auto storage = std::make_shared<CachedProgram::ProgramStorage>(code, flags);
    storage->programHash = program_hash;
    storage->cacheDir = program_dir;
    storage->device = device;
    storage->archTag = arch_tag;

    std::error_code ec;
    std::filesystem::create_directories(storage->cacheDir, ec);
    std::ofstream fp_out(storage->cacheDir / "fingerprint.txt", std::ios::trunc);
    if (fp_out) {
        fp_out << program_hash << "\n" << fingerprint;
    }

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
    if (const char* env = std::getenv("DEME_JIT_CACHE_DIR")) {
        return std::filesystem::path(env);
    }
    std::filesystem::path default_path = DEMERuntimeDataHelper::data_path / "jit_cache";
    if (std::filesystem::exists(DEMERuntimeDataHelper::data_path)) {
        return default_path;
    }
    return std::filesystem::temp_directory_path() / "dem-jit";
}

#ifdef DEME_USE_HIP

JitHelper::CachedProgram::LoadedKernelData::~LoadedKernelData() {
    unloadHipModuleNoThrow(module);
}

JitHelper::CachedProgram::CachedProgram(std::shared_ptr<ProgramStorage> storage) : m_storage(std::move(storage)) {}

JitHelper::CachedProgram::Kernel JitHelper::CachedProgram::kernel(const std::string& name,
                                                                  std::vector<std::string> options) const {
    return Kernel(m_storage, name, std::move(options));
}

JitHelper::CachedProgram::ProgramStorage::ProgramStorage(std::string code_in, std::vector<std::string> flags_in)
    : code(std::move(code_in)), flags(std::move(flags_in)) {}

JitHelper::CachedProgram::Kernel::Kernel(std::shared_ptr<ProgramStorage> storage,
                                         std::string name,
                                         std::vector<std::string> options)
    : m_storage(std::move(storage)), m_name(std::move(name)), m_options(std::move(options)) {}

std::shared_ptr<JitHelper::CachedProgram::LoadedKernelData> JitHelper::CachedProgram::Kernel::getLoadedKernel(
    const std::vector<std::string>& template_args) const {
    const std::string name_expression = makeHipNameExpression(m_name, template_args);

    std::vector<std::string> options_sorted = m_options;
    std::sort(options_sorted.begin(), options_sorted.end());

    std::ostringstream options_sig_stream;
    for (const auto& option : options_sorted) {
        options_sig_stream << option << '\n';
    }

    const std::string key_material =
        m_storage->programHash + "|" + name_expression + "|" + options_sig_stream.str() + "|backend:hiprtc";
    const std::string key = JitHelper::hashString(key_material);

    std::lock_guard<std::mutex> storage_lock(m_storage->mutex);
    if (auto it = m_storage->kernelCache.find(key); it != m_storage->kernelCache.end()) {
        return it->second;
    }

    const auto cache_base = m_storage->cacheDir / (sanitizeFilename(m_name) + "_" + key);
    const auto code_file = cache_base.string() + ".codeobj";
    const auto lowered_file = cache_base.string() + ".lowered";

    auto loaded = std::make_shared<LoadedKernelData>();

    auto try_load_module = [&](const std::string& lowered_name, std::string code_object) -> bool {
        if (lowered_name.empty() || code_object.empty()) {
            return false;
        }
        hipModule_t module = nullptr;
        if (hipModuleLoadData(&module, code_object.data()) != hipSuccess) {
            return false;
        }
        hipFunction_t function = nullptr;
        if (hipModuleGetFunction(&function, module, lowered_name.c_str()) != hipSuccess) {
            unloadHipModuleNoThrow(module);
            return false;
        }
        loaded->module = module;
        loaded->function = function;
        loaded->loweredName = lowered_name;
        loaded->codeObject = std::move(code_object);
        return true;
    };

    bool loaded_from_disk = false;
    if (std::filesystem::exists(code_file) && std::filesystem::exists(lowered_file)) {
        std::ifstream lowered_in(lowered_file);
        std::ifstream code_in(code_file, std::ios::binary);
        if (lowered_in && code_in) {
            std::string lowered_name;
            std::getline(lowered_in, lowered_name);
            std::string code_object((std::istreambuf_iterator<char>(code_in)), std::istreambuf_iterator<char>());
            loaded_from_disk = try_load_module(lowered_name, std::move(code_object));
        }
    }

    if (!loaded_from_disk) {
        hiprtcProgram program = nullptr;
        checkHipRtc(hiprtcCreateProgram(&program, m_storage->code.c_str(), "kernel.cu", 0, nullptr, nullptr),
                    "hiprtcCreateProgram failed");

        auto destroy_program = [&]() {
            if (program != nullptr) {
                hiprtcDestroyProgram(&program);
                program = nullptr;
            }
        };

        checkHipRtc(hiprtcAddNameExpression(program, name_expression.c_str()),
                    "hiprtcAddNameExpression failed for " + name_expression);

        const auto compile_flags = mergeHipCompileOptions(m_storage->flags, m_options, m_storage->archTag);
        std::vector<const char*> compile_flag_ptrs;
        compile_flag_ptrs.reserve(compile_flags.size());
        for (const auto& flag : compile_flags) {
            compile_flag_ptrs.push_back(flag.c_str());
        }

        const hiprtcResult compile_status =
            hiprtcCompileProgram(program, static_cast<int>(compile_flag_ptrs.size()), compile_flag_ptrs.data());

        size_t log_size = 0;
        hiprtcGetProgramLogSize(program, &log_size);
        std::string compile_log;
        if (log_size > 0) {
            compile_log.resize(log_size);
            hiprtcGetProgramLog(program, compile_log.data());
        }

        checkHipRtc(compile_status, "hiprtcCompileProgram failed for " + name_expression, compile_log);

        const char* lowered_name = nullptr;
        checkHipRtc(hiprtcGetLoweredName(program, name_expression.c_str(), &lowered_name),
                    "hiprtcGetLoweredName failed for " + name_expression);
        const std::string lowered_name_copy = lowered_name ? std::string(lowered_name) : name_expression;

        size_t code_size = 0;
        checkHipRtc(hiprtcGetCodeSize(program, &code_size), "hiprtcGetCodeSize failed");

        std::string code_object(code_size, '\0');
        checkHipRtc(hiprtcGetCode(program, code_object.data()), "hiprtcGetCode failed");

        destroy_program();

        checkHip(hipModuleLoadData(&loaded->module, code_object.data()), "hipModuleLoadData failed");
        checkHip(hipModuleGetFunction(&loaded->function, loaded->module, lowered_name_copy.c_str()),
                 "hipModuleGetFunction failed for lowered symbol '" + lowered_name_copy +
                     "' (name expression '" + name_expression + "')");

        loaded->loweredName = lowered_name_copy;
        loaded->codeObject = std::move(code_object);

        std::error_code ec;
        std::filesystem::create_directories(m_storage->cacheDir, ec);
        std::ofstream lowered_out(lowered_file, std::ios::trunc);
        if (lowered_out) {
            lowered_out << loaded->loweredName;
        }
        std::ofstream code_out(code_file, std::ios::binary | std::ios::trunc);
        if (code_out) {
            code_out.write(loaded->codeObject.data(), static_cast<std::streamsize>(loaded->codeObject.size()));
        }
    }

    m_storage->kernelCache[key] = loaded;
    return loaded;
}

JitHelper::CachedProgram::Kernel::KernelInstantiation JitHelper::CachedProgram::Kernel::instantiate(
    std::vector<std::string> template_args) const {
    return KernelInstantiation(m_storage, m_name, std::move(template_args), m_options);
}

JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelInstantiation(
    std::shared_ptr<ProgramStorage> storage,
    std::string kernelName,
    std::vector<std::string> templateArgs,
    std::vector<std::string> options)
    : m_storage(std::move(storage)),
      m_kernelName(std::move(kernelName)),
      m_templateArgs(std::move(templateArgs)),
      m_options(std::move(options)) {}

std::shared_ptr<JitHelper::CachedProgram::LoadedKernelData>
JitHelper::CachedProgram::Kernel::KernelInstantiation::getLoadedKernel() const {
    if (!m_loadedKernel) {
        const Kernel kernel(m_storage, m_kernelName, m_options);
        m_loadedKernel = kernel.getLoadedKernel(m_templateArgs);
    }
    return m_loadedKernel;
}

JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher::KernelLauncher(
    std::shared_ptr<ProgramStorage> storage,
    std::string kernelName,
    std::vector<std::string> templateArgs,
    std::vector<std::string> options,
    dim3 grid,
    dim3 block,
    unsigned int smem,
    hipStream_t stream)
    : m_storage(std::move(storage)),
      m_kernelName(std::move(kernelName)),
      m_templateArgs(std::move(templateArgs)),
      m_options(std::move(options)),
      m_grid(grid),
      m_block(block),
      m_smem(smem),
      m_stream(stream) {}

JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher
JitHelper::CachedProgram::Kernel::KernelInstantiation::configure_1d_max_occupancy(int max_block_size,
                                                                                  unsigned int smem,
                                                                                  CUoccupancyB2DSize smem_callback,
                                                                                  CUstream stream,
                                                                                  unsigned int flags) const {
    const auto loaded = getLoadedKernel();

    int min_grid_size = 1;
    int block_size = 1;
    size_t dyn_smem = smem;
    if (smem_callback != nullptr) {
        dyn_smem = smem_callback(max_block_size > 0 ? max_block_size : 1);
    }

    checkHip(hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
                 &min_grid_size, &block_size, loaded->function, dyn_smem, max_block_size, flags),
             "hipModuleOccupancyMaxPotentialBlockSizeWithFlags failed");

    return configure(dim3(static_cast<unsigned int>(min_grid_size), 1, 1),
                     dim3(static_cast<unsigned int>(block_size), 1, 1),
                     smem,
                     stream);
}

CUdeviceptr JitHelper::CachedProgram::Kernel::KernelInstantiation::get_global_ptr(const char* name, size_t* size) const {
    const auto loaded = getLoadedKernel();
    hipDeviceptr_t ptr = 0;
    size_t bytes = 0;
    checkHip(hipModuleGetGlobal(&ptr, &bytes, loaded->module, name), "hipModuleGetGlobal failed for " + std::string(name));
    if (size != nullptr) {
        *size = bytes;
    }
    return ptr;
}

const std::string& JitHelper::CachedProgram::Kernel::KernelInstantiation::mangled_name() const {
    return getLoadedKernel()->loweredName;
}

const std::string& JitHelper::CachedProgram::Kernel::KernelInstantiation::ptx() const {
    return getLoadedKernel()->codeObject;
}

const std::vector<std::string>& JitHelper::CachedProgram::Kernel::KernelInstantiation::link_files() const {
    return getLoadedKernel()->linkFiles;
}

const std::vector<std::string>& JitHelper::CachedProgram::Kernel::KernelInstantiation::link_paths() const {
    return getLoadedKernel()->linkPaths;
}

CUresult JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher::launch(
    std::vector<void*> arg_ptrs,
    std::vector<std::string>) const {
    const KernelInstantiation inst(m_storage, m_kernelName, m_templateArgs, m_options);
    const auto loaded = inst.getLoadedKernel();

    return hipModuleLaunchKernel(loaded->function,
                                 m_grid.x,
                                 m_grid.y,
                                 m_grid.z,
                                 m_block.x,
                                 m_block.y,
                                 m_block.z,
                                 m_smem,
                                 m_stream,
                                 arg_ptrs.empty() ? nullptr : arg_ptrs.data(),
                                 nullptr);
}

void JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher::safe_launch(
    std::vector<void*> arg_ptrs,
    std::vector<std::string> arg_types) const {
    const auto status = launch(std::move(arg_ptrs), std::move(arg_types));
    if (status != hipSuccess) {
        throwHipError("hipModuleLaunchKernel failed", status);
    }
}

#else

JitHelper::CachedProgram::CachedProgram(std::shared_ptr<ProgramStorage> storage) : m_storage(std::move(storage)) {}

JitHelper::CachedProgram::Kernel JitHelper::CachedProgram::kernel(const std::string& name,
                                                                  std::vector<std::string> options) const {
    return Kernel(m_storage, name, std::move(options));
}

JitHelper::CachedProgram::ProgramStorage::ProgramStorage(std::string code_in, std::vector<std::string> flags_in)
    : code(std::move(code_in)), flags(std::move(flags_in)) {}

JitHelper::CachedProgram::Kernel::Kernel(std::shared_ptr<ProgramStorage> storage,
                                         std::string name,
                                         std::vector<std::string> options)
    : m_storage(std::move(storage)), m_name(std::move(name)), m_options(std::move(options)) {}

std::shared_ptr<jitify::experimental::KernelInstantiation> JitHelper::CachedProgram::Kernel::getKernelInstantiation(
    const std::vector<std::string>& template_args) const {
    const std::string template_suffix =
        template_args.empty() ? std::string() : jitify::reflection::reflect_template(template_args);

    std::vector<std::string> options_sorted = m_options;
    std::sort(options_sorted.begin(), options_sorted.end());
    const std::string options_sig = jitify::reflection::reflect_list(options_sorted);

    const std::string key_material = m_storage->programHash + "|" + m_name + "|" + template_suffix + "|" + options_sig +
                                     "|cuda:" + std::to_string(getCudaVersion()) +
                                     "|api:" + std::to_string(DEME_API_VERSION) + "|" + m_storage->archTag;
    const std::string key = JitHelper::hashString(key_material);
    const std::filesystem::path cache_file = m_storage->cacheDir / (sanitizeFilename(m_name) + "_" + key + ".jit");
    std::lock_guard<std::mutex> storage_lock(m_storage->mutex);
    if (auto it = m_storage->kernelCache.find(key); it != m_storage->kernelCache.end()) {
        return it->second;
    }

    std::shared_ptr<jitify::experimental::KernelInstantiation> inst;
    if (std::filesystem::exists(cache_file)) {
        std::ifstream input(cache_file, std::ios::binary);
        if (input) {
            std::stringstream buffer;
            buffer << input.rdbuf();
            try {
                inst = std::make_shared<jitify::experimental::KernelInstantiation>(
                    jitify::experimental::KernelInstantiation::deserialize(buffer.str()));
            } catch (const std::exception&) {
                inst.reset();
            }
        }
    }
    if (!inst) {
        std::cout << "jit-compiling for " << m_name << " ..." << std::endl;
        if (!m_storage->program) {
            m_storage->program = std::make_unique<jitify::experimental::Program>(
                m_storage->code, std::vector<std::string>(), m_storage->flags);
        }
        auto kernel = m_storage->program->kernel(m_name, m_options);
        inst = std::make_shared<jitify::experimental::KernelInstantiation>(kernel, template_args);
        std::error_code ec;
        std::filesystem::create_directories(cache_file.parent_path(), ec);
        std::ofstream output(cache_file, std::ios::binary | std::ios::trunc);
        if (output) {
            output << inst->serialize();
        }
    }
    m_storage->kernelCache[key] = inst;
    return inst;
}

JitHelper::CachedProgram::Kernel::KernelInstantiation JitHelper::CachedProgram::Kernel::instantiate(
    std::vector<std::string> template_args) const {
    return KernelInstantiation(getKernelInstantiation(template_args));
}

JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelInstantiation(
    std::shared_ptr<jitify::experimental::KernelInstantiation> impl)
    : m_impl(std::move(impl)) {}

JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher
JitHelper::CachedProgram::Kernel::KernelInstantiation::configure_1d_max_occupancy(int max_block_size,
                                                                                  unsigned int smem,
                                                                                  CUoccupancyB2DSize smem_callback,
                                                                                  CUstream stream,
                                                                                  unsigned int flags) const {
    return KernelLauncher(m_impl,
                          m_impl->configure_1d_max_occupancy(max_block_size, smem, smem_callback, stream, flags));
}

CUdeviceptr JitHelper::CachedProgram::Kernel::KernelInstantiation::get_global_ptr(const char* name, size_t* size) const {
    return m_impl->get_global_ptr(name, size);
}

const std::string& JitHelper::CachedProgram::Kernel::KernelInstantiation::mangled_name() const {
    return m_impl->mangled_name();
}

const std::string& JitHelper::CachedProgram::Kernel::KernelInstantiation::ptx() const {
    return m_impl->ptx();
}

const std::vector<std::string>& JitHelper::CachedProgram::Kernel::KernelInstantiation::link_files() const {
    return m_impl->link_files();
}

const std::vector<std::string>& JitHelper::CachedProgram::Kernel::KernelInstantiation::link_paths() const {
    return m_impl->link_paths();
}

CUresult JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher::launch(
    std::vector<void*> arg_ptrs,
    std::vector<std::string> arg_types) const {
    return m_launcher.launch(std::move(arg_ptrs), std::move(arg_types));
}

void JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher::safe_launch(
    std::vector<void*> arg_ptrs,
    std::vector<std::string> arg_types) const {
    m_launcher.safe_launch(std::move(arg_ptrs), std::move(arg_types));
}

#endif