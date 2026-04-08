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

std::string applySubstitutions(std::string code,
                               const std::unordered_map<std::string, std::string>& substitutions) {
    std::vector<std::pair<std::string, std::string>> ordered_subs(substitutions.begin(), substitutions.end());
    std::sort(ordered_subs.begin(), ordered_subs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (auto& subst : ordered_subs) {
        code = std::regex_replace(code, std::regex(subst.first), subst.second);
    }
    return code;
}

void appendCudaIncludeFlags(std::vector<std::string>& flags) {
    std::vector<std::filesystem::path> include_paths;
    std::string dirs = DEME_CUDA_TOOLKIT_INCLUDE_DIRS;  // "dir1;dir2;dir3"
    std::stringstream ss(dirs);
    std::string dir;
    while (std::getline(ss, dir, ';')) {
        if (!dir.empty()) {
            include_paths.emplace_back(dir);
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
        add_inc(p / "cccl");
    }
}

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
                   const std::filesystem::path& output,
                   const std::vector<std::string>& flags,
                   const std::string& arch_tag,
                   const std::filesystem::path& log_path) {
    std::ostringstream cmd;
    cmd << "nvcc --cubin -arch=" << arch_tag << " -DDEME_DEFINE_SIM_PARAMS_CONST";
    for (const auto& flag : flags) {
        if (flag.rfind("-diag-suppress=", 0) == 0 || flag.rfind("-diag_suppress=", 0) == 0) {
            continue;
        }
        cmd << " " << flag;
    }
    cmd << " -o " << quoteArg(output) << " " << quoteArg(source);
    cmd << " > " << quoteArg(log_path) << " 2>&1";
    return std::system(cmd.str().c_str());
}

std::mutex g_runtime_compile_mutex;

}  // namespace

const std::filesystem::path JitHelper::KERNEL_DIR = DEMERuntimeDataHelper::data_path / "kernel";
const std::filesystem::path JitHelper::KERNEL_INCLUDE_DIR = DEMERuntimeDataHelper::include_path;
const std::filesystem::path JitHelper::CACHE_DIR = JitHelper::resolveCacheDir();
bool JitHelper::s_try_disable_runtime_compiler = false;

void JitHelper::setTryDisableRuntimeCompiler(bool disable) {
    s_try_disable_runtime_compiler = disable;
}

bool JitHelper::getTryDisableRuntimeCompiler() {
    return s_try_disable_runtime_compiler;
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
    const std::string flags_sig = jitify::reflection::reflect_list(flags_sorted);
    const std::string fingerprint = code + "|flags:" + flags_sig + "|api:" + std::to_string(DEME_API_VERSION) +
                                    "|cuda:" + std::to_string(getCudaVersion()) + "|arch:" + arch_tag;
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

    if (getTryDisableRuntimeCompiler()) {
        const std::filesystem::path source_out = storage->cacheDir / (sanitizeFilename(name) + ".cu");
        const std::filesystem::path compile_out = storage->cacheDir / (sanitizeFilename(name) + ".cubin");
        const std::filesystem::path log_out = storage->cacheDir / (sanitizeFilename(name) + ".log");
        bool have_cubin = std::filesystem::exists(compile_out);
        bool attempted_standard_compiler = false;

        if (!have_cubin) {
            std::lock_guard<std::mutex> lock(g_runtime_compile_mutex);
            have_cubin = std::filesystem::exists(compile_out);
            if (!have_cubin) {
                attempted_standard_compiler = true;
                std::ofstream src_out(source_out, std::ios::trunc);
                if (src_out) {
                    src_out << source_code;
                    src_out.close();
                    std::cout << "Trying standard compiler for " << name << " ..." << std::endl;
                    int ret = runNvccCompile(source_out, compile_out, flags, arch_tag, log_out);
                    have_cubin = (ret == 0 && std::filesystem::exists(compile_out));
                    if (!have_cubin) {
                        std::filesystem::remove(compile_out, ec);
                    }
                }
            }
        }

        if (have_cubin && cuInit(0) == CUDA_SUCCESS) {
            CUmodule module = nullptr;
            if (cuModuleLoad(&module, compile_out.string().c_str()) == CUDA_SUCCESS) {
                storage->module = module;
            }
        }

        if (!storage->module && attempted_standard_compiler) {
            std::cout << "Falling back to the runtime compiler for " << name << " ..." << std::endl;
        }
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
    // Prefer to keep cache alongside other runtime data in the build tree
    std::filesystem::path default_path = DEMERuntimeDataHelper::data_path / "jit_cache";
    if (std::filesystem::exists(DEMERuntimeDataHelper::data_path)) {
        return default_path;
    }
    return std::filesystem::temp_directory_path() / "dem-jit";
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
    std::lock_guard<std::mutex> storage_lock(m_storage->mutex);

    if (m_storage->module) {
        if (!template_args.empty()) {
            throw std::runtime_error("Template arguments are not supported when kernels are compiled with nvcc.");
        }
        if (auto it = m_storage->kernelFunctionCache.find(m_name); it != m_storage->kernelFunctionCache.end()) {
            return KernelInstantiation(m_storage, it->second);
        }
        CUfunction func = nullptr;
        CUresult res = cuModuleGetFunction(&func, m_storage->module, m_name.c_str());
        if (res != CUDA_SUCCESS || func == nullptr) {
            throw std::runtime_error("Failed to load kernel function " + m_name);
        }
        m_storage->kernelFunctionCache.emplace(m_name, func);
        return KernelInstantiation(m_storage, func);
    }

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
    if (auto it = m_storage->kernelCache.find(key); it != m_storage->kernelCache.end()) {
        return KernelInstantiation(it->second);
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
    return KernelInstantiation(inst);
}

JitHelper::CachedProgram::Kernel::KernelInstantiation JitHelper::CachedProgram::Kernel::instantiate(
    std::vector<std::string> template_args) const {
    return getKernelInstantiation(template_args);
}

JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelInstantiation(
    std::shared_ptr<jitify::experimental::KernelInstantiation> impl)
    : m_impl(std::move(impl)) {}

JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelInstantiation(
    std::shared_ptr<ProgramStorage> storage,
    CUfunction func)
    : m_storage(std::move(storage)), m_func(func) {}
