//	Copyright (c) 2021, SBEL GPU Development Team
//	Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_JIT_HELPER_H
#define DEME_JIT_HELPER_H

#include <filesystem>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <utility>
#include <cctype>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <stdexcept>

#ifdef DEME_USE_HIP
    #include "GpuRuntime.hpp"
    #if defined(__clang__)
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wattributes"
    #endif
    #include <hip/hiprtc.h>
    #if defined(__clang__)
        #pragma clang diagnostic pop
    #endif

    using CUresult = hipError_t;
    using CUdeviceptr = hipDeviceptr_t;
    using CUstream = hipStream_t;
    using CUoccupancyB2DSize = size_t (*)(int);
#else
    #include <jitify/jitify.hpp>
    #include <cuda.h>
    #include <cuda_runtime_api.h>
#endif

#if defined(_WIN32) || defined(_WIN64)
    #undef max
    #undef min
    #undef strtok_r
#endif

class JitHelper {
  public:
    class CachedProgram;

    class Header {
      public:
        Header(const std::filesystem::path& sourcefile);
        const std::string& getSource();
        void substitute(const std::string& symbol, const std::string& value);

      private:
        std::string _source;
    };

    static CachedProgram buildProgram(
        const std::string& name,
        const std::filesystem::path& source,
        std::unordered_map<std::string, std::string> substitutions = std::unordered_map<std::string, std::string>(),
        std::vector<std::string> flags = std::vector<std::string>());

    static void setTryDisableRuntimeCompiler(bool disable) { s_tryDisableRuntimeCompiler = disable; }
    static bool getTryDisableRuntimeCompiler() { return s_tryDisableRuntimeCompiler; }

    static const std::filesystem::path KERNEL_DIR;
    static const std::filesystem::path KERNEL_INCLUDE_DIR;
    static const std::filesystem::path CACHE_DIR;

  private:
    inline static bool s_tryDisableRuntimeCompiler = false;

    static std::string hashString(const std::string& in);
    static std::string toHex(uint64_t value);
    static std::filesystem::path resolveCacheDir();

    template <typename T>
    inline static std::string stringifyTemplateArg(const T& value) {
        using Decayed = std::decay_t<T>;
        if constexpr (std::is_same_v<Decayed, std::string>) {
            return value;
        } else if constexpr (std::is_same_v<Decayed, const char*> || std::is_same_v<Decayed, char*>) {
            return value ? std::string(value) : std::string();
        } else {
            std::ostringstream out;
            out << value;
            return out.str();
        }
    }

    inline static std::string loadSourceFile(const std::filesystem::path& sourcefile) {
        if (!std::filesystem::exists(sourcefile)) {
            return {};
        }

        std::ifstream input(sourcefile, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    };
};

class JitHelper::CachedProgram {
  public:
    class Kernel;

    Kernel kernel(const std::string& name, std::vector<std::string> options = {}) const;

    const std::string& key() const { return m_storage->programHash; }

  private:
#ifdef DEME_USE_HIP
    struct LoadedKernelData {
        ~LoadedKernelData();

        hipModule_t module = nullptr;
        hipFunction_t function = nullptr;
        std::string loweredName;
        std::string codeObject;
        std::vector<std::string> linkFiles;
        std::vector<std::string> linkPaths;
    };
#endif

    struct ProgramStorage {
        ProgramStorage(std::string code_in, std::vector<std::string> flags_in);

#ifdef DEME_USE_HIP
        std::string code;
        std::vector<std::string> flags;
        std::string programHash;
        std::filesystem::path cacheDir;
        int device = 0;
        std::string archTag;
        std::unordered_map<std::string, std::shared_ptr<LoadedKernelData>> kernelCache;
        std::mutex mutex;
#else
        std::unique_ptr<jitify::experimental::Program> program;
        std::string code;
        std::vector<std::string> flags;
        std::string programHash;
        std::filesystem::path cacheDir;
        int device = 0;
        std::string archTag;
        std::unordered_map<std::string, std::shared_ptr<jitify::experimental::KernelInstantiation>> kernelCache;
        std::mutex mutex;
#endif
    };

    std::shared_ptr<ProgramStorage> m_storage;
    explicit CachedProgram(std::shared_ptr<ProgramStorage> storage);

    friend class JitHelper;
};

class JitHelper::CachedProgram::Kernel {
  public:
    class KernelInstantiation;

    KernelInstantiation instantiate(std::vector<std::string> template_args = {}) const;

    template <typename... TemplateArgs>
    KernelInstantiation instantiate(TemplateArgs... targs) const;

  private:
    std::shared_ptr<ProgramStorage> m_storage;
    std::string m_name;
    std::vector<std::string> m_options;

    Kernel(std::shared_ptr<ProgramStorage> storage, std::string name, std::vector<std::string> options);

#ifdef DEME_USE_HIP
    std::shared_ptr<LoadedKernelData> getLoadedKernel(const std::vector<std::string>& template_args) const;
#else
    std::shared_ptr<jitify::experimental::KernelInstantiation> getKernelInstantiation(
        const std::vector<std::string>& template_args) const;
#endif

    friend class CachedProgram;
};

class JitHelper::CachedProgram::Kernel::KernelInstantiation {
  public:
    class KernelLauncher;

#ifdef DEME_USE_HIP
    KernelLauncher configure(dim3 grid, dim3 block, unsigned int smem = 0, hipStream_t stream = nullptr) const;
#else
    KernelLauncher configure(dim3 grid, dim3 block, unsigned int smem = 0, cudaStream_t stream = nullptr) const;
#endif
    KernelLauncher configure_1d_max_occupancy(int max_block_size = 0,
                                              unsigned int smem = 0,
                                              CUoccupancyB2DSize smem_callback = 0,
                                              CUstream stream = 0,
                                              unsigned int flags = 0) const;

    CUdeviceptr get_global_ptr(const char* name, size_t* size = nullptr) const;

    template <typename T>
    CUresult get_global_array(const char* name, T* data, size_t count, CUstream stream = 0) const;

    template <typename T>
    CUresult get_global_value(const char* name, T* value, CUstream stream = 0) const {
        return get_global_array(name, value, 1, stream);
    }

    template <typename T>
    CUresult set_global_array(const char* name, const T* data, size_t count, CUstream stream = 0) const;

    template <typename T>
    CUresult set_global_value(const char* name, const T& value, CUstream stream = 0) const {
        return set_global_array(name, &value, 1, stream);
    }

    const std::string& mangled_name() const;
    const std::string& ptx() const;
    const std::vector<std::string>& link_files() const;
    const std::vector<std::string>& link_paths() const;

  private:
#ifdef DEME_USE_HIP
    std::shared_ptr<ProgramStorage> m_storage;
    std::string m_kernelName;
    std::vector<std::string> m_templateArgs;
    std::vector<std::string> m_options;
    mutable std::shared_ptr<LoadedKernelData> m_loadedKernel;

    KernelInstantiation(std::shared_ptr<ProgramStorage> storage,
                        std::string kernelName,
                        std::vector<std::string> templateArgs,
                        std::vector<std::string> options);
    std::shared_ptr<LoadedKernelData> getLoadedKernel() const;
#else
    std::shared_ptr<jitify::experimental::KernelInstantiation> m_impl;

    KernelInstantiation(std::shared_ptr<jitify::experimental::KernelInstantiation> impl);
#endif

    friend class Kernel;
};

class JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher {
  public:
#ifdef DEME_USE_HIP
    KernelLauncher(std::shared_ptr<ProgramStorage> storage,
                   std::string kernelName,
                   std::vector<std::string> templateArgs,
                   std::vector<std::string> options,
                   dim3 grid,
                   dim3 block,
                   unsigned int smem,
                   hipStream_t stream);
#else
    KernelLauncher(std::shared_ptr<jitify::experimental::KernelInstantiation> inst,
                   jitify::experimental::KernelLauncher launcher)
        : m_inst(std::move(inst)), m_launcher(std::move(launcher)) {}
#endif

    CUresult launch(std::vector<void*> arg_ptrs = {}, std::vector<std::string> arg_types = {}) const;

    template <typename... ArgTypes>
    CUresult launch(const ArgTypes&... args) const;

    void safe_launch(std::vector<void*> arg_ptrs = {}, std::vector<std::string> arg_types = {}) const;

    template <typename... ArgTypes>
    void safe_launch(const ArgTypes&... args) const;

  private:
#ifdef DEME_USE_HIP
    std::shared_ptr<ProgramStorage> m_storage;
    std::string m_kernelName;
    std::vector<std::string> m_templateArgs;
    std::vector<std::string> m_options;
    dim3 m_grid;
    dim3 m_block;
    unsigned int m_smem;
    hipStream_t m_stream;
#else
    std::shared_ptr<jitify::experimental::KernelInstantiation> m_inst;
    jitify::experimental::KernelLauncher m_launcher;
#endif
};

#ifdef DEME_USE_HIP
template <typename... TemplateArgs>
inline JitHelper::CachedProgram::Kernel::KernelInstantiation JitHelper::CachedProgram::Kernel::instantiate(
    TemplateArgs... targs) const {
    return instantiate(std::vector<std::string>({JitHelper::stringifyTemplateArg(targs)...}));
}

inline JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher
JitHelper::CachedProgram::Kernel::KernelInstantiation::configure(dim3 grid,
                                                                 dim3 block,
                                                                 unsigned int smem,
                                                                 hipStream_t stream) const {
    return KernelLauncher(m_storage, m_kernelName, m_templateArgs, m_options, grid, block, smem, stream);
}
#else
template <typename... TemplateArgs>
inline JitHelper::CachedProgram::Kernel::KernelInstantiation JitHelper::CachedProgram::Kernel::instantiate(
    TemplateArgs... targs) const {
    return instantiate(std::vector<std::string>({jitify::reflection::reflect(targs)...}));
}

inline JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher
JitHelper::CachedProgram::Kernel::KernelInstantiation::configure(dim3 grid,
                                                                 dim3 block,
                                                                 unsigned int smem,
                                                                 cudaStream_t stream) const {
    return KernelLauncher(m_impl, m_impl->configure(grid, block, smem, stream));
}
#endif

template <typename T>
inline CUresult JitHelper::CachedProgram::Kernel::KernelInstantiation::get_global_array(const char* name,
                                                                                         T* data,
                                                                                         size_t count,
                                                                                         CUstream stream) const {
#ifdef DEME_USE_HIP
    size_t bytes = 0;
    const auto ptr = get_global_ptr(name, &bytes);
    if (ptr == 0) {
        return hipErrorNotFound;
    }
    const size_t requested = sizeof(T) * count;
    if (requested > bytes) {
        return hipErrorInvalidValue;
    }
    return stream ? hipMemcpyDtoHAsync(data, ptr, requested, stream) : hipMemcpyDtoH(data, ptr, requested);
#else
    return m_impl->get_global_array(name, data, count, stream);
#endif
}

template <typename T>
inline CUresult JitHelper::CachedProgram::Kernel::KernelInstantiation::set_global_array(const char* name,
                                                                                         const T* data,
                                                                                         size_t count,
                                                                                         CUstream stream) const {
#ifdef DEME_USE_HIP
    size_t bytes = 0;
    const auto ptr = get_global_ptr(name, &bytes);
    if (ptr == 0) {
        return hipErrorNotFound;
    }
    const size_t requested = sizeof(T) * count;
    if (requested > bytes) {
        return hipErrorInvalidValue;
    }
    return stream ? hipMemcpyHtoDAsync(ptr, data, requested, stream)
                  : hipMemcpyHtoD(ptr, data, requested);
#else
    return m_impl->set_global_array(name, data, count, stream);
#endif
}

template <typename... ArgTypes>
inline CUresult JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher::launch(const ArgTypes&... args) const {
#ifdef DEME_USE_HIP
    return launch(std::vector<void*>({const_cast<void*>(static_cast<const void*>(&args))...}), {});
#else
    return launch(std::vector<void*>({(void*)&args...}), {jitify::reflection::reflect<ArgTypes>()...});
#endif
}

template <typename... ArgTypes>
inline void JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher::safe_launch(
    const ArgTypes&... args) const {
#ifdef DEME_USE_HIP
    safe_launch(std::vector<void*>({const_cast<void*>(static_cast<const void*>(&args))...}), {});
#else
    safe_launch(std::vector<void*>({(void*)&args...}), {jitify::reflection::reflect<ArgTypes>()...});
#endif
}

#endif