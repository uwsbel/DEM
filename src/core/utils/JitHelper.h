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
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime_api.h>

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

    //// I'm pretty sure C++17 auto-converts this
    // static CachedProgram buildProgram(
    // 	const std::string& name, const std::string& code,
    // 	std::vector<Header> headers = 0,
    // 	std::vector<std::string> flags = 0
    // );

    static const std::filesystem::path KERNEL_DIR;
    static const std::filesystem::path KERNEL_INCLUDE_DIR;
    static void setCacheDir(const std::filesystem::path& dir);
    static const std::filesystem::path& getCacheDir();
    static void setCacheTag(const std::string& tag);
    static const std::string& getCacheTag();
    static void setRequireCache(bool require);
    static bool getRequireCache();

  private:
    static std::string hashString(const std::string& in);
    static std::string toHex(uint64_t value);
    static std::filesystem::path resolveCacheDir();

    static std::filesystem::path s_cache_dir;
    static std::string s_cache_tag;
    static bool s_require_cache;

    inline static std::string loadSourceFile(const std::filesystem::path& sourcefile) {
        std::string code;
        // If the file exists, read in the entire thing.
        if (std::filesystem::exists(sourcefile)) {
            std::ifstream input(sourcefile);
            std::getline(input, code, std::string::traits_type::to_char_type(std::string::traits_type::eof()));
        }
        return code;
    };
};

class JitHelper::CachedProgram {
  public:
    class Kernel;

    Kernel kernel(const std::string& name, std::vector<std::string> options = {}) const;

    const std::string& key() const { return m_storage->programHash; }

  private:
    struct ProgramStorage {
        ProgramStorage(std::string code_in, std::vector<std::string> flags_in);
        ~ProgramStorage();
        std::string code;
        std::vector<std::string> flags;
        std::string programHash;
        std::filesystem::path cacheDir;
        int device = 0;
        std::string archTag;
        CUmodule module = nullptr;
        std::unordered_map<std::string, CUfunction> bakedKernelCache;
        std::mutex mutex;
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

    KernelInstantiation getKernelInstantiation(const std::vector<std::string>& template_args) const;

    friend class CachedProgram;
};

class JitHelper::CachedProgram::Kernel::KernelInstantiation {
  public:
    class KernelLauncher;

    KernelLauncher configure(dim3 grid, dim3 block, unsigned int smem = 0, cudaStream_t stream = 0) const;
    KernelLauncher configure_1d_max_occupancy(int max_block_size = 0,
                                              unsigned int smem = 0,
                                              CUoccupancyB2DSize smem_callback = 0,
                                              cudaStream_t stream = 0,
                                              unsigned int flags = 0) const;

    CUdeviceptr get_global_ptr(const char* name, size_t* size = nullptr) const {
        size_t bytes = 0;
        CUdeviceptr ptr = 0;
        CUresult res = cuModuleGetGlobal(&ptr, &bytes, m_storage->module, name);
        if (size) {
            *size = bytes;
        }
        if (res != CUDA_SUCCESS) {
            return 0;
        }
        return ptr;
    }

    template <typename T>
    CUresult get_global_array(const char* name, T* data, size_t count, CUstream stream = 0) const {
        size_t bytes = 0;
        CUdeviceptr ptr = 0;
        CUresult res = cuModuleGetGlobal(&ptr, &bytes, m_storage->module, name);
        if (res != CUDA_SUCCESS) {
            return res;
        }
        const size_t copy_bytes = sizeof(T) * count;
        if (copy_bytes > bytes) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        return cuMemcpyDtoHAsync(data, ptr, copy_bytes, stream);
    }

    template <typename T>
    CUresult get_global_value(const char* name, T* value, CUstream stream = 0) const {
        return get_global_array(name, value, 1, stream);
    }

    template <typename T>
    CUresult set_global_array(const char* name, const T* data, size_t count, CUstream stream = 0) const {
        size_t bytes = 0;
        CUdeviceptr ptr = 0;
        CUresult res = cuModuleGetGlobal(&ptr, &bytes, m_storage->module, name);
        if (res != CUDA_SUCCESS) {
            return res;
        }
        const size_t copy_bytes = sizeof(T) * count;
        if (copy_bytes > bytes) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        return cuMemcpyHtoDAsync(ptr, data, copy_bytes, stream);
    }

    template <typename T>
    CUresult set_global_value(const char* name, const T& value, CUstream stream = 0) const {
        return set_global_array(name, &value, 1, stream);
    }

    const std::string& mangled_name() const {
        static const std::string empty;
        return empty;
    }
    const std::string& ptx() const {
        static const std::string empty;
        return empty;
    }

    const std::vector<std::string>& link_files() const {
        static const std::vector<std::string> empty;
        return empty;
    }
    const std::vector<std::string>& link_paths() const {
        static const std::vector<std::string> empty;
        return empty;
    }

  private:
    std::shared_ptr<ProgramStorage> m_storage;
    CUfunction m_func = nullptr;

    KernelInstantiation(std::shared_ptr<ProgramStorage> storage, CUfunction func);

    friend class Kernel;
};

class JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher {
  public:
    KernelLauncher(std::shared_ptr<ProgramStorage> storage,
                   CUfunction func,
                   dim3 grid,
                   dim3 block,
                   unsigned int smem,
                   cudaStream_t stream)
        : m_storage(std::move(storage)),
          m_func(func),
          m_grid(grid),
          m_block(block),
          m_smem(smem),
          m_stream(stream) {}

    CUresult launch(std::vector<void*> arg_ptrs = {}, std::vector<std::string> arg_types = {}) const {
        (void)arg_types;
        return cuLaunchKernel(m_func, m_grid.x, m_grid.y, m_grid.z, m_block.x, m_block.y, m_block.z, m_smem, m_stream,
                              arg_ptrs.data(), nullptr);
    }

    template <typename... ArgTypes>
    CUresult launch(const ArgTypes&... args) const {
        return launch(std::vector<void*>({(void*)&args...}), {});
    }

    void safe_launch(std::vector<void*> arg_ptrs = {}, std::vector<std::string> arg_types = {}) const {
        CUresult res = launch(std::move(arg_ptrs), std::move(arg_types));
        if (res != CUDA_SUCCESS) {
            const char* err = nullptr;
            cuGetErrorString(res, &err);
            throw std::runtime_error(err ? err : "Failed to launch baked kernel");
        }
    }

    template <typename... ArgTypes>
    void safe_launch(const ArgTypes&... args) const {
        safe_launch(std::vector<void*>({(void*)&args...}), {});
    }

  private:
    std::shared_ptr<ProgramStorage> m_storage;
    CUfunction m_func = nullptr;
    dim3 m_grid = dim3(0);
    dim3 m_block = dim3(0);
    unsigned int m_smem = 0;
    cudaStream_t m_stream = 0;
};

template <typename... TemplateArgs>
inline JitHelper::CachedProgram::Kernel::KernelInstantiation JitHelper::CachedProgram::Kernel::instantiate(
    TemplateArgs... targs) const {
    (void)sizeof...(targs);
    return instantiate(std::vector<std::string>());
}

inline JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher
JitHelper::CachedProgram::Kernel::KernelInstantiation::configure(dim3 grid,
                                                                 dim3 block,
                                                                 unsigned int smem,
                                                                 cudaStream_t stream) const {
    return KernelLauncher(m_storage, m_func, grid, block, smem, stream);
}

inline JitHelper::CachedProgram::Kernel::KernelInstantiation::KernelLauncher
JitHelper::CachedProgram::Kernel::KernelInstantiation::configure_1d_max_occupancy(int max_block_size,
                                                                                  unsigned int smem,
                                                                                  CUoccupancyB2DSize smem_callback,
                                                                                  cudaStream_t stream,
                                                                                  unsigned int flags) const {
    int min_grid = 1;
    int block = max_block_size;
    if (block <= 0) {
        (void)smem_callback;
        (void)flags;
        CUresult res = cuOccupancyMaxPotentialBlockSize(&min_grid, &block, m_func, nullptr, smem, 0);
        if (res != CUDA_SUCCESS) {
            block = 128;
            min_grid = 1;
        }
    }
    return KernelLauncher(m_storage, m_func, dim3(min_grid), dim3(block), smem, stream);
}

#endif
