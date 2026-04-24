//  Copyright (c) 2021, SBEL GPU Development Team
//  Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_DATA_MIGRATION_HPP
#define DEME_DATA_MIGRATION_HPP

#include <cassert>
#include <cctype>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <typeinfo>
#include <utility>
#include <unordered_map>
#include <vector>

#include "Logger.hpp"
#include "BaseClasses.hpp"
#include "CudaAllocator.hpp"
#include "../../DEM/VariableTypes.h"

namespace deme {


enum class MemoryRole {
    Unknown,
    PersistentState,
    StaticReadOnly,
    TransferSnapshot,
    ContactWorkspace,
    ScratchTemporary,
    OutputStaging,
    BorrowedView
};

enum class PreserveOnResize {
    Preserve,
    Discard
};


inline const char* MemoryRoleName(MemoryRole role) {
    switch (role) {
        case MemoryRole::PersistentState: return "PersistentState";
        case MemoryRole::StaticReadOnly: return "StaticReadOnly";
        case MemoryRole::TransferSnapshot: return "TransferSnapshot";
        case MemoryRole::ContactWorkspace: return "ContactWorkspace";
        case MemoryRole::ScratchTemporary: return "ScratchTemporary";
        case MemoryRole::OutputStaging: return "OutputStaging";
        case MemoryRole::BorrowedView: return "BorrowedView";
        case MemoryRole::Unknown:
        default: return "Unknown";
    }
}

namespace detail {


inline bool env_truthy(const char* env) {
    return env && *env && !(env[0] == '0' && env[1] == '\0') && env[0] != 'f' && env[0] != 'F' && env[0] != 'n' && env[0] != 'N';
}

inline bool memory_policy_compact() {
    const char* env = std::getenv("DEME_MEMORY_POLICY");
    if (!env || !*env)
        return false;
    std::string v(env);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return v == "compact" || v == "compact_single_gpu" || v == "compactsinglegpu";
}

inline size_t scratch_cache_limit_bytes() {
    // Freeing pooled scratch DeviceArrays is not a safe lifetime operation yet: many call sites mark temp vectors
    // free before all same-stream/consumer-stream uses are globally complete. Retaining those allocations preserves
    // the legacy behavior. Real scratch decommit must be implemented with scoped arenas/events or VMM pages.
    const char* enable_trim = std::getenv("DEME_EXPERIMENTAL_SCRATCH_TRIM");
    if (!env_truthy(enable_trim))
        return static_cast<size_t>(-1);

    const char* env = std::getenv("DEME_SCRATCH_CACHE_LIMIT_MB");
    if (env && *env) {
        char* end = nullptr;
        double mb = std::strtod(env, &end);
        if (end != env && mb >= 0.0) {
            return static_cast<size_t>(mb * 1024.0 * 1024.0);
        }
    }
    // Opt-in experimental trimming only. Throughput and compact modes both keep legacy unlimited caching unless
    // DEME_EXPERIMENTAL_SCRATCH_TRIM=1 is explicitly set.
    return 16ull * 1024ull * 1024ull;
}

inline bool immutable_metadata_share_enabled() {
    // Conservative same-GPU kT/dT sharing for metadata that is populated during Initialize/Update and then read-only
    // during dynamics. This is default-on because it only changes dT device pointers to views of kT-owned immutable
    // arrays when kT and dT run on the same CUDA device. It can be disabled explicitly for A/B testing.
    if (env_truthy(std::getenv("DEME_DISABLE_KTDT_IMMUTABLE_METADATA_SHARE")))
        return false;
    if (env_truthy(std::getenv("DEME_DISABLE_SAFE_STATIC_GEOMETRY_SHARE")))
        return false;

    // Backward-compatible override: DEME_SHARE_STATIC_GEOMETRY=0 disables all kT/dT sharing, while =1 enables the
    // broader experimental static-geometry list below in addition to this default immutable metadata set.
    const char* env = std::getenv("DEME_SHARE_STATIC_GEOMETRY");
    if (env && *env)
        return env_truthy(env);
    return true;
}

inline bool safe_static_share_enabled() {
    // Broader static-geometry/template sharing remains opt-in until every writer/update path is proven immutable.
    // The narrow immutable metadata set above is handled separately and is default-on for same-GPU kT/dT.
    if (env_truthy(std::getenv("DEME_DISABLE_SAFE_STATIC_GEOMETRY_SHARE")))
        return false;
    return env_truthy(std::getenv("DEME_SHARE_STATIC_GEOMETRY"));
}

inline bool mesh_static_contract_enabled() {
    // Mesh-node mutability contract. By default mesh local triangle-node positions are considered immutable after
    // initialization: rigid free-moving meshes and prescribed-motion rigid workpieces move via owner pose, not by
    // rewriting relPosNode*. Deformable meshes opt out via DEMMesh::SetDeformable(true) or by calling
    // SetTriNodeRelPos/UpdateTriNodeRelPos, which detaches dT views and allocates transfer buffers on demand.
    return !env_truthy(std::getenv("DEME_DISABLE_MESH_STATIC_CONTRACT"));
}

inline bool static_mesh_node_share_enabled() {
    // The contract-backed relPosNode sharing is default-on for same-GPU kT/dT when all loaded meshes are rigid-node
    // meshes. Keep the global sharing opt-out semantics: DEME_SHARE_STATIC_GEOMETRY=0 disables all static sharing.
    if (!mesh_static_contract_enabled())
        return false;
    if (env_truthy(std::getenv("DEME_DISABLE_STATIC_MESH_NODE_SHARE")))
        return false;
    const char* env = std::getenv("DEME_SHARE_STATIC_GEOMETRY");
    if (env && *env && !env_truthy(env))
        return false;
    return true;
}

inline bool omit_immutable_mesh_deform_buffers_enabled() {
    // Omitting relPosNode*_buffer gives another ~3*sizeof(float3)*nTri saving in purely immutable-mesh scenes, but a
    // previous default-lazy attempt exposed an illegal access in a triangle-heavy demo. Keep buffer omission explicit
    // until this path has passed broader CUDA validation. The default contract still shares dT relPosNode device
    // storage, which gives the safe same-GPU VRAM win without changing the legacy kT buffer layout.
    if (env_truthy(std::getenv("DEME_ALWAYS_ALLOC_MESH_DEFORM_BUFFERS")))
        return false;
    return mesh_static_contract_enabled() &&
           (env_truthy(std::getenv("DEME_OMIT_IMMUTABLE_MESH_DEFORM_BUFFERS")) ||
            env_truthy(std::getenv("DEME_MESH_STATIC_CONTRACT_STRICT")));
}

inline bool experimental_share_static_mesh_nodes_enabled() {
    // relPosNode1/2/3 are large, but some deformation paths treat them as mutable. Keep this off unless the
    // scene is known to use only static meshes and the user opts in explicitly.
    return env_truthy(std::getenv("DEME_EXPERIMENTAL_SHARE_STATIC_MESH_NODES"));
}

inline size_t safe_scratch_cache_limit_bytes() {
    // Scratch-pool entries marked free are not guaranteed to be true lifetime-dead objects in the current
    // architecture: some later phases still rely on cached scratch addresses or reclaim-by-pattern behavior.
    // Therefore physical scratch trimming must be explicit experimental opt-in, not automatic compact mode.
    // The previous default-on compact trim saved memory but caused illegal accesses in DEMdemo_Testsf before
    // sph-tri contacts.
    if (env_truthy(std::getenv("DEME_DISABLE_SAFE_SCRATCH_TRIM")))
        return static_cast<size_t>(-1);
    if (!env_truthy(std::getenv("DEME_EXPERIMENTAL_SAFE_SCRATCH_TRIM")) &&
        !env_truthy(std::getenv("DEME_EXPERIMENTAL_SCRATCH_TRIM")))
        return static_cast<size_t>(-1);
    const char* env = std::getenv("DEME_SCRATCH_CACHE_LIMIT_MB");
    if (env && *env) {
        char* end = nullptr;
        double mb = std::strtod(env, &end);
        if (end != env && mb >= 0.0)
            return static_cast<size_t>(mb * 1024.0 * 1024.0);
    }
    return 16ull * 1024ull * 1024ull;
}

inline bool lazy_mesh_deform_buffers_enabled() {
    // The old relPosNode*_buffer allocations appear to mask an existing downstream OOB in some triangle scenes.
    // Do not remove them by default; keep lazy allocation available only as an explicit experiment.
    return env_truthy(std::getenv("DEME_EXPERIMENTAL_LAZY_MESH_DEFORM_BUFFERS"));
}

inline bool mem_trace_enabled() {
    const char* env = std::getenv("DEME_MEM_TRACE");
    return env && *env && !(env[0] == '0' && env[1] == '\0');
}

inline std::string mem_trace_dir() {
    const char* env = std::getenv("DEME_MEM_TRACE_DIR");
    return (env && *env) ? std::string(env) : std::string(".");
}

inline std::string csv_escape(const std::string& s) {
    bool quote = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            quote = true;
            break;
        }
    }
    if (!quote)
        return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"')
            out += "\"\"";
        else
            out += c;
    }
    out += "\"";
    return out;
}

inline std::string sanitize_phase_name(const std::string& phase) {
    std::string out;
    out.reserve(phase.size());
    for (char c : phase) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-')
            out.push_back(c);
        else
            out.push_back('_');
    }
    return out.empty() ? std::string("snapshot") : out;
}

struct MemoryLedgerRecord {
    const void* array_object = nullptr;
    std::string owner;
    std::string name;
    std::string role;
    std::string type_name;
    size_t type_size = 0;
    size_t logical_bytes = 0;
    size_t capacity_bytes = 0;
    size_t committed_bytes = 0;
    size_t host_bytes = 0;
    size_t high_water_capacity_bytes = 0;
    const void* device_ptr = nullptr;
    bool is_view = false;
    bool has_host_mirror = false;
    std::string backend;
    std::string phase;
};

struct MemoryLedger {
    std::mutex mutex;
    std::unordered_map<const void*, MemoryLedgerRecord> records;
};

inline MemoryLedger& memory_ledger() {
    static MemoryLedger ledger;
    return ledger;
}

inline void upsert_memory_ledger_record(const MemoryLedgerRecord& rec) {
    if (!mem_trace_enabled())
        return;
    auto& ledger = memory_ledger();
    std::lock_guard<std::mutex> lock(ledger.mutex);
    ledger.records[rec.array_object] = rec;
}

inline void erase_memory_ledger_record(const void* array_object) {
    if (!mem_trace_enabled())
        return;
    auto& ledger = memory_ledger();
    std::lock_guard<std::mutex> lock(ledger.mutex);
    ledger.records.erase(array_object);
}

inline std::vector<MemoryLedgerRecord> memory_ledger_snapshot() {
    auto& ledger = memory_ledger();
    std::lock_guard<std::mutex> lock(ledger.mutex);
    std::vector<MemoryLedgerRecord> out;
    out.reserve(ledger.records.size());
    for (const auto& kv : ledger.records)
        out.push_back(kv.second);
    std::sort(out.begin(), out.end(), [](const MemoryLedgerRecord& a, const MemoryLedgerRecord& b) {
        if (a.committed_bytes != b.committed_bytes)
            return a.committed_bytes > b.committed_bytes;
        if (a.capacity_bytes != b.capacity_bytes)
            return a.capacity_bytes > b.capacity_bytes;
        return a.name < b.name;
    });
    return out;
}

inline void dump_memory_ledger_csv(const std::string& phase) {
    if (!mem_trace_enabled())
        return;
    auto records = memory_ledger_snapshot();
    const std::string file = mem_trace_dir() + "/memtrace_" + sanitize_phase_name(phase) + ".csv";
    std::ofstream os(file);
    if (!os)
        return;
    os << "phase,owner,name,role,type_name,type_size,logical_bytes,capacity_bytes,committed_bytes,host_bytes,"
          "high_water_capacity_bytes,device_ptr,array_object,is_view,has_host_mirror,backend,last_phase\n";
    for (const auto& r : records) {
        os << csv_escape(phase) << ',' << csv_escape(r.owner) << ',' << csv_escape(r.name) << ','
           << csv_escape(r.role) << ',' << csv_escape(r.type_name) << ',' << r.type_size << ',' << r.logical_bytes
           << ',' << r.capacity_bytes << ',' << r.committed_bytes << ',' << r.host_bytes << ','
           << r.high_water_capacity_bytes << ',' << r.device_ptr << ',' << r.array_object << ',' << (r.is_view ? 1 : 0)
           << ',' << (r.has_host_mirror ? 1 : 0) << ',' << csv_escape(r.backend) << ',' << csv_escape(r.phase) << '\n';
    }
}

inline void print_memory_ledger_summary(size_t top_n = 30) {
    if (!mem_trace_enabled())
        return;
    auto records = memory_ledger_snapshot();
    size_t committed = 0, capacity = 0, host = 0, views = 0;
    for (const auto& r : records) {
        committed += r.committed_bytes;
        capacity += r.capacity_bytes;
        host += r.host_bytes;
        views += r.is_view ? 1 : 0;
    }
    std::cout << "\n~~ DEME MEMORY LEDGER ~~\n";
    std::cout << "records=" << records.size() << ", views=" << views << ", device_committed=" << committed
              << ", device_capacity=" << capacity << ", host_mirror=" << host << " bytes\n";
    std::cout << "Top " << std::min(top_n, records.size()) << " allocations by committed bytes:\n";
    for (size_t i = 0; i < records.size() && i < top_n; ++i) {
        const auto& r = records[i];
        std::cout << "  " << std::setw(2) << i + 1 << ". " << r.owner << ":" << r.name << " role=" << r.role
                  << " logical=" << r.logical_bytes << " capacity=" << r.capacity_bytes
                  << " committed=" << r.committed_bytes << " host=" << r.host_bytes
                  << (r.is_view ? " view" : "") << "\n";
    }
    std::cout << "--------------------------\n";
}

}  // namespace detail

template <typename T>
class DualArray;
template <typename T>
class DeviceArray;
template <typename T>
bool swap_device_buffer(DualArray<T>& lhs, DeviceArray<T>& rhs);

namespace detail {

struct ProgramDeviceMemTracker {
    std::atomic<size_t> live_bytes{0};
    std::atomic<size_t> peak_bytes{0};
    std::mutex mutex;
    std::unordered_map<const void*, size_t> allocations;
};

inline ProgramDeviceMemTracker& tracked_device_mem() {
    static ProgramDeviceMemTracker tracker;
    return tracker;
}

inline void update_atomic_max(std::atomic<size_t>& target, size_t value) {
    size_t observed = target.load(std::memory_order_relaxed);
    while (observed < value &&
           !target.compare_exchange_weak(observed, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

inline void register_device_allocation(const void* ptr, size_t bytes) {
    if (!ptr || !bytes)
        return;
    auto& tracker = tracked_device_mem();
    {
        std::lock_guard<std::mutex> lock(tracker.mutex);
        tracker.allocations[ptr] = bytes;
    }
    const size_t live_now = tracker.live_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    update_atomic_max(tracker.peak_bytes, live_now);
}

inline void unregister_device_allocation(const void* ptr) {
    if (!ptr)
        return;
    auto& tracker = tracked_device_mem();
    size_t bytes = 0;
    {
        std::lock_guard<std::mutex> lock(tracker.mutex);
        auto it = tracker.allocations.find(ptr);
        if (it != tracker.allocations.end()) {
            bytes = it->second;
            tracker.allocations.erase(it);
        }
    }
    if (bytes)
        tracker.live_bytes.fetch_sub(bytes, std::memory_order_relaxed);
}

inline size_t get_tracked_program_device_peak_memory_usage() {
    return tracked_device_mem().peak_bytes.load(std::memory_order_relaxed);
}

inline void reset_tracked_program_device_peak_memory_usage() {
    auto& tracker = tracked_device_mem();
    tracker.peak_bytes.store(tracker.live_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

}  // namespace detail

// A to-device memcpy wrapper
template <typename T>
void CudaCopyToDevice(T* pD, T* pH) {
    DEME_GPU_CALL(cudaMemcpy(pD, pH, sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T>
void CudaCopyToDevice(T* pD, T* pH, size_t n) {
    DEME_GPU_CALL(cudaMemcpy(pD, pH, n * sizeof(T), cudaMemcpyHostToDevice));
}

// A to-host memcpy wrapper
template <typename T>
void CudaCopyToHost(T* pH, T* pD) {
    DEME_GPU_CALL(cudaMemcpy(pH, pD, sizeof(T), cudaMemcpyDeviceToHost));
}
template <typename T>
void CudaCopyToHost(T* pH, T* pD, size_t n) {
    DEME_GPU_CALL(cudaMemcpy(pH, pD, n * sizeof(T), cudaMemcpyDeviceToHost));
}

// ptr being a reference to a pointer is crucial
template <typename T>
inline void DevicePtrDealloc(T*& ptr) {
    if (!ptr)
        return;
    cudaPointerAttributes attrib;
    DEME_GPU_CALL(cudaPointerGetAttributes(&attrib, ptr));

    if (attrib.type != cudaMemoryType::cudaMemoryTypeUnregistered) {
        detail::unregister_device_allocation(ptr);
        DEME_GPU_CALL(cudaFree(ptr));
    }
}

// You have to deal with it yourself if ptr is an already-used device pointer
template <typename T>
inline void DevicePtrAlloc(T*& ptr, size_t size) {
    DEME_GPU_CALL(cudaMalloc((void**)&ptr, size * sizeof(T)));
    detail::register_device_allocation(ptr, size * sizeof(T));
}

template <typename T>
inline void HostPtrDealloc(T*& ptr) {
    if (!ptr)
        return;
    cudaPointerAttributes attrib;
    DEME_GPU_CALL(cudaPointerGetAttributes(&attrib, ptr));

    if (attrib.type != cudaMemoryType::cudaMemoryTypeUnregistered)
        DEME_GPU_CALL(cudaFreeHost(ptr));
}
template <typename T>
inline void HostPtrAlloc(T*& ptr, size_t size) {
    DEME_GPU_CALL(cudaMallocHost((void**)&ptr, size * sizeof(T)));
}

// Managed advise doesn't seem to do anything...
#define DEME_ADVISE_DEVICE(vec, device) \
    { advise(vec, ManagedAdvice::PREFERRED_LOC, device); }
#define DEME_MIGRATE_TO_DEVICE(vec, device, stream) \
    { migrate(vec, device, stream); }

// DEME_DUAL_ARRAY_RESIZE is a reminder for developers that a work array is resized, and this may automatically change
// the external device pointer this array's bound to. Therefore, after this call, syncing the data pointer bundle
// (granData) to device may be needed, and you remember to cudaSetDevice beforehand so it allocates to correct places.
#define DEME_DUAL_ARRAY_RESIZE(vec, newsize, val) \
    { vec.resize(newsize, val); }
#define DEME_DUAL_ARRAY_RESIZE_NOVAL(vec, newsize) \
    { vec.resize(newsize); }

// Simply a reminder that this is a device array resize, to distinguish from some general .resize calls
#define DEME_DEVICE_ARRAY_RESIZE(vec, newsize) \
    { vec.resize(newsize); }

// Use (void) to silence unused warnings.
// #define assertm(exp, msg) assert(((void)msg, exp))


// Non-owning device pointer plus size/capacity metadata. This is the lightweight role-split primitive used when
// an array is a borrowed view rather than an allocation owner.
template <typename T>
class DeviceView {
  public:
    DeviceView() = default;
    DeviceView(T* ptr, size_t size, size_t capacity = 0) : m_ptr(ptr), m_size(size), m_capacity(capacity ? capacity : size) {}

    void bind(T* ptr, size_t size, size_t capacity = 0) {
        m_ptr = ptr;
        m_size = size;
        m_capacity = capacity ? capacity : size;
    }

    T* data() { return m_ptr; }
    const T* data() const { return m_ptr; }
    T* device() { return m_ptr; }
    size_t size() const { return m_size; }
    size_t capacity() const { return m_capacity; }
    size_t getNumBytes() const { return m_capacity * sizeof(T); }
    explicit operator bool() const { return m_ptr != nullptr; }

  private:
    T* m_ptr = nullptr;
    size_t m_size = 0;
    size_t m_capacity = 0;
};

// Used for wrapping data structures so they become usable on GPU.
// We protect GPU-related data types with NonCopyable, because the device pointers inside these data types are too
// fragile for copying. If say a shallow copy is enforced to our array types in a vector-of-arrays resizing, then if you
// check the copied array's pointer from host, CUDA might not recognize it properly. The best practice is just using
// unique_ptr to manage these array classes if you expect to put them in places where some under-the-hood copying could
// happen.
template <typename T>
class DualStruct : private NonCopyable {
  private:
    T* host_data;           // Pointer to host memory (pinned)
    T* device_data;         // Pointer to device memory
    bool modified_on_host;  // Flag to track if host data has been modified
  public:
    // Constructor: Initialize and allocate memory for both host and device
    DualStruct() : modified_on_host(false) {
        DEME_GPU_CALL(cudaMallocHost((void**)&host_data, sizeof(T)));
        DEME_GPU_CALL(cudaMalloc((void**)&device_data, sizeof(T)));
    }

    // Constructor: Initialize and allocate memory for both host and device with init values
    DualStruct(T init_val) : modified_on_host(false) {
        DEME_GPU_CALL(cudaMallocHost((void**)&host_data, sizeof(T)));
        DEME_GPU_CALL(cudaMalloc((void**)&device_data, sizeof(T)));

        *host_data = init_val;

        toDevice();
    }

    // Destructor: Free memory
    ~DualStruct() { free(); }

    void free() {
        HostPtrDealloc(host_data);      // Free pinned memory
        DevicePtrDealloc(device_data);  // Free device memory
    }

    // Synchronize changes from host to device
    void toDevice() {
        DEME_GPU_CALL(cudaMemcpy(device_data, host_data, sizeof(T), cudaMemcpyHostToDevice));
        modified_on_host = false;
    }

    // Asynchronous host->device copy on a user stream. Host memory is pinned (cudaMallocHost),
    // so the caller must ensure host_data isn't modified until the copy completes.
    void toDeviceAsync(cudaStream_t stream) {
        DEME_GPU_CALL(cudaMemcpyAsync(device_data, host_data, sizeof(T), cudaMemcpyHostToDevice, stream));
        modified_on_host = false;
    }

    // Synchronize changes from device to host
    void toHost() {
        DEME_GPU_CALL(cudaMemcpy(host_data, device_data, sizeof(T), cudaMemcpyDeviceToHost));
        modified_on_host = false;
    }

    // Asynchronous device->host copy on a user stream. Host memory is pinned (cudaMallocHost).
    void toHostAsync(cudaStream_t stream) {
        DEME_GPU_CALL(cudaMemcpyAsync(host_data, device_data, sizeof(T), cudaMemcpyDeviceToHost, stream));
        modified_on_host = false;
    }

    // Synchronize change of one field of the struct to device
    template <typename MemberType>
    void syncMemberToDevice(ptrdiff_t offset) {
        DEME_GPU_CALL(cudaMemcpy(reinterpret_cast<char*>(device_data) + offset,
                                 reinterpret_cast<char*>(host_data) + offset, sizeof(MemberType),
                                 cudaMemcpyHostToDevice));
    }

    // Asynchronous partial host->device copy on a user stream.
    template <typename MemberType>
    void syncMemberToDeviceAsync(ptrdiff_t offset, cudaStream_t stream) {
        DEME_GPU_CALL(cudaMemcpyAsync(reinterpret_cast<char*>(device_data) + offset,
                                      reinterpret_cast<char*>(host_data) + offset, sizeof(MemberType),
                                      cudaMemcpyHostToDevice, stream));
    }

    // Synchronize change of one field of the struct to host
    template <typename MemberType>
    void syncMemberToHost(ptrdiff_t offset) {
        DEME_GPU_CALL(cudaMemcpy(reinterpret_cast<char*>(host_data) + offset,
                                 reinterpret_cast<char*>(device_data) + offset, sizeof(MemberType),
                                 cudaMemcpyDeviceToHost));
    }

    // Asynchronous partial device->host copy on a user stream.
    template <typename MemberType>
    void syncMemberToHostAsync(ptrdiff_t offset, cudaStream_t stream) {
        DEME_GPU_CALL(cudaMemcpyAsync(reinterpret_cast<char*>(host_data) + offset,
                                      reinterpret_cast<char*>(device_data) + offset, sizeof(MemberType),
                                      cudaMemcpyDeviceToHost, stream));
    }

    // Check if host data has been modified and not synced
    bool checkNoPendingModification() { return !modified_on_host; }

    void markModified() { modified_on_host = true; }

    void unmarkModified() { modified_on_host = false; }

    // Accessor for host data (using the arrow operator)
    T* operator->() { return host_data; }

    // Accessor for host data (using the arrow operator)
    T* operator->() const { return host_data; }

    // Dereference operator for simple types (like float) to access the value directly
    T& operator*() {
        return *host_data;  // Return a reference to the value
    }

    // Dereference operator for simple types (like float) to access the value directly
    T& operator*() const {
        return *host_data;  // Return a reference to the value
    }

    // Overloaded operator& for device pointer access
    T* operator&() const {
        return device_data;  // Return device pointer when using &
    }

    // Getter for the device pointer
    T* getDevicePointer() { return device_data; }

    // Getter for the host pointer
    T* getHostPointer() { return host_data; }

    // Get host or device size in bytes
    size_t getNumBytes() const { return sizeof(T); }
};

#ifndef DEME_USE_MANAGED_ARRAYS
// CPU--GPU unified array, leveraging pinned memory
template <typename T>
class DualArray : private NonCopyable {
  public:
    using PinnedVector = std::vector<T, PinnedAllocator<T>>;
    template <typename U>
    friend bool swap_device_buffer(DualArray<U>& lhs, DeviceArray<U>& rhs);

    explicit DualArray(size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        ensureHostVector();
    }

    DualArray(size_t n, size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        resize(n);
    }

    DualArray(size_t n, T val, size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        resize(n, val);
    }

    DualArray(const std::vector<T>& vec,
              size_t* host_external_counter = nullptr,
              size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        attachHostVector(&vec, /*deep_copy=*/true);
    }

    ~DualArray() { free(); }

    void resize(size_t n) {
        assert(m_host_vec_ptr == m_pinned_vec.get() && "resize() requires internal host ownership");
        resizeHost(n);
        resizeDevice(n);
    }

    // This resize flavor fills host values only!
    void resize(size_t n, const T& val) {
        assert(m_host_vec_ptr == m_pinned_vec.get() && "resize() requires internal host ownership");
        resizeHost(n, val);
        resizeDevice(n);
    }

    void resizeHost(size_t n) {
        ensureHostVector();  // allocates pinned vec if null
        size_t old_bytes = m_host_vec_ptr->size() * sizeof(T);
        m_host_vec_ptr->resize(n);
        size_t new_bytes = m_host_vec_ptr->size() * sizeof(T);
        updateHostMemCounter(static_cast<ssize_t>(new_bytes) - static_cast<ssize_t>(old_bytes));
        recordMemory("resizeHost");
    }

    void resizeHost(size_t n, const T& val) {
        ensureHostVector();  // allocates pinned vec if null
        size_t old_bytes = m_host_vec_ptr->size() * sizeof(T);
        m_host_vec_ptr->resize(n, val);
        size_t new_bytes = m_host_vec_ptr->size() * sizeof(T);
        updateHostMemCounter(static_cast<ssize_t>(new_bytes) - static_cast<ssize_t>(old_bytes));
        recordMemory("resizeHost");
    }

    // m_device_capacity is allocated memory, not array usable data range.
    // Also, this method preserves already-existing device data. If this DualArray
    // currently views an external DeviceArray, resizeDevice detaches from that
    // borrowed pointer and creates owned storage before returning.
    void resizeDevice(size_t n, bool allow_shrink = false) {
        if (m_device_owned && !allow_shrink && m_device_capacity >= n) {
            recordMemory("resizeDevice-reuse");
            return;
        }
        if (n == 0) {
            freeDevice();
            return;
        }

        T* new_device_ptr = nullptr;
        DevicePtrAlloc(new_device_ptr, n);

        // If previous data exists, copy the minimum amount. This is valid for both
        // owned storage and non-owning views of another device allocation.
        if (m_device_ptr && m_device_capacity > 0) {
            size_t copy_count = std::min(n, m_device_capacity);
            DEME_GPU_CALL(cudaMemcpy(new_device_ptr, m_device_ptr, copy_count * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        // Free old memory and update bookkeeping only when this DualArray owned it.
        if (m_device_owned) {
            updateDeviceMemCounter(-(ssize_t)(m_device_capacity * sizeof(T)));
            DevicePtrDealloc(m_device_ptr);
        }

        m_device_ptr = new_device_ptr;
        m_device_owned = true;
        updateBoundDevicePointer();

        updateDeviceMemCounter(static_cast<ssize_t>(n * sizeof(T)));
        m_device_capacity = n;
        recordMemory("resizeDevice");
    }

    // Borrow an already-owned device buffer without taking ownership. This is used
    // by same-GPU dT contact-array views: dT kernels read directly from the kT->dT
    // ping-pong buffer, while the DeviceArray remains responsible for freeing that
    // allocation. Host size is managed separately through resizeHost().
    void setDeviceView(T* external_device_ptr, size_t capacity) {
        if (m_device_owned) {
            updateDeviceMemCounter(-(ssize_t)(m_device_capacity * sizeof(T)));
            DevicePtrDealloc(m_device_ptr);
        }
        m_device_ptr = external_device_ptr;
        m_device_capacity = capacity;
        m_device_owned = false;
        updateBoundDevicePointer();
        recordMemory("setDeviceView");
    }

    bool isDeviceView() const { return !m_device_owned && m_device_ptr != nullptr; }

    void freeHost() {
        if (m_host_vec_ptr) {
            updateHostMemCounter(-(ssize_t)(m_host_vec_ptr->size() * sizeof(T)));
        }
        m_pinned_vec.reset();
        m_host_vec_ptr = nullptr;
        m_host_dirty = false;
        recordMemory("freeHost");
    }

    void freeDevice() {
        if (m_device_owned) {
            DevicePtrDealloc(m_device_ptr);
            updateDeviceMemCounter(-(ssize_t)(m_device_capacity * sizeof(T)));
        }
        m_device_ptr = nullptr;
        m_device_capacity = 0;
        m_device_owned = true;
        updateBoundDevicePointer();
        recordMemory("freeDevice");
    }

    void free() {
        freeDevice();
        freeHost();
    }

    void toDevice() {
        assert(m_host_vec_ptr);
        size_t count = size();
        if (!m_device_owned || count > m_device_capacity)
            resizeDevice(count);
        DEME_GPU_CALL(cudaMemcpy(m_device_ptr, m_host_vec_ptr->data(), count * sizeof(T), cudaMemcpyHostToDevice));
        m_host_dirty = false;
    }

    void toDevice(size_t start, size_t n) {
        assert(m_host_vec_ptr);
        const size_t need = start + n;
        if (!m_device_owned || need > m_device_capacity) {
            resizeDevice(std::max(size(), need));
        }
        DEME_GPU_CALL(
            cudaMemcpy(m_device_ptr + start, m_host_vec_ptr->data() + start, n * sizeof(T), cudaMemcpyHostToDevice));
        m_host_dirty = false;
    }

    void toDeviceAsync(cudaStream_t& stream) {
        assert(m_host_vec_ptr);
        size_t count = size();
        if (!m_device_owned || count > m_device_capacity)
            resizeDevice(count);
        DEME_GPU_CALL(
            cudaMemcpyAsync(m_device_ptr, m_host_vec_ptr->data(), count * sizeof(T), cudaMemcpyHostToDevice, stream));
        m_host_dirty = false;
    }

    // And partial update methods...
    // Normally this is preferred when they are used in tracker implementation
    void toDeviceAsync(cudaStream_t& stream, size_t start, size_t n) {
        assert(m_host_vec_ptr);
        const size_t need = start + n;
        if (!m_device_owned || need > m_device_capacity) {
            resizeDevice(std::max(size(), need));
        }
        DEME_GPU_CALL(cudaMemcpyAsync(m_device_ptr + start, m_host_vec_ptr->data() + start, n * sizeof(T),
                                      cudaMemcpyHostToDevice, stream));
        m_host_dirty = false;
    }

    void toHost() {
        assert(m_device_ptr && m_host_vec_ptr);
        if (size() > m_device_capacity) {
            DEME_ERROR("DualArray::toHost would read past device capacity for %s: host size %zu, device capacity %zu",
                       m_mem_name.c_str(), size(), m_device_capacity);
        }
        DEME_GPU_CALL(cudaMemcpy(m_host_vec_ptr->data(), m_device_ptr, size() * sizeof(T), cudaMemcpyDeviceToHost));
        m_host_dirty = false;
    }

    void toHost(size_t start, size_t n) {
        assert(m_device_ptr && m_host_vec_ptr);
        if (start + n > m_device_capacity) {
            DEME_ERROR("DualArray::toHost range would read past device capacity for %s: range [%zu,%zu), device capacity %zu",
                       m_mem_name.c_str(), start, start + n, m_device_capacity);
        }
        DEME_GPU_CALL(
            cudaMemcpy(m_host_vec_ptr->data() + start, m_device_ptr + start, n * sizeof(T), cudaMemcpyDeviceToHost));
        m_host_dirty = false;
    }

    void toHostAsync(cudaStream_t& stream) {
        assert(m_host_vec_ptr && m_device_ptr);
        if (size() > m_device_capacity) {
            DEME_ERROR("DualArray::toHostAsync would read past device capacity for %s: host size %zu, device capacity %zu",
                       m_mem_name.c_str(), size(), m_device_capacity);
        }
        DEME_GPU_CALL(
            cudaMemcpyAsync(m_host_vec_ptr->data(), m_device_ptr, size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        m_host_dirty = false;
    }

    void toHostAsync(cudaStream_t& stream, size_t start, size_t n) {
        assert(m_host_vec_ptr && m_device_ptr);
        if (start + n > m_device_capacity) {
            DEME_ERROR("DualArray::toHostAsync range would read past device capacity for %s: range [%zu,%zu), device capacity %zu",
                       m_mem_name.c_str(), start, start + n, m_device_capacity);
        }
        DEME_GPU_CALL(cudaMemcpyAsync(m_host_vec_ptr->data() + start, m_device_ptr + start, n * sizeof(T),
                                      cudaMemcpyDeviceToHost, stream));
        m_host_dirty = false;
    }

    T getVal(size_t start) {
        toHost(start, 1);  // sync from device to host
        return (*m_host_vec_ptr)[start];
    }

    std::vector<T> getVal(size_t start, size_t n) {
        toHost(start, n);  // sync from device to host
        return std::vector<T>(m_host_vec_ptr->begin() + start, m_host_vec_ptr->begin() + start + n);
    }

    void setVal(const T& data, size_t start) {
        (*m_host_vec_ptr)[start] = data;
        toDevice(start, 1);
    }

    void setVal(const std::vector<T>& data, size_t start, size_t n = 0) {
        size_t count = (n > 0) ? n : data.size();
        // Copy to host vector
        std::copy(data.begin(), data.begin() + count, m_host_vec_ptr->begin() + start);
        toDevice(start, count);
    }

    void setVal(cudaStream_t& stream, const T& data, size_t start) {
        (*m_host_vec_ptr)[start] = data;
        toDeviceAsync(stream, start, 1);
    }

    void setVal(cudaStream_t& stream, const std::vector<T>& data, size_t start, size_t n = 0) {
        size_t count = (n > 0) ? n : data.size();
        // Copy to host vector
        std::copy(data.begin(), data.begin() + count, m_host_vec_ptr->begin() + start);
        toDeviceAsync(stream, start, count);
    }

    void markHostModified() { m_host_dirty = true; }
    void unmarkHostModified() { m_host_dirty = false; }

    // Array's in-use data range is always stored on host by size()
    size_t size() const { return m_host_vec_ptr ? m_host_vec_ptr->size() : 0; }

    // Get host or device size in bytes
    size_t getNumBytes() const { return m_host_vec_ptr ? m_host_vec_ptr->size() * sizeof(T) : 0; }

    T* host() { return m_host_vec_ptr ? m_host_vec_ptr->data() : nullptr; }

    T* device() { return m_device_ptr; }

    // Overloaded operator& for device pointer access
    T* operator&() const { return m_device_ptr; }

    // data() returns device data for the ease of packing pointers
    T* data() { return device(); }

    PinnedVector& getHostVector() { return *m_host_vec_ptr; }

    void bindDevicePointer(T** external_ptr_to_ptr) {
        m_bound_device_ptr = external_ptr_to_ptr;
        updateBoundDevicePointer();
    }

    void unbindDevicePointer() { m_bound_device_ptr = nullptr; }

    void setHostMemoryCounter(size_t* counter) { m_host_mem_counter = counter; }
    void setDeviceMemoryCounter(size_t* counter) { m_device_mem_counter = counter; }

    void setMemoryContext(const std::string& owner, const std::string& name, MemoryRole role) {
        m_mem_owner = owner;
        m_mem_name = name;
        m_mem_role = role;
        recordMemory("setMemoryContext");
    }

    void setMemoryPhase(const std::string& phase) {
        m_mem_phase = phase;
        recordMemory(phase);
    }
    // You can use nullptr to unbind

    void attachHostVector(const std::vector<T>* external_vec, bool deep_copy = true) {
        freeHost();  // discard internal memory and update memory tracker
        if (deep_copy) {
            m_pinned_vec = std::make_unique<PinnedVector>(external_vec->begin(), external_vec->end());
            m_host_vec_ptr = m_pinned_vec.get();
            updateHostMemCounter(static_cast<ssize_t>(m_host_vec_ptr->size() * sizeof(T)));
        } else {
            m_host_vec_ptr = const_cast<PinnedVector*>(reinterpret_cast<const PinnedVector*>(external_vec));
        }
        m_host_dirty = true;
        recordMemory("attachHostVector");
    }

    T& operator[](size_t i) { return (*m_host_vec_ptr)[i]; }
    const T& operator[](size_t i) const { return (*m_host_vec_ptr)[i]; }
    T operator()(size_t i) { return getVal(i); }

  private:
    std::unique_ptr<PinnedVector> m_pinned_vec = nullptr;
    PinnedVector* m_host_vec_ptr = nullptr;

    size_t* m_host_mem_counter = nullptr;
    size_t* m_device_mem_counter = nullptr;

    T* m_device_ptr = nullptr;
    size_t m_device_capacity = 0;
    bool m_device_owned = true;

    T** m_bound_device_ptr = nullptr;

    bool m_host_dirty = false;

    std::string m_mem_owner = "unassigned";
    std::string m_mem_name;
    MemoryRole m_mem_role = MemoryRole::Unknown;
    std::string m_mem_phase = "construct";
    size_t m_high_water_capacity = 0;

    void ensureHostVector(size_t n = 0) {
        if (!m_host_vec_ptr) {
            m_pinned_vec = std::make_unique<PinnedVector>(n);
            m_host_vec_ptr = m_pinned_vec.get();
        }
    }

    void updateBoundDevicePointer() {
        if (m_bound_device_ptr)
            *m_bound_device_ptr = m_device_ptr;
    }

    void recordMemory(const std::string& phase) {
        if (!detail::mem_trace_enabled())
            return;
        m_mem_phase = phase;
        const size_t host_bytes = m_host_vec_ptr ? m_host_vec_ptr->size() * sizeof(T) : 0;
        const size_t cap_bytes = m_device_capacity * sizeof(T);
        m_high_water_capacity = std::max(m_high_water_capacity, cap_bytes);
        if (!host_bytes && !cap_bytes && !m_device_ptr) {
            detail::erase_memory_ledger_record(this);
            return;
        }
        detail::MemoryLedgerRecord rec;
        rec.array_object = this;
        rec.owner = m_mem_owner;
        rec.name = m_mem_name.empty() ? (std::string("DualArray<") + typeid(T).name() + ">@" +
                                         std::to_string(reinterpret_cast<std::uintptr_t>(this))) : m_mem_name;
        rec.role = MemoryRoleName(m_mem_role);
        rec.type_name = typeid(T).name();
        rec.type_size = sizeof(T);
        rec.logical_bytes = host_bytes;
        rec.capacity_bytes = cap_bytes;
        rec.committed_bytes = m_device_owned ? cap_bytes : 0;
        rec.host_bytes = host_bytes;
        rec.high_water_capacity_bytes = m_high_water_capacity;
        rec.device_ptr = m_device_ptr;
        rec.is_view = !m_device_owned && m_device_ptr != nullptr;
        rec.has_host_mirror = m_host_vec_ptr != nullptr;
        rec.backend = m_device_owned ? "legacy" : "view";
        rec.phase = m_mem_phase;
        detail::upsert_memory_ledger_record(rec);
    }

    void updateHostMemCounter(ssize_t delta) {
        if (m_host_mem_counter)
            *m_host_mem_counter += delta;
    }

    void updateDeviceMemCounter(ssize_t delta) {
        if (m_device_mem_counter)
            *m_device_mem_counter += delta;
    }
};
#else
// CPU--GPU unified array, leveraging managed memory
template <typename T>
class DualArray : private NonCopyable {
  public:
    using ManagedVector = std::vector<T, ManagedAllocator<T>>;
    template <typename U>
    friend bool swap_device_buffer(DualArray<U>& lhs, DeviceArray<U>& rhs);

    explicit DualArray(size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        ensureHostVector();
    }

    DualArray(size_t n, size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        resize(n);
    }

    DualArray(size_t n, T val, size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {
        resize(n, val);
    }

    ~DualArray() { free(); }

    void resize(size_t n) {
        assert(m_host_vec_ptr == m_pinned_vec.get() && "resize() requires internal host ownership");
        resizeHost(n);
        resizeDevice(n);
    }

    // This resize flavor fills host values only!
    void resize(size_t n, const T& val) {
        assert(m_host_vec_ptr == m_pinned_vec.get() && "resize() requires internal host ownership");
        resizeHost(n, val);
        resizeDevice(n);
    }

    void resizeHost(size_t n) {
        ensureHostVector();  // allocates pinned vec if null
        size_t old_bytes = m_host_vec_ptr->size() * sizeof(T);
        m_host_vec_ptr->resize(n);
        size_t new_bytes = m_host_vec_ptr->size() * sizeof(T);
        updateMemCounter(static_cast<ssize_t>(new_bytes) - static_cast<ssize_t>(old_bytes));
        updateBoundDevicePointer();
        recordMemory("resizeHost");
    }

    void resizeHost(size_t n, const T& val) {
        ensureHostVector();  // allocates pinned vec if null
        size_t old_bytes = m_host_vec_ptr->size() * sizeof(T);
        m_host_vec_ptr->resize(n, val);
        size_t new_bytes = m_host_vec_ptr->size() * sizeof(T);
        updateMemCounter(static_cast<ssize_t>(new_bytes) - static_cast<ssize_t>(old_bytes));
        updateBoundDevicePointer();
        recordMemory("resizeHost");
    }

    // m_device_capacity is allocated memory, not array usable data range
    void resizeDevice(size_t n, bool allow_shrink = false) {}

    void freeHost() {
        if (m_host_vec_ptr) {
            updateMemCounter(-(ssize_t)(m_host_vec_ptr->size() * sizeof(T)));
        }
        m_pinned_vec.reset();
        m_host_vec_ptr = nullptr;
        updateBoundDevicePointer();
        recordMemory("freeHost");
    }

    void freeDevice() {}

    void setDeviceView(T*, size_t) {}

    bool isDeviceView() const { return false; }

    void free() {
        freeDevice();
        freeHost();
    }

    void toDevice() {}

    void toDevice(size_t start, size_t n) {}

    void toDeviceAsync(cudaStream_t& stream) {}

    void toDeviceAsync(cudaStream_t& stream, size_t start, size_t n) {}

    void toHost() {}

    void toHost(size_t start, size_t n) {}

    void toHostAsync(cudaStream_t& stream) {}

    void toHostAsync(cudaStream_t& stream, size_t start, size_t n) {}

    T getVal(size_t start) { return (*m_host_vec_ptr)[start]; }

    std::vector<T> getVal(size_t start, size_t n) {
        return std::vector<T>(m_host_vec_ptr->begin() + start, m_host_vec_ptr->begin() + start + n);
    }

    void setVal(const T& data, size_t start) { (*m_host_vec_ptr)[start] = data; }

    void setVal(const std::vector<T>& data, size_t start, size_t n = 0) {
        size_t count = (n > 0) ? n : data.size();
        std::copy(data.begin(), data.begin() + count, m_host_vec_ptr->begin() + start);
    }

    void setVal(cudaStream_t& stream, const T& data, size_t start) { (*m_host_vec_ptr)[start] = data; }

    void setVal(cudaStream_t& stream, const std::vector<T>& data, size_t start, size_t n = 0) {
        size_t count = (n > 0) ? n : data.size();
        std::copy(data.begin(), data.begin() + count, m_host_vec_ptr->begin() + start);
    }

    void markHostModified() { m_host_dirty = true; }
    void unmarkHostModified() { m_host_dirty = false; }

    // Array's in-use data range is always stored on host by size()
    size_t size() const { return m_host_vec_ptr ? m_host_vec_ptr->size() : 0; }

    // Get host or device size in bytes
    size_t getNumBytes() const { return m_host_vec_ptr ? m_host_vec_ptr->size() * sizeof(T) : 0; }

    T* host() { return m_host_vec_ptr ? m_host_vec_ptr->data() : nullptr; }

    T* device() { return host(); }

    // Overloaded operator& for device pointer access
    T* operator&() const { return host(); }

    // data() returns device data for the ease of packing pointers
    T* data() { return host(); }

    ManagedVector& getHostVector() { return *m_host_vec_ptr; }

    void bindDevicePointer(T** external_ptr_to_ptr) {
        m_bound_device_ptr = external_ptr_to_ptr;
        updateBoundDevicePointer();
    }

    void unbindDevicePointer() { m_bound_device_ptr = nullptr; }

    void setHostMemoryCounter(size_t* counter) { m_host_mem_counter = counter; }
    void setDeviceMemoryCounter(size_t* counter) { m_device_mem_counter = counter; }

    void setMemoryContext(const std::string& owner, const std::string& name, MemoryRole role) {
        m_mem_owner = owner;
        m_mem_name = name;
        m_mem_role = role;
        recordMemory("setMemoryContext");
    }

    void setMemoryPhase(const std::string& phase) {
        m_mem_phase = phase;
        recordMemory(phase);
    }
    // You can use nullptr to unbind

    T& operator[](size_t i) { return (*m_host_vec_ptr)[i]; }
    const T& operator[](size_t i) const { return (*m_host_vec_ptr)[i]; }
    T operator()(size_t i) { return getVal(i); }

  private:
    std::unique_ptr<ManagedVector> m_pinned_vec = nullptr;
    ManagedVector* m_host_vec_ptr = nullptr;

    size_t* m_host_mem_counter = nullptr;
    size_t* m_device_mem_counter = nullptr;

    T** m_bound_device_ptr = nullptr;

    bool m_host_dirty = false;

    std::string m_mem_owner = "unassigned";
    std::string m_mem_name;
    MemoryRole m_mem_role = MemoryRole::Unknown;
    std::string m_mem_phase = "construct";
    size_t m_high_water_capacity = 0;

    void ensureHostVector(size_t n = 0) {
        if (!m_host_vec_ptr) {
            m_pinned_vec = std::make_unique<ManagedVector>(n);
            m_host_vec_ptr = m_pinned_vec.get();
        }
    }

    void updateBoundDevicePointer() {
        if (m_bound_device_ptr)
            *m_bound_device_ptr = host();
    }

    void recordMemory(const std::string& phase) {
        if (!detail::mem_trace_enabled())
            return;
        m_mem_phase = phase;
        const size_t bytes = m_host_vec_ptr ? m_host_vec_ptr->size() * sizeof(T) : 0;
        m_high_water_capacity = std::max(m_high_water_capacity, bytes);
        if (!bytes) {
            detail::erase_memory_ledger_record(this);
            return;
        }
        detail::MemoryLedgerRecord rec;
        rec.array_object = this;
        rec.owner = m_mem_owner;
        rec.name = m_mem_name.empty() ? (std::string("ManagedDualArray<") + typeid(T).name() + ">@" +
                                         std::to_string(reinterpret_cast<std::uintptr_t>(this))) : m_mem_name;
        rec.role = MemoryRoleName(m_mem_role);
        rec.type_name = typeid(T).name();
        rec.type_size = sizeof(T);
        rec.logical_bytes = bytes;
        rec.capacity_bytes = bytes;
        rec.committed_bytes = bytes;
        rec.host_bytes = bytes;
        rec.high_water_capacity_bytes = m_high_water_capacity;
        rec.device_ptr = m_host_vec_ptr ? m_host_vec_ptr->data() : nullptr;
        rec.is_view = false;
        rec.has_host_mirror = true;
        rec.backend = "managed";
        rec.phase = m_mem_phase;
        detail::upsert_memory_ledger_record(rec);
    }

    void updateMemCounter(ssize_t delta) {
        if (m_host_mem_counter)
            *m_host_mem_counter += delta;
        if (m_device_mem_counter)
            *m_device_mem_counter += delta;
    }
};
#endif

// Pure device data type, usually used for scratching space
template <typename T>
class DeviceArray : private NonCopyable {
  public:
    template <typename U>
    friend bool swap_device_buffer(DualArray<U>& lhs, DeviceArray<U>& rhs);
    DeviceArray(size_t* external_counter = nullptr) : m_mem_counter(external_counter) {}

    explicit DeviceArray(size_t n, size_t* external_counter = nullptr) : m_mem_counter(external_counter) { resize(n); }

    ~DeviceArray() { free(); }

    // In practice, we use device array as temp arrays. The default preserves legacy behavior, but scratch/transfer
    // call sites can explicitly use PreserveOnResize::Discard to avoid a pointless device-to-device copy.
    void resize(size_t n,
                bool allow_shrink = false,
                PreserveOnResize preserve = PreserveOnResize::Preserve) {
        // Preserve legacy semantics: resize(0) must not free an already allocated scratch buffer unless
        // the caller explicitly allows shrinking. Some call sites keep bound device pointers valid even
        // while the logical work count is zero.
        if (!allow_shrink && m_capacity >= n) {
            recordMemory("resize-reuse");
            return;
        }
        if (n == 0) {
            free();
            return;
        }
        T* new_device_ptr = nullptr;
        DevicePtrAlloc(new_device_ptr, n);

        // If previous data exists, copy the minimum amount only when the caller asked to preserve it.
        if (preserve == PreserveOnResize::Preserve && m_data && m_capacity > 0) {
            size_t copy_count = std::min(n, m_capacity);
            DEME_GPU_CALL(cudaMemcpy(new_device_ptr, m_data, copy_count * sizeof(T), cudaMemcpyDeviceToDevice));
        }
        // Free old memory and update bookkeeping
        updateMemCounter(-(ssize_t)(m_capacity * sizeof(T)));
        DevicePtrDealloc(m_data);

        m_data = new_device_ptr;
        updateBoundDevicePointer();

        updateMemCounter(static_cast<ssize_t>(n * sizeof(T)));
        m_capacity = n;
        recordMemory(preserve == PreserveOnResize::Discard ? "resizeDiscard" : "resize");
    }

    void resizeDiscard(size_t n, bool allow_shrink = false) {
        resize(n, allow_shrink, PreserveOnResize::Discard);
    }

    void free() {
        DevicePtrDealloc(m_data);
        updateMemCounter(-(ssize_t)(m_capacity * sizeof(T)));
        m_data = nullptr;
        m_capacity = 0;
        updateBoundDevicePointer();
        recordMemory("free");
    }

    void bindDevicePointer(T** external_ptr_to_ptr) {
        m_bound_device_ptr = external_ptr_to_ptr;
        updateBoundDevicePointer();
    }
    void unbindDevicePointer() { m_bound_device_ptr = nullptr; }

    size_t size() const { return m_capacity; }

    // Get host or device size in bytes
    size_t getNumBytes() const { return m_capacity * sizeof(T); }

    T* data() { return m_data; }

    const T* data() const { return m_data; }

    void setMemoryCounter(size_t* counter) { m_mem_counter = counter; }

    void setMemoryContext(const std::string& owner, const std::string& name, MemoryRole role) {
        m_mem_owner = owner;
        m_mem_name = name;
        m_mem_role = role;
        recordMemory("setMemoryContext");
    }

    void setMemoryPhase(const std::string& phase) {
        m_mem_phase = phase;
        recordMemory(phase);
    }

  private:
    T* m_data = nullptr;
    T** m_bound_device_ptr = nullptr;
    size_t m_capacity = 0;
    size_t* m_mem_counter = nullptr;

    std::string m_mem_owner = "unassigned";
    std::string m_mem_name;
    MemoryRole m_mem_role = MemoryRole::Unknown;
    std::string m_mem_phase = "construct";
    size_t m_high_water_capacity = 0;

    void updateBoundDevicePointer() {
        if (m_bound_device_ptr)
            *m_bound_device_ptr = m_data;
    }

    void recordMemory(const std::string& phase) {
        if (!detail::mem_trace_enabled())
            return;
        m_mem_phase = phase;
        const size_t cap_bytes = m_capacity * sizeof(T);
        m_high_water_capacity = std::max(m_high_water_capacity, cap_bytes);
        if (!cap_bytes && !m_data) {
            detail::erase_memory_ledger_record(this);
            return;
        }
        detail::MemoryLedgerRecord rec;
        rec.array_object = this;
        rec.owner = m_mem_owner;
        rec.name = m_mem_name.empty() ? (std::string("DeviceArray<") + typeid(T).name() + ">@" +
                                         std::to_string(reinterpret_cast<std::uintptr_t>(this))) : m_mem_name;
        rec.role = MemoryRoleName(m_mem_role);
        rec.type_name = typeid(T).name();
        rec.type_size = sizeof(T);
        rec.logical_bytes = cap_bytes;
        rec.capacity_bytes = cap_bytes;
        rec.committed_bytes = cap_bytes;
        rec.host_bytes = 0;
        rec.high_water_capacity_bytes = m_high_water_capacity;
        rec.device_ptr = m_data;
        rec.is_view = false;
        rec.has_host_mirror = false;
        rec.backend = "legacy";
        rec.phase = m_mem_phase;
        detail::upsert_memory_ledger_record(rec);
    }

    void updateMemCounter(ssize_t delta) {
        if (m_mem_counter)
            *m_mem_counter += delta;
    }
};

#ifndef DEME_USE_MANAGED_ARRAYS
template <typename T>
inline bool swap_device_buffer(DualArray<T>& lhs, DeviceArray<T>& rhs) {
    if (!lhs.m_device_owned) {
        return false;
    }
    using std::swap;
    swap(lhs.m_device_ptr, rhs.m_data);
    swap(lhs.m_device_capacity, rhs.m_capacity);
    lhs.updateBoundDevicePointer();
    lhs.recordMemory("swapDeviceBuffer");
    rhs.recordMemory("swapDeviceBuffer");
    return true;
}
#else
template <typename T>
inline bool swap_device_buffer(DualArray<T>&, DeviceArray<T>&) {
    return false;
}
#endif

/// @brief General abstraction of vector pool
/// @tparam T Array data type
/// @tparam DerivedEnclosedData The data type of the enclosed arrays in the derived pool class
template <typename T, typename DerivedEnclosedData>
class ResourcePool : private NonCopyable {
  protected:
    std::vector<std::unique_ptr<DerivedEnclosedData>> vectors;
    std::vector<std::optional<std::string>> in_use;
    std::unordered_map<std::string, size_t> name_to_index;

  public:
    virtual ~ResourcePool() { releaseAll(); }

    void resize(const std::string& name, size_t new_size) {
        auto it = name_to_index.find(name);
        if (it == name_to_index.end()) {
            DEME_ERROR(std::string("Cannot resize: name not found"));
        }
        vectors[it->second]->resize(new_size);
    }

    bool exist(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end()) {
            return false;
        } else {
            return true;
        }
    }

    void unclaim(const std::string& name) {
        auto it = name_to_index.find(name);
        if (it == name_to_index.end())
            return;
        in_use[it->second] = std::nullopt;
        name_to_index.erase(it);
    }

    void release(const std::string& name) {
        auto it = name_to_index.find(name);
        if (it == name_to_index.end()) {
            DEME_ERROR("Cannot release: name not found");
        }
        vectors[it->second]->free();
        in_use[it->second] = std::nullopt;
        name_to_index.erase(it);
    }

    void releaseAll() {
        for (size_t i = 0; i < vectors.size(); ++i) {
            vectors[i]->free();
        }
        in_use.clear();
        name_to_index.clear();
    }

    size_t cachedFreeBytes() const {
        size_t bytes = 0;
        for (size_t i = 0; i < vectors.size(); ++i) {
            if (i >= in_use.size() || !in_use[i]) {
                bytes += vectors[i]->getNumBytes();
            }
        }
        return bytes;
    }

    size_t trimFreeMemory(size_t keep_bytes = 0) {
        size_t cached = cachedFreeBytes();
        size_t freed = 0;
        while (cached > keep_bytes) {
            size_t best = static_cast<size_t>(-1);
            size_t best_bytes = 0;
            for (size_t i = 0; i < vectors.size(); ++i) {
                if (i < in_use.size() && in_use[i])
                    continue;
                const size_t bytes = vectors[i]->getNumBytes();
                if (bytes > best_bytes) {
                    best = i;
                    best_bytes = bytes;
                }
            }
            if (best == static_cast<size_t>(-1) || best_bytes == 0)
                break;
            vectors[best]->free();
            cached -= best_bytes;
            freed += best_bytes;
        }
        return freed;
    }

    void printStatus() const {
        for (size_t i = 0; i < in_use.size(); ++i) {
            if (in_use[i]) {
                const std::string& name = *in_use[i];
                const size_t index = name_to_index.at(name);
                const auto& vec = vectors[index];
                std::cout << "Storage vector[" << i << "] in use as \"" << name << "\", using " << vec->getNumBytes()
                          << " bytes.\n";
            } else {
                std::cout << "Storage vector[" << i << "] is free\n";
            }
        }
    }
};

/// @brief A pool that dispatches device arrays
/// @tparam T Array data type
template <typename T>
class DeviceVectorPool : public ResourcePool<T, DeviceArray<T>> {
  public:
    explicit DeviceVectorPool(size_t* external_counter = nullptr) : m_mem_counter(external_counter) {}

    T* claim(const std::string& name, size_t size, bool allow_duplicate = false) {
        // Referring to the base class that correspond to me
        using Base = ResourcePool<T, DeviceArray<T>>;

        if (Base::name_to_index.count(name)) {
            if (allow_duplicate) {
                size_t index = Base::name_to_index[name];
                Base::vectors[index]->setMemoryContext("scratch", name, MemoryRole::ScratchTemporary);
                Base::vectors[index]->resize(size);
                return Base::vectors[index]->data();
            } else {
                DEME_ERROR("Name already claimed: %s", name.c_str());
            }
        }

        // Preserve the legacy first-free reuse policy. A previous best-fit experiment looked attractive locally,
        // but in patch-heavy runs it prevented large scratch slots from being recycled by later phases and caused
        // several phase-local arrays to remain resident at once, regressing peak VRAM.
        for (size_t i = 0; i < Base::in_use.size(); i++) {
            if (!Base::in_use[i]) {
                Base::vectors[i]->setMemoryContext("scratch", name, MemoryRole::ScratchTemporary);
                Base::vectors[i]->resize(size);
                Base::in_use[i] = name;
                Base::name_to_index[name] = i;
                return Base::vectors[i]->data();
            }
        }

        Base::vectors.emplace_back(std::make_unique<DeviceArray<T>>(m_mem_counter));
        Base::in_use.emplace_back(name);
        size_t new_index = Base::vectors.size() - 1;
        Base::name_to_index[name] = new_index;
        Base::vectors[new_index]->setMemoryContext("scratch", name, MemoryRole::ScratchTemporary);
        Base::vectors[new_index]->resize(size);
        return Base::vectors[new_index]->data();
    }


    void resize(const std::string& name, size_t new_size) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end()) {
            DEME_ERROR(std::string("Cannot resize: name not found"));
        }
        this->vectors[it->second]->resize(new_size);
    }

    T* get(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            DEME_ERROR("Name not found: %s", name.c_str());
        return this->vectors[it->second]->data();
    }

    void setMemoryCounter(size_t* counter) {
        m_mem_counter = counter;
        for (auto& vec : this->vectors) {
            vec->setMemoryCounter(counter);
        }
    }

  private:
    size_t* m_mem_counter = nullptr;
};

/// @brief A pool that dispatches dual arrays
/// @tparam T Array data type
template <typename T>
class DualArrayPool : public ResourcePool<T, DualArray<T>> {
  public:
    explicit DualArrayPool(size_t* host_external_counter = nullptr, size_t* device_external_counter = nullptr)
        : m_host_mem_counter(host_external_counter), m_device_mem_counter(device_external_counter) {}

    DualArray<T>* claim(const std::string& name, size_t size, bool allow_duplicate = false) {
        // Referring to the base class that correspond to me
        using Base = ResourcePool<T, DualArray<T>>;

        if (Base::name_to_index.count(name)) {
            if (allow_duplicate) {
                size_t index = Base::name_to_index[name];
                Base::vectors[index]->setMemoryContext("scratch", name, MemoryRole::ScratchTemporary);
                Base::vectors[index]->resize(size);
                return Base::vectors[index].get();
            } else {
                DEME_ERROR("Name already claimed: %s", name.c_str());
            }
        }

        size_t chosen = static_cast<size_t>(-1);
        size_t chosen_cap = static_cast<size_t>(-1);
        size_t fallback = static_cast<size_t>(-1);
        size_t fallback_cap = static_cast<size_t>(-1);
        for (size_t i = 0; i < Base::in_use.size(); ++i) {
            if (Base::in_use[i])
                continue;
            const size_t cap = Base::vectors[i]->size();
            if (cap >= size) {
                if (cap < chosen_cap) {
                    chosen = i;
                    chosen_cap = cap;
                }
            } else if (cap < fallback_cap) {
                fallback = i;
                fallback_cap = cap;
            }
        }
        if (chosen == static_cast<size_t>(-1))
            chosen = fallback;
        if (chosen != static_cast<size_t>(-1)) {
            Base::vectors[chosen]->setMemoryContext("scratch", name, MemoryRole::ScratchTemporary);
            Base::vectors[chosen]->resize(size);
            Base::in_use[chosen] = name;
            Base::name_to_index[name] = chosen;
            return Base::vectors[chosen].get();
        }

        Base::vectors.emplace_back(std::make_unique<DualArray<T>>(m_host_mem_counter, m_device_mem_counter));
        Base::in_use.emplace_back(name);
        size_t new_index = Base::vectors.size() - 1;
        Base::name_to_index[name] = new_index;
        Base::vectors[new_index]->setMemoryContext("scratch", name, MemoryRole::ScratchTemporary);
        Base::vectors[new_index]->resize(size);
        return Base::vectors[new_index].get();
    }

    DualArray<T>* get(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            DEME_ERROR("Name not found: %s", name.c_str());
        return this->vectors[it->second].get();
    }

    T* getHost(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            DEME_ERROR("Name not found: %s", name.c_str());
        return this->vectors[it->second]->host();
    }

    T* getDevice(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            DEME_ERROR("Name not found: %s", name.c_str());
        return this->vectors[it->second]->device();
    }

    void setMemoryCounter(size_t* host_external_counter, size_t* device_external_counter) {
        m_host_mem_counter = host_external_counter;
        m_device_mem_counter = device_external_counter;
        for (auto& vec : this->vectors) {
            vec->setHostMemoryCounter(m_host_mem_counter);
            vec->setDeviceMemoryCounter(m_device_mem_counter);
        }
    }

  private:
    size_t* m_host_mem_counter = nullptr;
    size_t* m_device_mem_counter = nullptr;
};

/// @brief A pool that dispatches dual structs
/// @tparam T Struct data type
template <typename T>
class DualStructPool : public ResourcePool<T, DualStruct<T>> {
  public:
    explicit DualStructPool() {}

    DualStruct<T>* claim(const std::string& name, bool allow_duplicate = false) {
        // Referring to the base class that correspond to me
        using Base = ResourcePool<T, DualStruct<T>>;

        if (Base::name_to_index.count(name)) {
            if (allow_duplicate) {
                size_t index = Base::name_to_index[name];
                return Base::vectors[index].get();
            } else {
                DEME_ERROR("Name already claimed: %s", name.c_str());
            }
        }

        for (size_t i = 0; i < Base::in_use.size(); ++i) {
            if (!Base::in_use[i]) {
                Base::in_use[i] = name;
                Base::name_to_index[name] = i;
                return Base::vectors[i].get();
            }
        }

        Base::vectors.emplace_back(std::make_unique<DualStruct<T>>());
        Base::in_use.emplace_back(name);
        size_t new_index = Base::vectors.size() - 1;
        Base::name_to_index[name] = new_index;
        return Base::vectors[new_index].get();
    }

    DualStruct<T>* get(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            DEME_ERROR("Name not found: %s", name.c_str());
        return this->vectors[it->second].get();
    }

    T* getHost(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            DEME_ERROR("Name not found: %s", name.c_str());
        return this->vectors[it->second]->getHostPointer();
    }

    T* getDevice(const std::string& name) {
        auto it = this->name_to_index.find(name);
        if (it == this->name_to_index.end())
            DEME_ERROR("Name not found: %s", name.c_str());
        return this->vectors[it->second]->getDevicePointer();
    }
};

// A conceptual note: In DEME, DualStruct-managed simParams, granData etc. are in general considered "host-major",
// meaning that we generally believe the data on the host is more fresh. So, we change host data freely, copy host data
// to device more freely, but generally do not copy data from device to host. In contrast, DualArray-managed simulation
// status or work arrays are "device-major", meaning the device copy is believed to be more fresh. So, we copy from
// device to host more freely and concern-free, but when copying from host to device, that's either in a centrialized
// initialization stage, or we do piecemeal and fine-grain copying which reflects the user's forced system updates only.

namespace xfer {  // Memory Transfer Bundle Manager

enum class XferMode { Same, Peer, Stage };

struct HostBounce {
    void* ptr = nullptr;
    size_t cap = 0;
    ~HostBounce() {
        if (ptr)
            cudaFreeHost(ptr);
    }
    inline void ensure(size_t need) {
        if (cap >= need)
            return;
        if (ptr)
            cudaFreeHost(ptr);
        DEME_GPU_CALL(cudaMallocHost(&ptr, need));  // pinned
        cap = need;
    }
};

struct PeerCache {
    int dst{-1}, src{-1};
    int can{-1};
};  // -1 unknown, 0 no, 1 yes
static thread_local HostBounce __bounce;
static thread_local PeerCache __pc;

inline XferMode plan_mode(int dstDev, int srcDev) {
    if (dstDev == srcDev)
        return XferMode::Same;

    if (__pc.dst != dstDev || __pc.src != srcDev || __pc.can < 0) {
        __pc.dst = dstDev;
        __pc.src = srcDev;
        __pc.can = 0;
        int can = 0;
        DEME_GPU_CALL(cudaDeviceCanAccessPeer(&can, dstDev, srcDev));
        if (can) {
            int cur = -1;
            DEME_GPU_CALL(cudaGetDevice(&cur));
            if (cur != dstDev)
                DEME_GPU_CALL(cudaSetDevice(dstDev));
            cudaError_t st = cudaDeviceEnablePeerAccess(srcDev, 0);
            if (st == cudaErrorPeerAccessAlreadyEnabled)
                (void)cudaGetLastError();
            else if (st != cudaSuccess)
                can = 0;
            if (cur != dstDev)
                DEME_GPU_CALL(cudaSetDevice(cur));
        }
        __pc.can = can ? 1 : 0;
    }
    return __pc.can ? XferMode::Peer : XferMode::Stage;
}

inline size_t __sum_bytes(const size_t* bytes, int n) {
    size_t t = 0;
    for (int i = 0; i < n; ++i)
        t += bytes[i];
    return t;
}
inline size_t __max_bytes(const size_t* bytes, int n) {
    size_t m = 0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, bytes[i]);
    return m;
}

// Same device: D2D
inline void xfer_same(int dev,
                      void* const* dst,
                      const void* const* src,
                      const size_t* bytes,
                      int n,
                      cudaStream_t stream) {
    int cur = -1;
    DEME_GPU_CALL(cudaGetDevice(&cur));
    int dev_sw = (cur != dev) ? 1 : 0;
    if (dev_sw)
        DEME_GPU_CALL(cudaSetDevice(dev));

    for (int i = 0; i < n; ++i)
        if (bytes[i]) {
            DEME_GPU_CALL(cudaMemcpyAsync(dst[i], src[i], bytes[i], cudaMemcpyDeviceToDevice, stream));
        }
    if (dev_sw)
        DEME_GPU_CALL(cudaSetDevice(cur));
}

// Peer: cudaMemcpyPeer
inline void xfer_peer(int dstDev,
                      int srcDev,
                      void* const* dst,
                      const void* const* src,
                      const size_t* bytes,
                      int n,
                      cudaStream_t stream) {
    int cur = -1;
    DEME_GPU_CALL(cudaGetDevice(&cur));
    int dev_sw = (cur != dstDev) ? 1 : 0;
    if (dev_sw)
        DEME_GPU_CALL(cudaSetDevice(dstDev));
    for (int i = 0; i < n; ++i)
        if (bytes[i]) {
            DEME_GPU_CALL(cudaMemcpyPeerAsync(dst[i], dstDev, src[i], srcDev, bytes[i], stream));
        }
    if (dev_sw)
        DEME_GPU_CALL(cudaSetDevice(cur));
}

// Stage== No Peer
inline void xfer_stage_as_d2d(int dstDev,
                              int srcDev,
                              void* const* dst,
                              const void* const* src,
                              const size_t* bytes,
                              int n,
                              cudaStream_t /*stream*/) {
    for (int i = 0; i < n; ++i)
        if (bytes[i]) {
            DEME_GPU_CALL(cudaMemcpy(dst[i], src[i], bytes[i], cudaMemcpyDeviceToDevice));
        }
}

// Frontend
inline void D2D_bundle(int dstDev,
                       int srcDev,
                       void* const* dst,
                       const void* const* src,
                       const size_t* bytes,
                       int n,
                       cudaStream_t stream = 0) {
    if (n <= 0)
        return;
    XferMode mode = plan_mode(dstDev, srcDev);

    switch (mode) {
        case XferMode::Same:
            xfer_same(srcDev, dst, src, bytes, n, stream);
            break;
        case XferMode::Peer:
            xfer_peer(dstDev, srcDev, dst, src, bytes, n, stream);
            break;
        case XferMode::Stage:
            xfer_stage_as_d2d(dstDev, srcDev, dst, src, bytes, n, stream);
            break;
    }
}

// Builder
struct XferList {
    static constexpr int MAX = 16;
    void* dst[MAX]{};
    const void* src[MAX]{};
    size_t sz[MAX]{};
    int n{0};
    inline void add(void* d, const void* s, size_t b) {
        if (b && n < MAX) {
            dst[n] = d;
            src[n] = s;
            sz[n] = b;
            ++n;
        }
    }
    inline void run(int dstDev, int srcDev, cudaStream_t stream = 0) {
        D2D_bundle(dstDev, srcDev, dst, src, sz, n, stream);
    }
};

}  // namespace xfer

inline size_t GetTrackedProgramDevicePeakMemoryUsage() {
    return detail::get_tracked_program_device_peak_memory_usage();
}

inline void ResetTrackedProgramDevicePeakMemoryUsage() {
    detail::reset_tracked_program_device_peak_memory_usage();
}

}  // namespace deme

#endif
