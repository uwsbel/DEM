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
#include <cstring>
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

#include <cuda_runtime_api.h>
#if __has_include(<cuda.h>)
#include <cuda.h>
#define DEME_HAS_CUDA_DRIVER_VMM 1
#else
#define DEME_HAS_CUDA_DRIVER_VMM 0
#endif

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

inline bool vmm_temp_arena_requested() {
    // Default-on safe VMM path: only temp scratch vectors, no per-array VMM, no page decommit while kernels run.
    // It remains fully opt-out and falls back to the legacy pool automatically when CUDA VMM is unavailable.
    if (env_truthy(std::getenv("DEME_DISABLE_VMM_TEMP_ARENA")))
        return false;
    const char* arena = std::getenv("DEME_VMM_TEMP_ARENA");
    if (arena && *arena && !env_truthy(arena))
        return false;
    const char* vmm = std::getenv("DEME_VMM");
    if (vmm && *vmm && !env_truthy(vmm))
        return false;
    const char* scratch = std::getenv("DEME_VMM_SCRATCH");
    if (scratch && *scratch && !env_truthy(scratch))
        return false;
    return true;
}

inline size_t parse_env_mb_to_bytes(const char* env, double fallback_mb) {
    double mb = fallback_mb;
    if (env && *env) {
        char* end = nullptr;
        const double parsed = std::strtod(env, &end);
        if (end != env && parsed > 0.0)
            mb = parsed;
    }
    return static_cast<size_t>(mb * 1024.0 * 1024.0);
}

inline size_t parse_optional_env_mb_to_bytes(const char* env) {
    if (!env || !*env)
        return 0;
    char* end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || parsed <= 0.0)
        return 0;
    return static_cast<size_t>(parsed * 1024.0 * 1024.0);
}

inline size_t vmm_temp_arena_explicit_first_segment_bytes() {
    // Backward compatibility with earlier patches. This is only a virtual-reserve hint for the first segment; physical
    // pages are committed lazily. Leaving it unset enables the adaptive segmented policy below.
    size_t bytes = parse_optional_env_mb_to_bytes(std::getenv("DEME_VMM_TEMP_ARENA_RESERVE_MB"));
    if (!bytes)
        bytes = parse_optional_env_mb_to_bytes(std::getenv("DEME_VMM_SCRATCH_RESERVE_MB"));
    return bytes;
}

inline size_t vmm_temp_arena_min_segment_bytes() {
    // Small first segments avoid the v4 "always reserve 512 MiB" behavior while still amortizing CUDA VMM calls.
    return parse_env_mb_to_bytes(std::getenv("DEME_VMM_TEMP_ARENA_SEGMENT_MIN_MB"), 64.0);
}

inline size_t vmm_temp_arena_max_segment_bytes() {
    // A 128 MiB segment cap is large enough for DEME's current scratch bursts but prevents a single bad request from
    // reserving a huge virtual window. Larger scenes can raise this explicitly.
    return parse_env_mb_to_bytes(std::getenv("DEME_VMM_TEMP_ARENA_SEGMENT_MAX_MB"), 128.0);
}

inline size_t vmm_temp_arena_user_budget_bytes() {
    // Preferred explicit budget knob. Older HARD_LIMIT/SCRATCH_HARD_LIMIT names remain accepted.
    size_t bytes = parse_optional_env_mb_to_bytes(std::getenv("DEME_VMM_TEMP_ARENA_BUDGET_MB"));
    if (!bytes)
        bytes = parse_optional_env_mb_to_bytes(std::getenv("DEME_VMM_TEMP_ARENA_HARD_LIMIT_MB"));
    if (!bytes)
        bytes = parse_optional_env_mb_to_bytes(std::getenv("DEME_VMM_SCRATCH_HARD_LIMIT_MB"));
    return bytes;
}

inline double vmm_temp_arena_budget_fraction() {
    const char* env = std::getenv("DEME_VMM_TEMP_ARENA_BUDGET_FRACTION");
    if (!env || !*env)
        return 1.0 / 64.0;  // about 1.56% of currently free VRAM, clamped below
    char* end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || parsed <= 0.0)
        return 1.0 / 64.0;
    return parsed > 0.25 ? 0.25 : parsed;
}

inline size_t vmm_temp_arena_min_budget_bytes() {
    return parse_env_mb_to_bytes(std::getenv("DEME_VMM_TEMP_ARENA_MIN_BUDGET_MB"), 192.0);
}

inline size_t vmm_temp_arena_max_budget_bytes() {
    return parse_env_mb_to_bytes(std::getenv("DEME_VMM_TEMP_ARENA_MAX_BUDGET_MB"), 512.0);
}

inline double vmm_temp_arena_growth_factor() {
    const char* env = std::getenv("DEME_VMM_TEMP_ARENA_GROWTH");
    if (!env || !*env)
        return 1.5;
    char* end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || parsed < 1.0)
        return 1.5;
    return parsed > 3.0 ? 3.0 : parsed;
}

inline bool vmm_temp_arena_trace_enabled() {
    return env_truthy(std::getenv("DEME_VMM_TRACE")) || env_truthy(std::getenv("DEME_VMM_TEMP_ARENA_TRACE"));
}

inline bool vmm_temp_arena_validate_enabled() {
    // Tracing must stay cheap.  v3 accidentally made DEME_VMM_TRACE imply O(N^2) overlap validation on every
    // scratch claim/free, which made startup and early frames much slower.  Keep heavy validation explicit.
    return env_truthy(std::getenv("DEME_VMM_TEMP_ARENA_VALIDATE"));
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

inline void update_tracked_device_allocation(const void* ptr, size_t bytes) {
    if (!ptr)
        return;
    auto& tracker = tracked_device_mem();
    size_t old_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(tracker.mutex);
        auto it = tracker.allocations.find(ptr);
        if (it != tracker.allocations.end())
            old_bytes = it->second;
        if (bytes)
            tracker.allocations[ptr] = bytes;
        else if (it != tracker.allocations.end())
            tracker.allocations.erase(it);
    }
    if (bytes > old_bytes) {
        const size_t delta = bytes - old_bytes;
        const size_t live_now = tracker.live_bytes.fetch_add(delta, std::memory_order_relaxed) + delta;
        update_atomic_max(tracker.peak_bytes, live_now);
    } else if (old_bytes > bytes) {
        tracker.live_bytes.fetch_sub(old_bytes - bytes, std::memory_order_relaxed);
    }
}

inline size_t get_tracked_program_device_peak_memory_usage() {
    return tracked_device_mem().peak_bytes.load(std::memory_order_relaxed);
}

inline void reset_tracked_program_device_peak_memory_usage() {
    auto& tracker = tracked_device_mem();
    tracker.peak_bytes.store(tracker.live_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

#if DEME_HAS_CUDA_DRIVER_VMM
inline std::string cuda_driver_error_string(CUresult result) {
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    std::ostringstream os;
    os << (name ? name : "CUresult") << ": " << (text ? text : "unknown CUDA driver error");
    return os.str();
}
#endif

class VmmTempScratchArena : private NonCopyable {
  public:
    explicit VmmTempScratchArena(size_t* external_counter = nullptr) : m_mem_counter(external_counter) {}
    ~VmmTempScratchArena() { releaseAll(); }

    void setMemoryCounter(size_t* counter) { m_mem_counter = counter; }

    bool requested() const { return vmm_temp_arena_requested(); }
    bool enabled() {
        if (!requested())
            return false;
        initializeIfNeeded();
        return m_enabled;
    }

    void* allocate(const std::string& name, size_t bytes) {
        if (!bytes)
            bytes = 1;
        if (!enabled())
            return nullptr;
        if (m_name_to_block.count(name))
            DEME_ERROR("Scratch temp arena name already claimed: %s", name.c_str());

        const size_t aligned = alignUp(bytes, kAlignment);
        size_t chosen = static_cast<size_t>(-1);
        size_t chosen_size = static_cast<size_t>(-1);
        for (size_t i = 0; i < m_blocks.size(); ++i) {
            const Block& b = m_blocks[i];
            if (!b.free || b.size < aligned)
                continue;
            if (b.size < chosen_size) {
                chosen = i;
                chosen_size = b.size;
            }
        }

        if (chosen == static_cast<size_t>(-1)) {
            chosen = allocateFromUnbumpedSpace(name, bytes, aligned);
            if (chosen == static_cast<size_t>(-1))
                return nullptr;
        } else {
            // IMPORTANT: do not keep a Block& across push_back.  push_back may reallocate m_blocks,
            // and an invalidated reference here can leave the chosen block marked free while its
            // address is returned to a live kernel.  That produces silent slice aliasing and quickly
            // corrupts physics state.
            const size_t block_segment = m_blocks[chosen].segment;
            const size_t block_offset = m_blocks[chosen].offset;
            const size_t block_size = m_blocks[chosen].size;
            if (block_size >= aligned + kMinSplitBytes) {
                m_blocks[chosen].size = aligned;
                m_blocks.push_back(Block{block_segment, block_offset + aligned, block_size - aligned, 0, true,
                                         std::string()});
            }
            Block& b = m_blocks[chosen];
            b.requested = bytes;
            b.free = false;
            b.name = name;
        }

        m_name_to_block[name] = chosen;
        validateClaimedBlock(name, chosen, aligned);
        if (vmm_temp_arena_validate_enabled())
            validateNoLiveOverlap("arenaClaim");
        recordBlock(chosen, "arenaClaim");
        recordArena("arenaClaim");
        return ptrForBlock(m_blocks[chosen]);
    }

    bool owns(const std::string& name) const { return m_name_to_block.count(name) != 0; }

    void freeByName(const std::string& name) {
        auto it = m_name_to_block.find(name);
        if (it == m_name_to_block.end())
            return;
        const size_t idx = it->second;
        if (idx < m_blocks.size()) {
            Block& b = m_blocks[idx];
            b.free = true;
            b.name.clear();
            b.requested = 0;
            recordBlock(idx, "arenaFree");
        }
        m_name_to_block.erase(it);
        coalesceFreeBlocks();
        if (vmm_temp_arena_validate_enabled())
            validateNoLiveOverlap("arenaFree");
        recordArena("arenaFree");
    }

    void releaseAll() {
#if DEME_HAS_CUDA_DRIVER_VMM
        for (auto& seg : m_segments) {
            for (auto& h : seg.handles) {
                if (h.handle) {
                    cuMemUnmap(seg.base + h.offset, h.bytes);
                    cuMemRelease(h.handle);
                }
            }
            if (seg.base)
                cuMemAddressFree(seg.base, seg.reserved);
            update_tracked_device_allocation(reinterpret_cast<const void*>(seg.base), 0);
        }
#endif
        if (m_mem_counter && m_total_committed_bytes)
            *m_mem_counter -= std::min(*m_mem_counter, m_total_committed_bytes);
        eraseArenaAndBlocks();
        m_segments.clear();
        m_blocks.clear();
        m_name_to_block.clear();
        m_total_reserved_bytes = 0;
        m_total_committed_bytes = 0;
        m_peak_committed_bytes = 0;
        m_next_segment_reserve_bytes = 0;
        m_failed_allocations = 0;
        m_budget_bytes = static_cast<size_t>(-1);
        m_enabled = false;
        m_initialized = false;
    }

    size_t committedBytes() const { return m_total_committed_bytes; }
    size_t reservedBytes() const { return m_total_reserved_bytes; }
    size_t peakCommittedBytes() const { return m_peak_committed_bytes; }
    size_t failedAllocations() const { return m_failed_allocations; }
    size_t activeBytes() const {
        size_t out = 0;
        for (const auto& b : m_blocks)
            if (!b.free)
                out += b.requested;
        return out;
    }

    void printStatus() const {
        if (!m_enabled)
            return;
        std::cout << "VMM temp arena: segments=" << m_segments.size() << " reserved=" << m_total_reserved_bytes
                  << " committed=" << m_total_committed_bytes << " peak=" << m_peak_committed_bytes
                  << " budget=" << m_budget_bytes << " active=" << activeBytes()
                  << " failed=" << m_failed_allocations << "\n";
    }

  private:
    static constexpr size_t kAlignment = 256;
    static constexpr size_t kMinSplitBytes = 4096;

    struct Block {
        size_t segment = 0;
        size_t offset = 0;
        size_t size = 0;
        size_t requested = 0;
        bool free = true;
        std::string name;
    };

#if DEME_HAS_CUDA_DRIVER_VMM
    struct HandleRange {
        size_t offset = 0;
        size_t bytes = 0;
        CUmemGenericAllocationHandle handle = 0;
    };
    struct Segment {
        CUdeviceptr base = 0;
        size_t reserved = 0;
        size_t committed = 0;
        size_t bump = 0;
        std::vector<HandleRange> handles;
    };
#else
    struct Segment {
        uintptr_t base = 0;
        size_t reserved = 0;
        size_t committed = 0;
        size_t bump = 0;
    };
#endif

    size_t* m_mem_counter = nullptr;
    bool m_initialized = false;
    bool m_enabled = false;
    int m_device = 0;
    size_t m_granularity = 0;
    size_t m_total_reserved_bytes = 0;
    size_t m_total_committed_bytes = 0;
    size_t m_peak_committed_bytes = 0;
    size_t m_next_segment_reserve_bytes = 0;
    size_t m_failed_allocations = 0;
    size_t m_budget_bytes = static_cast<size_t>(-1);
    std::vector<Segment> m_segments;
    std::vector<Block> m_blocks;
    std::unordered_map<std::string, size_t> m_name_to_block;

    static size_t alignUp(size_t value, size_t alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    static size_t nextPow2(size_t value) {
        if (value <= 1)
            return 1;
        --value;
        for (size_t shift = 1; shift < sizeof(size_t) * 8; shift <<= 1)
            value |= value >> shift;
        return value + 1;
    }

    static size_t safeScale(size_t value, double factor) {
        if (factor <= 1.0)
            return value;
        const double scaled = static_cast<double>(value) * factor;
        const double max_size = static_cast<double>(static_cast<size_t>(-1));
        if (scaled >= max_size)
            return static_cast<size_t>(-1);
        return static_cast<size_t>(scaled);
    }

    void initializeIfNeeded() {
        if (m_initialized)
            return;
        m_initialized = true;
#if DEME_HAS_CUDA_DRIVER_VMM
        cudaError_t rt = cudaFree(nullptr);  // creates the primary context without allocating
        if (rt != cudaSuccess)
            return;
        rt = cudaGetDevice(&m_device);
        if (rt != cudaSuccess)
            return;
        CUresult cr = cuInit(0);
        if (cr != CUDA_SUCCESS)
            return;
        CUdevice cu_device;
        cr = cuDeviceGet(&cu_device, m_device);
        if (cr != CUDA_SUCCESS)
            return;

        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = m_device;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

        size_t gran = 0;
        cr = cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (cr != CUDA_SUCCESS || gran == 0)
            return;
        m_granularity = gran;

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        rt = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (rt != cudaSuccess || free_bytes == 0)
            return;
        m_budget_bytes = computeAdaptiveBudget(free_bytes, total_bytes);
        if (m_budget_bytes < m_granularity)
            return;

        m_enabled = true;
        if (vmm_temp_arena_trace_enabled()) {
            std::cerr << "DEME VMM temp scratch arena available on device " << m_device
                      << ": commit granularity " << (double)m_granularity / (1024.0 * 1024.0)
                      << " MiB, adaptive physical budget " << (double)m_budget_bytes / (1024.0 * 1024.0)
                      << " MiB, segmented virtual reserve on demand\n";
        }
#endif
    }

    size_t computeAdaptiveBudget(size_t free_bytes, size_t total_bytes) const {
        const size_t user_budget = vmm_temp_arena_user_budget_bytes();
        if (user_budget)
            return alignUp(user_budget, m_granularity);

        const double fraction = vmm_temp_arena_budget_fraction();
        size_t budget = static_cast<size_t>(static_cast<double>(free_bytes) * fraction);
        const size_t min_budget = vmm_temp_arena_min_budget_bytes();
        const size_t max_budget = vmm_temp_arena_max_budget_bytes();
        budget = std::max(budget, min_budget);
        budget = std::min(budget, max_budget);

        // Never let the automatic scratch arena take most of a small GPU.  This is a safety budget, not a target.
        budget = std::min(budget, free_bytes / 4);
        budget = std::min(budget, total_bytes / 8);
        return alignUp(budget, m_granularity);
    }

    size_t remainingBudgetBytes() const {
        if (m_budget_bytes == static_cast<size_t>(-1))
            return static_cast<size_t>(-1);
        return (m_total_committed_bytes < m_budget_bytes) ? (m_budget_bytes - m_total_committed_bytes) : 0;
    }

    size_t computeSegmentReserve(size_t required_bytes) const {
        const size_t min_seg = alignUp(std::max(vmm_temp_arena_min_segment_bytes(), m_granularity), m_granularity);
        const size_t max_cfg = std::max(vmm_temp_arena_max_segment_bytes(), min_seg);
        const size_t max_seg = alignUp(max_cfg, m_granularity);
        const size_t explicit_first = vmm_temp_arena_explicit_first_segment_bytes();

        size_t wanted = required_bytes;
        if (m_segments.empty() && explicit_first)
            wanted = std::max(wanted, explicit_first);
        else
            wanted = std::max(wanted, safeScale(required_bytes, vmm_temp_arena_growth_factor()));
        wanted = std::max(wanted, min_seg);
        wanted = nextPow2(wanted);
        wanted = alignUp(wanted, m_granularity);
        wanted = std::min(wanted, max_seg);
        if (wanted < required_bytes)
            wanted = alignUp(required_bytes, m_granularity);

        // Virtual reservations are cheap, but bounding them by the physical budget prevents v3-style runaway growth
        // and makes reserve size automatic instead of a fixed 512 MiB knob.  Requests larger than the budget fall
        // back to the legacy pool instead of risking OOM.
        if (m_budget_bytes != static_cast<size_t>(-1)) {
            const size_t reserved_left = (m_total_reserved_bytes < m_budget_bytes) ? (m_budget_bytes - m_total_reserved_bytes) : 0;
            if (reserved_left < required_bytes)
                return 0;
            wanted = std::min(wanted, reserved_left);
            wanted = alignUp(wanted, m_granularity);
        }
        return wanted;
    }

    void updateNextSegmentReserve(size_t just_reserved) {
        const size_t min_seg = alignUp(std::max(vmm_temp_arena_min_segment_bytes(), m_granularity), m_granularity);
        const size_t max_seg = alignUp(std::max(vmm_temp_arena_max_segment_bytes(), min_seg), m_granularity);
        size_t next = safeScale(just_reserved, vmm_temp_arena_growth_factor());
        next = std::max(next, min_seg);
        if (next > max_seg)
            next = max_seg;
        m_next_segment_reserve_bytes = alignUp(next, m_granularity);
    }

    bool addSegment(size_t required_bytes) {
#if DEME_HAS_CUDA_DRIVER_VMM
        const size_t reserve = computeSegmentReserve(required_bytes);
        if (!reserve)
            return false;
        CUdeviceptr base = 0;
        CUresult cr = cuMemAddressReserve(&base, reserve, 0, 0, 0);
        if (cr != CUDA_SUCCESS) {
            if (vmm_temp_arena_trace_enabled())
                std::cerr << "DEME VMM temp arena cuMemAddressReserve failed for " << reserve
                          << " bytes: " << cuda_driver_error_string(cr) << "\n";
            return false;
        }
        Segment seg;
        seg.base = base;
        seg.reserved = reserve;
        m_segments.push_back(std::move(seg));
        m_total_reserved_bytes += reserve;
        updateNextSegmentReserve(reserve);
        if (vmm_temp_arena_trace_enabled()) {
            std::cerr << "DEME VMM temp arena added segment " << (m_segments.size() - 1)
                      << ": reserved " << (double)reserve / (1024.0 * 1024.0)
                      << " MiB virtual, total reserved " << (double)m_total_reserved_bytes / (1024.0 * 1024.0)
                      << " MiB\n";
        }
        recordArena("arenaReserveSegment");
        return true;
#else
        (void)required_bytes;
        return false;
#endif
    }

    bool ensureCommitted(size_t segment_idx, size_t required_bytes) {
#if DEME_HAS_CUDA_DRIVER_VMM
        if (segment_idx >= m_segments.size())
            return false;
        Segment& seg = m_segments[segment_idx];
        required_bytes = alignUp(required_bytes, m_granularity);
        if (required_bytes <= seg.committed)
            return true;
        const size_t old_committed = seg.committed;
        const size_t grow = required_bytes - old_committed;
        if (m_budget_bytes != static_cast<size_t>(-1) && m_total_committed_bytes + grow > m_budget_bytes) {
            if (vmm_temp_arena_trace_enabled())
                std::cerr << "DEME VMM temp arena adaptive budget would be exceeded by " << grow
                          << " bytes; falling back to legacy scratch for this allocation\n";
            return false;
        }

        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = m_device;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

        CUmemGenericAllocationHandle handle = 0;
        CUresult cr = cuMemCreate(&handle, grow, &prop, 0);
        if (cr != CUDA_SUCCESS) {
            if (vmm_temp_arena_trace_enabled())
                std::cerr << "DEME VMM temp arena cuMemCreate failed for " << grow << " bytes: "
                          << cuda_driver_error_string(cr) << "\n";
            return false;
        }
        cr = cuMemMap(seg.base + old_committed, grow, 0, handle, 0);
        if (cr != CUDA_SUCCESS) {
            cuMemRelease(handle);
            if (vmm_temp_arena_trace_enabled())
                std::cerr << "DEME VMM temp arena cuMemMap failed: " << cuda_driver_error_string(cr) << "\n";
            return false;
        }
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = m_device;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cr = cuMemSetAccess(seg.base + old_committed, grow, &access, 1);
        if (cr != CUDA_SUCCESS) {
            cuMemUnmap(seg.base + old_committed, grow);
            cuMemRelease(handle);
            if (vmm_temp_arena_trace_enabled())
                std::cerr << "DEME VMM temp arena cuMemSetAccess failed: " << cuda_driver_error_string(cr) << "\n";
            return false;
        }
        seg.handles.push_back(HandleRange{old_committed, grow, handle});
        seg.committed = required_bytes;
        m_total_committed_bytes += grow;
        m_peak_committed_bytes = std::max(m_peak_committed_bytes, m_total_committed_bytes);
        update_tracked_device_allocation(reinterpret_cast<const void*>(seg.base), seg.committed);
        if (m_mem_counter)
            *m_mem_counter += grow;
        recordArena("arenaCommit");
        return true;
#else
        (void)segment_idx;
        (void)required_bytes;
        return false;
#endif
    }

    size_t allocateFromUnbumpedSpace(const std::string& name, size_t requested, size_t aligned) {
        for (size_t seg_idx = 0; seg_idx < m_segments.size(); ++seg_idx) {
            Segment& seg = m_segments[seg_idx];
            const size_t off = alignUp(seg.bump, kAlignment);
            if (off + aligned > seg.reserved)
                continue;
            if (!ensureCommitted(seg_idx, off + aligned)) {
                ++m_failed_allocations;
                return static_cast<size_t>(-1);
            }
            const size_t idx = m_blocks.size();
            m_blocks.push_back(Block{seg_idx, off, aligned, requested, false, name});
            seg.bump = off + aligned;
            return idx;
        }

        if (!addSegment(aligned)) {
            ++m_failed_allocations;
            return static_cast<size_t>(-1);
        }
        const size_t seg_idx = m_segments.size() - 1;
        Segment& seg = m_segments[seg_idx];
        const size_t off = alignUp(seg.bump, kAlignment);
        if (off + aligned > seg.reserved || !ensureCommitted(seg_idx, off + aligned)) {
            ++m_failed_allocations;
            return static_cast<size_t>(-1);
        }
        const size_t idx = m_blocks.size();
        m_blocks.push_back(Block{seg_idx, off, aligned, requested, false, name});
        seg.bump = off + aligned;
        return idx;
    }

    void* ptrForBlock(const Block& b) const {
        if (b.segment >= m_segments.size())
            return nullptr;
#if DEME_HAS_CUDA_DRIVER_VMM
        return reinterpret_cast<void*>(m_segments[b.segment].base + b.offset);
#else
        return reinterpret_cast<void*>(m_segments[b.segment].base + b.offset);
#endif
    }

    const void* firstSegmentPtr() const {
        if (m_segments.empty())
            return nullptr;
        return reinterpret_cast<const void*>(m_segments.front().base);
    }

    void coalesceFreeBlocks() {
        std::sort(m_blocks.begin(), m_blocks.end(), [](const Block& a, const Block& b) {
            if (a.segment != b.segment)
                return a.segment < b.segment;
            return a.offset < b.offset;
        });
        std::vector<Block> merged;
        merged.reserve(m_blocks.size());
        for (const auto& b : m_blocks) {
            if (!merged.empty() && merged.back().segment == b.segment && merged.back().free && b.free &&
                merged.back().offset + merged.back().size == b.offset) {
                merged.back().size += b.size;
            } else {
                merged.push_back(b);
            }
        }
        m_blocks.swap(merged);
        m_name_to_block.clear();
        for (size_t i = 0; i < m_blocks.size(); ++i)
            if (!m_blocks[i].free)
                m_name_to_block[m_blocks[i].name] = i;
    }

    void validateClaimedBlock(const std::string& name, size_t idx, size_t min_size) const {
        if (idx >= m_blocks.size())
            DEME_ERROR("VMM temp arena internal error: block index out of range for %s", name.c_str());
        const Block& b = m_blocks[idx];
        if (b.segment >= m_segments.size() || b.free || b.name != name || b.size < min_size)
            DEME_ERROR("VMM temp arena internal error: inconsistent live block for %s", name.c_str());
    }

    void validateNoLiveOverlap(const char* where) const {
        for (const auto& kv : m_name_to_block) {
            if (kv.second >= m_blocks.size())
                DEME_ERROR("VMM temp arena map corruption at %s: index out of range for %s", where, kv.first.c_str());
            const Block& b = m_blocks[kv.second];
            if (b.free || b.name != kv.first)
                DEME_ERROR("VMM temp arena map corruption at %s: stale/free entry for %s", where, kv.first.c_str());
        }
        for (size_t i = 0; i < m_blocks.size(); ++i) {
            if (m_blocks[i].free)
                continue;
            const size_t a0 = m_blocks[i].offset;
            const size_t a1 = m_blocks[i].offset + m_blocks[i].size;
            for (size_t j = i + 1; j < m_blocks.size(); ++j) {
                if (m_blocks[j].free || m_blocks[i].segment != m_blocks[j].segment)
                    continue;
                const size_t b0 = m_blocks[j].offset;
                const size_t b1 = m_blocks[j].offset + m_blocks[j].size;
                if (a0 < b1 && b0 < a1)
                    DEME_ERROR("VMM temp arena live slice overlap at %s: %s overlaps %s", where,
                               m_blocks[i].name.c_str(), m_blocks[j].name.c_str());
            }
        }
    }

    void recordArena(const std::string& phase) const {
        if (!mem_trace_enabled() || (!firstSegmentPtr() && !m_total_committed_bytes))
            return;
        MemoryLedgerRecord rec;
        rec.array_object = this;
        rec.owner = "scratch";
        rec.name = "VmmTempArena";
        rec.role = "ScratchTemporary";
        rec.type_name = "byte-arena";
        rec.type_size = 1;
        rec.logical_bytes = activeBytes();
        rec.capacity_bytes = usedVirtualBytes();
        rec.committed_bytes = m_total_committed_bytes;
        rec.host_bytes = 0;
        rec.high_water_capacity_bytes = m_peak_committed_bytes;
        rec.device_ptr = firstSegmentPtr();
        rec.is_view = false;
        rec.has_host_mirror = false;
        rec.backend = "vmm-temp-arena-adaptive";
        rec.phase = phase;
        upsert_memory_ledger_record(rec);
    }

    size_t usedVirtualBytes() const {
        size_t out = 0;
        for (const auto& seg : m_segments)
            out += seg.bump;
        return out;
    }

    void recordBlock(size_t idx, const std::string& phase) const {
        if (!mem_trace_enabled() || !env_truthy(std::getenv("DEME_VMM_TEMP_ARENA_SLICE_TRACE")) ||
            idx >= m_blocks.size())
            return;
        const Block& b = m_blocks[idx];
        const void* ptr = ptrForBlock(b);
        if (!ptr)
            return;
        MemoryLedgerRecord rec;
        rec.array_object = ptr;
        rec.owner = "scratch";
        rec.name = b.free ? std::string("arena-free") : b.name;
        rec.role = "ScratchTemporary";
        rec.type_name = "scratch_t";
        rec.type_size = 1;
        rec.logical_bytes = b.requested;
        rec.capacity_bytes = b.size;
        rec.committed_bytes = 0;  // physical bytes are owned by the aggregate arena record, avoiding double count
        rec.host_bytes = 0;
        rec.high_water_capacity_bytes = b.size;
        rec.device_ptr = ptr;
        rec.is_view = false;
        rec.has_host_mirror = false;
        rec.backend = "vmm-temp-arena-slice";
        rec.phase = phase;
        upsert_memory_ledger_record(rec);
    }

    void eraseArenaAndBlocks() const {
        if (!mem_trace_enabled())
            return;
        erase_memory_ledger_record(this);
        for (const auto& b : m_blocks) {
            const void* ptr = ptrForBlock(b);
            if (ptr)
                erase_memory_ledger_record(ptr);
        }
    }
};



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
