/* file: service_profiler.cpp */
/*******************************************************************************
* Copyright contributors to the oneDAL project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Recording, merging and reporting for the kernel profiler.
//
//  Each thread records into its own tree of call paths, so opening and closing a task takes no lock
//  and memory grows with the number of distinct paths, not with the number of runs. The threading
//  layer tells each thread which task started the parallel region of the chunk it runs, and the
//  trees are joined there when the report is built.
//--
*/

#include "src/services/service_profiler.h"
#include "src/threading/threading.h"

// Builds made with REQPROFILE=yes also report every task to the ITT API, so VTune and other ITT
// collectors show oneDAL's tasks on their own timelines, next to the GPU kernels they measure.
#ifdef ONEDAL_KERNEL_PROFILER
    #include <ittnotify.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #define DAAL_PROFILER_X86
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <cpuid.h>
        #include <x86intrin.h>
    #endif
#endif

#include <atomic>
#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace daal
{
namespace internal
{
namespace
{

constexpr int memory_kinds   = 2;
constexpr int transfer_kinds = 2;

inline std::uint64_t steady_ns()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

/// Whether timestamps come from the time-stamp counter, which costs about half as much as
/// steady_clock. Only an invariant counter ticks at a constant rate on every core;
/// ONEDAL_PROFILER_CLOCK=steady turns it off.
bool detect_tsc()
{
    const char * clock = std::getenv("ONEDAL_PROFILER_CLOCK");
    if (clock && std::strcmp(clock, "steady") == 0) return false;
#if defined(DAAL_PROFILER_X86) && defined(_MSC_VER)
    int regs[4] = {};
    __cpuid(regs, 0x80000000);
    if (static_cast<unsigned>(regs[0]) < 0x80000007u) return false;
    __cpuid(regs, 0x80000007);
    return (regs[3] >> 8) & 1;
#elif defined(DAAL_PROFILER_X86)
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid_max(0x80000000u, nullptr) < 0x80000007u) return false;
    __get_cpuid(0x80000007u, &a, &b, &c, &d);
    return (d >> 8) & 1;
#else
    return false;
#endif
}

const bool tsc_clock = detect_tsc();

/// A timestamp in the profiler's own units; ticks_to_ns() converts a difference of two.
inline std::uint64_t now_ticks()
{
#ifdef DAAL_PROFILER_X86
    if (tsc_clock) return __rdtsc();
#endif
    return steady_ns();
}

struct clock_origin
{
    std::uint64_t ticks = now_ticks();
    std::uint64_t ns    = steady_ns();
};

const clock_origin & origin_of_time()
{
    static const clock_origin origin;
    return origin;
}

/// Nanoseconds per tick, measured over everything the profiler has run so far.
double ns_per_tick()
{
    if (!tsc_clock) return 1.0;
    const clock_origin & origin = origin_of_time();
    // A short run gives a rough ratio; wait until the measured span is long enough.
    std::uint64_t ns = steady_ns();
    while (ns - origin.ns < 20000000) ns = steady_ns();
    const std::uint64_t ticks = now_ticks();
    return double(ns - origin.ns) / double(ticks - origin.ticks);
}

inline std::uint64_t ticks_to_ns(std::uint64_t ticks, double scale)
{
    return static_cast<std::uint64_t>(double(ticks) * scale + 0.5);
}

#ifdef ONEDAL_KERNEL_PROFILER
__itt_domain * itt_domain()
{
    static __itt_domain * const domain = __itt_domain_create("oneDAL");
    return domain;
}
#endif

inline void atomic_max(std::atomic<std::int64_t> & target, std::int64_t value)
{
    std::int64_t current = target.load(std::memory_order_relaxed);
    while (current < value && !target.compare_exchange_weak(current, value, std::memory_order_relaxed))
    {
    }
}

std::string format_time(std::uint64_t time_ns)
{
    std::ostringstream out;
    const double time = static_cast<double>(time_ns);
    if (time_ns == 0)
        out << "0 ns";
    else if (time >= 1e9)
        out << std::fixed << std::setprecision(2) << time / 1e9 << " s";
    else if (time >= 1e6)
        out << std::fixed << std::setprecision(2) << time / 1e6 << " ms";
    else if (time >= 1e3)
        out << std::fixed << std::setprecision(2) << time / 1e3 << " us";
    else
        out << time_ns << " ns";
    return out.str();
}

inline std::uint64_t positive_part(std::int64_t value)
{
    return value > 0 ? static_cast<std::uint64_t>(value) : 0;
}

/// Allocations, deallocations and copies charged to one task, not including its children.
struct own_stats
{
    std::uint64_t alloc_bytes[memory_kinds]      = {};
    std::uint64_t alloc_count[memory_kinds]      = {};
    std::uint64_t free_bytes[memory_kinds]       = {};
    std::uint64_t free_count[memory_kinds]       = {};
    std::uint64_t transfer_bytes[transfer_kinds] = {};
    std::uint64_t transfer_count[transfer_kinds] = {};
    std::uint64_t live_bytes[memory_kinds]       = {}; // still allocated when the report is built
    std::uint64_t live_count[memory_kinds]       = {};

    void add(const own_stats & other)
    {
        for (int k = 0; k < memory_kinds; ++k)
        {
            alloc_bytes[k] += other.alloc_bytes[k];
            alloc_count[k] += other.alloc_count[k];
            free_bytes[k] += other.free_bytes[k];
            free_count[k] += other.free_count[k];
            live_bytes[k] += other.live_bytes[k];
            live_count[k] += other.live_count[k];
        }
        for (int k = 0; k < transfer_kinds; ++k)
        {
            transfer_bytes[k] += other.transfer_bytes[k];
            transfer_count[k] += other.transfer_count[k];
        }
    }
};

/// A call path in one thread's tree. Only the owning thread writes to it.
struct node
{
    std::string name;
    const char * key = nullptr; // the pointer first seen for this name, compared before the string
    node * parent    = nullptr; // nullptr for a top-level task of the thread
    node * anchor    = nullptr; // for the first task of a chunk: the task that started the region
    bool threading   = false;
    bool in_parallel = false; // ran inside a parallel region at least once
    std::vector<node *> children;

    std::uint64_t count              = 0;
    std::uint64_t total              = 0;
    std::uint64_t min                = UINT64_MAX;
    std::uint64_t max                = 0;
    std::uint64_t peak[memory_kinds] = {};
    std::uint64_t self_total         = 0; // time not in child tasks or in regions this task waited for
    std::uint64_t sequential_total   = 0; // time of the runs outside parallel regions
    std::uint64_t wait_total         = 0; // time waiting for parallel regions beyond running its chunks
    std::uint64_t wait_count         = 0;
    std::uint64_t wait_max           = 0;
    std::uint64_t wait_min           = UINT64_MAX;
    own_stats own;
    bool logged = false; // the logger or the tracer wrote this task once already
#ifdef ONEDAL_KERNEL_PROFILER
    __itt_string_handle * itt_name = nullptr;
#endif
    std::uint64_t first       = 0;          // length of the first run
    std::uint64_t first_start = UINT64_MAX; // when the first run started

    // For a task in a parallel region: this thread's time per run of the region, so the report can
    // take the busiest thread of each run rather than of the whole program.
    std::vector<std::pair<std::uint64_t, std::uint64_t> > runs;
    std::uint64_t unsplit_total = 0; // time recorded after `runs` reached max_runs_per_node
};

constexpr std::size_t max_runs_per_node      = 1 << 16;
constexpr std::size_t max_regions_per_thread = 1 << 18;

// Allocations on a thread inside a parallel region reach the process-wide live count in steps of
// this size, so threads do not contend on it; larger blocks are counted at once.
constexpr std::int64_t live_batch_bytes = 64 << 10;

// Timeline events a thread reserves at a time from the process-wide limit.
constexpr std::int64_t trace_reserve = 1024;

// Blocks a thread keeps in its own table before it moves them to the shared one.
constexpr std::size_t max_recent_blocks = 128;

struct frame
{
    node * task                              = nullptr;
    std::uint64_t start                      = 0;
    std::uint64_t run                        = 0;     // the region run, or the sequential run, this task is part of
    std::uint64_t paused                     = 0;     // time this thread spent on chunks of other regions while this task waited
    std::uint64_t child_time                 = 0;     // time in child tasks and in parallel regions this task started
    bool counts_in_parent                    = false; // false for the first task of a chunk that interrupted other work
    bool in_parallel                         = false;
    bool sequential                          = false; // sequential tasks track memory process-wide, others per thread
    bool opens_chunk                         = false; // the first task of a chunk
    bool outermost                           = false; // a sequential task with nothing open around it
    std::int64_t live_at_start[memory_kinds] = {};
    std::int64_t peak_live[memory_kinds]     = {}; // per-thread tasks only
    std::int64_t outer_peak[memory_kinds]    = {}; // sequential tasks only: the enclosing task's peak so far
};

struct trace_event
{
    const node * task;
    std::uint64_t start;
    std::uint64_t end;
    std::uint64_t run; // for the first task of a chunk: the region; otherwise 0
    bool parallel;
};

struct memory_sample
{
    std::uint64_t time;
    std::int64_t live;
    int kind;
};

/// A chunk of a parallel region that a thread is running.
struct chunk
{
    node * parent          = nullptr; // the task the region was started from, or nullptr
    std::uint64_t run      = 0;
    std::size_t base       = 0;     // stack depth when the chunk started
    std::size_t pause_from = 0;     // first stack frame the chunk interrupted
    bool interrupts        = false; // the thread was busy with other work when it took the chunk
    std::uint64_t start    = 0;
    std::size_t recent     = 0;  // first entry of the thread's `recent` blocks allocated in the chunk
    int task               = -1; // depth of the task the profiler opened for the chunk, or -1
};

/// Size and kind of live recorded blocks, by address. Open addressing with linear probing, so
/// recording a block does not allocate unless the table grows.
struct block_table
{
    struct slot
    {
        const void * ptr        = nullptr;
        std::uint64_t size_kind = 0;       // size << 1 | kind
        const node * owner      = nullptr; // the task the allocation was charged to, if any
    };

    std::vector<slot> slots; // empty or a power of two
    std::size_t used = 0;

    static std::size_t home(const void * ptr, std::size_t mask)
    {
        return static_cast<std::size_t>(((reinterpret_cast<std::uintptr_t>(ptr) >> 4) * 0x9E3779B97F4A7C15ull) >> 16) & mask;
    }

    void insert(const void * ptr, std::uint64_t size_kind, const node * owner)
    {
        if (2 * (used + 1) > slots.size()) grow();
        const std::size_t mask = slots.size() - 1;
        for (std::size_t i = home(ptr, mask);; i = (i + 1) & mask)
        {
            if (slots[i].ptr == nullptr || slots[i].ptr == ptr)
            {
                used += slots[i].ptr == nullptr ? 1 : 0;
                slots[i] = { ptr, size_kind, owner };
                return;
            }
        }
    }

    bool erase(const void * ptr, std::uint64_t & size_kind)
    {
        if (slots.empty()) return false;
        const std::size_t mask = slots.size() - 1;
        std::size_t i          = home(ptr, mask);
        while (slots[i].ptr != ptr)
        {
            if (slots[i].ptr == nullptr) return false;
            i = (i + 1) & mask;
        }
        size_kind = slots[i].size_kind;
        // Backward-shift deletion: move each following entry into the hole unless its home slot lies
        // between the hole and the entry, so every entry stays reachable without tombstones.
        for (std::size_t j = (i + 1) & mask; slots[j].ptr != nullptr; j = (j + 1) & mask)
        {
            const std::size_t k = home(slots[j].ptr, mask);
            if (((j - k) & mask) >= ((j - i) & mask))
            {
                slots[i] = slots[j];
                i        = j;
            }
        }
        slots[i] = slot();
        --used;
        return true;
    }

    void grow()
    {
        std::vector<slot> previous(slots.empty() ? 64 : 2 * slots.size());
        previous.swap(slots);
        used = 0;
        for (const slot & entry : previous)
        {
            if (entry.ptr) insert(entry.ptr, entry.size_kind, entry.owner);
        }
    }
};

struct alignas(64) allocation_shard
{
    std::mutex mutex;
    block_table table;
};

struct thread_state
{
    std::size_t index = 0;
    node root;
    std::deque<node> nodes; // a deque keeps node addresses stable
    std::vector<frame> stack;
    std::vector<chunk> chunks; // innermost last

    // Blocking regions this thread started and is waiting for, innermost last.
    struct started_region
    {
        std::uint64_t run;
        std::uint64_t start;
        std::size_t depth;        // stack depth of the task that started it
        std::uint64_t child_time; // that task's child time when the region started
    };
    std::vector<started_region> started;
    std::uint64_t runs_started = 0;
    // Each parallel region this thread started.
    struct region_record
    {
        std::uint64_t run;
        std::uint64_t outer_run; // the run it was started in
        std::uint64_t start;
    };
    std::vector<region_record> regions;

    std::int64_t live[memory_kinds]    = {};
    std::int64_t pending[memory_kinds] = {}; // change in live memory not yet in the process-wide count
    // Blocks this thread allocated that are not in the shared table yet, newest last; most are
    // freed here, soon after they were allocated.
    std::vector<block_table::slot> recent;
    std::unordered_map<const node *, own_stats> anchored; // charged while idle inside a region
    const node * charged_anchor = nullptr;                // the last entry of `anchored` looked up
    own_stats * charged         = nullptr;
    own_stats outside;                // charged with no task open
    std::int64_t trace_slots     = 0; // reserved, not yet used timeline events
    std::uint64_t memory_records = 0; // allocations and frees recorded
    std::vector<trace_event> events;
    std::vector<memory_sample> samples;
};

constexpr std::size_t shard_count = 256;

struct alignas(64) padded_counter
{
    std::atomic<std::int64_t> value { 0 };
};

struct global_state
{
    std::mutex threads_mutex;
    std::vector<std::unique_ptr<thread_state> > threads;

    allocation_shard shards[shard_count];
    padded_counter live[memory_kinds];
    padded_counter region_peak[memory_kinds]; // peak since the innermost sequential task started
    padded_counter process_peak[memory_kinds];
    // Frees of blocks that were still in another thread's `recent` table, settled when it is spilled.
    std::mutex orphans_mutex;
    std::unordered_set<const void *> orphans;
    std::atomic<std::size_t> orphan_count { 0 };

    std::atomic<int> open_calls { 0 }; // outermost tasks open on any thread

    std::atomic<std::int64_t> trace_budget { 0 };
    std::atomic<std::uint64_t> trace_dropped { 0 };
    std::uint64_t origin = origin_of_time().ticks;

    global_state() { trace_budget = trace_limit(); }

    static std::int64_t trace_limit()
    {
        const char * value = std::getenv("ONEDAL_PROFILER_TRACE_LIMIT");
        if (value)
        {
            char * end        = nullptr;
            const long long n = std::strtoll(value, &end, 10);
            if (end != value && *end == '\0' && n >= 0) return n;
        }
        return 4000000;
    }
};

/// Where the logger, the tracer and the report go. Never destroyed, like global_state.
struct log_sink
{
    std::mutex mutex;
    std::ofstream file;
    bool to_file    = false;
    bool every_call = false; // otherwise each call site is logged once
    bool noted      = false;
    std::unordered_set<std::string> seen;

    log_sink()
    {
        const char * path = std::getenv("ONEDAL_PROFILER_OUTPUT");
        if (path && *path)
        {
            file.open(path, std::ios::out | std::ios::trunc);
            to_file = file.is_open();
            if (!to_file) std::cerr << "ONEDAL PROFILER: cannot open " << path << ", writing to stderr" << std::endl;
        }
        // A pipe that nobody reads while oneDAL runs (a notebook kernel) blocks the process once it is
        // full, so stderr gets one line per call site. A file takes every call.
        every_call = to_file;
    }

    std::ostream & out() { return to_file ? static_cast<std::ostream &>(file) : std::cerr; }
};

log_sink & sink()
{
    static log_sink * instance = new log_sink();
    return *instance;
}

void report_at_exit();
profiler_region region_capture(bool blocking);
void region_finish(const profiler_region & region);
void region_enter(const profiler_region & region);
void region_leave();

const profiler_region_hooks region_hooks = { region_capture, region_finish, region_enter, region_leave };

// Never destroyed: tasks can still close from static destructors that run after the report.
global_state & state()
{
    static global_state * instance = [] {
        global_state * s = new global_state();
        if (is_analyzer_enabled() || profiler_trace_path()) std::atexit(report_at_exit);
        // Regions started from now on report their chunks. Not done at load time: on Windows that
        // would load the threading library from inside DllMain.
        if (is_profiler_enabled()) _daal_threader_set_profiler_hooks(&region_hooks);
        return s;
    }();
    return *instance;
}

thread_local thread_state * current_thread_state = nullptr;

// Call site of the next parallel region this thread starts.
thread_local const profiler_site * pending_site = nullptr;

void spill_recent(thread_state & ts);

/// Spills the thread's recent blocks when the thread exits.
struct thread_exit_spill
{
    ~thread_exit_spill()
    {
        if (current_thread_state) spill_recent(*current_thread_state);
    }
};

thread_state & this_thread()
{
    if (!current_thread_state)
    {
        auto created  = std::make_unique<thread_state>();
        auto & global = state();
        std::lock_guard<std::mutex> lock(global.threads_mutex);
        created->index       = global.threads.size();
        current_thread_state = created.get();
        global.threads.push_back(std::move(created));
        static thread_local thread_exit_spill exit_spill;
        (void)exit_spill;
    }
    return *current_thread_state;
}

allocation_shard & shard_for(const void * ptr)
{
    const auto bits = reinterpret_cast<std::uintptr_t>(ptr) >> 4;
    return state().shards[(bits * 0x9E3779B97F4A7C15ull) >> 56];
}

/// A run number unique across threads; run 0 means none.
std::uint64_t new_run(thread_state & ts)
{
    return (static_cast<std::uint64_t>(ts.index + 1) << 40) | ++ts.runs_started;
}

/// Adds this thread's pending change in live memory of `kind` to the process-wide count.
void flush_live(thread_state & ts, int kind);

void flush_live(thread_state & ts)
{
    for (int k = 0; k < memory_kinds; ++k) flush_live(ts, k);
}

own_stats & charge_target(thread_state & ts);
void update_live(thread_state & ts, int kind, std::int64_t delta);

void charge_free(thread_state & ts, std::uint64_t size_kind)
{
    const std::uint64_t bytes = size_kind >> 1;
    const int k               = static_cast<int>(size_kind & 1);
    own_stats & own           = charge_target(ts);
    own.free_bytes[k] += bytes;
    own.free_count[k] += 1;
    update_live(ts, k, -static_cast<std::int64_t>(bytes));
}

/// Whether `ptr` was freed by another thread while it was in this thread's `recent` table.
/// Called with the shard of `ptr` locked, as every change to the orphans is.
bool take_orphan(const void * ptr)
{
    auto & global = state();
    if (global.orphan_count.load(std::memory_order_acquire) == 0) return false;
    std::lock_guard<std::mutex> lock(global.orphans_mutex);
    if (global.orphans.erase(ptr) == 0) return false;
    global.orphan_count.fetch_sub(1, std::memory_order_release);
    return true;
}

/// Removes entries [first, last) of this thread's recent blocks, keeping the chunks' offsets into them.
void erase_recent(thread_state & ts, std::size_t first, std::size_t last)
{
    ts.recent.erase(ts.recent.begin() + first, ts.recent.begin() + last);
    for (auto & c : ts.chunks)
    {
        if (c.recent > first) c.recent = c.recent >= last ? c.recent - (last - first) : first;
    }
}

/// Moves entries [first, last) of this thread's recent blocks to the shared table, where any thread
/// can free them.
void spill_recent(thread_state & ts, std::size_t first, std::size_t last)
{
    last = std::min(last, ts.recent.size());
    if (first >= last) return;
    for (std::size_t i = first; i < last; ++i)
    {
        const auto & entry = ts.recent[i];
        bool freed         = false;
        {
            auto & shard = shard_for(entry.ptr);
            std::lock_guard<std::mutex> lock(shard.mutex);
            freed = take_orphan(entry.ptr);
            if (!freed) shard.table.insert(entry.ptr, entry.size_kind, entry.owner);
        }
        if (freed) charge_free(ts, entry.size_kind);
    }
    erase_recent(ts, first, last);
}

void spill_recent(thread_state & ts)
{
    spill_recent(ts, 0, ts.recent.size());
}

/// Settles the frees other threads made of blocks still in this thread's `recent` table.
void settle_orphans(thread_state & ts)
{
    auto & global = state();
    if (global.orphan_count.load(std::memory_order_acquire) == 0 || ts.recent.empty()) return;
    std::vector<std::uint64_t> freed;
    for (std::size_t i = ts.recent.size(); i-- > 0;)
    {
        const auto & entry = ts.recent[i];
        auto & shard       = shard_for(entry.ptr);
        std::lock_guard<std::mutex> lock(shard.mutex);
        if (take_orphan(entry.ptr))
        {
            freed.push_back(entry.size_kind);
            erase_recent(ts, i, i + 1);
        }
    }
    for (const std::uint64_t size_kind : freed) charge_free(ts, size_kind);
}

/// @param[in] stable_name  Whether `name` outlives the profiler, as the string literals in the macros
///                         do; only then is its address kept for a faster lookup.
node * find_child(thread_state & ts, node * parent, node * anchor, const char * name, bool threading, bool stable_name = true)
{
    auto & children = parent ? parent->children : ts.root.children;
    for (node * child : children)
    {
        if (child->key == name && child->anchor == anchor) return child;
    }
    for (node * child : children)
    {
        if (child->anchor == anchor && child->name == name) return child;
    }
    ts.nodes.emplace_back();
    node * created     = &ts.nodes.back();
    created->name      = name;
    created->key       = stable_name ? name : nullptr;
    created->parent    = parent;
    created->anchor    = anchor;
    created->threading = threading;
    children.push_back(created);
    return created;
}

/// @param[in] shared   Whether the task tracks memory process-wide, logs and is traced.
///                     False only for the private calibration state.
constexpr std::size_t max_args_variants = 8;

/// The row of a task with arguments under `parent`: "name [args]", or "name [other args]" once the
/// task has max_args_variants rows there already. Existing rows are matched without building the label.
node * find_args_child(thread_state & ts, node * parent, node * anchor, const char * name, const std::string & args)
{
    std::size_t length = args.size();
    while (length > 0 && (args[length - 1] == ' ' || args[length - 1] == ';')) --length;
    const std::size_t name_length = std::strlen(name);
    std::size_t variants          = 0;
    for (node * sibling : parent ? parent->children : ts.root.children)
    {
        const std::string & label = sibling->name;
        if (sibling->anchor != anchor || label.size() < name_length + 3 || label.compare(0, name_length, name) != 0
            || label.compare(name_length, 2, " [") != 0)
        {
            continue;
        }
        if (label.size() == name_length + length + 3 && label.compare(name_length + 2, length, args, 0, length) == 0) return sibling;
        ++variants;
    }
    const std::string label = std::string(name) + " [" + (variants < max_args_variants ? args.substr(0, length) : "other args") + "]";
    return find_child(ts, parent, anchor, label.c_str(), false, false);
}

/// @param[in] args     The task's arguments, or nullptr
int begin_task(thread_state & ts, const char * name, bool threading, bool shared, const std::string * args = nullptr)
{
    auto & global = state();
    frame f;
    node * parent    = nullptr;
    node * anchor    = nullptr;
    bool in_parallel = true;

    const chunk * c = ts.chunks.empty() ? nullptr : &ts.chunks.back();
    if (c && ts.stack.size() == c->base)
    {
        // The first task of a chunk belongs under the task that started the region, whatever else
        // this thread has open.
        if (!ts.stack.empty() && ts.stack.back().task == c->parent)
            parent = c->parent;
        else
            anchor = c->parent;
        f.run         = c->run;
        f.opens_chunk = true;
    }
    else if (!ts.stack.empty())
    {
        parent      = ts.stack.back().task;
        in_parallel = threading || ts.stack.back().in_parallel;
        f.run       = ts.stack.back().run;
    }
    else
    {
        // Not inside a chunk the threading layer reported: a top-level task of this thread.
        in_parallel = threading;
    }
    if (args)
    {
        f.task = find_args_child(ts, parent, anchor, name, *args);
    }
    else
    {
        f.task = find_child(ts, parent, anchor, name, threading);
    }
    f.task->in_parallel |= in_parallel;
    f.in_parallel      = in_parallel;
    f.counts_in_parent = parent != nullptr;

    if (threading && shared && is_logger_enabled() && (is_every_call_logged() || !f.task->logged))
    {
        f.task->logged = true;
        profiler_log(("threading " + f.task->name).c_str(), "-----------------------------------------------------------------------------\n"
                                                            "THREADING Profiler task started "
                                                                + f.task->name + "\n");
    }

    f.sequential = !in_parallel && shared;
    f.outermost  = f.sequential && ts.stack.empty() && ts.chunks.empty();
    if (f.outermost) global.open_calls.fetch_add(1, std::memory_order_relaxed);
    if (f.sequential)
    {
        f.run = new_run(ts);
        flush_live(ts);
        for (int k = 0; k < memory_kinds; ++k)
        {
            f.live_at_start[k] = global.live[k].value.load(std::memory_order_relaxed);
            f.outer_peak[k]    = global.region_peak[k].value.exchange(f.live_at_start[k], std::memory_order_relaxed);
        }
    }
    else
    {
        for (int k = 0; k < memory_kinds; ++k)
        {
            f.live_at_start[k] = ts.live[k];
            f.peak_live[k]     = ts.live[k];
        }
    }
#ifdef ONEDAL_KERNEL_PROFILER
    if (shared)
    {
        if (!f.task->itt_name) f.task->itt_name = __itt_string_handle_create(f.task->name.c_str());
        __itt_task_begin(itt_domain(), __itt_null, __itt_null, f.task->itt_name);
    }
#endif
    ts.stack.push_back(f);
    // Taken last, so the bookkeeping above is charged to the parent rather than to this task.
    ts.stack.back().start = now_ticks();
    return static_cast<int>(ts.stack.size() - 1);
}

/// Whether the timeline can take one more event from this thread.
bool take_trace_slot(thread_state & ts)
{
    auto & global = state();
    if (ts.trace_slots == 0)
    {
        const std::int64_t left = global.trace_budget.fetch_sub(trace_reserve, std::memory_order_relaxed);
        ts.trace_slots          = std::max<std::int64_t>(0, std::min(left, trace_reserve));
    }
    if (ts.trace_slots == 0)
    {
        global.trace_dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    --ts.trace_slots;
    return true;
}

void end_one(thread_state & ts, std::uint64_t end, bool shared)
{
    auto & global = state();
    const frame f = ts.stack.back();
    ts.stack.pop_back();
#ifdef ONEDAL_KERNEL_PROFILER
    if (shared) __itt_task_end(itt_domain());
#endif

    node & task                 = *f.task;
    const std::uint64_t elapsed = end > f.start ? end - f.start : 0;
    const std::uint64_t length  = elapsed > f.paused ? elapsed - f.paused : 0;
    if (task.count == 0)
    {
        task.first       = length;
        task.first_start = f.start;
    }
    task.count += 1;
    task.total += length;
    task.self_total += length > f.child_time ? length - f.child_time : 0;
    if (f.counts_in_parent && !ts.stack.empty()) ts.stack.back().child_time += length;
    task.min = std::min(task.min, length);
    task.max = std::max(task.max, length);

    if (f.sequential)
    {
        flush_live(ts);
        for (int k = 0; k < memory_kinds; ++k)
        {
            const std::int64_t peak = global.region_peak[k].value.load(std::memory_order_relaxed);
            task.peak[k]            = std::max(task.peak[k], positive_part(peak - f.live_at_start[k]));
            // The enclosing task's peak is the larger of its own so far and this task's.
            atomic_max(global.region_peak[k].value, f.outer_peak[k]);
        }
    }
    if (!f.in_parallel)
    {
        task.sequential_total += length;
    }
    else
    {
        auto & runs = task.runs;
        if (!runs.empty() && runs.back().first == f.run)
            runs.back().second += length;
        else if (runs.size() < max_runs_per_node)
            runs.emplace_back(f.run, length);
        else
            task.unsplit_total += length;
        for (int k = 0; k < memory_kinds; ++k)
        {
            task.peak[k] = std::max(task.peak[k], positive_part(f.peak_live[k] - f.live_at_start[k]));
            if (!ts.stack.empty() && !ts.stack.back().sequential)
            {
                ts.stack.back().peak_live[k] = std::max(ts.stack.back().peak_live[k], f.peak_live[k]);
            }
        }
    }

    if (shared && profiler_trace_path() && take_trace_slot(ts))
        ts.events.push_back({ &task, f.start, end, f.opens_chunk ? f.run : 0, f.in_parallel });

    if (shared && is_tracer_enabled() && (is_every_call_logged() || !task.logged))
    {
        task.logged           = true;
        const std::string key = "tracer " + task.name;
        if (!task.threading)
            profiler_log(key.c_str(), task.name + " " + format_time(ticks_to_ns(length, ns_per_tick())) + "\n");
        else
            profiler_log(key.c_str(), "THREADING " + task.name + " finished on the main rank(time could be different for other ranks): "
                                          + format_time(ticks_to_ns(length, ns_per_tick())) + "\n");
    }
}

void report_call();

void end_task(thread_state & ts, int depth, bool shared)
{
    const std::uint64_t end = now_ticks();
    if (depth < 0) return;
    bool call_done = false;
    while (ts.stack.size() > static_cast<std::size_t>(depth))
    {
        const bool outermost = ts.stack.back().outermost;
        end_one(ts, end, shared);
        if (outermost) call_done = state().open_calls.fetch_sub(1, std::memory_order_relaxed) == 1;
    }
    if (call_done && is_report_per_call()) report_call();
}

/// The stats that an allocation or copy on this thread is charged to right now.
own_stats & charge_target(thread_state & ts)
{
    if (!ts.chunks.empty() && ts.stack.size() == ts.chunks.back().base)
    {
        const node * parent = ts.chunks.back().parent;
        if (!parent) return ts.outside;
        if (parent != ts.charged_anchor || !ts.charged)
        {
            ts.charged        = &ts.anchored[parent];
            ts.charged_anchor = parent;
        }
        return *ts.charged;
    }
    if (!ts.stack.empty()) return ts.stack.back().task->own;
    return ts.outside;
}

/// The task an allocation on this thread is charged to right now, as charge_target() picks it, or nullptr.
const node * charge_owner(const thread_state & ts)
{
    if (!ts.chunks.empty() && ts.stack.size() == ts.chunks.back().base) return ts.chunks.back().parent;
    return ts.stack.empty() ? nullptr : ts.stack.back().task;
}

void record_sample(thread_state & ts, int kind, std::int64_t live)
{
    if (take_trace_slot(ts)) ts.samples.push_back({ now_ticks(), live, kind });
}

void flush_live(thread_state & ts, int kind)
{
    const std::int64_t delta = ts.pending[kind];
    if (delta == 0) return;
    ts.pending[kind]         = 0;
    auto & global            = state();
    const std::int64_t value = global.live[kind].value.fetch_add(delta, std::memory_order_relaxed) + delta;
    if (delta > 0)
    {
        atomic_max(global.region_peak[kind].value, value);
        atomic_max(global.process_peak[kind].value, value);
    }
    if (profiler_trace_path()) record_sample(ts, kind, value);
}

void update_live(thread_state & ts, int kind, std::int64_t delta)
{
    ts.live[kind] += delta;
    const bool sequential = !ts.stack.empty() && ts.stack.back().sequential;
    if (delta > 0 && !ts.stack.empty() && !sequential)
    {
        ts.stack.back().peak_live[kind] = std::max(ts.stack.back().peak_live[kind], ts.live[kind]);
    }
    ts.pending[kind] += delta;
    // A sequential task counts every byte at once; elsewhere small blocks are counted in batches.
    if (sequential || ts.pending[kind] >= live_batch_bytes || ts.pending[kind] <= -live_batch_bytes) flush_live(ts, kind);
}

/// The row name for the chunks of regions started at `site`, interned so that tasks can key on it.
const char * site_name(const profiler_site & site)
{
    static thread_local std::unordered_map<const char *, std::unordered_map<int, const char *> > cache;
    const char *& cached = cache[site.file][site.line];
    if (cached) return cached;

    std::string file        = site.file ? site.file : "?";
    const std::size_t slash = file.find_last_of("/\\");
    if (slash != std::string::npos) file.erase(0, slash + 1);
    const std::string name =
        std::string("parallel loop: ") + (site.function ? site.function : "?") + " (" + file + ':' + std::to_string(site.line) + ')';

    static std::mutex mutex;
    static auto * names = new std::unordered_set<std::string>(); // never destroyed, like global_state
    std::lock_guard<std::mutex> lock(mutex);
    cached = names->insert(name).first->c_str();
    return cached;
}

profiler_region region_capture(bool blocking)
{
    const profiler_site * site = pending_site;
    pending_site               = nullptr;
    thread_state & ts          = this_thread();
    const chunk * c            = ts.chunks.empty() ? nullptr : &ts.chunks.back();
    const bool own             = !ts.stack.empty() && (!c || ts.stack.size() > c->base);
    node * parent              = nullptr;
    std::uint64_t outer_run    = 0;
    if (own)
    {
        parent    = ts.stack.back().task;
        outer_run = ts.stack.back().run;
    }
    else if (c)
    {
        parent    = c->parent;
        outer_run = c->run;
    }

    // Every region is a run of its own: the report takes the busiest thread of each, and counts it on
    // the line of the thread that started it within the enclosing run.
    const std::uint64_t run = new_run(ts);
    if (ts.regions.size() < max_regions_per_thread) ts.regions.push_back({ run, outer_run, now_ticks() });
    if (blocking && own) ts.started.push_back({ run, now_ticks(), ts.stack.size() - 1, ts.stack.back().child_time });
    return { parent, run, site ? site_name(*site) : nullptr };
}

void region_finish(const profiler_region & region)
{
    thread_state & ts = this_thread();
    for (std::size_t i = ts.started.size(); i-- > 0;)
    {
        const auto started = ts.started[i];
        if (started.run != region.run) continue;
        ts.started.erase(ts.started.begin() + i);
        if (started.depth < ts.stack.size())
        {
            // The wait counts as child time, except for the parts already counted: the chunks
            // this thread ran itself as child tasks.
            frame & f                   = ts.stack[started.depth];
            const std::uint64_t elapsed = now_ticks() - started.start;
            const std::uint64_t counted = f.child_time - started.child_time;
            const std::uint64_t wait    = elapsed > counted ? elapsed - counted : 0;
            f.child_time += wait;
            f.task->wait_total += wait;
            f.task->wait_count += 1;
            f.task->wait_max = std::max(f.task->wait_max, wait);
            f.task->wait_min = std::min(f.task->wait_min, wait);
        }
        return;
    }
}

void region_enter(const profiler_region & region)
{
    thread_state & ts = this_thread();
    chunk c;
    c.parent = static_cast<node *>(const_cast<void *>(region.parent));
    c.run    = region.run;
    c.base   = ts.stack.size();
    c.recent = ts.recent.size();

    // A thread that waits inside other work (for example in a nested parallel call) may be given a
    // chunk of an unrelated region. Its time is not part of the tasks it interrupted: those are the
    // ones above the region's parent, or above the outermost chunk of the same region.
    const node * innermost = ts.stack.empty() ? nullptr : ts.stack.back().task;
    if (!ts.chunks.empty() && ts.stack.size() == ts.chunks.back().base) innermost = ts.chunks.back().parent;
    if (!ts.stack.empty() && innermost != c.parent)
    {
        c.interrupts = true;
        c.pause_from = 0;
        bool found   = false;
        for (std::size_t i = ts.stack.size(); i-- > 0;)
        {
            if (ts.stack[i].task == c.parent)
            {
                c.pause_from = i + 1;
                found        = true;
                break;
            }
        }
        for (std::size_t i = 0; !found && i < ts.chunks.size(); ++i)
        {
            if (ts.chunks[i].parent == c.parent)
            {
                c.pause_from = ts.chunks[i].base;
                break;
            }
        }
        c.start = now_ticks();
    }
    ts.chunks.push_back(c);
    if (region.name) ts.chunks.back().task = begin_task(ts, region.name, true, true);
}

void region_leave()
{
    thread_state & ts = this_thread();
    if (ts.chunks.empty()) return;
    if (ts.chunks.back().task >= 0) end_task(ts, ts.chunks.back().task, true);
    const chunk c = ts.chunks.back();
    ts.chunks.pop_back();
    // What the chunk allocated and did not free may be freed by another thread after the region.
    spill_recent(ts, c.recent, ts.recent.size());
    if (c.interrupts)
    {
        const std::uint64_t elapsed = now_ticks() - c.start;
        for (std::size_t i = c.pause_from; i < c.base && i < ts.stack.size(); ++i) ts.stack[i].paused += elapsed;
    }
}

// Merged tree used by the report.
struct merged
{
    std::string name;
    merged * parent = nullptr;
    std::vector<std::unique_ptr<merged> > children;
    bool parallel                    = false;
    std::uint64_t count              = 0;
    std::uint64_t cpu                = 0;
    std::uint64_t min                = UINT64_MAX;
    std::uint64_t max                = 0;
    std::uint64_t peak[memory_kinds] = {};
    own_stats own;
    std::unordered_map<std::size_t, std::uint64_t> per_thread;

    struct run_time
    {
        std::uint64_t run;
        std::size_t thread;
        std::uint64_t time;
    };
    std::vector<run_time> runs;
    std::uint64_t unsplit_busiest = 0;
    std::uint64_t sequential_cpu  = 0; // runs outside any parallel region, which have no run entries
    std::uint64_t wait_total      = 0;
    std::uint64_t wait_count      = 0;
    std::uint64_t wait_max        = 0;
    std::uint64_t wait_min        = UINT64_MAX;
    std::uint64_t self_cpu        = 0;
    std::uint64_t first           = 0;
    std::uint64_t first_start     = UINT64_MAX;

    std::uint64_t wall        = 0;
    std::uint64_t busiest     = 0;
    std::uint64_t mean_thread = 0;
    std::uint64_t threads     = 0;
    std::uint64_t self        = 0;
    own_stats inclusive;
};

struct merged_tree
{
    merged root;
    own_stats outside;
    std::uint64_t task_runs      = 0;
    std::uint64_t memory_records = 0;
    std::size_t threads          = 0;
};

merged * merged_child(merged * parent, const std::string & name)
{
    for (auto & child : parent->children)
    {
        if (child->name == name) return child.get();
    }
    parent->children.push_back(std::make_unique<merged>());
    merged * created = parent->children.back().get();
    created->name    = name;
    created->parent  = parent;
    return created;
}

merged * resolve(merged_tree & tree, std::unordered_map<const node *, merged *> & map, const node * n)
{
    auto it = map.find(n);
    if (it != map.end()) return it->second;
    merged * parent = n->parent ? resolve(tree, map, n->parent) : (n->anchor ? resolve(tree, map, n->anchor) : &tree.root);
    merged * result = merged_child(parent, n->name);
    map.emplace(n, result);
    return result;
}

using region_map = std::unordered_map<std::uint64_t, std::pair<std::uint64_t, std::size_t> >;

const char * const waiting_row_name = "(in parallel regions, outside tasks)";

void finish(merged & m, const region_map & regions)
{
    for (auto & child : m.children) finish(*child, regions);

    m.wall        = m.cpu;
    m.busiest     = m.cpu;
    m.mean_thread = m.cpu;
    m.threads     = m.per_thread.size();
    if (m.parallel)
    {
        // Time per run and thread.
        std::unordered_map<std::uint64_t, std::unordered_map<std::size_t, std::uint64_t> > lines;
        for (const auto & entry : m.runs) lines[entry.run][entry.thread] += entry.time;

        m.busiest     = m.unsplit_busiest + m.sequential_cpu;
        m.mean_thread = m.unsplit_busiest + m.sequential_cpu;
        m.threads     = 0;
        for (const auto & run : lines)
        {
            std::uint64_t busiest = 0, sum = 0;
            for (const auto & line : run.second)
            {
                busiest = std::max(busiest, line.second);
                sum += line.second;
            }
            m.busiest += busiest;
            m.mean_thread += sum / run.second.size();
            m.threads = std::max<std::uint64_t>(m.threads, run.second.size());
        }

        // A run lasts as long as its busiest thread, and that time is spent on the line of the thread
        // that started the region, within the run it was started in. Innermost runs are folded first.
        const auto depth = [&](std::uint64_t run) {
            std::size_t d = 0;
            for (auto it = regions.find(run); it != regions.end() && it->second.first != 0 && d <= regions.size();
                 it      = regions.find(it->second.first))
            {
                ++d;
            }
            return d;
        };
        std::priority_queue<std::pair<std::size_t, std::uint64_t> > order;
        for (const auto & run : lines) order.push({ depth(run.first), run.first });
        m.wall = m.unsplit_busiest + m.sequential_cpu;
        while (!order.empty())
        {
            const std::uint64_t run = order.top().second;
            order.pop();
            const auto it = lines.find(run);
            if (it == lines.end()) continue;
            std::uint64_t busiest = 0;
            for (const auto & line : it->second) busiest = std::max(busiest, line.second);
            lines.erase(it);
            const auto region = regions.find(run);
            if (region != regions.end() && region->second.first != 0)
            {
                const std::uint64_t outer = region->second.first;
                if (lines.find(outer) == lines.end()) order.push({ depth(outer), outer });
                lines[outer][region->second.second] += busiest;
            }
            else
            {
                m.wall += busiest;
            }
        }
    }

    std::uint64_t children_wall = 0;
    for (const auto & child : m.children) children_wall += child->wall;
    // A parallel task's self time is its share of the CPU time, scaled to wall time; it cannot be more
    // than what its children leave.
    m.self = m.parallel && m.cpu > 0 ? static_cast<std::uint64_t>(double(m.self_cpu) * double(m.wall) / double(m.cpu)) : m.self_cpu;
    m.self = std::min(m.self, m.wall > children_wall ? m.wall - children_wall : 0);

    // Without this row, time spent waiting for parallel regions whose chunks open no task would show
    // up nowhere. The rows of the regions' tasks already take the busiest thread, which overlaps the
    // wait, so the row gets only what the task's own time and its other rows leave of the task.
    if (m.wait_total > 0)
    {
        const std::uint64_t accounted = m.self + children_wall;
        merged * waiting              = merged_child(&m, waiting_row_name);
        waiting->count                = m.wait_count;
        waiting->max                  = m.wait_max;
        waiting->min                  = m.wait_min;
        waiting->cpu                  = m.wait_total;
        waiting->wall    = m.parallel && m.cpu > 0 ? static_cast<std::uint64_t>(double(m.wait_total) * double(m.wall) / double(m.cpu)) : m.wait_total;
        waiting->wall    = std::min(waiting->wall, m.wall > accounted ? m.wall - accounted : 0);
        waiting->self    = waiting->wall;
        waiting->busiest = waiting->wall;
        waiting->mean_thread = waiting->wall;
        waiting->threads     = m.threads;
    }

    m.inclusive = m.own;
    for (auto & child : m.children)
    {
        m.inclusive.add(child->inclusive);
        for (int k = 0; k < memory_kinds; ++k) m.peak[k] = std::max(m.peak[k], child->peak[k]);
    }
}

/// Caps every row at the row it is in. A nested region's busiest thread may run while the thread
/// that started it was taken away by other work, which leaves its rows slightly longer than the
/// task they are in.
void cap_at_parent(merged & m, std::uint64_t limit)
{
    m.wall = std::min(m.wall, limit);
    m.self = std::min(m.self, m.wall);
    for (auto & child : m.children) cap_at_parent(*child, m.wall);
}

merged_tree build_tree()
{
    merged_tree tree;
    auto & global = state();
    std::lock_guard<std::mutex> lock(global.threads_mutex);
    std::unordered_map<const node *, merged *> map;
    for (const auto & ts : global.threads)
    {
        for (const node & n : ts->nodes)
        {
            merged * m = resolve(tree, map, &n);
            m->parallel |= n.in_parallel;
            m->count += n.count;
            m->cpu += n.total;
            if (n.count > 0)
            {
                m->min = std::min(m->min, n.min);
                m->max = std::max(m->max, n.max);
            }
            for (int k = 0; k < memory_kinds; ++k) m->peak[k] = std::max(m->peak[k], n.peak[k]);
            m->own.add(n.own);
            m->per_thread[ts->index] += n.total;
            for (const auto & run : n.runs) m->runs.push_back({ run.first, ts->index, run.second });
            m->self_cpu += n.self_total;
            m->sequential_cpu += n.sequential_total;
            m->wait_total += n.wait_total;
            m->wait_count += n.wait_count;
            m->wait_max        = std::max(m->wait_max, n.wait_max);
            m->wait_min        = std::min(m->wait_min, n.wait_min);
            m->unsplit_busiest = std::max(m->unsplit_busiest, n.unsplit_total);
            if (n.count > 0 && n.first_start < m->first_start)
            {
                m->first_start = n.first_start;
                m->first       = n.first;
            }
            tree.task_runs += n.count;
        }
        if (!ts->nodes.empty()) tree.threads += 1;
        tree.memory_records += ts->memory_records;
        for (const auto & entry : ts->anchored) resolve(tree, map, entry.first)->own.add(entry.second);
        tree.outside.add(ts->outside);
    }
    // Blocks still allocated, charged to the task that allocated them.
    const auto add_live = [&](const block_table::slot & entry) {
        const int k        = static_cast<int>(entry.size_kind & 1);
        own_stats & target = entry.owner ? resolve(tree, map, entry.owner)->own : tree.outside;
        target.live_bytes[k] += entry.size_kind >> 1;
        target.live_count[k] += 1;
    };
    for (auto & shard : global.shards)
    {
        std::lock_guard<std::mutex> shard_lock(shard.mutex);
        for (const auto & entry : shard.table.slots)
        {
            if (entry.ptr) add_live(entry);
        }
    }
    for (const auto & ts : global.threads)
    {
        for (const auto & entry : ts->recent) add_live(entry);
    }

    region_map regions;
    for (const auto & ts : global.threads)
    {
        for (const auto & region : ts->regions) regions.emplace(region.run, std::make_pair(region.outer_run, ts->index));
    }
    for (auto & child : tree.root.children)
    {
        finish(*child, regions);
        cap_at_parent(*child, child->wall);
    }
    return tree;
}

bool has_runs(const merged & m)
{
    if (m.count > 0) return true;
    for (const auto & child : m.children)
    {
        if (has_runs(*child)) return true;
    }
    return false;
}

void flatten(const merged & m, std::int64_t parent, std::int64_t depth, double scale, std::vector<profiler_report_row> & rows)
{
    if (!has_runs(m)) return;
    profiler_report_row row {};
    row.name           = m.name;
    row.parent         = parent;
    row.depth          = depth;
    row.parallel       = m.parallel;
    row.count          = m.count;
    row.time_ns        = ticks_to_ns(m.wall, scale);
    row.cpu_ns         = ticks_to_ns(m.cpu, scale);
    row.self_ns        = ticks_to_ns(m.self, scale);
    row.min_ns         = m.count > 0 ? ticks_to_ns(m.min, scale) : 0;
    row.max_ns         = ticks_to_ns(m.max, scale);
    row.first_ns       = ticks_to_ns(m.first, scale);
    row.threads        = m.threads;
    row.busiest_ns     = ticks_to_ns(m.busiest, scale);
    row.mean_thread_ns = ticks_to_ns(m.mean_thread, scale);
    for (int k = 0; k < memory_kinds; ++k)
    {
        row.alloc_bytes[k] = m.inclusive.alloc_bytes[k];
        row.alloc_count[k] = m.inclusive.alloc_count[k];
        row.free_bytes[k]  = m.inclusive.free_bytes[k];
        row.free_count[k]  = m.inclusive.free_count[k];
        row.peak_bytes[k]  = m.peak[k];
        row.live_bytes[k]  = m.inclusive.live_bytes[k];
        row.live_count[k]  = m.inclusive.live_count[k];
    }
    for (int k = 0; k < transfer_kinds; ++k)
    {
        row.transfer_bytes[k] = m.inclusive.transfer_bytes[k];
        row.transfer_count[k] = m.inclusive.transfer_count[k];
    }
    const std::int64_t index = static_cast<std::int64_t>(rows.size());
    rows.push_back(std::move(row));
    for (const auto & child : m.children) flatten(*child, index, depth + 1, scale, rows);
}

/// Cost of opening and closing one task, measured on a private state that publishes nothing.
double task_cost_ticks()
{
    constexpr int runs = 20000;
    thread_state scratch;
    const char * name         = "calibration";
    const std::uint64_t start = now_ticks();
    for (int i = 0; i < runs; ++i) end_task(scratch, begin_task(scratch, name, false, false), false);
    return double(now_ticks() - start) / runs;
}

/// Cost of recording one allocation or free on the thread that made the block, in ticks: the exported
/// entry points run on a private state the calling thread switches to.
double record_cost_ticks()
{
    constexpr int runs = 20000;
    thread_state scratch;
    std::vector<char> blocks(runs);
    thread_state * const own  = current_thread_state;
    current_thread_state      = &scratch;
    const std::uint64_t start = now_ticks();
    for (int i = 0; i < runs; ++i)
    {
        profiler_record_alloc(&blocks[i], 64, profiler_memory_kind::host);
        profiler_record_free(&blocks[i]);
    }
    const std::uint64_t end = now_ticks();
    current_thread_state    = own;
    return double(end - start) / runs / 2;
}

std::string format_bytes(std::uint64_t bytes)
{
    std::ostringstream out;
    const double value = static_cast<double>(bytes);
    if (bytes >= (1ull << 30))
        out << std::fixed << std::setprecision(2) << value / double(1ull << 30) << " GiB";
    else if (bytes >= (1ull << 20))
        out << std::fixed << std::setprecision(2) << value / double(1ull << 20) << " MiB";
    else if (bytes >= (1ull << 10))
        out << std::fixed << std::setprecision(2) << value / double(1ull << 10) << " KiB";
    else
        out << bytes << " B";
    return out.str();
}

std::string percent(std::uint64_t part, std::uint64_t whole)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << (whole > 0 ? 100.0 * double(part) / double(whole) : 0.0);
    return out.str();
}

std::string memory_line(const char * label, const std::uint64_t alloc_bytes, std::uint64_t alloc_count, std::uint64_t free_bytes,
                        std::uint64_t free_count)
{
    std::ostringstream out;
    out << label << "allocated " << format_bytes(alloc_bytes) << " in " << alloc_count << " blocks, freed " << format_bytes(free_bytes) << " in "
        << free_count << " blocks";
    return out.str();
}

void escape_json(std::ostream & out, const std::string & text)
{
    for (const char c : text)
    {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (static_cast<unsigned char>(c) < 0x20)
            out << ' ';
        else
            out << c;
    }
}

void write_trace(const char * path)
{
    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "ONEDAL PROFILER: cannot open " << path << " for writing the timeline" << std::endl;
        return;
    }
    auto & global      = state();
    const double scale = ns_per_tick();
    std::lock_guard<std::mutex> lock(global.threads_mutex);
    const auto micros = [&](std::uint64_t t) { return double(ticks_to_ns(t > global.origin ? t - global.origin : 0, scale)) / 1e3; };
    static const char * memory_names[memory_kinds] = { "host memory", "device memory" };

    // Where each region started, for the arrows from there to its chunks.
    std::unordered_map<std::uint64_t, std::pair<std::size_t, std::uint64_t> > region_starts;
    for (const auto & ts : global.threads)
    {
        for (const auto & region : ts->regions) region_starts.emplace(region.run, std::make_pair(ts->index, region.start));
    }

    out << std::fixed << std::setprecision(3);
    out << "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n";
    out << "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":0,\"tid\":0,\"args\":{\"name\":\"oneDAL\"}}";
    std::uint64_t flow_id = 0;
    for (const auto & ts : global.threads)
    {
        if (ts->events.empty() && ts->samples.empty()) continue;
        out << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":" << ts->index << ",\"args\":{\"name\":\"oneDAL thread " << ts->index
            << "\"}}";
        for (const auto & e : ts->events)
        {
            // A task with arguments is one slice name, with the arguments in its details.
            const std::string & name = e.task->name;
            const std::size_t args   = name.back() == ']' ? name.find(" [") : std::string::npos;
            out << ",\n{\"name\":\"";
            escape_json(out, args == std::string::npos ? name : name.substr(0, args));
            out << "\",\"cat\":\"" << (e.parallel ? "parallel" : "task") << "\",\"ph\":\"X\",\"pid\":0,\"tid\":" << ts->index
                << ",\"ts\":" << micros(e.start) << ",\"dur\":" << double(ticks_to_ns(e.end - e.start, scale)) / 1e3;
            if (args != std::string::npos)
            {
                out << ",\"args\":{\"args\":\"";
                escape_json(out, name.substr(args + 2, name.size() - args - 3));
                out << "\"}";
            }
            out << "}";
            const auto region = e.run ? region_starts.find(e.run) : region_starts.end();
            if (region != region_starts.end())
            {
                // An arrow from the task that started the region to this chunk.
                ++flow_id;
                out << ",\n{\"name\":\"region\",\"cat\":\"region\",\"ph\":\"s\",\"id\":" << flow_id << ",\"pid\":0,\"tid\":" << region->second.first
                    << ",\"ts\":" << micros(region->second.second) << "}";
                out << ",\n{\"name\":\"region\",\"cat\":\"region\",\"ph\":\"f\",\"bp\":\"e\",\"id\":" << flow_id << ",\"pid\":0,\"tid\":" << ts->index
                    << ",\"ts\":" << micros(e.start) << "}";
            }
        }
        for (const auto & sample : ts->samples)
        {
            out << ",\n{\"name\":\"" << memory_names[sample.kind] << "\",\"ph\":\"C\",\"pid\":0,\"ts\":" << micros(sample.time)
                << ",\"args\":{\"MiB\":" << double(sample.live) / double(1 << 20) << "}}";
        }
    }
    out << "\n]}\n";
    if (global.trace_dropped.load() > 0)
    {
        std::cerr << "ONEDAL PROFILER: the timeline is truncated, " << global.trace_dropped.load()
                  << " events were dropped; raise ONEDAL_PROFILER_TRACE_LIMIT to keep them" << std::endl;
    }
}

// Whether a report has been printed since the program started; in per-call mode the report at exit
// is skipped when no call ran after the last one.
bool report_rows_printed = false;

void report_call()
{
    std::ostringstream report;
    profiler_print_report(report);
    profiler_log(nullptr, report.str());
    profiler_clear();
}

void report_at_exit()
{
#ifdef DAAL_PROFILER_EXCEPTIONS_ENABLED
    try
    {
#endif
        if (is_analyzer_enabled() && (!is_report_per_call() || !report_rows_printed || !profiler_report().empty()))
        {
            std::ostringstream report;
            profiler_print_report(report);
            profiler_log(nullptr, report.str());
        }
        if (const char * path = profiler_trace_path()) write_trace(path);
#ifdef DAAL_PROFILER_EXCEPTIONS_ENABLED
    }
    catch (std::exception & e)
    {
        std::cerr << "ONEDAL PROFILER: unexpected error while writing the report: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "ONEDAL PROFILER: unexpected error while writing the report" << std::endl;
    }
#endif
}

/// Live memory of `kind`, including the changes threads have not added to the process-wide count yet.
std::int64_t live_now(int kind)
{
    auto & global      = state();
    std::int64_t value = global.live[kind].value.load();
    std::lock_guard<std::mutex> lock(global.threads_mutex);
    for (const auto & ts : global.threads) value += ts->pending[kind];
    return value;
}

void print_memory_summary(std::ostream & out, const merged_tree & tree)
{
    auto & global = state();
    if (is_memory_tracking_enabled())
    {
        own_stats all = tree.outside;
        for (const auto & child : tree.root.children) all.add(child->inclusive);
        static const char * labels[memory_kinds] = { "  host:   ", "  device: " };
        out << "\nMemory\n";
        for (int k = 0; k < memory_kinds; ++k)
        {
            if (all.alloc_count[k] == 0 && all.free_count[k] == 0) continue;
            out << memory_line(labels[k], all.alloc_bytes[k], all.alloc_count[k], all.free_bytes[k], all.free_count[k]) << ", peak "
                << format_bytes(positive_part(global.process_peak[k].value.load())) << ", live now " << format_bytes(positive_part(live_now(k)))
                << '\n';
            if (tree.outside.alloc_count[k] > 0)
            {
                out << "          of which outside any task: " << format_bytes(tree.outside.alloc_bytes[k]) << " in " << tree.outside.alloc_count[k]
                    << " blocks\n";
            }
        }
        if (all.transfer_count[0] > 0 || all.transfer_count[1] > 0)
        {
            out << "  copies: " << format_bytes(all.transfer_bytes[0]) << " to the device in " << all.transfer_count[0] << " copies, "
                << format_bytes(all.transfer_bytes[1]) << " to the host in " << all.transfer_count[1] << " copies\n";
        }
        const std::size_t unmatched = global.orphan_count.load();
        if (unmatched > 0) out << "  " << unmatched << " frees of blocks that were not recorded were ignored\n";
    }
}

/// Writes one line per call path, "outer;inner;task <self time in ns>", the input format of flame
/// graph tools such as speedscope and flamegraph.pl.
void print_folded(std::ostream & out, const std::vector<profiler_report_row> & rows)
{
    std::vector<std::string> paths(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        std::string name = rows[i].name;
        std::replace(name.begin(), name.end(), ';', ',');
        paths[i] = rows[i].parent >= 0 ? paths[rows[i].parent] + ';' + name : name;
        if (rows[i].self_ns > 0) out << paths[i] << ' ' << rows[i].self_ns << '\n';
    }
}

void print_json(std::ostream & out, const std::vector<profiler_report_row> & rows)
{
    const auto pair = [&](const char * key, const std::uint64_t (&values)[2]) {
        out << ",\"" << key << "\":[" << values[0] << ',' << values[1] << ']';
    };
    out << "{\"rows\":[";
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        const auto & row = rows[i];
        out << (i ? ",\n" : "\n") << "{\"name\":\"";
        escape_json(out, row.name);
        out << "\",\"parent\":" << row.parent << ",\"depth\":" << row.depth << ",\"parallel\":" << (row.parallel ? "true" : "false")
            << ",\"count\":" << row.count << ",\"time_ns\":" << row.time_ns << ",\"cpu_ns\":" << row.cpu_ns << ",\"self_ns\":" << row.self_ns
            << ",\"min_ns\":" << row.min_ns << ",\"max_ns\":" << row.max_ns << ",\"first_ns\":" << row.first_ns << ",\"threads\":" << row.threads
            << ",\"busiest_ns\":" << row.busiest_ns << ",\"mean_thread_ns\":" << row.mean_thread_ns;
        pair("alloc_bytes", row.alloc_bytes);
        pair("alloc_count", row.alloc_count);
        pair("free_bytes", row.free_bytes);
        pair("free_count", row.free_count);
        pair("peak_bytes", row.peak_bytes);
        pair("live_bytes", row.live_bytes);
        pair("live_count", row.live_count);
        pair("transfer_bytes", row.transfer_bytes);
        pair("transfer_count", row.transfer_count);
        out << '}';
    }
    out << "\n]}\n";
}

/// The share of the profiled time below which the text report folds a row into its parent's
/// "(N more ...)" row, from ONEDAL_PROFILER_MIN_PERCENT.
double min_percent()
{
    static const double value = [] {
        const char * text = std::getenv("ONEDAL_PROFILER_MIN_PERCENT");
        if (!text) return 0.0;
        char * end         = nullptr;
        const double value = std::strtod(text, &end);
        return (end != text && value > 0.0) ? value : 0.0;
    }();
    return value;
}

/// `rows` without the rows shorter than `threshold`, each parent's hidden children summed into one row.
std::vector<profiler_report_row> fold_small_rows(const std::vector<profiler_report_row> & rows, std::uint64_t threshold)
{
    std::vector<std::vector<std::size_t> > children(rows.size());
    std::vector<std::size_t> roots;
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        (rows[i].parent >= 0 ? children[rows[i].parent] : roots).push_back(i);
    }
    std::vector<profiler_report_row> kept;
    // Appends the rows of `group`, under the kept row `parent`.
    std::function<void(const std::vector<std::size_t> &, std::int64_t, std::int64_t)> add = [&](const std::vector<std::size_t> & group,
                                                                                                std::int64_t parent, std::int64_t depth) {
        profiler_report_row rest {};
        std::uint64_t hidden = 0;
        for (const std::size_t i : group)
        {
            if (rows[i].time_ns < threshold)
            {
                ++hidden;
                rest.count += rows[i].count;
                rest.time_ns += rows[i].time_ns;
                rest.cpu_ns += rows[i].cpu_ns;
                rest.self_ns += rows[i].time_ns;
                rest.max_ns = std::max(rest.max_ns, rows[i].max_ns);
                rest.parallel |= rows[i].parallel;
                for (int k = 0; k < memory_kinds; ++k)
                {
                    rest.alloc_bytes[k] += rows[i].alloc_bytes[k];
                    rest.peak_bytes[k] = std::max(rest.peak_bytes[k], rows[i].peak_bytes[k]);
                    rest.live_bytes[k] += rows[i].live_bytes[k];
                    rest.live_count[k] += rows[i].live_count[k];
                }
                for (int k = 0; k < transfer_kinds; ++k) rest.transfer_bytes[k] += rows[i].transfer_bytes[k];
                continue;
            }
            profiler_report_row row = rows[i];
            row.parent              = parent;
            row.depth               = depth;
            kept.push_back(row);
            add(children[i], static_cast<std::int64_t>(kept.size() - 1), depth + 1);
        }
        if (hidden > 0)
        {
            rest.name     = "(" + std::to_string(hidden) + (hidden == 1 ? " more task" : " more tasks") + " under the report threshold)";
            rest.parent   = parent;
            rest.depth    = depth;
            rest.threads  = 1;
            rest.parallel = false;
            kept.push_back(rest);
        }
    };
    add(roots, -1, 0);
    return kept;
}

/// Checks the recorded figures for patterns that usually mean something is wrong or slow, and
/// writes one line per finding, worst first.
///
/// @param[in] rows         The task tree, as profiler_report() returns it
/// @param[in] total        Profiled wall time in ns
/// @param[in] cost_share   The profiler's own cost as a share of the profiled CPU time
void print_health(std::ostream & out, const std::vector<profiler_report_row> & rows, std::uint64_t total, double cost_share)
{
    struct finding
    {
        int severity; // 2 warning, 1 check, 0 note
        std::string text;
    };
    std::vector<finding> findings;
    const auto share    = [&](std::uint64_t t) { return total > 0 ? double(t) / double(total) : 0.0; };
    const auto describe = [&](std::size_t i) {
        std::string path = rows[i].name;
        for (std::int64_t p = rows[i].parent, hops = 0; p >= 0 && hops < 2; p = rows[p].parent, ++hops) path = rows[p].name + " > " + path;
        return path;
    };
    const auto is_task = [&](std::size_t i) { return rows[i].name.front() != '('; };
    // A first run at least ten times as long as the average of the others, and at least 10 ms.
    const auto slow_first = [&](std::size_t i) {
        const auto & row = rows[i];
        if (row.count < 3 || row.first_ns < 10000000 || !is_task(i)) return false;
        const std::uint64_t rest = (row.cpu_ns > row.first_ns ? row.cpu_ns - row.first_ns : 0) / (row.count - 1);
        return row.first_ns >= 10 * std::max<std::uint64_t>(rest, 1);
    };

    std::uint64_t threads = 1;
    for (const auto & row : rows) threads = std::max(threads, row.threads);

    static const char * kinds[memory_kinds] = { "host", "device" };
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        const auto & row = rows[i];
        for (int k = 0; k < memory_kinds; ++k)
        {
            // The innermost row that holds the memory: a child holding all of it is reported instead.
            if (row.live_bytes[k] == 0) continue;
            bool child_holds_all = false;
            for (std::size_t j = i + 1; j < rows.size() && rows[j].depth > row.depth; ++j)
            {
                child_holds_all |= rows[j].parent == static_cast<std::int64_t>(i) && rows[j].live_bytes[k] == row.live_bytes[k];
            }
            if (child_holds_all) continue;
            std::ostringstream text;
            text << format_bytes(row.live_bytes[k]) << " of " << kinds[k] << " memory in " << row.live_count[k]
                 << (row.live_count[k] == 1 ? " block" : " blocks") << " allocated in " << describe(i)
                 << " is still allocated: a leak, unless it is a result the caller still holds";
            findings.push_back({ 2, text.str() });
        }
        if (!is_task(i)) continue;
        const std::uint64_t per_run = row.count > 0 ? row.cpu_ns / row.count : 0;
        if (row.parallel && row.threads > 1 && row.mean_thread_ns > 0 && share(row.time_ns) >= 0.05)
        {
            const double imbalance = double(row.busiest_ns) / double(row.mean_thread_ns);
            if (imbalance >= 2.0)
            {
                std::ostringstream text;
                text << std::fixed << std::setprecision(1) << describe(i) << ": load imbalance " << imbalance
                     << ", the busiest thread works that many times as long as the average one (" << percent(row.time_ns, total) << "% of the time)";
                findings.push_back({ 2, text.str() });
            }
        }
        if (row.parallel && row.count >= 1000 && per_run < 2000 && share(row.cpu_ns / std::max<std::uint64_t>(row.threads, 1)) >= 0.01)
        {
            std::ostringstream text;
            text << describe(i) << ": " << row.count << " runs of " << format_time(per_run)
                 << " on average; work this fine-grained spends much of its time scheduling, a larger grain size helps";
            findings.push_back({ 1, text.str() });
        }
        if (row.name.rfind("parallel loop: ", 0) == 0 && threads >= 8 && row.threads > 0 && row.threads * 4 <= threads && share(row.time_ns) >= 0.05)
        {
            std::ostringstream text;
            text << describe(i) << ": ran on at most " << row.threads << " of " << threads << " threads for " << percent(row.time_ns, total)
                 << "% of the time; the loop has too few iterations or too large a grain to use the rest";
            findings.push_back({ 1, text.str() });
        }
        if (slow_first(i))
        {
            bool child_explains = false;
            for (std::size_t j = i + 1; j < rows.size() && rows[j].depth > row.depth; ++j)
            {
                child_explains |= rows[j].parent == static_cast<std::int64_t>(i) && slow_first(j) && 10 * rows[j].first_ns >= 9 * row.first_ns;
            }
            const std::uint64_t rest = (row.cpu_ns > row.first_ns ? row.cpu_ns - row.first_ns : 0) / (row.count - 1);
            if (!child_explains)
            {
                std::ostringstream text;
                text << describe(i) << ": the first run took " << format_time(row.first_ns) << ", the others " << format_time(rest)
                     << " on average; on a GPU that is kernel compilation, which a warm-up call or a kernel cache avoids";
                findings.push_back({ 0, text.str() });
            }
        }
        const std::uint64_t allocs = row.alloc_count[0] + row.alloc_count[1];
        if (row.count > 0 && allocs / row.count >= 1000 && row.depth > 0)
        {
            std::ostringstream text;
            text << describe(i) << ": " << allocs / row.count << " allocations per run; allocating once outside the loop is usually faster";
            findings.push_back({ 1, text.str() });
        }
        if (row.transfer_count[1] >= 100 && row.count > 0 && row.transfer_count[1] / row.count >= 10 && row.depth == 0)
        {
            std::ostringstream text;
            text << describe(i) << ": " << row.transfer_count[1] / row.count << " copies to the host per call ("
                 << format_bytes(row.transfer_bytes[1] / row.count) << "); each one waits for the device";
            findings.push_back({ 1, text.str() });
        }
    }
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        const auto & row = rows[i];
        if (row.name != waiting_row_name || row.parent < 0) continue;
        const auto & parent = rows[row.parent];
        if (parent.time_ns > 0 && double(row.time_ns) >= 0.25 * double(parent.time_ns) && share(row.time_ns) >= 0.05)
        {
            std::ostringstream text;
            text << describe(static_cast<std::size_t>(row.parent)) << ": " << percent(row.time_ns, parent.time_ns)
                 << "% of it is spent in parallel regions outside any task, mostly waiting for the slowest thread or in loops without a known call "
                    "site";
            findings.push_back({ 0, text.str() });
        }
    }
    if (cost_share >= 0.05)
    {
        std::ostringstream text;
        text << "the profiler itself took about " << std::fixed << std::setprecision(1) << 100.0 * cost_share
             << "% of the CPU time; ONEDAL_PROFILER_MEMORY=0 removes the part spent on allocations";
        findings.push_back({ 0, text.str() });
    }

    std::stable_sort(findings.begin(), findings.end(), [](const finding & a, const finding & b) { return a.severity > b.severity; });
    static const char * labels[] = { "note   ", "check  ", "warning" };
    out << "\nHealth\n";
    if (findings.empty()) out << "  nothing unusual found\n";
    for (const auto & f : findings) out << "  " << labels[f.severity] << "  " << f.text << '\n';
}

} // namespace

DAAL_EXPORT void profiler_log(const char * once_key, const std::string & text)
{
    auto & s = sink();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (once_key && !s.every_call)
    {
        if (!s.seen.insert(once_key).second) return;
        if (!s.noted)
        {
            s.noted = true;
            s.out() << "ONEDAL PROFILER: each call site is logged once; set ONEDAL_PROFILER_OUTPUT=<file> to log every call\n";
        }
    }
    s.out() << text;
    s.out().flush();
}

DAAL_EXPORT int profiler_begin_task(const char * task_name, bool threading)
{
    return begin_task(this_thread(), task_name, threading, true);
}

DAAL_EXPORT int profiler_begin_task_with_args(const char * task_name, const std::string & args)
{
    return begin_task(this_thread(), task_name, false, true, &args);
}

DAAL_EXPORT void profiler_set_region_site(const profiler_site * site)
{
    pending_site = site;
}

DAAL_EXPORT const profiler_region_hooks & profiler_threading_hooks()
{
    return region_hooks;
}

DAAL_EXPORT void profiler_end_task(int depth)
{
    end_task(this_thread(), depth, true);
}

DAAL_EXPORT void profiler_record_alloc(const void * ptr, std::size_t bytes, profiler_memory_kind kind)
{
    if (!ptr) return;
    const int k = static_cast<int>(kind);
    auto & ts   = this_thread();
    settle_orphans(ts);
    ++ts.memory_records;
    if (ts.recent.size() == max_recent_blocks) spill_recent(ts, 0, max_recent_blocks / 2);
    ts.recent.push_back({ ptr, (static_cast<std::uint64_t>(bytes) << 1) | static_cast<std::uint64_t>(k), charge_owner(ts) });
    own_stats & own = charge_target(ts);
    own.alloc_bytes[k] += bytes;
    own.alloc_count[k] += 1;
    update_live(ts, k, static_cast<std::int64_t>(bytes));
}

DAAL_EXPORT void profiler_record_free(const void * ptr)
{
    if (!ptr) return;
    auto & ts = this_thread();
    ++ts.memory_records;
    settle_orphans(ts);
    std::uint64_t size_kind = 0;
    bool found              = false;
    for (std::size_t i = ts.recent.size(); i-- > 0;)
    {
        if (ts.recent[i].ptr == ptr)
        {
            size_kind = ts.recent[i].size_kind;
            erase_recent(ts, i, i + 1);
            found = true;
            break;
        }
    }
    if (!found)
    {
        auto & shard = shard_for(ptr);
        std::lock_guard<std::mutex> lock(shard.mutex);
        if (!shard.table.erase(ptr, size_kind))
        {
            // Still in the table of the thread that allocated it, or never recorded.
            auto & global = state();
            std::lock_guard<std::mutex> orphans_lock(global.orphans_mutex);
            if (global.orphans.insert(ptr).second) global.orphan_count.fetch_add(1, std::memory_order_release);
            return;
        }
    }
    charge_free(ts, size_kind);
}

DAAL_EXPORT void profiler_record_transfer(std::size_t bytes, profiler_transfer_kind kind)
{
    const int k     = static_cast<int>(kind);
    own_stats & own = charge_target(this_thread());
    own.transfer_bytes[k] += bytes;
    own.transfer_count[k] += 1;
}

DAAL_EXPORT std::vector<profiler_report_row> profiler_report()
{
    merged_tree tree = build_tree();
    std::vector<profiler_report_row> rows;
    for (const auto & child : tree.root.children) flatten(*child, -1, 0, ns_per_tick(), rows);
    return rows;
}

DAAL_EXPORT void profiler_print_report(std::ostream & out)
{
    profiler_print_report(out, report_format(), min_percent());
}

DAAL_EXPORT void profiler_print_report(std::ostream & out, profiler_report_format format, double min_percent)
{
    const merged_tree tree = build_tree();
    report_rows_printed    = true;
    std::vector<profiler_report_row> rows;
    for (const auto & child : tree.root.children) flatten(*child, -1, 0, ns_per_tick(), rows);
    auto & global = state();

    std::uint64_t total = 0;
    for (const auto & row : rows)
    {
        if (row.depth == 0) total += row.time_ns;
    }
    bool any_memory[memory_kinds] = {};
    bool any_live[memory_kinds]   = {};
    bool any_transfer             = false;
    for (const auto & row : rows)
    {
        for (int k = 0; k < memory_kinds; ++k)
        {
            any_memory[k] |= row.alloc_count[k] > 0 || row.free_count[k] > 0;
            any_live[k] |= row.live_count[k] > 0;
        }
        any_transfer |= row.transfer_count[0] > 0 || row.transfer_count[1] > 0;
    }

    std::uint64_t total_cpu = 0;
    for (const auto & row : rows)
    {
        if (row.depth == 0) total_cpu += row.cpu_ns;
    }
    const double cost        = task_cost_ticks() * ns_per_tick();
    const double record_cost = tree.memory_records > 0 ? record_cost_ticks() * ns_per_tick() : 0.0;
    const auto overhead      = static_cast<std::uint64_t>(cost * double(tree.task_runs) + record_cost * double(tree.memory_records));

    if (format == profiler_report_format::folded)
    {
        print_folded(out, rows);
        out << std::flush;
        return;
    }
    if (format == profiler_report_format::json)
    {
        print_json(out, rows);
        out << std::flush;
        return;
    }
    if (format == profiler_report_format::health)
    {
        out << "oneDAL profiler health\n";
        print_health(out, rows, total, double(overhead) / double(std::max<std::uint64_t>(total_cpu, 1)));
        print_memory_summary(out, tree);
        out << std::flush;
        return;
    }
    // The health checks see every row, the tree only the rows above the threshold.
    const std::vector<profiler_report_row> all_rows = rows;
    if (min_percent > 0.0) rows = fold_small_rows(rows, static_cast<std::uint64_t>(double(total) * min_percent / 100.0));

    const bool tree_only = format != profiler_report_format::text;
    out << "oneDAL profiler report\n";
    if (rows.empty())
    {
        out << "  no profiled task ran\n";
        if (!tree_only) print_memory_summary(out, tree);
        out << std::flush;
        return;
    }
    if (!tree_only)
    {
        out << "  profiled wall time " << format_time(total) << " on " << tree.threads << " threads, " << tree.task_runs << " task runs\n";
        out << "  profiler cost about " << format_time(static_cast<std::uint64_t>(cost)) << " per task run";
        if (tree.memory_records > 0)
        {
            out << " and " << format_time(static_cast<std::uint64_t>(record_cost)) << " per allocation or free (" << tree.memory_records
                << " recorded; more when another thread frees the block)";
        }
        out << ", " << format_time(overhead) << " of CPU time in total (" << percent(overhead, total_cpu) << "% of the profiled CPU time)\n";
        out << "  time: wall time, a parallel task taking its busiest thread in each run; self: time not in child tasks\n";
        if (any_memory[0] || any_memory[1])
            out << "  alloc: bytes allocated in the task and its children; peak: highest live memory above the task's start\n";
        if (any_live[0] || any_live[1]) out << "  live: allocated in the task and its children and not freed when this report was written\n";
        out << '\n';
    }

    // Columns, then the tree.
    std::ostringstream header;
    header << std::setw(10) << "time" << std::setw(7) << "%" << std::setw(10) << "self" << std::setw(9) << "calls" << std::setw(10) << "avg"
           << std::setw(10) << "max" << std::setw(10) << "first" << std::setw(8) << "thr";
    if (any_memory[0]) header << std::setw(11) << "host alloc" << std::setw(11) << "host peak";
    if (any_live[0]) header << std::setw(11) << "host live";
    if (any_memory[1]) header << std::setw(11) << "dev alloc" << std::setw(11) << "dev peak";
    if (any_live[1]) header << std::setw(11) << "dev live";
    if (any_transfer) header << std::setw(11) << "to dev" << std::setw(11) << "to host";
    header << "  task";
    out << header.str() << '\n';

    std::vector<bool> last_child(rows.size(), false);
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        last_child[i] = true;
        for (std::size_t j = i + 1; j < rows.size(); ++j)
        {
            if (rows[j].depth < rows[i].depth) break;
            if (rows[j].depth == rows[i].depth && rows[j].parent == rows[i].parent)
            {
                last_child[i] = false;
                break;
            }
        }
    }

    std::vector<std::string> prefixes(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        const auto & row = rows[i];
        std::string prefix;
        if (row.parent >= 0)
        {
            const auto & parent_prefix = prefixes[row.parent];
            prefix                     = parent_prefix;
            if (!prefix.empty())
            {
                // Continue the parent's branch line only if the parent has siblings below it.
                prefix.replace(prefix.size() - 4, 4, last_child[row.parent] ? "    " : "|   ");
            }
            prefix += last_child[i] ? "`-- " : "|-- ";
        }
        prefixes[i] = prefix;

        out << std::setw(10) << format_time(row.time_ns) << std::setw(7) << percent(row.time_ns, total) << std::setw(10) << format_time(row.self_ns)
            << std::setw(9) << row.count << std::setw(10) << format_time(row.count > 0 ? row.cpu_ns / row.count : 0) << std::setw(10)
            << format_time(row.max_ns) << std::setw(10) << (row.name.front() == '(' ? std::string("-") : format_time(row.first_ns)) << std::setw(8)
            << row.threads;
        if (any_memory[0]) out << std::setw(11) << format_bytes(row.alloc_bytes[0]) << std::setw(11) << format_bytes(row.peak_bytes[0]);
        if (any_live[0]) out << std::setw(11) << format_bytes(row.live_bytes[0]);
        if (any_memory[1]) out << std::setw(11) << format_bytes(row.alloc_bytes[1]) << std::setw(11) << format_bytes(row.peak_bytes[1]);
        if (any_live[1]) out << std::setw(11) << format_bytes(row.live_bytes[1]);
        if (any_transfer) out << std::setw(11) << format_bytes(row.transfer_bytes[0]) << std::setw(11) << format_bytes(row.transfer_bytes[1]);
        out << "  " << prefix << row.name;
        if (row.parallel && row.threads > 1 && row.mean_thread_ns > 0)
        {
            // Busiest thread over the average thread, per run of the region: 1.00 is perfectly balanced.
            const double imbalance = double(row.busiest_ns) / double(row.mean_thread_ns);
            out << "  [parallel, cpu " << format_time(row.cpu_ns) << ", imbalance " << std::fixed << std::setprecision(2) << imbalance << "]";
        }
        out << '\n';
    }

    if (tree_only)
    {
        out << std::flush;
        return;
    }

    // Hotspots: self time summed over every place a task name appears.
    struct hotspot
    {
        std::uint64_t self  = 0;
        std::uint64_t count = 0;
    };
    std::unordered_map<std::string, hotspot> by_name;
    for (const auto & row : rows)
    {
        by_name[row.name].self += row.self_ns;
        by_name[row.name].count += row.count;
    }
    std::vector<std::pair<std::string, hotspot> > hotspots(by_name.begin(), by_name.end());
    std::sort(hotspots.begin(), hotspots.end(), [](const auto & a, const auto & b) { return a.second.self > b.second.self; });
    out << "\nHotspots by self time\n";
    out << std::setw(10) << "self" << std::setw(7) << "%" << std::setw(9) << "calls" << "  task\n";
    for (std::size_t i = 0; i < hotspots.size() && i < 15; ++i)
    {
        out << std::setw(10) << format_time(hotspots[i].second.self) << std::setw(7) << percent(hotspots[i].second.self, total) << std::setw(9)
            << hotspots[i].second.count << "  " << hotspots[i].first << '\n';
    }

    print_health(out, all_rows, total, double(overhead) / double(std::max<std::uint64_t>(total_cpu, 1)));
    print_memory_summary(out, tree);
    out << std::flush;
}

DAAL_EXPORT void profiler_clear()
{
    auto & global = state();
    std::lock_guard<std::mutex> lock(global.threads_mutex);
    for (auto & ts : global.threads)
    {
        for (node & n : ts->nodes)
        {
            n.count = n.total = n.self_total = n.sequential_total = n.unsplit_total = 0;
            n.wait_total = n.wait_count = n.wait_max = 0;
            n.min = n.wait_min = UINT64_MAX;
            n.max              = 0;
            n.first            = 0;
            n.first_start      = UINT64_MAX;
            n.own              = own_stats();
            n.runs.clear();
            for (int k = 0; k < memory_kinds; ++k) n.peak[k] = 0;
        }
        ts->regions.clear();
        ts->anchored.clear();
        ts->charged_anchor = nullptr;
        ts->charged        = nullptr;
        ts->outside        = own_stats();
        ts->memory_records = 0;
    }
    for (int k = 0; k < memory_kinds; ++k)
    {
        global.region_peak[k].value  = global.live[k].value.load();
        global.process_peak[k].value = global.live[k].value.load();
    }
}

DAAL_EXPORT void profiler_reset()
{
    auto & global = state();
    std::lock_guard<std::mutex> lock(global.threads_mutex);
    // Live blocks stay recorded, but the tasks they were charged to are dropped.
    for (auto & shard : global.shards)
    {
        std::lock_guard<std::mutex> shard_lock(shard.mutex);
        for (auto & entry : shard.table.slots) entry.owner = nullptr;
    }
    for (auto & ts : global.threads)
    {
        for (auto & entry : ts->recent) entry.owner = nullptr;
    }
    for (auto & ts : global.threads)
    {
        ts->root.children.clear();
        ts->nodes.clear();
        ts->stack.clear();
        ts->chunks.clear();
        ts->started.clear();
        ts->anchored.clear();
        ts->outside = own_stats();
        ts->events.clear();
        ts->samples.clear();
    }
    for (auto & ts : global.threads)
    {
        ts->regions.clear();
        ts->charged_anchor = nullptr;
        ts->charged        = nullptr;
        ts->trace_slots    = 0;
        ts->memory_records = 0;
    }
    for (int k = 0; k < memory_kinds; ++k)
    {
        global.region_peak[k].value  = global.live[k].value.load();
        global.process_peak[k].value = global.live[k].value.load();
    }
    global.trace_budget  = global_state::trace_limit();
    global.trace_dropped = 0;
}

} // namespace internal
} // namespace daal

extern "C" DAAL_EXPORT void _onedal_profiler_report(const char * path, int clear)
{
#ifdef DAAL_PROFILER_EXCEPTIONS_ENABLED
    try
    {
#endif
        std::ostringstream report;
        daal::internal::profiler_print_report(report);
        if (path && *path)
        {
            std::ofstream out(path, std::ios::out | std::ios::app);
            out << report.str();
        }
        else
        {
            daal::internal::profiler_log(nullptr, report.str());
        }
        if (clear) daal::internal::profiler_clear();
#ifdef DAAL_PROFILER_EXCEPTIONS_ENABLED
    }
    catch (std::exception & e)
    {
        std::cerr << "ONEDAL PROFILER: unexpected error while writing the report: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "ONEDAL PROFILER: unexpected error while writing the report" << std::endl;
    }
#endif
}
