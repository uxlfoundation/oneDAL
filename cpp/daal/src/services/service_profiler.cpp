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
//  Kernel profiler state, shared by every library of the process.
//
//  Each thread records into its own tree of call paths, without a lock. A task opened on a thread
//  goes under the task that thread is in; on a thread running a chunk of a parallel region it goes
//  under the task that started the region, which the threading layer reports through
//  profiler_region_hooks. The report merges the threads' trees by call path.
//
//  Allocations made through oneDAL's allocators are charged to the task open on the allocating
//  thread. Live memory is counted process-wide, so the peak of a sequential task also includes
//  what other threads of the process allocate at the same time.
//--
*/

#include "src/services/service_profiler.h"
#include "src/threading/threading.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace daal
{
namespace internal
{
namespace
{
constexpr int memory_kinds = 2;

inline std::uint64_t now_ns()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

/// A call path in one thread's tree. Only the owning thread writes to it; its parent may belong
/// to another thread.
struct node
{
    const node * parent;
    const char * name;
    std::uint64_t seq; // creation order, which the report keeps
    std::uint64_t count    = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t max_ns   = 0;
    bool threading         = false; // ran as the body of a parallel region
    bool on_queue          = false; // timed on a SYCL queue
    bool traced            = false; // the tracer has written this path once
    // Written by any thread running a chunk under this task, hence atomic.
    std::atomic<std::uint64_t> alloc_bytes[memory_kinds] = {};
    std::atomic<std::uint64_t> alloc_count[memory_kinds] = {};
    std::uint64_t peak_bytes[memory_kinds]               = {};
};

/// An open task or, with task == nullptr, a chunk of a parallel region whose tasks go under anchor.
struct frame
{
    const node * anchor;
    node * task;
    std::uint64_t start;
    std::uint64_t paused                     = 0; // time of chunks of other regions this thread ran while in the task
    std::size_t pause_from                   = 0; // for an interrupting chunk: the first frame its time is taken out of
    bool interrupts                          = false;
    bool parallel                            = false; // inside a parallel region, where no peak is kept
    std::int64_t live_at_start[memory_kinds] = {};
    std::int64_t outer_peak[memory_kinds]    = {}; // the enclosing sequential task's peak so far
};

struct node_key
{
    const node * parent;
    const char * name;
    bool operator==(const node_key & other) const { return parent == other.parent && name == other.name; }
};

struct node_key_hash
{
    std::size_t operator()(const node_key & key) const { return std::hash<const void *>()(key.parent) * 31u ^ std::hash<const void *>()(key.name); }
};

struct thread_state
{
    std::vector<frame> stack;
    std::unordered_map<node_key, node *, node_key_hash> index;
    std::vector<std::unique_ptr<node> > nodes;
};

/// A recorded block that is not freed yet.
struct block
{
    std::size_t size;
    int kind;
    const node * owner; // the task it was charged to, or nullptr
};

struct block_shard
{
    std::mutex mutex;
    std::unordered_map<const void *, block> blocks;
};

constexpr std::size_t block_shards = 64;

struct global_state
{
    std::mutex threads_mutex;
    std::vector<thread_state *> threads; // never freed: a thread that exits keeps its tree in the report
    std::atomic<std::uint64_t> next_seq { 0 };
    block_shard shards[block_shards];
    std::atomic<std::int64_t> live[memory_kinds]         = {};
    std::atomic<std::int64_t> region_peak[memory_kinds]  = {}; // since the innermost sequential task started
    std::atomic<std::int64_t> process_peak[memory_kinds] = {};
};

block_shard & shard_for(global_state & global, const void * ptr)
{
    const std::size_t h = std::hash<const void *>()(ptr) >> 6;
    return global.shards[h % block_shards];
}

void atomic_max(std::atomic<std::int64_t> & target, std::int64_t value)
{
    std::int64_t current = target.load(std::memory_order_relaxed);
    while (current < value && !target.compare_exchange_weak(current, value, std::memory_order_relaxed))
    {
    }
}

void report_at_exit();
const void * region_capture();
void region_enter(const void * region);
void region_leave();

const profiler_region_hooks region_hooks = { region_capture, region_enter, region_leave };

// Never destroyed: tasks can still close from static destructors that run after the report.
global_state & state()
{
    static global_state * instance = new global_state();
    return *instance;
}

void register_exit_report()
{
    static std::once_flag once;
    std::call_once(once, [] {
        if (is_analyzer_enabled()) std::atexit(report_at_exit);
    });
}

thread_local thread_state * current_thread_state = nullptr;

thread_state & this_thread()
{
    if (!current_thread_state)
    {
        register_exit_report();
        // Regions started from now on report their chunks. Not done at load time: on Windows that
        // would load the threading library from inside DllMain.
        static std::once_flag hooks_once;
        std::call_once(hooks_once, [] {
            if (is_profiler_enabled()) _daal_threader_set_profiler_hooks(&region_hooks);
        });
        thread_state * created = new thread_state();
        global_state & global  = state();
        std::lock_guard<std::mutex> lock(global.threads_mutex);
        global.threads.push_back(created);
        current_thread_state = created;
    }
    return *current_thread_state;
}

std::mutex & log_mutex()
{
    static std::mutex m;
    return m;
}

std::string format_time(std::uint64_t time_ns)
{
    std::ostringstream out;
    const double time = static_cast<double>(time_ns);
    if (time_ns == 0)
        out << "0.00s";
    else if (time > 1e9)
        out << std::fixed << std::setprecision(2) << time / 1e9 << "s";
    else if (time > 1e6)
        out << std::fixed << std::setprecision(2) << time / 1e6 << "ms";
    else if (time > 1e3)
        out << std::fixed << std::setprecision(2) << time / 1e3 << "us";
    else
        out << time_ns << "ns";
    return out.str();
}

std::string format_bytes(std::uint64_t bytes)
{
    std::ostringstream out;
    const double value = static_cast<double>(bytes);
    if (bytes >= (1ull << 30))
        out << std::fixed << std::setprecision(2) << value / (1ull << 30) << "GB";
    else if (bytes >= (1ull << 20))
        out << std::fixed << std::setprecision(2) << value / (1ull << 20) << "MB";
    else if (bytes >= (1ull << 10))
        out << std::fixed << std::setprecision(2) << value / (1ull << 10) << "KB";
    else
        out << bytes << "B";
    return out.str();
}

/// The task an allocation on this thread is charged to now, or nullptr.
node * charge_target(const thread_state & ts)
{
    if (ts.stack.empty()) return nullptr;
    const frame & f = ts.stack.back();
    // A chunk with no task open charges the task that started the region, which another thread owns.
    return f.task ? f.task : const_cast<node *>(f.anchor);
}

node * find_node(thread_state & ts, const node * parent, const char * name)
{
    const node_key key { parent, name };
    auto found = ts.index.find(key);
    if (found != ts.index.end()) return found->second;
    ts.nodes.emplace_back(new node { parent, name, state().next_seq.fetch_add(1, std::memory_order_relaxed) });
    node * created = ts.nodes.back().get();
    ts.index.emplace(key, created);
    return created;
}

/// Whether the tracer has not yet written the call path of `n`, as run on any thread.
bool claim_trace(const node * n)
{
    std::vector<const char *> path;
    for (const node * at = n; at; at = at->parent) path.push_back(at->name);
    static std::vector<std::vector<const char *> > traced;
    std::lock_guard<std::mutex> lock(log_mutex());
    if (std::find(traced.begin(), traced.end(), path) != traced.end()) return false;
    traced.push_back(path);
    return true;
}

void close_task(thread_state & ts, const frame & f, std::uint64_t end)
{
    node * n                    = f.task;
    const std::uint64_t elapsed = end > f.start ? end - f.start : 0;
    const std::uint64_t spent   = elapsed > f.paused ? elapsed - f.paused : 0;
    n->count += 1;
    n->total_ns += spent;
    n->max_ns = std::max(n->max_ns, spent);
    if (!f.parallel)
    {
        global_state & global = state();
        for (int k = 0; k < memory_kinds; ++k)
        {
            const std::int64_t peak = global.region_peak[k].load(std::memory_order_relaxed) - f.live_at_start[k];
            if (peak > 0) n->peak_bytes[k] = std::max(n->peak_bytes[k], static_cast<std::uint64_t>(peak));
            atomic_max(global.region_peak[k], f.outer_peak[k]);
        }
    }
    // A top-level task is traced on every run, a nested path the first time it runs, everything at level 5.
    if (is_tracer_enabled())
    {
        const bool first = !n->traced && claim_trace(n);
        n->traced        = true;
        if (ts.stack.empty() || first || is_service_debug_enabled())
        {
            std::ostringstream out;
            out << n->name << ' ' << format_time(spent) << '\n';
            const std::string text = out.str();
            profiler_log(text.data(), text.size());
        }
    }
}

int begin_task(thread_state & ts, const char * name, bool threading, bool on_queue)
{
    const node * parent = ts.stack.empty() ? nullptr : ts.stack.back().anchor;
    const bool in_chunk = !ts.stack.empty() && !ts.stack.back().task;
    node * n            = find_node(ts, parent, name);
    if (threading || in_chunk) n->threading = true;
    if (on_queue) n->on_queue = true;
    frame f { n, n, 0 };
    f.parallel = threading || (!ts.stack.empty() && ts.stack.back().parallel);
    if (!f.parallel)
    {
        global_state & global = state();
        for (int k = 0; k < memory_kinds; ++k)
        {
            f.live_at_start[k] = global.live[k].load(std::memory_order_relaxed);
            f.outer_peak[k]    = global.region_peak[k].exchange(f.live_at_start[k], std::memory_order_relaxed);
        }
    }
    f.start = now_ns();
    ts.stack.push_back(f);
    return static_cast<int>(ts.stack.size()) - 1;
}

void end_task(thread_state & ts, int depth)
{
    if (depth < 0 || depth >= static_cast<int>(ts.stack.size())) return;
    const std::uint64_t end = now_ns();
    while (static_cast<int>(ts.stack.size()) > depth)
    {
        // A chunk of a parallel region is closed by region_leave(), not by the tasks under it.
        if (!ts.stack.back().task) break;
        const frame f = ts.stack.back();
        ts.stack.pop_back();
        close_task(ts, f, end);
    }
}

const void * region_capture()
{
    thread_state & ts = this_thread();
    return ts.stack.empty() ? nullptr : ts.stack.back().anchor;
}

void region_enter(const void * region)
{
    thread_state & ts   = this_thread();
    const node * parent = static_cast<const node *>(region);
    frame chunk { parent, nullptr, 0 };
    chunk.parallel = true;
    // A thread that waits inside other work, for example in a nested parallel loop, may be given a
    // chunk of an unrelated region. Its time is not part of the tasks it interrupts: those above the
    // region's parent, or every open task when the parent is not on this thread.
    if (!ts.stack.empty() && ts.stack.back().anchor != parent)
    {
        chunk.interrupts = true;
        chunk.start      = now_ns();
        for (std::size_t i = ts.stack.size(); i-- > 0;)
        {
            if (ts.stack[i].anchor == parent)
            {
                chunk.pause_from = i + 1;
                break;
            }
        }
    }
    ts.stack.push_back(chunk);
}

void region_leave()
{
    thread_state & ts       = this_thread();
    const std::uint64_t end = now_ns();
    while (!ts.stack.empty())
    {
        const frame f = ts.stack.back();
        ts.stack.pop_back();
        if (f.task)
        {
            close_task(ts, f, end);
            continue;
        }
        if (f.interrupts)
        {
            const std::uint64_t elapsed = end > f.start ? end - f.start : 0;
            for (std::size_t i = f.pause_from; i < ts.stack.size(); ++i) ts.stack[i].paused += elapsed;
        }
        break;
    }
}

// Merged tree used by the report.
struct merged
{
    std::string name;
    std::vector<std::unique_ptr<merged> > children;
    std::vector<std::pair<const thread_state *, std::uint64_t> > time_by_thread;
    std::uint64_t seq     = ~std::uint64_t(0);
    std::uint64_t count   = 0;
    std::uint64_t max_ns  = 0;
    std::uint64_t time_ns = 0;
    std::uint64_t self_ns = 0;
    bool parallel         = false;
    bool on_queue         = false;
    // Of the task alone until finish() adds the children in, except the peak.
    std::uint64_t alloc_bytes[memory_kinds] = {};
    std::uint64_t alloc_count[memory_kinds] = {};
    std::uint64_t peak_bytes[memory_kinds]  = {};
    std::uint64_t held_bytes[memory_kinds]  = {};
};

struct node_ptr_hash
{
    std::size_t operator()(const node * n) const { return std::hash<const void *>()(n); }
};

merged * merged_child(merged & parent, const char * name)
{
    for (auto & child : parent.children)
    {
        if (child->name == name) return child.get();
    }
    parent.children.emplace_back(new merged());
    parent.children.back()->name = name;
    return parent.children.back().get();
}

merged * resolve(merged & root, std::unordered_map<const node *, merged *, node_ptr_hash> & memo, const node * n)
{
    auto found = memo.find(n);
    if (found != memo.end()) return found->second;
    merged * parent = n->parent ? resolve(root, memo, n->parent) : &root;
    merged * m      = merged_child(*parent, n->name);
    memo.emplace(n, m);
    return m;
}

void add_time(merged & m, const thread_state * ts, std::uint64_t time_ns)
{
    for (auto & entry : m.time_by_thread)
    {
        if (entry.first == ts)
        {
            entry.second += time_ns;
            return;
        }
    }
    m.time_by_thread.emplace_back(ts, time_ns);
}

/// Settles the times: a parallel task takes the busiest thread's time, a sequential one the sum.
/// A row is capped at the row it is in: under nested parallel loops one thread may run the inner
/// chunks of several outer tasks, so its inner total can exceed any one outer task.
void finish(merged & m, bool parent_parallel, std::uint64_t limit)
{
    m.parallel |= parent_parallel;
    m.time_ns = 0;
    for (const auto & entry : m.time_by_thread) m.time_ns = m.parallel ? std::max(m.time_ns, entry.second) : m.time_ns + entry.second;
    m.time_ns = std::min(m.time_ns, limit);
    std::sort(m.children.begin(), m.children.end(), [](const auto & a, const auto & b) { return a->seq < b->seq; });
    std::uint64_t child_time = 0;
    for (auto & child : m.children)
    {
        finish(*child, m.parallel, m.time_ns);
        child_time += child->time_ns;
        for (int k = 0; k < memory_kinds; ++k)
        {
            m.alloc_bytes[k] += child->alloc_bytes[k];
            m.alloc_count[k] += child->alloc_count[k];
            m.held_bytes[k] += child->held_bytes[k];
        }
    }
    m.self_ns = m.time_ns > child_time ? m.time_ns - child_time : 0;
}

std::unique_ptr<merged> build_tree()
{
    std::unique_ptr<merged> root(new merged());
    std::unordered_map<const node *, merged *, node_ptr_hash> memo;
    global_state & global = state();
    std::lock_guard<std::mutex> lock(global.threads_mutex);
    for (const thread_state * ts : global.threads)
    {
        for (const auto & n : ts->nodes)
        {
            merged * m = resolve(*root, memo, n.get());
            m->count += n->count;
            m->max_ns = std::max(m->max_ns, n->max_ns);
            m->seq    = std::min(m->seq, n->seq);
            m->parallel |= n->threading;
            m->on_queue |= n->on_queue;
            add_time(*m, ts, n->total_ns);
            for (int k = 0; k < memory_kinds; ++k)
            {
                m->alloc_bytes[k] += n->alloc_bytes[k].load(std::memory_order_relaxed);
                m->alloc_count[k] += n->alloc_count[k].load(std::memory_order_relaxed);
                m->peak_bytes[k] = std::max(m->peak_bytes[k], n->peak_bytes[k]);
            }
        }
    }
    for (block_shard & shard : global.shards)
    {
        std::lock_guard<std::mutex> shard_lock(shard.mutex);
        for (const auto & entry : shard.blocks)
        {
            if (entry.second.owner) resolve(*root, memo, entry.second.owner)->held_bytes[entry.second.kind] += entry.second.size;
        }
    }
    for (auto & child : root->children) finish(*child, false, ~std::uint64_t(0));
    return root;
}

bool has_runs(const merged & m)
{
    if (m.count) return true;
    for (const auto & child : m.children)
    {
        if (has_runs(*child)) return true;
    }
    return false;
}

void flatten(const merged & m, std::int64_t parent, std::int64_t depth, std::vector<profiler_report_row> & rows)
{
    for (const auto & child : m.children)
    {
        if (!has_runs(*child)) continue;
        const std::int64_t index = static_cast<std::int64_t>(rows.size());
        rows.push_back({ child->name, parent, depth, child->parallel, child->count, child->time_ns, child->self_ns, child->max_ns,
                         static_cast<std::uint64_t>(child->time_by_thread.size()) });
        profiler_report_row & row = rows.back();
        row.on_queue              = child->on_queue;
        for (int k = 0; k < memory_kinds; ++k)
        {
            row.alloc_bytes[k] = child->alloc_bytes[k];
            row.alloc_count[k] = child->alloc_count[k];
            row.peak_bytes[k]  = child->peak_bytes[k];
            row.held_bytes[k]  = child->held_bytes[k];
        }
        flatten(*child, index, depth + 1, rows);
    }
}

void report_at_exit()
{
    std::ostringstream out;
    profiler_print_report(out);
    const std::string text = out.str();
    profiler_log(text.data(), text.size());
}

} // namespace

DAAL_EXPORT void print_header()
{
    if (!is_profiler_enabled()) return;
    register_exit_report();
    daal::services::LibraryVersionInfo ver;
    std::ostringstream out;
    out << "Major version:          " << ver.majorVersion << '\n';
    out << "Minor version:          " << ver.minorVersion << '\n';
    out << "Update version:         " << ver.updateVersion << '\n';
    out << "Product status:         " << ver.productStatus << '\n';
    out << "Build:                  " << ver.build << '\n';
    out << "Build revision:         " << ver.build_rev << '\n';
    out << "Name:                   " << ver.name << '\n';
    out << "Processor optimization: " << ver.processor << '\n';
    out << '\n';
    const std::string text = out.str();
    profiler_log(text.data(), text.size());
}

DAAL_EXPORT bool profiler_claim_log_site(const char * site)
{
    if (is_service_debug_enabled()) return true;
    static std::vector<std::string> logged;
    std::lock_guard<std::mutex> lock(log_mutex());
    if (std::find(logged.begin(), logged.end(), site) != logged.end()) return false;
    logged.push_back(site);
    return true;
}

DAAL_EXPORT void profiler_log(const char * text, std::size_t length)
{
    std::lock_guard<std::mutex> lock(log_mutex());
    std::fwrite(text, 1, length, stderr);
    std::fflush(stderr);
}

DAAL_EXPORT int profiler_begin_task(const char * task_name, bool threading, bool on_queue)
{
    return begin_task(this_thread(), task_name, threading, on_queue);
}

DAAL_EXPORT void profiler_end_task(int depth)
{
    end_task(this_thread(), depth);
}

DAAL_EXPORT void profiler_record_alloc(const void * ptr, std::size_t size, profiler_memory_kind kind)
{
    if (!ptr) return;
    const int k           = static_cast<int>(kind);
    node * owner          = charge_target(this_thread());
    global_state & global = state();
    {
        block_shard & shard = shard_for(global, ptr);
        std::lock_guard<std::mutex> lock(shard.mutex);
        shard.blocks[ptr] = { size, k, owner };
    }
    if (owner)
    {
        owner->alloc_bytes[k].fetch_add(size, std::memory_order_relaxed);
        owner->alloc_count[k].fetch_add(1, std::memory_order_relaxed);
    }
    const std::int64_t live = global.live[k].fetch_add(static_cast<std::int64_t>(size), std::memory_order_relaxed) + static_cast<std::int64_t>(size);
    atomic_max(global.region_peak[k], live);
    atomic_max(global.process_peak[k], live);
}

DAAL_EXPORT void profiler_record_free(const void * ptr)
{
    if (!ptr) return;
    global_state & global = state();
    block freed;
    {
        block_shard & shard = shard_for(global, ptr);
        std::lock_guard<std::mutex> lock(shard.mutex);
        auto found = shard.blocks.find(ptr);
        if (found == shard.blocks.end()) return;
        freed = found->second;
        shard.blocks.erase(found);
    }
    global.live[freed.kind].fetch_sub(static_cast<std::int64_t>(freed.size), std::memory_order_relaxed);
}

DAAL_EXPORT const profiler_region_hooks & profiler_threading_hooks()
{
    return region_hooks;
}

DAAL_EXPORT std::vector<profiler_report_row> profiler_report()
{
    std::vector<profiler_report_row> rows;
    flatten(*build_tree(), -1, 0, rows);
    return rows;
}

DAAL_EXPORT void profiler_print_report(std::ostream & out)
{
    const std::vector<profiler_report_row> rows = profiler_report();
    std::uint64_t total                         = 0;
    for (const auto & row : rows)
    {
        if (row.depth == 0) total += row.time_ns;
    }
    out << "Algorithm tree analyzer" << '\n';
    for (const auto & row : rows)
    {
        std::string prefix;
        for (std::int64_t level = 0; level < row.depth; ++level) prefix += "|  ";
        out << prefix << "|-- " << row.name << " time: " << format_time(row.time_ns) << ' ' << std::fixed << std::setprecision(2)
            << (total > 0 ? 100.0 * static_cast<double>(row.time_ns) / static_cast<double>(total) : 0.0) << "% " << row.count << " times ";
        if (row.parallel)
            out << "in a parallel region on " << row.threads << " threads";
        else if (row.on_queue)
            out << "on a SYCL queue";
        else
            out << "in a sequential region";
        static const char * const kind_names[memory_kinds] = { "host", "device" };
        for (int k = 0; k < memory_kinds; ++k)
        {
            if (!row.alloc_count[k]) continue;
            out << ", " << kind_names[k] << " memory " << format_bytes(row.alloc_bytes[k]) << " in " << row.alloc_count[k]
                << (row.alloc_count[k] == 1 ? " block" : " blocks");
            if (!row.parallel) out << " peak " << format_bytes(row.peak_bytes[k]);
            if (row.held_bytes[k]) out << " held " << format_bytes(row.held_bytes[k]);
        }
        out << '\n';
    }
    out << "|--(end)" << '\n';
    out << "DAAL KERNEL_PROFILER: kernels total time " << format_time(total) << '\n';
    if (is_memory_tracking_enabled())
    {
        global_state & global = state();
        out << "DAAL KERNEL_PROFILER: memory peak host " << format_bytes(global.process_peak[0].load(std::memory_order_relaxed)) << ", device "
            << format_bytes(global.process_peak[1].load(std::memory_order_relaxed)) << '\n';
    }
}

DAAL_EXPORT void profiler_reset()
{
    global_state & global = state();
    std::lock_guard<std::mutex> lock(global.threads_mutex);
    for (thread_state * ts : global.threads)
    {
        ts->stack.clear();
        ts->index.clear();
        ts->nodes.clear();
    }
    // Blocks still allocated stay counted as live, but no longer charged to a task.
    for (block_shard & shard : global.shards)
    {
        std::lock_guard<std::mutex> shard_lock(shard.mutex);
        for (auto & entry : shard.blocks) entry.second.owner = nullptr;
    }
    for (int k = 0; k < memory_kinds; ++k)
    {
        const std::int64_t live = global.live[k].load(std::memory_order_relaxed);
        global.region_peak[k].store(live, std::memory_order_relaxed);
        global.process_peak[k].store(live, std::memory_order_relaxed);
    }
}

} // namespace internal
} // namespace daal

extern "C" DAAL_EXPORT void _onedal_profiler_report(const char * path, int clear)
{
    std::ostringstream out;
    daal::internal::profiler_print_report(out);
    const std::string text = out.str();
    if (path)
    {
        std::ofstream file(path, std::ios::app);
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
    }
    else
    {
        daal::internal::profiler_log(text.data(), text.size());
    }
    if (clear) daal::internal::profiler_reset();
}
