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

/// Unit tests for the for-loop and NUMA entry points of the threading layer.
///
/// Before this suite nothing exercised `_daal_threader_for*` except whichever algorithm happened
/// to call it, so a change to an iteration space, to a callback convention, or to the meaning of a
/// parameter only surfaced as a wrong number somewhere downstream -- or did not surface at all.
/// The cases below are deliberately mechanical, because that is the class of defect they are here
/// to catch; three concrete ones found on this branch:
///
/// * `threader_for_int64ptr` did not compile. It passed the `int64_t` thunk where the entry point
///   wants the `const int64_t *` one, and a wrapper template with no caller is never instantiated,
///   so neither it nor `threader_for_int32ptr` nor `threader_for_int64` on `main` had ever been
///   compiled. Every wrapper therefore gets a case here, if only to instantiate it.
/// * `threader_for_blocked` and the `_blocked_size` entry point it absorbs had divergent callback
///   conventions, `(first, last)` against `(first, length)`. `checkTiling` tells them apart.
/// * `conditional_threader_for` kept forwarding the legacy `threader_for(n, n, ...)`, where the
///   second argument used to be an ignored thread-count request. As a grain size, `n` collapses
///   the loop into one chunk, which left 21 call sites silently serial. The thread-id assertions
///   catch that; an every-index-visited assertion on its own does not.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "services/env_detect.h"
#include "src/threading/threading.h"
#include "src/threading/service_thread_pinner.h"
#include "src/services/service_topo.h"

#if defined(__linux__)
    #include <sys/syscall.h>
    #include <unistd.h>
#endif

namespace daal::test
{
/// Records how many times each index of an iteration space was visited.
///
/// Catch2's assertion macros are not thread-safe, so a loop body never asserts: it only marks the
/// index it was given, and the properties of the whole iteration space are checked after the
/// parallel region has completed.
class VisitLog
{
public:
    /// @param[in] count Number of indices in the iteration space.
    explicit VisitLog(int64_t count) : _visits(static_cast<size_t>(count))
    {
        for (auto & visit : _visits)
        {
            visit.store(0, std::memory_order_relaxed);
        }
    }

    /// Record a visit of the index `i`. Safe to call concurrently.
    ///
    /// @param[in] i Index that the loop body was given.
    void mark(int64_t i)
    {
        if (i < 0 || static_cast<size_t>(i) >= _visits.size())
        {
            _outOfRange.store(true, std::memory_order_relaxed);
            return;
        }
        _visits[static_cast<size_t>(i)].fetch_add(1, std::memory_order_relaxed);
    }

    /// @return `true` if every index of the iteration space was visited exactly once and no index
    ///         outside of it was ever passed to the loop body.
    bool visitedExactlyOnce() const
    {
        if (_outOfRange.load(std::memory_order_relaxed)) return false;
        for (const auto & visit : _visits)
        {
            if (visit.load(std::memory_order_relaxed) != 1) return false;
        }
        return true;
    }

private:
    std::vector<std::atomic<int32_t> > _visits;
    std::atomic<bool> _outOfRange { false };
};

/// Records the `[first, last)` blocks that a blocked loop handed to its body.
class BlockLog
{
public:
    /// Record one block. Safe to call concurrently.
    ///
    /// @param[in] first Index of the first iteration of the block.
    /// @param[in] last  Index past the last iteration of the block.
    void add(int64_t first, int64_t last)
    {
        std::lock_guard<std::mutex> guard(_mutex);
        _blocks.push_back(std::make_pair(first, last));
    }

    /// @return The recorded blocks, ordered by their first index.
    std::vector<std::pair<int64_t, int64_t> > sortedBlocks() const
    {
        std::vector<std::pair<int64_t, int64_t> > blocks = _blocks;
        std::sort(blocks.begin(), blocks.end());
        return blocks;
    }

private:
    mutable std::mutex _mutex;
    std::vector<std::pair<int64_t, int64_t> > _blocks;
};

/// Check that the recorded blocks tile `[0, n)` exactly: they start at `0`, each one begins where
/// the previous one ended, none is empty, and the last one ends at `n`.
///
/// This is what tells a `(first, last)` callback convention from a `(first, length)` one: with
/// lengths, the blocks after the first would start too early and the final block would stop short
/// of `n`.
///
/// @param[in] log Blocks recorded by the loop under test.
/// @param[in] n   Number of iterations the loop was asked to run.
void checkTiling(const BlockLog & log, int64_t n)
{
    const auto blocks = log.sortedBlocks();
    REQUIRE(!blocks.empty());

    int64_t expectedFirst = 0;
    for (const auto & block : blocks)
    {
        REQUIRE(block.first == expectedFirst);
        REQUIRE(block.second > block.first);
        expectedFirst = block.second;
    }
    REQUIRE(expectedFirst == n);
}

/// Records which thread indices a loop body ran on.
///
/// Like `VisitLog`, the body never asserts: it only records the index of the thread it ran on, and
/// the shape of the schedule is checked after the parallel region has completed.
class ThreadIdLog
{
public:
    ThreadIdLog() : _seen(static_cast<size_t>(threader_get_max_threads_number()))
    {
        for (auto & seen : _seen)
        {
            seen.store(false, std::memory_order_relaxed);
        }
    }

    /// Record that the calling thread ran an iteration. Safe to call concurrently.
    void mark()
    {
        const int threadId = threader_get_max_current_thread_index();
        if (threadId < 0 || static_cast<size_t>(threadId) >= _seen.size())
        {
            _outOfRange.store(true, std::memory_order_relaxed);
            return;
        }
        _seen[static_cast<size_t>(threadId)].store(true, std::memory_order_relaxed);
    }

    /// @return `true` if every thread index handed out was within `[0, max_threads)`.
    bool inRange() const { return !_outOfRange.load(std::memory_order_relaxed); }

    /// @return Number of distinct thread indices that ran at least one iteration.
    size_t distinctCount() const
    {
        size_t count = 0;
        for (const auto & seen : _seen)
        {
            if (seen.load(std::memory_order_relaxed)) ++count;
        }
        return count;
    }

private:
    std::vector<std::atomic<bool> > _seen;
    std::atomic<bool> _outOfRange { false };
};

/// @return `true` if the threading layer is configured to run loops in parallel, so that
///         assertions on the *shape* of the partitioning are meaningful. The loops fall back to a
///         plain sequential pass otherwise.
bool isParallel()
{
    return threader_get_threads_number() > 1;
}

/// Both plain index loops in one case, as for the `_int32` ones below. The property checked is the
/// only one that holds whatever the backend's partitioning does: every index of the space reaches
/// the body exactly once, for an empty space as well as a large one, and for any grain size the
/// caller is allowed to pass.
TEST("the 64-bit index loops visit every index exactly once", "[threading][unit]")
{
    for (const int64_t n : { int64_t(0), int64_t(1), int64_t(17), int64_t(1024), int64_t(100000) })
    {
        for (const int64_t grainSize : { int64_t(1), int64_t(7), int64_t(4096) })
        {
            VisitLog plain(n);
            threader_for(n, grainSize, [&](int64_t i) { plain.mark(i); });
            REQUIRE(plain.visitedExactlyOnce());

            VisitLog simple(n);
            threader_for_simple(n, grainSize, [&](int64_t i) { simple.mark(i); });
            REQUIRE(simple.visitedExactlyOnce());
        }
    }
}

TEST("threader_for_optional visits every index exactly once, nested or not", "[threading][unit]")
{
    constexpr int64_t n = 5000;

    VisitLog flat(n);
    threader_for_optional(n, 1, [&](int64_t i) { flat.mark(i); });
    REQUIRE(flat.visitedExactlyOnce());

    // Outside of a parallel region the loop is a real parallel one, so with more than one thread
    // available its iterations are spread over more than one of them. The iteration space is much
    // larger than above because the body is trivial and the partitioner only splits a space it
    // considers worth splitting.
    if (isParallel())
    {
        constexpr int64_t wide = 2000000;
        ThreadIdLog threads;
        threader_for_optional(wide, 1, [&](int64_t i) {
            (void)i;
            threads.mark();
        });
        REQUIRE(threads.inRange());
        REQUIRE(threads.distinctCount() > 1u);
    }

    // Called from inside a parallel region, the loop runs sequentially in the calling thread
    // instead of nesting a second one -- it still has to cover the whole iteration space.
    constexpr int64_t outerCount = 8;
    VisitLog nested(outerCount * n);

    // One entry per outer iteration, holding the index of the thread that ran the inner loop for
    // it, or `noThread` until the first inner iteration claims it. A nested `threader_for_optional`
    // that did spawn a parallel region would leave at least one entry claimed by one thread and
    // then visited by another.
    constexpr int noThread = -1;
    std::vector<std::atomic<int> > innerThread(static_cast<size_t>(outerCount));
    for (auto & thread : innerThread)
    {
        thread.store(noThread, std::memory_order_relaxed);
    }
    std::atomic<bool> innerMigrated { false };

    threader_for(outerCount, 1, [&](int64_t outer) {
        threader_for_optional(n, 1, [&](int64_t inner) {
            nested.mark(outer * n + inner);

            const int threadId = threader_get_max_current_thread_index();
            auto & claimed     = innerThread[static_cast<size_t>(outer)];
            int expected       = noThread;
            if (!claimed.compare_exchange_strong(expected, threadId, std::memory_order_relaxed) && expected != threadId)
            {
                innerMigrated.store(true, std::memory_order_relaxed);
            }
        });
    });
    REQUIRE(nested.visitedExactlyOnce());
    REQUIRE(!innerMigrated.load(std::memory_order_relaxed));
}

TEST("threader_for_blocked tiles the iteration space with (first, last) blocks", "[threading][unit]")
{
    for (const int64_t n : { int64_t(1), int64_t(9), int64_t(1000), int64_t(65536) })
    {
        for (const int64_t grainSize : { int64_t(1), int64_t(32) })
        {
            BlockLog log;
            threader_for_blocked(n, grainSize, [&](int64_t first, int64_t last) { log.add(first, last); });
            checkTiling(log, n);
        }
    }
}

TEST("threader_for_blocked hands the grain size to the threading backend", "[threading][unit]")
{
    constexpr int64_t n = 1000000;

    // A grain size of `n` makes the whole iteration space indivisible, so the backend must produce
    // exactly one block however many threads are available. Before the grain size reached the
    // backend the chunk size was hardcoded to 1 and this loop was split into one block per thread.
    BlockLog single;
    threader_for_blocked(n, n, [&](int64_t first, int64_t last) { single.add(first, last); });
    REQUIRE(single.sortedBlocks().size() == 1u);
    checkTiling(single, n);

    // The converse: with the smallest grain size an iteration space this large is split, as long
    // as there is more than one thread to split it across.
    if (isParallel())
    {
        BlockLog split;
        threader_for_blocked(n, 1, [&](int64_t first, int64_t last) { split.add(first, last); });
        REQUIRE(split.sortedBlocks().size() > 1u);
        checkTiling(split, n);
    }
}

TEST("threader_for_blocked spans an iteration space wider than INT32_MAX", "[threading][unit]")
{
    // Only the blocks are visited, not the individual iterations, so the whole 64-bit space is
    // covered at the cost of a few hundred callbacks.
    const int64_t n         = int64_t(std::numeric_limits<int32_t>::max()) + 100000;
    const int64_t grainSize = n / 64;

    BlockLog log;
    threader_for_blocked(n, grainSize, [&](int64_t first, int64_t last) { log.add(first, last); });
    checkTiling(log, n);

    const auto blocks = log.sortedBlocks();
    // A 32-bit induction variable would have wrapped long before reaching the end of the space,
    // so no block would have ended at `n` and the tiling above would not have closed.
    REQUIRE(blocks.back().second == n);
    REQUIRE(blocks.back().second > int64_t(std::numeric_limits<int32_t>::max()));
    REQUIRE(blocks.size() > 1u);
}

// Weekly scope: unlike the blocked case below, this one has to run the body once per index, so its
// cost scales with 1/thread count. It is negligible on a many-core host and a few seconds on a
// two-core runner. The default scope keeps the blocked case, which covers the same 64-bit boundary
// arithmetic at the cost of a few hundred callbacks.
TEST("threader_for reaches indices beyond INT32_MAX", "[threading][unit][weekly]")
{
    const int64_t n = int64_t(std::numeric_limits<int32_t>::max()) + 3;

    // An iteration space this large cannot be tracked index by index, so the body only reports
    // whether it was ever handed a negative index -- which is what a 32-bit counter would
    // eventually produce -- and whether the last index was reached.
    std::atomic<bool> sawNegative { false };
    std::atomic<bool> sawLast { false };
    threader_for(n, 4096, [&](int64_t i) {
        if (i < 0) sawNegative.store(true, std::memory_order_relaxed);
        if (i == n - 1) sawLast.store(true, std::memory_order_relaxed);
    });

    REQUIRE(!sawNegative.load(std::memory_order_relaxed));
    REQUIRE(sawLast.load(std::memory_order_relaxed));
}

TEST("threader_for_int64ptr visits every pointer of the range exactly once", "[threading][unit]")
{
    constexpr int64_t n = 4096;

    std::vector<int64_t> values(n);
    for (int64_t i = 0; i < n; ++i)
    {
        values[static_cast<size_t>(i)] = i * 3;
    }

    VisitLog log(n);
    std::atomic<bool> wrongValue { false };
    threader_for_int64ptr(values.data(), values.data() + n, [&](const int64_t * i) {
        const int64_t index = i - values.data();
        if (*i != index * 3) wrongValue.store(true, std::memory_order_relaxed);
        log.mark(index);
    });

    REQUIRE(!wrongValue.load(std::memory_order_relaxed));
    REQUIRE(log.visitedExactlyOnce());
}

TEST("static_threader_for gives every thread one contiguous block", "[threading][unit]")
{
    constexpr int64_t n       = 10000;
    const size_t maxThreads   = static_cast<size_t>(threader_get_max_threads_number());
    constexpr size_t noThread = static_cast<size_t>(-1);
    std::vector<size_t> threadOfIdx(static_cast<size_t>(n), noThread);

    VisitLog log(n);
    std::atomic<bool> threadIdOutOfRange { false };
    static_threader_for(n, [&](int64_t i, size_t tid) {
        if (tid >= maxThreads)
        {
            threadIdOutOfRange.store(true, std::memory_order_relaxed);
            return;
        }
        threadOfIdx[static_cast<size_t>(i)] = tid;
        log.mark(i);
    });

    REQUIRE(!threadIdOutOfRange.load(std::memory_order_relaxed));
    REQUIRE(log.visitedExactlyOnce());

    // The static schedule is what callers rely on to index per-thread scratch: the indices given
    // to one thread must form a single contiguous block, and the blocks must come in thread order.
    bool monotonicInThreadId = true;
    for (int64_t i = 1; i < n; ++i)
    {
        if (threadOfIdx[static_cast<size_t>(i)] < threadOfIdx[static_cast<size_t>(i - 1)])
        {
            monotonicInThreadId = false;
            break;
        }
    }
    REQUIRE(monotonicInThreadId);
}

TEST("threader_for_break covers the space when nothing breaks", "[threading][unit]")
{
    constexpr int64_t n = 10000;

    VisitLog log(n);
    threader_for_break(n, 1, [&](int64_t i, bool & needBreak) {
        log.mark(i);
        needBreak = false;
    });
    REQUIRE(log.visitedExactlyOnce());
}

TEST("threader_for_break stops the loop early", "[threading][unit]")
{
    // Large enough that cancellation is observed long before the space is exhausted, whether the
    // loop runs sequentially (it breaks on the first iteration) or in parallel (the chunks already
    // running finish, but no new one is started once the group is cancelled).
    constexpr int64_t n = 10000000;

    std::atomic<int64_t> visited { 0 };
    threader_for_break(n, 1, [&](int64_t i, bool & needBreak) {
        (void)i;
        visited.fetch_add(1, std::memory_order_relaxed);
        needBreak = true;
    });

    REQUIRE(visited.load(std::memory_order_relaxed) > 0);
    REQUIRE(visited.load(std::memory_order_relaxed) < n);
}

TEST("the int32 loops visit every index exactly once", "[threading][unit]")
{
    constexpr int n = 5000;

    VisitLog plain(n);
    threader_for_int32(n, 1, [&](int i) { plain.mark(i); });
    REQUIRE(plain.visitedExactlyOnce());

    VisitLog simple(n);
    threader_for_simple_int32(n, 16, [&](int i) { simple.mark(i); });
    REQUIRE(simple.visitedExactlyOnce());

    VisitLog optional(n);
    threader_for_optional_int32(n, 1, [&](int i) { optional.mark(i); });
    REQUIRE(optional.visitedExactlyOnce());

    VisitLog breakable(n);
    threader_for_break_int32(n, 1, [&](int i, bool & needBreak) {
        breakable.mark(i);
        needBreak = false;
    });
    REQUIRE(breakable.visitedExactlyOnce());
}

TEST("threader_for_blocked_int32 tiles the iteration space with (first, last) blocks", "[threading][unit]")
{
    for (const int n : { 1, 9, 1000, 65536 })
    {
        BlockLog log;
        threader_for_blocked_int32(n, 32, [&](int first, int last) { log.add(first, last); });
        checkTiling(log, n);
    }
}

TEST("the conditional loops cover the space and honour inParallel", "[threading][unit]")
{
    constexpr size_t n = 5000;

    for (const bool inParallel : { false, true })
    {
        VisitLog plain(static_cast<int64_t>(n));
        conditional_threader_for(inParallel, n, [&](size_t i) { plain.mark(static_cast<int64_t>(i)); });
        REQUIRE(plain.visitedExactlyOnce());

        VisitLog stat(static_cast<int64_t>(n));
        conditional_static_threader_for(inParallel, n, [&](size_t i, size_t tid) {
            (void)tid;
            stat.mark(static_cast<int64_t>(i));
        });
        REQUIRE(stat.visitedExactlyOnce());
    }

    // `inParallel` has to buy actual parallelism, not just a parallel-looking call. These loops
    // forward to `threader_for`/`static_threader_for`, whose grain size the conditional wrappers
    // choose on the caller's behalf -- a grain size as large as the iteration space would leave the
    // whole loop in one chunk and run on a single thread while still visiting every index.
    if (isParallel())
    {
        constexpr size_t wide = 2000000;

        ThreadIdLog plainThreads;
        conditional_threader_for(true, wide, [&](size_t i) {
            (void)i;
            plainThreads.mark();
        });
        REQUIRE(plainThreads.inRange());
        REQUIRE(plainThreads.distinctCount() > 1u);

        ThreadIdLog staticThreads;
        conditional_static_threader_for(true, wide, [&](size_t i, size_t tid) {
            (void)i;
            (void)tid;
            staticThreads.mark();
        });
        REQUIRE(staticThreads.inRange());
        REQUIRE(staticThreads.distinctCount() > 1u);
    }
}

TEST("threader_execute_in_numa runs the callable once, inside or outside the node range", "[threading][unit]")
{
    const int nodes = threader_numa_count();
    REQUIRE(nodes >= 1);

    for (int node = -1; node <= nodes; ++node)
    {
        int calls = 0;
        threader_execute_in_numa(node, [&]() { ++calls; });
        REQUIRE(calls == 1);
    }
}

TEST("loops run through threader_execute_in_numa cover their space with arena-relative thread indices", "[threading][unit]")
{
    constexpr int64_t n = 200000;
    const int nodes     = threader_numa_count();

    // All nodes at once from a parallel loop, the way a caller spreads work over the arenas.
    std::vector<std::unique_ptr<VisitLog> > logs;
    for (int node = 0; node < nodes; ++node)
    {
        logs.push_back(std::make_unique<VisitLog>(n));
    }
    std::atomic<bool> threadIdOutOfRange { false };

    threader_for(nodes, 1, [&](int64_t node) {
        threader_execute_in_numa(static_cast<int>(node), [&]() {
            ThreadIdLog threads;
            threader_for(n, 1, [&](int64_t i) {
                logs[static_cast<size_t>(node)]->mark(i);
                threads.mark();
            });
            if (!threads.inRange()) threadIdOutOfRange.store(true, std::memory_order_relaxed);
        });
    });

    for (const auto & log : logs)
    {
        REQUIRE(log->visitedExactlyOnce());
    }
    REQUIRE(!threadIdOutOfRange.load(std::memory_order_relaxed));
}

TEST("NUMA arenas share the thread count set by setNumberOfThreads", "[threading][unit]")
{
    const int nodes = threader_numa_count();
    if (nodes < 2) return;

    const size_t savedThreads = threader_get_threads_number();
    auto arenaThreads         = [&]() {
        std::vector<int> result(static_cast<size_t>(nodes));
        for (int node = 0; node < nodes; ++node)
        {
            threader_execute_in_numa(node, [&]() { result[static_cast<size_t>(node)] = threader_get_max_threads_number(); });
        }
        return result;
    };

    const std::vector<int> initial = arenaThreads();
    int total                      = 0;
    for (const int threads : initial)
    {
        REQUIRE(threads >= 1);
        total += threads;
    }
    REQUIRE(static_cast<size_t>(total) <= std::max(savedThreads, static_cast<size_t>(nodes)));

    // One thread per node gives every arena a single thread; restoring the thread count resizes
    // the arenas back. The count is restored before asserting so later tests are unaffected.
    services::Environment::getInstance()->setNumberOfThreads(static_cast<size_t>(nodes));
    const std::vector<int> reduced = arenaThreads();
    services::Environment::getInstance()->setNumberOfThreads(savedThreads);
    const std::vector<int> restored = arenaThreads();

    for (const int threads : reduced)
    {
        REQUIRE(threads == 1);
    }
    REQUIRE(restored == initial);
}

#if defined(__linux__)
/// @return OS index of the NUMA node of the CPU the calling thread runs on.
int currentOsNumaNode()
{
    unsigned cpu  = 0;
    unsigned node = 0;
    if (syscall(SYS_getcpu, &cpu, &node, nullptr) != 0) return -1;
    return static_cast<int>(node);
}

TEST("threader_execute_in_numa keeps each arena's threads on one NUMA node", "[threading][unit]")
{
    const int nodes = threader_numa_count();
    if (nodes < 2) return;

    // OS node indices need not equal the threading layer's node indices, so each arena is only
    // required to stay on one OS node, and different arenas on different ones.
    constexpr int64_t n = 2000000;
    std::vector<int> osNodeOfArena(static_cast<size_t>(nodes), -1);
    std::atomic<bool> leftNode { false };

    for (int node = 0; node < nodes; ++node)
    {
        threader_execute_in_numa(node, [&]() {
            const int expected = currentOsNumaNode();
            threader_for(n, 1, [&](int64_t i) {
                (void)i;
                if (currentOsNumaNode() != expected) leftNode.store(true, std::memory_order_relaxed);
            });
            osNodeOfArena[static_cast<size_t>(node)] = expected;
        });
    }

    REQUIRE(!leftNode.load(std::memory_order_relaxed));
    std::vector<int> sorted = osNodeOfArena;
    std::sort(sorted.begin(), sorted.end());
    REQUIRE(sorted.front() >= 0);
    REQUIRE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
}
#endif

#if !(defined DAAL_THREAD_PINNING_DISABLED)
/// Runs `func` in the thread pinner's arena.
template <typename F>
class PinnedTask : public services::internal::thread_pinner_task_t
{
public:
    explicit PinnedTask(const F & func) : _func(func) {}
    void operator()() override { _func(); }

private:
    const F & _func;
};

TEST("threads pinned by enableThreadPinning see one node and run threader_execute_in_numa inline", "[threading][unit]")
{
    const int nodes = threader_numa_count();
    if (nodes < 2) return;

    services::Environment::getInstance()->enableThreadPinning(true);
    services::internal::thread_pinner_t * pinner = services::internal::getThreadPinner(false, read_topology, delete_topology);
    if (pinner == nullptr || !pinner->get_pinning())
    {
        services::Environment::getInstance()->enableThreadPinning(false);
        return;
    }

    constexpr int64_t n = 10000;
    std::atomic<bool> sawManyNodes { false };
    std::atomic<bool> leftThread { false };
    std::atomic<int64_t> calls { 0 };
    const auto body = [&]() {
        threader_for(n, 1, [&](int64_t) {
            if (threader_numa_count() != 1) sawManyNodes.store(true, std::memory_order_relaxed);
            const std::thread::id caller = std::this_thread::get_id();
            threader_execute_in_numa(nodes - 1, [&]() {
                if (std::this_thread::get_id() != caller) leftThread.store(true, std::memory_order_relaxed);
                calls.fetch_add(1, std::memory_order_relaxed);
            });
        });
    };
    PinnedTask<decltype(body)> task(body);
    pinner->execute(task);
    services::Environment::getInstance()->enableThreadPinning(false);

    REQUIRE(!sawManyNodes.load(std::memory_order_relaxed));
    REQUIRE(!leftThread.load(std::memory_order_relaxed));
    REQUIRE(calls.load(std::memory_order_relaxed) == n);
    // Leaving the pinner's arena unpins the thread, so it sees every node again.
    REQUIRE(threader_numa_count() == nodes);
}
#endif

} // namespace daal::test
