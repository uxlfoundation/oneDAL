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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "src/threading/threading.h"

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

/// @return `true` if the threading layer is configured to run loops in parallel, so that
///         assertions on the *shape* of the partitioning are meaningful. The loops fall back to a
///         plain sequential pass otherwise.
bool isParallel()
{
    return threader_get_threads_number() > 1;
}

TEST("threader_for visits every index exactly once", "[threading][unit]")
{
    for (const int64_t n : { int64_t(0), int64_t(1), int64_t(17), int64_t(1024), int64_t(100000) })
    {
        for (const int64_t grainSize : { int64_t(1), int64_t(7), int64_t(4096) })
        {
            VisitLog log(n);
            threader_for(n, grainSize, [&](int64_t i) { log.mark(i); });
            REQUIRE(log.visitedExactlyOnce());
        }
    }
}

TEST("threader_for_simple visits every index exactly once", "[threading][unit]")
{
    for (const int64_t n : { int64_t(0), int64_t(1), int64_t(33), int64_t(10000) })
    {
        for (const int64_t grainSize : { int64_t(1), int64_t(16) })
        {
            VisitLog log(n);
            threader_for_simple(n, grainSize, [&](int64_t i) { log.mark(i); });
            REQUIRE(log.visitedExactlyOnce());
        }
    }
}

TEST("threader_for_optional visits every index exactly once, nested or not", "[threading][unit]")
{
    constexpr int64_t n = 5000;

    VisitLog flat(n);
    threader_for_optional(n, 1, [&](int64_t i) { flat.mark(i); });
    REQUIRE(flat.visitedExactlyOnce());

    // Called from inside a parallel region, the loop runs sequentially in the calling thread
    // instead of nesting a second one -- it still has to cover the whole iteration space.
    constexpr int64_t outerCount = 8;
    VisitLog nested(outerCount * n);
    threader_for(outerCount, 1, [&](int64_t outer) { threader_for_optional(n, 1, [&](int64_t inner) { nested.mark(outer * n + inner); }); });
    REQUIRE(nested.visitedExactlyOnce());
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

TEST("threader_for reaches indices beyond INT32_MAX", "[threading][unit]")
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

} // namespace daal::test
