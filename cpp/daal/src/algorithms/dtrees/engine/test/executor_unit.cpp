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
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "services/env_detect.h"
#include "src/algorithms/dtrees/engine/executor.h"
#include "src/algorithms/dtrees/engine/philox_counter.h"

namespace daal::algorithms::dtrees::engine::internal::test
{
constexpr auto cpu = DAAL_BASE_CPU;

struct MockItem
{
    size_t iTree       = 0;
    uint64_t key       = 0;
    size_t rows        = 0;
    size_t depth       = 0;
    double improvement = 0;
};

struct NodeRecord
{
    uint64_t key;
    size_t rows;
    size_t depth;
    bool leaf;
    bool operator<(const NodeRecord & o) const { return key < o.key; }
    bool operator==(const NodeRecord & o) const { return key == o.key && rows == o.rows && depth == o.depth && leaf == o.leaf; }
};

// The tree shape is a function of the node keys only, as for the real engine with philox.
struct TreeShape
{
    size_t rootRows;
    size_t maxDepth;

    uint64_t rootKey(size_t iTree) const { return mix64<cpu>(iTree + 1); }
    bool splittable(const MockItem & item) const { return item.rows >= 2 && item.depth < maxDepth && mix64<cpu>(item.key ^ 1) % 5 != 0; }
    // Few distinct values, so best-first trees meet ties.
    double improvement(const MockItem & item) const { return double(mix64<cpu>(item.key ^ 2) % 64); }
    void split(const MockItem & item, MockItem & left, MockItem & right) const
    {
        const size_t nLeft = 1 + mix64<cpu>(item.key ^ 3) % (item.rows - 1);
        left               = { item.iTree, childNodeKey<cpu>(item.key, 0), nLeft, item.depth + 1, 0 };
        right              = { item.iTree, childNodeKey<cpu>(item.key, 1), item.rows - nLeft, item.depth + 1, 0 };
    }
};

struct TreeLog
{
    std::mutex mutex; // evaluate, commit and makeLeaf of one tree run concurrently
    std::vector<NodeRecord> nodes;
    std::vector<uint64_t> commits;
    int begun    = 0;
    int ended    = 0;
    bool builtOk = false;
};

class MockBuilder
{
public:
    using WorkItem = MockItem;
    struct TreeContext
    {
        bool initialized = false;
    };

    MockBuilder(const TreeShape & shape, size_t nTrees, uint64_t failKey = 0) : _shape(shape), _failKey(failKey)
    {
        for (size_t i = 0; i < nTrees; ++i) logs.emplace_back(new TreeLog);
    }

    services::Status initContext(TreeContext & ctx)
    {
        ctx.initialized = true;
        ++contexts;
        return services::Status();
    }

    services::Status beginTree(TreeContext & ctx, size_t iTree, MockItem & root)
    {
        if (!ctx.initialized) contractViolated = true;
        ++logs[iTree]->begun;
        const size_t now = ++inFlight;
        size_t seen      = maxInFlight.load();
        while (now > seen && !maxInFlight.compare_exchange_weak(seen, now))
        {
        }
        root = { iTree, _shape.rootKey(iTree), _shape.rootRows, 0, 0 };
        return services::Status();
    }

    services::Status evaluate(TreeContext &, MockItem & item, bool & splittable)
    {
        splittable = _shape.splittable(item);
        if (splittable) item.improvement = _shape.improvement(item);
        return services::Status();
    }

    services::Status commit(TreeContext &, MockItem & item, MockItem & left, MockItem & right)
    {
        if (item.key == _failKey) return services::Status(services::ErrorIncorrectParameter);
        _shape.split(item, left, right);
        TreeLog & log = *logs[item.iTree];
        std::lock_guard<std::mutex> lock(log.mutex);
        log.nodes.push_back({ item.key, item.rows, item.depth, false });
        log.commits.push_back(item.key);
        return services::Status();
    }

    services::Status makeLeaf(TreeContext &, MockItem & item)
    {
        TreeLog & log = *logs[item.iTree];
        std::lock_guard<std::mutex> lock(log.mutex);
        log.nodes.push_back({ item.key, item.rows, item.depth, true });
        return services::Status();
    }

    services::Status endTree(TreeContext &, size_t iTree, const services::Status & built)
    {
        ++logs[iTree]->ended;
        logs[iTree]->builtOk = built.ok();
        --inFlight;
        return built;
    }

    size_t rows(const MockItem & item) const { return item.rows; }
    double improvement(const MockItem & item) const { return item.improvement; }
    uint64_t nodeKey(const MockItem & item) const { return item.key; }

    std::vector<std::unique_ptr<TreeLog> > logs;
    std::atomic<size_t> contexts { 0 };
    std::atomic<size_t> inFlight { 0 };
    std::atomic<size_t> maxInFlight { 0 };
    std::atomic<bool> contractViolated { false };

private:
    TreeShape _shape;
    uint64_t _failKey;
};

std::vector<NodeRecord> referenceDepthFirst(const TreeShape & shape, size_t iTree)
{
    std::vector<NodeRecord> nodes;
    std::vector<MockItem> stack = { { iTree, shape.rootKey(iTree), shape.rootRows, 0, 0 } };
    while (!stack.empty())
    {
        const MockItem item = stack.back();
        stack.pop_back();
        const bool split = shape.splittable(item);
        nodes.push_back({ item.key, item.rows, item.depth, !split });
        if (!split) continue;
        MockItem left, right;
        shape.split(item, left, right);
        stack.push_back(left);
        stack.push_back(right);
    }
    std::sort(nodes.begin(), nodes.end());
    return nodes;
}

// Best-first in the order the executor promises: largest improvement first, ties to the smaller key.
void referenceBestFirst(const TreeShape & shape, size_t iTree, size_t maxLeafNodes, std::vector<NodeRecord> & nodes, std::vector<uint64_t> & commits)
{
    auto less = [&](const MockItem & a, const MockItem & b) {
        const double ia = shape.improvement(a);
        const double ib = shape.improvement(b);
        return ia < ib || (ia == ib && a.key > b.key);
    };
    std::priority_queue<MockItem, std::vector<MockItem>, decltype(less)> heap(less);
    const MockItem root = { iTree, shape.rootKey(iTree), shape.rootRows, 0, 0 };
    nodes.clear();
    commits.clear();
    if (!shape.splittable(root))
    {
        nodes.push_back({ root.key, root.rows, root.depth, true });
        return;
    }
    heap.push(root);
    size_t budget = maxLeafNodes - 1;
    while (!heap.empty())
    {
        const MockItem item = heap.top();
        heap.pop();
        if (!budget)
        {
            nodes.push_back({ item.key, item.rows, item.depth, true });
            continue;
        }
        --budget;
        nodes.push_back({ item.key, item.rows, item.depth, false });
        commits.push_back(item.key);
        MockItem children[2];
        shape.split(item, children[0], children[1]);
        for (const auto & child : children)
        {
            if (shape.splittable(child))
                heap.push(child);
            else
                nodes.push_back({ child.key, child.rows, child.depth, true });
        }
    }
    std::sort(nodes.begin(), nodes.end());
}

class ThreadCountGuard
{
public:
    explicit ThreadCountGuard(size_t nThreads) : _saved(services::Environment::getInstance()->getNumberOfThreads())
    {
        services::Environment::getInstance()->setNumberOfThreads(nThreads);
    }
    ~ThreadCountGuard() { services::Environment::getInstance()->setNumberOfThreads(_saved); }

private:
    size_t _saved;
};

void checkLifecycle(const MockBuilder & builder, const ExecutorParams & params)
{
    REQUIRE(!builder.contractViolated);
    REQUIRE(builder.inFlight == 0);
    for (const auto & log : builder.logs)
    {
        REQUIRE(log->begun == 1);
        REQUIRE(log->ended == 1);
        REQUIRE(log->builtOk);
    }
    if (params.treesInFlight)
    {
        REQUIRE(builder.contexts <= params.treesInFlight);
        REQUIRE(builder.maxInFlight <= params.treesInFlight);
    }
}

constexpr size_t nTreesTested = 12;
const TreeShape shape         = { 3000, 14 };

TEST("pushMaxHeap and popMaxHeap pop in descending order", "[unit]")
{
    std::mt19937 gen(777);
    std::vector<int> values(1000);
    for (auto & v : values) v = int(gen() % 100);
    std::vector<int> heap;
    for (int v : values)
    {
        heap.push_back(v);
        daal::algorithms::internal::pushMaxHeap<cpu>(heap.begin(), heap.end(), std::less<int>());
    }
    std::vector<int> popped;
    for (auto end = heap.end(); end != heap.begin(); --end)
    {
        daal::algorithms::internal::popMaxHeap<cpu>(heap.begin(), end, std::less<int>());
        popped.push_back(*(end - 1));
    }
    std::sort(values.rbegin(), values.rend());
    REQUIRE(popped == values);
}

TEST("depth-first trees do not depend on threads, serial subtrees or trees in flight", "[unit]")
{
    std::vector<std::vector<NodeRecord> > expected;
    for (size_t i = 0; i < nTreesTested; ++i) expected.push_back(referenceDepthFirst(shape, i));

    for (size_t nThreads : { 1, 2, 8 })
    {
        ThreadCountGuard threads(nThreads);
        for (size_t serialRows : { size_t(0), size_t(64), std::numeric_limits<size_t>::max() })
        {
            for (size_t treesInFlight : { 0, 1, 3 })
            {
                MockBuilder builder(shape, nTreesTested);
                ExecutorParams params;
                params.nTrees            = nTreesTested;
                params.serialSubtreeRows = serialRows;
                params.treesInFlight     = treesInFlight;
                Executor<MockBuilder, cpu> executor(builder, params);
                REQUIRE(executor.run().ok());

                checkLifecycle(builder, params);
                for (size_t i = 0; i < nTreesTested; ++i)
                {
                    auto nodes = builder.logs[i]->nodes;
                    std::sort(nodes.begin(), nodes.end());
                    REQUIRE(nodes == expected[i]);
                }
            }
        }
    }
}

TEST("best-first trees expand by improvement, ties to the smaller node key", "[unit]")
{
    for (size_t maxLeafNodes : { 1, 2, 7, 100000 })
    {
        for (size_t nThreads : { 1, 8 })
        {
            ThreadCountGuard threads(nThreads);
            MockBuilder builder(shape, nTreesTested);
            ExecutorParams params;
            params.nTrees       = nTreesTested;
            params.maxLeafNodes = maxLeafNodes;
            Executor<MockBuilder, cpu> executor(builder, params);
            REQUIRE(executor.run().ok());

            checkLifecycle(builder, params);
            for (size_t i = 0; i < nTreesTested; ++i)
            {
                std::vector<NodeRecord> expectedNodes;
                std::vector<uint64_t> expectedCommits;
                referenceBestFirst(shape, i, maxLeafNodes, expectedNodes, expectedCommits);
                auto nodes = builder.logs[i]->nodes;
                std::sort(nodes.begin(), nodes.end());
                REQUIRE(nodes == expectedNodes);
                REQUIRE(builder.logs[i]->commits == expectedCommits);
                const size_t nLeaves = size_t(std::count_if(nodes.begin(), nodes.end(), [](const NodeRecord & r) { return r.leaf; }));
                REQUIRE(nLeaves <= maxLeafNodes);
            }
        }
    }
}

TEST("no trees means no runner contexts", "[unit]")
{
    MockBuilder builder(shape, 0);
    ExecutorParams params;
    Executor<MockBuilder, cpu> executor(builder, params);
    REQUIRE(executor.run().ok());
    REQUIRE(builder.contexts == 0);
}

TEST("a failed split stops the fit and every started tree is ended", "[unit]")
{
    constexpr size_t failingTree = 5;
    for (size_t maxLeafNodes : { 0, 50 })
    {
        MockBuilder builder(shape, nTreesTested, shape.rootKey(failingTree));
        ExecutorParams params;
        params.nTrees       = nTreesTested;
        params.maxLeafNodes = maxLeafNodes;
        Executor<MockBuilder, cpu> executor(builder, params);
        REQUIRE(!executor.run().ok());

        REQUIRE(builder.inFlight == 0);
        REQUIRE(builder.logs[failingTree]->begun == 1);
        REQUIRE(!builder.logs[failingTree]->builtOk);
        for (const auto & log : builder.logs) REQUIRE(log->ended == log->begun);
    }
}

TEST("the default serial subtree fits half of L2", "[unit]")
{
    REQUIRE(defaultSerialSubtreeRows<cpu>(0) == 0);
    for (size_t bytesPerRow : { 64, 128, 832 })
    {
        const size_t rows = defaultSerialSubtreeRows<cpu>(bytesPerRow);
        REQUIRE(rows * bytesPerRow <= services::internal::getL2CacheSize() / serialSubtreeL2Divisor);
    }
}

} // namespace daal::algorithms::dtrees::engine::internal::test
