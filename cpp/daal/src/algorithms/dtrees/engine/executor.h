/* file: executor.h */
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
//  Schedules tree building: trees are taken from one shared counter by runners
//  inside one arena per NUMA node, and the nodes of a tree run as tasks in the
//  tree's arena.
//--
*/

#ifndef __DTREES_ENGINE_EXECUTOR_H__
#define __DTREES_ENGINE_EXECUTOR_H__

#include <cstddef>
#include <cstdint>

#include "services/daal_atomic_int.h"
#include "src/algorithms/service_error_handling.h"
#include "src/algorithms/service_heap.h"
#include "src/algorithms/service_threading.h"
#include "src/services/service_arrays.h"
#include "src/services/service_defines.h"
#include "src/services/service_environment.h"
#include "src/threading/threading.h"

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace engine
{
namespace internal
{
using daal::internal::CpuType;

// A serial subtree's rows should stay in one core's L2 (design 4.8); the other half is left to
// histograms and scratch. The profiling follow-up sets the final value.
constexpr size_t serialSubtreeL2Divisor = 2;

/// Rows below which a node builds its whole subtree in one task, for rows touching bytesPerRow bytes each.
template <CpuType cpu>
size_t defaultSerialSubtreeRows(size_t bytesPerRow)
{
    const size_t l2 = services::internal::getL2CacheSize();
    return bytesPerRow ? l2 / serialSubtreeL2Divisor / bytesPerRow : 0;
}

struct ExecutorParams
{
    size_t nTrees            = 0;
    size_t maxLeafNodes      = 0; // 0 builds depth-first; otherwise best-first with this many leaves at most
    size_t treesInFlight     = 0; // 0 means no limit
    size_t serialSubtreeRows = 0; // nodes with fewer rows build their subtree in one task; 0 disables it
};

/**
 *  \brief Builds params.nTrees trees through a Builder
 *
 *  Builder provides:
 *  - WorkItem, a copyable node description, and TreeContext, default-constructible per-runner state.
 *  - Status initContext(TreeContext &, int numaNode): called once per runner, inside the arena of NUMA
 *    node numaNode, in [0, threader_numa_count()). Per-tree buffers allocated here are reused by every
 *    tree the runner builds and stay on that node; numaNode also selects the node's data replicas.
 *  - Status beginTree(TreeContext &, size_t iTree, WorkItem & root)
 *  - Status evaluate(TreeContext &, WorkItem &, bool & splittable): finds the item's split. In
 *    best-first trees it also sets the improvement the heap orders by.
 *  - Status commit(TreeContext &, WorkItem &, WorkItem & left, WorkItem & right): applies the split;
 *    each child has at least one row and fewer rows than the parent.
 *  - Status makeLeaf(TreeContext &, WorkItem &)
 *  - Status endTree(TreeContext &, size_t iTree, const Status & built): called after every beginTree,
 *    with the build status; it emits the tree only if built is ok.
 *  - size_t rows(const WorkItem &), double improvement(const WorkItem &), uint64_t nodeKey(const WorkItem &)
 *
 *  evaluate, commit and makeLeaf run concurrently for distinct items of one tree and for items of
 *  different trees. Every node goes through evaluate and then commit or makeLeaf, in that order, by
 *  the same split routine in all scheduling modes, so the trees do not depend on the mode.
 *
 *  Code run here uses daal::tls, not static_tls: thread indices are arena-relative (threading.h).
 */
template <typename Builder, CpuType cpu>
class Executor
{
public:
    using WorkItem    = typename Builder::WorkItem;
    using TreeContext = typename Builder::TreeContext;

    Executor(Builder & builder, const ExecutorParams & params) : _builder(builder), _params(params) {}

    services::Status run()
    {
        _nextTree.set(0);
        _runners.set(0);
        SafeStatus safeStat;
        const int nArenas = threader_numa_count();
        if (nArenas == 1)
        {
            runArena(0, safeStat);
        }
        else
        {
            daal::threader_for(nArenas, 1, [&](size_t iArena) {
                daal::threader_execute_in_numa(static_cast<int>(iArena), [&]() { runArena(static_cast<int>(iArena), safeStat); });
            });
        }
        return safeStat.detach();
    }

private:
    struct TreeState
    {
        TreeState(TreeContext & context) : ctx(context) {}
        TreeContext & ctx;
        daal::task_group group;
        SafeStatus status;
    };

    struct NodeTask
    {
        Executor * executor;
        TreeState * tree;
        WorkItem item;
        void operator()() { executor->processNode(*tree, item); }
    };

    // One runner per thread of the arena; each runs trees one at a time until none are left.
    void runArena(int numaNode, SafeStatus & safeStat)
    {
        const int nRunners = threader_get_max_threads_number();
        daal::task_group group;
        auto runner = [&]() { runTrees(numaNode, safeStat); };
        for (int i = 0; i < nRunners; ++i) group.run(runner);
        group.wait();
    }

    void runTrees(int numaNode, SafeStatus & safeStat)
    {
        // Runners hold the per-tree buffers, so bounding them bounds the trees in flight.
        if (_params.treesInFlight && _runners.inc() > _params.treesInFlight) return;
        TreeContext ctx;
        bool initialized = false;
        while (safeStat.ok())
        {
            const size_t iTree = _nextTree.inc() - 1;
            if (iTree >= _params.nTrees) return;
            if (!initialized)
            {
                const services::Status s = _builder.initContext(ctx, numaNode);
                DAAL_CHECK_STATUS_THR(s);
                initialized = true;
            }
            const services::Status s = buildTree(ctx, iTree);
            DAAL_CHECK_STATUS_THR(s);
        }
    }

    services::Status buildTree(TreeContext & ctx, size_t iTree)
    {
        WorkItem root;
        services::Status s = _builder.beginTree(ctx, iTree, root);
        if (s) s = _params.maxLeafNodes ? buildBestFirst(ctx, root) : buildDepthFirst(ctx, root);
        return _builder.endTree(ctx, iTree, s);
    }

    // Evaluates the item and either makes it a leaf or splits it into left and right.
    services::Status splitNode(TreeContext & ctx, WorkItem & item, WorkItem & left, WorkItem & right, bool & split)
    {
        split              = false;
        services::Status s = _builder.evaluate(ctx, item, split);
        if (!s) return s;
        return split ? _builder.commit(ctx, item, left, right) : _builder.makeLeaf(ctx, item);
    }

    services::Status buildDepthFirst(TreeContext & ctx, WorkItem & root)
    {
        TreeState tree(ctx);
        processNode(tree, root);
        tree.group.wait();
        return tree.status.detach();
    }

    // Spawns the left child and continues with the right one in the same task.
    void processNode(TreeState & tree, WorkItem item)
    {
        while (tree.status.ok())
        {
            if (_builder.rows(item) < _params.serialSubtreeRows)
            {
                tree.status |= buildSerialSubtree(tree.ctx, item);
                return;
            }
            WorkItem left;
            WorkItem right;
            bool split         = false;
            services::Status s = splitNode(tree.ctx, item, left, right, split);
            if (!s)
            {
                tree.status |= s;
                return;
            }
            if (!split) return;
            NodeTask task { this, &tree, left };
            tree.group.run(task);
            item = right;
        }
    }

    // Depth-first from a stack, right child pushed first. The stack holds at most one pending right
    // child per level plus the current node, and a subtree of n rows has fewer than n levels.
    services::Status buildSerialSubtree(TreeContext & ctx, const WorkItem & root)
    {
        const size_t capacity = _builder.rows(root) + 1;
        services::internal::TArray<WorkItem, cpu> stack(capacity);
        DAAL_CHECK_MALLOC(stack.get());
        size_t n   = 0;
        stack[n++] = root;
        while (n)
        {
            WorkItem item = stack[--n];
            WorkItem left;
            WorkItem right;
            bool split         = false;
            services::Status s = splitNode(ctx, item, left, right, split);
            DAAL_CHECK_STATUS_VAR(s);
            if (!split) continue;
            DAAL_ASSERT(n + 2 <= capacity);
            stack[n++] = right;
            stack[n++] = left;
        }
        return services::Status();
    }

    // Expands the evaluated node with the largest improvement, ties to the smaller node key, until
    // maxLeafNodes - 1 splits are made; the nodes left in the heap become leaves.
    services::Status buildBestFirst(TreeContext & ctx, WorkItem & root)
    {
        bool splittable    = false;
        services::Status s = _builder.evaluate(ctx, root, splittable);
        DAAL_CHECK_STATUS_VAR(s);
        if (!splittable) return _builder.makeLeaf(ctx, root);

        // A split pops one node and pushes at most two, so the heap holds at most 1 + splits nodes,
        // and a tree has at most maxLeafNodes leaves and one leaf per row.
        const size_t capacity = _params.maxLeafNodes < _builder.rows(root) ? _params.maxLeafNodes : _builder.rows(root);
        services::internal::TArray<WorkItem, cpu> heap(capacity);
        DAAL_CHECK_MALLOC(heap.get());
        auto less = [&](const WorkItem & a, const WorkItem & b) -> bool {
            const double ia = _builder.improvement(a);
            const double ib = _builder.improvement(b);
            return ia < ib || (ia == ib && _builder.nodeKey(a) > _builder.nodeKey(b));
        };

        size_t n      = 0;
        heap[n++]     = root;
        size_t budget = _params.maxLeafNodes - 1;
        while (n)
        {
            daal::algorithms::internal::popMaxHeap<cpu>(heap.get(), heap.get() + n, less);
            WorkItem item = heap[--n];
            if (!budget)
            {
                s = _builder.makeLeaf(ctx, item);
                DAAL_CHECK_STATUS_VAR(s);
                continue;
            }
            --budget;
            WorkItem children[2];
            s = _builder.commit(ctx, item, children[0], children[1]);
            DAAL_CHECK_STATUS_VAR(s);
            for (auto & child : children)
            {
                s = _builder.evaluate(ctx, child, splittable);
                DAAL_CHECK_STATUS_VAR(s);
                if (!splittable)
                {
                    s = _builder.makeLeaf(ctx, child);
                    DAAL_CHECK_STATUS_VAR(s);
                    continue;
                }
                DAAL_ASSERT(n < capacity);
                heap[n++] = child;
                daal::algorithms::internal::pushMaxHeap<cpu>(heap.get(), heap.get() + n, less);
            }
        }
        return services::Status();
    }

    Builder & _builder;
    const ExecutorParams _params;
    services::Atomic<size_t> _nextTree;
    services::Atomic<size_t> _runners;
};

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
