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

struct ExecutorParams
{
    size_t nTrees        = 0;
    size_t maxLeafNodes  = 0; // 0 builds depth-first; otherwise best-first with this many leaves at most
    size_t treesInFlight = 0; // 0 means no limit
    size_t numaNodes     = 0; // arenas: 0 queries the topology for every NUMA node; 1 runs in the calling arena
};

/**
 *  \brief Builds params.nTrees trees through a Builder
 *
 *  Builder provides:
 *  - WorkItem, a copyable node description, and TreeContext, default-constructible per-runner state.
 *  - Status initContext(TreeContext &, int numaNode): called once per runner, inside the arena of NUMA
 *    node numaNode, in [0, ExecutorParams::numaNodes). Per-tree buffers allocated here are reused by every
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
        SafeStatus safeStat;
        const size_t inFlight = treesInFlight(_params);
        if (!inFlight) return services::Status();
        const size_t numaNodes = _params.numaNodes ? _params.numaNodes : size_t(threader_numa_count());
        const size_t nArenas   = numaNodes < inFlight ? numaNodes : inFlight;
        if (nArenas == 1)
        {
            runArena(0, inFlight, safeStat);
        }
        else
        {
            daal::threader_for(nArenas, 1, [&](size_t iArena) {
                daal::threader_execute_in_numa(static_cast<int>(iArena),
                                               [&]() { runArena(static_cast<int>(iArena), runnersPerArena(inFlight, nArenas, iArena), safeStat); });
            });
        }
        return safeStat.detach();
    }

    /// Trees built at once: nTrees, or fewer with params.treesInFlight.
    static size_t treesInFlight(const ExecutorParams & params)
    {
        return params.treesInFlight && params.treesInFlight < params.nTrees ? params.treesInFlight : params.nTrees;
    }

    /// Runners arena iArena of nArenas starts for inFlight trees in flight, before its thread count caps them.
    static size_t runnersPerArena(size_t inFlight, size_t nArenas, size_t iArena)
    {
        return inFlight / nArenas + (iArena < inFlight % nArenas ? 1 : 0);
    }

private:
    struct TreeState
    {
        TreeState(TreeContext & context) : ctx(context) {}
        TreeContext & ctx;
        daal::task_group group;
        SafeStatus status;
        // Set with the first error. Node tasks read it per node; SafeStatus::ok would take a lock each time.
        services::Atomic<int> failed;

        void add(const services::Status & s)
        {
            if (s) return;
            status |= s;
            failed.set(1);
        }
    };

    struct NodeTask
    {
        Executor * executor;
        TreeState * tree;
        WorkItem item;
        void operator()() { executor->processNode(*tree, item); }
    };

    // Up to nRunners runners, at most one per thread of the arena; each runs trees one at a time until none are
    // left. A runner owns its tree (buffers, begin, wait, emit); the tree's node tasks run on every thread of the
    // arena, so runners beyond the trees in flight would only wake threads, and on a first fit create them, to find
    // no tree.
    void runArena(int numaNode, size_t nRunners, SafeStatus & safeStat)
    {
        const size_t nThreads = size_t(threader_get_max_threads_number());
        if (nRunners > nThreads) nRunners = nThreads;
        daal::task_group group;
        auto runner = [&]() { runTrees(numaNode, safeStat); };
        for (size_t i = 0; i < nRunners; ++i) group.run(runner);
        group.wait();
    }

    void runTrees(int numaNode, SafeStatus & safeStat)
    {
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
        while (!tree.failed.get())
        {
            WorkItem left;
            WorkItem right;
            bool split         = false;
            services::Status s = splitNode(tree.ctx, item, left, right, split);
            if (!s)
            {
                tree.add(s);
                return;
            }
            if (!split) return;
            NodeTask task { this, &tree, left };
            tree.group.run(task);
            item = right;
        }
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
};

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
