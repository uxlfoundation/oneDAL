/* file: tree_soa.h */
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
//  The working tree of one tree build: nodes in a pool that node tasks append
//  to concurrently (design 4.14).
//--
*/

#ifndef __DTREES_ENGINE_TREE_SOA_H__
#define __DTREES_ENGINE_TREE_SOA_H__

#include <cstddef>
#include <cstdint>

#include "services/daal_atomic_int.h"
#include "src/algorithms/dtrees/engine/numa_alloc.h"
#include "src/algorithms/service_threading.h"
#include "src/services/service_arrays.h"

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

/// A node of the working tree. Leaves keep their record range, from which emission computes the leaf value.
struct TreeNode
{
    int32_t feature;         // split feature, -1 for a leaf
    uint32_t left;           // split: left child; the right child is left + 1
    double threshold;        // split: rows with x <= threshold (categorical: x == threshold) go left
    double impurity;         // Gini or MSE of the node
    double weightedDecrease; // split: weighted impurity decrease, for MDI importance
    uint32_t start;          // leaf: first record in the response buffer of its depth
    uint32_t nRecords;       // records (unique in-bag rows) in the node
    uint32_t nSamples;       // sum of the records' bootstrap counts
    uint32_t depth;          // a chain of single-row splits can exceed 2^16 levels
    uint8_t categorical;
    uint8_t defaultLeft; // side with more weight, for missing values at prediction
};

// Nodes per pool chunk: 2^16 nodes of 56 bytes, 3.5 MB, so a small tree takes one chunk.
constexpr size_t nodeChunkShift = 16;
constexpr size_t nodeChunkSize  = size_t(1) << nodeChunkShift;

/**
 *  \brief Nodes of one tree: the root is node 0, and each split's children are an adjacent pair
 *
 *  allocatePair may run concurrently from node tasks. Chunks are allocated when first needed, so
 *  memory follows the nodes the tree has, not its capacity.
 */
template <CpuType cpu>
class NodePool
{
public:
    /// Prepares for up to capacity nodes; chunks from an earlier tree are kept and reused.
    services::Status reset(size_t capacity)
    {
        const size_t nChunks = (capacity + nodeChunkSize - 1) / nodeChunkSize;
        if (nChunks > _slots.size())
        {
            // The new tree overwrites every node it uses, so the old chunks are dropped rather than copied.
            DAAL_CHECK_MALLOC(_slots.reset(nChunks) && _chunks.reset(nChunks));
            for (size_t i = 0; i < nChunks; ++i) _slots[i].set(0);
        }
        _capacity = capacity;
        _pairs.set(0);
        return chunkFor(0) ? services::Status() : services::Status(services::ErrorMemoryAllocationFailed);
    }

    /// Allocates two adjacent nodes; returns the first, or 0 on failure (node 0 is the root).
    uint32_t allocatePair()
    {
        const size_t pair  = _pairs.inc() - 1;
        const size_t first = 1 + 2 * pair;
        if (first + 1 >= _capacity || !chunkFor(first) || !chunkFor(first + 1)) return 0;
        return uint32_t(first);
    }

    size_t size() const { return 1 + 2 * _pairs.get(); }

    TreeNode & operator[](size_t i) { return chunkAt(i)[i & (nodeChunkSize - 1)]; }
    const TreeNode & operator[](size_t i) const { return chunkAt(i)[i & (nodeChunkSize - 1)]; }

private:
    TreeNode * chunkAt(size_t i) const { return reinterpret_cast<TreeNode *>(_slots[i >> nodeChunkShift].get()); }

    // The chunk holding node i, allocated on first use. Slots are read without the lock; a chunk is
    // published once, under it.
    TreeNode * chunkFor(size_t i)
    {
        const size_t c = i >> nodeChunkShift;
        if (c >= _slots.size()) return nullptr;
        if (TreeNode * p = chunkAt(i)) return p;
        AutoLock lock(_mutex);
        if (TreeNode * p = chunkAt(i)) return p;
        TreeNode * chunk = _chunks[c].reset(nodeChunkSize);
        _slots[c].set(reinterpret_cast<uintptr_t>(chunk));
        return chunk;
    }

    services::internal::TArray<LargeArray<TreeNode, cpu>, cpu> _chunks;
    services::internal::TArray<services::Atomic<uintptr_t>, cpu> _slots;
    services::Atomic<size_t> _pairs;
    size_t _capacity = 0;
    Mutex _mutex;
};

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
