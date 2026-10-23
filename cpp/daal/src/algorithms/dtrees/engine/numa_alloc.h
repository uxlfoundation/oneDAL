/* file: numa_alloc.h */
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
//  Arrays placed on NUMA nodes by first touch: one copy per node, or one copy
//  whose blocks are touched by the arenas in turn.
//--
*/

#ifndef __DTREES_ENGINE_NUMA_ALLOC_H__
#define __DTREES_ENGINE_NUMA_ALLOC_H__

#include <cstddef>
#include <cstdint>

#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_memory.h"
#include "src/services/service_arrays.h"
#include "src/services/service_defines.h"
#include "src/services/service_utils.h"
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

/**
 *  \brief Uninitialized array for large per-fit and per-tree buffers
 *
 *  Elements are not constructed, so the pages are first touched by the code that fills them, on
 *  that thread's NUMA node. Buffers are not advised for transparent huge pages: on a host with
 *  defrag=madvise and fragmented memory that made binning Higgs 3-4x slower (synchronous compaction
 *  on every fault); whether huge pages pay elsewhere is part of the design's NUMA measurements.
 */
template <typename T, CpuType cpu>
class LargeArray
{
public:
    LargeArray() = default;
    ~LargeArray() { release(); }
    LargeArray(const LargeArray &)             = delete;
    LargeArray & operator=(const LargeArray &) = delete;

    /// Replaces the contents with n uninitialized elements; returns nullptr if n is 0 or allocation fails.
    T * reset(size_t n)
    {
        release();
        if (!n || n > ~size_t(0) / sizeof(T)) return nullptr;
        _data = services::internal::service_malloc<T, cpu>(n);
        if (!_data) return nullptr;
        _size = n;
        return _data;
    }

    T * get() { return _data; }
    const T * get() const { return _data; }
    size_t size() const { return _size; }
    T & operator[](size_t i) { return _data[i]; }
    const T & operator[](size_t i) const { return _data[i]; }

private:
    void release()
    {
        if (_data) services::internal::service_free<T, cpu>(_data);
        _data = nullptr;
        _size = 0;
    }

    T * _data    = nullptr;
    size_t _size = 0;
};

/// The NUMA nodes a fit spreads its data and work over, and whether each node holds its own copy of the data.
struct NumaPlacement
{
    size_t nodes   = 1;     // 1: one copy, used from the calling thread's arena, without querying the topology
    bool replicate = false; // a copy per node when nodes > 1

    /// Every NUMA node of the host, which queries the topology (hwloc through TBB), replicated if requested.
    static NumaPlacement host(bool replicate) { return { size_t(threader_numa_count()), replicate }; }
};

/**
 *  \brief An array of n elements held once per NUMA node, or once in total
 *
 *  Pages are placed by the thread that first writes them, so the copies are filled through
 *  forEachBlock: a copy per node is written inside that node's arena, and a single copy on a
 *  multi-node host is written by the arenas in turn, one block each.
 */
template <typename T, CpuType cpu>
class NumaReplicated
{
public:
    /// Allocates one copy per node of the placement if it replicates over more than one node, else one copy.
    services::Status allocate(size_t n, const NumaPlacement & placement)
    {
        _nNodes  = placement.nodes ? placement.nodes : 1;
        _nCopies = (placement.replicate && _nNodes > 1) ? _nNodes : 1;
        _size    = n;
        _copies.reset(_nCopies);
        DAAL_CHECK_MALLOC(_copies.get());
        for (size_t i = 0; i < _nCopies; ++i)
        {
            if (n) DAAL_CHECK_MALLOC(_copies[i].reset(n));
        }
        return services::Status();
    }

    size_t size() const { return _size; }
    size_t copies() const { return _nCopies; }

    /// The copy local to NUMA node numaNode; with one copy, that copy.
    T * data(int numaNode) { return _copies[copyIndex(numaNode)].get(); }
    const T * data(int numaNode) const { return _copies[copyIndex(numaNode)].get(); }

    /**
     *  \brief Runs func(iBlock) for every block in [0, nBlocks) so that copy iCopy is first touched on its node
     *
     *  func returns a services::Status; the first error is returned.
     */
    template <typename Func>
    services::Status forEachBlock(size_t iCopy, size_t nBlocks, const Func & func) const
    {
        SafeStatus safeStat;
        auto runBlocks = [&](size_t first, size_t step) {
            const size_t n = first < nBlocks ? (nBlocks - first + step - 1) / step : 0;
            daal::threader_for(n, 1, [&](size_t i) {
                const services::Status s = func(first + i * step);
                DAAL_CHECK_STATUS_THR(s);
            });
        };
        if (_nCopies > 1)
        {
            daal::threader_execute_in_numa(static_cast<int>(iCopy), [&]() { runBlocks(0, 1); });
        }
        else if (_nNodes > 1)
        {
            const size_t nNodes = _nNodes;
            daal::threader_for(nNodes, 1,
                               [&](size_t node) { daal::threader_execute_in_numa(static_cast<int>(node), [&]() { runBlocks(node, nNodes); }); });
        }
        else
        {
            runBlocks(0, 1);
        }
        return safeStat.detach();
    }

    /// Copies copy 0 into the other copies, each inside its own node's arena.
    services::Status replicate(size_t blockSize)
    {
        const size_t nBlocks = (_size + blockSize - 1) / blockSize;
        for (size_t iCopy = 1; iCopy < _nCopies; ++iCopy)
        {
            const T * src      = _copies[0].get();
            T * dst            = _copies[iCopy].get();
            services::Status s = forEachBlock(iCopy, nBlocks, [&](size_t iBlock) {
                const size_t begin = iBlock * blockSize;
                const size_t end   = begin + blockSize < _size ? begin + blockSize : _size;
                services::internal::tmemcpy<T, cpu>(dst + begin, src + begin, end - begin);
                return services::Status();
            });
            DAAL_CHECK_STATUS_VAR(s);
        }
        return services::Status();
    }

private:
    size_t copyIndex(int numaNode) const { return (numaNode >= 0 && size_t(numaNode) < _nCopies) ? size_t(numaNode) : 0; }

    services::internal::TArray<LargeArray<T, cpu>, cpu> _copies;
    size_t _size    = 0;
    size_t _nCopies = 0;
    size_t _nNodes  = 1;
};

/// Rows read per block when a table is copied or scanned.
constexpr size_t tableBlockRows = 4096;

/**
 *  \brief Row-major copy of x in algorithmFPType, one per NUMA node (design 4.9)
 *
 *  Each copy is read from x through ReadRows blocks inside its own node's arena.
 */
template <typename algorithmFPType, CpuType cpu>
services::Status copyTableRows(const data_management::NumericTable & x, const NumaPlacement & placement, NumaReplicated<algorithmFPType, cpu> & out)
{
    const size_t nRows = x.getNumberOfRows();
    const size_t nCols = x.getNumberOfColumns();
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, nCols);
    services::Status s = out.allocate(nRows * nCols, placement);
    DAAL_CHECK_STATUS_VAR(s);
    const size_t nBlocks = (nRows + tableBlockRows - 1) / tableBlockRows;
    for (size_t iCopy = 0; iCopy < out.copies(); ++iCopy)
    {
        algorithmFPType * dst = out.data(static_cast<int>(iCopy));
        s                     = out.forEachBlock(iCopy, nBlocks, [&](size_t iBlock) {
            const size_t begin = iBlock * tableBlockRows;
            const size_t n     = begin + tableBlockRows < nRows ? tableBlockRows : nRows - begin;
            daal::internal::ReadRows<algorithmFPType, cpu> rows(const_cast<data_management::NumericTable *>(&x), begin, n);
            DAAL_CHECK_BLOCK_STATUS(rows);
            services::internal::tmemcpy<algorithmFPType, cpu>(dst + begin * nCols, rows.get(), n * nCols);
            return services::Status();
        });
        DAAL_CHECK_STATUS_VAR(s);
    }
    return services::Status();
}

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
