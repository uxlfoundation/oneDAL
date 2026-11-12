/* file: block_array.h */
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
//  Uninitialized arrays for a fit's large buffers, and the table data of a fit
//  filled in parallel blocks.
//--
*/

#ifndef __DTREES_ENGINE_BLOCK_ARRAY_H__
#define __DTREES_ENGINE_BLOCK_ARRAY_H__

#include <cstddef>
#include <cstdint>

#include "data_management/data/homogen_numeric_table.h"
#include "src/algorithms/dtrees/engine/prefetch.h"
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
 *  Elements are not constructed, so the pages are first touched by the code that fills them. Buffers
 *  are not advised for transparent huge pages: on a host with defrag=madvise and fragmented memory
 *  that made binning Higgs 3-4x slower (synchronous compaction on every fault).
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

/**
 *  \brief An array of n elements of a fit's table data, owned or borrowed from the caller
 *
 *  Pages are placed by the thread that first writes them, so an owned array is filled through forEachBlock.
 */
template <typename T, CpuType cpu>
class BlockArray
{
public:
    services::Status allocate(size_t n)
    {
        _borrowed = nullptr;
        _size     = n;
        if (n) DAAL_CHECK_MALLOC(_data.reset(n));
        return services::Status();
    }

    /// The caller's n elements at p, read only, which must outlive this object's use.
    void borrow(const T * p, size_t n)
    {
        _data.reset(0);
        _borrowed = p;
        _size     = n;
    }

    size_t size() const { return _size; }
    bool borrowed() const { return _borrowed != nullptr; }

    /// A borrowed array is read through the const overload only.
    T * data()
    {
        DAAL_ASSERT(!_borrowed);
        return _borrowed ? nullptr : _data.get();
    }
    const T * data() const { return _borrowed ? _borrowed : _data.get(); }

    /**
     *  \brief Runs func(iBlock) for every block in [0, nBlocks) in parallel
     *
     *  func returns a services::Status; the first error is returned.
     */
    template <typename Func>
    services::Status forEachBlock(size_t nBlocks, const Func & func) const
    {
        return forEachTile(nBlocks, 1, [&](size_t iBlock, size_t) { return func(iBlock); });
    }

    /// forEachBlock over nTiles tasks func(iBlock, iTile) per block.
    template <typename Func>
    services::Status forEachTile(size_t nBlocks, size_t nTiles, const Func & func) const
    {
        SafeStatus safeStat;
        // Tile-major order, so threads that start together work on different blocks.
        daal::threader_for(nBlocks * nTiles, 1, [&](size_t i) {
            const services::Status s = func(i % nBlocks, i / nBlocks);
            DAAL_CHECK_STATUS_THR(s);
        });
        return safeStat.detach();
    }

private:
    LargeArray<T, cpu> _data;
    const T * _borrowed = nullptr;
    size_t _size        = 0;
};

/// Rows read per block when a table is scanned with fixed blocks, and the most rows of a passBlockRows block.
constexpr size_t tableBlockRows = 4096;

// Tasks per thread a pass aims for, and the fewest input bytes a task reads. Untuned initial values.
constexpr size_t passTasksPerThread = 4;
constexpr size_t minTaskBytes       = size_t(64) << 10;

/**
 *  \brief Rows per block of a parallel pass over nRows rows, with tasksPerBlock tasks per block
 *
 *  About passTasksPerThread tasks per thread, each reading at least minTaskBytes at taskRowBytes per row, and at
 *  most tableBlockRows rows. Blocks start on a cache line of the written rows of writeRowBytes. The result depends
 *  on the thread count: passes that add floating-point values per block keep fixed blocks instead.
 */
template <CpuType cpu>
inline size_t passBlockRows(size_t nRows, size_t nThreads, size_t tasksPerBlock, size_t taskRowBytes, size_t writeRowBytes)
{
    const size_t tasks    = (nThreads ? nThreads : 1) * passTasksPerThread;
    const size_t perBlock = tasksPerBlock ? tasksPerBlock : 1;
    const size_t nBlocks  = (tasks + perBlock - 1) / perBlock;
    const size_t minRows  = taskRowBytes ? (minTaskBytes + taskRowBytes - 1) / taskRowBytes : tableBlockRows;
    size_t rows           = (nRows + nBlocks - 1) / nBlocks;
    rows                  = rows > minRows ? rows : minRows;
    rows                  = rows < tableBlockRows ? rows : tableBlockRows;
    // unit rows of writeRowBytes are whole cache lines; a power of two up to tableBlockRows, so it divides it.
    size_t low        = writeRowBytes & (~writeRowBytes + 1);
    low               = low && low < cacheLineBytes ? low : cacheLineBytes;
    const size_t unit = cacheLineBytes / low;
    return (rows + unit - 1) / unit * unit;
}

/**
 *  \brief Row-major x in algorithmFPType (design 4.9)
 *
 *  A homogen table of algorithmFPType, already row-major, is borrowed; its pages stay where the caller placed them.
 *  Other tables are copied through ReadRows blocks.
 */
template <typename algorithmFPType, CpuType cpu>
services::Status copyTableRows(const data_management::NumericTable & x, BlockArray<algorithmFPType, cpu> & out)
{
    const size_t nRows = x.getNumberOfRows();
    const size_t nCols = x.getNumberOfColumns();
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, nCols);
    const auto * h = dynamic_cast<const data_management::HomogenNumericTable<algorithmFPType> *>(&x);
    if (h && h->getArray())
    {
        out.borrow(h->getArray(), nRows * nCols);
        return services::Status();
    }
    services::Status s = out.allocate(nRows * nCols);
    DAAL_CHECK_STATUS_VAR(s);
    const size_t rowBytes  = nCols * sizeof(algorithmFPType);
    const size_t blockRows = passBlockRows<cpu>(nRows, threader_get_threads_number(), 1, rowBytes, rowBytes);
    const size_t nBlocks   = (nRows + blockRows - 1) / blockRows;
    algorithmFPType * dst  = out.data();
    return out.forEachBlock(nBlocks, [&](size_t iBlock) {
        const size_t begin = iBlock * blockRows;
        const size_t n     = begin + blockRows < nRows ? blockRows : nRows - begin;
        daal::internal::ReadRows<algorithmFPType, cpu> rows(const_cast<data_management::NumericTable *>(&x), begin, n);
        DAAL_CHECK_BLOCK_STATUS(rows);
        services::internal::tmemcpy<algorithmFPType, cpu>(dst + begin * nCols, rows.get(), n * nCols);
        return services::Status();
    });
}

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
