/* file: prefetch.h */
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
//  Software prefetch for passes that read rows through an index list. Node row
//  indices ascend with gaps the hardware prefetcher does not follow.
//--
*/

#ifndef __DTREES_ENGINE_PREFETCH_H__
#define __DTREES_ENGINE_PREFETCH_H__

#include <cstddef>
#include <cstdint>

#include "src/services/service_defines.h"

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

// Cache line size of the supported x86-64, Arm and RISC-V server CPUs; the bin matrix row layout assumes it.
constexpr size_t cacheLineBytes = 64;

/// Cache lines a pass keeps in flight ahead of the row it processes. Every CpuType starts from the 10 rows of one
/// line of ComputeGHSumByRows (gbt_train_hist_kernel.i) until the prefetch benchmark sets it.
template <CpuType cpu>
constexpr size_t defaultPrefetchLines()
{
    return 10;
}

/// How far ahead a pass over rows prefetches (design 4.5), in lines rather than rows, so wide rows do not put more
/// lines in flight than the fill buffers hold.
template <CpuType cpu>
struct PrefetchBudget
{
    size_t lines   = defaultPrefetchLines<cpu>(); // 0: no prefetch
    size_t l2Bytes = 0;                           // a pass whose lines fit half of L2 after the parent's pass does not prefetch

    /// Rows of lead for n rows of linesPerRow prefetched lines each: 0 if their lines fit half of L2, else about
    /// lines / linesPerRow, at least 1.
    size_t rows(size_t n, size_t linesPerRow) const
    {
        const size_t perRow = linesPerRow ? linesPerRow : 1;
        if (!lines || n * perRow * cacheLineBytes * 2 < l2Bytes) return 0;
        return lines / perRow ? lines / perRow : 1;
    }
};

/**
 *  \brief Runs body(i) for i in [0, n) in order and prefetch(i) distance rows before body(i)
 *
 *  A prologue prefetches rows [0, distance), the main loop prefetches row i + distance and then
 *  processes row i, and the last distance rows run without prefetch. Every row is prefetched once.
 *  distance == 0 runs body only.
 */
template <CpuType cpu, typename Prefetch, typename Body>
inline void prefetchedLoop(size_t n, size_t distance, const Prefetch & prefetch, const Body & body)
{
    if (distance == 0)
    {
        for (size_t i = 0; i < n; ++i) body(i);
        return;
    }
    if (distance > n) distance = n;
    for (size_t i = 0; i < distance; ++i) prefetch(i);
    const size_t nMain = n - distance;
    size_t i           = 0;
    for (; i < nMain; ++i)
    {
        prefetch(i + distance);
        body(i);
    }
    for (; i < n; ++i) body(i);
}

/// Prefetches the cache lines of one row listed in lines, each an index of a line within the row.
template <CpuType cpu>
inline void prefetchRowLines(const void * row, const uint32_t * lines, size_t nLines)
{
    const char * base = static_cast<const char *>(row);
    for (size_t j = 0; j < nLines; ++j) DAAL_PREFETCH_READ_T0(base + size_t(lines[j]) * cacheLineBytes);
}

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
