/* file: phase_hist.i */
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
//  Histogram phase of the tree builder (design 4.6): per-node histograms of the
//  drawn features over the bin matrix, built in chunks for large nodes.
//  Included by tree_builder.i.
//--
*/

#ifndef __DTREES_ENGINE_PHASE_HIST_I__
#define __DTREES_ENGINE_PHASE_HIST_I__

#include "src/algorithms/dtrees/engine/bin_matrix.h"
#include "src/algorithms/dtrees/engine/cost_function.h"
#include "src/algorithms/dtrees/engine/prefetch.h"
#include "src/algorithms/dtrees/engine/response_set.h"

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
namespace hist
{
using services::internal::TArray;

// Lanes of a plane row are padded to 16, so each bin's row of u32 or f32 statistics starts on a cache line.
constexpr size_t laneAlign = 16;

/// Per-bin statistics of a record layout (design 4.10): record counts u32; classification class weights u32
/// (bootstrap counts) or f32 (user weights); regression weight f32 and weighted response sum f64.
template <typename Record>
struct Layout;
template <>
struct Layout<ClsCountRecord>
{
    using Stat                           = uint32_t;
    static constexpr bool classification = true;
};
template <typename algorithmFPType>
struct Layout<ClsWeightRecord<algorithmFPType> >
{
    using Stat                           = float;
    static constexpr bool classification = true;
};
template <typename algorithmFPType>
struct Layout<RegCountRecord<algorithmFPType> >
{
    using Stat                           = float;
    static constexpr bool classification = false;
};
template <typename algorithmFPType>
struct Layout<RegWeightRecord<algorithmFPType> >
{
    using Stat                           = float;
    static constexpr bool classification = false;
};

/**
 *  \brief Histograms of a node's lanes (drawn features)
 *
 *  Each plane is [bin][lane]: records per bin; classification, one plane per class of class weights;
 *  regression, a plane of weights and a plane of weighted response sums.
 */
template <typename Record, CpuType cpu>
struct Planes
{
    using Stat = typename Layout<Record>::Stat;

    size_t nBins    = 0;
    size_t stride   = 0; // lanes, padded
    size_t nClasses = 0;
    LargeArray<uint32_t, cpu> records;
    LargeArray<Stat, cpu> stat; // classification: [class][bin][lane]; regression: weight [bin][lane]
    LargeArray<double, cpu> sumY;

    bool init(size_t bins, size_t lanes, size_t classes)
    {
        nBins          = bins;
        stride         = (lanes + laneAlign - 1) / laneAlign * laneAlign;
        nClasses       = classes;
        const size_t n = nBins * stride;
        const size_t s = Layout<Record>::classification ? nClasses * n : n;
        if (!records.reset(n) || !stat.reset(s)) return false;
        if (!Layout<Record>::classification && !sumY.reset(n)) return false;
        for (size_t i = 0; i < n; ++i) records[i] = 0;
        for (size_t i = 0; i < s; ++i) stat[i] = 0;
        if (!Layout<Record>::classification)
            for (size_t i = 0; i < n; ++i) sumY[i] = 0;
        return true;
    }

    void add(const Planes & o)
    {
        const size_t n = nBins * stride;
        const size_t s = Layout<Record>::classification ? nClasses * n : n;
        for (size_t i = 0; i < n; ++i) records[i] += o.records[i];
        for (size_t i = 0; i < s; ++i) stat[i] += o.stat[i];
        if (!Layout<Record>::classification)
            for (size_t i = 0; i < n; ++i) sumY[i] += o.sumY[i];
    }

    size_t at(size_t bin, size_t lane) const { return bin * stride + lane; }
    Stat classStat(size_t c, size_t bin, size_t lane) const { return stat[c * nBins * stride + at(bin, lane)]; }
};

/// Adds records [begin, end) to the planes, reading each record's bin row once and prefetching distance rows
/// ahead the lines that hold the lanes' features.
template <typename Record, typename BinT, CpuType cpu>
void accumulate(const Record * rec, size_t begin, size_t end, const BinMatrix<cpu> & bins, int numaNode, const uint32_t * features, size_t nLanes,
                const uint32_t * lines, size_t nLines, size_t distance, Planes<Record, cpu> & h)
{
    uint32_t * records                        = h.records.get();
    typename Planes<Record, cpu>::Stat * stat = h.stat.get();
    double * sumY                             = h.sumY.get();
    const size_t planeSize                    = h.nBins * h.stride;
    const size_t stride                       = h.stride;
    prefetchedLoop<cpu>(
        end - begin, distance, [&](size_t i) { prefetchRowLines<cpu>(bins.template row<uint8_t>(numaNode, rec[begin + i].row), lines, nLines); },
        [&](size_t i) {
            const Record & r = rec[begin + i];
            const BinT * b   = bins.template row<BinT>(numaNode, r.row);
            if constexpr (Layout<Record>::classification)
            {
                const size_t offset = size_t(recordClass(r)) * planeSize;
                const auto weight   = static_cast<typename Planes<Record, cpu>::Stat>(recordWeight(r));
                for (size_t j = 0; j < nLanes; ++j)
                {
                    const size_t idx = size_t(b[features[j]]) * stride + j;
                    ++records[idx];
                    stat[offset + idx] += weight;
                }
            }
            else
            {
                const double w  = recordWeight(r);
                const double wy = w * recordResponse(r);
                for (size_t j = 0; j < nLanes; ++j)
                {
                    const size_t idx = size_t(b[features[j]]) * stride + j;
                    ++records[idx];
                    stat[idx] += float(w);
                    sumY[idx] += wy;
                }
            }
        });
}

/// Chunks of a node of n records: ceil(n / chunkRows), at most maxChunks. Boundaries depend on n only.
inline size_t chunkCount(size_t n, size_t chunkRows, size_t maxChunks)
{
    const size_t c = chunkRows ? (n + chunkRows - 1) / chunkRows : 1;
    return c < 1 ? 1 : (c > maxChunks ? maxChunks : c);
}

inline void chunkRange(size_t n, size_t nChunks, size_t c, size_t & begin, size_t & end)
{
    const size_t size = (n + nChunks - 1) / nChunks;
    begin             = c * size < n ? c * size : n;
    end               = begin + size < n ? begin + size : n;
}

/// Ascending list of the cache lines of a bin row that hold the features.
inline size_t rowLines(const uint32_t * features, size_t nLanes, size_t binBytes, uint32_t * lines)
{
    size_t n = 0;
    for (size_t j = 0; j < nLanes; ++j)
    {
        const uint32_t line = uint32_t(size_t(features[j]) * binBytes / cacheLineBytes);
        size_t k            = 0;
        while (k < n && lines[k] < line) ++k;
        if (k < n && lines[k] == line) continue;
        for (size_t m = n; m > k; --m) lines[m] = lines[m - 1];
        lines[k] = line;
        ++n;
    }
    return n;
}

/**
 *  \brief Histograms of nLanes features over a node's n records
 *
 *  One chunk runs in the calling task. Several chunks each fill partial planes, in parallel, reduced by a
 *  fixed pairwise tree over chunk index, so the result does not depend on the thread count.
 */
template <typename Record, CpuType cpu>
services::Status build(const Record * rec, size_t n, const BinMatrix<cpu> & bins, int numaNode, const uint32_t * features, size_t nLanes,
                       size_t nClasses, size_t nChunks, size_t distance, Planes<Record, cpu> & out)
{
    size_t nBins = 0;
    for (size_t j = 0; j < nLanes; ++j)
    {
        const size_t f = features[j];
        const size_t b = bins.nBins(f) + (bins.hasMissing(f) ? 1 : 0);
        nBins          = b > nBins ? b : nBins;
    }
    TArray<uint32_t, cpu> lines(nLanes);
    DAAL_CHECK_MALLOC(lines.get());
    const size_t nLines = rowLines(features, nLanes, bins.binBytes(), lines.get());

    auto pass = [&](size_t begin, size_t end, Planes<Record, cpu> & h) {
        if (bins.binBytes() == 1)
            accumulate<Record, uint8_t, cpu>(rec, begin, end, bins, numaNode, features, nLanes, lines.get(), nLines, distance, h);
        else
            accumulate<Record, uint16_t, cpu>(rec, begin, end, bins, numaNode, features, nLanes, lines.get(), nLines, distance, h);
    };

    DAAL_CHECK_MALLOC(out.init(nBins, nLanes, nClasses));
    if (nChunks <= 1)
    {
        pass(0, n, out);
        return services::Status();
    }
    TArray<Planes<Record, cpu>, cpu> partial(nChunks);
    DAAL_CHECK_MALLOC(partial.get());
    SafeStatus safeStat;
    daal::threader_for(nChunks, 1, [&](size_t c) {
        Planes<Record, cpu> & h = c ? partial[c] : out;
        DAAL_CHECK_THR(c == 0 || h.init(nBins, nLanes, nClasses), services::ErrorMemoryAllocationFailed);
        size_t begin, end;
        chunkRange(n, nChunks, c, begin, end);
        pass(begin, end, h);
    });
    DAAL_CHECK_SAFE_STATUS();
    for (size_t width = 1; width < nChunks; width *= 2)
    {
        const size_t nPairs = (nChunks + 2 * width - 1) / (2 * width);
        daal::threader_for(nPairs, 1, [&](size_t q) {
            const size_t a = q * 2 * width;
            const size_t b = a + width;
            if (b >= nChunks) return;
            (a ? partial[a] : out).add(partial[b]);
        });
    }
    return services::Status();
}

} // namespace hist
} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
