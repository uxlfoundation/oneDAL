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
//  drawn features over the bin matrix, built in chunks for large nodes, and the
//  partition pass that builds the children's histograms as it routes the rows.
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

// Lanes of a plane row are padded to 16, so each bin's row of u32 or f64 statistics starts on a cache line.
constexpr size_t laneAlign = 16;

/// Per-bin statistics of a record layout (design 4.10): record counts u32; class weights, or the regression
/// weight, u32 for bootstrap counts and f64 for user weights, which float sums would round (legacy moved to
/// f64 for the same reason); the regression weighted response sum f64.
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
    using Stat                           = double;
    static constexpr bool classification = true;
};
template <typename algorithmFPType>
struct Layout<RegCountRecord<algorithmFPType> >
{
    using Stat                           = uint32_t;
    static constexpr bool classification = false;
};
template <typename algorithmFPType>
struct Layout<RegWeightRecord<algorithmFPType> >
{
    using Stat                           = double;
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

    void release()
    {
        records.reset(0);
        stat.reset(0);
        sumY.reset(0);
    }

    size_t at(size_t bin, size_t lane) const { return bin * stride + lane; }
    Stat classStat(size_t c, size_t bin, size_t lane) const { return stat[c * nBins * stride + at(bin, lane)]; }
};

/// Adds one record, whose bin row is b, to the planes of nLanes features.
template <typename Record, typename BinT, CpuType cpu>
inline void addRow(const Record & r, const BinT * b, const uint32_t * features, size_t nLanes, Planes<Record, cpu> & h)
{
    uint32_t * records  = h.records.get();
    auto * stat         = h.stat.get();
    const size_t stride = h.stride;
    if constexpr (Layout<Record>::classification)
    {
        const size_t offset = size_t(recordClass(r)) * h.nBins * stride;
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
        double * sumY   = h.sumY.get();
        const double w  = recordWeight(r);
        const double wy = w * recordResponse(r);
        const auto ws   = static_cast<typename Planes<Record, cpu>::Stat>(w);
        for (size_t j = 0; j < nLanes; ++j)
        {
            const size_t idx = size_t(b[features[j]]) * stride + j;
            ++records[idx];
            stat[idx] += ws;
            sumY[idx] += wy;
        }
    }
}

/// Adds records [begin, end) to the planes, reading each record's bin row once and prefetching distance rows
/// ahead the lines that hold the lanes' features.
template <typename Record, typename BinT, CpuType cpu>
void accumulate(const Record * rec, size_t begin, size_t end, const BinMatrix<cpu> & bins, int numaNode, const uint32_t * features, size_t nLanes,
                const uint32_t * lines, size_t nLines, size_t distance, Planes<Record, cpu> & h)
{
    prefetchedLoop<cpu>(
        end - begin, distance, [&](size_t i) { prefetchRowLines<cpu>(bins.template row<uint8_t>(numaNode, rec[begin + i].row), lines, nLines); },
        [&](size_t i) {
            const Record & r = rec[begin + i];
            addRow<Record, BinT, cpu>(r, bins.template row<BinT>(numaNode, r.row), features, nLanes, h);
        });
}

/// Bins of the planes of nLanes features: the largest data bin count plus missing bin over them.
template <CpuType cpu>
size_t planeBins(const BinMatrix<cpu> & bins, const uint32_t * features, size_t nLanes)
{
    size_t nBins = 0;
    for (size_t j = 0; j < nLanes; ++j)
    {
        const size_t f = features[j];
        const size_t b = bins.nBins(f) + (bins.hasMissing(f) ? 1 : 0);
        nBins          = b > nBins ? b : nBins;
    }
    return nBins;
}

/// Adds the planes of chunks 1 .. nChunks - 1 into chunk 0's by a fixed pairwise tree over chunk index.
template <typename PlanesOf>
void reducePairwise(size_t nChunks, const PlanesOf & planesOf)
{
    for (size_t width = 1; width < nChunks; width *= 2)
    {
        const size_t nPairs = (nChunks + 2 * width - 1) / (2 * width);
        daal::threader_for(nPairs, 1, [&](size_t q) {
            const size_t a = q * 2 * width;
            const size_t b = a + width;
            if (b < nChunks) planesOf(a).add(planesOf(b));
        });
    }
}

/// Chunks of a node of n records: ceil(n / chunkRows), at most maxChunks. Boundaries depend on n only.
inline size_t chunkCount(size_t n, size_t chunkRows, size_t maxChunks)
{
    const size_t c = chunkRows ? (n + chunkRows - 1) / chunkRows : 1;
    return c < 1 ? 1 : (c > maxChunks ? maxChunks : c);
}

// Bytes the partial planes of one chunked node may take. A node whose chunks would need more runs as fewer,
// larger chunks, as many-class planes (1000 classes, 256 bins, 50 lanes: 131 MB) would otherwise take 8 GB.
constexpr size_t partialPlaneBytes = size_t(1) << 28;

/// Bytes of one set of planes of nLanes features with nBins bins.
template <typename Record>
inline size_t planeBytes(size_t nBins, size_t nLanes, size_t nClasses)
{
    const size_t n = nBins * ((nLanes + laneAlign - 1) / laneAlign * laneAlign);
    const size_t s = Layout<Record>::classification ? nClasses * n : n;
    return n * sizeof(uint32_t) + s * sizeof(typename Layout<Record>::Stat) + (Layout<Record>::classification ? 0 : n * sizeof(double));
}

/// nChunks, capped so that the partial planes of chunks 1.. take at most partialPlaneBytes. The cap depends on
/// the plane shape only, so results still do not depend on the thread count.
inline size_t chunksWithinBudget(size_t nChunks, size_t bytesPerChunk)
{
    if (!bytesPerChunk) return nChunks;
    const size_t fit = 1 + partialPlaneBytes / bytesPerChunk;
    return nChunks < fit ? nChunks : fit;
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
    const size_t nBins = planeBins(bins, features, nLanes);
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
    nChunks = chunksWithinBudget(nChunks, planeBytes<Record>(nBins, nLanes, nClasses));
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
    reducePairwise(nChunks, [&](size_t c) -> Planes<Record, cpu> & { return c ? partial[c] : out; });
    return services::Status();
}

/// One child's histograms in a fused partition: nLanes features into planes, or none when planes is null.
template <typename Record, CpuType cpu>
struct Target
{
    const uint32_t * features    = nullptr;
    size_t nLanes                = 0;
    Planes<Record, cpu> * planes = nullptr;
};

/**
 *  \brief Stable partition of n records into out, the nLeft rows with left(binRow) first, building each
 *  child's histograms from the bin rows the routing reads (design 4.6, fused partition)
 *
 *  One chunk routes, scatters and adds in one pass. Several chunks route and add in parallel, keeping the
 *  side of each row and partial planes per chunk and child, reduced in chunk order; a second pass
 *  scatters from the per-chunk offsets. Results do not depend on the thread count.
 */
template <typename Record, typename BinT, typename Left, CpuType cpu>
services::Status fusedPartition(const Record * in, Record * out, size_t n, size_t nLeft, const BinMatrix<cpu> & bins, int numaNode,
                                uint32_t splitFeature, const Left & left, const Target<Record, cpu> (&targets)[2], size_t nClasses, size_t nChunks,
                                size_t distance)
{
    // Lines to prefetch: the split feature's and the children's features'.
    size_t nFeatures = 1 + targets[0].nLanes + targets[1].nLanes;
    TArray<uint32_t, cpu> features(nFeatures);
    TArray<uint32_t, cpu> lines(nFeatures);
    DAAL_CHECK_MALLOC(features.get() && lines.get());
    features[0] = splitFeature;
    nFeatures   = 1;
    for (const auto & t : targets)
        for (size_t j = 0; j < t.nLanes; ++j) features[nFeatures++] = t.features[j];
    const size_t nLines = rowLines(features.get(), nFeatures, bins.binBytes(), lines.get());

    size_t nBins[2] = { 0, 0 };
    size_t bytes    = 0;
    for (size_t s = 0; s < 2; ++s)
    {
        if (!targets[s].planes) continue;
        nBins[s] = planeBins(bins, targets[s].features, targets[s].nLanes);
        DAAL_CHECK_MALLOC(targets[s].planes->init(nBins[s], targets[s].nLanes, nClasses));
        bytes += planeBytes<Record>(nBins[s], targets[s].nLanes, nClasses);
    }
    nChunks       = chunksWithinBudget(nChunks, bytes);
    auto prefetch = [&](size_t i) { prefetchRowLines<cpu>(bins.template row<uint8_t>(numaNode, in[i].row), lines.get(), nLines); };

    if (nChunks <= 1)
    {
        size_t iLeft = 0, iRight = nLeft;
        prefetchedLoop<cpu>(n, distance, prefetch, [&](size_t i) {
            const Record & r            = in[i];
            const BinT * b              = bins.template row<BinT>(numaNode, r.row);
            const size_t s              = left(b) ? 0 : 1;
            out[s ? iRight++ : iLeft++] = r;
            if (targets[s].planes) addRow<Record, BinT, cpu>(r, b, targets[s].features, targets[s].nLanes, *targets[s].planes);
        });
        if (iLeft != nLeft) return services::Status(services::ErrorIncorrectInternalFunctionParameter);
        return services::Status();
    }

    TArray<uint8_t, cpu> side(n);
    TArray<size_t, cpu> lefts(nChunks + 1);
    TArray<Planes<Record, cpu>, cpu> partial(2 * nChunks);
    DAAL_CHECK_MALLOC(side.get() && lefts.get() && partial.get());
    auto planesOf = [&](size_t s, size_t c) -> Planes<Record, cpu> & { return c ? partial[2 * c + s] : *targets[s].planes; };
    SafeStatus safeStat;
    daal::threader_for(nChunks, 1, [&](size_t c) {
        for (size_t s = 0; s < 2; ++s)
            DAAL_CHECK_THR(!c || !targets[s].planes || planesOf(s, c).init(nBins[s], targets[s].nLanes, nClasses),
                           services::ErrorMemoryAllocationFailed);
        size_t begin, end, count = 0;
        chunkRange(n, nChunks, c, begin, end);
        prefetchedLoop<cpu>(
            end - begin, distance, [&](size_t i) { prefetch(begin + i); },
            [&](size_t i) {
                const Record & r = in[begin + i];
                const BinT * b   = bins.template row<BinT>(numaNode, r.row);
                const size_t s   = left(b) ? 0 : 1;
                side[begin + i]  = uint8_t(s);
                count += 1 - s;
                if (targets[s].planes) addRow<Record, BinT, cpu>(r, b, targets[s].features, targets[s].nLanes, planesOf(s, c));
            });
        lefts[c + 1] = count;
    });
    DAAL_CHECK_SAFE_STATUS();
    lefts[0] = 0;
    for (size_t c = 0; c < nChunks; ++c) lefts[c + 1] += lefts[c];
    if (lefts[nChunks] != nLeft) return services::Status(services::ErrorIncorrectInternalFunctionParameter);
    for (size_t s = 0; s < 2; ++s)
        if (targets[s].planes) reducePairwise(nChunks, [&](size_t c) -> Planes<Record, cpu> & { return planesOf(s, c); });
    daal::threader_for(nChunks, 1, [&](size_t c) {
        size_t begin, end;
        chunkRange(n, nChunks, c, begin, end);
        size_t iLeft = lefts[c], iRight = nLeft + begin - lefts[c];
        for (size_t i = begin; i < end; ++i) out[side[i] ? iRight++ : iLeft++] = in[i];
    });
    return services::Status();
}

} // namespace hist
} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
