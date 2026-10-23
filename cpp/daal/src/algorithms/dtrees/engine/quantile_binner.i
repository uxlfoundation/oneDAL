/* file: quantile_binner.i */
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
//  Implementation of the quantile binner. Included only by
//  quantile_binner_fpt_cpu.cpp, so each CpuType gets its own search and sort.
//--
*/

#ifndef __DTREES_ENGINE_QUANTILE_BINNER_I__
#define __DTREES_ENGINE_QUANTILE_BINNER_I__

#include "src/algorithms/dtrees/engine/quantile_binner.h"
#include "src/externals/service_rng.h"

#include "src/algorithms/dtrees/engine/response_set.h"
#include "src/algorithms/service_error_handling.h"
#include "src/algorithms/service_sort.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_memory.h"
#include "src/services/service_data_utils.h"
#include "src/threading/threading.h"

#if defined(TARGET_X86_64) && defined(DAAL_CPU) && ((__CPUID__(DAAL_CPU) == __avx512__) || (__CPUID__(DAAL_CPU) == __avx2__))
    #include <immintrin.h>
#endif

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
using services::internal::TArray;

// Borders the k-ary SIMD search covers: 16 coarse borders, then 16 within the selected range.
constexpr size_t simdSearchBorders = 256;
constexpr size_t simdSearchBlock   = 16;

// A feature's sample is sorted as chunks of at least this many values, in parallel, when the group
// has fewer features than threads. Initial value; the binning benchmark tunes it.
constexpr size_t minSortChunk = size_t(1) << 16;

// Data-bin cap of categorical features, which get one bin per category; the u16 limit is checked after.
constexpr size_t unlimitedBins = ~size_t(0);

namespace binner
{
/// Number of the n sorted values below x.
template <CpuType cpu>
inline size_t countBelow(const double * sorted, size_t n, double x)
{
    size_t lo = 0;
    while (n > 0)
    {
        const size_t half = n / 2;
        if (sorted[lo + half] < x)
        {
            lo += half + 1;
            n -= half + 1;
        }
        else
        {
            n = half;
        }
    }
    return lo;
}

/// Number of sample values not above v: the upper end of v's run in s.
template <typename algorithmFPType>
inline size_t countNotAbove(const algorithmFPType * s, size_t k, double v)
{
    size_t lo = 0;
    size_t n  = k;
    while (n > 0)
    {
        const size_t half = n / 2;
        if (!(v < double(s[lo + half])))
        {
            lo += half + 1;
            n -= half + 1;
        }
        else
        {
            n = half;
        }
    }
    return lo;
}

/// Thread-local state of an assignment pass.
template <CpuType cpu>
struct PassStats
{
    TArray<uint8_t, cpu> nan;
    TArray<uint8_t, cpu> miss;
    TArray<double, cpu> min;
    TArray<double, cpu> max;
    TArray<uint32_t, cpu> bins;
    bool invalidCode = false;

    bool init(size_t nFeatures)
    {
        if (!nan.reset(nFeatures) || !miss.reset(nFeatures) || !min.reset(nFeatures) || !max.reset(nFeatures) || !bins.reset(tableBlockRows))
            return false;
        for (size_t f = 0; f < nFeatures; ++f)
        {
            nan[f]  = 0;
            miss[f] = 0;
            min[f]  = services::internal::MaxVal<double>::get();
            max[f]  = -services::internal::MaxVal<double>::get();
        }
        return true;
    }
};
} // namespace binner

template <typename algorithmFPType, CpuType cpu>
size_t QuantileBinner<algorithmFPType, cpu>::sortSample(size_t n, algorithmFPType * values, int32_t * scratch)
{
    return daal::algorithms::internal::qSort<algorithmFPType, int32_t, cpu>(n, values, scratch, true);
}

template <typename algorithmFPType, CpuType cpu>
void QuantileBinner<algorithmFPType, cpu>::searchBins(const double * inner, size_t nInner, const algorithmFPType * x, size_t n, size_t xStride,
                                                      uint32_t * bins)
{
#if defined(TARGET_X86_64) && defined(DAAL_CPU) && ((__CPUID__(DAAL_CPU) == __avx512__) || (__CPUID__(DAAL_CPU) == __avx2__))
    if (nInner < simdSearchBorders)
    {
        // Padding with the largest double keeps every padded border at or above any finite x; the
        // result is clamped for x = +inf.
        alignas(64) double padded[simdSearchBorders];
        alignas(64) double coarse[simdSearchBlock];
        for (size_t i = 0; i < simdSearchBorders; ++i) padded[i] = i < nInner ? inner[i] : services::internal::MaxVal<double>::get();
        for (size_t i = 0; i < simdSearchBlock; ++i) coarse[i] = padded[i * simdSearchBlock + simdSearchBlock - 1];
        for (size_t i = 0; i < n; ++i)
        {
            const double v = double(x[i * xStride]);
    #if (__CPUID__(DAAL_CPU) == __avx512__)
            const __m512d xv = _mm512_set1_pd(v);
            uint32_t q       = _mm_popcnt_u32(_mm512_cmp_pd_mask(_mm512_load_pd(coarse), xv, _CMP_LT_OQ))
                         + _mm_popcnt_u32(_mm512_cmp_pd_mask(_mm512_load_pd(coarse + 8), xv, _CMP_LT_OQ));
            q                 = q < simdSearchBlock ? q : simdSearchBlock - 1;
            const double * bl = padded + q * simdSearchBlock;
            const uint32_t r  = _mm_popcnt_u32(_mm512_cmp_pd_mask(_mm512_load_pd(bl), xv, _CMP_LT_OQ))
                               + _mm_popcnt_u32(_mm512_cmp_pd_mask(_mm512_load_pd(bl + 8), xv, _CMP_LT_OQ));
    #else
            const __m256d xv = _mm256_set1_pd(v);
            uint32_t q       = 0;
            for (size_t j = 0; j < simdSearchBlock; j += 4)
                q += _mm_popcnt_u32(uint32_t(_mm256_movemask_pd(_mm256_cmp_pd(_mm256_load_pd(coarse + j), xv, _CMP_LT_OQ))));
            q                 = q < simdSearchBlock ? q : simdSearchBlock - 1;
            const double * bl = padded + q * simdSearchBlock;
            uint32_t r        = 0;
            for (size_t j = 0; j < simdSearchBlock; j += 4)
                r += _mm_popcnt_u32(uint32_t(_mm256_movemask_pd(_mm256_cmp_pd(_mm256_load_pd(bl + j), xv, _CMP_LT_OQ))));
    #endif
            const size_t bin = size_t(q) * simdSearchBlock + r;
            bins[i]          = uint32_t(bin < nInner ? bin : nInner);
        }
        return;
    }
#endif
    for (size_t i = 0; i < n; ++i) bins[i] = uint32_t(binner::countBelow<cpu>(inner, nInner, double(x[i * xStride])));
}

template <typename algorithmFPType, CpuType cpu>
services::Status QuantileBinner<algorithmFPType, cpu>::computeBorders(const algorithmFPType * s, size_t k, size_t maxDataBins, size_t nRows,
                                                                      const BinningParams & params, bool allowExact, TArray<double, cpu> & borders,
                                                                      size_t & nBins, bool & exact)
{
    nBins = 0;
    exact = false;
    if (!k) return services::Status();

    size_t nDistinct = 1;
    for (size_t i = 1; i < k; ++i) nDistinct += s[i] != s[i - 1];

    if (allowExact && nDistinct <= maxDataBins)
    {
        DAAL_CHECK_MALLOC(borders.reset(nDistinct));
        size_t n     = 0;
        borders[n++] = double(s[0]);
        for (size_t i = 1; i < k; ++i)
            if (s[i] != s[i - 1]) borders[n++] = double(s[i]);
        nBins = nDistinct;
        exact = true;
        return services::Status();
    }

    const size_t B    = maxDataBins ? maxDataBins : 1;
    const double low  = double(s[0]);
    const double high = double(s[k - 1]);
    TArray<double, cpu> candidates(B);
    DAAL_CHECK_MALLOC(candidates.get());
    size_t nCandidates = 0;
    for (size_t j = 1; j < B; ++j)
    {
        // Quantiles: the smallest value with at least j / B of the sample at or below it.
        const double v =
            params.strategy == decision_forest::training::averages ? low + (high - low) * double(j) / double(B) : double(s[(j * k + B - 1) / B - 1]);
        if (v < high && (!nCandidates || v > candidates[nCandidates - 1])) candidates[nCandidates++] = v;
    }

    // Merge adjacent bins whose estimated row count is below minBinSize into the next bin, and a
    // small last bin into the previous one.
    DAAL_CHECK_MALLOC(borders.reset(nCandidates + 1));
    const double scale = double(nRows) / double(k);
    size_t n           = 0;
    size_t closed      = 0;
    for (size_t i = 0; i < nCandidates; ++i)
    {
        const size_t upto = binner::countNotAbove(s, k, candidates[i]);
        if (double(upto - closed) * scale >= double(params.minBinSize))
        {
            borders[n++] = candidates[i];
            closed       = upto;
        }
    }
    if (n && double(k - closed) * scale < double(params.minBinSize)) --n;
    borders[n++] = high;
    nBins        = n;
    return services::Status();
}

namespace binner
{
/// Rows of the binning sample, grouped by table block: block b holds sample positions [offset[b], offset[b + 1]).
template <CpuType cpu>
struct SampleRows
{
    bool all = true; // every row is in the sample
    size_t k = 0;
    TArray<size_t, cpu> offset;
    TArray<uint32_t, cpu> rows;

    services::Status selectAll(size_t nRows)
    {
        const size_t nBlocks = (nRows + tableBlockRows - 1) / tableBlockRows;
        DAAL_CHECK_MALLOC(offset.reset(nBlocks + 1));
        for (size_t b = 0; b <= nBlocks; ++b) offset[b] = b * tableBlockRows < nRows ? b * tableBlockRows : nRows;
        all = true;
        k   = nRows;
        return services::Status();
    }

    /// Every row when nRows <= sampleRows; otherwise a Bernoulli sample: row r is in it if the stream's r-th
    /// uniform draw is below sampleRows / nRows. The draws are read serially, in row order, so the sample
    /// depends on the stream alone, for every engine and thread count.
    services::Status select(size_t nRows, size_t sampleRows, void * stream)
    {
        if (nRows <= sampleRows) return selectAll(nRows);
        DAAL_CHECK(stream, services::ErrorNullPtr);
        const size_t nBlocks = (nRows + tableBlockRows - 1) / tableBlockRows;
        const size_t nWords  = (nRows + 63) / 64;
        TArray<uint64_t, cpu> selected(nWords);
        TArray<double, cpu> draws(tableBlockRows);
        DAAL_CHECK_MALLOC(offset.reset(nBlocks + 1) && selected.get() && draws.get());
        for (size_t w = 0; w < nWords; ++w) selected[w] = 0;
        all               = false;
        const double rate = double(sampleRows) / double(nRows);
        daal::internal::RNGsInst<double, cpu> rng;
        offset[0] = 0;
        for (size_t b = 0; b < nBlocks; ++b)
        {
            const size_t begin = b * tableBlockRows;
            const size_t n     = begin + tableBlockRows < nRows ? tableBlockRows : nRows - begin;
            if (rng.uniform(n, draws.get(), stream, 0.0, 1.0)) return services::Status(services::ErrorIncorrectErrorcodeFromGenerator);
            size_t count = 0;
            for (size_t i = 0; i < n; ++i)
            {
                if (!(draws[i] < rate)) continue;
                selected[(begin + i) >> 6] |= uint64_t(1) << ((begin + i) & 63);
                ++count;
            }
            offset[b + 1] = offset[b] + count;
        }
        k = offset[nBlocks];
        if (k) DAAL_CHECK_MALLOC(rows.reset(k));
        daal::threader_for(nBlocks, 1, [&](size_t b) {
            const size_t begin = b * tableBlockRows;
            const size_t end   = begin + tableBlockRows < nRows ? begin + tableBlockRows : nRows;
            size_t i           = offset[b];
            for (size_t r = begin; r < end; ++r)
                if ((selected[r >> 6] >> (r & 63)) & 1) rows[i++] = uint32_t(r);
        });
        return services::Status();
    }
};

/// Copies features [f0, f0 + nf) of the sample rows into dst, feature-major: dst[j * k + i].
template <typename algorithmFPType, CpuType cpu>
services::Status gatherSample(const data_management::NumericTable & x, const SampleRows<cpu> & sample, size_t f0, size_t nf, algorithmFPType * dst)
{
    const size_t nRows   = x.getNumberOfRows();
    const size_t p       = x.getNumberOfColumns();
    const size_t nBlocks = (nRows + tableBlockRows - 1) / tableBlockRows;
    const size_t k       = sample.k;
    SafeStatus safeStat;
    daal::threader_for(nBlocks, 1, [&](size_t b) {
        const size_t i0 = sample.offset[b];
        const size_t i1 = sample.offset[b + 1];
        if (i0 == i1) return;
        const size_t begin = b * tableBlockRows;
        const size_t n     = begin + tableBlockRows < nRows ? tableBlockRows : nRows - begin;
        daal::internal::ReadRows<algorithmFPType, cpu> rows(const_cast<data_management::NumericTable *>(&x), begin, n);
        DAAL_CHECK_BLOCK_STATUS_THR(rows);
        const algorithmFPType * block = rows.get();
        for (size_t i = i0; i < i1; ++i)
        {
            const size_t r                = sample.all ? i : size_t(sample.rows[i]);
            const algorithmFPType * value = block + (r - begin) * p + f0;
            for (size_t j = 0; j < nf; ++j) dst[j * k + i] = value[j];
        }
    });
    return safeStat.detach();
}

/// Gathered sample values, sort payload and merge buffer, grown on demand.
template <typename algorithmFPType, CpuType cpu>
struct SampleScratch
{
    LargeArray<algorithmFPType, cpu> values;
    LargeArray<algorithmFPType, cpu> tmp;
    LargeArray<int32_t, cpu> payload;

    bool ensure(size_t n)
    {
        if (n <= values.size()) return true;
        return values.reset(n) && tmp.reset(n) && payload.reset(n);
    }
};

template <typename T, CpuType cpu>
void mergeRuns(const T * a, size_t na, const T * b, size_t nb, T * out)
{
    size_t i = 0, j = 0, o = 0;
    while (i < na && j < nb) out[o++] = (b[j] < a[i]) ? b[j++] : a[i++];
    while (i < na) out[o++] = a[i++];
    while (j < nb) out[o++] = b[j++];
}
} // namespace binner

namespace binner
{
/**
 *  Sorts each feature of a group with NaN excluded. A feature is sorted in nChunks chunks merged
 *  pairwise in chunk order; the sorted values are unique, so the chunking does not change them.
 *  sorted[j] points into values or tmp; nNotNaN[j] counts its non-NaN values.
 */
template <typename algorithmFPType, CpuType cpu>
void sortGroup(size_t nf, size_t k, size_t nChunks, algorithmFPType * values, algorithmFPType * tmp, int32_t * scratch,
               const algorithmFPType ** sorted, size_t * nNotNaN)
{
    const size_t chunk = (k + nChunks - 1) / nChunks;
    TArray<size_t, cpu> runBegin(nf * nChunks);
    TArray<size_t, cpu> runLen(nf * nChunks);
    daal::threader_for(nf * nChunks, 1, [&](size_t t) {
        const size_t j     = t / nChunks;
        const size_t c     = t % nChunks;
        const size_t begin = c * chunk < k ? c * chunk : k;
        const size_t end   = begin + chunk < k ? begin + chunk : k;
        runBegin[t]        = begin;
        runLen[t] = end > begin ? QuantileBinner<algorithmFPType, cpu>::sortSample(end - begin, values + j * k + begin, scratch + j * k + begin) : 0;
    });

    // Merge levels: at width w the runs are at indices that are multiples of w. Runs r and r + w
    // merge into one run at r, written compactly after the runs before it.
    algorithmFPType * src = values;
    algorithmFPType * dst = tmp;
    for (size_t width = 1; width < nChunks; width *= 2)
    {
        const size_t nPairs = (nChunks + 2 * width - 1) / (2 * width);
        TArray<size_t, cpu> outBegin(nf * nPairs);
        for (size_t j = 0; j < nf; ++j)
        {
            size_t offset = 0;
            for (size_t q = 0; q < nPairs; ++q)
            {
                outBegin[j * nPairs + q] = offset;
                const size_t r0          = q * 2 * width;
                const size_t r1          = r0 + width;
                offset += runLen[j * nChunks + r0] + (r1 < nChunks ? runLen[j * nChunks + r1] : 0);
            }
        }
        daal::threader_for(nf * nPairs, 1, [&](size_t t) {
            const size_t j             = t / nPairs;
            const size_t q             = t % nPairs;
            const size_t a             = j * nChunks + q * 2 * width;
            const size_t r1            = q * 2 * width + width;
            const size_t na            = runLen[a];
            const size_t nb            = r1 < nChunks ? runLen[j * nChunks + r1] : 0;
            const algorithmFPType * pa = src + j * k + runBegin[a];
            const algorithmFPType * pb = nb ? src + j * k + runBegin[j * nChunks + r1] : pa;
            binner::mergeRuns<algorithmFPType, cpu>(pa, na, pb, nb, dst + j * k + outBegin[t]);
            runBegin[a] = outBegin[t];
            runLen[a]   = na + nb;
        });
        algorithmFPType * swap = src;
        src                    = dst;
        dst                    = swap;
    }
    for (size_t j = 0; j < nf; ++j)
    {
        sorted[j]  = src + j * k + runBegin[j * nChunks];
        nNotNaN[j] = runLen[j * nChunks];
    }
}
} // namespace binner

template <typename algorithmFPType, CpuType cpu>
services::Status QuantileBinner<algorithmFPType, cpu>::build(const data_management::NumericTable & x, const BinningParams & params,
                                                             void * sampleStream, BinMatrix<cpu> & out, bool & binned)
{
    using data_management::NumericTableIface;
    binned             = false;
    const size_t nRows = x.getNumberOfRows();
    const size_t p     = x.getNumberOfColumns();
    if (!nRows || !p) return services::Status(services::ErrorIncorrectNumberOfObservations);
    DAAL_CHECK_STATUS_VAR(checkRowCount(nRows));
    if (x.getDataLayout() & NumericTableIface::csrArray) return services::Status(services::ErrorMethodNotSupported);
    if (!params.maxBins || params.maxBins > maxU16Bins) return services::Status();

    TArray<bool, cpu> categorical(p);
    TArray<size_t, cpu> nBins(p);
    TArray<bool, cpu> exact(p);
    TArray<bool, cpu> hasMissing(p);
    TArray<TArray<double, cpu>, cpu> borders(p);
    DAAL_CHECK_MALLOC(categorical.get() && nBins.get() && exact.get() && hasMissing.get() && borders.get());
    for (size_t f = 0; f < p; ++f) categorical[f] = x.getFeatureType(f) == data_management::features::DAAL_CATEGORICAL;

    binner::SampleRows<cpu> sample;
    services::Status s = sample.select(nRows, params.sampleRows, sampleStream);
    DAAL_CHECK_STATUS_VAR(s);
    const size_t nThreads = threader_get_threads_number();

    // One scratch set for all feature groups; pages of the merge buffer are only touched when used.
    binner::SampleScratch<algorithmFPType, cpu> sampleScratch;

    // Sorts features [f0, f0 + nf) of a sample and sets their bins. Data bins are capped at maxBins,
    // one fewer for a feature with NaN; categorical features get one bin per category.
    auto binFeatures = [&](const binner::SampleRows<cpu> & rows, size_t f0, size_t nf, const bool * redo) -> services::Status {
        const size_t k = rows.k;
        if (!k) return services::Status();
        DAAL_CHECK_MALLOC(sampleScratch.ensure(nf * k));
        algorithmFPType * values = sampleScratch.values.get();
        TArray<const algorithmFPType *, cpu> sorted(nf);
        TArray<size_t, cpu> nNotNaN(nf);
        DAAL_CHECK_MALLOC(sorted.get() && nNotNaN.get());
        services::Status st = binner::gatherSample<algorithmFPType, cpu>(x, rows, f0, nf, values);
        DAAL_CHECK_STATUS_VAR(st);
        size_t nChunks = 1;
        if (nf < nThreads)
        {
            const size_t byThreads = (nThreads + nf - 1) / nf;
            const size_t bySize    = k / minSortChunk ? k / minSortChunk : 1;
            nChunks                = byThreads < bySize ? byThreads : bySize;
        }
        binner::sortGroup<algorithmFPType, cpu>(nf, k, nChunks, values, sampleScratch.tmp.get(), sampleScratch.payload.get(), sorted.get(),
                                                nNotNaN.get());
        SafeStatus safeStat;
        daal::threader_for(nf, 1, [&](size_t j) {
            const size_t f = f0 + j;
            if (redo && !redo[j]) return;
            hasMissing[f]        = hasMissing[f] || nNotNaN[j] < k;
            const size_t maxData = categorical[f] ? unlimitedBins : params.maxBins - (hasMissing[f] && params.maxBins > 1 ? 1 : 0);
            // A redo after a value missed the exact set uses quantile borders unless the sample is the full column.
            const bool allowExact  = categorical[f] || !redo || rows.all;
            services::Status local = computeBorders(sorted[j], nNotNaN[j], maxData, nRows, params, allowExact, borders[f], nBins[f], exact[f]);
            DAAL_CHECK_STATUS_THR(local);
        });
        return safeStat.detach();
    };

    const size_t bytesPerValue = 2 * sizeof(algorithmFPType) + sizeof(int32_t);
    const size_t groupSize     = binGroupFeatures(sample.k, bytesPerValue, nThreads);
    for (size_t f = 0; f < p; ++f) hasMissing[f] = false;
    for (size_t f0 = 0; f0 < p; f0 += groupSize)
    {
        s = binFeatures(sample, f0, f0 + groupSize < p ? groupSize : p - f0, nullptr);
        DAAL_CHECK_STATUS_VAR(s);
    }
    // A categorical feature has one bin per category plus possibly a missing-value bin, so more than
    // 65535 categories do not fit a u16 index.
    auto tooManyCategories = [&](size_t f) { return categorical[f] && nBins[f] >= maxU16Bins; };
    for (size_t f = 0; f < p; ++f)
        if (tooManyCategories(f)) return services::Status();

    // A bound on each feature's bins plus missing-value bin after the pass. With a sample, NaN may
    // still appear: ordered features then stay within maxBins (a full feature is rebinned with one
    // data bin fewer); categorical ones are checked again after the pass.
    auto totalBins = [&](size_t f) { return nBins[f] + size_t(hasMissing[f]); };
    auto boundBins = [&](size_t f) -> size_t {
        if (sample.all) return totalBins(f);
        if (categorical[f]) return nBins[f] + 1;
        return nBins[f] + 1 < params.maxBins ? nBins[f] + 1 : params.maxBins;
    };
    size_t maxTotal = 0;
    for (size_t f = 0; f < p; ++f) maxTotal = boundBins(f) > maxTotal ? boundBins(f) : maxTotal;
    size_t binBytes = maxTotal <= maxU8Bins ? 1 : 2;

    TArray<double, cpu> featMin(p);
    TArray<double, cpu> featMax(p);
    TArray<uint8_t, cpu> nanSeen(p);
    TArray<uint8_t, cpu> missed(p);
    DAAL_CHECK_MALLOC(featMin.get() && featMax.get() && nanSeen.get() && missed.get());

    // Assigns the listed features of every row and records, per feature, NaN, values outside an
    // exact bin set and the value range.
    auto assign = [&](const size_t * features, size_t nf) -> services::Status {
        daal::tls<binner::PassStats<cpu> *> tlsStats([&]() -> binner::PassStats<cpu> * {
            binner::PassStats<cpu> * st = new binner::PassStats<cpu>();
            if (st && !st->init(p))
            {
                delete st;
                st = nullptr;
            }
            return st;
        });
        const size_t nBlocks   = (nRows + tableBlockRows - 1) / tableBlockRows;
        services::Status local = out.storage().forEachBlock(0, nBlocks, [&](size_t b) -> services::Status {
            binner::PassStats<cpu> * st = tlsStats.local();
            DAAL_CHECK_MALLOC(st);
            const size_t begin = b * tableBlockRows;
            const size_t n     = begin + tableBlockRows < nRows ? tableBlockRows : nRows - begin;
            daal::internal::ReadRows<algorithmFPType, cpu> rows(const_cast<data_management::NumericTable *>(&x), begin, n);
            DAAL_CHECK_BLOCK_STATUS(rows);
            const algorithmFPType * block = rows.get();
            uint32_t * bins               = st->bins.get();
            for (size_t jf = 0; jf < nf; ++jf)
            {
                const size_t f      = features[jf];
                const size_t nb     = nBins[f];
                const double * bord = borders[f].get();
                const size_t nInner = nb ? nb - 1 : 0;
                searchBins(bord, nInner, block + f, n, p, bins);
                for (size_t i = 0; i < n; ++i)
                {
                    const algorithmFPType v = block[i * p + f];
                    if (daal::algorithms::internal::isNaN<algorithmFPType, cpu>(v))
                    {
                        st->nan[f] = 1;
                        bins[i]    = uint32_t(nb);
                        continue;
                    }
                    const double d = double(v);
                    if (categorical[f] && !(d >= 0 && d <= maxCategoryCode && d == double(int64_t(d)))) st->invalidCode = true;
                    if (!nb || (exact[f] && bord[bins[i]] != d)) st->miss[f] = 1;
                    st->min[f] = d < st->min[f] ? d : st->min[f];
                    st->max[f] = d > st->max[f] ? d : st->max[f];
                }
                if (binBytes == 1)
                    for (size_t i = 0; i < n; ++i) out.rowBytes(0, begin + i)[f] = uint8_t(bins[i]);
                else
                    for (size_t i = 0; i < n; ++i) reinterpret_cast<uint16_t *>(out.rowBytes(0, begin + i))[f] = uint16_t(bins[i]);
            }
            return services::Status();
        });
        bool invalid           = false;
        tlsStats.reduce([&](binner::PassStats<cpu> * st) {
            if (!st) return;
            invalid = invalid || st->invalidCode;
            for (size_t jf = 0; jf < nf; ++jf)
            {
                const size_t f = features[jf];
                nanSeen[f] |= st->nan[f];
                missed[f] |= st->miss[f];
                featMin[f] = st->min[f] < featMin[f] ? st->min[f] : featMin[f];
                featMax[f] = st->max[f] > featMax[f] ? st->max[f] : featMax[f];
            }
            delete st;
        });
        DAAL_CHECK_STATUS_VAR(local);
        return invalid ? services::Status(services::ErrorIncorrectDataRange) : services::Status();
    };

    TArray<size_t, cpu> features(p);
    DAAL_CHECK_MALLOC(features.get());
    auto resetStats = [&](const size_t * list, size_t nf) {
        for (size_t j = 0; j < nf; ++j)
        {
            const size_t f = list[j];
            featMin[f]     = services::internal::MaxVal<double>::get();
            featMax[f]     = -services::internal::MaxVal<double>::get();
            nanSeen[f]     = 0;
            missed[f]      = 0;
        }
    };
    auto assignAll = [&]() -> services::Status {
        for (size_t f = 0; f < p; ++f) features[f] = f;
        services::Status st = out.allocateFeatures(p);
        DAAL_CHECK_STATUS_VAR(st);
        st = out.allocateRows(nRows, binBytes, params.placement);
        DAAL_CHECK_STATUS_VAR(st);
        resetStats(features.get(), p);
        return assign(features.get(), p);
    };
    s = assignAll();
    DAAL_CHECK_STATUS_VAR(s);

    // NaN met by a feature that uses all maxBins data bins, or a value outside an exact bin set:
    // rebin the feature (from its full column if the sample has no value of it) and assign it again.
    size_t nRedo = 0;
    TArray<bool, cpu> redo(p);
    DAAL_CHECK_MALLOC(redo.get());
    for (size_t f = 0; f < p; ++f)
    {
        const bool fullOrdered = !categorical[f] && nanSeen[f] && !hasMissing[f] && nBins[f] == params.maxBins;
        redo[f]                = missed[f] || fullOrdered;
        hasMissing[f]          = hasMissing[f] || nanSeen[f];
        if (redo[f]) features[nRedo++] = f;
    }
    if (nRedo)
    {
        binner::SampleRows<cpu> fullColumn;
        bool fullSelected = false;
        for (size_t j = 0; j < nRedo; ++j)
        {
            const size_t f = features[j];
            // Categorical misses and samples with no value of the feature need the whole column.
            const bool needFull = categorical[f] || nBins[f] == 0;
            if (needFull && !fullSelected)
            {
                s = fullColumn.selectAll(nRows);
                DAAL_CHECK_STATUS_VAR(s);
                fullSelected = true;
            }
            bool one = true;
            s        = binFeatures(needFull ? fullColumn : sample, f, 1, &one);
            DAAL_CHECK_STATUS_VAR(s);
            if (tooManyCategories(f)) return services::Status();
        }
        maxTotal = 0;
        for (size_t f = 0; f < p; ++f) maxTotal = totalBins(f) > maxTotal ? totalBins(f) : maxTotal;
        const size_t needBytes = maxTotal <= maxU8Bins ? 1 : 2;
        if (needBytes > binBytes)
        {
            binBytes = needBytes;
            s        = assignAll();
            DAAL_CHECK_STATUS_VAR(s);
        }
        else
        {
            resetStats(features.get(), nRedo);
            s = assign(features.get(), nRedo);
            DAAL_CHECK_STATUS_VAR(s);
        }
        for (size_t j = 0; j < nRedo; ++j) DAAL_ASSERT(!missed[features[j]]);
    }

    // The last border is the feature's maximum, so the random splitter's bounds cover every row.
    TArray<size_t, cpu> offsets(p + 1);
    DAAL_CHECK_MALLOC(offsets.get());
    offsets[0] = 0;
    for (size_t f = 0; f < p; ++f) offsets[f + 1] = offsets[f] + nBins[f];
    TArray<double, cpu> allBorders(offsets[p] ? offsets[p] : 1);
    DAAL_CHECK_MALLOC(allBorders.get());
    for (size_t f = 0; f < p; ++f)
    {
        const bool anyValue = nBins[f] > 0;
        if (anyValue) borders[f][nBins[f] - 1] = featMax[f];
        for (size_t b = 0; b < nBins[f]; ++b) allBorders[offsets[f] + b] = borders[f][b];
        out.setFeature(f, hasMissing[f], exact[f], anyValue ? featMin[f] : 0, anyValue ? featMax[f] : 0);
    }
    s = out.setBorders(offsets.get(), allBorders.get());
    DAAL_CHECK_STATUS_VAR(s);
    s = out.storage().replicate(tableBlockRows * out.rowStride());
    DAAL_CHECK_STATUS_VAR(s);
    binned = true;
    return services::Status();
}

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
