/* file: response_set.h */
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
//  Per-tree response records: one per in-bag row, carrying the row index, the
//  response and the row's bootstrap count or weight.
//--
*/

#ifndef __DTREES_ENGINE_RESPONSE_SET_H__
#define __DTREES_ENGINE_RESPONSE_SET_H__

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "src/algorithms/dtrees/engine/numa_alloc.h"
#include "src/services/service_arrays.h"
#include "src/services/service_data_utils.h"

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

// Row indices are u32 and the model stores node sample counts as int.
constexpr size_t maxTrainingRows = (size_t(1) << 31) - 1;

// Classes the 8-byte record's u16 class field holds.
constexpr size_t maxCompactClasses = (size_t(1) << 16) - 1;

/// Rejects tables whose row indices do not fit the engine's u32 indices and the model's int counts.
inline services::Status checkRowCount(size_t nRows)
{
    return nRows > maxTrainingRows ? services::Status(services::ErrorIncorrectNumberOfObservations) : services::Status();
}

// Record layouts (design 4.2). No field is narrower than in the legacy code; counts are exact.
struct ClsCountRecord
{
    uint32_t row;
    uint16_t cls;
    uint16_t count;
};

template <typename algorithmFPType>
struct ClsWeightRecord
{
    uint32_t row;
    uint32_t cls;
    algorithmFPType w;
};

template <typename algorithmFPType>
struct RegCountRecord;

template <>
struct RegCountRecord<float>
{
    uint32_t row;
    float y;
    uint32_t count;
};

template <>
struct RegCountRecord<double>
{
    uint32_t row;
    uint32_t count;
    double y;
};

template <typename algorithmFPType>
struct RegWeightRecord
{
    uint32_t row;
    algorithmFPType y;
    algorithmFPType w;
};

template <>
struct RegWeightRecord<double>
{
    uint32_t row;
    uint32_t pad;
    double y;
    double w;
};

static_assert(sizeof(ClsCountRecord) == 8, "");
static_assert(sizeof(ClsWeightRecord<float>) == 12 && sizeof(ClsWeightRecord<double>) == 16, "");
static_assert(sizeof(RegCountRecord<float>) == 12 && sizeof(RegCountRecord<double>) == 16, "");
static_assert(sizeof(RegWeightRecord<float>) == 12 && sizeof(RegWeightRecord<double>) == 24, "");

/// Count or weight of a record, the statistic histograms and costs sum.
inline double recordWeight(const ClsCountRecord & r)
{
    return r.count;
}
template <typename algorithmFPType>
inline double recordWeight(const ClsWeightRecord<algorithmFPType> & r)
{
    return r.w;
}
template <typename algorithmFPType>
inline double recordWeight(const RegCountRecord<algorithmFPType> & r)
{
    return r.count;
}
template <typename algorithmFPType>
inline double recordWeight(const RegWeightRecord<algorithmFPType> & r)
{
    return r.w;
}

/**
 *  \brief Responses and normalized user weights of the training rows, one copy per NUMA node
 *
 *  Weights are scaled to a maximum of 1 and weights not above 10 machine epsilons are set to 0, as
 *  the legacy normalizeWeights and DataHelper do. If no weight is positive the fit is unweighted.
 *  Regression records hold the responses less responseShift(); computeShift explains why and how it is chosen.
 */
template <typename algorithmFPType, CpuType cpu>
class ResponseSource
{
public:
    services::Status build(const data_management::NumericTable & y, const data_management::NumericTable * w, bool classification, bool replicate)
    {
        _nRows          = y.getNumberOfRows();
        _classification = classification;
        services::Status s;
        if (classification)
            s = readColumn(y, replicate, _cls, [](algorithmFPType v) { return uint32_t(v); });
        else
            s = readColumn(y, replicate, _y, [](algorithmFPType v) { return v; });
        DAAL_CHECK_STATUS_VAR(s);
        _weighted = false;
        _shift    = 0;
        _mean     = 0;
        if (w)
        {
            s = readWeights(*w, replicate);
            DAAL_CHECK_STATUS_VAR(s);
        }
        return classification ? s : computeShift();
    }

    size_t nRows() const { return _nRows; }
    bool weighted() const { return _weighted; }
    const uint32_t * classes(int numaNode) const { return _cls.data(numaNode); }
    const algorithmFPType * responses(int numaNode) const { return _y.data(numaNode); }
    const algorithmFPType * weights(int numaNode) const { return _w.data(numaNode); }
    double responseShift() const { return _shift; }
    double responseMean() const { return _mean; } // weighted if the fit is; 0 for classification

private:
    services::Status readWeights(const data_management::NumericTable & w, bool replicate)
    {
        services::Status s;
        const algorithmFPType maxW = maxValue(w, s);
        DAAL_CHECK_STATUS_VAR(s);
        if (!(maxW > 0)) return s;
        const algorithmFPType eps = algorithmFPType(10) * services::internal::EpsilonVal<algorithmFPType>::get();
        s                         = readColumn(w, replicate, _w, [=](algorithmFPType v) {
            const algorithmFPType scaled = v / maxW;
            return scaled > eps ? scaled : algorithmFPType(0);
        });
        _weighted                 = s.ok();
        return s;
    }

    /**
     *  \brief Sets the shift subtracted from every regression response, and the mean response
     *
     *  Why shift. MseCost scores splits from raw sums: impurity = sum(w y^2) / sum(w) - mean^2, and the split
     *  proxy sumL^2 / wL + sumR^2 / wR. When every response is close to a large constant C (y = C + x, x of spread
     *  s), each term is about C^2 times the weight, while the quantities that matter, the differences between
     *  them, are about s^2 times the weight. Double keeps about 16 digits, so once C / s nears 1e8 the
     *  differences are rounding noise: with y = 1e8 + x, x in [0, 1), impurity computes to 0 and the root
     *  becomes a leaf. Subtracting a constant from every response changes none of the results: the variance is
     *  unchanged, and the proxy of every candidate split of a node moves by the same amount as the node's own
     *  proxy, so the chosen split and its impurity decrease are the same. Leaves add the shift back
     *  (TreeBuilder::regressionLeaf). Legacy avoided the problem with Welford updates instead of raw sums.
     *
     *  Why not subtract the mean itself. The subtraction must not add rounding of its own. Responses are often
     *  on a grid, integers for example, and sums of integers in double are exact. y - 3.71 is not an integer,
     *  so sums of shifted responses would round. Exact sums matter here: the histogram phase adds a node's
     *  responses bin by bin and the exact phase adds them in sorted order, and only exact sums make both phases
     *  score every candidate identically, so that they pick the same split (the test "hist with one bin per
     *  value equals the exact phase on the same draws" relies on it).
     *
     *  The rule. Let range = max(y) - min(y) and unit = the largest power of two not above range (range in
     *  [unit, 2 unit)). The shift is the (weighted) mean rounded to the nearest multiple of unit.
     *  - It still centers: the mean lies in [min, max] and the shift is within unit / 2 <= range / 2 of it, so
     *    every shifted response is at most 1.5 range in magnitude, whatever C is.
     *  - It keeps grids: if the responses are multiples of a power of two g (g = 1 for integers, g = 0.25 for
     *    quarters, any float's spacing), their range is a nonzero multiple of g, so unit >= g and unit is a
     *    multiple of g. The shift, a multiple of unit, is then a multiple of g too, so each y - shift is a
     *    multiple of g no larger than about the responses themselves: it is computed exactly and sums that were
     *    exact before the shift stay exact.
     *  Example: y = {1e8, 1e8 + 1, 1e8 + 2, 1e8 + 3}: mean 1e8 + 1.5, range 3, unit 2, shift 1e8 + 2, shifted
     *  responses {-2, -1, 0, 1}.
     *  Constant responses (range 0) shift by their value, so every shifted response is 0. Responses whose sum
     *  or range is not finite (overflow, NaN) are not shifted.
     *
     *  The mean is kept as well: a leaf without records (a tree of zero-weight rows) predicts it, not the
     *  rounded shift. Per-block sums are added in block order, so the result does not depend on the thread count.
     */
    services::Status computeShift()
    {
        if (!_nRows) return services::Status();
        const size_t nBlocks = (_nRows + tableBlockRows - 1) / tableBlockRows;
        services::internal::TArray<double, cpu> stats(4 * nBlocks);
        DAAL_CHECK_MALLOC(stats.get());
        const algorithmFPType * y = _y.data(0);
        const algorithmFPType * w = _weighted ? _w.data(0) : nullptr;
        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            const size_t begin = iBlock * tableBlockRows;
            const size_t end   = begin + tableBlockRows < _nRows ? begin + tableBlockRows : _nRows;
            double sy = 0, sw = 0, lo = double(y[begin]), hi = double(y[begin]);
            for (size_t i = begin; i < end; ++i)
            {
                const double yi = double(y[i]);
                const double wi = w ? double(w[i]) : 1.;
                sy += wi * yi;
                sw += wi;
                lo = yi < lo ? yi : lo;
                hi = yi > hi ? yi : hi;
            }
            double * out = stats.get() + 4 * iBlock;
            out[0]       = sy;
            out[1]       = sw;
            out[2]       = lo;
            out[3]       = hi;
        });
        double sy = 0, sw = 0, lo = stats[2], hi = stats[3];
        for (size_t i = 0; i < nBlocks; ++i)
        {
            sy += stats[4 * i];
            sw += stats[4 * i + 1];
            lo = stats[4 * i + 2] < lo ? stats[4 * i + 2] : lo;
            hi = stats[4 * i + 3] > hi ? stats[4 * i + 3] : hi;
        }
        const double mean  = sw > 0 ? sy / sw : 0;
        const double range = hi - lo;
        // Overflowing sums, and NaN responses, leave the responses unshifted.
        if (!(mean - mean == 0) || !(range - range == 0)) return services::Status();
        _mean = mean;
        if (range == 0)
        {
            _shift = lo;
            return services::Status();
        }
        int e = 0;
        std::frexp(range, &e); // range in [2^(e-1), 2^e)
        const double unit = std::ldexp(1., e - 1);
        _shift            = std::nearbyint(mean / unit) * unit;
        return services::Status();
    }

    template <typename T, typename Convert>
    services::Status readColumn(const data_management::NumericTable & t, bool replicate, NumaReplicated<T, cpu> & out, const Convert & convert)
    {
        services::Status s = out.allocate(_nRows, replicate);
        DAAL_CHECK_STATUS_VAR(s);
        const size_t nBlocks = (_nRows + tableBlockRows - 1) / tableBlockRows;
        for (size_t iCopy = 0; iCopy < out.copies(); ++iCopy)
        {
            T * dst = out.data(static_cast<int>(iCopy));
            s       = out.forEachBlock(iCopy, nBlocks, [&](size_t iBlock) {
                const size_t begin = iBlock * tableBlockRows;
                const size_t n     = begin + tableBlockRows < _nRows ? tableBlockRows : _nRows - begin;
                daal::internal::ReadRows<algorithmFPType, cpu> rows(const_cast<data_management::NumericTable *>(&t), begin, n);
                DAAL_CHECK_BLOCK_STATUS(rows);
                const algorithmFPType * src = rows.get();
                for (size_t i = 0; i < n; ++i) dst[begin + i] = convert(src[i]);
                return services::Status();
            });
            DAAL_CHECK_STATUS_VAR(s);
        }
        return s;
    }

    algorithmFPType maxValue(const data_management::NumericTable & t, services::Status & s) const
    {
        algorithmFPType result = 0;
        for (size_t begin = 0; begin < _nRows && s.ok(); begin += tableBlockRows)
        {
            const size_t n = begin + tableBlockRows < _nRows ? tableBlockRows : _nRows - begin;
            daal::internal::ReadRows<algorithmFPType, cpu> rows(const_cast<data_management::NumericTable *>(&t), begin, n);
            s |= rows.status();
            if (!s) break;
            const algorithmFPType * src = rows.get();
            for (size_t i = 0; i < n; ++i) result = src[i] > result ? src[i] : result;
        }
        return result;
    }

    size_t _nRows        = 0;
    bool _classification = false;
    bool _weighted       = false;
    double _shift        = 0;
    double _mean         = 0;
    NumaReplicated<uint32_t, cpu> _cls;
    NumaReplicated<algorithmFPType, cpu> _y;
    NumaReplicated<algorithmFPType, cpu> _w;
};

/**
 *  \brief The response records of one tree, in two buffers used in turn by the levels of the tree
 *
 *  Buffer 0 holds the in-bag rows in ascending row order. A node at depth d reads buffer d % 2 over
 *  its range and partitions into the other buffer over the same range.
 */
template <typename Record, CpuType cpu>
class ResponseSet
{
public:
    /// Allocates both buffers once for up to capacity records; call inside the arena that uses them.
    services::Status allocate(size_t capacity)
    {
        DAAL_CHECK_MALLOC(_buffers[0].reset(capacity) && _buffers[1].reset(capacity));
        _capacity = capacity;
        _size     = 0;
        return services::Status();
    }

    size_t size() const { return _size; }
    Record * buffer(size_t depth) { return _buffers[depth & 1].get(); }
    const Record * buffer(size_t depth) const { return _buffers[depth & 1].get(); }

    /**
     *  \brief Fills buffer 0 from per-row bootstrap counts, counts[r] draws of row r
     *
     *  counts == nullptr means every row once. Rows with no draws or a zero weight get no record.
     *  totalWeight is the sum of the records' counts or weights.
     */
    template <typename algorithmFPType>
    services::Status fill(const uint16_t * counts, const ResponseSource<algorithmFPType, cpu> & source, int numaNode, double & totalWeight)
    {
        const size_t nRows = source.nRows();
        DAAL_CHECK_STATUS_VAR(checkRowCount(nRows));
        Record * dst = _buffers[0].get();
        size_t n     = 0;
        double total = 0;
        for (size_t r = 0; r < nRows; ++r)
        {
            const uint32_t count = counts ? counts[r] : 1;
            if (!count) continue;
            if (n == _capacity) return services::Status(services::ErrorIncorrectNumberOfObservations);
            if (!makeRecord(dst[n], uint32_t(r), count, source, numaNode)) continue;
            total += recordWeight(dst[n]);
            ++n;
        }
        _size       = n;
        totalWeight = total;
        return services::Status();
    }

private:
    template <typename algorithmFPType>
    static bool makeRecord(ClsCountRecord & rec, uint32_t row, uint32_t count, const ResponseSource<algorithmFPType, cpu> & src, int node)
    {
        DAAL_ASSERT(!src.weighted());
        DAAL_ASSERT(src.classes(node)[row] <= maxCompactClasses);
        rec = { row, uint16_t(src.classes(node)[row]), uint16_t(count) };
        return true;
    }

    template <typename algorithmFPType>
    static bool makeRecord(ClsWeightRecord<algorithmFPType> & rec, uint32_t row, uint32_t count, const ResponseSource<algorithmFPType, cpu> & src,
                           int node)
    {
        const algorithmFPType w = src.weighted() ? algorithmFPType(count) * src.weights(node)[row] : algorithmFPType(count);
        rec                     = { row, src.classes(node)[row], w };
        return w > 0;
    }

    template <typename algorithmFPType>
    static bool makeRecord(RegCountRecord<algorithmFPType> & rec, uint32_t row, uint32_t count, const ResponseSource<algorithmFPType, cpu> & src,
                           int node)
    {
        DAAL_ASSERT(!src.weighted());
        rec.row   = row;
        rec.y     = algorithmFPType(double(src.responses(node)[row]) - src.responseShift());
        rec.count = count;
        return true;
    }

    template <typename algorithmFPType>
    static bool makeRecord(RegWeightRecord<algorithmFPType> & rec, uint32_t row, uint32_t count, const ResponseSource<algorithmFPType, cpu> & src,
                           int node)
    {
        rec     = RegWeightRecord<algorithmFPType> {};
        rec.row = row;
        rec.y   = algorithmFPType(double(src.responses(node)[row]) - src.responseShift());
        rec.w   = src.weighted() ? algorithmFPType(count) * src.weights(node)[row] : algorithmFPType(count);
        return rec.w > 0;
    }

    LargeArray<Record, cpu> _buffers[2];
    size_t _capacity = 0;
    size_t _size     = 0;
};

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
