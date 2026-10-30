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
 */
template <typename algorithmFPType, CpuType cpu>
class ResponseSource
{
public:
    services::Status build(const data_management::NumericTable & y, const data_management::NumericTable * w, bool classification,
                           const NumaPlacement & placement)
    {
        _nRows          = y.getNumberOfRows();
        _classification = classification;
        services::Status s;
        if (classification)
            s = readColumn(y, placement, _cls, [](algorithmFPType v) { return uint32_t(v); });
        else
            s = readColumn(y, placement, _y, [](algorithmFPType v) { return v; });
        DAAL_CHECK_STATUS_VAR(s);
        _weighted = false;
        _mean     = 0;
        if (w)
        {
            s = readWeights(*w, placement);
            DAAL_CHECK_STATUS_VAR(s);
        }
        return classification ? s : computeMean();
    }

    size_t nRows() const { return _nRows; }
    bool weighted() const { return _weighted; }
    const uint32_t * classes(int numaNode) const { return _cls.data(numaNode); }
    const algorithmFPType * responses(int numaNode) const { return _y.data(numaNode); }
    const algorithmFPType * weights(int numaNode) const { return _w.data(numaNode); }
    double responseMean() const { return _mean; } // weighted if the fit is; 0 for classification

private:
    services::Status readWeights(const data_management::NumericTable & w, const NumaPlacement & placement)
    {
        services::Status s;
        const algorithmFPType maxW = maxValue(w, s);
        DAAL_CHECK_STATUS_VAR(s);
        if (!(maxW > 0)) return s;
        const algorithmFPType eps = algorithmFPType(10) * services::internal::EpsilonVal<algorithmFPType>::get();
        s                         = readColumn(w, placement, _w, [=](algorithmFPType v) {
            const algorithmFPType scaled = v / maxW;
            return scaled > eps ? scaled : algorithmFPType(0);
        });
        _weighted                 = s.ok();
        return s;
    }

    /// The (weighted) mean response, the root's center (MseCost::setCenter) and the prediction of a leaf without
    /// records. Per-block sums add in block order, so it does not depend on the thread count; a sum that is
    /// not finite (overflow, NaN) leaves it 0.
    services::Status computeMean()
    {
        if (!_nRows) return services::Status();
        const size_t nBlocks = (_nRows + tableBlockRows - 1) / tableBlockRows;
        services::internal::TArray<double, cpu> stats(2 * nBlocks);
        DAAL_CHECK_MALLOC(stats.get());
        const algorithmFPType * y = _y.data(0);
        const algorithmFPType * w = _weighted ? _w.data(0) : nullptr;
        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            const size_t begin = iBlock * tableBlockRows;
            const size_t end   = begin + tableBlockRows < _nRows ? begin + tableBlockRows : _nRows;
            double sy = 0, sw = 0;
            for (size_t i = begin; i < end; ++i)
            {
                const double wi = w ? double(w[i]) : 1.;
                sy += wi * double(y[i]);
                sw += wi;
            }
            stats[2 * iBlock]     = sy;
            stats[2 * iBlock + 1] = sw;
        });
        double sy = 0, sw = 0;
        for (size_t i = 0; i < nBlocks; ++i)
        {
            sy += stats[2 * i];
            sw += stats[2 * i + 1];
        }
        const double mean = sw > 0 ? sy / sw : 0;
        if (mean - mean == 0) _mean = mean;
        return services::Status();
    }

    template <typename T, typename Convert>
    services::Status readColumn(const data_management::NumericTable & t, const NumaPlacement & placement, NumaReplicated<T, cpu> & out,
                                const Convert & convert)
    {
        services::Status s = out.allocate(_nRows, placement);
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
        rec.y     = src.responses(node)[row];
        rec.count = count;
        return true;
    }

    template <typename algorithmFPType>
    static bool makeRecord(RegWeightRecord<algorithmFPType> & rec, uint32_t row, uint32_t count, const ResponseSource<algorithmFPType, cpu> & src,
                           int node)
    {
        rec     = RegWeightRecord<algorithmFPType> {};
        rec.row = row;
        rec.y   = src.responses(node)[row];
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
