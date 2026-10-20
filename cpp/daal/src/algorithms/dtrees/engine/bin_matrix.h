/* file: bin_matrix.h */
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
//  Row-major bin indices of the training table and the bins of each feature.
//--
*/

#ifndef __DTREES_ENGINE_BIN_MATRIX_H__
#define __DTREES_ENGINE_BIN_MATRIX_H__

#include <cstddef>
#include <cstdint>

#include "src/algorithms/dtrees/engine/numa_alloc.h"
#include "src/algorithms/dtrees/engine/prefetch.h"
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

// Bins a u8 and a u16 index can address, including a feature's missing-value bin.
constexpr size_t maxU8Bins  = size_t(1) << 8;
constexpr size_t maxU16Bins = size_t(1) << 16;

/// Bytes between consecutive rows: the next power of two >= binBytes * nFeatures while that fits a
/// cache line, else binBytes * nFeatures rounded up to whole lines. A row never crosses a line.
template <CpuType cpu>
size_t binRowStride(size_t nFeatures, size_t binBytes)
{
    const size_t bytes = nFeatures * binBytes;
    if (bytes > cacheLineBytes) return (bytes + cacheLineBytes - 1) / cacheLineBytes * cacheLineBytes;
    size_t stride = 1;
    while (stride < bytes) stride <<= 1;
    return stride;
}

/**
 *  \brief Bin indices of every row and feature, and the bins of each feature
 *
 *  Feature f has nBins(f) data bins: bin b holds values in (border(f, b - 1), border(f, b)], with the
 *  first bin open below and the last bin's border equal to the feature's maximum. NaN goes to bin
 *  nBins(f), present when hasMissing(f). exactBins(f) means each data bin holds one value, its border.
 */
template <CpuType cpu>
class BinMatrix
{
public:
    services::Status allocateFeatures(size_t nFeatures)
    {
        _nFeatures = nFeatures;
        DAAL_CHECK_MALLOC(_nBins.reset(nFeatures) && _borderOffset.reset(nFeatures + 1) && _hasMissing.reset(nFeatures) && _exactBins.reset(nFeatures)
                          && _min.reset(nFeatures) && _max.reset(nFeatures));
        return services::Status();
    }

    /// Sets the borders of all features at once; offsets[f] .. offsets[f + 1] index borders of feature f.
    services::Status setBorders(const size_t * offsets, const double * borders)
    {
        const size_t total = offsets[_nFeatures];
        if (total) DAAL_CHECK_MALLOC(_borders.reset(total));
        for (size_t f = 0; f <= _nFeatures; ++f) _borderOffset[f] = offsets[f];
        if (total) services::internal::tmemcpy<double, cpu>(_borders.get(), borders, total);
        for (size_t f = 0; f < _nFeatures; ++f) _nBins[f] = offsets[f + 1] - offsets[f];
        return services::Status();
    }

    services::Status allocateRows(size_t nRows, size_t binBytes, bool replicate)
    {
        _nRows    = nRows;
        _binBytes = binBytes;
        _stride   = binRowStride<cpu>(_nFeatures, binBytes);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, _stride);
        return _rows.allocate(nRows * _stride, replicate);
    }

    size_t nRows() const { return _nRows; }
    size_t nFeatures() const { return _nFeatures; }
    size_t binBytes() const { return _binBytes; }
    size_t rowStride() const { return _stride; }
    size_t copies() const { return _rows.copies(); }

    size_t nBins(size_t f) const { return _nBins[f]; }
    size_t missingBin(size_t f) const { return _nBins[f]; }
    bool hasMissing(size_t f) const { return _hasMissing[f]; }
    bool exactBins(size_t f) const { return _exactBins[f]; }
    /// A feature whose rows all share one bin.
    bool isConstant(size_t f) const { return _nBins[f] + size_t(_hasMissing[f]) <= 1; }
    double min(size_t f) const { return _min[f]; }
    double max(size_t f) const { return _max[f]; }
    /// nBins(f) borders, the last one the feature's maximum.
    const double * borders(size_t f) const { return _borders.get() + _borderOffset[f]; }

    template <typename BinT>
    const BinT * row(int numaNode, size_t iRow) const
    {
        return reinterpret_cast<const BinT *>(_rows.data(numaNode) + iRow * _stride);
    }

    // Writers used while the matrix is built.
    uint8_t * rowBytes(size_t iCopy, size_t iRow) { return _rows.data(static_cast<int>(iCopy)) + iRow * _stride; }
    NumaReplicated<uint8_t, cpu> & storage() { return _rows; }
    void setFeature(size_t f, bool hasMissing, bool exactBins, double min, double max)
    {
        _hasMissing[f] = hasMissing;
        _exactBins[f]  = exactBins;
        _min[f]        = min;
        _max[f]        = max;
    }

private:
    size_t _nRows     = 0;
    size_t _nFeatures = 0;
    size_t _binBytes  = 0;
    size_t _stride    = 0;
    NumaReplicated<uint8_t, cpu> _rows;
    services::internal::TArray<size_t, cpu> _nBins;
    services::internal::TArray<size_t, cpu> _borderOffset;
    services::internal::TArray<double, cpu> _borders;
    services::internal::TArray<bool, cpu> _hasMissing;
    services::internal::TArray<bool, cpu> _exactBins;
    services::internal::TArray<double, cpu> _min;
    services::internal::TArray<double, cpu> _max;
};

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
