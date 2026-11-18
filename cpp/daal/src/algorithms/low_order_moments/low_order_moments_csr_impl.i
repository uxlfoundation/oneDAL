/* file: low_order_moments_csr_impl.i */
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
//  Low order moments of CSR data, computed from the non-zero values only
//--
*/

#ifndef __LOW_ORDER_MOMENTS_CSR_IMPL_I__
#define __LOW_ORDER_MOMENTS_CSR_IMPL_I__

#include "src/algorithms/low_order_moments/low_order_moments_kernel.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/services/service_arrays.h"
#include "src/services/service_data_utils.h"
#include "src/threading/threading.h"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace internal
{
namespace csr
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

/// Per-feature moments of a set of rows, in double whatever the table type, as
/// scikit-learn's mean_variance_axis accumulates them.
template <CpuType cpu>
struct Moments
{
    Status init(size_t nFeatures)
    {
        sum.reset(nFeatures);
        sumSq.reset(nFeatures);
        sumSqCen.reset(nFeatures);
        mean.reset(nFeatures);
        min.reset(nFeatures);
        max.reset(nFeatures);
        DAAL_CHECK_MALLOC(sum.get() && sumSq.get() && sumSqCen.get() && mean.get() && min.get() && max.get());
        return Status();
    }

    size_t nRows = 0;
    TArray<double, cpu> sum;
    TArray<double, cpu> sumSq;
    TArray<double, cpu> sumSqCen;
    TArray<double, cpu> mean;
    TArray<double, cpu> min;
    TArray<double, cpu> max;
};

/// Running moments of one column inside a chunk, packed into one cache line so that a
/// stored value updates a single line.
enum ColumnSlot
{
    slotCount = 0,
    slotMean,
    slotM2,
    slotSum,
    slotSumSq,
    slotMin,
    slotMax,
    slotsPerColumn = 8
};

/// Splits the rows into chunks with their own accumulators. The count does not depend on the
/// number of threads, and the chunks are merged in order, so the result is the same bit for bit.
///
/// @param[in]  nRows      Number of rows
/// @param[in]  nFeatures  Number of columns
/// @param[in]  nnz        Number of stored values
/// @return     Number of rows per chunk
inline size_t rowsPerChunk(size_t nRows, size_t nFeatures, size_t nnz)
{
    // A chunk initializes and merges slotsPerColumn values per column: 64 stored values per
    // column keep that under a few percent of its work, 2^15 keeps the task overhead small.
    constexpr size_t valuesPerColumn   = 64;
    constexpr size_t minValuesPerChunk = size_t(1) << 15;
    // Enough chunks to occupy any thread count; fixed so that the chunking stays the same.
    constexpr size_t maxChunks = 1024;

    const size_t byColumns = valuesPerColumn * nFeatures;
    const size_t perChunk  = byColumns > minValuesPerChunk ? byColumns : minValuesPerChunk;
    size_t nChunks         = nnz / perChunk;
    nChunks                = nChunks < 1 ? 1 : nChunks;
    nChunks                = nChunks > maxChunks ? maxChunks : nChunks;
    nChunks                = nChunks > nRows ? nRows : nChunks;
    return (nRows + nChunks - 1) / nChunks;
}

/// Merges the moments (cB, meanB, m2B) into (cA, meanA, m2A), Chan et al.; either side may be empty.
inline void mergeMoments(double & cA, double & meanA, double & m2A, double cB, double meanB, double m2B)
{
    const double c     = cA + cB;
    const double inv   = c > 0.0 ? 1.0 / c : 0.0;
    const double delta = meanB - meanA;
    meanA += delta * (cB * inv);
    m2A += m2B + delta * delta * (cA * cB * inv);
    cA = c;
}

/// Moments of all rows of a CSR table, from its stored values only. Each chunk runs Welford's
/// update over its values; the chunks are merged, and the implicit zeros join as one final
/// observation of value 0 and weight (nRows - nnz_j), as a weighted dense column would.
///
/// @param[in]  table      The CSR table
/// @param[in]  userSums   Precomputed column sums used for the mean, or nullptr
/// @param[out] moments    The moments; `sum` holds `userSums` when they are given
template <typename algorithmFPType, CpuType cpu>
Status computeMoments(CSRNumericTableIface & table, size_t nRows, size_t nFeatures, const algorithmFPType * userSums, Moments<cpu> & moments)
{
    Status st;
    DAAL_CHECK_STATUS(st, moments.init(nFeatures));
    moments.nRows = nRows;
    if (nRows == 0) return st;

    const size_t chunkRows = rowsPerChunk(nRows, nFeatures, table.getDataSize());
    const size_t nChunks   = (nRows + chunkRows - 1) / chunkRows;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures, slotsPerColumn);
    const size_t stride = nFeatures * slotsPerColumn;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nChunks, stride);
    TArrayScalable<double, cpu> acc(nChunks * stride);
    DAAL_CHECK_MALLOC(acc.get());

    SafeStatus safeStat;
    daal::threader_for(nChunks, 1, [&](size_t iChunk) {
        double * const a = acc.get() + iChunk * stride;
        PRAGMA_OMP_SIMD
        for (size_t j = 0; j < nFeatures; j++)
        {
            double * const col = a + j * slotsPerColumn;
            col[slotCount]     = 0.0;
            col[slotMean]      = 0.0;
            col[slotM2]        = 0.0;
            col[slotSum]       = 0.0;
            col[slotSumSq]     = 0.0;
            col[slotMin]       = MaxVal<double>::get();
            col[slotMax]       = -MaxVal<double>::get();
        }

        const size_t start = iChunk * chunkRows;
        const size_t rows  = (start + chunkRows > nRows) ? nRows - start : chunkRows;
        ReadRowsCSR<algorithmFPType, cpu> block(&table, start, rows);
        DAAL_CHECK_BLOCK_STATUS_THR(block);
        const algorithmFPType * const values = block.values();
        const size_t * const cols            = block.cols();
        const size_t nnz                     = block.rows()[rows] - block.rows()[0];
        // A sparse block can report success with no buffer when converting the value type fails to allocate.
        DAAL_CHECK_THR(nnz == 0 || (values && cols), ErrorMemoryAllocationFailed);

        // Not vectorized: a row may repeat a column index, which the format does not rule out.
        for (size_t k = 0; k < nnz; k++)
        {
            double * const col = a + (cols[k] - 1) * slotsPerColumn;
            const double x     = values[k];
            const double c     = col[slotCount] + 1.0;
            const double delta = x - col[slotMean];
            col[slotCount]     = c;
            col[slotMean] += delta / c;
            col[slotM2] += delta * (x - col[slotMean]);
            col[slotSum] += x;
            col[slotSumSq] += x * x;
            col[slotMin] = x < col[slotMin] ? x : col[slotMin];
            col[slotMax] = x > col[slotMax] ? x : col[slotMax];
        }
    });
    DAAL_CHECK_SAFE_STATUS();

    /* Merge the chunks in order, a block of columns per task, then add the implicit zeros */
    constexpr size_t columnsPerTask = 256;
    const size_t nColumnBlocks      = (nFeatures + columnsPerTask - 1) / columnsPerTask;
    const double n                  = double(nRows);
    const double invN               = 1.0 / n;
    daal::threader_for(nColumnBlocks, 1, [&](size_t iBlock) {
        const size_t jBegin = iBlock * columnsPerTask;
        const size_t width  = (jBegin + columnsPerTask > nFeatures) ? nFeatures - jBegin : columnsPerTask;
        double c[columnsPerTask], m[columnsPerTask], m2[columnsPerTask], s[columnsPerTask];
        double s2[columnsPerTask], lo[columnsPerTask], hi[columnsPerTask];

        const double * const first = acc.get() + jBegin * slotsPerColumn;
        PRAGMA_OMP_SIMD
        for (size_t t = 0; t < width; t++)
        {
            const double * const col = first + t * slotsPerColumn;
            c[t]                     = col[slotCount];
            m[t]                     = col[slotMean];
            m2[t]                    = col[slotM2];
            s[t]                     = col[slotSum];
            s2[t]                    = col[slotSumSq];
            lo[t]                    = col[slotMin];
            hi[t]                    = col[slotMax];
        }
        for (size_t iChunk = 1; iChunk < nChunks; iChunk++)
        {
            const double * const chunk = acc.get() + iChunk * stride + jBegin * slotsPerColumn;
            PRAGMA_OMP_SIMD
            for (size_t t = 0; t < width; t++)
            {
                const double * const col = chunk + t * slotsPerColumn;
                mergeMoments(c[t], m[t], m2[t], col[slotCount], col[slotMean], col[slotM2]);
                s[t] += col[slotSum];
                s2[t] += col[slotSumSq];
                lo[t] = col[slotMin] < lo[t] ? col[slotMin] : lo[t];
                hi[t] = col[slotMax] > hi[t] ? col[slotMax] : hi[t];
            }
        }

        PRAGMA_OMP_SIMD
        for (size_t t = 0; t < width; t++)
        {
            // A row can repeat a column, so the stored values may outnumber the rows.
            const double zeros = c[t] < n ? n - c[t] : 0.0;
            lo[t]              = (zeros > 0.0 && 0.0 < lo[t]) ? 0.0 : lo[t];
            hi[t]              = (zeros > 0.0 && 0.0 > hi[t]) ? 0.0 : hi[t];
            mergeMoments(c[t], m[t], m2[t], zeros, 0.0, 0.0);
        }
        if (userSums)
        {
            PRAGMA_OMP_SIMD
            for (size_t t = 0; t < width; t++)
            {
                const double userMean = double(userSums[jBegin + t]) * invN;
                m2[t] += n * (m[t] - userMean) * (m[t] - userMean);
                m[t] = userMean;
                s[t] = double(userSums[jBegin + t]);
            }
        }

        double * const sum   = moments.sum.get() + jBegin;
        double * const sumSq = moments.sumSq.get() + jBegin;
        double * const ssc   = moments.sumSqCen.get() + jBegin;
        double * const mean  = moments.mean.get() + jBegin;
        double * const min   = moments.min.get() + jBegin;
        double * const max   = moments.max.get() + jBegin;
        PRAGMA_OMP_SIMD
        for (size_t t = 0; t < width; t++)
        {
            sum[t]   = s[t];
            sumSq[t] = s2[t];
            ssc[t]   = m2[t];
            mean[t]  = m[t];
            min[t]   = lo[t];
            max[t]   = hi[t];
        }
    });
    return st;
}

/// Precomputed column sums of `sumCSR`, read from the table's basic statistics.
template <typename algorithmFPType, CpuType cpu>
Status readUserSums(NumericTable * dataTable, size_t nFeatures, TArray<algorithmFPType, cpu> & sums)
{
    NumericTable * const sumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();
    if (!sumsTable) return Status(ErrorPrecomputedSumNotAvailable);
    ReadRows<algorithmFPType, cpu> rows(sumsTable, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(rows);
    sums.reset(nFeatures);
    DAAL_CHECK_MALLOC(sums.get());
    for (size_t j = 0; j < nFeatures; j++)
    {
        sums[j] = rows.get()[j];
    }
    return Status();
}

/// Moments of the whole table for the batch CSR methods.
template <typename algorithmFPType, Method method, CpuType cpu>
Status computeBatch(NumericTable * dataTable, Result * result)
{
    CSRNumericTableIface * const csr = dynamic_cast<CSRNumericTableIface *>(dataTable);
    DAAL_CHECK(csr, ErrorIncorrectTypeOfInputNumericTable);
    const size_t nRows     = dataTable->getNumberOfRows();
    const size_t nFeatures = dataTable->getNumberOfColumns();

    Status st;
    TArray<algorithmFPType, cpu> userSums;
    if (method == sumCSR)
    {
        DAAL_CHECK_STATUS(st, (readUserSums<algorithmFPType, cpu>(dataTable, nFeatures, userSums)));
    }

    Moments<cpu> m;
    DAAL_CHECK_STATUS(st, (computeMoments<algorithmFPType, cpu>(*csr, nRows, nFeatures, userSums.get(), m)));
    if (nRows == 0) return st;

    WriteOnlyRows<algorithmFPType, cpu> out[lastResultId + 1];
    algorithmFPType * res[lastResultId + 1];
    for (size_t i = 0; i < lastResultId + 1; i++)
    {
        out[i].set(result->get((ResultId)i).get(), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(out[i]);
        res[i] = out[i].get();
    }

    const double invN   = 1.0 / double(nRows);
    const double invNM1 = 1.0 / double(nRows - 1);
    for (size_t j = 0; j < nFeatures; j++)
    {
        const double meanJ           = m.mean[j];
        const double varianceJ       = m.sumSqCen[j] * invNM1;
        res[minimum][j]              = algorithmFPType(m.min[j]);
        res[maximum][j]              = algorithmFPType(m.max[j]);
        res[sum][j]                  = algorithmFPType(m.sum[j]);
        res[sumSquares][j]           = algorithmFPType(m.sumSq[j]);
        res[sumSquaresCentered][j]   = algorithmFPType(m.sumSqCen[j]);
        res[mean][j]                 = algorithmFPType(meanJ);
        res[secondOrderRawMoment][j] = algorithmFPType(m.sumSq[j] * invN);
        res[variance][j]             = algorithmFPType(varianceJ);
    }
    MathInst<algorithmFPType, cpu>::vSqrt(nFeatures, res[variance], res[standardDeviation]);
    for (size_t j = 0; j < nFeatures; j++)
    {
        res[variation][j] = res[standardDeviation][j] / res[mean][j];
    }
    return st;
}

/// Moments of one block merged into the partial result for the online CSR methods.
template <typename algorithmFPType, Method method, CpuType cpu>
Status computeOnline(NumericTable * dataTable, PartialResult * partialResult, bool isOnline)
{
    CSRNumericTableIface * const csr = dynamic_cast<CSRNumericTableIface *>(dataTable);
    DAAL_CHECK(csr, ErrorIncorrectTypeOfInputNumericTable);
    const size_t nRows     = dataTable->getNumberOfRows();
    const size_t nFeatures = dataTable->getNumberOfColumns();

    Status st;
    TArray<algorithmFPType, cpu> userSums;
    if (method == sumCSR)
    {
        DAAL_CHECK_STATUS(st, (readUserSums<algorithmFPType, cpu>(dataTable, nFeatures, userSums)));
    }

    Moments<cpu> m;
    DAAL_CHECK_STATUS(st, (computeMoments<algorithmFPType, cpu>(*csr, nRows, nFeatures, userSums.get(), m)));

    WriteRows<algorithmFPType, cpu> part[lastPartialResultId + 1];
    algorithmFPType * res[lastPartialResultId + 1];
    for (size_t i = 0; i < lastPartialResultId + 1; i++)
    {
        NumericTable * t = partialResult->get((PartialResultId)i).get();
        DAAL_CHECK(t, ErrorNullPartialResult);
        part[i].set(t, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(part[i]);
        res[i] = part[i].get();
    }

    const double n1 = isOnline ? double(res[nObservations][0]) : 0.0;
    const double n2 = double(nRows);
    if (n2 == 0.0)
    {
        if (!isOnline) res[nObservations][0] = 0;
        return st;
    }

    if (n1 == 0.0)
    {
        for (size_t j = 0; j < nFeatures; j++)
        {
            res[partialMinimum][j]            = algorithmFPType(m.min[j]);
            res[partialMaximum][j]            = algorithmFPType(m.max[j]);
            res[partialSum][j]                = algorithmFPType(m.sum[j]);
            res[partialSumSquares][j]         = algorithmFPType(m.sumSq[j]);
            res[partialSumSquaresCentered][j] = algorithmFPType(m.sumSqCen[j]);
        }
    }
    else
    {
        /* Chan et al. pairwise update; (mean2 - mean1)^2 does not cancel the way the expanded form does */
        const double coeff = n1 * n2 / (n1 + n2);
        for (size_t j = 0; j < nFeatures; j++)
        {
            const double sum1  = res[partialSum][j];
            const double delta = m.mean[j] - sum1 / n1;
            const double min1  = res[partialMinimum][j];
            const double max1  = res[partialMaximum][j];

            res[partialMinimum][j]            = algorithmFPType(m.min[j] < min1 ? m.min[j] : min1);
            res[partialMaximum][j]            = algorithmFPType(m.max[j] > max1 ? m.max[j] : max1);
            res[partialSum][j]                = algorithmFPType(sum1 + m.sum[j]);
            res[partialSumSquares][j]         = algorithmFPType(double(res[partialSumSquares][j]) + m.sumSq[j]);
            res[partialSumSquaresCentered][j] = algorithmFPType(double(res[partialSumSquaresCentered][j]) + m.sumSqCen[j] + coeff * delta * delta);
        }
    }
    res[nObservations][0] = algorithmFPType(n1 + n2);
    return st;
}

} // namespace csr
} // namespace internal
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
