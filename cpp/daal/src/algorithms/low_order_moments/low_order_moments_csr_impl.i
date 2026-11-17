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
        min.reset(nFeatures);
        max.reset(nFeatures);
        DAAL_CHECK_MALLOC(sum.get() && sumSq.get() && sumSqCen.get() && min.get() && max.get());
        return Status();
    }

    size_t nRows = 0;
    TArray<double, cpu> sum;
    TArray<double, cpu> sumSq;
    TArray<double, cpu> sumSqCen;
    TArray<double, cpu> min;
    TArray<double, cpu> max;
};

/// Thread-local accumulators of one pass over the non-zero values.
template <CpuType cpu>
struct PassAccumulators
{
    static PassAccumulators * create(size_t nFeatures, size_t nArrays)
    {
        PassAccumulators * acc = new PassAccumulators();
        if (!acc) return nullptr;
        acc->values.reset(nFeatures * nArrays);
        if (!acc->values.get())
        {
            delete acc;
            return nullptr;
        }
        return acc;
    }

    TArrayScalable<double, cpu> values;
};

/// Splits the rows into blocks that each hold enough non-zero values to amortize
/// the per-block accumulators of length nFeatures.
///
/// @param[in]  nRows      Number of rows
/// @param[in]  nFeatures  Number of columns
/// @param[in]  nnz        Number of stored values
/// @return     Number of rows per block
inline size_t rowsPerBlock(size_t nRows, size_t nFeatures, size_t nnz)
{
    // A block reads at least this many values per accumulator entry it allocates, initializes and reduces.
    constexpr size_t valuesPerAccumulator = 16;
    const size_t nThreads                 = threader_get_threads_number();
    const size_t minNnzPerBlock           = valuesPerAccumulator * nFeatures;
    size_t nBlocks                        = nThreads;
    const size_t byNnz                    = nnz / minNnzPerBlock;
    nBlocks                               = byNnz < nBlocks ? byNnz : nBlocks;
    nBlocks                               = nBlocks < 1 ? 1 : nBlocks;
    nBlocks                               = nRows < nBlocks ? nRows : nBlocks;
    return (nRows + nBlocks - 1) / nBlocks;
}

/// Moments of all rows of a CSR table, from its non-zero values only: implicit
/// zeros enter the sums as zero, the minimum and maximum as a zero value, and the
/// centered sum of squares as (nRows - nnz_j) * mean_j^2.
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

    const size_t blockRows = rowsPerBlock(nRows, nFeatures, table.getDataSize());
    const size_t nBlocks   = (nRows + blockRows - 1) / blockRows;

    /* Pass 1: sums, sums of squares, extremes and non-zero counts */
    TArray<double, cpu> count(nFeatures);
    DAAL_CHECK_MALLOC(count.get());
    {
        SafeStatus safeStat;
        daal::tls<PassAccumulators<cpu> *> tls([&]() {
            PassAccumulators<cpu> * acc = PassAccumulators<cpu>::create(nFeatures, 5);
            if (!acc)
            {
                safeStat.add(ErrorMemoryAllocationFailed);
                return acc;
            }
            double * v = acc->values.get();
            for (size_t j = 0; j < nFeatures; j++)
            {
                v[j]                 = 0.0;
                v[nFeatures + j]     = 0.0;
                v[2 * nFeatures + j] = MaxVal<double>::get();
                v[3 * nFeatures + j] = -MaxVal<double>::get();
                v[4 * nFeatures + j] = 0.0;
            }
            return acc;
        });

        daal::threader_for(nBlocks, 1, [&](size_t iBlock) {
            PassAccumulators<cpu> * acc = tls.local();
            if (!acc) return;
            const size_t start = iBlock * blockRows;
            const size_t rows  = (start + blockRows > nRows) ? nRows - start : blockRows;
            ReadRowsCSR<algorithmFPType, cpu> block(&table, start, rows);
            DAAL_CHECK_BLOCK_STATUS_THR(block);
            const algorithmFPType * values = block.values();
            const size_t * cols            = block.cols();
            const size_t nnz               = block.rows()[rows] - block.rows()[0];

            double * sum   = acc->values.get();
            double * sumSq = sum + nFeatures;
            double * min   = sum + 2 * nFeatures;
            double * max   = sum + 3 * nFeatures;
            double * cnt   = sum + 4 * nFeatures;
            for (size_t k = 0; k < nnz; k++)
            {
                const size_t j = cols[k] - 1;
                const double x = values[k];
                sum[j] += x;
                sumSq[j] += x * x;
                min[j] = x < min[j] ? x : min[j];
                max[j] = x > max[j] ? x : max[j];
                cnt[j] += 1.0;
            }
        });
        DAAL_CHECK_SAFE_STATUS();

        double * sum   = moments.sum.get();
        double * sumSq = moments.sumSq.get();
        double * min   = moments.min.get();
        double * max   = moments.max.get();
        double * cnt   = count.get();
        for (size_t j = 0; j < nFeatures; j++)
        {
            sum[j]   = 0.0;
            sumSq[j] = 0.0;
            min[j]   = MaxVal<double>::get();
            max[j]   = -MaxVal<double>::get();
            cnt[j]   = 0.0;
        }
        tls.reduce([&](PassAccumulators<cpu> * acc) {
            if (!acc) return;
            const double * v = acc->values.get();
            for (size_t j = 0; j < nFeatures; j++)
            {
                sum[j] += v[j];
                sumSq[j] += v[nFeatures + j];
                min[j] = v[2 * nFeatures + j] < min[j] ? v[2 * nFeatures + j] : min[j];
                max[j] = v[3 * nFeatures + j] > max[j] ? v[3 * nFeatures + j] : max[j];
                cnt[j] += v[4 * nFeatures + j];
            }
            delete acc;
        });

        const double n = double(nRows);
        for (size_t j = 0; j < nFeatures; j++)
        {
            if (cnt[j] < n)
            {
                min[j] = 0.0 < min[j] ? 0.0 : min[j];
                max[j] = 0.0 > max[j] ? 0.0 : max[j];
            }
            if (userSums) sum[j] = double(userSums[j]);
        }
    }

    /* Pass 2: squared deviations of the stored values from the mean */
    {
        const double invN = 1.0 / double(nRows);
        TArray<double, cpu> mean(nFeatures);
        DAAL_CHECK_MALLOC(mean.get());
        for (size_t j = 0; j < nFeatures; j++)
        {
            mean[j] = moments.sum[j] * invN;
        }

        SafeStatus safeStat;
        daal::tls<PassAccumulators<cpu> *> tls([&]() {
            PassAccumulators<cpu> * acc = PassAccumulators<cpu>::create(nFeatures, 1);
            if (!acc)
            {
                safeStat.add(ErrorMemoryAllocationFailed);
                return acc;
            }
            service_memset_seq<double, cpu>(acc->values.get(), 0.0, nFeatures);
            return acc;
        });

        const double * m = mean.get();
        daal::threader_for(nBlocks, 1, [&](size_t iBlock) {
            PassAccumulators<cpu> * acc = tls.local();
            if (!acc) return;
            const size_t start = iBlock * blockRows;
            const size_t rows  = (start + blockRows > nRows) ? nRows - start : blockRows;
            ReadRowsCSR<algorithmFPType, cpu> block(&table, start, rows);
            DAAL_CHECK_BLOCK_STATUS_THR(block);
            const algorithmFPType * values = block.values();
            const size_t * cols            = block.cols();
            const size_t nnz               = block.rows()[rows] - block.rows()[0];

            double * ssc = acc->values.get();
            for (size_t k = 0; k < nnz; k++)
            {
                const size_t j = cols[k] - 1;
                const double d = double(values[k]) - m[j];
                ssc[j] += d * d;
            }
        });
        DAAL_CHECK_SAFE_STATUS();

        double * ssc = moments.sumSqCen.get();
        service_memset_seq<double, cpu>(ssc, 0.0, nFeatures);
        tls.reduce([&](PassAccumulators<cpu> * acc) {
            if (!acc) return;
            const double * v = acc->values.get();
            for (size_t j = 0; j < nFeatures; j++)
            {
                ssc[j] += v[j];
            }
            delete acc;
        });

        const double n   = double(nRows);
        const double * c = count.get();
        for (size_t j = 0; j < nFeatures; j++)
        {
            ssc[j] += (n - c[j]) * m[j] * m[j];
        }
    }
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
        const double meanJ           = m.sum[j] * invN;
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
            const double delta = m.sum[j] / n2 - sum1 / n1;
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
