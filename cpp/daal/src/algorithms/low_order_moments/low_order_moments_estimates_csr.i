/* file: low_order_moments_estimates_csr.i */
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
//  Low order moments of CSR data for the estimates enabled by the _*_ENABLE_ macros; no include guard
//--
*/

/* The mean and the centered sum of squares are updated together; the raw sum of squares is on its own */
#undef _CSR_CENTERED_ENABLE_
#if defined _MEAN_ENABLE_ || defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
    #define _CSR_CENTERED_ENABLE_
#endif
#undef _CSR_SUM2_ENABLE_
#if defined _SUM2_ENABLE_ || defined _SORM_ENABLE_
    #define _CSR_SUM2_ENABLE_
#endif

/// Welford's update of one column's running moments with the observation `x`. The count is
/// always kept: it gives the number of implicit zeros of the column.
///
/// @tparam withSum  Whether the sum of the column is kept
template <bool withSum>
inline void addObservation(double * const col, const double x)
{
    const double c = col[slotCount] + 1.0;
    col[slotCount] = c;
#ifdef _CSR_CENTERED_ENABLE_
    const double delta = x - col[slotMean];
    col[slotMean] += delta / c;
    col[slotM2] += delta * (x - col[slotMean]);
#endif
    if (withSum) col[slotSum] += x;
#ifdef _CSR_SUM2_ENABLE_
    col[slotSumSq] += x * x;
#endif
#ifdef _MIN_ENABLE_
    col[slotMin] = x < col[slotMin] ? x : col[slotMin];
#endif
#ifdef _MAX_ENABLE_
    col[slotMax] = x > col[slotMax] ? x : col[slotMax];
#endif
}

/// Moments of all rows of a CSR table, from its stored values only. Each chunk runs Welford's
/// update over its values; the chunks are merged, and the implicit zeros join as one final
/// observation of value 0 and weight (nRows - nnz_j), as a weighted dense column would.
///
/// @tparam online         Whether the moments go to a partial result
/// @param[in]  table      The CSR table
/// @param[in]  userSums   Precomputed column sums used for the mean, or nullptr
/// @param[out] moments    The moments of the enabled estimates; `sum` holds `userSums` when they are given
template <typename algorithmFPType, CpuType cpu, bool online>
Status computeMoments(CSRNumericTableIface & table, size_t nRows, size_t nFeatures, const algorithmFPType * userSums, Moments<cpu> & moments)
{
    Status st;
    DAAL_CHECK_STATUS(st, moments.init(nFeatures));
    moments.nRows = nRows;
    if (nRows == 0) return st;

    /* The sum is a result of its own, or what an online partial result keeps for the mean */
#if defined _SUM_ENABLE_
    constexpr bool withSum = true;
#elif defined _MEAN_ENABLE_
    constexpr bool withSum = online;
#else
    constexpr bool withSum = false;
#endif

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
#ifdef _CSR_CENTERED_ENABLE_
            col[slotMean] = 0.0;
            col[slotM2]   = 0.0;
#endif
            if (withSum) col[slotSum] = 0.0;
#ifdef _CSR_SUM2_ENABLE_
            col[slotSumSq] = 0.0;
#endif
#ifdef _MIN_ENABLE_
            col[slotMin] = MaxVal<double>::get();
#endif
#ifdef _MAX_ENABLE_
            col[slotMax] = -MaxVal<double>::get();
#endif
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

        // Values of a column repeated in a row are summed, as in the dense row. Rows with sorted
        // indices sum the adjacent repeats; the others go through a per-column scratch.
        const size_t * const rowOffsets = block.rows();
        TArrayScalable<size_t, cpu> seenRow;
        TArrayScalable<double, cpu> pending;
        for (size_t r = 0; r < rows; r++)
        {
            const size_t begin = rowOffsets[r] - rowOffsets[0];
            const size_t end   = rowOffsets[r + 1] - rowOffsets[0];
            bool sorted        = true;
            for (size_t k = begin + 1; k < end && sorted; k++)
            {
                sorted = cols[k - 1] <= cols[k];
            }
            if (sorted)
            {
                for (size_t k = begin; k < end;)
                {
                    const size_t j = cols[k];
                    double x       = values[k];
                    while (++k < end && cols[k] == j)
                    {
                        x += values[k];
                    }
                    addObservation<withSum>(a + (j - 1) * slotsPerColumn, x);
                }
                continue;
            }
            if (!seenRow.get())
            {
                seenRow.reset(nFeatures);
                pending.reset(nFeatures);
                DAAL_CHECK_THR(seenRow.get() && pending.get(), ErrorMemoryAllocationFailed);
                services::internal::service_memset_seq<size_t, cpu>(seenRow.get(), noRow, nFeatures);
            }
            for (size_t k = begin; k < end; k++)
            {
                const size_t j = cols[k] - 1;
                pending[j]     = (seenRow[j] == r) ? pending[j] + values[k] : double(values[k]);
                seenRow[j]     = r;
            }
            for (size_t k = begin; k < end; k++)
            {
                const size_t j = cols[k] - 1;
                if (seenRow[j] != r) continue;
                addObservation<withSum>(a + j * slotsPerColumn, pending[j]);
                seenRow[j] = noRow;
            }
        }
    });
    DAAL_CHECK_SAFE_STATUS();

    /* Merge the chunks in order, a block of columns per task, then add the implicit zeros */
    const size_t nColumnBlocks = (nFeatures + columnsPerTask - 1) / columnsPerTask;
    const double n             = double(nRows);
    daal::threader_for(nColumnBlocks, 1, [&](size_t iBlock) {
        const size_t jBegin = iBlock * columnsPerTask;
        const size_t width  = (jBegin + columnsPerTask > nFeatures) ? nFeatures - jBegin : columnsPerTask;
        double c[columnsPerTask], s[columnsPerTask];
#ifdef _CSR_CENTERED_ENABLE_
        double m[columnsPerTask], m2[columnsPerTask];
#endif
#ifdef _CSR_SUM2_ENABLE_
        double s2[columnsPerTask];
#endif
#ifdef _MIN_ENABLE_
        double lo[columnsPerTask];
#endif
#ifdef _MAX_ENABLE_
        double hi[columnsPerTask];
#endif

        const double * const first = acc.get() + jBegin * slotsPerColumn;
        PRAGMA_OMP_SIMD
        for (size_t t = 0; t < width; t++)
        {
            const double * const col = first + t * slotsPerColumn;
            c[t]                     = col[slotCount];
#ifdef _CSR_CENTERED_ENABLE_
            m[t]  = col[slotMean];
            m2[t] = col[slotM2];
#endif
            if (withSum) s[t] = col[slotSum];
#ifdef _CSR_SUM2_ENABLE_
            s2[t] = col[slotSumSq];
#endif
#ifdef _MIN_ENABLE_
            lo[t] = col[slotMin];
#endif
#ifdef _MAX_ENABLE_
            hi[t] = col[slotMax];
#endif
        }
        for (size_t iChunk = 1; iChunk < nChunks; iChunk++)
        {
            const double * const chunk = acc.get() + iChunk * stride + jBegin * slotsPerColumn;
            PRAGMA_OMP_SIMD
            for (size_t t = 0; t < width; t++)
            {
                const double * const col = chunk + t * slotsPerColumn;
#ifdef _CSR_CENTERED_ENABLE_
                mergeMoments(c[t], m[t], m2[t], col[slotCount], col[slotMean], col[slotM2]);
#else
                c[t] += col[slotCount];
#endif
                if (withSum) s[t] += col[slotSum];
#ifdef _CSR_SUM2_ENABLE_
                s2[t] += col[slotSumSq];
#endif
#ifdef _MIN_ENABLE_
                lo[t] = col[slotMin] < lo[t] ? col[slotMin] : lo[t];
#endif
#ifdef _MAX_ENABLE_
                hi[t] = col[slotMax] > hi[t] ? col[slotMax] : hi[t];
#endif
            }
        }

        PRAGMA_OMP_SIMD
        for (size_t t = 0; t < width; t++)
        {
            const double zeros = n - c[t];
#ifdef _MIN_ENABLE_
            lo[t] = (zeros > 0.0 && 0.0 < lo[t]) ? 0.0 : lo[t];
#endif
#ifdef _MAX_ENABLE_
            hi[t] = (zeros > 0.0 && 0.0 > hi[t]) ? 0.0 : hi[t];
#endif
#ifdef _CSR_CENTERED_ENABLE_
            mergeMoments(c[t], m[t], m2[t], zeros, 0.0, 0.0);
#endif
        }
#ifdef _CSR_CENTERED_ENABLE_
        if (userSums)
        {
            const double invN = 1.0 / n;
            PRAGMA_OMP_SIMD
            for (size_t t = 0; t < width; t++)
            {
                const double userMean = double(userSums[jBegin + t]) * invN;
                m2[t] += n * (m[t] - userMean) * (m[t] - userMean);
                m[t] = userMean;
                if (withSum) s[t] = double(userSums[jBegin + t]);
            }
        }
#endif

        PRAGMA_OMP_SIMD
        for (size_t t = 0; t < width; t++)
        {
#ifdef _CSR_CENTERED_ENABLE_
            moments.mean[jBegin + t]     = m[t];
            moments.sumSqCen[jBegin + t] = m2[t];
#endif
            if (withSum) moments.sum[jBegin + t] = s[t];
#ifdef _CSR_SUM2_ENABLE_
            moments.sumSq[jBegin + t] = s2[t];
#endif
#ifdef _MIN_ENABLE_
            moments.min[jBegin + t] = lo[t];
#endif
#ifdef _MAX_ENABLE_
            moments.max[jBegin + t] = hi[t];
#endif
        }
    });
    return st;
}

/// Moments of the whole table for the batch CSR methods; writes the results of the enabled estimates.
template <typename algorithmFPType, Method method, CpuType cpu>
Status computeBatch(NumericTable * dataTable, Result * result)
{
    CSRNumericTableIface * const csr = dynamic_cast<CSRNumericTableIface *>(dataTable);
    DAAL_CHECK(csr, ErrorIncorrectTypeOfInputNumericTable);
    const size_t nRows     = dataTable->getNumberOfRows();
    const size_t nFeatures = dataTable->getNumberOfColumns();

    Status st;
    TArray<algorithmFPType, cpu> userSums;
#if defined _SUM_ENABLE_ || defined _MEAN_ENABLE_
    if (method == sumCSR)
    {
        DAAL_CHECK_STATUS(st, (readUserSums<algorithmFPType, cpu>(dataTable, nFeatures, userSums)));
    }
#endif

    Moments<cpu> m;
    DAAL_CHECK_STATUS(st, (computeMoments<algorithmFPType, cpu, false>(*csr, nRows, nFeatures, userSums.get(), m)));
    if (nRows == 0) return st;

    WriteOnlyRows<algorithmFPType, cpu> out[lastResultId + 1];
    algorithmFPType * res[lastResultId + 1];
    for (size_t i = 0; i < lastResultId + 1; i++)
    {
        out[i].set(result->get((ResultId)i).get(), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(out[i]);
        res[i] = out[i].get();
    }

#if defined _SORM_ENABLE_ || defined _VARC_ENABLE_
    const double n = double(nRows);
#endif
    PRAGMA_OMP_SIMD
    for (size_t j = 0; j < nFeatures; j++)
    {
#ifdef _MIN_ENABLE_
        res[minimum][j] = algorithmFPType(m.min[j]);
#endif
#ifdef _MAX_ENABLE_
        res[maximum][j] = algorithmFPType(m.max[j]);
#endif
#ifdef _SUM_ENABLE_
        res[sum][j] = algorithmFPType(m.sum[j]);
#endif
#ifdef _SUM2_ENABLE_
        res[sumSquares][j] = algorithmFPType(m.sumSq[j]);
#endif
#ifdef _SUM2C_ENABLE_
        res[sumSquaresCentered][j] = algorithmFPType(m.sumSqCen[j]);
#endif
#ifdef _MEAN_ENABLE_
        res[mean][j] = algorithmFPType(m.mean[j]);
#endif
#ifdef _SORM_ENABLE_
        res[secondOrderRawMoment][j] = algorithmFPType(m.sumSq[j] / n);
#endif
#ifdef _VARC_ENABLE_
        res[variance][j] = algorithmFPType(m.sumSqCen[j] / (n - 1.0));
#endif
    }
#ifdef _STDEV_ENABLE_
    MathInst<algorithmFPType, cpu>::vSqrt(nFeatures, res[variance], res[standardDeviation]);
#endif
#ifdef _VART_ENABLE_
    PRAGMA_OMP_SIMD
    for (size_t j = 0; j < nFeatures; j++)
    {
        res[variation][j] = res[standardDeviation][j] / res[mean][j];
    }
#endif
    return st;
}

/// Moments of one block merged into the partial result for the online CSR methods; updates the
/// partial results of the enabled estimates.
template <typename algorithmFPType, Method method, CpuType cpu>
Status computeOnline(NumericTable * dataTable, PartialResult * partialResult, bool isOnline)
{
    CSRNumericTableIface * const csr = dynamic_cast<CSRNumericTableIface *>(dataTable);
    DAAL_CHECK(csr, ErrorIncorrectTypeOfInputNumericTable);
    const size_t nRows     = dataTable->getNumberOfRows();
    const size_t nFeatures = dataTable->getNumberOfColumns();

    Status st;
    TArray<algorithmFPType, cpu> userSums;
#if defined _SUM_ENABLE_ || defined _MEAN_ENABLE_
    if (method == sumCSR)
    {
        DAAL_CHECK_STATUS(st, (readUserSums<algorithmFPType, cpu>(dataTable, nFeatures, userSums)));
    }
#endif

    Moments<cpu> m;
    DAAL_CHECK_STATUS(st, (computeMoments<algorithmFPType, cpu, true>(*csr, nRows, nFeatures, userSums.get(), m)));

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
        PRAGMA_OMP_SIMD
        for (size_t j = 0; j < nFeatures; j++)
        {
#ifdef _MIN_ENABLE_
            res[partialMinimum][j] = algorithmFPType(m.min[j]);
#endif
#ifdef _MAX_ENABLE_
            res[partialMaximum][j] = algorithmFPType(m.max[j]);
#endif
#if defined _SUM_ENABLE_ || defined _MEAN_ENABLE_
            res[partialSum][j] = algorithmFPType(m.sum[j]);
#endif
#ifdef _CSR_SUM2_ENABLE_
            res[partialSumSquares][j] = algorithmFPType(m.sumSq[j]);
#endif
#if defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
            res[partialSumSquaresCentered][j] = algorithmFPType(m.sumSqCen[j]);
#endif
        }
    }
    else
    {
        PRAGMA_OMP_SIMD
        for (size_t j = 0; j < nFeatures; j++)
        {
#ifdef _MIN_ENABLE_
            const double min1      = res[partialMinimum][j];
            res[partialMinimum][j] = algorithmFPType(m.min[j] < min1 ? m.min[j] : min1);
#endif
#ifdef _MAX_ENABLE_
            const double max1      = res[partialMaximum][j];
            res[partialMaximum][j] = algorithmFPType(m.max[j] > max1 ? m.max[j] : max1);
#endif
#if defined _SUM_ENABLE_ || defined _MEAN_ENABLE_
            const double sum1 = res[partialSum][j];
    #if defined _SUM2C_ENABLE_ || defined _VARC_ENABLE_ || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
            /* Chan et al. pairwise update; (mean2 - mean1)^2 does not cancel the way the expanded form does */
            const double coeff                = n1 * n2 / (n1 + n2);
            const double delta                = m.mean[j] - sum1 / n1;
            res[partialSumSquaresCentered][j] = algorithmFPType(double(res[partialSumSquaresCentered][j]) + m.sumSqCen[j] + coeff * delta * delta);
    #endif
            res[partialSum][j] = algorithmFPType(sum1 + m.sum[j]);
#endif
#ifdef _CSR_SUM2_ENABLE_
            res[partialSumSquares][j] = algorithmFPType(double(res[partialSumSquares][j]) + m.sumSq[j]);
#endif
        }
    }
    res[nObservations][0] = algorithmFPType(n1 + n2);
    return st;
}
