/* file: covariance_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Covariance matrix computation algorithm implementation
//--
*/

#ifndef __COVARIANCE_IMPL_I__
#define __COVARIANCE_IMPL_I__

#include "data_management/data/numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_lapack.h"
#include "src/externals/service_spblas.h"
#include "src/externals/service_stat.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/threading/threading.h"
#include "src/services/service_profiler.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status prepareSums(NumericTable * dataTable, algorithmFPType * sums)
{
    DAAL_PROFILER_TASK(Covariance::prepareSums);

    const size_t nFeatures = dataTable->getNumberOfColumns();
    int result             = 0;

    if (method == sumDense || method == sumCSR)
    {
        NumericTable * dataSumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();
        DEFINE_TABLE_BLOCK(ReadRows, userSumsBlock, dataSumsTable);

        const size_t nFeaturesSize = nFeatures * sizeof(algorithmFPType);
        result                     = daal::services::internal::daal_memcpy_s(sums, nFeaturesSize, userSumsBlock.get(), nFeaturesSize);
    }
    else
    {
        const algorithmFPType zero = 0.0;
        services::internal::service_memset<algorithmFPType, cpu>(sums, zero, nFeatures);
    }

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
services::Status prepareCrossProduct(size_t nFeatures, algorithmFPType * crossProduct)
{
    DAAL_PROFILER_TASK_WITH_ARGS(Covariance::prepareCrossProduct, nFeatures);

    const algorithmFPType zero = 0.0;
    services::internal::service_memset<algorithmFPType, cpu>(crossProduct, zero, nFeatures * nFeatures);
    return services::Status();
}

enum class ReducerErrorCode
{
    ok                  = 0, /// No error
    memAllocationFailed = 1, /// Memory allocation failed
    intOverflow         = 2, /// Integer overflow
    badCast             = 3  /// Cannot cast base daal::Reducer to derived class
};

/// Implements daal::Reducer interface for the sums reduction part of the Covariance algorithm computations.
///
/// @tparam algorithmFPType     Data type to store partial results: double or float.
/// @tparam cpu                 Variant of the CPU instruction set: SSE2, AVX2, AVX512, ARM SVE, etc.
template <typename algorithmFPType, CpuType cpu>
class SumsReducer : public daal::Reducer
{
public:
    /// Status of the computation.
    ReducerErrorCode errorCode;

    /// Get pointer to the array of partial sums.
    inline algorithmFPType * sums() { return _sumsArray.get(); }

    /// Get pointer to the constant array of partial sums.
    inline const algorithmFPType * sums() const { return _sumsArray.get(); }

    /// Construct and initialize the thread-local partial results
    ///
    /// @param[in] dataTable        Input data table that stores matrix X for which the cross-product matrix and sums are computed
    /// @param[in] numRowsInBlock   Number of rows in the block of the input data table - a mininal number of rows to be processed by a thread
    /// @param[in] numBlocks        Number of blocks of rows in the input data table
    SumsReducer(NumericTable * dataTable, DAAL_INT numRowsInBlock, size_t numBlocks)
        : _dataTable(dataTable), _numRowsInBlock(numRowsInBlock), _numBlocks(numBlocks), _nFeatures(dataTable->getNumberOfColumns())
    {
        if (!_sumsArray.reset(_nFeatures))
        {
            errorCode = ReducerErrorCode::memAllocationFailed;
            return;
        }
        errorCode = ReducerErrorCode::ok;
    }

    /// New and delete operators are overloaded to use scalable memory allocator that doesn't block threads
    /// if memory allocations are executed concurrently.
    void * operator new(size_t size) { return service_scalable_malloc<unsigned char, cpu>(size); }

    void operator delete(void * p) { service_scalable_free<unsigned char, cpu>((unsigned char *)p); }

    /// Constructs a thread-local partial result and initializes it with zeros.
    /// Must be able to run concurrently with `update` and `join` methods.
    ///
    /// @return Pointer to the partial result of the covariance algorithm.
    virtual ReducerUniquePtr create() const override
    {
        return daal::internal::makeUnique<SumsReducer<algorithmFPType, cpu>, DAAL_BASE_CPU>(_dataTable, _numRowsInBlock, _numBlocks);
    }

    /// Updates partial sums array with the data
    /// from the blocks of input data table in the sub-interval [begin, end).
    ///
    /// @param begin Index of the starting block of the input data table.
    /// @param end   Index of the block after the last one in the sub-range.
    virtual void update(size_t begin, size_t end) override
    {
        DAAL_PROFILER_THREADING_TASK(sums_reducer.update);
        if (errorCode != ReducerErrorCode::ok)
        {
            return;
        }

        const size_t numRowsInLastBlock = _numRowsInBlock + _dataTable->getNumberOfRows() - _numBlocks * _numRowsInBlock;
        algorithmFPType one             = 1.0;

        /// Process blocks of the input data table
        for (size_t iBlock = begin; iBlock < end; ++iBlock)
        {
            size_t nRows    = ((iBlock + 1 < _numBlocks) ? _numRowsInBlock : numRowsInLastBlock);
            size_t startRow = iBlock * _numRowsInBlock;

            ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD(_dataTable, startRow, nRows);

            algorithmFPType * dataBlock = const_cast<algorithmFPType *>(dataTableBD.get());

            algorithmFPType * sumsPtr = sums();
            if (!sumsPtr || !dataBlock)
            {
                errorCode = ReducerErrorCode::memAllocationFailed;
                return;
            }

            for (DAAL_INT i = 0; i < nRows; i++)
            {
                PRAGMA_OMP_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (DAAL_INT j = 0; j < _nFeatures; j++)
                {
                    sumsPtr[j] += dataBlock[i * _nFeatures + j];
                }
            }
        }
    }

    /// Merge the partial result with the data from another thread.
    ///
    /// @param otherReducer Pointer to the other thread's partial result.
    virtual void join(Reducer * otherReducer) override
    {
        if (errorCode != ReducerErrorCode::ok)
        {
            return;
        }
        DAAL_PROFILER_THREADING_TASK(reducer.join);
        SumsReducer<algorithmFPType, cpu> * other = dynamic_cast<SumsReducer<algorithmFPType, cpu> *>(otherReducer);
        if (!other)
        {
            errorCode = ReducerErrorCode::badCast;
            return;
        }
        if (other->errorCode != ReducerErrorCode::ok)
        {
            errorCode = other->errorCode;
            return;
        }
        const algorithmFPType * otherSums = other->sums();
        algorithmFPType * thisSums        = sums();

        if (!thisSums || !otherSums)
        {
            errorCode = ReducerErrorCode::memAllocationFailed;
            return;
        }

        PRAGMA_OMP_SIMD_ARGS(aligned(thisSums, otherSums : DAAL_MALLOC_DEFAULT_ALIGNMENT))
        for (size_t i = 0; i < _nFeatures; i++)
        {
            thisSums[i] += otherSums[i];
        }
    }

private:
    /// Pointer to the input data table that stores matrix X for which the cross-product matrix and sums are computed.
    NumericTable * _dataTable;
    /// Number of features in the input data table.
    DAAL_INT _nFeatures;
    /// Number of rows in the block of the input data table - a mininal number of rows to be processed by a thread.
    DAAL_INT _numRowsInBlock;
    /// Number of blocks of rows in the input data table.
    size_t _numBlocks;
    /// Thread-local array of partial sums of size `_nFeatures`.
    /// The array is used only if the input data is not normalized.
    TArrayScalableCalloc<algorithmFPType, cpu> _sumsArray;
};

/// Implements daal::Reducer interface for the dense Covariance algorithm computations.
///
/// @tparam algorithmFPType     Data type to store partial results: double or float.
/// @tparam cpu                 Variant of the CPU instruction set: SSE2, AVX2, AVX512, ARM SVE, etc.
template <typename algorithmFPType, CpuType cpu>
class CovarianceReducer : public daal::Reducer
{
public:
    /// Status of the computation.
    ReducerErrorCode errorCode;

    /// Get pointer to the partial cross-product matrix.
    inline algorithmFPType * means() { return _meansArray.get(); }
    /// Get pointer to the partial cross-product matrix.
    inline algorithmFPType * crossProduct() { return _crossProductArray.get(); }

    /// Get pointer to the constant partial means array.
    inline const algorithmFPType * means() const { return _meansArray.get(); }
    /// Get pointer to the constant partial cross-product matrix.
    inline const algorithmFPType * crossProduct() const { return _crossProductArray.get(); }

    /// Construct and initialize the thread-local partial results
    ///
    /// @param[in] dataTable        Input data table that stores matrix X for which the cross-product matrix and sums are computed
    /// @param[in] numRowsInBlock   Number of rows in the block of the input data table - a mininal number of rows to be processed by a thread
    /// @param[in] numBlocks        Number of blocks of rows in the input data table
    /// @param[in] isNormalized     Flag that specifies whether the input data is normalized
    CovarianceReducer(NumericTable * dataTable, const algorithmFPType * sums, DAAL_INT numRowsInBlock, size_t numBlocks, bool isNormalized)
        : _dataTable(dataTable),
          _sums(sums),
          _numRowsInBlock(numRowsInBlock),
          _numBlocks(numBlocks),
          _nFeatures(dataTable->getNumberOfColumns()),
          _isNormalized(isNormalized)
    {
        bool isOverflow = false;
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_BOOL(size_t, _nFeatures, _nFeatures, isOverflow);
        if (isOverflow)
        {
            errorCode = ReducerErrorCode::intOverflow;
            return;
        }
        if (!_crossProductArray.reset(_nFeatures * _nFeatures))
        {
            errorCode = ReducerErrorCode::memAllocationFailed;
            return;
        }
        if (!isNormalized)
        {
            if (!sums)
            {
                errorCode = ReducerErrorCode::memAllocationFailed;
                return;
            }
            if (!_meansArray.reset(_nFeatures))
            {
                errorCode = ReducerErrorCode::memAllocationFailed;
                return;
            }
            algorithmFPType * meansPtr     = means();
            const algorithmFPType invNRows = 1.0 / (algorithmFPType)_dataTable->getNumberOfRows();
            for (DAAL_INT i = 0; i < _nFeatures; i++)
            {
                meansPtr[i] = sums[i] * invNRows;
            }
            if (!_centeredDataBlockArray.reset(_numRowsInBlock * _nFeatures))
            {
                errorCode = ReducerErrorCode::memAllocationFailed;
                return;
            }
        }

        errorCode = ReducerErrorCode::ok;
    }

    /// New and delete operators are overloaded to use scalable memory allocator that doesn't block threads
    /// if memory allocations are executed concurrently.
    void * operator new(size_t size) { return service_scalable_malloc<unsigned char, cpu>(size); }

    void operator delete(void * p) { service_scalable_free<unsigned char, cpu>((unsigned char *)p); }

    /// Constructs a thread-local partial result and initializes it with zeros.
    /// Must be able to run concurrently with `update` and `join` methods.
    ///
    /// @return Pointer to the partial result of the covariance algorithm.
    virtual ReducerUniquePtr create() const override
    {
        return daal::internal::makeUnique<CovarianceReducer<algorithmFPType, cpu>, DAAL_BASE_CPU>(_dataTable, _sums, _numRowsInBlock, _numBlocks,
                                                                                                  _isNormalized);
    }

    /// Updates partial cross-product matrix and, if required, sums with the data
    /// from the blocks of input data table in the sub-interval [begin, end).
    ///
    /// @param begin Index of the starting block of the input data table.
    /// @param end   Index of the block after the last one in the sub-range.
    virtual void update(size_t begin, size_t end) override
    {
        DAAL_PROFILER_THREADING_TASK(reducer.update);
        if (errorCode != ReducerErrorCode::ok)
        {
            return;
        }
        algorithmFPType * crossProductPtr = crossProduct();

        if (!crossProductPtr)
        {
            errorCode = ReducerErrorCode::memAllocationFailed;
            return;
        }

        bool isOverflow = false;
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_BOOL(size_t, _numRowsInBlock, _nFeatures, isOverflow);
        if (isOverflow)
        {
            errorCode = ReducerErrorCode::intOverflow;
            return;
        }
        const size_t numRowsInLastBlock = _numRowsInBlock + _dataTable->getNumberOfRows() - _numBlocks * _numRowsInBlock;
        algorithmFPType one             = 1.0;

        algorithmFPType * centeredDataBlock = nullptr;

        /// Process blocks of the input data table
        for (size_t iBlock = begin; iBlock < end; ++iBlock)
        {
            size_t nRows    = ((iBlock + 1 < _numBlocks) ? _numRowsInBlock : numRowsInLastBlock);
            size_t startRow = iBlock * _numRowsInBlock;

            ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD(_dataTable, startRow, nRows);
            if (!dataTableBD.get())
            {
                errorCode = ReducerErrorCode::memAllocationFailed;
                return;
            }
            algorithmFPType * dataBlock = const_cast<algorithmFPType *>(dataTableBD.get());
            centeredDataBlock           = dataBlock;

            if (!_isNormalized)
            {
                centeredDataBlock = _centeredDataBlockArray.get();
                if (!_centeredDataBlockArray.get())
                {
                    errorCode = ReducerErrorCode::memAllocationFailed;
                    return;
                }

                const algorithmFPType * meansPtr = means();

                /* Sum input array elements in case of non-normalized data */
                for (DAAL_INT i = 0; i < nRows; i++)
                {
                    PRAGMA_OMP_SIMD
                    PRAGMA_VECTOR_ALWAYS
                    for (DAAL_INT j = 0; j < _nFeatures; j++)
                    {
                        centeredDataBlock[i * _nFeatures + j] = dataBlock[i * _nFeatures + j] - meansPtr[j];
                    }
                }
            }
            /// Update the cross-product matrix with the data from the block
            {
                DAAL_PROFILER_THREADING_TASK(reducer.update.syrkData);
                BlasInst<algorithmFPType, cpu>::xsyrk("U", "N", &_nFeatures, reinterpret_cast<DAAL_INT *>(&nRows), &one, centeredDataBlock,
                                                      &_nFeatures, &one, crossProductPtr, &_nFeatures);
            }
        }
    }

    /// Merge the partial result with the data from another thread.
    ///
    /// @param otherReducer Pointer to the other thread's partial result.
    virtual void join(Reducer * otherReducer) override
    {
        if (errorCode != ReducerErrorCode::ok)
        {
            return;
        }
        DAAL_PROFILER_THREADING_TASK(reducer.join);
        CovarianceReducer<algorithmFPType, cpu> * other = dynamic_cast<CovarianceReducer<algorithmFPType, cpu> *>(otherReducer);
        if (!other)
        {
            errorCode = ReducerErrorCode::badCast;
            return;
        }
        if (other->errorCode != ReducerErrorCode::ok)
        {
            errorCode = other->errorCode;
            return;
        }
        const algorithmFPType * otherCrossProduct = other->crossProduct();
        algorithmFPType * thisCrossProduct        = crossProduct();

        if (!thisCrossProduct || !otherCrossProduct)
        {
            errorCode = ReducerErrorCode::memAllocationFailed;
            return;
        }

        /// It is safe to use aligned loads and stores because the data in TArrayScalableCalloc data structures is aligned
        PRAGMA_OMP_SIMD_ARGS(aligned(thisCrossProduct, otherCrossProduct : DAAL_MALLOC_DEFAULT_ALIGNMENT))
        for (size_t i = 0; i < (_nFeatures * _nFeatures); i++)
        {
            thisCrossProduct[i] += otherCrossProduct[i];
        }
    }

private:
    /// Pointer to the input data table that stores matrix X for which the cross-product matrix and sums are computed.
    NumericTable * _dataTable;
    /// Number of features in the input data table.
    DAAL_INT _nFeatures;
    /// Number of rows in the block of the input data table - a mininal number of rows to be processed by a thread.
    DAAL_INT _numRowsInBlock;
    /// Number of blocks of rows in the input data table.
    size_t _numBlocks;
    /// Pointer to the array of sums of size `_nFeatures`.
    /// The array is used only if the input data is not normalized.
    const algorithmFPType * _sums;
    /// Thread-local array of means of size `_nFeatures`.
    /// The array is used only if the input data is not normalized.
    TArrayScalableCalloc<algorithmFPType, cpu> _meansArray;
    /// Thread-local matrix of size `_numRowsInBlock * _nFeatures` that stores the centered data block.
    /// The array is used only if the input data is not normalized.
    TArrayScalableCalloc<algorithmFPType, cpu> _centeredDataBlockArray;
    /// Thread-local partial cross-product matrix of size `_nFeatures * _nFeatures`.
    TArrayScalableCalloc<algorithmFPType, cpu> _crossProductArray;
    /// Flag that specifies whether the input data is normalized.
    bool _isNormalized;
};

/* Optimal block size for AVX512 low dimensions case (1024) and other CPU's and cases (140) */
template <CpuType cpu>
static inline size_t getBlockSize(size_t nrows)
{
    return 140;
}

#if defined(TARGET_X86_64)
    #define DAAL_CPU_TYPE avx512
#elif defined(TARGET_ARM)
    #define DAAL_CPU_TYPE sve
#elif defined(TARGET_RISCV64)
    #define DAAL_CPU_TYPE rv64
#endif

template <>
inline size_t getBlockSize<DAAL_CPU_TYPE>(size_t nrows)
{
    return (nrows > 5000 && nrows <= 50000) ? 1024 : 140;
}

template <typename algorithmFPType, CpuType cpu>
services::Status computeDenseCrossProductsAndSumsBatched(const size_t nFeatures, const size_t nVectors, NumericTable * dataTable,
                                                         algorithmFPType * crossProduct, algorithmFPType * sums, const bool isNormalized,
                                                         const bool assumeCentered, const DAAL_INT64 numRowsInBlock, const DAAL_INT64 grainSize)
{
    DAAL_PROFILER_TASK_WITH_ARGS(Covariance::computeDenseCrossProductsAndSumsBatched, numRowsInBlock, grainSize);
    /* Inverse number of rows (for normalization) */
    const algorithmFPType nVectorsInv = 1.0 / (double)(nVectors);

    /* Split rows by blocks */

    size_t numBlocks = nVectors / numRowsInBlock;
    if (numBlocks * numRowsInBlock < nVectors)
    {
        numBlocks++;
    }

    if (!isNormalized && !assumeCentered)
    {
        SumsReducer<algorithmFPType, cpu> sumsResult(dataTable, numRowsInBlock, numBlocks);
        if (!sumsResult.sums())
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }

        const size_t sumsNumBlocks     = numBlocks;
        const DAAL_INT64 sumsGrainSize = grainSize;

        daal::static_threader_reduce(sumsNumBlocks, sumsGrainSize, sumsResult);
        if (sumsResult.errorCode != ReducerErrorCode::ok)
        {
            if (sumsResult.errorCode == ReducerErrorCode::memAllocationFailed)
            {
                return services::Status(services::ErrorMemoryAllocationFailed);
            }
            if (sumsResult.errorCode == ReducerErrorCode::intOverflow)
            {
                return services::Status(services::ErrorBufferSizeIntegerOverflow);
            }
            if (sumsResult.errorCode == ReducerErrorCode::badCast)
            {
                return services::Status(services::ErrorCovarianceInternal);
            }
        }

        const algorithmFPType * resultSums = sumsResult.sums();
        daal::services::internal::daal_memcpy_s(sums, nFeatures * sizeof(algorithmFPType), resultSums, nFeatures * sizeof(algorithmFPType));
    }

    CovarianceReducer<algorithmFPType, cpu> result(dataTable, sums, numRowsInBlock, numBlocks, isNormalized);
    if (!result.crossProduct())
    {
        return services::Status(services::ErrorMemoryAllocationFailed);
    }

    /* Reduce input matrix X into cross product Xt X and a vector of column sums */
    daal::static_threader_reduce(numBlocks, grainSize, result);
    if (result.errorCode != ReducerErrorCode::ok)
    {
        if (result.errorCode == ReducerErrorCode::memAllocationFailed)
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }
        if (result.errorCode == ReducerErrorCode::intOverflow)
        {
            return services::Status(services::ErrorBufferSizeIntegerOverflow);
        }
        if (result.errorCode == ReducerErrorCode::badCast)
        {
            return services::Status(services::ErrorCovarianceInternal);
        }
    }

    const algorithmFPType * resultCrossProduct = result.crossProduct();
    if (result.errorCode != ReducerErrorCode::ok || !resultCrossProduct)
    {
        return services::Status(services::ErrorMemoryAllocationFailed);
    }

    /* If data is not normalized, perform subtractions of(sums[i]*sums[j])/n */

    daal::services::internal::daal_memcpy_s(crossProduct, nFeatures * nFeatures * sizeof(algorithmFPType), resultCrossProduct,
                                            nFeatures * nFeatures * sizeof(algorithmFPType));

    return services::Status();
}

/// Computes the cross-product matrix and, if required, the sums of the input data table
/// without splitting the data into blocks.
///
/// If the data is not normalized and is not assumed to be centered, the cross-product matrix is
/// computed on the mean-centered copy of the data. This avoids the catastrophic cancellation that
/// occurs when the cross-product of the raw data is corrected by the outer product of the sums
/// afterwards, which is critical for the data with large means and small variances.
///
/// @tparam algorithmFPType         Data type to store the results: double or float.
/// @tparam cpu                     Variant of the CPU instruction set.
///
/// @param[in]     nFeatures            Number of features (columns) in the input data table.
/// @param[in]     nVectors             Number of observations (rows) in the input data table.
/// @param[in]     dataTable            Input data table that stores matrix X.
/// @param[out]    crossProduct         Resulting cross-product matrix of size `nFeatures` * `nFeatures`.
/// @param[in,out] sums                 Array of column sums of size `nFeatures`. It is an output if
///                                     `computeSumsAndCenter` is true, an input if `useCurrentSums` is true,
///                                     and is left untouched otherwise.
/// @param[in]     computeSumsAndCenter Flag that specifies whether the column sums are to be computed
///                                     and the data is to be centered with the resulting means.
/// @param[in]     useCurrentSums       Flag that specifies whether the data is to be centered with the means
///                                     derived from the sums provided in `sums`.
template <typename algorithmFPType, CpuType cpu>
services::Status computeDenseCrossProductsAndSumsNonBatched(const size_t nFeatures, const size_t nVectors, NumericTable * dataTable,
                                                            algorithmFPType * crossProduct, algorithmFPType * sums, const bool computeSumsAndCenter,
                                                            const bool useCurrentSums)
{
    DAAL_PROFILER_TASK(Covariance::computeDenseCrossProductsAndSumsNonBatched);

    const bool center = computeSumsAndCenter || useCurrentSums;

    ReadRows<algorithmFPType, cpu, NumericTable> dataReader(dataTable, 0, nVectors);
    const algorithmFPType * dataPointer = dataReader.get();
    DAAL_CHECK_MALLOC(dataPointer);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures, nVectors);

    /// Copy of the input data with the column means subtracted. Allocated only if the centering is required.
    daal::services::internal::TArray<algorithmFPType, cpu> dataCentered(center ? (nFeatures * nVectors) : 0);
    daal::services::internal::TArray<algorithmFPType, cpu> meansArray(center ? nFeatures : 0);

    if (center)
    {
        algorithmFPType * dataCenteredPtr = dataCentered.get();
        algorithmFPType * means           = meansArray.get();
        DAAL_CHECK_MALLOC(dataCenteredPtr && means);

        if (computeSumsAndCenter)
        {
            StatisticsInst<algorithmFPType, cpu>::xmeansOnePass(dataPointer, nFeatures, nVectors, means);
        }
        else
        {
            /// The means are derived from the sums provided by the user
            const int result =
                daal::services::internal::daal_memcpy_s(means, nFeatures * sizeof(algorithmFPType), sums, nFeatures * sizeof(algorithmFPType));
            DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);

            const DAAL_INT one                = 1;
            const DAAL_INT nCols              = nFeatures;
            const algorithmFPType nVectors_fp = nVectors;
            LapackInst<algorithmFPType, cpu>::xxrscl(&nCols, &nVectors_fp, means, &one);
        }

        threader_for(nVectors, 0, [=](const int vector) {
            daal::internal::MathInst<algorithmFPType, cpu>::vSub(nFeatures, dataPointer + vector * nFeatures, means,
                                                                 dataCenteredPtr + vector * nFeatures);
        });

        if (computeSumsAndCenter)
        {
            /// `finalizeCovariance` and `mergeCrossProductAndSums` expect the sums, not the means
            const DAAL_INT one                = 1;
            const DAAL_INT nCols              = nFeatures;
            const algorithmFPType nVectors_fp = nVectors;
            const int result =
                daal::services::internal::daal_memcpy_s(sums, nFeatures * sizeof(algorithmFPType), means, nFeatures * sizeof(algorithmFPType));
            DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
            BlasInst<algorithmFPType, cpu>::xscal(&nCols, &nVectors_fp, sums, &one);
        }
    }

    const DAAL_INT nCols         = nFeatures;
    const DAAL_INT nRows         = nVectors;
    const algorithmFPType zero   = 0.0;
    const algorithmFPType one_fp = 1.0;
    BlasInst<algorithmFPType, cpu>::xsyrk("U", "N", &nCols, &nRows, &one_fp, center ? dataCentered.get() : dataPointer, &nCols, &zero, crossProduct,
                                          &nCols);

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status computeDenseCrossProductsAndSumsServiceStats(const size_t nFeatures, const size_t nVectors, NumericTable * dataTable,
                                                              algorithmFPType * crossProduct, algorithmFPType * sums, algorithmFPType * nObservations,
                                                              const DAAL_INT64 method)
{
    __int64 mklMethod = __DAAL_VSL_SS_METHOD_FAST;
    switch (method)
    {
    case defaultDense: mklMethod = __DAAL_VSL_SS_METHOD_FAST; break;
    case singlePassDense: mklMethod = __DAAL_VSL_SS_METHOD_1PASS; break;
    case sumDense: mklMethod = __DAAL_VSL_SS_METHOD_FAST_USER_MEAN; break;
    default: break;
    }

    DAAL_PROFILER_TASK_WITH_ARGS(Covariance::computeDenseCrossProductsAndSumsServiceStats, mklMethod);

    DEFINE_TABLE_BLOCK(ReadRows, dataBlock, dataTable);
    algorithmFPType * dataBlockPtr = const_cast<algorithmFPType *>(dataBlock.get());

    int errcode =
        StatisticsInst<algorithmFPType, cpu>::xcp(dataBlockPtr, (__int64)nFeatures, (__int64)nVectors, nObservations, sums, crossProduct, mklMethod);
    if (errcode)
    {
        return services::Status(services::ErrorCovarianceInternal);
    }
    else
    {
        return services::Status();
    }
}

/********************* updateDenseCrossProductAndSums ********************************************/
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status updateDenseCrossProductAndSums(bool isNormalized, size_t nFeatures, size_t nVectors, NumericTable * dataTable,
                                                algorithmFPType * crossProduct, algorithmFPType * sums, algorithmFPType * nObservations,
                                                const Parameter * parameter, const Hyperparameter * hyperparameter)
{
    DAAL_INT64 numRowsInBlock     = getBlockSize<cpu>(nVectors); // number of rows in a data block
    DAAL_INT64 grainSize          = 1;                           // minimal number of data blocks to be processed by a thread
    DAAL_INT64 maxColsBatched     = 4096;
    DAAL_INT64 smallRowsThreshold = 10'000; DAAL_INT64 smallRowsMaxColsBatched = 1024;

        if (hyperparameter)
    {
        services::Status status = hyperparameter->find(denseUpdateStepBlockSize, numRowsInBlock);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK(numRowsInBlock > 0ll, services::ErrorHyperparameterBadValue);
        status = hyperparameter->find(denseUpdateStepGrainSize, grainSize);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK(grainSize > 0ll, services::ErrorHyperparameterBadValue);
        status = hyperparameter->find(denseUpdateMaxColsBatched, maxColsBatched);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK(maxColsBatched >= 0ll, services::ErrorHyperparameterBadValue);
        status = hyperparameter->find(denseUpdateSmallRowsThreshold, smallRowsThreshold);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK(smallRowsThreshold >= 0ll, services::ErrorHyperparameterBadValue);
        status = hyperparameter->find(denseUpdateSmallRowsMaxColsBatched, smallRowsMaxColsBatched);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK(smallRowsMaxColsBatched >= 0ll, services::ErrorHyperparameterBadValue);
    }
    bool assumeCentered = parameter->assumeCentered;
    DAAL_PROFILER_TASK_WITH_ARGS(Covariance::updateDenseCrossProductAndSums, numRowsInBlock, grainSize, maxColsBatched, smallRowsThreshold,
                                 smallRowsMaxColsBatched, assumeCentered, isNormalized);

    services::Status status;
    const bool prefer_non_batched = (nFeatures >= maxColsBatched) || (nVectors <= smallRowsThreshold && nFeatures >= smallRowsMaxColsBatched);

    if (prefer_non_batched)
    {
        if (method == defaultDense || method == sumDense || isNormalized || assumeCentered)
        {
            status = computeDenseCrossProductsAndSumsNonBatched<algorithmFPType, cpu>(nFeatures, nVectors, dataTable, crossProduct, sums,
                                                                                      method == defaultDense && !isNormalized && !assumeCentered,
                                                                                      method == sumDense && !isNormalized && !assumeCentered);
        }
        else
        {
            status = computeDenseCrossProductsAndSumsServiceStats<algorithmFPType, cpu>(nFeatures, nVectors, dataTable, crossProduct, sums,
                                                                                        nObservations, method);
        }
    }
    else
    {
        if (((isNormalized) || ((!isNormalized) && ((method == defaultDense) || (method == sumDense)))))
        {
            status = computeDenseCrossProductsAndSumsBatched<algorithmFPType, cpu>(nFeatures, nVectors, dataTable, crossProduct, sums, isNormalized,
                                                                                   assumeCentered, numRowsInBlock, grainSize);
        }
        else
        {
            status = computeDenseCrossProductsAndSumsServiceStats<algorithmFPType, cpu>(nFeatures, nVectors, dataTable, crossProduct, sums,
                                                                                        nObservations, method);
        }
    }

    if (status != services::Status()) return status;

    *nObservations += (algorithmFPType)nVectors;
    return status;
}

/********************** updateCSRCrossProductAndSums *********************************************/
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status updateCSRCrossProductAndSums(size_t nFeatures, size_t nVectors, algorithmFPType * dataBlock, size_t * colIndices,
                                              size_t * rowOffsets, algorithmFPType * crossProduct, algorithmFPType * sums,
                                              algorithmFPType * nObservations, const Hyperparameter * hyperparameter)
{
    char transa = 'T';
    SpBlasInst<algorithmFPType, cpu>::xcsrmultd(&transa, (DAAL_INT *)&nVectors, (DAAL_INT *)&nFeatures, (DAAL_INT *)&nFeatures, dataBlock,
                                                (DAAL_INT *)colIndices, (DAAL_INT *)rowOffsets, dataBlock, (DAAL_INT *)colIndices,
                                                (DAAL_INT *)rowOffsets, crossProduct, (DAAL_INT *)&nFeatures);

    if (method != sumCSR)
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors, sizeof(algorithmFPType));

        TArray<algorithmFPType, cpu> onesArray(nVectors);
        DAAL_CHECK_MALLOC(onesArray.get());

        algorithmFPType one    = 1.0;
        algorithmFPType * ones = onesArray.get();
        daal::services::internal::service_memset<algorithmFPType, cpu>(ones, one, nVectors);

        char matdescra[6];
        matdescra[0] = 'G'; // general matrix
        matdescra[3] = 'F'; // 1-based indexing

        matdescra[1] = (char)0;
        matdescra[2] = (char)0;
        matdescra[4] = (char)0;
        matdescra[5] = (char)0;
        SpBlasInst<algorithmFPType, cpu>::xcsrmv(&transa, (DAAL_INT *)&nVectors, (DAAL_INT *)&nFeatures, &one, matdescra, dataBlock,
                                                 (DAAL_INT *)colIndices, (DAAL_INT *)rowOffsets, (DAAL_INT *)rowOffsets + 1, ones, &one, sums);
    }

    nObservations[0] += (algorithmFPType)nVectors;
    return services::Status();
}

/*********************** mergeCrossProductAndSums ************************************************/
template <typename algorithmFPType, CpuType cpu>
void mergeCrossProductAndSums(size_t nFeatures, const algorithmFPType * partialCrossProduct, const algorithmFPType * partialSums,
                              const algorithmFPType * partialNObservations, algorithmFPType * crossProduct, algorithmFPType * sums,
                              algorithmFPType * nObservations, const Hyperparameter * hyperparameter)
{
    DAAL_PROFILER_TASK(Covariance::mergeCrossProductAndSums);
    /* Merge cross-products */
    algorithmFPType partialNObsValue = partialNObservations[0];

    if (partialNObsValue != 0)
    {
        algorithmFPType nObsValue = nObservations[0];

        if (nObsValue == 0)
        {
            daal::threader_for(nFeatures, nFeatures, [=](size_t i) {
                PRAGMA_OMP_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    crossProduct[i * nFeatures + j] += partialCrossProduct[i * nFeatures + j];
                    crossProduct[j * nFeatures + i] = crossProduct[i * nFeatures + j];
                }
            });
        }
        else
        {
            algorithmFPType invPartialNObs = 1.0 / partialNObsValue;
            algorithmFPType invNObs        = 1.0 / nObsValue;
            algorithmFPType invNewNObs     = 1.0 / (nObsValue + partialNObsValue);

            daal::threader_for(nFeatures, nFeatures, [=](size_t i) {
                PRAGMA_OMP_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    crossProduct[i * nFeatures + j] += partialCrossProduct[i * nFeatures + j];
                    crossProduct[i * nFeatures + j] += partialSums[i] * partialSums[j] * invPartialNObs;
                    crossProduct[i * nFeatures + j] += sums[i] * sums[j] * invNObs;
                    crossProduct[i * nFeatures + j] -= (partialSums[i] + sums[i]) * (partialSums[j] + sums[j]) * invNewNObs;
                    crossProduct[j * nFeatures + i] = crossProduct[i * nFeatures + j];
                }
            });
        }

        /* Merge number of observations */
        nObservations[0] += partialNObservations[0];

        /* Merge sums */
        daal::internal::MathInst<algorithmFPType, cpu>::vAdd(nFeatures, sums, partialSums, sums);
    }
}

/*********************** finalizeCovariance ******************************************************/
template <typename algorithmFPType, CpuType cpu>
services::Status finalizeCovariance(size_t nFeatures, algorithmFPType nObservations, algorithmFPType * crossProduct, algorithmFPType * sums,
                                    algorithmFPType * cov, algorithmFPType * mean, const Parameter * parameter, const Hyperparameter * hyperparameter)
{
    DAAL_PROFILER_TASK(compute.finalizeCovariance);

    algorithmFPType invNObservations   = 1.0 / nObservations;
    algorithmFPType invNObservationsM1 = 1.0;
    if (nObservations > 1.0)
    {
        invNObservationsM1 = 1.0 / (nObservations - 1.0);
    }

    algorithmFPType multiplier = invNObservationsM1;
    if (parameter->bias)
    {
        multiplier = invNObservations;
    }

    /* Calculate resulting mean vector */
    PRAGMA_OMP_SIMD
    for (size_t i = 0; i < nFeatures; i++)
    {
        mean[i] = sums[i] * invNObservations;
    }

    if (parameter->outputMatrixType == covariance::correlationMatrix)
    {
        /* Calculate resulting correlation matrix */
        TArray<algorithmFPType, cpu> diagInvSqrtsArray(nFeatures);
        DAAL_CHECK_MALLOC(diagInvSqrtsArray.get());

        algorithmFPType * diagInvSqrts = diagInvSqrtsArray.get();
        daal::internal::MathInst<algorithmFPType, cpu>::vInvSqrtI(nFeatures, crossProduct, nFeatures + 1, diagInvSqrts, 1);

        for (size_t i = 0; i < nFeatures; i++)
        {
            PRAGMA_OMP_SIMD
            for (size_t j = 0; j < i; j++)
            {
                cov[i * nFeatures + j] = crossProduct[i * nFeatures + j] * diagInvSqrts[i] * diagInvSqrts[j];
            }
            cov[i * nFeatures + i] = 1.0; //diagonal element
        }
    }
    else
    {
        /* Calculate resulting covariance matrix */
        for (size_t i = 0; i < nFeatures; i++)
        {
            PRAGMA_OMP_SIMD
            for (size_t j = 0; j <= i; j++)
            {
                cov[i * nFeatures + j] = crossProduct[i * nFeatures + j] * multiplier;
            }
        }
    }

    /* Copy results into symmetric upper triangle */
    for (size_t i = 0; i < nFeatures; i++)
    {
        PRAGMA_OMP_SIMD
        for (size_t j = 0; j < i; j++)
        {
            cov[j * nFeatures + i] = cov[i * nFeatures + j];
        }
    }

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status finalizeCovariance(NumericTable * nObservationsTable, NumericTable * crossProductTable, NumericTable * sumTable,
                                    NumericTable * covTable, NumericTable * meanTable, const Parameter * parameter,
                                    const Hyperparameter * hyperparameter)
{
    const size_t nFeatures = covTable->getNumberOfColumns();

    DEFINE_TABLE_BLOCK(ReadRows, sumBlock, sumTable);
    DEFINE_TABLE_BLOCK(ReadRows, crossProductBlock, crossProductTable);
    DEFINE_TABLE_BLOCK(ReadRows, nObservationsBlock, nObservationsTable);
    DEFINE_TABLE_BLOCK(WriteOnlyRows, covBlock, covTable);
    DEFINE_TABLE_BLOCK(WriteOnlyRows, meanBlock, meanTable);

    algorithmFPType * cov           = covBlock.get();
    algorithmFPType * mean          = meanBlock.get();
    algorithmFPType * sums          = const_cast<algorithmFPType *>(sumBlock.get());
    algorithmFPType * crossProduct  = const_cast<algorithmFPType *>(crossProductBlock.get());
    algorithmFPType * nObservations = const_cast<algorithmFPType *>(nObservationsBlock.get());

    return finalizeCovariance<algorithmFPType, cpu>(nFeatures, *nObservations, crossProduct, sums, cov, mean, parameter, hyperparameter);
}

} // namespace internal
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
