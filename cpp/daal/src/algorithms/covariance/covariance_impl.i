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

/// Implements daal::Reducer interface for the dense Covariance algorithm computations.
///
/// @tparam algorithmFPType     Data type to store partial results: double or float.
/// @tparam cpu                 Variant of the CPU instruction set: SSE2, SSE4.2, AVX2, AVX512, ARM SVE, etc.
template <typename algorithmFPType, CpuType cpu>
class CovarianceReducer : public daal::Reducer
{
public:
    enum ErrorCode
    {
        ok                  = 0, /// No error
        memAllocationFailed = 1, /// Memory allocation failed
        intOverflow         = 2, /// Integer overflow
        badCast             = 3  /// Cannot cast base daal::Reducer to derived class
    };
    /// Status of the computation.
    ErrorCode errorCode;

    /// Get pointer to the array of partial sums.
    inline algorithmFPType * sums() { return _sumsArray.get(); }
    /// Get pointer to the partial cross-product matrix.
    inline algorithmFPType * crossProduct() { return _crossProductArray.get(); }

    /// Get pointer to the constant array of partial sums.
    inline const algorithmFPType * sums() const { return _sumsArray.get(); }
    /// Get pointer to the constant partial cross-product matrix.
    inline const algorithmFPType * crossProduct() const { return _crossProductArray.get(); }

    /// Construct and initialize the thread-local partial results
    ///
    /// @param[in] dataTable        Input data table that stores matrix X for which the cross-product matrix and sums are computed
    /// @param[in] numRowsInBlock   Number of rows in the block of the input data table - a mininal number of rows to be processed by a thread
    /// @param[in] numBlocks        Number of blocks of rows in the input data table
    /// @param[in] isNormalized     Flag that specifies whether the input data is normalized
    CovarianceReducer(NumericTable * dataTable, DAAL_INT numRowsInBlock, size_t numBlocks, bool isNormalized)
        : _dataTable(dataTable),
          _numRowsInBlock(numRowsInBlock),
          _numBlocks(numBlocks),
          _nFeatures(dataTable->getNumberOfColumns()),
          _isNormalized(isNormalized)
    {
        bool isOverflow = false;
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_BOOL(size_t, _nFeatures, _nFeatures, isOverflow);
        if (isOverflow)
        {
            errorCode = ErrorCode::intOverflow;
            return;
        }
        if (!_crossProductArray.reset(_nFeatures * _nFeatures))
        {
            errorCode = ErrorCode::memAllocationFailed;
            return;
        }

        if (!_isNormalized)
        {
            if (!_sumsArray.reset(_nFeatures))
            {
                errorCode = ErrorCode::memAllocationFailed;
                return;
            }
        }
        errorCode = ErrorCode::ok;
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
        return daal::internal::makeUnique<CovarianceReducer<algorithmFPType, cpu>, DAAL_BASE_CPU>(_dataTable, _numRowsInBlock, _numBlocks,
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
        if (errorCode != ErrorCode::ok)
        {
            return;
        }
        algorithmFPType * crossProductPtr = crossProduct();

        if (!crossProductPtr)
        {
            errorCode = ErrorCode::memAllocationFailed;
            return;
        }

        bool isOverflow = false;
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_BOOL(size_t, _numRowsInBlock, _nFeatures, isOverflow);
        if (isOverflow)
        {
            errorCode = ErrorCode::intOverflow;
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
            if (!dataTableBD.get())
            {
                errorCode = ErrorCode::memAllocationFailed;
                return;
            }
            algorithmFPType * dataBlock = const_cast<algorithmFPType *>(dataTableBD.get());

            /// Update the cross-product matrix with the data from the block
            {
                DAAL_PROFILER_THREADING_TASK(reducer.update.syrkData);
                BlasInst<algorithmFPType, cpu>::xsyrk("U", "N", &_nFeatures, reinterpret_cast<DAAL_INT *>(&nRows), &one, dataBlock, &_nFeatures, &one,
                                                      crossProductPtr, &_nFeatures);
            }

            if (!_isNormalized)
            {
                DAAL_PROFILER_THREADING_TASK(reducer.update.sums);
                algorithmFPType * sumsPtr = sums();
                if (!sumsPtr)
                {
                    errorCode = ErrorCode::memAllocationFailed;
                    return;
                }
                /* Sum input array elements in case of non-normalized data */
                for (DAAL_INT i = 0; i < nRows; i++)
                {
                    PRAGMA_FORCE_SIMD
                    PRAGMA_VECTOR_ALWAYS
                    for (DAAL_INT j = 0; j < _nFeatures; j++)
                    {
                        sumsPtr[j] += dataBlock[i * _nFeatures + j];
                    }
                }
            }
        }
    }

    /// Merge the partial result with the data from another thread.
    ///
    /// @param otherReducer Pointer to the other thread's partial result.
    virtual void join(Reducer * otherReducer) override
    {
        if (errorCode != ErrorCode::ok)
        {
            return;
        }
        DAAL_PROFILER_THREADING_TASK(reducer.join);
        CovarianceReducer<algorithmFPType, cpu> * other = dynamic_cast<CovarianceReducer<algorithmFPType, cpu> *>(otherReducer);
        if (!other)
        {
            errorCode = ErrorCode::badCast;
            return;
        }
        if (other->errorCode != ErrorCode::ok)
        {
            errorCode = other->errorCode;
            return;
        }
        const algorithmFPType * otherCrossProduct = other->crossProduct();
        algorithmFPType * thisCrossProduct        = crossProduct();

        if (!thisCrossProduct || !otherCrossProduct)
        {
            errorCode = ErrorCode::memAllocationFailed;
            return;
        }

        /// It is safe to use aligned loads and stores because the data in TArrayScalableCalloc data structures is aligned
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        PRAGMA_VECTOR_ALIGNED
        for (size_t i = 0; i < (_nFeatures * _nFeatures); i++)
        {
            thisCrossProduct[i] += otherCrossProduct[i];
        }
        if (!_isNormalized)
        {
            algorithmFPType * thisSums        = sums();
            const algorithmFPType * otherSums = other->sums();
            if (!thisSums || !otherSums)
            {
                errorCode = ErrorCode::memAllocationFailed;
                return;
            }
            /// It is safe to use aligned loads and stores because the data is aligned
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            PRAGMA_VECTOR_ALIGNED
            for (size_t i = 0; i < _nFeatures; i++)
            {
                thisSums[i] += otherSums[i];
            }
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

/********************* updateDenseCrossProductAndSums ********************************************/
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status updateDenseCrossProductAndSums(bool isNormalized, size_t nFeatures, size_t nVectors, NumericTable * dataTable,
                                                algorithmFPType * crossProduct, algorithmFPType * sums, algorithmFPType * nObservations,
                                                const Parameter * parameter, const Hyperparameter * hyperparameter)
{
    DAAL_INT64 numRowsInBlock = getBlockSize<cpu>(nVectors); // number of rows in a data block
    DAAL_INT64 grainSize      = 1;                           // minimal number of data blocks to be processed by a thread
    if (hyperparameter)
    {
        services::Status status = hyperparameter->find(denseUpdateStepBlockSize, numRowsInBlock);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK(numRowsInBlock > 0ll, services::ErrorHyperparameterBadValue);
        status = hyperparameter->find(denseUpdateStepGrainSize, grainSize);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK(grainSize > 0ll, services::ErrorHyperparameterBadValue);
    }
    DAAL_PROFILER_TASK_WITH_ARGS(Covariance::updateDenseCrossProductAndSums, numRowsInBlock, grainSize);
    bool assumeCentered = parameter->assumeCentered;
    if (((isNormalized) || ((!isNormalized) && ((method == defaultDense) || (method == sumDense)))))
    {
        /* Inverse number of rows (for normalization) */
        const algorithmFPType nVectorsInv = 1.0 / (double)(nVectors);

        /* Split rows by blocks */

        size_t numBlocks = nVectors / numRowsInBlock;
        if (numBlocks * numRowsInBlock < nVectors)
        {
            numBlocks++;
        }

        CovarianceReducer<algorithmFPType, cpu> result(dataTable, numRowsInBlock, numBlocks, isNormalized);
        if (!result.crossProduct() || !result.sums())
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }

        /* Reduce input matrix X into cross product Xt X and a vector of column sums */
        daal::static_threader_reduce(numBlocks, grainSize, result);
        if (result.errorCode != CovarianceReducer<algorithmFPType, cpu>::ok)
        {
            if (result.errorCode == CovarianceReducer<algorithmFPType, cpu>::memAllocationFailed)
            {
                return services::Status(services::ErrorMemoryAllocationFailed);
            }
            if (result.errorCode == CovarianceReducer<algorithmFPType, cpu>::intOverflow)
            {
                return services::Status(services::ErrorBufferSizeIntegerOverflow);
            }
            if (result.errorCode == CovarianceReducer<algorithmFPType, cpu>::badCast)
            {
                return services::Status(services::ErrorCovarianceInternal);
            }
        }

        const algorithmFPType * resultCrossProduct = result.crossProduct();
        if (result.errorCode != CovarianceReducer<algorithmFPType, cpu>::ok || !resultCrossProduct)
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }

        /* If data is not normalized, perform subtractions of(sums[i]*sums[j])/n */
        if (!isNormalized && !assumeCentered)
        {
            const algorithmFPType * resultSums = result.sums();
            if (!resultSums)
            {
                return services::Status(services::ErrorMemoryAllocationFailed);
            }
            for (size_t i = 0; i < nFeatures; i++)
            {
                sums[i] = resultSums[i];
            }
            for (size_t i = 0; i < nFeatures; i++)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < nFeatures; j++)
                {
                    crossProduct[i * nFeatures + j] = resultCrossProduct[i * nFeatures + j] - (nVectorsInv * sums[i] * sums[j]);
                }
            }
        }
        else
        {
            for (size_t i = 0; i < nFeatures * nFeatures; i++)
            {
                crossProduct[i] = resultCrossProduct[i];
            }
        }
    }
    else
    {
        __int64 mklMethod = __DAAL_VSL_SS_METHOD_FAST;
        switch (method)
        {
        case defaultDense: mklMethod = __DAAL_VSL_SS_METHOD_FAST; break;
        case singlePassDense: mklMethod = __DAAL_VSL_SS_METHOD_1PASS; break;
        case sumDense: mklMethod = __DAAL_VSL_SS_METHOD_FAST_USER_MEAN; break;
        default: break;
        }

        DEFINE_TABLE_BLOCK(ReadRows, dataBlock, dataTable);
        algorithmFPType * dataBlockPtr = const_cast<algorithmFPType *>(dataBlock.get());

        int errcode = StatisticsInst<algorithmFPType, cpu>::xcp(dataBlockPtr, (__int64)nFeatures, (__int64)nVectors, nObservations, sums,
                                                                crossProduct, mklMethod);
        DAAL_CHECK(errcode == 0, services::ErrorCovarianceInternal);
    }

    *nObservations += (algorithmFPType)nVectors;
    return services::Status();
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
                PRAGMA_FORCE_SIMD
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
                PRAGMA_FORCE_SIMD
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
        for (size_t i = 0; i < nFeatures; i++)
        {
            sums[i] += partialSums[i];
        }
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
        for (size_t i = 0; i < nFeatures; i++)
        {
            diagInvSqrts[i] = 1.0 / daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(crossProduct[i * nFeatures + i]);
        }

        for (size_t i = 0; i < nFeatures; i++)
        {
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
            for (size_t j = 0; j <= i; j++)
            {
                cov[i * nFeatures + j] = crossProduct[i * nFeatures + j] * multiplier;
            }
        }
    }

    /* Copy results into symmetric upper triangle */
    for (size_t i = 0; i < nFeatures; i++)
    {
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
