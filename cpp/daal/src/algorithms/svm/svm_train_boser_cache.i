/* file: svm_train_boser_cache.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  SVM cache structure implementation
//--
*/

#ifndef __SVM_TRAIN_BOSER_CACHE_I__
#define __SVM_TRAIN_BOSER_CACHE_I__

#include "src/services/service_utils.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_profiler.h"
#include "src/algorithms/svm/svm_train_cache.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

/**
 * Common interface for cache that stores kernel function values
 */
template <typename algorithmFPType, CpuType cpu>
class SVMCacheIface<boser, algorithmFPType, cpu> : public SVMCacheCommonIface<algorithmFPType, cpu>
{
public:
    virtual ~SVMCacheIface() {}

    /**
     * Get block of values from the row of the matrix Q (kernel(x[i], x[j]))
     * \param[in] rowIndex      Index of the requested row
     * \param[in] startColIndex Starting columns index of the requested block of values
     * \param[in] blockSize     Number of requested values
     * \param[out] block        Pointer to the block of values
     * \return status of the call
     */
    virtual Status getRowBlock(size_t rowIndex, size_t startColIndex, size_t blockSize, const algorithmFPType *& block) = 0;

    /**
     * Get blocks of values from the two rows of the matrix Q (kernel(x[i], x[j]))
     * \param[in] rowIndex1     Index of the first requested row
     * \param[in] rowIndex2     Index of the second requested row
     * \param[in] startColIndex Starting columns index of the requested blocks of values
     * \param[in] blockSize     Number of requested values in each block
     * \param[out] block1       Pointer to the first  block of values
     * \param[out] block2       Pointer to the second block of values
     * \return status of the call
     */
    virtual Status getTwoRowsBlock(size_t rowIndex1, size_t rowIndex2, size_t startColIndex, size_t blockSize, const algorithmFPType *& block1,
                                   const algorithmFPType *& block2) = 0;

    /**
     * Move the indices of the shrunk feature vector to the end of the array
     *
     * \param[in] nActiveVectors Number of observations in a training data set that are used
     *                           in sequential minimum optimization at the current iteration
     * \param[in] I              Array of flags that describe the status of feature vectors
     * \return status of the call
     */
    virtual Status updateShrinkingRowIndices(size_t nActiveVectors, const char * I) = 0;
};

/**
 * Common implementation for cache that stores kernel function values
 */
template <typename algorithmFPType, CpuType cpu>
class SVMCacheImpl : public SVMCacheIface<boser, algorithmFPType, cpu>
{
public:
    virtual ~SVMCacheImpl() {}

    virtual size_t getDataRowIndex(size_t rowIndex) const DAAL_C11_OVERRIDE { return _doShrinking ? _shrinkingRowIndices[rowIndex] : rowIndex; }

protected:
    /**
     * Constructs cache
     *
     * \param[in] lineSize      Number of elements in the cache line
     * \param[in] doShrinking   Flag that enables use of the shrinking optimization technique
     * \param[in] kernel        Kernel function
     */
    SVMCacheImpl(size_t lineSize, bool doShrinking, const kernel_function::KernelIfacePtr & kernel)
        : _lineSize(lineSize), _doShrinking(doShrinking), _kernel(kernel)
    {}

    Status init()
    {
        if (_doShrinking)
        {
            _shrinkingRowIndices.reset(_lineSize);
            DAAL_CHECK_MALLOC(_shrinkingRowIndices.get());
            for (size_t i = 0; i < _lineSize; i++) _shrinkingRowIndices[i] = i;
        }
        return Status();
    }

protected:
    TArray<algorithmFPType, cpu> _cache;
    const size_t _lineSize;                        /*!< Number of elements in the cache line */
    const kernel_function::KernelIfacePtr _kernel; /*!< Kernel function */
    const bool _doShrinking;                       /*!< Flag that enables use of the shrinking optimization technique */
    TArray<size_t, cpu> _shrinkingRowIndices;      /*!< Array of input data row indices used with shrinking technique */
};

/**
 * Simple cache: all elements of kernel matrix fit into cache
 */
template <typename algorithmFPType, CpuType cpu>
class SVMCache<boser, simpleCache, algorithmFPType, cpu> : public SVMCacheImpl<algorithmFPType, cpu>
{
    typedef SVMCacheImpl<algorithmFPType, cpu> super;
    typedef SVMCache<boser, simpleCache, algorithmFPType, cpu> this_type;

    using super::_cache;
    using super::_kernel;
    using super::_lineSize;
    using super::_shrinkingRowIndices;
    using super::_doShrinking;

public:
    DAAL_NEW_DELETE();
    /**
     * Constructs simple cache
     *
     * \param[in] lineSize      Number of elements in the cache line
     * \param[in] doShrinking   Flag that enables use of the shrinking optimization technique
     * \param[in] xTable        Input data set
     * \param[in] kernel        Kernel function
     */
    static SVMCache * create(size_t lineSize, bool doShrinking, const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel,
                             Status & s)
    {
        s.clear();
        this_type * res = new this_type(lineSize, doShrinking, xTable, kernel);
        if (!res)
            s.add(ErrorMemoryAllocationFailed);
        else
        {
            s = res->init(xTable);
            if (!s)
            {
                delete res;
                res = nullptr;
            }
        }
        return res;
    }

    virtual Status getRowBlock(size_t rowIndex, size_t startColIndex, size_t blockSize, const algorithmFPType *& block) DAAL_C11_OVERRIDE
    {
        block = _cache.get() + rowIndex * _lineSize + startColIndex;
        return Status();
    }

    virtual Status getTwoRowsBlock(size_t rowIndex1, size_t rowIndex2, size_t startColIndex, size_t blockSize, const algorithmFPType *& block1,
                                   const algorithmFPType *& block2) DAAL_C11_OVERRIDE
    {
        block1 = _cache.get() + rowIndex1 * _lineSize + startColIndex;
        block2 = _cache.get() + rowIndex2 * _lineSize + startColIndex;
        return Status();
    }

    virtual Status updateShrinkingRowIndices(size_t nActiveVectors, const char * I) DAAL_C11_OVERRIDE;

    ~SVMCache() {}

protected:
    Status init(const NumericTablePtr & xTable)
    {
        DAAL_PROFILER_TASK(CACHE.computeKernel);

        Status s = super::init();
        if (!s) return s;
        _cache.reset(_lineSize * _nLines);
        DAAL_CHECK_MALLOC(_cache.get());
        if (_doShrinking)
        {
            _tmp.reset(_lineSize);
            DAAL_CHECK_MALLOC(_tmp.get());
        }

        _cacheTable = HomogenNumericTableCPU<algorithmFPType, cpu>::create(_cache.get(), _lineSize, _nLines, &s);
        DAAL_CHECK_STATUS_VAR(s);

        _kernel->getParameter()->computationMode = kernel_function::matrixMatrix;
        _kernel->getInput()->set(kernel_function::X, xTable);
        _kernel->getInput()->set(kernel_function::Y, xTable);

        auto kfResultPtr = new kernel_function::Result();
        DAAL_CHECK_MALLOC(kfResultPtr)
        kernel_function::ResultPtr shRes(kfResultPtr);
        shRes->set(kernel_function::values, _cacheTable);
        _kernel->setResult(shRes);
        return _kernel->computeNoThrow();
    }

    /**
    * \param[in] lineSize      Number of elements in the cache line
    * \param[in] doShrinking   Flag that enables use of the shrinking optimization technique
    * \param[in] xTable        Input data set
    * \param[in] kernel        Kernel function
    */
    SVMCache(size_t lineSize, bool doShrinking, const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel)
        : super(lineSize, doShrinking, kernel), _nLines(lineSize)
    {}

protected:
    size_t _nLines;
    services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > _cacheTable;
    TArray<algorithmFPType, cpu> _tmp;
};

/**
 * No cache: kernel function values are not cached
 */
template <typename algorithmFPType, CpuType cpu>
class SVMCache<boser, noCache, algorithmFPType, cpu> : public SVMCacheImpl<algorithmFPType, cpu>
{
    typedef SVMCacheImpl<algorithmFPType, cpu> super;
    typedef SVMCache<boser, noCache, algorithmFPType, cpu> this_type;
    using super::_cache;
    using super::_kernel;
    using super::_lineSize;
    using super::_shrinkingRowIndices;
    using super::_doShrinking;

public:
    DAAL_NEW_DELETE();
    /**
    * Constructs simple cache
     *
     * \param[in] lineSize      Number of elements in the cache line
     * \param[in] doShrinking   Flag that enables use of the shrinking optimization technique
     * \param[in] xTable        Input data set
     * \param[in] kernel        Kernel function
     */
    static SVMCache * create(size_t cacheSize, size_t lineSize, bool doShrinking, const NumericTablePtr & xTable,
                             const kernel_function::KernelIfacePtr & kernel, Status & s)
    {
        s.clear();
        this_type * res = new this_type(lineSize, doShrinking, xTable, kernel);
        if (!res)
            s.add(ErrorMemoryAllocationFailed);
        else
        {
            s = res->init(cacheSize, xTable);
            if (!s)
            {
                delete res;
                res = nullptr;
            }
        }
        return res;
    }

    virtual Status getRowBlock(size_t rowIndex, size_t startColIndex, size_t blockSize, const algorithmFPType *& block) DAAL_C11_OVERRIDE
    {
        return getRowBlockImpl(rowIndex, startColIndex, blockSize, 0, block);
    }

    virtual Status getTwoRowsBlock(size_t rowIndex1, size_t rowIndex2, size_t startColIndex, size_t blockSize, const algorithmFPType *& block1,
                                   const algorithmFPType *& block2) DAAL_C11_OVERRIDE
    {
        Status s = getRowBlockImpl(rowIndex1, startColIndex, blockSize, 0, block1);
        s |= getRowBlockImpl(rowIndex2, startColIndex, blockSize, blockSize, block2);
        return s;
    }

    virtual Status updateShrinkingRowIndices(size_t nActiveVectors, const char * I) DAAL_C11_OVERRIDE;

    ~SVMCache() {}

protected:
    /**
     * Constructs the cache that doesn't cache kernel function values
     *
     * \param[in] lineSize      Number of elements in the cache line
     * \param[in] doShrinking   Flag that enables use of the shrinking optimization technique
     * \param[in] xTable        Input data set
     * \param[in] kernel        Kernel function
     */
    SVMCache(size_t lineSize, bool doShrinking, const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel)
        : super(lineSize, doShrinking, kernel)
    {}

    Status init(size_t cacheSize, const NumericTablePtr & xTable)
    {
        Status s = super::init();
        if (!s) return s;
        _cache.reset(cacheSize);
        DAAL_CHECK_MALLOC(_cache.get());
        _cacheTable = HomogenNumericTableCPU<algorithmFPType, cpu>::create(NULL, 1, _lineSize, &s);
        DAAL_CHECK_STATUS_VAR(s);
        _kernel->getParameter()->computationMode = kernel_function::vectorVector;
        _kernel->getInput()->set(kernel_function::X, xTable);
        _kernel->getInput()->set(kernel_function::Y, xTable);

        auto kfResultPtr = new kernel_function::Result();
        DAAL_CHECK_MALLOC(kfResultPtr)
        kernel_function::ResultPtr shRes(kfResultPtr);
        shRes->set(kernel_function::values, _cacheTable);
        _kernel->setResult(shRes);
        return s;
    }

    Status getRowBlockImpl(size_t rowIndex, size_t startColIndex, size_t blockSize, size_t cacheOffset, const algorithmFPType *& block)
    {
        DAAL_PROFILER_TASK(NOT_CACHE.computeKernel);

        _cacheTable->setArray(_cache.get() + cacheOffset, _cacheTable->getNumberOfRows());
        _kernel->getParameter()->rowIndexY = _doShrinking ? _shrinkingRowIndices[rowIndex] : rowIndex;
        Status s;
        for (size_t i = 0; i < blockSize; i++)
        {
            _kernel->getParameter()->rowIndexX      = _doShrinking ? _shrinkingRowIndices[startColIndex + i] : startColIndex + i;
            _kernel->getParameter()->rowIndexResult = i;
            s.add(_kernel->computeNoThrow());
        }
        block = _cache.get() + cacheOffset;
        return s;
    }

protected:
    services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > _cacheTable;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
