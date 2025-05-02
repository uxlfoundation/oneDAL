/* file: svm_train_thunder_cache.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef __SVM_TRAIN_THUNDER_CACHE_H__
#define __SVM_TRAIN_THUNDER_CACHE_H__

#include "src/services/service_utils.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/svm/svm_train_cache.h"
#include "src/externals/service_service.h"
#include "data_management/data/soa_numeric_table.h"

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

/**
 * Common interface for cache that stores kernel function values
 */
template <typename algorithmFPType, CpuType cpu>
class SVMCacheIface<thunder, algorithmFPType, cpu> : public SVMCacheCommonIface<algorithmFPType, cpu>
{
public:
    virtual ~SVMCacheIface() {}

    virtual services::Status getRowsBlock(const uint32_t * const indices, const size_t n, algorithmFPType **& soablock) = 0;

    virtual size_t getDataRowIndex(size_t rowIndex) const override { return rowIndex; }

    virtual services::Status clear() = 0;

    virtual services::Status resize(const size_t nSize) = 0;

protected:
    SVMCacheIface(const size_t cacheSize, const size_t lineSize, const kernel_function::KernelIfacePtr & kernel)
        : _lineSize(lineSize), _cacheSize(cacheSize), _kernel(kernel)
    {}

    const size_t _lineSize;                        /*!< Number of elements in the cache line */
    const size_t _cacheSize;                       /*!< Number of cache lines */
    const kernel_function::KernelIfacePtr _kernel; /*!< Kernel function */
};

/**
 * LRU cache: kernel function values are cached
 */
template <typename algorithmFPType, CpuType cpu>
class SVMCache<thunder, lruCache, algorithmFPType, cpu> : public SVMCacheIface<thunder, algorithmFPType, cpu>
{
    using super    = SVMCacheIface<thunder, algorithmFPType, cpu>;
    using thisType = SVMCache<thunder, lruCache, algorithmFPType, cpu>;
    using super::_kernel;
    using super::_lineSize;
    using super::_cacheSize;

public:
    ~SVMCache() {}

    DAAL_NEW_DELETE();

    static SVMCachePtr<thunder, algorithmFPType, cpu> create(const size_t cacheSize, const size_t nSize, const size_t lineSize,
                                                             const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel,
                                                             services::Status & status)
    {
        services::SharedPtr<thisType> res = services::SharedPtr<thisType>(new thisType(cacheSize, lineSize, xTable, kernel));
        if (!res)
        {
            status.add(ErrorMemoryAllocationFailed);
        }
        else
        {
            status = res->init(nSize);
            if (!status)
            {
                res.reset();
            }
        }
        return SVMCachePtr<thunder, algorithmFPType, cpu>(res);
    }

    services::Status clear() override
    {
        _blockTask.reset();
        _kernelOriginalIndex.reset();
        _kernelIndex.reset();
        _cache.reset();
        _cacheData.reset();
        _soaData.reset();
        return services::Status();
    }

    services::Status getRowsBlock(const uint32_t * const indices, const size_t n, algorithmFPType **& soablock) override
    {
        services::Status status;

        const size_t nVectors = _xTable->getNumberOfRows();
        if (_soaData.size() < n)
        {
            _soaData.reset(n);
            DAAL_CHECK_MALLOC(_soaData.get());
        }

        size_t nIndicesForKernel = 0;

        for (int i = 0; i < n; ++i)
        {
            const uint32_t dataIndex = indices[i] % nVectors;
            int64_t cacheIndex       = _lruCache.get(dataIndex);
            if (cacheIndex != -1)
            {
                // If index in cache
                DAAL_ASSERT(cacheIndex < _cacheSize)
                algorithmFPType * const cachei = _cache[cacheIndex];
                _soaData[i]                    = cachei;
            }
            else
            {
                _lruCache.put(dataIndex);
                cacheIndex = _lruCache.getFreeIndex();
                DAAL_ASSERT(cacheIndex < _cacheSize)
                algorithmFPType * const cachei          = _cache[cacheIndex];
                _soaData[i]                             = cachei;
                _kernelIndex[nIndicesForKernel]         = cacheIndex;
                _kernelOriginalIndex[nIndicesForKernel] = dataIndex;
                ++nIndicesForKernel;
            }
        }
        if (nIndicesForKernel != 0)
        {
            DAAL_CHECK_STATUS(status, computeKernel(nIndicesForKernel, _kernelOriginalIndex.get()));
        }

        soablock = _soaData.get();
        return status;
    }

    services::Status resize(const size_t nSize) override
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(cache.resize);

        services::Status status;
        status |= initKernelIndex(nSize);
        status |= initBlockTask(nSize);
        return status;
    }

protected:
    SVMCache(const size_t cacheSize, const size_t lineSize, const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel)
        : super(cacheSize, lineSize, kernel), _lruCache(cacheSize), _xTable(xTable)
    {}

    services::Status computeKernel(const size_t nWorkElements, const uint32_t * indices)
    {
        services::Status status;
        auto kernelComputeTable = SOANumericTableCPU<cpu>::create(nWorkElements, _lineSize, DictionaryIface::FeaturesEqual::equal, &status);
        DAAL_CHECK_STATUS_VAR(status);

        for (size_t i = 0; i < nWorkElements; ++i)
        {
            const size_t cacheIndex = _kernelIndex[i];
            auto cachei             = _cache[cacheIndex];
            DAAL_CHECK_STATUS(status, kernelComputeTable->template setArray<algorithmFPType>(cachei, i));
        }

        DAAL_CHECK_STATUS(status, _blockTask->copyDataByIndices(indices, nWorkElements, _xTable));

        DAAL_CHECK_STATUS_VAR(status);
        _kernel->getParameter()->computationMode = kernel_function::matrixMatrix;

        _kernel->getInput()->set(kernel_function::X, _xTable);
        _kernel->getInput()->set(kernel_function::Y, _blockTask->getTableData());

        kernel_function::ResultPtr shRes(new kernel_function::Result());
        shRes->set(kernel_function::values, kernelComputeTable);
        _kernel->setResult(shRes);
        DAAL_CHECK_STATUS(status, _kernel->computeNoThrow());

        return status;
    }

    services::Status initKernelIndex(const size_t nSize)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(cache.initKernelIndex);

        services::Status status;

        _kernelIndex.reset(nSize);
        DAAL_CHECK_MALLOC(_kernelIndex.get());
        _kernelOriginalIndex.reset(nSize);
        DAAL_CHECK_MALLOC(_kernelOriginalIndex.get());

        return status;
    }

    services::Status initCache()
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(cache.initCache);

        services::Status status;

        const size_t bytes            = _lineSize * sizeof(algorithmFPType);
        const size_t alignedBytesSize = bytes & 63 ? (bytes & (~63)) + 64 : bytes;  // nearest number aligned on 64
        const size_t newLineSize      = alignedBytesSize / sizeof(algorithmFPType); // to elements

        _cacheData.reset(newLineSize * _cacheSize);
        DAAL_CHECK_MALLOC(_cacheData.get());
        _cache.reset(_cacheSize);
        DAAL_CHECK_MALLOC(_cache.get());

        for (size_t i = 0; i < _cacheSize; ++i)
        {
            _cache[i] = &_cacheData[i * newLineSize]; // _cache[i] - always aligned on 64 bytes
        }

        return status;
    }

    services::Status initBlockTask(const size_t nSize)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(cache.initBlockTask);

        services::Status status;

        SubDataTaskBase<algorithmFPType, cpu> * task = nullptr;
        if (_xTable->getDataLayout() == NumericTableIface::csrArray)
        {
            task = SubDataTaskCSR<algorithmFPType, cpu>::create(_xTable, nSize);
        }
        else
        {
            task = SubDataTaskDense<algorithmFPType, cpu>::create(_xTable->getNumberOfColumns(), nSize);
        }

        DAAL_CHECK_MALLOC(task);
        if (_blockTask.get() != nullptr)
        {
            _blockTask.reset();
        }
        _blockTask = SubDataTaskBasePtr<algorithmFPType, cpu>(task);

        return status;
    }

    services::Status init(const size_t nSize)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(cache.init);

        services::Status status;
        status |= initKernelIndex(nSize);
        status |= initCache();
        status |= initBlockTask(nSize);
        return status;
    }

protected:
    LRUCache<cpu, uint32_t> _lruCache;
    const NumericTablePtr & _xTable;
    SubDataTaskBasePtr<algorithmFPType, cpu> _blockTask;
    TArray<uint32_t, cpu> _kernelOriginalIndex;
    TArray<uint32_t, cpu> _kernelIndex;
    TArrayScalable<algorithmFPType *, cpu> _cache;
    TArrayScalable<algorithmFPType, cpu> _cacheData;
    TArrayScalable<algorithmFPType *, cpu> _soaData;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
