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

#include "service/kernel/service_utils.h"
#include "externals/service_memory.h"
#include "service/kernel/data_management/service_micro_table.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "data_management/data/numeric_table_sycl_homogen.h"
#include "algorithms/kernel/svm/svm_train_cache.h"

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
class SVMCacheIface<thunder, algorithmFPType, cpu>
{
public:
    virtual ~SVMCacheIface() {}

    virtual services::Status compute(const uint32_t * wsIndices) = 0;

    virtual const algorithmFPType * getRowsBlock() const = 0;
    virtual services::Status copyLastToFirst()           = 0;

protected:
    SVMCacheIface(const size_t blockSize, const size_t lineSize, const kernel_function::KernelIfacePtr & kernel)
        : _lineSize(lineSize), _blockSize(blockSize), _kernel(kernel)
    {}

    const size_t _lineSize;                        /*!< Number of elements in the cache line */
    const size_t _blockSize;                       /*!< Number of cache lines */
    const kernel_function::KernelIfacePtr _kernel; /*!< Kernel function */
};

/**
 * No cache: kernel function values are not cached
 */
template <typename algorithmFPType, CpuType cpu>
class SVMCache<thunder, noCache, algorithmFPType, cpu> : public SVMCacheIface<thunder, algorithmFPType, cpu>
{
    using super    = SVMCacheIface<thunder, algorithmFPType, cpu>;
    using thisType = SVMCache<thunder, noCache, algorithmFPType, cpu>;
    using super::_kernel;
    using super::_lineSize;
    using super::_blockSize;

public:
    ~SVMCache() {}

    DAAL_NEW_DELETE();

    static SVMCachePtr<thunder, algorithmFPType, cpu> create(const size_t cacheSize, const size_t blockSize, const size_t lineSize,
                                                             const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel,
                                                             services::Status & status)
    {
        status.clear();
        services::SharedPtr<thisType> res = services::SharedPtr<thisType>(new thisType(blockSize, lineSize, xTable, kernel));
        if (!res)
        {
            status.add(ErrorMemoryAllocationFailed);
        }
        else
        {
            status = res->init(cacheSize);
            if (!status)
            {
                res.reset();
            }
        }
        return SVMCachePtr<thunder, algorithmFPType, cpu>(res);
    }

    const algorithmFPType * getRowsBlock() const override { return _cache.get(); }

    services::Status copyDataByIndices(const uint32_t * wsIndices)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(copyDataByIndices);
        services::Status status;
        NumericTable & x = *_xTable.get();
        const size_t p   = x.getNumberOfColumns();
        // TODO: tbb::parallel_for
        for (size_t i = _nSelectRows; i < _blockSize; i++)
        {
            size_t iRows = wsIndices[i];
            ReadRows<algorithmFPType, cpu> mtX(x, iRows, 1);
            DAAL_CHECK_BLOCK_STATUS(mtX);
            // DAAL_CHECK_BLOCK_STATUS_THR(mtX);
            const algorithmFPType * const dataInt = mtX.get();
            algorithmFPType * dataOut             = _xBlock.get() + i * p;
            services::internal::daal_memcpy_s(dataOut, p * sizeof(algorithmFPType), dataInt, p * sizeof(algorithmFPType));
        }
        return status;
    }

    services::Status compute(const uint32_t * wsIndices) override
    {
        services::Status status;
        DAAL_CHECK_STATUS(status, copyDataByIndices(wsIndices));
        DAAL_ITTNOTIFY_SCOPED_TASK(cacheCompute);
        DAAL_CHECK_STATUS(status, _kernel->computeNoThrow());
        return status;
    }

    services::Status copyLastToFirst() override
    {
        _nSelectRows        = _blockSize / 2;
        _ifComputeSubKernel = true;
        services::Status status;

        // auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        // context.copy(_cache, 0, _cache, _nSelectRows * _lineSize, _nSelectRows * _lineSize, &status);
        return status;
    }

protected:
    SVMCache(const size_t blockSize, const size_t lineSize, const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel)
        : super(blockSize, lineSize, kernel), _nSelectRows(0), _ifComputeSubKernel(false), _xTable(xTable)
    {}

    services::Status init(const size_t cacheSize)
    {
        services::Status status;
        _cache.reset(_lineSize * _blockSize);
        DAAL_CHECK_MALLOC(_cache.get());

        auto cacheTable = HomogenNumericTableCPU<algorithmFPType, cpu>::create(_cache.get(), _lineSize, _blockSize, &status);
        DAAL_CHECK_STATUS_VAR(status);

        const size_t p = _xTable->getNumberOfColumns();
        _xBlock.reset(_blockSize * p);
        DAAL_CHECK_MALLOC(_xBlock.get());

        const NumericTablePtr xWSTable = HomogenNumericTableCPU<algorithmFPType, cpu>::create(_xBlock.get(), p, _blockSize, &status);
        DAAL_CHECK_STATUS_VAR(status);

        _kernel->getParameter()->computationMode = kernel_function::matrixMatrix;
        _kernel->getInput()->set(kernel_function::X, xWSTable);
        _kernel->getInput()->set(kernel_function::Y, _xTable);

        kernel_function::ResultPtr shRes(new kernel_function::Result());
        shRes->set(kernel_function::values, cacheTable);
        _kernel->setResult(shRes);

        return status;
    }

protected:
    size_t _nSelectRows;
    bool _ifComputeSubKernel;
    const NumericTablePtr & _xTable;
    TArray<algorithmFPType, cpu> _xBlock; // [nWS x p]
    TArray<algorithmFPType, cpu> _cache;  // [nWS x n]
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
