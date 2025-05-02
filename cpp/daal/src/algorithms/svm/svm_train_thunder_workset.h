/* file: svm_train_thunder_workset.h */
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
//  SVM workset structure implementation
//--
*/

#ifndef __SVM_TRAIN_THUNDER_WORKSET_H__
#define __SVM_TRAIN_THUNDER_WORKSET_H__

#include "src/services/service_utils.h"
#include "src/algorithms/service_sort.h"
#include "src/algorithms/svm/svm_train_kernel.h"

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
using namespace daal::services::internal;

template <typename algorithmFPType, CpuType cpu>
struct TaskWorkingSet
{
    using IndexType  = uint32_t;
    using IdxValType = daal::IdxValType<algorithmFPType>;

    TaskWorkingSet(const size_t nNonZeroWeights, const size_t nVectors, const size_t maxWS, SvmType svmType)
        : _nNonZeroWeights(nNonZeroWeights), _nVectors(nVectors), _maxWS(maxWS), _svmType(svmType)
    {}

    services::Status init()
    {
        services::Status status;
        _sortedFIndices.reset(_nVectors);
        DAAL_CHECK_MALLOC(_sortedFIndices.get());

        _indicator.reset(_nVectors);
        DAAL_CHECK_MALLOC(_indicator.get());
        services::internal::service_memset_seq<bool, cpu>(_indicator.get(), false, _nVectors);

        _nWS       = services::internal::min<cpu, algorithmFPType>(maxPowTwo(_nNonZeroWeights), _maxWS);
        _nSelected = 0;

        _wsIndices.reset(_nWS);
        DAAL_CHECK_MALLOC(_wsIndices.get());

        return status;
    }

    size_t getSize() const { return _nWS; }

    services::Status copyLastToFirst()
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(select.copyLastToFirst);

        services::Status status;
        const size_t q = _nWS / 2;

        services::internal::daal_memcpy_s(_wsIndices.get(), q * sizeof(IndexType), _wsIndices.get() + _nWS - q, q * sizeof(IndexType));
        _nSelected = q;
        services::internal::service_memset_seq<bool, cpu>(_indicator.get(), false, _nVectors);
        for (size_t i = 0; i < q; ++i)
        {
            _indicator[_wsIndices[i]] = true;
        }

        return status;
    }

    services::Status select(const algorithmFPType * y, const algorithmFPType * alpha, const algorithmFPType * f, const algorithmFPType * cw)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(select);
        services::Status status;
        IdxValType * sortedFIndices = _sortedFIndices.get();

        /* The operation copy is lightweight, therefore a large size is chosen
        so that the number of blocks is a reasonable number. */
        const size_t blockSize = 16384;
        const size_t nBlocks   = _nVectors / blockSize + !!(_nVectors % blockSize);
        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            const size_t startRow = iBlock * blockSize;
            const size_t endRow   = (iBlock != nBlocks - 1) ? startRow + blockSize : _nVectors;
            for (size_t i = startRow; i < endRow; ++i)
            {
                sortedFIndices[i].value = f[i];
                sortedFIndices[i].index = i;
            }
        });

        daal::parallel_sort(sortedFIndices, sortedFIndices + _nVectors);

        if (_svmType == SvmType::nu_classification || _svmType == SvmType::nu_regression)
        {
            int64_t pLeftPos  = 0;
            int64_t pLeftNeg  = 0;
            int64_t pRightPos = _nVectors - 1;
            int64_t pRightNeg = _nVectors - 1;
            while (_nSelected < _nWS && (pRightPos >= 0 || pRightNeg >= 0 || pLeftPos < _nVectors || pLeftNeg < _nVectors))
            {
                moveRight(pLeftPos, sortedFIndices, y, alpha, cw, SignNuType::positive);
                if (_nSelected == _nWS) break;
                moveRight(pLeftNeg, sortedFIndices, y, alpha, cw, SignNuType::negative);
                if (_nSelected == _nWS) break;
                moveLeft(pRightPos, sortedFIndices, y, alpha, cw, SignNuType::positive);
                if (_nSelected == _nWS) break;
                moveLeft(pRightNeg, sortedFIndices, y, alpha, cw, SignNuType::negative);
            }
        }
        else
        {
            int64_t pLeft  = 0;
            int64_t pRight = _nVectors - 1;
            while (_nSelected < _nWS && (pRight >= 0 || pLeft < _nVectors))
            {
                moveRight(pLeft, sortedFIndices, y, alpha, cw);
                if (_nSelected == _nWS) break;
                moveLeft(pRight, sortedFIndices, y, alpha, cw);
            }
        }

        // For cases, when weights are zero
        int64_t pLeft = 0;
        while (_nSelected < _nWS)
        {
            if (!_indicator[pLeft])
            {
                _wsIndices[_nSelected] = pLeft;
                _indicator[pLeft]      = true;
                ++_nSelected;
            }
            ++pLeft;
        }
        DAAL_ASSERT(_nSelected == _nWS);
        _nSelected = 0;
        return status;
    }

    const IndexType * getIndices() const { return _wsIndices.get(); }

protected:
    size_t maxPowTwo(size_t n)
    {
        if (!(n & (n - 1)))
        {
            return n;
        }

        size_t count = 0;
        while (n > 1)
        {
            n >>= 1;
            ++count;
        }
        return 1 << count;
    }

    void moveRight(int64_t & pLeft, const IdxValType * sortedFIndices, const algorithmFPType * y, const algorithmFPType * alpha,
                   const algorithmFPType * cw, SignNuType signNuType = SignNuType::none)
    {
        if (pLeft < _nVectors)
        {
            IndexType i = sortedFIndices[pLeft].index;
            while (_indicator[i] || !HelperTrainSVM<algorithmFPType, cpu>::isUpper(y[i], alpha[i], cw[i], signNuType))
            {
                pLeft++;
                if (pLeft == _nVectors)
                {
                    break;
                }
                i = sortedFIndices[pLeft].index;
            }
            if (pLeft < _nVectors)
            {
                _wsIndices[_nSelected] = i;
                _indicator[i]          = true;
                ++_nSelected;
            }
        }
    }

    void moveLeft(int64_t & pRight, const IdxValType * sortedFIndices, const algorithmFPType * y, const algorithmFPType * alpha,
                  const algorithmFPType * cw, SignNuType signNuType = SignNuType::none)
    {
        if (pRight >= 0)
        {
            IndexType i = sortedFIndices[pRight].index;
            while (_indicator[i] || !HelperTrainSVM<algorithmFPType, cpu>::isLower(y[i], alpha[i], cw[i], signNuType))
            {
                pRight--;
                if (pRight == -1)
                {
                    break;
                }
                i = sortedFIndices[pRight].index;
            }
            if (pRight >= 0)
            {
                _wsIndices[_nSelected] = i;
                _indicator[i]          = true;
                ++_nSelected;
            }
        }
    }

private:
    size_t _nNonZeroWeights;
    size_t _nVectors;
    size_t _maxWS;
    size_t _nSelected;
    size_t _nWS;

    TArray<IdxValType, cpu> _sortedFIndices;
    TArray<bool, cpu> _indicator;
    TArray<IndexType, cpu> _wsIndices;

    SvmType _svmType;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
