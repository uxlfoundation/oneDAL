/* file: svm_train_common_impl.i */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  SVM training algorithm implementation
//--
*/

#ifndef __SVM_TRAIN_COMMON_IMPL_I__
#define __SVM_TRAIN_COMMON_IMPL_I__

#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_utils.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_profiler.h"
#include "src/algorithms/svm/svm_train_result.h"
#include "src/algorithms/svm/svm_train_common.h"

#if defined(DAAL_INTEL_CPP_COMPILER)
    #if defined(_M_AMD64) || defined(__amd64) || defined(__x86_64) || defined(__x86_64__)
        #if (__CPUID__(DAAL_CPU) == __avx512__)

            #include <immintrin.h>
            #include "src/algorithms/svm/svm_train_common_avx512_impl.i"

        #endif // __CPUID__(DAAL_CPU) == __avx512__
    #endif     // defined (_M_AMD64) || defined (__amd64) || defined (__x86_64) || defined (__x86_64__)
#elif defined(TARGET_ARM)
    #if (__CPUID__(DAAL_CPU) == __sve__)
        #include "src/algorithms/svm/svm_train_common_sve_impl.i"
    #endif // __CPUID__(DAAL_CPU) == __sve__
#endif

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
/**
 * \brief Working set selection (WSS3) function.
 *        Select an index i from a pair of indices B = {i, j} using WSS 3 algorithm from [1].
 *
 * \param[in] nActiveVectors    number of observations in a training data set that are used
 *                              in sequential minimum optimization at the current iteration
 * \param[out] Bi            resulting index i
 *
 * \return The function returns m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha)
 */
template <typename algorithmFPType, CpuType cpu>
algorithmFPType HelperTrainSVM<algorithmFPType, cpu>::WSSi(size_t nActiveVectors, const algorithmFPType * grad, const char * I, int & Bi,
                                                           SignNuType signNuType)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(findMaximumViolatingPair.WSSi);

    Bi                   = -1;
    algorithmFPType GMin = (MaxVal<algorithmFPType>::get()); // some big negative number
    const char sign      = getSign(signNuType);

    /* Find i index of the working set (Bi) */
    for (size_t i = 0; i < nActiveVectors; ++i)
    {
        const algorithmFPType objFunc = grad[i];
        if ((I[i] & sign) && (I[i] & up) && objFunc < GMin)
        {
            GMin = objFunc;
            Bi   = i;
        }
    }
    return GMin;
}

template <typename algorithmFPType, CpuType cpu>
void HelperTrainSVM<algorithmFPType, cpu>::WSSjLocalBaseline(const size_t jStart, const size_t jEnd, const algorithmFPType * KiBlock,
                                                             const algorithmFPType * kernelDiag, const algorithmFPType * grad, const char * I,
                                                             const algorithmFPType GMin, const algorithmFPType Kii, const algorithmFPType tau,
                                                             int & Bj, algorithmFPType & GMax, algorithmFPType & GMax2, algorithmFPType & delta,
                                                             SignNuType signNuType)
{
    algorithmFPType fpMax = MaxVal<algorithmFPType>::get();
    GMax                  = -fpMax; // some big negative number
    GMax2                 = -fpMax;

    const algorithmFPType zero(0.0);
    const algorithmFPType two(2.0);

    const char sign = getSign(signNuType);

    for (size_t j = jStart; j < jEnd; j++)
    {
        const algorithmFPType gradj = grad[j];
        if (!(I[j] & sign))
        {
            continue;
        }
        if ((I[j] & low) != low)
        {
            continue;
        }
        if (gradj > GMax2)
        {
            GMax2 = gradj;
        }
        if (gradj < GMin)
        {
            continue;
        }

        const algorithmFPType b = GMin - gradj;
        algorithmFPType a       = Kii + kernelDiag[j] - two * KiBlock[j - jStart];
        if (a <= zero)
        {
            a = tau;
        }
        const algorithmFPType dt      = b / a;
        const algorithmFPType objFunc = b * dt;
        if (objFunc > GMax)
        {
            GMax  = objFunc;
            Bj    = j;
            delta = -dt;
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
void HelperTrainSVM<algorithmFPType, cpu>::WSSjLocal(const size_t jStart, const size_t jEnd, const algorithmFPType * KiBlock,
                                                     const algorithmFPType * kernelDiag, const algorithmFPType * grad, const char * I,
                                                     const algorithmFPType GMin, const algorithmFPType Kii, const algorithmFPType tau, int & Bj,
                                                     algorithmFPType & GMax, algorithmFPType & GMax2, algorithmFPType & delta, SignNuType signNuType)
{
    WSSjLocalBaseline(jStart, jEnd, KiBlock, kernelDiag, grad, I, GMin, Kii, tau, Bj, GMax, GMax2, delta, signNuType);
}

template <CpuType cpu, typename TKey>
void LRUCache<cpu, TKey>::put(const TKey key)
{
    LRUNode * node = nullptr;
    if (_hashmap.find(key, node))
    {
        enqueue(node);
    }
    else
    {
        node = LRUNode::create(key, _freeIndexCache + 1);
        if (!node) return;
        enqueue(node);
        if (_count == _capacity)
        {
            const int64_t freeIndex = dequeue();
            node->setValue(freeIndex);
            _freeIndexCache = freeIndex;
        }
        else
        {
            ++_freeIndexCache;
        }
        _hashmap.insert(key, node);
        ++_count;
    }
}

template <CpuType cpu, typename TKey>
int64_t LRUCache<cpu, TKey>::get(const TKey key)
{
    LRUNode * node = nullptr;
    if (_hashmap.find(key, node))
    {
        enqueue(node);
        return node->getValue();
    }
    else
    {
        return -1;
    }
}

template <CpuType cpu, typename TKey>
void LRUCache<cpu, TKey>::enqueue(LRUNode * node)
{
    if (!_head)
    {
        _head = node;
        _tail = node;
    }
    else if (node != _head)
    {
        if (node == _tail)
        {
            _tail = node->prev;
        }
        LRUNode * prev = node->prev;
        if (prev)
        {
            prev->next = node->next;
        }
        if (node->next)
        {
            node->next->prev = prev;
        }
        _head->prev = node;
        node->next  = _head;
        _head       = node;
        _head->prev = nullptr;
        _tail->next = nullptr;
    }
}

template <CpuType cpu, typename TKey>
int64_t LRUCache<cpu, TKey>::dequeue()
{
    int64_t value = -1;
    if (_head == _tail)
    {
        delete _head;
        _head = nullptr;
        _tail = nullptr;
    }
    else
    {
        _hashmap.erase(_tail->getKey());
        value          = _tail->getValue();
        LRUNode * prev = _tail->prev;
        if (_tail->prev)
        {
            _tail->prev->next = nullptr;
        }
        delete _tail;
        _tail = prev;
        --_count;
    }
    return value;
}

template <typename algorithmFPType, CpuType cpu>
services::Status SubDataTaskCSR<algorithmFPType, cpu>::copyDataByIndices(const uint32_t * wsIndices, const size_t nSubsetVectors,
                                                                         const NumericTablePtr & xTable)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(cache.copyDataByIndices.CSR);
    services::Status status;
    CSRNumericTableIface * csrIface = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(xTable.get()));
    DAAL_CHECK(csrIface, services::ErrorEmptyCSRNumericTable);
    DAAL_CHECK_STATUS(status, this->_dataTable->resize(nSubsetVectors));

    _rowOffsets[0] = 1;
    for (size_t i = 0; i < nSubsetVectors; ++i)
    {
        size_t iRows = wsIndices[i];
        ReadRowsCSR<algorithmFPType, cpu> mtX(csrIface, iRows, 1);
        DAAL_CHECK_BLOCK_STATUS(mtX);
        const size_t nNonZeroValuesInRow     = mtX.rows()[1] - mtX.rows()[0];
        const size_t * const colIndices      = mtX.cols();
        const algorithmFPType * const values = mtX.values();
        const size_t offsetOut               = _rowOffsets[i] - _rowOffsets[0];
        {
            // Copy values
            const algorithmFPType * const dataIn = values;
            algorithmFPType * const dataOut      = this->_data.get() + offsetOut;
            DAAL_CHECK(!services::internal::daal_memcpy_s(dataOut, nNonZeroValuesInRow * sizeof(algorithmFPType), dataIn,
                                                          nNonZeroValuesInRow * sizeof(algorithmFPType)),
                       services::ErrorMemoryCopyFailedInternal);
        }
        {
            // Copy col indices
            const size_t * const dataIn = colIndices;
            size_t * const dataOut      = _colIndices.get() + offsetOut;
            DAAL_CHECK(
                !services::internal::daal_memcpy_s(dataOut, nNonZeroValuesInRow * sizeof(size_t), dataIn, nNonZeroValuesInRow * sizeof(size_t)),
                services::ErrorMemoryCopyFailedInternal);
        }
        _rowOffsets[i + 1] = _rowOffsets[i] + nNonZeroValuesInRow;
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status SubDataTaskDense<algorithmFPType, cpu>::copyDataByIndices(const uint32_t * wsIndices, const size_t nSubsetVectors,
                                                                           const NumericTablePtr & xTable)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(cache.copyDataByIndices.Dense);
    services::Status status;
    NumericTable & x = *xTable.get();
    const size_t p   = x.getNumberOfColumns();
    DAAL_CHECK_STATUS(status, this->_dataTable->resize(nSubsetVectors));
    const size_t nBlock = nSubsetVectors;

    SafeStatus safeStat;
    daal::threader_for(nBlock, nBlock, [&](const size_t iBlock) {
        size_t iRows = wsIndices[iBlock];
        ReadRows<algorithmFPType, cpu> mtX(x, iRows, 1);
        DAAL_CHECK_BLOCK_STATUS_THR(mtX);
        const algorithmFPType * const dataIn = mtX.get();
        algorithmFPType * dataOut            = this->_data.get() + iBlock * p;
        DAAL_CHECK_THR(!services::internal::daal_memcpy_s(dataOut, p * sizeof(algorithmFPType), dataIn, p * sizeof(algorithmFPType)),
                       services::ErrorMemoryCopyFailedInternal);
    });

    return safeStat.detach();
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
