/* file: ridge_regression_train_dense_normeq_helper_impl.i */
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
//  Implementation of auxiliary functions for ridge regression Normal Equations (normEqDense) method.
//--
*/

#ifndef __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_HELPER_IMPL_I__
#define __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_HELPER_IMPL_I__

#include "src/algorithms/ridge_regression/ridge_regression_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace internal
{
using namespace daal::algorithms::linear_model::normal_equations::training::internal;

template <typename algorithmFPType, CpuType cpu>
Status KernelHelper<algorithmFPType, cpu>::computeBetasImpl(DAAL_INT p, const algorithmFPType * a, algorithmFPType * aCopy, DAAL_INT ny,
                                                            algorithmFPType * b, bool interceptFlag) const
{
    /* Ridge parameters are stored as a 1 x nResponses table (one value per response) or a 1 x 1 table */
    const size_t nRidge = _ridge.getNumberOfRows() * _ridge.getNumberOfColumns();
    ReadRows<algorithmFPType, cpu> ridgeBlock(const_cast<NumericTable &>(_ridge), 0, _ridge.getNumberOfRows());
    DAAL_CHECK_BLOCK_STATUS(ridgeBlock);
    const algorithmFPType * ridge = ridgeBlock.get();
    DAAL_CHECK(nRidge == 1 || nRidge == static_cast<size_t>(ny), services::ErrorIncorrectNumberOfColumns);

    const DAAL_INT pToFix = (interceptFlag ? p - 1 : p);

    Status st;
    int result = 0;
    if (nRidge == 1)
    {
        for (DAAL_INT i = 0, idx = 0; i < pToFix; i++, idx += (p + 1))
        {
            aCopy[idx] += *ridge;
        }

        st |= FinalizeKernel<algorithmFPType, cpu>::solveSystem(p, aCopy, ny, b, getInternalErrorCode());
        DAAL_CHECK_STATUS_VAR(st);
    }
    else
    {
        algorithmFPType * bPtr    = b;
        const size_t aSizeInBytes = p * p * sizeof(algorithmFPType);

        /* The solver overwrites aCopy, so if the caller passed the same buffer as a and aCopy,
           save the original matrix to restore it for each response */
        const algorithmFPType * aOrig = a;
        TArray<algorithmFPType, cpu> aSavedArray;
        if (a == aCopy)
        {
            algorithmFPType * aSaved = aSavedArray.reset(p * p);
            DAAL_CHECK_MALLOC(aSaved);
            DAAL_CHECK(!daal::services::internal::daal_memcpy_s(aSaved, aSizeInBytes, a, aSizeInBytes), services::ErrorMemoryCopyFailedInternal);
            aOrig = aSaved;
        }

        /* b is of size ny x p, the j-th row contains the right hand side for the j-th response */
        for (DAAL_INT j = 0; j < ny; j++, bPtr += p)
        {
            result |= daal::services::internal::daal_memcpy_s(aCopy, aSizeInBytes, aOrig, aSizeInBytes);
            for (DAAL_INT i = 0, idx = 0; i < pToFix; i++, idx += (p + 1))
            {
                aCopy[idx] += ridge[j];
            }

            DAAL_INT one(1);

            st |= FinalizeKernel<algorithmFPType, cpu>::solveSystem(p, aCopy, one, bPtr, getInternalErrorCode());
            DAAL_CHECK_STATUS_VAR(st);
        }
    }
    return (!result) ? st : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
ErrorID KernelHelper<algorithmFPType, cpu>::getInternalErrorCode() const
{
    return ErrorRidgeRegressionInternal;
}

} // namespace internal
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
