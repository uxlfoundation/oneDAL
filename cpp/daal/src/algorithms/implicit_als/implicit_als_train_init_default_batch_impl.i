/* file: implicit_als_train_init_default_batch_impl.i */
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
//  Implementation of defaultDense method for impicit ALS initialization
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_INIT_DEFAULT_BATCH_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_INIT_DEFAULT_BATCH_IMPL_I__

#include "src/algorithms/distributions/uniform/uniform_kernel.h"
#include "src/algorithms/distributions/uniform/uniform_impl.i"
#include "src/algorithms/service_error_handling.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSInitKernelBase<algorithmFPType, cpu>::randFactors(size_t nItems, size_t nFactors, algorithmFPType * itemsFactors,
                                                                              engines::BatchBase & engine)
{
    const size_t nTheads       = threader_get_threads_number();
    const size_t nBlocks       = nTheads;
    const size_t blockSize     = (nItems * nFactors) / nBlocks;
    const size_t lastBlockSize = (nItems * nFactors) - (nBlocks - 1) * blockSize;

    TArray<services::SharedPtr<engines::BatchBase>, cpu> engines(nBlocks - 1);
    for (size_t i = 0; i < nBlocks - 1; i++)
    {
        engines[i] = services::dynamicPointerCast<engines::BatchBase>(engine.clone());
    }

    daal::SafeStatus safeStatus;
    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
        const size_t start          = blockSize * iBlock;
        const size_t nElems         = (iBlock != nBlocks - 1) ? blockSize : lastBlockSize;
        algorithmFPType * const arr = itemsFactors + start;

        if (iBlock != 0)
        {
            engines[iBlock - 1]->skipAhead(start);
            safeStatus |=
                distributions::uniform::internal::UniformKernelDefault<algorithmFPType, cpu>::compute(0.0f, 1.0f, *engines[iBlock - 1], nElems, arr);
        }
        else
        {
            safeStatus |= distributions::uniform::internal::UniformKernelDefault<algorithmFPType, cpu>::compute(0.0f, 1.0f, engine, nElems, arr);
        }
    });

    return safeStatus.detach();
}

} // namespace internal
} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
