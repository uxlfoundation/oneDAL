/* file: implicit_als_train_init_kernel.h */
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
//  Declaration of structure containing kernels for implicit ALS
//  initialization.
//--
*/

#ifndef __IMPLICIT_ALS_INIT_TRAIN_KERNEL_H__
#define __IMPLICIT_ALS_INIT_TRAIN_KERNEL_H__

#include "algorithms/implicit_als/implicit_als_training_init_batch.h"
#include "algorithms/implicit_als/implicit_als_model.h"
#include "src/algorithms/kernel.h"
#include "src/algorithms/engines/engine_impl.h"

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
using namespace daal::data_management;
using namespace daal::internal;

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitKernelBase : public daal::algorithms::Kernel
{
public:
    services::Status randFactors(size_t nItems, size_t nFactors, algorithmFPType * itemsFactors, engines::BatchBase & engine);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSInitKernel : public ImplicitALSInitKernelBase<algorithmFPType, cpu>
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitKernel<algorithmFPType, fastCSR, cpu> : public ImplicitALSInitKernelBase<algorithmFPType, cpu>
{
public:
    services::Status compute(const NumericTable * data, NumericTable * itemsFactors, NumericTable * usersFactors, const Parameter * parameter,
                             engines::BatchBase & engine);

protected:
    services::Status reduceSumByColumns(algorithmFPType ** arrSum, const size_t nItems, const size_t nBlocks, algorithmFPType * const arrForReduce);

    services::Status computeSumByColumnsCSR(const algorithmFPType * data, const size_t * colIndices, const size_t * rowOffsets, const size_t nUsers,
                                            const size_t nItems, const size_t nFactors, algorithmFPType * const itemsFactors,
                                            algorithmFPType * const itemsSum, algorithmFPType * const notNullElemSum, const bool oneAsBase);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitKernel<algorithmFPType, defaultDense, cpu> : public ImplicitALSInitKernelBase<algorithmFPType, cpu>
{
public:
    services::Status compute(const NumericTable * data, NumericTable * itemsFactors, NumericTable * usersFactors, const Parameter * parameter,
                             engines::BatchBase & engine);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSInitDistrKernelBase
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>
{
protected:
    void computeOffsets(size_t nParts, const int * partition, NumericTable ** offsets);
    services::Status computeBlocksToLocal(size_t nItems, size_t fullNUsers, const size_t * rowIndices, const size_t * colOffsets, size_t nParts,
                                          const int * partition, NumericTable ** blocksToLocal);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSInitDistrKernel
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitDistrKernel<algorithmFPType, fastCSR, cpu> : public ImplicitALSInitKernelBase<algorithmFPType, cpu>,
                                                                  public ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>
{
public:
    services::Status compute(const NumericTable * data, const NumericTable * partitionTable, NumericTable ** dataParts, NumericTable ** blocksToLocal,
                             NumericTable ** userOffsets, NumericTable * partialFactors, const DistributedParameter * parameter,
                             engines::BatchBase & engine);

protected:
    using ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>::computeOffsets;
    using ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>::computeBlocksToLocal;

    services::Status transposeAndSplitCSRTable(size_t nItems, size_t fullNUsers, const algorithmFPType * tdata, const size_t * rowIndices,
                                               const size_t * colOffsets, size_t nParts, const int * partition, NumericTable ** dataParts);

    services::Status computePartialFactors(const size_t nItems, const size_t nFactors, const algorithmFPType * tdata, const size_t * rowIndices,
                                           algorithmFPType * const itemsFactors);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSInitDistrStep2Kernel
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSInitDistrStep2Kernel<algorithmFPType, fastCSR, cpu> : public daal::algorithms::Kernel,
                                                                       public ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>
{
public:
    services::Status compute(size_t nParts, NumericTable ** dataParts, NumericTable * data, NumericTable ** blocksToLocal,
                             NumericTable ** itemOffsets);

protected:
    using ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>::computeOffsets;
    using ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>::computeBlocksToLocal;

    services::Status mergeCSRTables(size_t nParts, NumericTable ** dataParts, size_t nRows, algorithmFPType * data, size_t * rowOffsets,
                                    size_t * colIndices);
};

} // namespace internal
} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
