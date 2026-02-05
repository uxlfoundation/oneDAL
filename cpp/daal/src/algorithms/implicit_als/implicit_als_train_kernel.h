/* file: implicit_als_train_kernel.h */
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
//  training.
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_KERNEL_H__
#define __IMPLICIT_ALS_TRAIN_KERNEL_H__

#include "algorithms/implicit_als/implicit_als_training_batch.h"
#include "algorithms/implicit_als/implicit_als_model.h"
#include "src/algorithms/kernel.h"
#include "src/threading/threading.h"

#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_memory.h"

using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
struct ImplicitALSTrainTaskBase;

template <typename algorithmFPType, Method method, CpuType cpu>
struct ImplicitALSTrainTask;

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainKernelCommon : public daal::algorithms::Kernel
{
protected:
    void computeXtX(size_t * nRows, size_t * nCols, algorithmFPType * beta, algorithmFPType * x, size_t * ldx, algorithmFPType * xtx, size_t * ldxtx);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainKernelBase : public ImplicitALSTrainKernelCommon<algorithmFPType, cpu>
{
public:
    static void updateSystem(size_t nCols, const algorithmFPType * x, const algorithmFPType * coeff, const algorithmFPType * p, algorithmFPType * a,
                             algorithmFPType * b);

    static bool solve(size_t nCols, algorithmFPType * a, algorithmFPType * b);

protected:
    friend struct ImplicitALSTrainTaskBase<algorithmFPType, cpu>;
    friend struct ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu>;
    friend struct ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu>;

    services::Status computeFactors(size_t nRows, size_t nCols, const algorithmFPType * data, const size_t * colIndices, const size_t * rowOffsets,
                                    size_t nFactors, algorithmFPType * colFactors, algorithmFPType * rowFactors, algorithmFPType alpha,
                                    algorithmFPType lambda, algorithmFPType * xtx, daal::tls<algorithmFPType *> & lhs);

    virtual void formSystem(size_t i, size_t nCols, const algorithmFPType * data, const size_t * colIndices, const size_t * rowOffsets,
                            size_t nFactors, algorithmFPType * colFactors, algorithmFPType alpha, algorithmFPType * lhs, algorithmFPType * rhs,
                            algorithmFPType lambda) = 0;

    virtual void computeCostFunction(size_t nUsers, size_t nItems, size_t nFactors, algorithmFPType * data, size_t * colIndices, size_t * rowOffsets,
                                     algorithmFPType * itemsFactors, algorithmFPType * usersFactors, algorithmFPType alpha, algorithmFPType lambda,
                                     algorithmFPType * costFunctionPtr) = 0;
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSTrainKernel : public ImplicitALSTrainKernelBase<algorithmFPType, cpu>
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainKernel<algorithmFPType, fastCSR, cpu> : public ImplicitALSTrainKernelBase<algorithmFPType, cpu>
{
protected:
    virtual void formSystem(size_t i, size_t nCols, const algorithmFPType * data, const size_t * colIndices, const size_t * rowOffsets,
                            size_t nFactors, algorithmFPType * colFactors, algorithmFPType alpha, algorithmFPType * lhs, algorithmFPType * rhs,
                            algorithmFPType lambda) override;

    virtual void computeCostFunction(size_t nUsers, size_t nItems, size_t nFactors, algorithmFPType * data, size_t * colIndices, size_t * rowOffsets,
                                     algorithmFPType * itemsFactors, algorithmFPType * usersFactors, algorithmFPType alpha, algorithmFPType lambda,
                                     algorithmFPType * costFunctionPtr) override;
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainKernel<algorithmFPType, defaultDense, cpu> : public ImplicitALSTrainKernelBase<algorithmFPType, cpu>
{
protected:
    virtual void formSystem(size_t i, size_t nCols, const algorithmFPType * data, const size_t * colIndices, const size_t * rowOffsets,
                            size_t nFactors, algorithmFPType * colFactors, algorithmFPType alpha, algorithmFPType * lhs, algorithmFPType * rhs,
                            algorithmFPType lambda) override;

    virtual void computeCostFunction(size_t nUsers, size_t nItems, size_t nFactors, algorithmFPType * data, size_t * colIndices, size_t * rowOffsets,
                                     algorithmFPType * itemsFactors, algorithmFPType * usersFactors, algorithmFPType alpha, algorithmFPType lambda,
                                     algorithmFPType * costFunctionPtr) override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSTrainBatchKernel : public ImplicitALSTrainKernel<algorithmFPType, method, cpu>
{};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainBatchKernel<algorithmFPType, fastCSR, cpu> : public ImplicitALSTrainKernel<algorithmFPType, fastCSR, cpu>
{
public:
    services::Status compute(const NumericTable * data, implicit_als::Model * initModel, implicit_als::Model * model, const Parameter * parameter);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainBatchKernel<algorithmFPType, defaultDense, cpu> : public ImplicitALSTrainKernel<algorithmFPType, defaultDense, cpu>
{
public:
    services::Status compute(const NumericTable * data, implicit_als::Model * initModel, implicit_als::Model * model, const Parameter * parameter);
};

template <typename algorithmFPType, CpuType cpu>
struct ImplicitALSTrainTaskBase
{
    ImplicitALSTrainTaskBase(const NumericTable * dataTable, implicit_als::Model * model, const Parameter * parameter);
    services::Status init(const NumericTable * dataTable, implicit_als::Model * initModel, const Parameter * parameter);

    size_t nItems;
    size_t nUsers;
    size_t nFactors;

    daal::internal::WriteOnlyRows<algorithmFPType, cpu> mtItemsFactors;
    daal::internal::WriteOnlyRows<algorithmFPType, cpu> mtUsersFactors;
    daal::internal::TArray<algorithmFPType, cpu> xtx;
};

template <typename algorithmFPType, Method method, CpuType cpu>
struct ImplicitALSTrainTask : ImplicitALSTrainTaskBase<algorithmFPType, cpu>
{};

template <typename algorithmFPType, CpuType cpu>
struct ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu> : ImplicitALSTrainTaskBase<algorithmFPType, cpu>
{
    typedef ImplicitALSTrainTaskBase<algorithmFPType, cpu> super;
    ImplicitALSTrainTask(const NumericTable * dataTable, implicit_als::Model * model, const Parameter * parameter);

    services::Status init(const NumericTable * dataTable, implicit_als::Model * initModel, const Parameter * parameter);

    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::nItems;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::nUsers;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::nFactors;

    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::mtItemsFactors;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::mtUsersFactors;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::xtx;

    daal::internal::ReadRowsCSR<algorithmFPType, cpu> mtData;
    daal::internal::TArray<algorithmFPType, cpu> tdata;
    daal::internal::TArray<size_t, cpu> rowIndices;
    daal::internal::TArray<size_t, cpu> colOffsets;
};

template <typename algorithmFPType, CpuType cpu>
struct ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu> : ImplicitALSTrainTaskBase<algorithmFPType, cpu>
{
    typedef ImplicitALSTrainTaskBase<algorithmFPType, cpu> super;
    ImplicitALSTrainTask(const NumericTable * dataTable, implicit_als::Model * model, const Parameter * parameter);
    services::Status init(const NumericTable * dataTable, implicit_als::Model * initModel, const Parameter * parameter);
    void transpose(size_t nItems, size_t nUsers, const algorithmFPType * data, algorithmFPType * tdata);

    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::nItems;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::nUsers;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::nFactors;

    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::mtItemsFactors;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::mtUsersFactors;
    using ImplicitALSTrainTaskBase<algorithmFPType, cpu>::xtx;

    daal::internal::ReadRows<algorithmFPType, cpu> mtData;
    daal::internal::TArray<algorithmFPType, cpu> tdata;
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainDistrStep1Kernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(implicit_als::PartialModel * partialModel, data_management::NumericTable * crossProduct, const Parameter * parameter);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainDistrStep2Kernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(size_t nParts, data_management::NumericTable ** partialCrossProducts, data_management::NumericTable * crossProduct,
                             const Parameter * parameter);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainDistrStep3Kernel : public ImplicitALSTrainKernelCommon<algorithmFPType, cpu>
{
public:
    services::Status compute(implicit_als::PartialModel * partialModel, data_management::NumericTable * offsetTable,
                             data_management::KeyValueDataCollection * models, const Parameter * parameter);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class ImplicitALSTrainDistrStep4Kernel : public ImplicitALSTrainKernel<algorithmFPType, method, cpu>
{
public:
    services::Status compute(data_management::KeyValueDataCollection * models, data_management::NumericTable * dataTable,
                             data_management::NumericTable * cpTable, implicit_als::PartialModel * partialModel, const Parameter * parameter);
};

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSTrainDistrStep4Kernel<algorithmFPType, fastCSR, cpu> : public ImplicitALSTrainKernel<algorithmFPType, fastCSR, cpu>
{
public:
    services::Status compute(data_management::KeyValueDataCollection * models, data_management::NumericTable * dataTable,
                             data_management::NumericTable * cpTable, implicit_als::PartialModel * partialModel, const Parameter * parameter);
};

} // namespace internal
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
