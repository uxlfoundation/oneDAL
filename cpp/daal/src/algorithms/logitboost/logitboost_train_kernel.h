/* file: logitboost_train_kernel.h */
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
//  Declaration of structure containing kernels for logit boost model
//  training.
//--
*/

#ifndef __LOGITBOOST_TRAIN_KERNEL_H__
#define __LOGITBOOST_TRAIN_KERNEL_H__

#include "algorithms/boosting/logitboost_model.h"
#include "algorithms/boosting/logitboost_training_types.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace training
{
namespace internal
{
/**
 *  \brief Construct Logit Boost classifier model.
 *
 *  \param a[in]    Array of numeric tables containing input data
 *                  a[0] holds input matrix of features X
 *                  a[1] holds input matrix of class labels Y
 *  \param r[out]   Resulting model
 *  \param par[in]  Logit Boost algorithm parameters
 */
template <Method method, typename algorithmFPType, CpuType cpu>
struct LogitBoostTrainKernel : public Kernel
{
    typedef typename daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef typename services::SharedPtr<HomogenNT> HomogenNTPtr;
    services::Status compute(const size_t na, NumericTablePtr a[], Model * r, const Parameter * par);
};
} // namespace internal
} // namespace training
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
