/* file: gbt_regression_train_kernel.h */
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
//  Declaration of structure containing kernels for gradient boosted trees
//  training.
//--
*/

#ifndef __GBT_REGRESSION_TRAIN_KERNEL_H__
#define __GBT_REGRESSION_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_training_types.h"
#include "src/algorithms/engines/engine_batch_impl.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, Method method, CpuType cpu>
class RegressionTrainBatchKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, gbt::regression::Model & m, Result & res,
                             const Parameter & par, engines::internal::BatchBaseImpl & engine);
};

} // namespace internal
} // namespace training
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
