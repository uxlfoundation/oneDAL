/* file: elastic_net_train_kernel.h */
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
//  Declaration of structure containing kernels for elastic net
//  training.
//--
*/

#ifndef __ELASTIC_NET_TRAIN_KERNEL_H__
#define __ELASTIC_NET_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/optimization_solver/objective_function/mse_batch.h"
#include "algorithms/elastic_net/elastic_net_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace elastic_net
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class TrainBatchKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTablePtr & x, const NumericTablePtr & y, elastic_net::Model & m, Result & res, const Parameter & par,
                             services::SharedPtr<daal::algorithms::optimization_solver::mse::Batch<algorithmFPType> > & objFunc);
};

} // namespace internal
} // namespace training
} // namespace elastic_net
} // namespace algorithms
} // namespace daal

#endif
