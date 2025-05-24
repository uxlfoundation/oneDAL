/* file: df_classification_train_kernel.h */
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
//  Declaration of structure containing kernels for decision forest
//  training.
//--
*/

#ifndef __DF_CLASSFICATION_TRAIN_KERNEL_H__
#define __DF_CLASSFICATION_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/decision_forest/decision_forest_training_parameter.h"
#include "src/algorithms/dtrees/forest/df_hyperparameter_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class ClassificationTrainBatchKernel : public daal::algorithms::Kernel
{
public:
    typedef daal::algorithms::decision_forest::classification::training::internal::Hyperparameter HyperparameterType;
    services::Status compute(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, const NumericTable * w,
                             decision_forest::classification::Model & m, Result & res,
                             const decision_forest::classification::training::Parameter & par, const HyperparameterType * hyperparameter = nullptr);
};

} // namespace internal
} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
