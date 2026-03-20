/* file: gbt_classification_train_kernel.h */
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

#ifndef __GBT_CLASSIFICATION_TRAIN_KERNEL_H__
#define __GBT_CLASSIFICATION_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/gradient_boosted_trees/gbt_classification_training_types.h"
#include "src/algorithms/engines/engine_batch_impl.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, Method method, CpuType cpu>
class ClassificationTrainBatchKernel : public daal::algorithms::Kernel
{
public:
    /**
     *  \brief Trains gradient boosted trees classification model.
     *
     *  \param x[in]            Matrix of input variables X of size [N x P], where N is a number of observations
     *                          and P is a number of features
     *  \param y[in]            Matrix of dependent variables Y of size [N x 1], where N is a number of observations
     *  \param m[out]           Gradient boosted trees classification model to be trained
     *  \param res[out]         Structure to store the training results such as model
     *  \param par[in]          Gradient boosted trees classification algorithm parameters
     *  \param engine[in]       Engine for random number generation to be used for training
     */
    services::Status compute(const NumericTable * x, const NumericTable * y, gbt::classification::Model & m, Result & res,
                             const interface2::Parameter & par, engines::internal::BatchBaseImpl & engine);
};

} // namespace internal
} // namespace training
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
