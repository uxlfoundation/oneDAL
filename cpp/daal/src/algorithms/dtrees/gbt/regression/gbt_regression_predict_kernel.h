/* file: gbt_regression_predict_kernel.h */
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
//  Declaration of template function that computes gradient boosted trees
//  prediction results.
//--
*/

#ifndef __GBT_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__
#define __GBT_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__

#include "algorithms/gradient_boosted_trees/gbt_regression_predict.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace prediction
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;

template <typename algorithmFpType, gbt::regression::prediction::Method method, CpuType cpu>
class PredictKernel : public daal::algorithms::Kernel
{
public:
    /**
     *  \brief Compute gradient boosted trees prediction results.
     *
     *  \param a[in]    Matrix of input variables X
     *  \param m[in]    gradient boosted trees model obtained on training stage
     *  \param r[out]   Prediction results
     *  \param nIterations[in]  Number of iterations to predict in gradient boosted trees algorithm parameter
     *  \param predShapContributions[in] Predict SHAP contributions
     *  \param predShapInteractions[in] Predict SHAP interactions
     */
    services::Status compute(services::HostAppIface * pHostApp, const NumericTable * a, const regression::Model * m, NumericTable * r,
                             size_t nIterations, bool predShapContributions, bool predShapInteractions);
};

} // namespace internal
} // namespace prediction
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
