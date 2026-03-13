/* file: df_regression_predict_dense_default_batch.h */
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
//  Declaration of template function that computes decision forest
//  prediction results.
//--
*/

#ifndef __DF_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__
#define __DF_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__

#include "algorithms/decision_forest/decision_forest_regression_predict.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace prediction
{
namespace internal
{
template <typename algorithmFpType, decision_forest::regression::prediction::Method method, CpuType cpu>
class PredictKernel : public daal::algorithms::Kernel
{
public:
    /**
     *  \brief Compute decision forest regression prediction results.
     *
     *  \param a[in]    Matrix of input variables X of size [N x P], where N is a number of observations
     *                  and P is a number of features.
     *  \param m[in]    decision forest model obtained on training stage
     *  \param r[out]   Prediction results stored in the numeric table of size [N x 1]
     */
    services::Status compute(const NumericTable * a, const regression::Model * m, NumericTable * r);
};

} // namespace internal
} // namespace prediction
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
