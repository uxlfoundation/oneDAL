/* file: logistic_regression_predict_kernel.h */
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
//  Declaration of template function that computes logistic regression
//  prediction results.
//--
*/

#ifndef __LOGISTIC_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__
#define __LOGISTIC_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_H__

#include "algorithms/logistic_regression/logistic_regression_predict.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace prediction
{
namespace internal
{
template <typename algorithmFpType, logistic_regression::prediction::Method method, CpuType cpu>
class PredictKernel : public daal::algorithms::Kernel
{
public:
    /**
     *  \brief Compute logistic regression prediction results.
     *
     *  \param a[in]   Matrix of input variables X
     *  \param m[in]   Logistic regression model obtained on training stage
     *  \param nClasses[in] Number of classes in logistic regression algorithm parameter
     *  \param pRes[out] Prediction results
     *  \param pProbab[out] Probability prediction results
     *  \param pLogProbab[out] Log of probability prediction results
     */
    services::Status compute(const NumericTable * a, const logistic_regression::Model * m, size_t nClasses, NumericTable * pRes,
                             NumericTable * pProbab, NumericTable * pLogProbab);
};

} // namespace internal
} // namespace prediction
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal

#endif
