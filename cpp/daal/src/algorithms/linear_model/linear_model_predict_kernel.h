/* file: linear_model_predict_kernel.h */
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
//  Declaration of template function that computes linear regression
//  prediction results.
//--
*/

#ifndef __LINEAR_MODEL_PREDICT_KERNEL_H__
#define __LINEAR_MODEL_PREDICT_KERNEL_H__

#include "algorithms/linear_model/linear_model_predict.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/externals/service_blas.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace prediction
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;

template <typename algorithmFpType, prediction::Method method, CpuType cpu>
class PredictKernel : public daal::algorithms::Kernel
{
public:
    /**
     *  \brief Compute linear regression prediction results.
     *
     *  \param a[in]    Matrix of input variables X
     *  \param m[in]    Linear regression model obtained on training stage
     *  \param r[out]   Prediction results
     */
    services::Status compute(const NumericTable * a, const linear_model::Model * m, NumericTable * r);
};

template <typename algorithmFpType, CpuType cpu>
class PredictKernel<algorithmFpType, defaultDense, cpu> : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable * a, const linear_model::Model * m, NumericTable * r);
    services::Status compute_impl(const NumericTable * a, const NumericTable * b, NumericTable * r, bool intercept_flag);

protected:
    services::Status computeBlockOfResponses(size_t startRow, size_t numRows, NumericTable * dataTable, size_t numBetas, const algorithmFpType * beta,
                                             size_t numResponses, algorithmFpType * responseBlock, bool findBeta0);
    services::Status computeBlockOfResponsesSOA(size_t startRow, size_t numRows, NumericTable * dataTable, size_t numBetas,
                                                const algorithmFpType * beta, size_t numResponses, algorithmFpType * responseBlock, bool findBeta0,
                                                bool isHomogeneous, TlsMem<algorithmFpType, cpu> & tlsData);
    static const size_t blockSizeRows    = 512;
    static const size_t blockSizeColumns = 512;
};

} // namespace internal
} // namespace prediction
} // namespace linear_model
} // namespace algorithms
} // namespace daal

#endif
