/* file: linear_regression_single_beta_dense_default_batch_kernel.h */
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
//  Declaration of template class that computes linear regression quality metrics.
//--
*/

#ifndef __LINEAR_REGRESSION_SINGLE_BETA_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __LINEAR_REGRESSION_SINGLE_BETA_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "algorithms/linear_regression/linear_regression_single_beta_types.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
namespace single_beta
{
namespace internal
{
using namespace daal::data_management;
struct SingleBetaOutput
{
    NumericTable * rms;
    NumericTable * variance;
    NumericTable ** betaCovariances;
    NumericTable * zScore;
    NumericTable * confidenceIntervals;
    NumericTable * inverseOfXtX;

    SingleBetaOutput(size_t nResponses);
    SingleBetaOutput(const SingleBetaOutput & other)             = delete;
    SingleBetaOutput & operator=(const SingleBetaOutput & other) = delete;

    ~SingleBetaOutput();
};

template <Method method, typename algorithmFPType, CpuType cpu>
class SingleBetaKernel : public daal::algorithms::Kernel
{
public:
    virtual ~SingleBetaKernel() {}
    services::Status compute(const NumericTable * y, const NumericTable * z, size_t p, const NumericTable * betas, const NumericTable * xtx,
                             bool bModelNe, algorithmFPType accuracyThreshold, algorithmFPType alpha, SingleBetaOutput & out);

protected:
    static const size_t _nRowsInBlock = 1024;
    services::Status computeTestStatistics(const NumericTable * betas, const algorithmFPType * v, algorithmFPType alpha,
                                           algorithmFPType accuracyThreshold, SingleBetaOutput & out);
    services::Status computeRmsVariance(const NumericTable * y, const NumericTable * z, size_t p, NumericTable * rms, NumericTable * variance);
    services::Status computeInverseXtX(const NumericTable * xtx, bool bModelNe, NumericTable * xtxInv);
};

} // namespace internal
} // namespace single_beta
} // namespace quality_metric
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
