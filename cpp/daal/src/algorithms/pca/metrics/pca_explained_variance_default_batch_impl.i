/* file: pca_explained_variance_default_batch_impl.i */
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

#ifndef __PCA_EXPLAINED_VARIANCE_DEFAULT_BATCH_IMPL_I__
#define __PCA_EXPLAINED_VARIANCE_DEFAULT_BATCH_IMPL_I__

#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "src/externals/service_lapack.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_data_utils.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace quality_metric
{
namespace explained_variance
{
namespace internal
{
template <Method method, typename algorithmFPType, CpuType cpu>
Status ExplainedVarianceKernel<method, algorithmFPType, cpu>::compute(const NumericTable & eigenvalues, NumericTable & explainedVariances,
                                                                      NumericTable & explainedVariancesRatios, NumericTable & noiseVariance)
{
    ReadRows<algorithmFPType, cpu, NumericTable> rowsEigenvalues(const_cast<NumericTable &>(eigenvalues), 0, 1);
    const algorithmFPType * pEigenvalues = rowsEigenvalues.get();

    DEFINE_TABLE_BLOCK(WriteOnlyRows, rowsExplainedVariances, &explainedVariances);
    DEFINE_TABLE_BLOCK(WriteOnlyRows, rowsExplainedVariancesRatios, &explainedVariancesRatios);
    DEFINE_TABLE_BLOCK(WriteOnlyRows, rowsNoiseVariance, &noiseVariance);

    algorithmFPType * pExplainedVariances       = rowsExplainedVariances.get();
    algorithmFPType * pExplainedVariancesRatios = rowsExplainedVariancesRatios.get();
    algorithmFPType * pNoiseVariance            = rowsNoiseVariance.get();

    size_t nFeatures   = eigenvalues.getNumberOfColumns();
    size_t nComponents = explainedVariances.getNumberOfColumns();

    algorithmFPType sum      = 0;
    algorithmFPType noiseSum = 0;

    for (size_t id = 0; id < nFeatures; ++id)
    {
        sum += pEigenvalues[id];
        if (id >= nComponents) noiseSum += pEigenvalues[id];
    }

    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t id = 0; id < nComponents; ++id)
    {
        pExplainedVariances[id]       = pEigenvalues[id];
        pExplainedVariancesRatios[id] = pEigenvalues[id] / sum;
    }

    int delta         = nFeatures - nComponents;
    pNoiseVariance[0] = (delta > 0) ? noiseSum / delta : 0;

    return Status();
}

} // namespace internal
} // namespace explained_variance
} // namespace quality_metric
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
