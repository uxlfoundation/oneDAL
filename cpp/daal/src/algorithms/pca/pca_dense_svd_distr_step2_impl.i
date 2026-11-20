/* file: pca_dense_svd_distr_step2_impl.i */
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
//  Functuons that are used in PCA algorithm
//--
*/

#ifndef __PCA_DENSE_SVD_DISTR_STEP2_IMPL_I__
#define __PCA_DENSE_SVD_DISTR_STEP2_IMPL_I__

#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
using namespace daal::internal;
using namespace daal::data_management;

template <typename algorithmFPType, CpuType cpu>
services::Status PCASVDStep2MasterKernel<algorithmFPType, cpu>::finalizeMerge(InputDataType type, const DataCollectionPtr & inputPartialResults,
                                                                              NumericTable & eigenvalues, NumericTable & eigenvectors)
{
    if (type == correlation) return services::Status(services::ErrorInputCorrelationNotSupportedInOnlineAndDistributed);

    size_t nObservations = 0;

    size_t nPartialResults = inputPartialResults->size();
    DataCollection rTables;

    for (size_t i = 0; i < nPartialResults; i++)
    {
        services::SharedPtr<PartialResult<svdDense> > partialRes =
            services::staticPointerCast<PartialResult<svdDense>, SerializationIface>(inputPartialResults->get(i));

        size_t nBlocks = partialRes->get(pca::auxiliaryData)->size();
        for (size_t j = 0; j < nBlocks; j++)
        {
            rTables.push_back(partialRes->get(pca::auxiliaryData, j));
        }

        NumericTable * nCurrentObservationsTable = partialRes->get(pca::nObservationsSVD).get();
        nObservations += nCurrentObservationsTable->getValue<int>(0, 0);
    }

    svd::Parameter kmPar;
    kmPar.leftSingularMatrix = svd::notRequired;

    const size_t nPartialBlocks = rTables.size();
    const size_t nInputs        = nPartialBlocks * 2;
    TArray<NumericTable *, cpu> svdInputs(nInputs);
    DAAL_CHECK_MALLOC(svdInputs.get());
    for (size_t i = 0; i < nPartialBlocks; i++)
    {
        svdInputs[i]                  = static_cast<NumericTable *>(rTables[i].get());
        svdInputs[i + nPartialBlocks] = 0;
    }

    const size_t nResults               = 3;
    NumericTable * svdResults[nResults] = { &eigenvalues, nullptr, &eigenvectors };

    daal::algorithms::svd::internal::SVDOnlineKernel<algorithmFPType, svd::defaultDense, cpu> svdKernel;
    services::Status s = svdKernel.finalizeCompute(nInputs, svdInputs.get(), nResults, svdResults, &kmPar);

    if (s) s = this->scaleSingularValues(eigenvalues, nObservations);
    return s;
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
