/* file: em_gmm_init_dense_batch_fpt.cpp */
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
//  Implementation of EMforKernel
//--
*/

#include "src/algorithms/em/em_gmm_init_dense_default_batch_kernel.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace init
{
namespace internal
{
template <typename algorithmFPType>
ErrorID EMforKernel<algorithmFPType>::run(data_management::NumericTable & inputData, data_management::NumericTable & inputWeights,
                                          data_management::NumericTable & inputMeans, data_management::DataCollectionPtr & inputCov,
                                          const em_gmm::CovarianceStorageId covType, algorithmFPType & loglikelihood)
{
    this->input.set(daal::algorithms::em_gmm::data, NumericTablePtr(&inputData, EmptyDeleter()));
    this->input.set(daal::algorithms::em_gmm::inputWeights, NumericTablePtr(&inputWeights, EmptyDeleter()));
    this->input.set(daal::algorithms::em_gmm::inputMeans, NumericTablePtr(&inputMeans, EmptyDeleter()));
    this->input.set(daal::algorithms::em_gmm::inputCovariances, inputCov);
    this->parameter.covarianceStorage = covType;

    daal::algorithms::em_gmm::ResultPtr emResult(new daal::algorithms::em_gmm::Result());
    emResult->set(daal::algorithms::em_gmm::weights, NumericTablePtr(&inputWeights, EmptyDeleter()));
    emResult->set(daal::algorithms::em_gmm::means, NumericTablePtr(&inputMeans, EmptyDeleter()));
    emResult->set(daal::algorithms::em_gmm::covariances, inputCov);

    services::Status status;
    SharedPtr<HomogenNumericTable<algorithmFPType> > loglikelihoodValueTable =
        HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, &status);
    if (!status)
    {
        return ErrorMemoryAllocationFailed;
    }

    NumericTablePtr nIterationsValueTable = HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status);
    if (!status)
    {
        return ErrorMemoryAllocationFailed;
    }
    emResult->set(daal::algorithms::em_gmm::goalFunction, loglikelihoodValueTable);
    emResult->set(daal::algorithms::em_gmm::nIterations, nIterationsValueTable);

    this->setResult(emResult);
    services::Status s = this->computeNoThrow();
    if (!s)
    {
        return ErrorEMInitNoTrialConverges;
    }
    loglikelihood = loglikelihoodValueTable->getArray()[0];
    return ErrorID(0);
}

template class EMforKernel<DAAL_FPTYPE>;

} // namespace internal
} // namespace init
} // namespace em_gmm
} // namespace algorithms
} // namespace daal
