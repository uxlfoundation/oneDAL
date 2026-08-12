/* file: kmeans_partialresult.h */
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
//  Implementation of kmeans classes.
//--
*/

#ifndef __KMEANS_PARTIALRESULT_
#define __KMEANS_PARTIALRESULT_

#include "algorithms/kmeans/kmeans_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
/**
 * Allocates memory to store partial results of the K-Means algorithm
 * \param[in] input        Pointer to the structure of the input objects
 * \param[in] parameter    Pointer to the structure of the algorithm parameters
 * \param[in] method       Computation method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                     const int method)
{
    const interface2::Parameter * kmPar2 = dynamic_cast<const interface2::Parameter *>(parameter);
    if (kmPar2 == nullptr) return services::Status(daal::services::ErrorNullParameterNotSupported);

    size_t nFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
    size_t nClusters = kmPar2->nClusters;

    services::Status status;
    set(nObservations, HomogenNumericTable<algorithmFPType>::create(1, nClusters, NumericTable::doAllocate, &status));
    DAAL_CHECK_STATUS_VAR(status);
    set(partialSums, HomogenNumericTable<algorithmFPType>::create(nFeatures, nClusters, NumericTable::doAllocate, &status));
    DAAL_CHECK_STATUS_VAR(status);
    set(partialObjectiveFunction, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, &status));
    DAAL_CHECK_STATUS_VAR(status);
    // partialCandidatesDistances layout is 2 columns per row:
    //   col 0 = squared distance from the candidate row to its nearest centroid
    //   col 1 = source cluster ID (encoded in algorithmFPType) the row was
    //           assigned to on the local node, needed by step2 finalizeCompute
    //           to correct clusterS0/clusterS1 when the row is used to seed an
    //           empty cluster elsewhere. Column count is an internal detail --
    //           get(partialCandidatesDistances) still returns the same
    //           NumericTablePtr type; callers must not assume 1 column.
    set(partialCandidatesDistances, HomogenNumericTable<algorithmFPType>::create(2, nClusters, NumericTable::doAllocate, &status));
    DAAL_CHECK_STATUS_VAR(status);
    set(partialCandidatesCentroids, HomogenNumericTable<algorithmFPType>::create(nFeatures, nClusters, NumericTable::doAllocate, &status));
    DAAL_CHECK_STATUS_VAR(status);

    const Input * step1Input = dynamic_cast<const Input *>(input);

    if (kmPar2)
    {
        if ((kmPar2->resultsToEvaluate & computeAssignments || kmPar2->assignFlag) && step1Input)
        {
            const size_t nRows = step1Input->get(data)->getNumberOfRows();
            set(partialAssignments, HomogenNumericTable<int>::create(1, nRows, NumericTable::doAllocate, &status));
        }
    }

    return status;
}

} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
