/* file: kmeans_partialresult_types.cpp */
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

#include "algorithms/kmeans/kmeans_types.h"
#include "services/daal_defines.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_KMEANS_PARTIAL_RESULT_ID);
PartialResult::PartialResult() : daal::algorithms::PartialResult(lastPartialResultId + 1) {}

/**
 * Returns a partial result of the K-Means algorithm
 * \param[in] id   Identifier of the partial result
 * \return         Partial result that corresponds to the given identifier
 */
NumericTablePtr PartialResult::get(PartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets a partial result of the K-Means algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the object
 */
void PartialResult::set(PartialResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns the number of features in the Input data table
* \return Number of features in the Input data table
*/

size_t PartialResult::getNumberOfFeatures() const
{
    NumericTablePtr sums = get(partialSums);
    return sums->getNumberOfColumns();
}

/**
* Checks partial results of the K-Means algorithm
* \param[in] input   %Input object of the algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/
services::Status PartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    const interface2::Parameter * kmPar2 = dynamic_cast<const interface2::Parameter *>(par);
    if (kmPar2 == nullptr) return services::Status(daal::services::ErrorNullParameterNotSupported);

    size_t inputFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
    size_t nClusters     = kmPar2->nClusters;

    const int unexpectedLayouts = (int)packed_mask;

    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(nObservations).get(), nObservationsStr(), unexpectedLayouts, 0, 1, nClusters));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(partialSums).get(), partialSumsStr(), unexpectedLayouts, 0, inputFeatures, nClusters));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(partialObjectiveFunction).get(), partialObjectiveFunctionStr(), unexpectedLayouts, 0, 1, 1));
    DAAL_CHECK_STATUS(s,
                      checkNumericTable(get(partialCandidatesDistances).get(), partialCandidatesDistancesStr(), unexpectedLayouts, 0, 2, nClusters));
    DAAL_CHECK_STATUS(
        s, checkNumericTable(get(partialCandidatesCentroids).get(), partialCandidatesCentroidsStr(), unexpectedLayouts, 0, inputFeatures, nClusters));
    if (kmPar2)
    {
        if (kmPar2->resultsToEvaluate & computeAssignments || kmPar2->assignFlag)
        {
            Input * algInput = dynamic_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
            if (!algInput)
            {
                return s;
            }
            const size_t nRows = algInput->get(data)->getNumberOfRows();
            s                  = checkNumericTable(get(partialAssignments).get(), partialAssignmentsStr(), unexpectedLayouts, 0, 1, nRows);
        }
    }
    return s;
}

/**
 * Checks partial results of the K-Means algorithm
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Parameter * par, int method) const
{
    const interface2::Parameter * kmPar = static_cast<const interface2::Parameter *>(par);
    const int unexpectedLayouts         = (int)packed_mask;
    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(nObservations).get(), nObservationsStr(), unexpectedLayouts, 0, 1, kmPar->nClusters));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(partialSums).get(), partialSumsStr(), unexpectedLayouts, 0, 0, kmPar->nClusters));
    return s;
}

} // namespace interface1
} // namespace kmeans
} // namespace algorithms
} // namespace daal
