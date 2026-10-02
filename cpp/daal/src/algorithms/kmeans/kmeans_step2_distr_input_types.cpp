/* file: kmeans_step2_distr_input_types.cpp */
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
DistributedStep2MasterInput::DistributedStep2MasterInput() : InputIface(lastMasterInputId + 1)
{
    Argument::set(partialResults, DataCollectionPtr(new DataCollection()));
}

/**
* Returns an input object for the K-Means algorithm in the second step of the distributed processing mode
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
DataCollectionPtr DistributedStep2MasterInput::get(MasterInputId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

/**
* Sets an input object for the K-Means algorithm in the second step of the distributed processing mode
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the object
*/
void DistributedStep2MasterInput::set(MasterInputId id, const DataCollectionPtr & ptr)
{
    Argument::set(id, staticPointerCast<SerializationIface, DataCollection>(ptr));
}

/**
 * Adds partial results computed on local nodes to the input for the K-Means algorithm
 * in the second step of the distributed processing mode
 * \param[in] id    Identifier of the input object
 * \param[in] value Pointer to the object
 */
void DistributedStep2MasterInput::add(MasterInputId id, const PartialResultPtr & value)
{
    DataCollectionPtr collection = staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
    collection->push_back(value);
}

/**
* Returns the number of features in the Input data table in the second step of the distributed processing mode
* \return Number of features in the Input data table
*/
size_t DistributedStep2MasterInput::getNumberOfFeatures() const
{
    DataCollectionPtr collection = get(partialResults);
    PartialResultPtr pres        = staticPointerCast<PartialResult, SerializationIface>((*collection)[0]);
    return pres->getNumberOfFeatures();
}

/**
* Checks an input object for the K-Means algorithm in the second step of the distributed processing mode
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/
services::Status DistributedStep2MasterInput::check(const daal::algorithms::Parameter * par, int method) const
{
    const interface2::Parameter * kmPar2 = dynamic_cast<const interface2::Parameter *>(par);
    if (kmPar2 == nullptr) return services::Status(daal::services::ErrorNullParameterNotSupported);

    size_t nClusters = kmPar2->nClusters;

    DataCollectionPtr collection = get(partialResults);
    DAAL_CHECK(collection, ErrorNullInputDataCollection);

    size_t nBlocks = collection->size();
    DAAL_CHECK(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables);

    PartialResultPtr firstPres = dynamicPointerCast<PartialResult, SerializationIface>((*collection)[0]);
    DAAL_CHECK(firstPres, ErrorIncorrectElementInPartialResultCollection);

    const int unexpectedLayouts = (int)packed_mask;
    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(firstPres->get(nObservations).get(), nObservationsStr(), unexpectedLayouts, 0, 1, nClusters));
    DAAL_CHECK_STATUS(s, checkNumericTable(firstPres->get(partialSums).get(), partialSumsStr(), unexpectedLayouts, 0, 0, nClusters));
    DAAL_CHECK_STATUS(s,
                      checkNumericTable(firstPres->get(partialObjectiveFunction).get(), partialObjectiveFunctionStr(), unexpectedLayouts, 0, 1, 1));

    const size_t inputFeatures = firstPres->get(partialSums)->getNumberOfColumns();

    DAAL_CHECK_STATUS(
        s, checkNumericTable(firstPres->get(partialCandidatesDistances).get(), partialCandidatesDistancesStr(), unexpectedLayouts, 0, 2, nClusters));
    DAAL_CHECK_STATUS(s, checkNumericTable(firstPres->get(partialCandidatesCentroids).get(), partialCandidatesCentroidsStr(), unexpectedLayouts, 0,
                                           inputFeatures, nClusters));

    if (kmPar2)
    {
        if (kmPar2->resultsToEvaluate & computeAssignments || kmPar2->assignFlag)
        {
            DAAL_CHECK_STATUS(s, checkNumericTable(firstPres->get(partialAssignments).get(), partialAssignmentsStr(), unexpectedLayouts, 0, 1));
        }
    }

    for (size_t i = 1; i < nBlocks; i++)
    {
        PartialResultPtr pres = dynamicPointerCast<PartialResult, SerializationIface>((*collection)[i]);
        DAAL_CHECK(pres, ErrorIncorrectElementInPartialResultCollection);

        DAAL_CHECK_STATUS(s, checkNumericTable(pres->get(nObservations).get(), nObservationsStr(), unexpectedLayouts, 0, 1, nClusters));
        DAAL_CHECK_STATUS(s, checkNumericTable(pres->get(partialSums).get(), partialSumsStr(), unexpectedLayouts, 0, inputFeatures, nClusters));
        DAAL_CHECK_STATUS(s, checkNumericTable(pres->get(partialObjectiveFunction).get(), partialObjectiveFunctionStr(), unexpectedLayouts, 0, 1, 1));
        DAAL_CHECK_STATUS(s, checkNumericTable(firstPres->get(partialCandidatesDistances).get(), partialCandidatesDistancesStr(), unexpectedLayouts,
                                               0, 2, nClusters));
        DAAL_CHECK_STATUS(s, checkNumericTable(firstPres->get(partialCandidatesCentroids).get(), partialCandidatesCentroidsStr(), unexpectedLayouts,
                                               0, inputFeatures, nClusters));

        if (kmPar2)
        {
            if (kmPar2->resultsToEvaluate & computeAssignments || kmPar2->assignFlag)
            {
                DAAL_CHECK_STATUS(s, checkNumericTable(pres->get(partialAssignments).get(), partialAssignmentsStr(), unexpectedLayouts, 0, 1));
            }
        }
    }
    return s;
}

} // namespace interface1
} // namespace kmeans
} // namespace algorithms
} // namespace daal
