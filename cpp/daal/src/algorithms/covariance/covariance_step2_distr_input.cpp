/* file: covariance_step2_distr_input.cpp */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#include "algorithms/covariance/covariance_types.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace interface1
{
DistributedInput<step2Master>::DistributedInput() : InputIface(lastMasterInputId + 1)
{
    Argument::set(partialResults, DataCollectionPtr(new DataCollection()));
}

DistributedInput<step2Master>::DistributedInput(const DistributedInput & other)                          = default;
DistributedInput<step2Master> & DistributedInput<step2Master>::operator=(const DistributedInput & other) = default;
DistributedInput<step2Master>::~DistributedInput()                                                       = default;

size_t DistributedInput<step2Master>::getNumberOfFeatures() const
{
    DataCollectionPtr collectionOfPartialResults = staticPointerCast<DataCollection, SerializationIface>(Argument::get(partialResults));
    if (collectionOfPartialResults)
    {
        PartialResultPtr onePartialResult = staticPointerCast<PartialResult, SerializationIface>((*collectionOfPartialResults)[0]);
        if (onePartialResult.get() != NULL)
        {
            NumericTablePtr ntPtr = onePartialResult->get(sum);
            if (ntPtr)
            {
                return ntPtr->getNumberOfColumns();
            }
        }
    }
    return 0;
}

void DistributedInput<step2Master>::add(MasterInputId id, const PartialResultPtr & partialResult)
{
    DataCollectionPtr collection = staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
    collection->push_back(staticPointerCast<SerializationIface, PartialResult>(partialResult));
}

DataCollectionPtr DistributedInput<step2Master>::get(MasterInputId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(partialResults));
}

services::Status DistributedInput<step2Master>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DataCollectionPtr collection = staticPointerCast<DataCollection, SerializationIface>(Argument::get(partialResults));
    DAAL_CHECK_EX(collection, ErrorNullInputDataCollection, ArgumentName, partialResultsStr());

    size_t nBlocks = collection->size();
    DAAL_CHECK_EX(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, partialResultsStr());

    int packedLayouts = packed_mask;
    int csrLayout     = (int)NumericTableIface::csrArray;
    int crossProductUnexpectedLayout =
        (int)NumericTableIface::csrArray | (int)NumericTableIface::upperPackedTriangularMatrix | (int)NumericTableIface::lowerPackedTriangularMatrix;

    services::Status s;
    for (size_t j = 0; j < nBlocks; j++)
    {
        PartialResultPtr partialResult = staticPointerCast<PartialResult, SerializationIface>((*collection)[j]);
        DAAL_CHECK_EX(partialResult, ErrorIncorrectElementInPartialResultCollection, ArgumentName, partialResultsStr());

        /* Check partial number of observations */
        NumericTable * nObservationsTable = static_cast<NumericTable *>(partialResult->get(nObservations).get());
        s |= checkNumericTable(nObservationsTable, nObservationsStr(), csrLayout, 0, 1, 1);
        if (!s) return s;

        size_t nFeatures = getNumberOfFeatures();
        /* Check partial cross-products */
        NumericTable * crossProductTable = static_cast<NumericTable *>(partialResult->get(crossProduct).get());

        s |= checkNumericTable(crossProductTable, crossProductStr(), crossProductUnexpectedLayout, 0, nFeatures, nFeatures);
        if (!s) return s;

        /* Check partial sums */
        NumericTable * sumTable = static_cast<NumericTable *>(partialResult->get(sum).get());
        s |= checkNumericTable(sumTable, sumStr(), packedLayouts, 0, nFeatures, 1);
    }
    return s;
}

} //namespace interface1
} //namespace covariance
} // namespace algorithms
} // namespace daal
