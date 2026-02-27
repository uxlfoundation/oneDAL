/* file: moments_online.h */
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
//  Implementation of LowOrderMoments algorithm and types methods
//--
*/
#ifndef __MOMENTS_ONLINE__
#define __MOMENTS_ONLINE__

#include "algorithms/moments/low_order_moments_types.h"
#include "src/data_management/service_numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

using namespace daal::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
/**
 * Allocates memory to store partial results of the low order %moments algorithm
 * \param[in] input     Pointer to the structure with input objects
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                     const int method)
{
    services::Status s;
    size_t nFeatures = 0;
    DAAL_CHECK_STATUS(s, static_cast<const InputIface *>(input)->getNumberOfColumns(nFeatures));

    set(nObservations, HomogenNumericTable<size_t>::create(1, 1, NumericTable::doAllocate, &s));
    for (size_t i = 1; i < lastPartialResultId + 1; i++)
    {
        Argument::set(i, HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, NumericTable::doAllocate, &s));
    }
    return s;
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::initialize(const daal::algorithms::Input * _in, const daal::algorithms::Parameter * parameter,
                                                       const int method)
{
    Input * input = const_cast<Input *>(static_cast<const Input *>(_in));

    services::Status s;

    DAAL_CHECK_STATUS(s, get(nObservations)->assign((algorithmFPType)0.0))

    DAAL_CHECK_STATUS(s, get(partialSum)->assign((algorithmFPType)0.0))
    DAAL_CHECK_STATUS(s, get(partialSumSquares)->assign((algorithmFPType)0.0))
    DAAL_CHECK_STATUS(s, get(partialSumSquaresCentered)->assign((algorithmFPType)0.0))

    ReadRows<algorithmFPType, DAAL_BASE_CPU> dataBlock(input->get(data).get(), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(dataBlock)
    const algorithmFPType * firstRow = dataBlock.get();

    WriteOnlyRows<algorithmFPType, DAAL_BASE_CPU> partialMinimumBlock(get(partialMinimum).get(), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(partialMinimumBlock)
    algorithmFPType * partialMinimumArray = partialMinimumBlock.get();

    WriteOnlyRows<algorithmFPType, DAAL_BASE_CPU> partialMaximumBlock(get(partialMaximum).get(), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(partialMaximumBlock)
    algorithmFPType * partialMaximumArray = partialMaximumBlock.get();

    size_t nColumns = input->get(data)->getNumberOfColumns();

    for (size_t j = 0; j < nColumns; j++)
    {
        partialMinimumArray[j] = firstRow[j];
        partialMaximumArray[j] = firstRow[j];
    }

    return s;
}

} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
