/* file: cosdistance_fpt.cpp */
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
//  Implementation of cosine distance algorithm and types methods.
//--
*/

#include "algorithms/distance/cosine_distance_types.h"

namespace daal
{
namespace algorithms
{
namespace cosine_distance
{
namespace interface1
{
/**
 * Allocates memory to store results of the cosine distance algorithm
 * \param[in] input  Pointer to input structure
 * \param[in] par    Pointer to parameter structure
 * \param[in] method Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    Input * algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    size_t dim       = algInput->get(data)->getNumberOfRows();
    services::Status status;
    Argument::set(cosineDistance, data_management::NumericTablePtr(data_management::HomogenNumericTable<algorithmFPType>::create(
                                      dim, dim, data_management::NumericTable::doAllocate, &status)));
    return status;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace interface1
} // namespace cosine_distance
} // namespace algorithms
} // namespace daal
