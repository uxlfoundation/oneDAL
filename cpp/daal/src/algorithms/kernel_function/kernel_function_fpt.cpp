/* file: kernel_function_fpt.cpp */
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
//  Implementation of kernel function algorithm and types methods.
//--
*/

#include "src/algorithms/kernel_function/kernel_function_types.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace interface1
{
/**
 * Allocates memory to store results of the kernel function algorithm
 * \param[in] input  Pointer to the structure with the input objects
 * \param[in] par    Pointer to the structure of the algorithm parameters
 * \param[in] method       Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    const Input * algInput = static_cast<const Input *>(input);

    const size_t nVectors1 = algInput->get(X)->getNumberOfRows();
    const size_t nVectors2 = algInput->get(Y)->getNumberOfRows();

    services::Status status;

    set(values,
        data_management::HomogenNumericTable<algorithmFPType>::create(nVectors2, nVectors1, data_management::NumericTable::doAllocate, &status));

    return status;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace interface1
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
