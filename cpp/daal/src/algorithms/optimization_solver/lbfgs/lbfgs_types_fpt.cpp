/* file: lbfgs_types_fpt.cpp */
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
//  Implementation of lbfgs solver classes.
//--
*/

#include "algorithms/optimization_solver/lbfgs/lbfgs_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace lbfgs
{
namespace interface2
{
/**
* Allocates memory to store the results of the iterative solver algorithm
* \param[in] input  Pointer to the input structure
* \param[in] par    Pointer to the parameter structure
* \param[in] method Computation method of the algorithm
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    services::Status s = super::allocate<algorithmFPType>(input, par, method);
    DAAL_CHECK_STATUS_VAR(s);
    const Parameter * algParam = static_cast<const Parameter *>(par);
    if (!algParam->optionalResultRequired) return services::Status();
    algorithms::OptionalArgumentPtr pOpt = get(iterative_solver::optionalResult);
    if (pOpt.get())
    {
        if (pOpt->size() != lastOptionalData + 1) return services::Status(); //error, will be found in check
    }
    else
    {
        pOpt = algorithms::OptionalArgumentPtr(new algorithms::OptionalArgument(lastOptionalData + 1));
        Argument::set(iterative_solver::optionalResult, pOpt);
    }
    const Input * algInput                = static_cast<const Input *>(input);
    const size_t argumentSize             = algInput->get(iterative_solver::inputArgument)->getNumberOfRows();
    data_management::NumericTablePtr pTbl = data_management::NumericTable::cast(pOpt->get(correctionPairs));
    if (!pTbl.get())
    {
        pTbl = data_management::NumericTablePtr(
            new data_management::HomogenNumericTable<algorithmFPType>(argumentSize, 2 * algParam->m, data_management::NumericTable::doAllocate, 0.0));
        pOpt->set(correctionPairs, pTbl);
    }
    pTbl = data_management::NumericTable::cast(pOpt->get(correctionIndices));
    if (!pTbl.get())
    {
        pTbl = data_management::NumericTablePtr(new data_management::HomogenNumericTable<int>(2, 1, data_management::NumericTable::doAllocate, 0));
        pOpt->set(correctionIndices, pTbl);
    }
    pTbl = data_management::NumericTable::cast(pOpt->get(averageArgumentLIterations));
    if (!pTbl.get())
    {
        pTbl = data_management::NumericTablePtr(
            new data_management::HomogenNumericTable<algorithmFPType>(argumentSize, 2, data_management::NumericTable::doAllocate, 0.0));
        pOpt->set(averageArgumentLIterations, pTbl);
    }

    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace interface2
} // namespace lbfgs
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
