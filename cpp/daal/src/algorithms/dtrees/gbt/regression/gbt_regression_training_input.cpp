/* file: gbt_regression_training_input.cpp */
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
//  Implementation of gradient boosted trees algorithm classes.
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_training_types.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
Status checkImpl(const gbt::training::Parameter & prm);
}

namespace regression
{
namespace training
{
namespace interface1
{
Parameter::Parameter() : loss(squared), varImportance(0) {}
Status Parameter::check() const
{
    return gbt::training::checkImpl(*this);
}

/** Default constructor */
Input::Input() : algorithms::regression::training::Input(lastInputId + 1) {}
Input::Input(const Input & other)             = default;
Input & Input::operator=(const Input & other) = default;

/**
 * Returns an input object for gradient boosted trees model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return algorithms::regression::training::Input::get(algorithms::regression::training::InputId(id));
}

/**
 * Sets an input object for gradient boosted trees model-based training
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr & value)
{
    algorithms::regression::training::Input::set(algorithms::regression::training::InputId(id), value);
}

/**
* Checks an input object for the gradient boosted trees algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/

Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, algorithms::regression::training::Input::check(par, method));
    NumericTablePtr dataTable              = get(data);
    NumericTablePtr dependentVariableTable = get(dependentVariable);

    DAAL_CHECK_EX(dataTable.get(), ErrorNullInputNumericTable, ArgumentName, dataStr());
    DAAL_CHECK_EX(dependentVariableTable->getNumberOfColumns() == 1, ErrorIncorrectNumberOfColumns, ArgumentName, dependentVariableStr());

    const Parameter * parameter = static_cast<const Parameter *>(par);
    const size_t nSamplesPerTree(parameter->observationsPerTreeFraction * dataTable->getNumberOfRows());
    DAAL_CHECK_EX(nSamplesPerTree > 0, ErrorIncorrectParameter, ParameterName, observationsPerTreeFractionStr());
    const auto nFeatures = dataTable->getNumberOfColumns();
    DAAL_CHECK_EX(parameter->featuresPerNode <= nFeatures, ErrorIncorrectParameter, ParameterName, featuresPerNodeStr());
    return s;
}

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
