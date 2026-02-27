/* file: implicit_als_train_result.cpp */
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "algorithms/implicit_als/implicit_als_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_IMPLICIT_ALS_MODEL_ID);
}

namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_IMPLICIT_ALS_TRAINING_RESULT_ID);

/** Default constructor */
Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
daal::algorithms::implicit_als::ModelPtr Result::get(ResultId id) const
{
    return services::staticPointerCast<daal::algorithms::implicit_als::Model, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const daal::algorithms::implicit_als::ModelPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the implicit ALS training algorithm
 * \param[in] input       %Input object for the algorithm
 * \param[in] parameter   %Parameter of the algorithm
 * \param[in] method      Computation method of the algorithm
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    const Input * algInput    = static_cast<const Input *>(input);
    NumericTablePtr dataTable = algInput->get(data);
    const size_t nUsers       = dataTable->getNumberOfRows();
    const size_t nItems       = dataTable->getNumberOfColumns();

    const Parameter * algParameter = static_cast<const Parameter *>(parameter);
    const size_t nFactors          = algParameter->nFactors;

    ModelPtr trainedModel = get(model);
    DAAL_CHECK(trainedModel, ErrorNullModel);

    const int unexpectedLayouts = (int)packed_mask;
    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(trainedModel->getUsersFactors().get(), usersFactorsStr(), unexpectedLayouts, 0, nFactors, nUsers));
    DAAL_CHECK_STATUS(s, checkNumericTable(trainedModel->getItemsFactors().get(), itemsFactorsStr(), unexpectedLayouts, 0, nFactors, nItems));
    return s;
}

} // namespace interface1
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
