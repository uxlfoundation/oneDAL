/* file: apriori.cpp */
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
//  Implementation of the interface for the association rules algorithm
//--
*/

#include "algorithms/association_rules/apriori_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_ASSOCIATION_RULES_RESULT_ID);

Parameter::Parameter(double minSupport, double minConfidence, size_t nUniqueItems, size_t nTransactions, bool discoverRules,
                     ItemsetsOrder itemsetsOrder, RulesOrder rulesOrder, size_t minSize, size_t maxSize)
    : minSupport(minSupport),
      minConfidence(minConfidence),
      nUniqueItems(nUniqueItems),
      nTransactions(nTransactions),
      discoverRules(discoverRules),
      itemsetsOrder(itemsetsOrder),
      rulesOrder(rulesOrder),
      minItemsetSize(minSize),
      maxItemsetSize(maxSize)
{}

/**
 * Checks parameters of the association rules algorithm
 */
Status Parameter::check() const
{
    DAAL_CHECK_EX(minSupport >= 0 && minSupport < 1, ErrorIncorrectParameter, ParameterName, minSupportStr());
    DAAL_CHECK_EX(minConfidence >= 0 && minConfidence < 1, ErrorIncorrectParameter, ParameterName, minConfidenceStr());
    DAAL_CHECK_EX(minItemsetSize <= maxItemsetSize, ErrorIncorrectParameter, ParameterName, minItemsetSizeStr());
    return Status();
}

Input::Input() : daal::algorithms::Input(lastInputId + 1) {}

/**
 * Returns the input object of the association rules algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the input object of the association rules algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks parameters of the association rules algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method of the algorithm
 */
Status Input::check(const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == (lastInputId + 1), ErrorIncorrectNumberOfInputNumericTables);

    const size_t nColumns          = 2;
    const size_t nRows             = get(data)->getNumberOfRows();
    const Parameter * algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(data).get(), dataStr(), 0, 0, nColumns));
    DAAL_CHECK_EX(algParameter->nUniqueItems <= nRows, ErrorIncorrectParameter, ParameterName, nUniqueItemsStr());
    DAAL_CHECK_EX(algParameter->nTransactions <= nRows, ErrorIncorrectParameter, ParameterName, nTransactionsStr());
    DAAL_CHECK_EX(
        (algParameter->maxItemsetSize <= nRows) && (algParameter->nUniqueItems <= 0 || algParameter->maxItemsetSize <= algParameter->nUniqueItems),
        ErrorIncorrectParameter, ParameterName, maxItemsetSizeStr());
    return s;
}

Result::Result() : daal::algorithms::Result((lastResultId + 1)) {}

/**
 * Returns the final result of the association rules algorithm
 * \param[in] id   Identifier of the result
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the final result of the association rules algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the association rules algorithm
 * \param[in] input   %Input of algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    Status s;
    const Parameter * algParameter = static_cast<const Parameter *>(par);
    const int unexpectedLayouts    = packed_mask;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(largeItemsets).get(), largeItemsetsStr(), unexpectedLayouts, 0, 2, 0, false));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(largeItemsetsSupport).get(), largeItemsetsSupportStr(), unexpectedLayouts, 0, 2, 0, false));

    if (algParameter->discoverRules)
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(get(antecedentItemsets).get(), antecedentItemsetsStr(), unexpectedLayouts, 0, 2, 0, false));
        DAAL_CHECK_STATUS(s, checkNumericTable(get(consequentItemsets).get(), consequentItemsetsStr(), unexpectedLayouts, 0, 2, 0, false));
        DAAL_CHECK_STATUS(s, checkNumericTable(get(confidence).get(), confidenceStr(), unexpectedLayouts, 0, 1, 0, false));
    }

    return s;
}

} // namespace interface1
} // namespace association_rules
} // namespace algorithms
} // namespace daal
