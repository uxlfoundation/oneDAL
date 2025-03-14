/* file: stump_regression_training_types.h */
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
//  Implementation of the interface of the decision stump training algorithm.
//--
*/

#ifndef __STUMP_REGRESSION_TRAINING_TYPES_H__
#define __STUMP_REGRESSION_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/regression/regression_training_types.h"
#include "algorithms/stump/stump_regression_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes to work with the decision stump training algorithm
 */
namespace stump
{
/**
 * \brief Contains classes of the Decision stump regression algorithm
 */
namespace regression
{
/**
 * @defgroup stump_regression_training Training
 * \copydoc daal::algorithms::stump::regression::training
 * @ingroup stump_regression
 * @{
 */
/**
 * \brief Contains classes to train the decision stump model
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__STUMP__REGRESSION__TRAINING__METHOD"></a>
 * Available methods to train the decision stump model
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__STUMP__REGRESSION__TRAINING__RESULTNUMERICTABLEID"></a>
* \brief Available identifiers of the result of  training
*/
enum ResultNumericTableId
{
    variableImportance =
        daal::algorithms::regression::training::lastResultId + 1, /*!< %Numeric table 1x(number of features) containing variable importance value.
                                                               Computed when parameter.varImportance != none */
    lastResultId = variableImportance
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__REGRESSION__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the decision stump training algorithm
 * in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::regression::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    virtual ~Result() {}

    /**
     * Returns the model trained with the Stump algorithm
     * \param[in] id    Identifier of the result, \ref daal::algorithms::regression::training::ResultId
     * \return          Model trained with the Stump algorithm
     */
    daal::algorithms::stump::regression::ModelPtr get(daal::algorithms::regression::training::ResultId id) const;

    /**
     * Sets the result of the training stage of the stump algorithm
     * \param[in] id      Identifier of the result, \ref daal::algorithms::regression::training::ResultId
     * \param[in] value   Pointer to the training result
     */
    void set(daal::algorithms::regression::training::ResultId id, daal::algorithms::stump::regression::ModelPtr & value);

    /**
    * Returns the result of the stump algorithm
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultNumericTableId id) const;

    /**
    * Sets the result of the stump algorithm
    * \param[in] id      Identifier of the result
    * \param[in] value   Result
    */
    void set(ResultNumericTableId id, const data_management::NumericTablePtr & value);

    /**
     * Allocates memory to store final results of the decision stump training algorithm
     * \tparam algorithmFPType  Data type to store prediction results
     * \param[in] input         %Input objects for the decision stump training algorithm
     * \param[in] parameter     Parameters of the decision stump training algorithm
     * \param[in] method        Decision stump training method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks the result of the decision stump training algorithm
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
} // namespace interface1
using interface1::Result;
using interface1::ResultPtr;

} // namespace training
/** @} */
} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal
#endif // __stump__regression_TRAINING_TYPES_H__
