/* file: gbt_regression_training_types.h */
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
//  Implementation of the gradient boosted trees regression training algorithm interface
//--
*/

#ifndef __GBT_REGRESSION_TRAINIG_TYPES_H__
#define __GBT_REGRESSION_TRAINIG_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_model.h"
#include "algorithms/gradient_boosted_trees/gbt_training_parameter.h"
#include "algorithms/regression/regression_training_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the gradient boosted trees algorithm
 */
namespace gbt
{
namespace regression
{
/**
 * @defgroup gbt_regression_training Training
 * \copydoc daal::algorithms::gbt::regression::training
 * @ingroup gbt_regression
 * @{
 */
/**
 * \brief Contains a class for model-based training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__METHOD"></a>
 * \brief Computation methods for gradient boosted trees classification model-based training
 */
enum Method
{
    xboost       = 0, /*!< Extreme boosting (second-order approximation of objective function,
                           regularization on number of leaves and their weights), Chen et al. */
    defaultDense = 0  /*!< Default training method */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__LOSS_FUNCTION_TYPE"></a>
* \brief Loss function type
*/
enum LossFunctionType
{
    squared, /* L(y,f) = ([y-f(x)]^2)/2 */
    custom   /* Should be differentiable up to the second order */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__INPUTID"></a>
 * \brief Available identifiers of input objects for model-based training
 */
enum InputId
{
    data              = algorithms::regression::training::data,               /*!< %Input data table */
    dependentVariable = algorithms::regression::training::dependentVariables, /*!< %Values of the dependent variable for the input data */
    lastInputId       = dependentVariable
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__REGRESSION__TRAINING__RESULTID"></a>
 * \brief Available identifiers of the result of model-based training
 */
enum ResultId
{
    model        = algorithms::regression::training::model, /*!< model */
    lastResultId = model
};

enum ResultNumericTableId
{
    variableImportanceByWeight = lastResultId + 1,
    variableImportanceByTotalCover,
    variableImportanceByCover,
    variableImportanceByTotalGain,
    variableImportanceByGain,
    lastResultNumericTableId = variableImportanceByGain
};

/**
 * \brief Contains version 1.0 of the Intel(R) oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__PARAMETER"></a>
 * \brief Parameters for the gradient boosted trees algorithm
 *
 * \snippet gradient_boosted_trees/gbt_regression_training_types.h Parameter source code
 */
/* [Parameter source code] */
class DAAL_EXPORT Parameter : public daal::algorithms::Parameter, public daal::algorithms::gbt::training::Parameter
{
public:
    Parameter();
    services::Status check() const DAAL_C11_OVERRIDE;

    LossFunctionType loss;     /*!< Loss function type */
    DAAL_UINT64 varImportance; /*!< 64 bit integer flag VariableImportanceModes that indicates the variable importance computation modes */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__INPUT"></a>
 * \brief %Input objects for model-based training
 */
class DAAL_EXPORT Input : public algorithms::regression::training::Input
{
public:
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input & other);

    Input & operator=(const Input & other);

    virtual ~Input() {};

    /**
     * Returns an input object for model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets an input object for model-based training
     * \param[in] id      Identifier of the input object
     * \param[in] value   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr & value);

    /**
    * Checks an input object for the gradient boosted trees algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    * \return Status of checking
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSSION__TRAINING__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of model-based training
 */
class DAAL_EXPORT Result : public algorithms::regression::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    /**
     * Allocates memory to store the result of model-based training
     * \param[in] input Pointer to an object containing the input data
     * \param[in] method Computation method for the algorithm
     * \param[in] parameter %Parameter of model-based training
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const Parameter * parameter, const int method);

    /**
     * Returns the result of model-based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    gbt::regression::ModelPtr get(ResultId id) const;

    /**
     * Sets the result of model-based training
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultId id, const ModelPtr & value);

    /**
     * Returns the result of model-based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultNumericTableId id) const;

    /**
     * Sets the result of model-based training
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultNumericTableId id, const data_management::NumericTablePtr & value);

    /**
     * Checks the result of model-based training
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     * \return Status of checking
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
using interface1::Parameter;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace training
/** @} */
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
#endif
