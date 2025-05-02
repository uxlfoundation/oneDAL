/* file: linear_model_training_types.h */
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
//  Implementation of the regression algorithm interface
//--
*/

#ifndef __LINEAR_MODEL_TRAINING_TYPES_H__
#define __LINEAR_MODEL_TRAINING_TYPES_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_types.h"
#include "algorithms/regression/regression_training_types.h"
#include "algorithms/linear_model/linear_model_model.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
/**
 * @defgroup linear_model_training Training
 * \copydoc daal::algorithms::linear_model::training
 * @ingroup linear_model
 * @{
 */
/**
 * \brief Contains a class for regression model-based training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_MODEL__TRAINING__INPUTID"></a>
 * \brief Available identifiers of input objects for regression model-based training
 */
enum InputId
{
    data               = regression::training::data,               /*!< %Input data table */
    dependentVariables = regression::training::dependentVariables, /*!< Values of the dependent variable for the input data */
    lastInputId        = dependentVariables
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_MODEL__TRAINING__RESULTID"></a>
 * \brief Available identifiers of the result of regression model-based training
 */
enum ResultId
{
    model        = regression::training::model, /*!< Regression model */
    lastResultId = model
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__TRAINING__INPUT"></a>
 * \brief %Input objects for the regression model-based training
 */
class DAAL_EXPORT Input : public regression::training::Input
{
public:
    /**
     * Constructs input objects for the regression training algorithm
     * \param[in] nElements Number of input objects
     */
    Input(size_t nElements);
    Input(const Input & other);
    Input & operator=(const Input & other);

    virtual ~Input() {}

    /**
     * Returns an input object for the regression model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets an input object for linear regression model-based training
     * \param[in] id      Identifier of the input object
     * \param[in] value   Input numeric table
     */
    void set(InputId id, const data_management::NumericTablePtr & value);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__TRAINING__PARTIALRESULT"></a>
 * \brief Provides methods to access a partial result obtained with the compute() method of
 *        the linear model-based training in the online processing mode
 */
class DAAL_EXPORT PartialResult : public regression::training::PartialResult
{
public:
    DAAL_CAST_OPERATOR(PartialResult)
    /**
     * Constructs the partial results of the linear model training algorithm
     * \param[in] nElements Number of partial results
     */
    PartialResult(size_t nElements = 0);

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        regression::training::PartialResult::serialImpl<Archive, onDeserialize>(arch);

        return services::Status();
    }

    services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<data_management::InputDataArchive, false>(arch);

        return services::Status();
    }

    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const data_management::OutputDataArchive, true>(arch);

        return services::Status();
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__TRAINING__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the regression model-based training
 */
class DAAL_EXPORT Result : public regression::training::Result
{
public:
    DAAL_CAST_OPERATOR(Result)
    /**
     * Constructs the results of the regression training algorithm
     * \param[in] nElements Number of results
     */
    Result(size_t nElements = 0);

    /**
     * Returns the result of the regression model-based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    linear_model::ModelPtr get(ResultId id) const;

    /**
     * Sets the result of the regression model-based training
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultId id, const linear_model::ModelPtr & value);

    /**
     * Checks the result of the regression model-based training
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
        regression::training::Result::serialImpl<Archive, onDeserialize>(arch);

        return services::Status();
    }

    services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<data_management::InputDataArchive, false>(arch);

        return services::Status();
    }

    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const data_management::OutputDataArchive, true>(arch);

        return services::Status();
    }
};
typedef services::SharedPtr<Result> ResultPtr;
typedef services::SharedPtr<const Result> ResultConstPtr;
typedef services::SharedPtr<PartialResult> PartialResultPtr;
typedef services::SharedPtr<const PartialResult> PartialResultConstPtr;
} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
using interface1::ResultConstPtr;
using interface1::PartialResult;
using interface1::PartialResultPtr;
using interface1::PartialResultConstPtr;
} // namespace training
/** @} */
} // namespace linear_model
} // namespace algorithms
} // namespace daal
#endif
