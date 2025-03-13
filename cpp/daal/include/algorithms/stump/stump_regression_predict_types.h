/* file: stump_regression_predict_types.h */
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
//  Implementation of the base classes used in the prediction stage
//  of the regression algorithm
//--
*/

#ifndef __STUMP_REGRESSION_PREDICT_TYPES_H__
#define __STUMP_REGRESSION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/stump/stump_regression_model.h"

#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/regression/regression_predict_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the Decision stump algorithm
 */
namespace stump
{
/**
 * \brief Contains classes of the Decision stump regression algorithm
 */
namespace regression
{
/**
 * @defgroup stump_regression_prediction Prediction
 * \copydoc daal::algorithms::stump::regression::prediction
 * @ingroup stump_regression
 * @{
 */
/**
 * \brief Contains classes for making prediction based on the regression model
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__STUMP__PREDICTION__RESULTID"></a>
 * \brief Available identifiers of the result for making decision tree model-based prediction
 */
enum ResultId
{
    prediction   = algorithms::regression::prediction::prediction, /*!< Result of decision tree model-based prediction */
    lastResultId = prediction
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__REGRESSION__PREDICTION__INPUT"></a>
 * \brief Input objects in the prediction stage of the stump algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::regression::prediction::Input
{
    typedef daal::algorithms::regression::prediction::Input super;

public:
    Input();
    Input(const Input & other);

    virtual ~Input() {}

    using super::get;
    using super::set;

    /**
     * Returns the input Numeric Table object in the prediction stage of the regression algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(daal::algorithms::regression::prediction::NumericTableInputId id) const;

    /**
     * Returns the input Model object in the prediction stage of the Stump algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    stump::regression::ModelPtr get(daal::algorithms::regression::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the regression algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(daal::algorithms::regression::prediction::NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the Stump algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(daal::algorithms::regression::prediction::ModelInputId id, const stump::regression::ModelPtr & ptr);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__REGRESSION__PREDICTION__RESULT"></a>
 * \brief Provides interface for the result of stump model-based prediction
 */
class DAAL_EXPORT Result : public algorithms::regression::prediction::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    /**
     * Returns the result of decision tree model-based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of decision tree model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
     * Allocates memory to store a partial result of decision tree model-based prediction
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Algorithm method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
     * Checks the result of stump model-based prediction
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
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
typedef services::SharedPtr<const Result> ResultConstPtr;

} // namespace interface1

using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
using interface1::ResultConstPtr;

} // namespace prediction
/** @} */
} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal
#endif
