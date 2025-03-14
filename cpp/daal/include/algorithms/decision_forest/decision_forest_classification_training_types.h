/* file: decision_forest_classification_training_types.h */
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
//  Implementation of decision forest classification training algorithm interface.
//--
*/

#ifndef __DECISION_FOREST_CLASSIFICATION_TRAINING_TYPES_H__
#define __DECISION_FOREST_CLASSIFICATION_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"
#include "algorithms/decision_forest/decision_forest_training_parameter.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
/**
 * @defgroup decision_forest_classification_training Training
 * \copydoc daal::algorithms::decision_forest::classification::training
 * @ingroup decision_forest_classification
 * @{
 */
/**
 * \brief Contains classes for Decision forest models training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__CLASSIFICAION__TRAINING__METHOD"></a>
 * \brief Computation methods for decision forest classification model-based training
 */
enum Method
{
    defaultDense = 0, /*!< Bagging, random choice of features, Gini impurity */
    hist         = 1  /*!< Subset of splits(bins), bagging, random choice of features, variance-based impurity */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__RESULT_NUMERIC_TABLEID"></a>
* \brief Available identifiers of the result of decision forest model-based training
*/
enum ResultNumericTableId
{
    outOfBagError = classifier::training::model + 1, /*!< %Numeric table 1x1 containing out-of-bag error.
                                                           Computed when computeOutOfBagError option is on */
    variableImportance,                              /*!< %Numeric table 1x(number of features) containing variable importance value.
                                                           Computed when parameter.varImportance != none */
    outOfBagErrorPerObservation,                     /*!< %Numeric table 1x(number of observations) containing out-of-bag error value computed.
                                                           Computed when computeOutOfBagErrorPerObservation option is on */
    outOfBagErrorAccuracy,                           /*!< %Numeric table 1x1 containing accuracy related to out-of-bag error.
                                                           Computed when computeOutOfBagErrorAccuracy option is on */
    outOfBagErrorDecisionFunction,                   /*!< %Numeric table (number of observations)x(number of classes)
                                                           containing probabilities related to out-of-bag error computed.
                                                           Computed when computeOutOfBagErrorDecisionFunction option is on */
    lastResultId = outOfBagErrorDecisionFunction
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__RESULT_NUMERIC_TABLEID"></a>
* \brief Available identifiers of the result of decision forest model-based training
*/
enum ResultEngineId
{
    updatedEngine      = lastResultId + 1, /*!< %Engine updated after computations. */
    lastResultEngineId = updatedEngine
};

/**
 * \brief Contains version 3.0 of oneAPI Data Analytics Library interface.
 */
namespace interface3
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__PARAMETER"></a>
 * \brief Decision forest algorithm parameters
 *
 * \snippet decision_forest/decision_forest_classification_training_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::Parameter, public daal::algorithms::decision_forest::training::Parameter
{
    /** Default constructor */
    Parameter(size_t nClasses) : classifier::Parameter(nClasses) {}
    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */
} // namespace interface3

namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method
 *        of the LogitBoost training algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)

    Result();
    virtual ~Result() DAAL_C11_OVERRIDE;

    /**
     * Returns the model trained with the LogitBoost algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the LogitBoost algorithm
     */
    ModelPtr get(classifier::training::ResultId id) const;

    /**
    * Sets the result of decision forest model-based training
    * \param[in] id      Identifier of the result
    * \param[in] value   Result
    */
    void set(classifier::training::ResultId id, const ModelPtr & value);

    /**
    * Returns the result of decision forest model-based training
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultNumericTableId id) const;

    /**
    * Sets the result of decision forest model-based training
    * \param[in] id      Identifier of the result
    * \param[in] value   Result
    */
    void set(ResultNumericTableId id, const data_management::NumericTablePtr & value);

    /**
     * Allocates memory to store final results of the LogitBoost training algorithm
     * \param[in] input         %Input of the LogitBoost training algorithm
     * \param[in] parameter     Parameters of the algorithm
     * \param[in] method        LogitBoost computation method
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
    * Checks the result of decision forest model-based training
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method
    * \return Status of checking
    */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the engine updated after computations
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    engines::EnginePtr get(ResultEngineId id) const;

protected:
    using daal::algorithms::interface1::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }

public:
    class ResultImpl;

    ResultImpl * impl() { return _impl; }

    Result(const Result & other);

private:
    ResultImpl * _impl;

    Result & operator=(const Result &);
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface1
using interface3::Parameter;
using interface1::Result;
using interface1::ResultPtr;

} // namespace training
/** @} */
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif // __DECISION_FOREST_CLASSIFICATION_TRAINING_TYPES_H__
