/* file: adaboost_training_types.h */
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
//  Implementation of Ada Boost training algorithm interface.
//--
*/

#ifndef __ADA_BOOST_TRAINING_TYPES_H__
#define __ADA_BOOST_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/boosting/adaboost_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup adaboost Adaboost Classifier
 * \copydoc daal::algorithms::adaboost
 * @ingroup boosting
 */
namespace adaboost
{
/**
 * @defgroup adaboost_training Training
 * \copydoc daal::algorithms::adaboost::training
 * @ingroup adaboost
 * @{
 */
/**
 * \brief Contains classes for AdaBoost models training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ADABOOST__TRAINING__METHOD"></a>
 * Available methods for AdaBoost model training
 */
enum Method
{
    defaultDense = 0,            /*!< Default method */
    samme        = defaultDense, /*!< SAMME algorithm */
    sammeR       = 1             /*!< SAMME.R algorithm, need probabilities from weak learner prediction result */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__ADABOOST__CLASSIFICATION__TRAINING__RESULTNUMERICTABLEID"></a>
* \brief Available identifiers of the result of AdaBoost model-based training
*/
enum ResultNumericTableId
{
    weakLearnersErrors       = classifier::training::model + 1, /*!< %Numeric table 1 x maxIterations containing
                                                                weak learners classification errors */
    lastResultNumericTableId = weakLearnersErrors
};

/**
 * \brief Contains version 2.0 of oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ADABOOST__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method
 *        of the AdaBoost training algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)

    Result();
    virtual ~Result() {}

    /**
     * Allocates memory to store final results of AdaBoost training
     * \param[in] input         %Input of the AdaBoost training algorithm
     * \param[in] parameter     Parameters of the algorithm
     * \param[in] method        AdaBoost computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns the model trained with the AdaBoost algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the LogitBoost algorithm
     */
    ModelPtr get(classifier::training::ResultId id) const;

    /**
    * Sets the result of AdaBoost model-based training
    * \param[in] id      Identifier of the result
    * \param[in] value   Result
    */
    void set(classifier::training::ResultId id, const ModelPtr & value);

    /**
    * Returns the result of AdaBoost model-based training
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultNumericTableId id) const;

    /**
    * Sets the result of AdaBoost model-based training
    * \param[in] id      Identifier of the result
    * \param[in] value   Result
    */
    void set(ResultNumericTableId id, const data_management::NumericTablePtr & value);

    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

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
} // namespace interface2
using interface2::Result;
using interface2::ResultPtr;

} // namespace training
/** @} */
} // namespace adaboost
} // namespace algorithms
} // namespace daal
#endif // __ADA_BOOST_TRAINING_TYPES_H__
