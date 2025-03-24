/* file: weak_learner_training_types.h */
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
//  Implementation of the base classes used on the training stage
//  of weak learners algorithms
//--
*/

#ifndef __WEAK_LEARNER_TRAINING_TYPES_H__
#define __WEAK_LEARNER_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/weak_learner/weak_learner_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for working with weak learners
 */
namespace weak_learner
{
/**
 * @defgroup weak_learner_training Training
 * \copydoc daal::algorithms::weak_learner::training
 * @ingroup weak_learner
 * @{
 */
/**
 * \brief Contains classes for training models of the weak learners algorithms
 */
namespace training
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__WEAK_LEARNER__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with compute() method of Batch
 *        or finalizeCompute() method of Online and Distributed weak learners algorithms
 */
class DAAL_EXPORT Result : public daal::algorithms::classifier::training::interface1::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();
    virtual ~Result() {}

    /**
     * Returns the model trained with the weak learner  algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the weak learner  algorithm
     */
    daal::algorithms::weak_learner::ModelPtr get(classifier::training::ResultId id) const;

    /**
     * Sets the result of the training stage of the weak learner algorithm
     * \param[in] id      Identifier of the result, \ref classifier::training::ResultId
     * \param[in] value   Pointer to the training result
     */
    void set(classifier::training::ResultId id, daal::algorithms::weak_learner::ModelPtr & value);

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::classifier::training::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
} // namespace interface1
using interface1::Result;
using interface1::ResultPtr;

} // namespace training
/** @} */
} // namespace weak_learner
} // namespace algorithms
} // namespace daal
#endif
