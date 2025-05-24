/* file: brownboost_training_types.h */
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
//  Implementation of Brown Boost training algorithm interface.
//--
*/

#ifndef __BROWN_BOOST_TRAINING_TYPES_H__
#define __BROWN_BOOST_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/boosting/brownboost_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup brownboost Brownboost Classifier
 * \copydoc daal::algorithms::brownboost
 * @ingroup boosting
 */
namespace brownboost
{
/**
 * @defgroup brownboost_training Training
 * \copydoc daal::algorithms::brownboost::training
 * @ingroup brownboost
 * @{
 */
/**
 * \brief Contains classes for BrownBoost models training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__BROWNBOOST__TRAINING__METHOD"></a>
 * Available methods for BrownBoost model training
 */
enum Method
{
    defaultDense = 0 /*!< Default method proposed by Y. Freund */
};

/**
 * \brief Contains version 2.0 of oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BROWNBOOST__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method
 *        of the BrownBoost training algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)

    virtual ~Result() {}

    /**
     * Returns the model trained with the BrownBoost algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the BrownBoost algorithm
     */
    daal::algorithms::brownboost::ModelPtr get(classifier::training::ResultId id) const;

    /**
     * Allocates memory to store final results of BrownBoost training
     * \param[in] input        %Input of the BrownBoost training algorithm
     * \param[in] parameter     Parameters of the algorithm
     * \param[in] method        BrownBoost computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

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
} // namespace brownboost
} // namespace algorithms
} // namespace daal
#endif // __BROWN_BOOST_TRAINING_TYPES_H__
