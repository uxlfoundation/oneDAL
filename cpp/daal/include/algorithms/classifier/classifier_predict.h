/* file: classifier_predict.h */
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
//  Implementation of the prediction stage of the classification algorithm interface.
//--
*/

#ifndef __CLASSIFIER_PREDICT_H__
#define __CLASSIFIER_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace prediction
{
namespace interface2
{
/**
 * @defgroup classifier_prediction_batch Batch
 * @ingroup prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__PREDICTION__BATCH"></a>
 *  \brief Base class for making predictions based on the model of the classification algorithms
 *
 *  \par Enumerations
 *      - \ref classifier::prediction::NumericTableInputId  Identifiers of input NumericTable objects
 *                                                          of the classifier prediction algorithm
 *      - \ref classifier::prediction::ModelInputId         Identifiers of input Model objects
 *                                                          of the classifier prediction algorithm
 *      - \ref classifier::prediction::ResultId             Identifiers of prediction results of the classifier algorithm
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref interface1::Model "Model" class
 */
class Batch : public daal::algorithms::Prediction
{
public:
    typedef algorithms::classifier::prediction::Input InputType;
    typedef algorithms::classifier::Parameter ParameterType;
    typedef algorithms::classifier::prediction::Result ResultType;

    Batch() { initialize(); }

    /**
     * Constructs a classifier prediction algorithm by copying input objects and parameters
     * of another classifier prediction algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch & /*other*/) { initialize(); }

    virtual ~Batch() {}

    /**
     * Get input objects for the classifier prediction algorithm
     * \return %Input objects for the classifier prediction algorithm
     */

    virtual InputType * getInput() = 0;
    /**
     * Gets parameter objects for the classifier model prediction algorithm
     * \return %Parameter objects for the classifier model prediction algorithm
     */
    ParameterType & parameter() { return *static_cast<ParameterType *>(this->getBaseParameter()); }

    /**
     * Gets parameter objects for the classifier model prediction algorithm
     * \return %Parameter objects for the classifier model prediction algorithm
     */
    // const ParameterType& parameter() const { return *static_cast<const ParameterType*>(this->getBaseParameter()); }

    /**
     * Returns the structure that contains computed prediction results
     * \return Structure that contains computed prediction results
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory for storing the prediction results
     * \param[in] result Structure for storing the prediction results
     *
     * \return Status of computation
     */
    services::Status setResult(const ResultPtr & result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated classifier prediction algorithm with a copy of input objects
     * and parameters of this classifier prediction algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch> clone() const { return services::SharedPtr<Batch>(cloneImpl()); }

protected:
    void initialize() { _result.reset(new ResultType()); }
    virtual Batch * cloneImpl() const override = 0;
    ResultPtr _result;

private:
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface2
using interface2::Batch;

} // namespace prediction
} // namespace classifier
} // namespace algorithms
} // namespace daal
#endif
