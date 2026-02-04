/* file: classifier_training_batch.h */
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
//  Implementation of the interface for the classifier model training algorithm.
//--
*/

#ifndef __CLASSIFIER_TRAINING_BATCH_H__
#define __CLASSIFIER_TRAINING_BATCH_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace training
{
namespace interface2
{
/**
 * @defgroup classifier_training_batch Batch
 * @ingroup training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__BATCH"></a>
 * \brief Algorithm class for training the classifier model
 *
 * \par Enumerations
 *      - \ref InputId  Identifiers of input objects of the classifier model training algorithm
 *      - \ref ResultId Identifiers of results of the classifier model training algorithm
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref interface1::Model "Model" class
 */
class Batch : public Training<batch>
{
public:
    typedef algorithms::classifier::training::Input InputType;
    typedef algorithms::classifier::Parameter ParameterType;
    typedef algorithms::classifier::training::Result ResultType;

    virtual ~Batch() {}

    /**
     * Get input objects for the classifier model training algorithm
     * \return %Input objects for the classifier model training algorithm
     */
    virtual InputType * getInput() = 0;

    /**
     * Gets parameter objects for the classifier model training algorithm
     * \return %Parameter objects for the classifier model training algorithm
     */
    ParameterType & parameter() { return *static_cast<ParameterType *>(this->getBaseParameter()); }

    /**
     * Gets parameter objects for the classifier model training algorithm
     * \return %Parameter objects for the classifier model training algorithm
     */
    // const ParameterType& parameter() const { return *static_cast<const ParameterType*>(this->getBaseParameter()); }

    /**
     * Registers user-allocated memory for storing results of the classifier model training algorithm
     * \param[in] res    Structure for storing results of the classifier model training algorithm
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains the trained classifier model
     * \return Structure that contains the trained classifier model
     */
    ResultPtr getResult() { return _result; }

    /**
     * Resets the results of the classifier model training algorithm
     * \return Status of the operation
     */
    virtual services::Status resetResult() = 0;

    /**
     * Returns a pointer to the newly allocated classifier training algorithm with a copy of input objects
     * and parameters of this classifier training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch> clone() const { return services::SharedPtr<Batch>(cloneImpl()); }

protected:
    virtual Batch * cloneImpl() const override = 0;
    ResultPtr _result;
};
/** @} */
} // namespace interface2
using interface2::Batch;

} // namespace training
} // namespace classifier
} // namespace algorithms
} // namespace daal
#endif
