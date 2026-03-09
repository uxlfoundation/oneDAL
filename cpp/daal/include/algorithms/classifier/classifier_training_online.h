/* file: classifier_training_online.h */
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

#ifndef __CLASSIFIER_TRAINING_ONLINE_H__
#define __CLASSIFIER_TRAINING_ONLINE_H__

#include "services/daal_defines.h"

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
 * @defgroup classifier_training_online Online
 * @ingroup training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__ONLINE"></a>
 *  \brief Algorithm class for training the classifier model in the online processing mode
 *
 * \par Enumerations
 *      - \ref InputId  %Input objects of the classifier model training algorithm
 *      - \ref ResultId Results of the classifier model training algorithm
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref interface1::Model "Model" class
 */
class DAAL_EXPORT Online : public Training<online>
{
public:
    typedef algorithms::classifier::training::Input InputType;
    typedef algorithms::classifier::Parameter ParameterType;
    typedef algorithms::classifier::training::Result ResultType;
    typedef algorithms::classifier::training::PartialResult PartialResultType;

    Online() {}

    /**
     * Constructs a classifier training algorithm by copying input objects and parameters
     * of another classifier training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Online(const Online & /*other*/) {}

    virtual ~Online() {}

    /**
     * Get input objects for the classifier model training algorithm
     * \return %Input objects for the classifier model training algorithm
     */
    DAAL_DEPRECATED_VIRTUAL virtual InputType * getInput() = 0;

    /**
     * Registers user-allocated memory for storing partial training results
     * \param[in] partialResult Structure for storing partial results
     * \param[in] initFlag Flag if partial result initialized or not
     */
    services::Status setPartialResult(const PartialResultPtr & partialResult, bool initFlag = false)
    {
        DAAL_CHECK(partialResult, services::ErrorNullPartialResult)
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        setInitFlag(initFlag);
        return services::Status();
    }

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
     * Returns the structure that contains computed partial results of the classification algorithm
     * \return Structure that contains computed partial results
     */
    PartialResultPtr getPartialResult() { return _partialResult; }

    /**
     * Returns the structure that contains results of the classification algorithm
     * \return Structure that contains computed results
     */
    ResultPtr getResult() { return _result; }

    /**
     * Returns a pointer to the newly allocated classifier training algorithm with a copy of input objects
     * and parameters of this classifier training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Online> clone() const { return services::SharedPtr<Online>(cloneImpl()); }

protected:
    PartialResultPtr _partialResult;
    ResultPtr _result;

    virtual Online * cloneImpl() const override = 0;

private:
    Online & operator=(const Online &);
};
/** @} */
} // namespace interface2
using interface2::Online;

} // namespace training
} // namespace classifier
} // namespace algorithms
} // namespace daal
#endif
