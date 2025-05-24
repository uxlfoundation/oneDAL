/* file: algorithm_quality_metric_set_batch.h */
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
//  Interface for the quality metric set in the batch processing mode.
//--
*/

#ifndef __ALGORITHM_QUALITY_METRIC_SET_BATCH_H__
#define __ALGORITHM_QUALITY_METRIC_SET_BATCH_H__

#include "algorithms/algorithm_quality_metric_set_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes to compute a quality metric set
 */
namespace quality_metric_set
{
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @addtogroup base_algorithms
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC_SET__BATCH"></a>
 * \brief Provides methods to compute a quality metric set of an algorithm in the batch processing mode.
 */
class Batch
{
public:
    InputAlgorithmsCollection inputAlgorithms; /*!< Collection of quality metrics algorithms */

    Batch(bool useDefaultMetrics = true) : _useDefaultMetrics(useDefaultMetrics) {}

    virtual ~Batch() {}
    /**
     * Returns the structure that contains a computed quality metric set
     * \return Structure that contains a computed quality metric set
     */
    ResultCollectionPtr getResultCollection() { return _resultCollection; }

    /**
     * Returns the collection of input objects for the quality metrics algorithm
     * \return The collection of input objects for the quality metrics algorithm
     */
    InputDataCollectionPtr getInputDataCollection() { return _inputData; }

    /**
     * Computes results for a quality metric set in the batch processing mode.
     *
     * \return Status of computations
     */
    services::Status compute()
    {
        this->_status = computeNoThrow();
        return services::throwIfPossible(this->_status);
    }

    /**
     * Computes results for a quality metric set in the batch processing mode.
     *
     * \return Status of computations
     */
    services::Status computeNoThrow()
    {
        DAAL_CHECK(inputAlgorithms.size(), services::ErrorEmptyInputAlgorithmsCollection)
        services::Status s;
        for (size_t i = 0; i < inputAlgorithms.size(); i++)
        {
            size_t key = inputAlgorithms.getKeyByIndex((int)i);
            inputAlgorithms[key]->setInput(_inputData->getInput(key).get());
            s = inputAlgorithms[key]->computeNoThrow();
            if (!s) break;
            _resultCollection->add(key, inputAlgorithms[key]->getResult());
        }
        return s;
    }

    /**
     * Returns errors that happened during the computations
     * \return Errors that happened during the computations
     */
    services::SharedPtr<services::ErrorCollection> getErrors() { return _status.getCollection(); }

protected:
    virtual void initializeQualityMetrics() = 0;

    bool _useDefaultMetrics;
    InputDataCollectionPtr _inputData; /*!< Collection of input objects of quality metrics algorithms */
    ResultCollectionPtr _resultCollection;
    services::Status _status;
};
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace quality_metric_set
} // namespace algorithms
} // namespace daal
#endif
