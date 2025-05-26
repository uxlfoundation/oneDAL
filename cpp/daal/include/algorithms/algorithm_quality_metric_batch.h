/* file: algorithm_quality_metric_batch.h */
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
//  Interface for the quality metrics in the batch processing mode.
//--
*/

#ifndef __ALGORITHM_QUALITY_METRIC_BATCH_H__
#define __ALGORITHM_QUALITY_METRIC_BATCH_H__

#include "algorithms/analysis.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes to compute quality metrics
 */
namespace quality_metric
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
 * <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC__BATCH"></a>
 * \brief Provides methods to compute quality metrics of an algorithm in the batch processing mode.
 *        Quality metric is a numerical characteristic or a set of connected numerical characteristics
 *        that represents the qualitative aspect of a computed statistical estimate, model,
 *        or decision-making result.
 */
class DAAL_EXPORT Batch : public Analysis<batch>
{
public:
    Batch() : Analysis<batch>() {}
    virtual ~Batch() {}

    /**
     * Sets an input object
     * \param[in] input Pointer to the input object
     */
    virtual void setInput(const algorithms::Input * input) = 0;

    /**
     * Returns the structure that contains computed quality metrics
     * \return Structure that contains computed quality metrics
     */
    algorithms::ResultPtr getResult() const { return getResultImpl(); }

protected:
    virtual algorithms::ResultPtr getResultImpl() const = 0;
};
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace quality_metric
} // namespace algorithms
} // namespace daal
#endif
