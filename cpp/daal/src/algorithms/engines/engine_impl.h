/* file: engine_impl.h */
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
//  Implementation of engine internal classes (deprecated, for internal use only).
//--
*/

#ifndef __ENGINE_IMPL_H__
#define __ENGINE_IMPL_H__

#include "services/daal_defines.h"
#include "algorithms/engines/engine.h"
#include "algorithms/algorithm.h"
#include "src/algorithms/engines/engine_types.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__BATCHBASE"></a>
 *  \brief Class representing an engine (internal implementation)
 *
 * \DAAL_DEPRECATED Internal use only
 */
class BatchBase : public EngineIface, public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::engines::Input InputType;
    typedef algorithms::engines::Result ResultType;

    InputType input; /*!< Input of the engine */

    virtual ~BatchBase() {}

    /**
     * Saves current engine state to destination
     * \param[in] dest  Destination to save the state
     *
     * \return Status of computations
     */
    services::Status saveState(byte * dest) const { return saveStateImpl(dest); }

    /**
     * Rewrites current state with source one
     * \param[in] src  Source state
     *
     * \return Status of computations
     */
    services::Status loadState(const byte * src) { return loadStateImpl(src); }

    /**
     * Enables the usage of current engine in parallel regions of code with leapfrog method
     * \param[in] threadIdx  Index of the thread
     * \param[in] nThreads   Number of threads
     *
     * \return Status of computations
     */
    services::Status leapfrog(size_t threadIdx, size_t nThreads) { return leapfrogImpl(threadIdx, nThreads); }

    /**
     * Enables the usage of current engine in parallel regions of code with skipAhead method
     * \param[in] nSkip  Number of elements that will be skipped
     *
     * \return Status of computations
     */
    services::Status skipAhead(size_t nSkip) { return skipAheadImpl(nSkip); }

    /**
     * Returns a pointer to the newly allocated engine
     * with a copy of input objects and parameters of this engine
     * \return Pointer to the newly allocated engine
     */
    services::SharedPtr<EngineIface> clone() const override { return cloneBatch(); }

    /**
     * Returns a pointer to the newly allocated engine (BatchBase type)
     * with a copy of input objects and parameters of this engine
     * \return Pointer to the newly allocated engine
     */
    services::SharedPtr<BatchBase> cloneBatch() const
    {
        daal::algorithms::Algorithm<batch> * alg = this->Analysis<batch>::cloneImpl();
        return services::SharedPtr<BatchBase>(static_cast<BatchBase*>(alg));
    }

protected:
    virtual services::Status saveStateImpl(byte * /*dest*/) const { return services::Status(); }
    virtual services::Status loadStateImpl(const byte * /*src*/) { return services::Status(); }
    virtual services::Status leapfrogImpl(size_t /*threadNum*/, size_t /*nThreads*/) { return services::Status(services::ErrorMethodNotSupported); }
    virtual services::Status skipAheadImpl(size_t /*nSkip*/) { return services::Status(); }
};
typedef services::SharedPtr<BatchBase> EnginePtr;

} // namespace interface1
using interface1::BatchBase;
using interface1::EnginePtr;
} // namespace engines
} // namespace algorithms
} // namespace daal
#endif
