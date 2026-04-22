/* file: engine.h */
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
//  Public interface for random number engines.
//--
*/

#ifndef __ENGINE_H__
#define __ENGINE_H__

#include "services/daal_defines.h"
#include "services/daal_shared_ptr.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
/**
 * @ingroup engines
 * @{
 */
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ENGINES__ENGINETYPE"></a>
 * \brief Available random number engine types
 */
enum EngineType
{
    mt19937Engine       = 0, /*!< Mersenne Twister engine (mt19937) */
    mt2203Engine        = 1, /*!< Mersenne Twister engine (mt2203) */
    mcg59Engine         = 2, /*!< Multiplicative congruential generator (mcg59) */
    mrg32k3aEngine      = 3, /*!< Combined multiple recursive generator (mrg32k3a) */
    philox4x32x10Engine = 4  /*!< Philox 4x32-10 counter-based engine */
};

namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__ENGINEIFACE"></a>
 *  \brief Minimal public interface for random number engines
 */
class DAAL_EXPORT EngineIface
{
public:
    virtual ~EngineIface() {}

    /**
     * Returns a pointer to the newly allocated engine
     * with a copy of input objects and parameters of this engine
     * \return Pointer to the newly allocated engine
     */
    virtual services::SharedPtr<EngineIface> clone() const = 0;

protected:
    EngineIface() {}
};
typedef services::SharedPtr<EngineIface> EngineIfacePtr;

/**
 * Creates a random number engine of specified type
 * \tparam algorithmFPType  Data type for the engine (float or double)
 * \param[in] type  Type of the engine to create
 * \param[in] seed  Initial seed for the engine
 * \return Pointer to the newly created engine
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE>
DAAL_EXPORT EngineIfacePtr createEngine(EngineType type, size_t seed = 777);

} // namespace interface1
using interface1::EngineIface;
using interface1::EngineIfacePtr;
using interface1::createEngine;
/** @} */
} // namespace engines
} // namespace algorithms
} // namespace daal
#endif
