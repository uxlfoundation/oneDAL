/* file: engine_family.h */
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
//  Implementation engine.
//--
*/

#ifndef __ENGINE_FAMILY_H__
#define __ENGINE_FAMILY_H__

#include "algorithms/engines/engine.h"

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
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__FAMILYBATCHBASE"></a>
 *  \brief Class representing an engine that has collection of independent streams obtained from RNGs from same family.
 *
 * \DAAL_DEPRECATED
 */
class DAAL_EXPORT FamilyBatchBase : public engines::BatchBase
{
public:
    typedef engines::BatchBase super;

    typedef super::InputType InputType;
    typedef super::ResultType ResultType;

    DAAL_DEPRECATED FamilyBatchBase();
    virtual ~FamilyBatchBase() {}

    /**
     * Add 'numberOfStreams' random number streams to the engine
     * \param[in] numberOfStreams  Number of streams to add
     *
     * \return Status of computations
     */
    services::Status add(size_t numberOfStreams);

    /**
     * Get engine with one stream from collection.
     * \param[in] i index of stream from the collection
     *
     * \return SharedPtr to engine instance with ith stream from the collection
     */
    services::SharedPtr<FamilyBatchBase> get(size_t i) const;

    /**
     * Return the number of streams in the collection.
     *
     * \return Number of streams in the collection
     */
    size_t getNumberOfStreams() const;

    /**
     * Return the maximum number of streams can be obtained from RNGs in given family
     *
     * \return Maximum number of streams can be obtained from RNGs in given family
     */
    size_t getMaxNumberOfStreams() const;

protected:
    virtual services::Status addImpl(size_t /*numberOfStreams*/) { return services::Status(); }
    virtual services::SharedPtr<FamilyBatchBase> getImpl(size_t /*i*/) const { return services::SharedPtr<FamilyBatchBase>(); }
    virtual size_t getNumberOfStreamsImpl() const { return 0; }
    virtual size_t getMaxNumberOfStreamsImpl() const { return 0; }

    FamilyBatchBase(const FamilyBatchBase & other);

private:
    FamilyBatchBase & operator=(const FamilyBatchBase &);
};
typedef services::SharedPtr<FamilyBatchBase> FamilyEnginePtr;

} // namespace interface1
using interface1::FamilyBatchBase;
using interface1::FamilyEnginePtr;
/** @} */
} // namespace engines
} // namespace algorithms
} // namespace daal
#endif
