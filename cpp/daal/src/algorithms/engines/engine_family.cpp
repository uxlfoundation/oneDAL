/* file: engine_family.cpp */
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

//++
//  Implementation of initializer types.
//--

#include "src/algorithms/engines/engine_family.h"

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
namespace internal
{
FamilyBatchBase::FamilyBatchBase() {}

FamilyBatchBase::FamilyBatchBase(const FamilyBatchBase & other) : FamilyBatchBase::super(other) {}

services::Status FamilyBatchBase::add(size_t numberOfStreams)
{
    return addImpl(numberOfStreams);
}

services::SharedPtr<FamilyBatchBase> FamilyBatchBase::get(size_t i) const
{
    return getImpl(i);
}

size_t FamilyBatchBase::getNumberOfStreams() const
{
    return getNumberOfStreamsImpl();
}

size_t FamilyBatchBase::getMaxNumberOfStreams() const
{
    return getMaxNumberOfStreamsImpl();
}

} // namespace internal
} // namespace engines
} // namespace algorithms
} // namespace daal
