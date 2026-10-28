/* file: service_environment.h */
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

#ifndef __SERVICE_ENVIRONMENT_H__
#define __SERVICE_ENVIRONMENT_H__

#include "services/daal_defines.h"

namespace daal
{
namespace services
{
namespace internal
{
//returns size of L1 cache in bytes
DAAL_EXPORT size_t getL1CacheSize();
//returns size of L2 cache in bytes
DAAL_EXPORT size_t getL2CacheSize();
//returns size of LL (last level) cache in bytes
DAAL_EXPORT size_t getLLCacheSize();

//returns number of elements that fit into the memory of given size
//param sizeofAnElement - size of an element in bytes
//param defaultNumElements - return this number of element if sizeofMemory is 0
size_t getNumElementsFitInMemory(size_t sizeofMemory, size_t sizeofAnElement, size_t defaultNumElements);

//returns number of elements that fit into L1 cache
//param sizeofAnElement - size of an element in bytes
//param defaultNumElements - return this number of elements if cache size system method call failed
size_t getNumElementsFitInL1Cache(size_t sizeofAnElement, size_t defaultNumElements);

//returns number of elements that fit into LL cache
//param sizeofAnElement - size of an element in bytes
//param defaultNumElements - return this number of element if cache size system method call failed
size_t getNumElementsFitInLLCache(size_t sizeofAnElement, size_t defaultNumElements);

} // namespace internal
} // namespace services
} // namespace daal

#endif /* __SERVICE_ENVIRONMENT_H__ */
