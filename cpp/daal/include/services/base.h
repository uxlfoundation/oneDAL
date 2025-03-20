/* file: base.h */
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
//  Implementation of a dummy base class needed to fix ABI inconsistency between
//  Visual Studio* 2012 and 2013.
//--
*/

#ifndef __BASE_H__
#define __BASE_H__

#include "services/daal_defines.h"
#include "services/daal_memory.h"

namespace daal
{
/**
 * @ingroup services
 * @{
 */
/**
 * <a name="DAAL-CLASS-__BASE"></a>
 * \brief %Base class for oneAPI Data Analytics Library objects
 */
class DAAL_EXPORT Base
{
public:
    DAAL_NEW_DELETE();
    virtual ~Base() {}
};
/** @} */
} // namespace daal
#endif
