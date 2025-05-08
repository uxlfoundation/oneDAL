/* file: host_app.h */
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
//  Interface of host application class used by the library
//--
*/

#ifndef __DAAL_HOST_APP_H__
#define __DAAL_HOST_APP_H__

#include "services/daal_defines.h"
#include "services/base.h"
#include "services/daal_shared_ptr.h"

namespace daal
{
namespace services
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 *  <a name="DAAL-CLASS-SERVICES__HOSTAPPIFACE"></a>
 *  \brief Abstract class which defines callback interface for the host application of this library
 *         to enable such features as computation cancelling, progress bar, status bar, verbose, etc.
 */
class DAAL_EXPORT HostAppIface : public Base
{
public:
    DAAL_NEW_DELETE();
    HostAppIface();
    HostAppIface(const HostAppIface & other)             = delete;
    HostAppIface & operator=(const HostAppIface & other) = delete;
    virtual ~HostAppIface();
    /**
     * This callback is called by compute() methods of the library algorithms.
     * If it returns true then compute() stops and returns 'ErrorUserCancelled' status
     * \return True when algorithm should be aborted
     */
    virtual bool isCancelled() = 0;

private:
    Base * _impl;
};
typedef services::SharedPtr<HostAppIface> HostAppIfacePtr;

} // namespace interface1
using interface1::HostAppIface;
using interface1::HostAppIfacePtr;

} // namespace services
} // namespace daal
#endif //__DAAL_HOST_APP_H__
