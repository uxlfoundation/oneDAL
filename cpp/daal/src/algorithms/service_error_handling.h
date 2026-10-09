/* file: service_error_handling.h */
/*******************************************************************************
* Copyright 2015 Intel Corporation
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
//  Declaration of service error handling classes
//--
*/
#ifndef __SERVICE_ERROR_HANDLING_H__
#define __SERVICE_ERROR_HANDLING_H__

#include "services/error_handling.h"
#include "src/algorithms/service_threading.h"

namespace daal
{
//Thread safe holder of Status
class SafeStatus
{
public:
    explicit SafeStatus();
    explicit SafeStatus(const services::Status & s);

    bool ok() const;
    operator bool() const { return ok(); }

    SafeStatus & add(services::ErrorID id);
    SafeStatus & add(const services::ErrorPtr & e);
    SafeStatus & add(const services::Status & s);
    SafeStatus & operator|=(const services::Status & other) { return add(other); }

    //Return services::Status and clean this class
    services::Status detach();

private:
    services::Status _val;
    mutable Mutex _m;
};

} // namespace daal

#endif
