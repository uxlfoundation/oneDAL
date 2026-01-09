/* file: service_topo.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
* Copyright contributors to the oneDAL project
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

#ifndef __SERVICE_TOPO_H__
#define __SERVICE_TOPO_H__

#include "services/daal_defines.h"

#if !defined(DAAL_CPU_TOPO_DISABLED)

    #define _MSGTYP_GENERAL_ERROR                  0x80000000
    #define _MSGTYP_INT_OVERFLOW                   0xA0000000
    #define _MSGTYP_CANNOT_SET_AFFINITY_BIT        0x90000000
    #define _MSGTYP_OS_PROC_COUNT_EXCEEDED         0x82000000 // The number of processors supported by current OS exceeded
    #define _MSGTYP_OS_GROUP_COUNT_EXCEEDED        0x81000000 // The number of processor groups supported by current OS exceeded
    #define _MSGTYP_INVALID_SNAPSHOT_HANDLE        0x80800000 // General unknown error. Do not confuse with
    #define _MSGTYP_FAILED_TO_INIT_PROC_AFFINITY   0x80400000 // General unknown error. Do not confuse with
    #define _MSGTYP_USER_AFFINITY_ERROR            0x80200000 // Error in user specified affinity mask
    #define _MSGTYP_CANNOT_TEST_AFFINITY_BIT       0x80100000 // Cannot test bit in generalized affinity mask
    #define _MSGTYP_MEMORY_ALLOCATION_FAILED       0x80080000
    #define _MSGTYP_THREAD_REPORTING_FAILED        0x80040000
    #define _MSGTYP_SET_THREAD_AFFINITY_FAILED     0x80020000 // Cannot set affinity for a thread
    #define _MSGTYP_RESTORE_THREAD_AFFINITY_FAILED 0x80010000 // Cannot restore affinity for a thread
    #define _MSGTYP_INVALID_THREAD_INDEX           0x80008000 // Thread index is greater than number of threads available in the affinity mask of the process
    #define _MSGTYP_INVALID_PACKAGE_INDEX          0x80004000 // Package index is greater than number of threads available in the affinity mask of the process
    #define _MSGTYP_INVALID_CORE_INDEX             0x80002000 // Core index is greater than the size of the allocated core ID array
    #define _MSGTYP_INVALID_THREAD_COUNT_INDEX     0x80001000 // Thread count index is greater than the size of the allocated thread count array
    #define _MSGTYP_TOPOLOGY_NOT_ANALYZED          0x80000800

namespace daal
{
namespace services
{
namespace internal
{
unsigned _internal_daal_GetProcessorCoreCount();
unsigned _internal_daal_GetStatus();
unsigned _internal_daal_GetSysLogicalProcessorCount();
} // namespace internal
} // namespace services
} // namespace daal

void read_topology(int & status, int & nthreads, int & max_threads, int ** cpu_queue);
void delete_topology(void * ptr);

#endif /* #if !defined (DAAL_CPU_TOPO_DISABLED) */

#endif /* __SERVICE_TOPO_H__ */
