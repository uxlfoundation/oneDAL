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

    #define _MSGTYP_GENERAL_ERROR             0x80000000
    #define _MSGTYP_ABORT                     0xc0000000
    #define _MSGTYP_CHECKBIOS_CPUIDMAXSETTING 0x88000000
    #define _MSGTYP_OSAFFCAP_ERROR            0xC4000000 // The # of processors supported by current OS exceed
    //                      those of legacy win32 (32 processors) and win64 API (64)
    #define _MSGTYP_UNKNOWNERR_OS         0xC2000000
    #define _MSGTYP_USERAFFINITYERR       0xC1000000
    #define _MSGTYP_TOPOLOGY_NOTANALYZED  0xC0800000
    #define _MSGTYP_UNK_AFFINTY_OPERATION 0x84000000

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
