/* file: service_topo.cpp */
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
//  Implementation of CPU topology reading routines
//--
*/
#include "services/daal_defines.h"

#if !(defined DAAL_CPU_TOPO_DISABLED)

    #include "src/externals/service_memory.h"
    #include "src/services/service_topo.h"

    #ifdef __FreeBSD__
static int sched_setaffinity(pid_t pid, size_t cpusetsize, cpu_set_t * mask)
{
    return cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_PID, pid == 0 ? -1 : pid, cpusetsize, mask);
}
static int sched_getaffinity(pid_t pid, size_t cpusetsize, cpu_set_t * mask)
{
    return cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_PID, pid == 0 ? -1 : pid, cpusetsize, mask);
}

        #undef stdout
        #undef stderr
FILE * stdout = __stdoutp;
FILE * stderr = __stderrp;
    #endif

    #define _INTERNAL_DAAL_MALLOC(x)              daal_malloc((x), 64)
    #define _INTERNAL_DAAL_FREE(x)                daal_free((x))
    #define _INTERNAL_DAAL_MEMSET(a1, a2, a3)     __internal_daal_memset((a1), (a2), (a3))
    #define _INTERNAL_DAAL_MEMCPY(a1, a2, a3, a4) daal::services::internal::daal_memcpy_s((a1), (a2), (a3), (a4))

namespace daal
{
namespace services
{
namespace internal
{
static glktsn glbl_obj;

static char scratch[BLOCKSIZE_4K]; // scratch space large enough for OS to write SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX

static void * __internal_daal_memset(void * s, int c, size_t n)
{
    unsigned char * p = (unsigned char *)s;
    while (n--) *p++ = (unsigned char)c;
    return s;
}

/*
 * __internal_daal_bindContext
 * A wrapper function that can compile under two OS environments
 * The size of the bitmap that underlies cpu_set_t is configurable
 * at Linux Kernel Compile time. Each distro may set limit on its own.
 * Some newer Linux distro may support 256 logical processors,
 * For simplicity we don't show the check for range on the ordinal index
 * of the target cpu in Linux, interested reader can check Linux kernel documentation.
 * Prior to Windows OS with version signature 0601H, it has size limit of 64 cpu
 * in 64-bit mode, 32 in 32-bit mode, the size limit is checked.
 * Starting with Windows OS version 0601H (e.g. Windows 7), it supports up to 4 sets of
 * affinity masks, referred to as "GROUP_AFFINITY".
 * Constext switch within the same group is done by the same API as was done in previous
 * generations of windows (such as Vista). In order to bind the current executing
 * process context to a logical processor in a different group, it must be be binded
 * using a new API to the target processor group, followed by a similar
 * SetThreadAffinityMask API
 * New API related to GROUP_AFFINITY are present only in kernel32.dll of the OS with the
 * relevant version signatures. So we dynamically examine the presence of thse API
 * and fall back of legacy AffinityMask API if the new APIs are not available.
 * Limitation, New Windows APIs that support GROUP_AFFINITY requires
 *  Windows platform SDK 7.0a. The executable
 *  using this SDK and recent MS compiler should be able to perform topology enumeration on
 *  Windows 7 and prior versions
 *  If the executable is compiled with prior versions of platform SDK,
 *  the topology enumeration will not use API and data structures defined in SDK 7.0a,
 *  So enumeration will be limited to the active processor group.
 * Arguments:
 *      cpu :   the ordinal index to reference a logical processor in the system
 * Return:        0 is no error
 */
static int __internal_daal_bindContext(unsigned int cpu, void * prevAffinity)
{
    int ret = -1;
    #if defined(__linux__) || defined(__FreeBSD__)
    cpu_set_t currentCPU;
    // add check for size of cpumask_t.
    MY_CPU_ZERO(&currentCPU);
    // turn on the equivalent bit inside the bitmap corresponding to affinitymask
    MY_CPU_SET(cpu, &currentCPU);
    sched_getaffinity(0, sizeof(*((cpu_set_t *)prevAffinity)), (cpu_set_t *)prevAffinity);
    if (!sched_setaffinity(0, sizeof(currentCPU), &currentCPU)) ret = 0;
    #else
        // compile with SDK 7.0a will allow EXE to run on Windows versions
        // with and without GROUP_AFFINITY support
        #if (_WIN32_WINNT >= 0x0601)
    //we resolve API dynamically at runtime, data structures required by new API is determined at compile time

    unsigned int cpu_beg = 0, cpu_cnt, j;
    DWORD cnt;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX * pSystem_rel_info = NULL;
    GROUP_AFFINITY grp_affinity;
    if (cpu >= MAX_WIN7_LOG_CPU) return ret;

    cnt = BLOCKSIZE_4K;
    _INTERNAL_DAAL_MEMSET(&scratch[0], 0, cnt);
    pSystem_rel_info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)&scratch[0];

    if (!GetLogicalProcessorInformationEx(RelationGroup, pSystem_rel_info, &cnt)) return ret;

    if (pSystem_rel_info->Relationship != RelationGroup) return ret;
    // to determine the input ordinal 'cpu' number belong to which processor group,
    // we consider each processor group to have its logical processors assigned with
    // numerical index consecutively in increasing order
    for (j = 0; j < pSystem_rel_info->Group.ActiveGroupCount; j++)
    {
        cpu_cnt = pSystem_rel_info->Group.GroupInfo[j].ActiveProcessorCount;
        // if the 'cpu' value is within the lower and upper bounds of a
        // processor group, we can use the new API to bind current thread to
        // the target thread affinity bit in the target processor group
        if (cpu >= cpu_beg && cpu < (cpu_beg + cpu_cnt))
        {
            _INTERNAL_DAAL_MEMSET(&grp_affinity, 0, sizeof(GROUP_AFFINITY));
            grp_affinity.Group = j;
            grp_affinity.Mask = (KAFFINITY)((DWORD_PTR)(LNX_MY1CON << (cpu - cpu_beg)));
            if (!SetThreadGroupAffinity(GetCurrentThread(), &grp_affinity, (GROUP_AFFINITY *)prevAffinity))
            {
                GetLastError();
                return ret;
            }

            return 0;
        }
        // if the value of 'cpu' is not this processor group, we move to the next group
        cpu_beg += cpu_cnt;
    }
        #else // If SDK version does not support GROUP_AFFINITY,
    DWORD_PTR affinity;

    // only the active processor group and be succesfully queried and analyzed for topology information
    if (cpu >= MAX_PREWIN7_LOG_CPU) return ret;
    // flip on the bit in the affinity mask corresponding to the input ordinal index

    affinity                     = (DWORD_PTR)(LNX_MY1CON << cpu);
    *((DWORD_PTR *)prevAffinity) = SetThreadAffinityMask(GetCurrentThread(), affinity) if ((DWORD_PTR)(*prevAffinity)) ret = 0;
        #endif
    #endif
    return ret;
}

static void __internal_daal_restoreContext(void * prevAffinity)
{
    #if defined(__linux__) || defined(__FreeBSD__)
    sched_setaffinity(0, sizeof(*((cpu_set_t *)prevAffinity)), (cpu_set_t *)prevAffinity);
    #else
        #if (_WIN32_WINNT >= 0x0601)
    SetThreadGroupAffinity(GetCurrentThread(), (GROUP_AFFINITY *)prevAffinity, NULL);
        #else // If SDK version does not support GROUP_AFFINITY,
    SetThreadAffinityMask(GetCurrentThread(), *((DWORD_PTR *)prevAffinity));
        #endif
    #endif
}

/*
 * _internal_daal_GetMaxCPUSupportedByOS
 * A wrapper function that calls OS specific system API to find out
 * how many logical processor the OS supports
 * Return:        a non-zero value
 */
unsigned int _internal_daal_GetMaxCPUSupportedByOS()
{
    unsigned int lcl_OSProcessorCount = 0;
    #if defined(__linux__) || defined(__FreeBSD__)

    lcl_OSProcessorCount = sysconf(_SC_NPROCESSORS_CONF); //This will tell us how many CPUs are currently enabled.

    #else
        #if (_WIN32_WINNT >= 0x0601)
    unsigned short grpCnt;
    DWORD cnt;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX * pSystem_rel_info = NULL;

    // runtime check if os version is greater than 0601h

    lcl_OSProcessorCount = 0;
    // if Windows version support processor groups
    // tally actually populated logical processors in each group
    grpCnt = (WORD)GetActiveProcessorGroupCount();
    cnt = BLOCKSIZE_4K;
    _INTERNAL_DAAL_MEMSET(&scratch[0], 0, cnt);
    pSystem_rel_info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)&scratch[0];

    if (!GetLogicalProcessorInformationEx(RelationGroup, pSystem_rel_info, &cnt))
    {
        glbl_obj.error |= _MSGTYP_UNKNOWNERR_OS;
        return 0;
    }
    if (pSystem_rel_info->Relationship != RelationGroup)
    {
        glbl_obj.error |= _MSGTYP_UNKNOWNERR_OS;
        return 0;
    }
    for (unsigned int i = 0; i < grpCnt; i++) lcl_OSProcessorCount += pSystem_rel_info->Group.GroupInfo[i].ActiveProcessorCount;
        #else
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    lcl_OSProcessorCount = si.dwNumberOfProcessors;
        #endif
    #endif
    return lcl_OSProcessorCount;
}

/*
 * __internal_daal_setChkProcessAffinityConsistency
 *
 * A wrapper function that calls OS specific system API to find out
 * the number of logical processor support by OS matches
 * the same set of logical processors this process is allowed to run on
 * if the two set matches, set the corresponding bits in our
 * generic affinity mask construct.
 * if inconsistency is found, app-specific error code is set
 *
 * Return: none
 */
static void __internal_daal_setChkProcessAffinityConsistency(unsigned int lcl_OSProcessorCount)
{
    unsigned int i, sum = 0;
    #if defined(__linux__) || defined(__FreeBSD__)
    cpu_set_t allowedCPUs;

    sched_getaffinity(0, sizeof(allowedCPUs), &allowedCPUs);
    for (i = 0; i < lcl_OSProcessorCount; i++)
    {
        if (MY_CPU_ISSET(i, &allowedCPUs) == 0)
        {
            glbl_obj.error |= _MSGTYP_USERAFFINITYERR;
        }
        else
        {
            __internal_daal_setGenericAffinityBit(&glbl_obj.cpu_generic_processAffinity, i);
            __internal_daal_setGenericAffinityBit(&glbl_obj.cpu_generic_systemAffinity, i);
        }
    }

    #else
    DWORD_PTR processAffinity;
    DWORD_PTR systemAffinity;

        #if (_WIN32_WINNT >= 0x0601)

    GROUP_AFFINITY grp_affinity, prev_grp_affinity;
    unsigned int cpu_cnt;
    DWORD cnt;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX * pSystem_rel_info = NULL;

    {
        cnt = BLOCKSIZE_4K;
        _INTERNAL_DAAL_MEMSET(&scratch[0], 0, cnt);
        pSystem_rel_info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)&scratch[0];

        if (!GetLogicalProcessorInformationEx(RelationGroup, pSystem_rel_info, &cnt))
        {
            glbl_obj.error |= _MSGTYP_UNKNOWNERR_OS;
            return;
        }
        if (pSystem_rel_info->Relationship != RelationGroup)
        {
            glbl_obj.error |= _MSGTYP_UNKNOWNERR_OS;
            return;
        }
        if (lcl_OSProcessorCount > MAX_WIN7_LOG_CPU)
        {
            glbl_obj.error |= _MSGTYP_OSAFFCAP_ERROR; // If the os supports more processors than allowed, make change as required.
        }
        const unsigned short grpCnt = GetActiveProcessorGroupCount();
        for (i = 0; i < grpCnt; i++)
        {
            unsigned short grpAffinity[MAX_THREAD_GROUPS_WIN7];
            unsigned short grpCntArg = grpCnt;
            if (!GetProcessGroupAffinity(GetCurrentProcess(), &grpCntArg, &grpAffinity[0]))
            {
                //throw some exception here, no full affinity for the process
                glbl_obj.error |= _MSGTYP_UNKNOWNERR_OS;
                break;
            }
            else
            {
                cpu_cnt = pSystem_rel_info->Group.GroupInfo[i].ActiveProcessorCount;
                _INTERNAL_DAAL_MEMSET(&grp_affinity, 0, sizeof(GROUP_AFFINITY));
                grp_affinity.Group = (WORD)i;
                if (cpu_cnt == (sizeof(DWORD_PTR) * 8))
                    grp_affinity.Mask = (DWORD_PTR)(-LNX_MY1CON);
                else
                    grp_affinity.Mask = (DWORD_PTR)(((DWORD_PTR)LNX_MY1CON << cpu_cnt) - 1);
                if (!SetThreadGroupAffinity(GetCurrentThread(), &grp_affinity, &prev_grp_affinity))
                {
                    glbl_obj.error |= _MSGTYP_UNKNOWNERR_OS;
                    return;
                }

                GetThreadGroupAffinity(GetCurrentProcess(), &grp_affinity);
                sum += __internal_daal_countBits(grp_affinity.Mask); // count bits on each target affinity group

                if (sum > lcl_OSProcessorCount)
                {
                    //throw some exception here, no full affinity for the process
                    glbl_obj.error |= _MSGTYP_USERAFFINITYERR;
                    break;
                }
            }
        }
        if (sum != lcl_OSProcessorCount) // check cumulative bit counts matches processor count
        {
            // if this process is restricted and not able to run on all logical processors managed by OS
            // the LS bytes can be extracted to indicate the affinity restrictions
            glbl_obj.error |= _MSGTYP_USERAFFINITYERR + sum;
            return;
        }

        for (i = 0; i < lcl_OSProcessorCount; i++)
        {
            __internal_daal_setGenericAffinityBit(&glbl_obj.cpu_generic_processAffinity, i);
        }

        return;
    }
        #else
    {
        if (lcl_OSProcessorCount > MAX_PREWIN7_LOG_CPU)
        {
            glbl_obj.error |= _MSGTYP_OSAFFCAP_ERROR; // If the os supports more processors than existing win32 or win64 API,
                                                      // we need to know the new API interface in that OS
        }
        GetProcessAffinityMask(GetCurrentProcess(), &processAffinity, &systemAffinity);
        sum = __internal_daal_countBits(processAffinity);
        if (lcl_OSProcessorCount != (unsigned long)sum)
        {
            //throw some exception here, no full affinity for the process
            glbl_obj.error |= _MSGTYP_USERAFFINITYERR + sum;
        }

        if (lcl_OSProcessorCount != (unsigned long)__internal_daal_countBits(systemAffinity))
        {
            //throw some exception here, no full system affinity
            glbl_obj.error |= _MSGTYP_UNKNOWNERR_OS;
        }
    }
        #endif
    for (i = 0; i < lcl_OSProcessorCount; i++)
    {
        // This logic assumes that looping over OSProcCount will let us inspect all the affinity bits
        // That is, we can't need more than OSProcessorCount bits in the affinityMask
        if (((unsigned long)systemAffinity & (LNX_MY1CON << i)) == 0)
            glbl_obj.error |= _MSGTYP_USERAFFINITYERR;
        else
            __internal_daal_setGenericAffinityBit(&glbl_obj.cpu_generic_systemAffinity, i);

        if (((unsigned long)processAffinity & (LNX_MY1CON << i)) == 0)
            glbl_obj.error |= _MSGTYP_USERAFFINITYERR;
        else
            __internal_daal_setGenericAffinityBit(&glbl_obj.cpu_generic_processAffinity, i);
    }
    #endif
}

static void __internal_daal_getCpuidInfo(CPUIDinfo * info, unsigned int eax, unsigned int ecx)
{
    uint32_t abcd[4];
    run_cpuid(eax, ecx, abcd);
    info->EAX = abcd[0];
    info->EBX = abcd[1];
    info->ECX = abcd[2];
    info->EDX = abcd[3];
}

/*
 * __internal_daal_cpuid_
 *
 * Returns the raw data reported in 4 registers by _internal_daal_CPUID, support sub-leaf reporting
 *          The _internal_daal_CPUID instrinsic in MSC does not support sub-leaf reporting
 *
 * Arguments:
 *     info       Point to strucrture to get output from CPIUD instruction
 *     func       _internal_daal_CPUID Leaf number
 *     subfunc    _internal_daal_CPUID Subleaf number
 * Return:        None
 */
static void __cdecl __internal_daal_cpuid_(CPUIDinfo * info, const unsigned int func, const unsigned int subfunc)
{
    __internal_daal_getCpuidInfo(info, func, subfunc);
}

// Simpler version for __internal_daal_cpuid leaf that do not require sub-leaf reporting
static void __internal_daal_cpuid(CPUIDinfo * info, const unsigned int func)
{
    __internal_daal_cpuid_(info, func, 0);
}

/*
 * __internal_daal_getBitsFromDWORD
 *
 * Returns of bits [to:from] of DWORD
 *
 * Arguments:
 *     val        DWORD to extract bits from
 *     from       Low order bit
 *     to         High order bit
 * Return:        Specified bits from DWORD val
 */
static unsigned long __internal_daal_getBitsFromDWORD(const unsigned int val, const char from, const char to)
{
    if (to == 31) return val >> from;

    unsigned long mask = (1 << (to + 1)) - 1;

    return (val & mask) >> from;
}

/*
 * __internal_daal_countBits
 *
 *  count the number of bits that are set to 1 from the input
 *
 * Arguments:
 *     x - Argument to count set bits in, no restriction on set bits distribution
 * Return: Number of bits set to 1
 */
static int __internal_daal_countBits(DWORD_PTR x)
{
    int res = 0, i;
    LNX_PTR2INT myll;

    myll = (LNX_PTR2INT)(x);
    for (i = 0; i < (8 * sizeof(myll)); i++)
    {
        if ((myll & (LNX_MY1CON << i)) != 0)
        {
            res++;
        }
    }

    return res;
}

/*
 * __internal_daal_myBitScanReverse
 *
 * Returns of bits [to:from] of DWORD
 * This c-emulation of the BSR instruction is shown here for tool portability
 *
 * Arguments:
 *     index      bit offset of the most significant bit that's not 0 found in mask
 *     mask       input data to search the most significant bit
  * Return:        1 if a non-zero bit is found, otherwise 0
 */
static unsigned char __internal_daal_myBitScanReverse(unsigned * index, unsigned long mask)
{
    unsigned long i;

    for (i = (8 * sizeof(unsigned long)); i > 0; i--)
    {
        if ((mask & (LNX_MY1CON << (i - 1))) != 0)
        {
            *index = (unsigned long)(i - 1);
            break;
        }
    }

    return (unsigned char)(mask != 0);
}

/* __internal_daal_createMask
 *
 * Derive a bit mask and associated mask width (# of bits) such that
 *  the bit mask is wide enough to select the specified number of
 *  distinct values "numEntries" within the bit field defined by maskWidth.
 *
 * Arguments:
 *     numEntries - The number of entries in the bit field for which a mask needs to be created
 *     maskWidth - Optional argument, pointer to argument that get the mask width (# of bits)
 * Return: Created mask of all 1's up to the maskWidth
 */
static unsigned __internal_daal_createMask(unsigned numEntries, unsigned * maskWidth)
{
    unsigned i;
    unsigned long k;

    // NearestPo2(numEntries) is the nearest power of 2 integer that is not less than numEntries
    // The most significant bit of (numEntries * 2 -1) matches the above definition

    k = (unsigned long)(numEntries)*2 - 1;

    if (__internal_daal_myBitScanReverse(&i, k) == 0)
    {
        // No bits set
        if (maskWidth) *maskWidth = 0;

        return 0;
    }

    if (maskWidth) *maskWidth = i;

    if (i == 31) return (unsigned)-1;

    return (1 << i) - 1;
}

/* __internal_daal_allocateGenericAffinityMask
 *
 * Allocate the memory needed for the bitmap of a generic affinity mask
 *  The memory buffer needed must be large enough to cover the specified maxcpu number
 *  Each cpu is represented by one bit in the bitmap
 *
 * Arguments:
 *     GenericAffintyMask - ptr to be filled in.
 *     maxcpu - max number of logical processors the bitmap of this generic affinity mask needs to  handle
 * Return: 0 if no errors, -1 otherwise
 */
static int __internal_daal_allocateGenericAffinityMask(GenericAffinityMask * pAffinityMap, unsigned maxcpu)
{
    int bytes;
    // Allocate an unsigned char string long enough (cpus/8 + 1) bytes to have 1 bit per cpu.
    // cpu 0 is the lsb of byte0, cpu1 is byte[0]:1, cpu2 is byte[0]:2 etc

    bytes                      = (maxcpu >> 3) + 1;
    pAffinityMap->AffinityMask = (unsigned char *)_INTERNAL_DAAL_MALLOC(bytes * sizeof(unsigned char));
    if (pAffinityMap->AffinityMask == NULL) return -1;

    pAffinityMap->maxByteLength = bytes;
    _INTERNAL_DAAL_MEMSET(pAffinityMap->AffinityMask, 0, bytes * sizeof(unsigned char));

    return 0;
}

/*
 * __internal_daal_freeGenericAffinityMask
 *
 * free up the memory allocated to the bitmap of a generic affinity mask
 *
 * Arguments:
 *     pAffinityMap - pointer to a generic affinity mask
 * Return: none
 */
static void __internal_daal_freeGenericAffinityMask(GenericAffinityMask * pAffinityMap)
{
    _INTERNAL_DAAL_FREE(pAffinityMap->AffinityMask);
    pAffinityMap->maxByteLength = 0;
}

/*
 * __internal_daal_clearGenericAffinityMask
 *
 * Clear all the affinity bits in the bitmap of a generic affinity mask
 *
 * Arguments:
 *     pAffinityMap - pointer to a generic affinity mask
 * Return: none
 */
static void __internal_daal_clearGenericAffinityMask(GenericAffinityMask * pAffinityMap)
{
    _INTERNAL_DAAL_MEMSET(pAffinityMap->AffinityMask, 0, pAffinityMap->maxByteLength);
}

/*
 * __internal_daal_setGenericAffinityBit
 *
 * Set the affinity bit corresponding to a specified logical processor .
 * The bitmap in a generic affinity mask is
 *
 * Arguments:
 *     pAffinityMap - pointer to a generic affinity mask
 *     cpu - an ordinal number that reference a logical processor visible to the OS
 * Return: none, abort if error occured
 */
static void __internal_daal_setGenericAffinityBit(GenericAffinityMask * pAffinityMap, unsigned cpu)
{
    if (cpu < (pAffinityMap->maxByteLength << 3)) pAffinityMap->AffinityMask[cpu >> 3] |= 1 << (cpu % 8);
}

/*
 * __internal_daal_compareEqualGenericAffinity
 *
 * compare two generic affinity masks if the identical set of
 * logical processors are set in two generic affinity mask bitmaps.
 * Since each generic affinity mask is allocated by user and can be of different length,
 * if the length of one mask is shorter then check the shorter part first.
 * If the second mask is longer than the first mask, check that the longer part is all zero.
 *
 * Arguments:
 *     pAffinityMap1 - pointer to a generic affinity mask
 *     pAffinityMap2 - pointer to another generic affinity mask
 * Return: 0 if equal, 1 otherwise
 */
static int __internal_daal_compareEqualGenericAffinity(GenericAffinityMask * pAffinityMap1, GenericAffinityMask * pAffinityMap2)
{
    int rc;
    unsigned i, smaller;

    smaller = pAffinityMap1->maxByteLength;
    if (smaller > pAffinityMap2->maxByteLength) smaller = pAffinityMap2->maxByteLength;
    if (!smaller) return 1;

    rc = memcmp(pAffinityMap1->AffinityMask, pAffinityMap2->AffinityMask, smaller);
    if (rc != 0)
    {
        return 1;
    }
    else
    {
        if (pAffinityMap1->maxByteLength == pAffinityMap2->maxByteLength)
        {
            return 0;
        }
        else if (pAffinityMap1->maxByteLength > pAffinityMap2->maxByteLength)
        {
            for (i = smaller; i < pAffinityMap1->maxByteLength; i++)
            {
                if (pAffinityMap1->AffinityMask[i] != 0) return 1;
            }

            return 0;
        }
        else
        {
            for (i = smaller; i < pAffinityMap2->maxByteLength; i++)
            {
                if (pAffinityMap2->AffinityMask[i] != 0) return 1;
            }
            return 0;
        }
    }
}

/*
 * __internal_daal_clearGenericAffinityBit
 *
 * Clear (set to 0) the cpu'th bit in the generic affinity mask.
 *
 * Arguments:
 *     generic affinty mask ptr
 * Return: 0 if no error, -1 otherise
 */
static int __internal_daal_clearGenericAffinityBit(GenericAffinityMask * pAffinityMap, unsigned cpu)
{
    if (cpu < (pAffinityMap->maxByteLength << 3))
    {
        pAffinityMap->AffinityMask[cpu >> 3] ^= 1 << (cpu % 8);
        return 0;
    }
    else
    {
        return -1;
    }
}

/*
 * __internal_daal_testGenericAffinityBit
 *
 * check the cpu'th bit of the affinity mask and return 1 if the bit is set and 0 if the bit is 0.
 *
 * Arguments:
 *     generic affinty mask ptr
 *     cpu - cpu number. Signifies bit in the unsigned char affinity mask to be checked
 * Return: 0 if the cpu's bit is clear, returns 1 if the bit is set, -1 if an error
 *   The error condition makes it where you can't just check for '0' or 'not 0'
 *
 */
static unsigned char __internal_daal_testGenericAffinityBit(GenericAffinityMask * pAffinityMap, unsigned cpu)
{
    if (cpu < (pAffinityMap->maxByteLength << 3))
    {
        if ((pAffinityMap->AffinityMask[cpu >> 3] & (1 << (cpu % 8))))
            return 1;
        else
            return 0;
    }
    else
    {
        return 0xff;
    }
}

/*
 * __internal_daal_getApicID
 *
 * Returns APIC ID from leaf B if it else from leaf 1
 *
 * Arguments: None
 * Return: APIC ID
 */
static unsigned __internal_daal_getApicID()
{
    CPUIDinfo info;

    if (glbl_obj.hasLeafB)
    {
        __internal_daal_cpuid(&info, 0xB); // query subleaf 0 of leaf B
        return info.EDX;                   //  x2APIC ID
    }

    __internal_daal_cpuid(&info, 1);

    return (BYTE)(__internal_daal_getBitsFromDWORD(info.EBX, 24, 31)); // zero extend 8-bit initial APIC ID
}

// select the system-wide ordinal number of the first logical processor the is located
// within the entity specified from the hierarchical ordinal number scheme
// package: ordinal number of a package; ENUM_ALL means any package
// core   : ordinal number of a core in the specified package; ENUM_ALL means any core
// logical: ordinal number of a logical processor within the specifed core; ; ENUM_ALL means any thread
// returns: the system wide ordinal number of the first logical processor that matches the
//    specified (package, core, logical) triplet specification.
static unsigned __internal_daal_slectOrdfromPkg(unsigned package, unsigned core, unsigned logical)
{
    unsigned i;

    if (package != ENUM_ALL && package >= _internal_daal_GetSysProcessorPackageCount()) return 0;

    for (i = 0; i < _internal_daal_GetOSLogicalProcessorCount(); i++)
    {
        ////
        if (!(package == ENUM_ALL || glbl_obj.pApicAffOrdMapping[i].packageORD == package)) continue;

        if (!(core == ENUM_ALL || core == glbl_obj.pApicAffOrdMapping[i].coreORD)) continue;

        if (!(logical == ENUM_ALL || logical == glbl_obj.pApicAffOrdMapping[i].threadORD)) continue;

        return i;
    }

    return 0;
}

// Derive bitmask extraction parameters used to extract/decompose x2APIC ID.
// The algorithm assumes __internal_daal_cpuid feature symmetry across all physical packages.
// Since __internal_daal_cpuid reporting by each logical processor in a physical package are identical, we only execute __internal_daal_cpuid
// on one logical processor to derive these system-wide parameters
static int __internal_daal_cpuTopologyLeafBConstants()
{
    CPUIDinfo infoB;
    int wasCoreReported            = 0;
    int wasThreadReported          = 0;
    int subLeaf                    = 0, levelType, levelShift;
    unsigned long coreplusSMT_Mask = 0;

    do
    {
        // we already tested __internal_daal_cpuid leaf 0BH contain valid sub-leaves
        __internal_daal_cpuid_(&infoB, 0xB, subLeaf);
        if (infoB.EBX == 0)
        {
            // if EBX ==0 then this subleaf is not valid, we can exit the loop
            break;
        }

        levelType  = __internal_daal_getBitsFromDWORD(infoB.ECX, 8, 15);
        levelShift = __internal_daal_getBitsFromDWORD(infoB.EAX, 0, 4);

        switch (levelType)
        {
        case 1:
            // level type is SMT, so levelShift is the SMT_Mask_Width
            glbl_obj.SMTSelectMask = ~((-1) << levelShift);
            glbl_obj.SMTMaskWidth  = levelShift;
            wasThreadReported      = 1;
            break;
        case 2:
            // level type is Core, so levelShift is the CorePlsuSMT_Mask_Width
            coreplusSMT_Mask            = ~((-1) << levelShift);
            glbl_obj.PkgSelectMaskShift = levelShift;
            glbl_obj.PkgSelectMask      = (-1) ^ coreplusSMT_Mask;
            wasCoreReported             = 1;
            break;
        default:
            // handle in the future
            break;
        }

        subLeaf++;
    } while (1);

    if (wasThreadReported && wasCoreReported)
    {
        glbl_obj.CoreSelectMask = coreplusSMT_Mask ^ glbl_obj.SMTSelectMask;
    }
    else if (!wasCoreReported && wasThreadReported)
    {
        glbl_obj.CoreSelectMask     = 0;
        glbl_obj.PkgSelectMaskShift = glbl_obj.SMTMaskWidth;
        glbl_obj.PkgSelectMask      = (-1) ^ glbl_obj.SMTSelectMask;
    }
    else //(case where !wasThreadReported)
    {
        // throw an error, this should not happen if hardware function normally
        glbl_obj.error |= _MSGTYP_GENERAL_ERROR;
    }

    if (glbl_obj.error) return -1;

    return 0;
}

// Calculate parameters used to extract/decompose Initial APIC ID.
// The algorithm assumes __internal_daal_cpuid feature symmetry across all physical packages.
// Since __internal_daal_cpuid reporting by each logical processor in a physical package are identical, we only execute __internal_daal_cpuid
// on one logical processor to derive these system-wide parameters
/*
 * __internal_daal_cpuTopologyLegacyConstants
 *
 * Derive bitmask extraction parameter using __internal_daal_cpuid leaf 1 and leaf 4
 *
 * Arguments:
 *     info - Point to strucrture containing CPIUD instruction leaf 1 data
 *     maxCPUID - Maximum __internal_daal_cpuid Leaf number supported by the processor
 * Return: 0 is no error
 */
static int __internal_daal_cpuTopologyLegacyConstants(CPUIDinfo * pinfo, DWORD maxCPUID)
{
    unsigned corePlusSMTIDMaxCnt;
    unsigned coreIDMaxCnt       = 1;
    unsigned SMTIDPerCoreMaxCnt = 1;

    corePlusSMTIDMaxCnt = __internal_daal_getBitsFromDWORD(pinfo->EBX, 16, 23);
    if (maxCPUID >= 4)
    {
        CPUIDinfo info4;
        __internal_daal_cpuid_(&info4, 4, 0);
        coreIDMaxCnt       = __internal_daal_getBitsFromDWORD(info4.EAX, 26, 31) + 1; // related to coreMaskWidth
        SMTIDPerCoreMaxCnt = corePlusSMTIDMaxCnt / coreIDMaxCnt;
    }
    else
    {
        // no support for __internal_daal_cpuid leaf 4 but caller has verified  HT support
        if (!glbl_obj.Alert_BiosCPUIDmaxLimitSetting)
        {
            coreIDMaxCnt       = 1;
            SMTIDPerCoreMaxCnt = corePlusSMTIDMaxCnt / coreIDMaxCnt;
        }
        else
        {
            // we got here most likely because IA32_MISC_ENABLES[22] was set to 1 by BIOS
            glbl_obj.error |= _MSGTYP_CHECKBIOS_CPUIDMAXSETTING; // IA32_MISC_ENABLES[22] may have been set to 1, will cause inaccurate reporting
        }
    }

    glbl_obj.SMTSelectMask  = __internal_daal_createMask(SMTIDPerCoreMaxCnt, &glbl_obj.SMTMaskWidth);
    glbl_obj.CoreSelectMask = __internal_daal_createMask(coreIDMaxCnt, &glbl_obj.PkgSelectMaskShift);
    glbl_obj.PkgSelectMaskShift += glbl_obj.SMTMaskWidth;
    glbl_obj.CoreSelectMask <<= glbl_obj.SMTMaskWidth;
    glbl_obj.PkgSelectMask = (-1) ^ (glbl_obj.CoreSelectMask | glbl_obj.SMTSelectMask);

    return 0;
}

/*
 * __internal_daal_getCacheTotalLize
 *
 * Caluculates the total capacity (bytes) from the cache parameters reported by __internal_daal_cpuid leaf 4
 *          the caller provides the raw data reported by the target sub leaf of cpuid leaf 4
 *
 * Arguments:
 *     maxCPUID - Maximum __internal_daal_cpuid Leaf number supported by the processor, provided by parent
 *     cache_type - the cache type encoding recognized by CPIUD instruction leaf 4 for the target cache level
 * Return: the sub-leaf index corresponding to the largest cache of specified cache type
 */
static unsigned long __internal_daal_getCacheTotalLize(CPUIDinfo info)
{
    unsigned long LnSz, SectorSz, WaySz, SetSz;

    LnSz     = __internal_daal_getBitsFromDWORD(info.EBX, 0, 11) + 1;
    SectorSz = __internal_daal_getBitsFromDWORD(info.EBX, 12, 21) + 1;
    WaySz    = __internal_daal_getBitsFromDWORD(info.EBX, 22, 31) + 1;
    SetSz    = __internal_daal_getBitsFromDWORD(info.ECX, 0, 31) + 1;

    return (SetSz * WaySz * SectorSz * LnSz);
}

/*
 * __internal_daal_findEachCacheIndex
 *
 * Find the subleaf index of __internal_daal_cpuid leaf 4 corresponding to the input subleaf
 *
 * Arguments:
 *     maxCPUID - Maximum __internal_daal_cpuid Leaf number supported by the processor, provided by parent
 *     cache_subleaf - the cache subleaf encoding recognized by CPIUD instruction leaf 4 for the target cache level
 * Return: the sub-leaf index corresponding to the largest cache of specified cache type
 */
static int __internal_daal_findEachCacheIndex(DWORD maxCPUID, unsigned cache_subleaf)
{
    unsigned i, type;
    unsigned long cap;
    CPUIDinfo info4;
    int target_index = -1;

    if (maxCPUID < 4)
    {
        // if we can't get detailed parameters from cpuid leaf 4, this is stub pointers for older processors
        if (maxCPUID >= 2)
        {
            if (!cache_subleaf)
            {
                glbl_obj.cacheDetail[cache_subleaf].sizeKB = 0;
                return cache_subleaf;
            }
        }
        return -1;
    }

    __internal_daal_cpuid_(&info4, 4, cache_subleaf);
    type = __internal_daal_getBitsFromDWORD(info4.EAX, 0, 4);
    cap  = __internal_daal_getCacheTotalLize(info4);

    if (type > 0)
    {
        glbl_obj.cacheDetail[cache_subleaf].level = __internal_daal_getBitsFromDWORD(info4.EAX, 5, 7);
        glbl_obj.cacheDetail[cache_subleaf].type  = type;
        for (i = 0; i <= cache_subleaf; i++)
        {
            if (glbl_obj.cacheDetail[i].type == type) glbl_obj.cacheDetail[i].how_many_caches_share_level++;
        }
        glbl_obj.cacheDetail[cache_subleaf].sizeKB = cap / 1024;
        target_index                               = cache_subleaf;
    }

    return target_index;
}

/*
 * __internal_daal_initStructuredLeafBuffers
 *
 * Allocate buffers to store cpuid leaf data including buffers for subleaves within leaf 4 and 11
 *
 * Arguments: none
 * Return: none
 *
 */
static void __internal_daal_initStructuredLeafBuffers()
{
    unsigned j, kk, qeidmsk;
    unsigned maxCPUID;
    CPUIDinfo info;

    __internal_daal_cpuid(&info, 0);
    maxCPUID                            = info.EAX;
    glbl_obj.cpuid_values[0].subleaf[0] = (CPUIDinfo *)_INTERNAL_DAAL_MALLOC(sizeof(CPUIDinfo));
    _INTERNAL_DAAL_MEMCPY(glbl_obj.cpuid_values[0].subleaf[0], sizeof(CPUIDinfo), &info, 4 * sizeof(unsigned int));
    // Mark this combo of cpu, leaf, subleaf is valid
    glbl_obj.cpuid_values[0].subleaf_max = 1;

    for (j = 1; j <= maxCPUID; j++)
    {
        __internal_daal_cpuid(&info, j);
        glbl_obj.cpuid_values[j].subleaf[0] = (CPUIDinfo *)_INTERNAL_DAAL_MALLOC(sizeof(CPUIDinfo));
        _INTERNAL_DAAL_MEMCPY(glbl_obj.cpuid_values[j].subleaf[0], sizeof(CPUIDinfo), &info, 4 * sizeof(unsigned int));
        glbl_obj.cpuid_values[j].subleaf_max = 1;

        if (j == 0xd)
        {
            int subleaf                          = 2;
            glbl_obj.cpuid_values[j].subleaf_max = 1;
            __internal_daal_cpuid_(&info, j, subleaf);
            while (info.EAX && subleaf < MAX_CACHE_SUBLEAFS)
            {
                glbl_obj.cpuid_values[j].subleaf_max = subleaf;
                subleaf++;
                __internal_daal_cpuid_(&info, j, subleaf);
            }
        }
        else if (j == 0x10 || j == 0xf)
        {
            int subleaf = 1;
            __internal_daal_cpuid_(&info, j, subleaf);
            if (j == 0xf)
                qeidmsk = info.EDX; // sub-leaf value are derived from valid resource id's
            else
                qeidmsk = info.EBX;
            kk = 1;
            while (kk < 32)
            {
                __internal_daal_cpuid_(&info, j, kk);
                if ((qeidmsk & (1 << kk)) != 0) glbl_obj.cpuid_values[j].subleaf_max = kk;
                kk++;
            }
        }
        else if ((j == 0x4 || j == 0xb))
        {
            int subleaf       = 1;
            unsigned int type = 1;
            while (type && subleaf < MAX_CACHE_SUBLEAFS)
            {
                __internal_daal_cpuid_(&info, j, subleaf);
                if (j == 0x4)
                    type = __internal_daal_getBitsFromDWORD(info.EAX, 0, 4);
                else
                    type = 0xffff & info.EBX;
                glbl_obj.cpuid_values[j].subleaf[subleaf] = (CPUIDinfo *)_INTERNAL_DAAL_MALLOC(sizeof(CPUIDinfo));
                _INTERNAL_DAAL_MEMCPY(glbl_obj.cpuid_values[j].subleaf[subleaf], sizeof(CPUIDinfo), &info, 4 * sizeof(unsigned int));
                subleaf++;
                glbl_obj.cpuid_values[j].subleaf_max = subleaf;
            }
        }
    }
}

/*
 * __internal_daal_eachCacheTopologyParams
 *
 * Calculates the select mask that can be used to extract Cache_ID from an APIC ID
 *          the caller must specify which target cache level it wishes to extract Cache_IDs
 *
 * Arguments:
 *     targ_subleaf - the subleaf index to execute CPIUD instruction leaf 4 for the target cache level, provided by parent
 *     maxCPUID - Maximum __internal_daal_cpuid Leaf number supported by the processor, provided by parent
 * Return: 0 is no error
 */
static int __internal_daal_eachCacheTopologyParams(unsigned targ_subleaf, DWORD maxCPUID)
{
    unsigned long SMTMaxCntPerEachCache;
    CPUIDinfo info;

    __internal_daal_cpuid(&info, 1);
    if (maxCPUID >= 4)
    {
        CPUIDinfo info4;
        __internal_daal_cpuid_(&info4, 4, targ_subleaf);

        SMTMaxCntPerEachCache = __internal_daal_getBitsFromDWORD(info4.EAX, 14, 25) + 1;
    }
    else if (__internal_daal_getBitsFromDWORD(info.EDX, 28, 28))
    {
        // no support for __internal_daal_cpuid leaf 4 but HT is supported
        SMTMaxCntPerEachCache = __internal_daal_getBitsFromDWORD(info.EBX, 16, 23);
    }
    else
    {
        // no support for __internal_daal_cpuid leaf 4 and no HT
        SMTMaxCntPerEachCache = 1;
    }

    glbl_obj.EachCacheSelectMask[targ_subleaf] = __internal_daal_createMask(SMTMaxCntPerEachCache, &glbl_obj.EachCacheMaskWidth[targ_subleaf]);

    return 0;
}

// Calculate parameters used to extract/decompose APIC ID for cache topology
// The algorithm assumes __internal_daal_cpuid feature symmetry across all physical packages.
// Since __internal_daal_cpuid reporting by each logical processor in a physical package are identical, we only execute __internal_daal_cpuid
// on one logical processor to derive these system-wide parameters
// return 0 if successful, non-zero if error occurred
static int __internal_daal_cacheTopologyParams()
{
    DWORD maxCPUID;
    CPUIDinfo info;
    int targ_index;

    __internal_daal_cpuid_(&info, 0, 0);
    maxCPUID = info.EAX;

    // Let's also examine cache topology.
    // As an example choose the largest unified cache as target level
    if (maxCPUID >= 4)
    {
        unsigned subleaf;
        __internal_daal_initStructuredLeafBuffers();

        glbl_obj.maxCacheSubleaf = 0;

        for (subleaf = 0; subleaf < glbl_obj.cpuid_values[4].subleaf_max; subleaf++)
        {
            targ_index = __internal_daal_findEachCacheIndex(maxCPUID, subleaf); // unified cache is type 3 under leaf 4
            if (targ_index >= 0)
            {
                glbl_obj.maxCacheSubleaf = targ_index;
                __internal_daal_eachCacheTopologyParams(targ_index, maxCPUID);
            }
            else
            {
                break;
            }
        }
    }
    else if (maxCPUID >= 2)
    {
        int subleaf;
        glbl_obj.maxCacheSubleaf = 0;

        for (subleaf = 0; subleaf < 4; subleaf++)
        {
            targ_index = __internal_daal_findEachCacheIndex(maxCPUID, subleaf);
            if (targ_index >= 0)
            {
                glbl_obj.maxCacheSubleaf = targ_index;
                __internal_daal_eachCacheTopologyParams(targ_index, maxCPUID);
            }
            else
            {
                break;
            }
        }
    }

    if (glbl_obj.error) return -1;

    return 0;
}

// Derive parameters used to extract/decompose APIC ID for CPU topology
// The algorithm assumes __internal_daal_cpuid feature symmetry across all physical packages.
// Since __internal_daal_cpuid reporting by each logical processor in a physical package are
// identical, we only execute __internal_daal_cpuid on one logical processor to derive these
// system-wide parameters
// return 0 if successful, non-zero if error occurred
static int __internal_daal_cpuTopologyParams()
{
    DWORD maxCPUID; // highest __internal_daal_cpuid leaf index this processor supports
    CPUIDinfo info; // data structure to store register data reported by __internal_daal_cpuid

    __internal_daal_cpuid_(&info, 0, 0);
    maxCPUID = info.EAX;

    // cpuid leaf B detection
    if (maxCPUID >= 0xB)
    {
        CPUIDinfo CPUInfoB;
        __internal_daal_cpuid_(&CPUInfoB, 0xB, 0);
        //glbl_ptr points to assortment of global data, workspace, etc
        glbl_obj.hasLeafB = (CPUInfoB.EBX != 0);
    }

    __internal_daal_cpuid_(&info, 1, 0);

    // Use HWMT feature flag __internal_daal_cpuid.01:EDX[28] to treat three configurations:
    if (__internal_daal_getBitsFromDWORD(info.EDX, 28, 28))
    {
        if (glbl_obj.hasLeafB)
        {
            // #1, Processors that support __internal_daal_cpuid leaf 0BH
            // use __internal_daal_cpuid leaf B to derive extraction parameters
            __internal_daal_cpuTopologyLeafBConstants();
        }
        else
        {
            //#2, Processors that support legacy parameters
            //  using __internal_daal_cpuid leaf 1 and leaf 4
            __internal_daal_cpuTopologyLegacyConstants(&info, maxCPUID);
        }
    }
    else
    {
        //#3, Prior to HT, there is only one logical processor in a physical package
        glbl_obj.CoreSelectMask     = 0;
        glbl_obj.SMTMaskWidth       = 0;
        glbl_obj.PkgSelectMask      = (unsigned)(-1);
        glbl_obj.PkgSelectMaskShift = 0;
        glbl_obj.SMTSelectMask      = 0;
    }

    if (glbl_obj.error) return -1;

    return 0;
}

/*
 * __internal_daal_allocArrays
 *
 * allocate the dynamic arrays in the glbl_ptr containing various buffers for analyzing
 *  cpu topology and cache topology of a system with N logical processors
 *
 * Arguments: number of logical processors
 * Return: 0 is no error, -1 is error
 */
static int __internal_daal_allocArrays(unsigned cpus)
{
    unsigned i;

    i                           = cpus + 1;
    glbl_obj.pApicAffOrdMapping = (idAffMskOrdMapping_t *)_INTERNAL_DAAL_MALLOC(i * sizeof(idAffMskOrdMapping_t));
    _INTERNAL_DAAL_MEMSET(glbl_obj.pApicAffOrdMapping, 0, i * sizeof(idAffMskOrdMapping_t));

    glbl_obj.perPkg_detectedCoresCount.data = (unsigned *)_INTERNAL_DAAL_MALLOC(i * sizeof(unsigned));
    _INTERNAL_DAAL_MEMSET(glbl_obj.perPkg_detectedCoresCount.data, 0, i * sizeof(unsigned));
    glbl_obj.perPkg_detectedCoresCount.dim[0] = i;

    glbl_obj.perCore_detectedThreadsCount.data = (unsigned *)_INTERNAL_DAAL_MALLOC(MAX_CORES * i * sizeof(unsigned));
    _INTERNAL_DAAL_MEMSET(glbl_obj.perCore_detectedThreadsCount.data, 0, MAX_CORES * i * sizeof(unsigned));
    glbl_obj.perCore_detectedThreadsCount.dim[0] = i;
    glbl_obj.perCore_detectedThreadsCount.dim[1] = MAX_CORES;

    // workspace for storing hierarchical counts relative to the cache topology
    // of the largest unified cache (may be shared by several cores)
    glbl_obj.perCache_detectedCoreCount.data = (unsigned *)_INTERNAL_DAAL_MALLOC(i * sizeof(unsigned));
    _INTERNAL_DAAL_MEMSET(glbl_obj.perCache_detectedCoreCount.data, 0, i * sizeof(unsigned));
    glbl_obj.perCache_detectedCoreCount.dim[0] = i;

    glbl_obj.perEachCache_detectedThreadCount.data = (unsigned *)_INTERNAL_DAAL_MALLOC(MAX_CACHE_SUBLEAFS * i * sizeof(unsigned));
    _INTERNAL_DAAL_MEMSET(glbl_obj.perEachCache_detectedThreadCount.data, 0, MAX_CACHE_SUBLEAFS * i * sizeof(unsigned));
    glbl_obj.perEachCache_detectedThreadCount.dim[0] = i;
    glbl_obj.perEachCache_detectedThreadCount.dim[1] = MAX_CACHE_SUBLEAFS;

    glbl_obj.cpuid_values = (CPUIDinfox *)_INTERNAL_DAAL_MALLOC(MAX_LEAFS * i * sizeof(CPUIDinfox));
    _INTERNAL_DAAL_MEMSET(glbl_obj.cpuid_values, 0, MAX_LEAFS * i * sizeof(CPUIDinfox));

    return 0;
}

/*
 * __internal_daal_parseIDS4EachThread
 *
 * after execution context has already bound to the target logical processor
 * Query the 32-bit x2APIC ID if the processor supports it, or
 * Query the 8bit initial APIC ID for older processors
 * Apply various system-wide topology constant to parse the APIC ID into various sub IDs
 *
 * Arguments:
 *      i - the ordinal index to reference a logical processor in the system
 *      numMappings - running count ot how many processors we've parsed
 * Return: 0 is no error
 */
static unsigned __internal_daal_parseIDS4EachThread(unsigned i, unsigned numMappings)
{
    unsigned APICID;
    unsigned subleaf;

    APICID = glbl_obj.pApicAffOrdMapping[numMappings].APICID = __internal_daal_getApicID();
    glbl_obj.pApicAffOrdMapping[numMappings].OrdIndexOAMsk   = i; // this an ordinal number that can relate to generic affinitymask
    glbl_obj.pApicAffOrdMapping[numMappings].pkg_IDAPIC      = ((APICID & glbl_obj.PkgSelectMask) >> glbl_obj.PkgSelectMaskShift);
    glbl_obj.pApicAffOrdMapping[numMappings].Core_IDAPIC     = ((APICID & glbl_obj.CoreSelectMask) >> glbl_obj.SMTMaskWidth);
    glbl_obj.pApicAffOrdMapping[numMappings].SMT_IDAPIC      = (APICID & glbl_obj.SMTSelectMask);

    if (glbl_obj.maxCacheSubleaf != -1)
    {
        for (subleaf = 0; subleaf <= glbl_obj.maxCacheSubleaf; subleaf++)
        {
            glbl_obj.pApicAffOrdMapping[numMappings].EaCacheSMTIDAPIC[subleaf] = (APICID & glbl_obj.EachCacheSelectMask[subleaf]);
            glbl_obj.pApicAffOrdMapping[numMappings].EaCacheIDAPIC[subleaf]    = (APICID & (-1 ^ glbl_obj.EachCacheSelectMask[subleaf]));
        }
    }

    return 0;
}

/*
 * __internal_daal_queryParseSubIDs
 *
 * Use OS specific service to find out how many logical processors can be accessed
 * by this application.
 * Querying __internal_daal_cpuid on each logical processor requires using OS-specific API to
 * bind current context to each logical processor first.
 * After gathering the APIC ID's for each logical processor,
 * we can parse APIC ID into sub IDs for each topological levels
 * The thread affnity API to bind the current context limits us
 * in dealing with the limit of specific OS
 * The iteration loop to go through each logical processor managed by the OS can be done
 * in a manner that abstract the OS-specific affinity mask data structure.
 * Here, we construct a generic affinity mask that can handle arbitrary number of logical processors.
 *
 * Arguments: none
 * Return: 0 is no error
 */
static int __internal_daal_queryParseSubIDs(void)
{
    unsigned i;
    int numMappings = 0;
    unsigned lcl_OSProcessorCount;
    #if defined(__linux__) || defined(__FreeBSD__)
    cpu_set_t pa;
    #else
        #if (_WIN32_WINNT >= 0x0601)
    GROUP_AFFINITY pa;
        #else // If SDK version does not support GROUP_AFFINITY,
    DWORD_PTR pa;
        #endif
    #endif

    // we already queried OS how many logical processor it sees.
    lcl_OSProcessorCount = glbl_obj.OSProcessorCount;

    // we will use our generic affinity bitmap that can be generalized from
    // OS specific affinity mask constructs or the bitmap representation of an OS
    if (__internal_daal_allocateGenericAffinityMask(&glbl_obj.cpu_generic_processAffinity, lcl_OSProcessorCount)) return -1;
    if (__internal_daal_allocateGenericAffinityMask(&glbl_obj.cpu_generic_systemAffinity, lcl_OSProcessorCount)) return -1;

    // Set the affinity bits of our generic affinity bitmap according to
    // the system affinity mask and process affinity mask
    __internal_daal_setChkProcessAffinityConsistency(lcl_OSProcessorCount);
    if (glbl_obj.error)
    {
        __internal_daal_freeGenericAffinityMask(&glbl_obj.cpu_generic_processAffinity);
        __internal_daal_freeGenericAffinityMask(&glbl_obj.cpu_generic_systemAffinity);
        return -1;
    }

    for (i = 0; i < glbl_obj.OSProcessorCount; i++)
    {
        // can't asume OS affinity bit mask is contiguous,
        // but we are using our generic bitmap representation for affinity
        if (__internal_daal_testGenericAffinityBit(&glbl_obj.cpu_generic_processAffinity, i) == 1)
        {
            // bind the execution context to the ith logical processor
            // using OS-specifi API
            if (__internal_daal_bindContext(i, (void *)(&pa)))
            {
                glbl_obj.error |= _MSGTYP_UNKNOWNERR_OS;
                break;
            }

            // now the execution context is on the i'th cpu, call the parsing routine
            __internal_daal_parseIDS4EachThread(i, numMappings);
            __internal_daal_restoreContext((void *)(&pa));

            numMappings++;
        }
    }

    glbl_obj.EnumeratedThreadCount = numMappings;

    __internal_daal_freeGenericAffinityMask(&glbl_obj.cpu_generic_processAffinity);
    __internal_daal_freeGenericAffinityMask(&glbl_obj.cpu_generic_systemAffinity);

    if (glbl_obj.error) return -1;

    return numMappings;
}

/*
 * __internal_daal_analyzeCPUHierarchy
 *
 * Analyze the Pkg_ID, Core_ID to derive hierarchical ordinal numbering scheme
 *
 * Arguments:
 *      numMappings - the number of logical processors successfully queried with SMT_ID, Core_ID, Pkg_ID extracted
 * Return: 0 is no error
 */
static int __internal_daal_analyzeCPUHierarchy(unsigned numMappings)
{
    unsigned i, ckDim, maxPackageDetetcted = 0;
    unsigned packageID, coreID;
    unsigned *pDetectCoreIDsperPkg, *pDetectedPkgIDs;

    // allocate workspace to sort parents and siblings in the topology
    // starting from pkg_ID and work our ways down each inner level
    pDetectedPkgIDs = (unsigned *)_INTERNAL_DAAL_MALLOC(numMappings * sizeof(unsigned));
    if (pDetectedPkgIDs == NULL) return -1;

    // we got a 1-D array to store unique Pkg_ID as we sort thru
    // each logical processor
    _INTERNAL_DAAL_MEMSET(pDetectedPkgIDs, 0xff, numMappings * sizeof(unsigned));
    ckDim                = numMappings * (1 << glbl_obj.PkgSelectMaskShift);
    pDetectCoreIDsperPkg = (unsigned *)_INTERNAL_DAAL_MALLOC(ckDim * sizeof(unsigned));
    if (pDetectCoreIDsperPkg == NULL)
    {
        _INTERNAL_DAAL_FREE(pDetectedPkgIDs);
        return -1;
    }

    // we got a 2-D array to store unique Core_ID within each Pkg_ID,
    // as we sort thru each logical processor
    _INTERNAL_DAAL_MEMSET(pDetectCoreIDsperPkg, 0xff, ckDim * sizeof(unsigned));

    // iterate throught each logical processor in the system.
    // mark up each unique physical package with a zero-based numbering scheme
    // Within each distinct package, mark up distinct cores within that package
    // with a zero-based numbering scheme
    for (i = 0; i < numMappings; i++)
    {
        BOOL PkgMarked;
        unsigned h;
        packageID = glbl_obj.pApicAffOrdMapping[i].pkg_IDAPIC;
        coreID    = glbl_obj.pApicAffOrdMapping[i].Core_IDAPIC;

        PkgMarked = FALSE;
        for (h = 0; h < maxPackageDetetcted; h++)
        {
            if (pDetectedPkgIDs[h] == packageID)
            {
                BOOL foundCore = FALSE;
                unsigned k;
                PkgMarked                                 = TRUE;
                glbl_obj.pApicAffOrdMapping[i].packageORD = h;

                // look for core in marked packages
                for (k = 0; k < glbl_obj.perPkg_detectedCoresCount.data[h]; k++)
                {
                    if (coreID == pDetectCoreIDsperPkg[h * numMappings + k])
                    {
                        foundCore = TRUE;
                        // add thread - can't be that the thread already exists, breaks uniqe APICID spec
                        glbl_obj.pApicAffOrdMapping[i].coreORD   = k;
                        glbl_obj.pApicAffOrdMapping[i].threadORD = glbl_obj.perCore_detectedThreadsCount.data[h * MAX_CORES + k];
                        glbl_obj.perCore_detectedThreadsCount.data[h * MAX_CORES + k]++;
                        break;
                    }
                }

                if (!foundCore)
                {
                    // mark up the Core_ID of an unmarked core in a marked package
                    unsigned core                                = glbl_obj.perPkg_detectedCoresCount.data[h];
                    pDetectCoreIDsperPkg[h * numMappings + core] = coreID;

                    // keep track of respective hierarchical counts
                    glbl_obj.perCore_detectedThreadsCount.data[h * MAX_CORES + core] = 1;
                    glbl_obj.perPkg_detectedCoresCount.data[h]++;

                    // build a set of numbering system to iterate each topological hierarchy
                    glbl_obj.pApicAffOrdMapping[i].coreORD   = core;
                    glbl_obj.pApicAffOrdMapping[i].threadORD = 0;
                    glbl_obj.EnumeratedCoreCount++; // this is an unmarked core, increment system core count by 1
                }

                break;
            }
        }

        if (!PkgMarked)
        {
            // mark up the pkg_ID and Core_ID of an unmarked package
            pDetectedPkgIDs[maxPackageDetetcted]                        = packageID;
            pDetectCoreIDsperPkg[maxPackageDetetcted * numMappings + 0] = coreID;

            // keep track of respective hierarchical counts
            glbl_obj.perPkg_detectedCoresCount.data[maxPackageDetetcted]                    = 1;
            glbl_obj.perCore_detectedThreadsCount.data[maxPackageDetetcted * MAX_CORES + 0] = 1;

            // build a set of zero-based numbering acheme so that
            // each logical processor in the same core can be referenced by a zero-based index
            // each core in the same package can be referenced by another zero-based index
            // each package in the system can be referenced by a third zero-based index scheme.
            // each system wide index i can be mapped to a triplet of zero-based hierarchical indices
            glbl_obj.pApicAffOrdMapping[i].packageORD = maxPackageDetetcted;
            glbl_obj.pApicAffOrdMapping[i].coreORD    = 0;
            glbl_obj.pApicAffOrdMapping[i].threadORD  = 0;

            maxPackageDetetcted++;          // this is an unmarked pkg, increment pkg count by 1
            glbl_obj.EnumeratedCoreCount++; // there is at least one core in a package
        }
    }

    glbl_obj.EnumeratedPkgCount = maxPackageDetetcted;

    _INTERNAL_DAAL_FREE(pDetectedPkgIDs);
    _INTERNAL_DAAL_FREE(pDetectCoreIDsperPkg);

    return 0;
}

/*
 * __internal_daal_analyzeEachCHierarchy
 *
 *   this is an example illustrating cache topology analysis of the largest unified cache
 *   the largest unified cache may be shared by multiple cores
 *          parse APIC ID into sub IDs for each topological levels
 *  This example illustrates several mapping relationships:
 *      1. count distinct target level cache in the system;     2 count distinct cores sharing the same cache
 *      3. Establish a hierarchical numbering scheme (0-based, ordinal number) for each distinct entity within a hierarchical level
 *
 * Arguments:
 *      numMappings - the number of logical processors successfully queried with SMT_ID, Core_ID, Pkg_ID extracted
 * Return: 0 is no error
 */
static int __internal_daal_analyzeEachCHierarchy(unsigned subleaf, unsigned numMappings)
{
    unsigned i;
    unsigned maxCacheDetected = 0, maxThreadsDetected = 0;
    unsigned threadID;
    unsigned CacheID;
    unsigned *pDetectThreadIDsperEachC, *pDetectedEachCIDs;
    unsigned *pThreadIDsperEachC, *pEachCIDs;

    if (glbl_obj.EachCacheMaskWidth[subleaf] == 0xffffffff) return -1;

    pEachCIDs = (unsigned *)_INTERNAL_DAAL_MALLOC(numMappings * sizeof(unsigned));
    if (pEachCIDs == NULL) return -1;
    _INTERNAL_DAAL_MEMSET(pEachCIDs, 0xff, numMappings * sizeof(unsigned));

    pThreadIDsperEachC = (unsigned *)_INTERNAL_DAAL_MALLOC(numMappings * sizeof(unsigned));
    if (pThreadIDsperEachC == NULL)
    {
        _INTERNAL_DAAL_FREE(pEachCIDs);
        return -1;
    }
    _INTERNAL_DAAL_MEMSET(pThreadIDsperEachC, 0xff, numMappings * sizeof(unsigned));

    // enumerate distinct caches of the same subleaf index to get counts of how many caches only
    // mark up each unique cache associated with subleaf index based on the cache_ID
    maxCacheDetected = 0;
    for (i = 0; i < numMappings; i++)
    {
        unsigned j;
        for (j = 0; j < maxCacheDetected; j++)
        {
            if (pEachCIDs[j] == glbl_obj.pApicAffOrdMapping[i].EaCacheIDAPIC[subleaf])
            {
                break;
            }
        }
        if (j >= maxCacheDetected)
        {
            pEachCIDs[maxCacheDetected++] = glbl_obj.pApicAffOrdMapping[i].EaCacheIDAPIC[subleaf];
        }
    }

    // enumerate distinct SMT threads within a caches of the subleaf index only without relation to core topology
    // mark up the distinct logical processors sharing a distinct cache level associated subleaf index
    maxThreadsDetected = 0;
    for (i = 0; i < numMappings; i++)
    {
        unsigned j;
        for (j = 0; j < maxThreadsDetected; j++)
        {
            if (pThreadIDsperEachC[j] == glbl_obj.pApicAffOrdMapping[i].EaCacheSMTIDAPIC[subleaf])
            {
                break;
            }
        }

        if (j >= maxThreadsDetected)
        {
            pThreadIDsperEachC[maxThreadsDetected++] = glbl_obj.pApicAffOrdMapping[i].EaCacheSMTIDAPIC[subleaf];
        }
    }

    glbl_obj.EnumeratedEachCacheCount[subleaf] = maxCacheDetected;

    _INTERNAL_DAAL_MEMSET(pEachCIDs, 0xff, numMappings * sizeof(unsigned));
    _INTERNAL_DAAL_MEMSET(pThreadIDsperEachC, 0xff, numMappings * sizeof(unsigned));

    pDetectedEachCIDs = (unsigned *)_INTERNAL_DAAL_MALLOC(numMappings * sizeof(unsigned));
    if (pDetectedEachCIDs == NULL)
    {
        _INTERNAL_DAAL_FREE(pEachCIDs);
        _INTERNAL_DAAL_FREE(pThreadIDsperEachC);
        return -1;
    }
    _INTERNAL_DAAL_MEMSET(pDetectedEachCIDs, 0xff, numMappings * sizeof(unsigned));

    pDetectThreadIDsperEachC = (unsigned *)_INTERNAL_DAAL_MALLOC(numMappings * maxCacheDetected * sizeof(unsigned));
    if (pDetectThreadIDsperEachC == NULL)
    {
        _INTERNAL_DAAL_FREE(pEachCIDs);
        _INTERNAL_DAAL_FREE(pThreadIDsperEachC);
        _INTERNAL_DAAL_FREE(pDetectedEachCIDs);
        return -1;
    }
    _INTERNAL_DAAL_MEMSET(pDetectThreadIDsperEachC, 0xff, numMappings * maxCacheDetected * sizeof(unsigned));

    // enumerate distinct SMT threads and cores relative to a cache level of the subleaf index
    // the enumeration below gets the counts and establishes zero-based numbering scheme for cores and SMT threads under each cache
    maxCacheDetected = 0;

    for (i = 0; i < numMappings; i++)
    {
        glbl_obj.pApicAffOrdMapping[i].EachCacheORD[subleaf] = (unsigned)-1;
    }

    for (i = 0; i < numMappings; i++)
    {
        BOOL CacheMarked;
        unsigned h;

        CacheID  = glbl_obj.pApicAffOrdMapping[i].EaCacheIDAPIC[subleaf]; // sub ID to enumerate different caches in the system
        threadID = glbl_obj.pApicAffOrdMapping[i].EaCacheSMTIDAPIC[subleaf];

        CacheMarked = FALSE;
        for (h = 0; h < maxCacheDetected; h++)
        {
            if (pDetectedEachCIDs[h] == CacheID)
            {
                BOOL foundThread = FALSE;
                unsigned k;

                CacheMarked                                          = TRUE;
                glbl_obj.pApicAffOrdMapping[i].EachCacheORD[subleaf] = h;

                // look for cores sharing the same target cache level
                for (k = 0; k < glbl_obj.perEachCache_detectedThreadCount.data[h * MAX_CACHE_SUBLEAFS + subleaf]; k++)
                {
                    if (threadID == pDetectThreadIDsperEachC[h * numMappings + k])
                    {
                        foundThread                                                 = TRUE;
                        glbl_obj.pApicAffOrdMapping[i].threadPerEaCacheORD[subleaf] = k;
                        break;
                    }
                }

                if (!foundThread)
                {
                    // mark up the thread_ID of an unmarked core in a marked package
                    unsigned thread = glbl_obj.perEachCache_detectedThreadCount.data[h * MAX_CACHE_SUBLEAFS + subleaf];
                    pDetectThreadIDsperEachC[h * numMappings + thread] = threadID;

                    // keep track of respective hierarchical counts
                    glbl_obj.perEachCache_detectedThreadCount.data[h * MAX_CACHE_SUBLEAFS + subleaf]++;

                    // build a set of numbering system to iterate the child hierarchy below the target cache
                    glbl_obj.pApicAffOrdMapping[i].threadPerEaCacheORD[subleaf] = thread;
                }

                break;
            }
        }

        if (!CacheMarked)
        {
            // mark up the pkg_ID and Core_ID of an unmarked package
            pDetectedEachCIDs[maxCacheDetected]                          = CacheID;
            pDetectThreadIDsperEachC[maxCacheDetected * numMappings + 0] = threadID;

            // keep track of respective hierarchical counts
            glbl_obj.perEachCache_detectedThreadCount.data[maxCacheDetected * MAX_CACHE_SUBLEAFS + subleaf] = 1;

            // build a set of numbering system to iterate each topological hierarchy
            glbl_obj.pApicAffOrdMapping[i].EachCacheORD[subleaf]        = maxCacheDetected;
            glbl_obj.pApicAffOrdMapping[i].threadPerEaCacheORD[subleaf] = 0;

            maxCacheDetected++; // this is an unmarked cache, increment cache count by 1
        }
    }

    _INTERNAL_DAAL_FREE(pEachCIDs);
    _INTERNAL_DAAL_FREE(pThreadIDsperEachC);
    _INTERNAL_DAAL_FREE(pDetectedEachCIDs);
    _INTERNAL_DAAL_FREE(pDetectThreadIDsperEachC);

    return 0;
}

/*
 * __internal_daal_buildSystemTopologyTables
 *
 * Construct the processor topology tables and values necessary to
 * support the external functions that display CPU topology and/or
 * cache topology derived from system topology enumeration.
 *
 * Arguments: None
 * Return: None, sets glbl_obj.error if tables or values can not be calculated.
 */
static void __internal_daal_buildSystemTopologyTables()
{
    unsigned lcl_OSProcessorCount, subleaf;
    int numMappings = 0;

    // call OS-specific service to find out how many logical processors
    // are supported by the OS
    glbl_obj.OSProcessorCount = lcl_OSProcessorCount = _internal_daal_GetMaxCPUSupportedByOS();

    // allocated the memory buffers within the global pointer
    __internal_daal_allocArrays(lcl_OSProcessorCount);

    // Gather all the system-wide constant parameters needed to derive topology information
    if (__internal_daal_cpuTopologyParams()) return;
    if (__internal_daal_cacheTopologyParams()) return;

    // For each logical processor, collect APIC ID and parse sub IDs for each APIC ID
    numMappings = __internal_daal_queryParseSubIDs();
    if (numMappings < 0) return;

    // Derived separate numbering schemes for each level of the cpu topology
    if (__internal_daal_analyzeCPUHierarchy(numMappings) < 0)
    {
        glbl_obj.error |= _MSGTYP_TOPOLOGY_NOTANALYZED;
    }

    // an example of building cache topology info for each cache level
    if (glbl_obj.maxCacheSubleaf != -1)
    {
        for (subleaf = 0; subleaf <= glbl_obj.maxCacheSubleaf; subleaf++)
        {
            if (glbl_obj.EachCacheMaskWidth[subleaf] != 0xffffffff)
            {
                // ensure there is at least one core in the target level cache
                if (__internal_daal_analyzeEachCHierarchy(subleaf, numMappings) < 0) glbl_obj.error |= _MSGTYP_TOPOLOGY_NOTANALYZED;
            }
        }
    }

    glbl_obj.isInit = 1;
}

/*
 * __internal_daal_initCpuTopology
 *
 * Initialize the CPU topology structures if they have not already been initialized
 *
 * Arguments: None
 * Return: None
 */
static void __internal_daal_initCpuTopology()
{
    if (!glbl_obj.isInit) __internal_daal_buildSystemTopologyTables();
}

/*
 * _internal_daal_GetEnumerateAPICID
 *
 * Returns APIC ID for the specified processor from enumerated table
 *
 * Arguments:
 *     processor - Os processor number of for which the the APIC ID is returned
 * Return: APIC ID for the specified processor
 */
unsigned _internal_daal_GetEnumerateAPICID(unsigned processor)
{
    if (!glbl_obj.isInit) __internal_daal_initCpuTopology();

    if (glbl_obj.error) return 0xffffffff;

    if (processor >= glbl_obj.OSProcessorCount) return 0xffffffff; // allow caller to intercept error

    return glbl_obj.pApicAffOrdMapping[processor].APICID;
}

/*
 * _internal_daal_GetEnumeratedCoreCount
 *
 * Returns numbers of cores active on a given package, based on enumerated result
 *
 * Arguments:
 *     package - ordinal
 * Return: number of cores active on specified package, 0 if can not calulate
 */
unsigned _internal_daal_GetEnumeratedCoreCount(unsigned package_ordinal)
{
    if (!glbl_obj.isInit) __internal_daal_initCpuTopology();

    if (glbl_obj.error || package_ordinal >= glbl_obj.EnumeratedPkgCount) return 0;

    return glbl_obj.perPkg_detectedCoresCount.data[package_ordinal];
}

/*
 * _internal_daal_GetEnumeratedThreadCount
 *
 * Returns numbers of Threads active on a given package/core
 *
 * Arguments:
 *     package - ordinal and core ordinal
 * Return: number of threads active on specified package and core, 0 if can not calulate
 */
unsigned _internal_daal_GetEnumeratedThreadCount(unsigned package_ordinal, unsigned core_ordinal)
{
    if (!glbl_obj.isInit) __internal_daal_initCpuTopology();

    if (glbl_obj.error || package_ordinal >= glbl_obj.EnumeratedPkgCount) return 0;

    if (core_ordinal >= glbl_obj.perPkg_detectedCoresCount.data[package_ordinal]) return 0;

    return glbl_obj.perCore_detectedThreadsCount.data[package_ordinal * MAX_CORES + core_ordinal];
}

/*
 * _internal_daal_GetSysEachCacheCount
 *
 * Returns count of distinct target level cache in the system
 *
 * Arguments: None
 * Return: Number of caches or 0 if number can not be calculated
 */
unsigned _internal_daal_GetSysEachCacheCount(unsigned subleaf)
{
    if (!glbl_obj.isInit) __internal_daal_initCpuTopology();

    if (glbl_obj.error) return 0;

    return glbl_obj.EnumeratedEachCacheCount[subleaf];
}

/*
 * _internal_daal_GetOSLogicalProcessorCount
 *
 * Returns count of logical processors in the system as seen by OS
 *
 * Arguments: None
 * Return: Number of logical processors or 0 if number can not be calculated
 */
unsigned _internal_daal_GetOSLogicalProcessorCount()
{
    if (!glbl_obj.isInit) __internal_daal_initCpuTopology();

    if (glbl_obj.error) return 0;

    return glbl_obj.OSProcessorCount;
}

/*
 * _internal_daal_GetSysLogicalProcessorCount
 *
 * Returns count of logical processors in the system that were enumerated by this app
 *
 * Arguments: None
 * Return: Number of logical processors or 0 if number can not be calculated
 */
unsigned _internal_daal_GetSysLogicalProcessorCount()
{
    if (!glbl_obj.isInit) __internal_daal_initCpuTopology();

    if (glbl_obj.error) return 0;

    return glbl_obj.EnumeratedThreadCount;
}

/*
 * _internal_daal_GetProcessorCoreCount
 *
 * Returns count of processor cores in the system that were enumerated by this app
 *
 * Arguments: None
 * Return: Number of physical processors or 0 if number can not be calculated
 */
unsigned _internal_daal_GetProcessorCoreCount()
{
    if (!glbl_obj.isInit) __internal_daal_initCpuTopology();

    if (glbl_obj.error) return 0;

    return glbl_obj.EnumeratedCoreCount;
}

/*
 * _internal_daal_GetSysProcessorPackageCount
 *
 * Returns count of processor packages in the system that were enumerated by this app
 *
 * Arguments: None
 * Return: Number of processor packages in the system or 0 if number can not be calculated
 */
unsigned _internal_daal_GetSysProcessorPackageCount()
{
    if (!glbl_obj.isInit) __internal_daal_initCpuTopology();

    if (glbl_obj.error) return 0;

    return glbl_obj.EnumeratedPkgCount;
}

/*
 * _internal_daal_GetCoreCountPerEachCache
 *
 * Returns numbers of cores active sharing the target level cache
 *
 * Arguments: Cache ordinal
 * Return: number of cores active on specified target level cache, 0 if can not calulate
 */
unsigned _internal_daal_GetCoreCountPerEachCache(unsigned subleaf, unsigned cache_ordinal)
{
    if (!glbl_obj.isInit) __internal_daal_initCpuTopology();

    if (glbl_obj.error || cache_ordinal >= glbl_obj.EnumeratedEachCacheCount[subleaf]) return 0;

    return glbl_obj.perEachCache_detectedThreadCount.data[cache_ordinal * MAX_CACHE_SUBLEAFS + subleaf];
}

unsigned _internal_daal_GetLogicalProcessorQueue(int * queue)
{
    const int cpus = _internal_daal_GetSysLogicalProcessorCount();
    int cores      = _internal_daal_GetProcessorCoreCount();

    if (cores == 0) cores = 1;

    int ht = cpus / cores;
    if (ht < 1 || ht >= cpus)
    {
        glbl_obj.error |= _MSGTYP_GENERAL_ERROR;
        return glbl_obj.error;
    }

    int q = 0;
    for (unsigned pkg = 0; pkg < _internal_daal_GetSysProcessorPackageCount(); pkg++)
    {
        for (unsigned i = 0; i < _internal_daal_GetSysEachCacheCount(0); i++)
        {
            if (_internal_daal_GetCoreCountPerEachCache(0, i) > 0)
            {
                for (unsigned j = 0; j < _internal_daal_GetSysLogicalProcessorCount(); j++)
                {
                    if (glbl_obj.pApicAffOrdMapping[j].packageORD == pkg && glbl_obj.pApicAffOrdMapping[j].EachCacheORD[0] == i)
                    {
                        int jj = ((q / ht) + (cores * (q % ht))) % cpus;
                        if (jj < cpus) queue[jj] = j;
                        q++;
                    }
                }
            }
        }
    }

    return glbl_obj.error;
}

unsigned _internal_daal_GetStatus()
{
    return glbl_obj.error;
}

//service_environment.h implementation

    #define __extract_bitmask_value(val, mask, shift) (((val) & (mask)) >> shift)

    #define _CPUID_CACHE_INFO_GET_TYPE(__eax) __extract_bitmask_value(__eax /*4:0*/, 0x1fU, 0)

    #define _CPUID_CACHE_INFO_GET_LEVEL(__eax) __extract_bitmask_value(__eax /*7:5*/, 0xe0U, 5)

    #define _CPUID_CACHE_INFO_GET_SETS(__ecx) ((__ecx) + 1)

    #define _CPUID_CACHE_INFO_GET_LINE_SIZE(__ebx) (__extract_bitmask_value(__ebx /*11:0*/, 0x7ffU, 0) + 1)

    #define _CPUID_CACHE_INFO_GET_PARTITIONS(__ebx) (__extract_bitmask_value(__ebx /*21:11*/, 0x3ff800U, 11) + 1)

    #define _CPUID_CACHE_INFO_GET_WAYS(__ebx) (__extract_bitmask_value(__ebx /*31:22*/, 0xffc00000U, 22) + 1)

    #define _CPUID_CACHE_INFO 0x4U

static __inline void get_cache_info(int cache_num, int * type, int * level, long long * sets, int * line_size, int * partitions, int * ways)
{
    static uint32_t abcd[4];
    run_cpuid(_CPUID_CACHE_INFO, cache_num, abcd);
    const uint32_t eax = abcd[0];
    const uint32_t ebx = abcd[1];
    const uint32_t ecx = abcd[2];
    const uint32_t edx = abcd[3];
    *type              = _CPUID_CACHE_INFO_GET_TYPE(eax);
    *level             = _CPUID_CACHE_INFO_GET_LEVEL(eax);
    *sets              = _CPUID_CACHE_INFO_GET_SETS(ecx);
    *line_size         = _CPUID_CACHE_INFO_GET_LINE_SIZE(ebx);
    *partitions        = _CPUID_CACHE_INFO_GET_PARTITIONS(ebx);
    *ways              = _CPUID_CACHE_INFO_GET_WAYS(ebx);
}

    #define _CPUID_CACHE_INFO_TYPE_NULL 0
    #define _CPUID_CACHE_INFO_TYPE_DATA 1
    #define _CPUID_CACHE_INFO_TYPE_INST 2
    #define _CPUID_CACHE_INFO_TYPE_UNIF 3

static __inline void detect_data_caches(int cache_sizes_len, volatile long long * cache_sizes)
{
    int cache_num = 0, cache_sizes_idx = 0;
    while (cache_sizes_idx < cache_sizes_len)
    {
        int type, level, line_size, partitions, ways;
        long long sets, size;
        get_cache_info(cache_num++, &type, &level, &sets, &line_size, &partitions, &ways);

        if (type == _CPUID_CACHE_INFO_TYPE_NULL) break;
        if (type == _CPUID_CACHE_INFO_TYPE_INST) continue;

        size                           = ways * partitions * line_size * sets;
        cache_sizes[cache_sizes_idx++] = size;
    }
}

    #define MAX_CACHE_LEVELS 4
volatile static bool cache_sizes_read                   = false;
volatile static long long cache_sizes[MAX_CACHE_LEVELS] = { 0 };

static __inline void update_cache_sizes()
{
    int cbwr_branch;

    if (cache_sizes_read) return;

    if (!cache_sizes_read) detect_data_caches(MAX_CACHE_LEVELS, cache_sizes);
    cache_sizes_read = true;
}

long long getCacheSize(int cache_num)
{
    if (cache_num < 1 || cache_num > MAX_CACHE_LEVELS)
        return -1;
    else
    {
        update_cache_sizes();
        return cache_sizes[cache_num - 1];
    }
}

size_t getL1CacheSize()
{
    return getCacheSize(1);
}

size_t getL2CacheSize()
{
    return getCacheSize(2);
}

size_t getLLCacheSize()
{
    return getCacheSize(3);
}

void glktsn::FreeArrays()
{
    isInit = 0;
    _INTERNAL_DAAL_FREE(pApicAffOrdMapping);
    _INTERNAL_DAAL_FREE(perPkg_detectedCoresCount.data);
    _INTERNAL_DAAL_FREE(perCore_detectedThreadsCount.data);
    _INTERNAL_DAAL_FREE(perCache_detectedCoreCount.data);
    _INTERNAL_DAAL_FREE(perEachCache_detectedThreadCount.data);

    if (cpuid_values)
    {
        for (unsigned int i = 0; i <= OSProcessorCount; i++)
        {
            _INTERNAL_DAAL_FREE(cpuid_values[i].subleaf[0]);

            if ((i == 0x4 || i == 0xb))
            {
                for (unsigned int j = 1; j < cpuid_values[i].subleaf_max; j++)
                {
                    _INTERNAL_DAAL_FREE(cpuid_values[i].subleaf[j]);
                }
            }
        }
        _INTERNAL_DAAL_FREE(cpuid_values);
    }
}

} // namespace internal
} // namespace services
} // namespace daal

void read_topology(int & status, int & nthreads, int & max_threads, int ** cpu_queue)
{
    status      = 0;
    max_threads = 0;
    *cpu_queue  = NULL;

    /* Maximum cpu's amount */
    max_threads = daal::services::internal::_internal_daal_GetSysLogicalProcessorCount();
    if (!max_threads)
    {
        status--;
        return;
    }

    /* Allocate memory for CPU queue */
    *cpu_queue = (int *)daal::services::daal_malloc(max_threads * sizeof(int), 64);
    if (!(*cpu_queue))
    {
        status--;
        return;
    }

    /* Create cpu queue */
    daal::services::internal::_internal_daal_GetLogicalProcessorQueue(*cpu_queue);

    /* Check if errors happened during topology reading */
    if (daal::services::internal::_internal_daal_GetStatus() != 0)
    {
        status--;
        return;
    }

    return;
}

void delete_topology(void * ptr)
{
    daal::services::daal_free(ptr);
}

#else

namespace daal
{
namespace services
{
namespace internal
{
size_t getL1CacheSize()
{
    return 32 * 1024;
}

size_t getL2CacheSize()
{
    return 256 * 1024;
}

size_t getLLCacheSize()
{
    return 25 * 1024 * 1024; //estimate based on mac pro
}

} // namespace internal
} // namespace services
} // namespace daal

#endif /* #if !(defined DAAL_CPU_TOPO_DISABLED) */

namespace daal
{
namespace services
{
namespace internal
{
size_t getNumElementsFitInMemory(size_t sizeofMemory, size_t sizeofAnElement, size_t defaultNumElements)
{
    const size_t n = sizeofMemory / sizeofAnElement;
    // TODO substitute this if-statement with DAAL_ASSERT
    if (n < 1) return sizeofMemory ? 1 : defaultNumElements;

    return n;
}

size_t getNumElementsFitInL1Cache(size_t sizeofAnElement, size_t defaultNumElements)
{
    return getNumElementsFitInMemory(getL1CacheSize(), sizeofAnElement, defaultNumElements);
}

size_t getNumElementsFitInLLCache(size_t sizeofAnElement, size_t defaultNumElements)
{
    return getNumElementsFitInMemory(getLLCacheSize(), sizeofAnElement, defaultNumElements);
}

} // namespace internal
} // namespace services
} // namespace daal
