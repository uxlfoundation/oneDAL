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

    #if defined(__linux__) || defined(__FreeBSD__)

        #ifndef _GNU_SOURCE
            #define _GNU_SOURCE
        #endif

        #include <stdio.h>
        #include <stdlib.h>
        #include <unistd.h>
        #include <string.h>
        #include <sched.h>

        #ifdef __FreeBSD__
            #include <sys/param.h>
            #include <sys/cpuset.h>
typedef cpuset_t cpu_set_t;
        #endif

        #ifdef __linux__
            #include <alloca.h>
        #endif

        #include <stdarg.h>

        #ifdef __CPU_ISSET
            #define MY_CPU_SET   __CPU_SET
            #define MY_CPU_ZERO  __CPU_ZERO
            #define MY_CPU_ISSET __CPU_ISSET
        #else
            #define MY_CPU_SET   CPU_SET
            #define MY_CPU_ZERO  CPU_ZERO
            #define MY_CPU_ISSET CPU_ISSET
        #endif

        #define __cdecl

        #if defined(TARGET_X86_64)
            #define LNX_PTR2INT unsigned long long
            #define LNX_MY1CON  1LL
        #elif defined(TARGET_ARM)
using LNX_PTR2INT                = uintptr_t;
constexpr LNX_PTR2INT LNX_MY1CON = 1LL;
        #elif defined(TARGET_RISCV64)
            #define LNX_PTR2INT uintptr_t
            #define LNX_MY1CON  1LL
        #else
            #define LNX_PTR2INT unsigned int
            #define LNX_MY1CON  1
        #endif

        #ifndef __int64
            #define __int64 long long
        #endif

        #ifndef __int32
            #define __int32 int
        #endif

        #ifndef DWORD
            #define DWORD unsigned long
        #endif

        #ifndef DWORD_PTR
            #define DWORD_PTR unsigned long *
        #endif

        #ifndef FALSE
            #define FALSE 0
        #endif

        #ifndef TRUE
            #define TRUE 1
        #endif

        #ifndef BYTE
            #define BYTE unsigned char
        #endif

        #ifndef BOOL
            #define BOOL char
        #endif

        #ifdef __x86_64__
            #define AFFINITY_MASK unsigned __int64
        #else
            #define AFFINITY_MASK unsigned __int32
        #endif

    #else /* WINDOWS */
        #define NOMINMAX
        #include <windows.h>

        #ifdef _M_IA64
            #define LNX_PTR2INT unsigned long long
            #define LNX_MY1CON  1LL
        #else
            #ifdef _M_X64
                #define LNX_PTR2INT __int64
                #define LNX_MY1CON  1LL
            #else
                #define LNX_PTR2INT unsigned int
                #define LNX_MY1CON  1
            #endif
        #endif

        #if defined(_MSC_VER)
            #if (_MSC_FULL_VER >= 160040219)
                #include <intrin.h>
            #else
                #error "min VS2010 SP1 compiler is required"
            #endif
        #endif

        #if _MSC_VER < 1300
            #ifndef DWORD_PTR
                #define DWORD_PTR unsigned long *
            #endif
        #endif

        #ifdef _M_IA64
typedef unsigned __int64 AFFINITY_MASK;
        #else
typedef unsigned __int32 AFFINITY_MASK;
        #endif

    #endif /* WINDOWS */

    #define MAX_LOG_CPU            (8 * sizeof(DWORD_PTR) * 8)
    #define MAX_WIN7_LOG_CPU       (4 * sizeof(DWORD_PTR) * 8)
    #define MAX_PREWIN7_LOG_CPU    (sizeof(DWORD_PTR) * 8)
    #define MAX_PACKAGES           MAX_LOG_CPU
    #define MAX_CORES              MAX_LOG_CPU
    #define BLOCKSIZE_4K           4096
    #define MAX_THREAD_GROUPS_WIN7 4

    #define MAX_CPUS_ARRAY     64
    #define MAX_LEAFS          80
    #define MAX_CACHE_SUBLEAFS 16 // max allocation limit of data structure per sub leaf of cpuid leaf 4 enumerated results
    #define MAX_LEAFS_EXT      MAX_LEAFS

    #define ENUM_ALL (0xffffffff)

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
typedef struct dcache_str
{
    int L1;       // L1Data size
    int L1i;      // L1 instruction size
    int L1i_type; // L1 instruction type (0 == size in bytes, 1= size in trace cache uops)
    int L2;       // L2 size
    int L3;       // L3 size
    int cl;       // cache line size
    int sectored; // cache lines sectored
} leaf2_cache_struct;

typedef struct
{
    unsigned __int32 EAX, EBX, ECX, EDX;
} CPUIDinfo;

struct CPUIDinfox
{
    CPUIDinfo * subleaf[MAX_CACHE_SUBLEAFS];
    unsigned __int32 subleaf_max = 0;
};

typedef struct
{
    unsigned maxByteLength;
    unsigned char * AffinityMask;
} GenericAffinityMask;
// The width of affinity mask in legacy Windows API is 32 or 64, depending on
// 32-bit or 64-bit OS.
// Linux abstract its equivalent bitmap cpumask_t from direct programmer access,
// cpumask_t can support more than 64 cpu, but is configured at kernel
// compile time by configuration parameter.
// To abstract the size difference of the bitmap used by different OSes
// for topology analysis, we use an unsigned char buffer that can
// map to different OS implementation's affinity mask data structure

typedef struct
{
    char description[256];
    char descShort[64];
    unsigned level; // start at 1
    unsigned type;  // cache type (instruction, data, combined)
    unsigned sizeKB;
    unsigned how_many_threads_share_cache;
    unsigned how_many_caches_share_level;
} cacheDetail_str;

struct Dyn2Arr_str
{
    unsigned dim[2];        // xdim and ydim
    unsigned * data = NULL; // data array to be malloc'd
};

struct Dyn1Arr_str
{
    unsigned dim[1];        // xdim
    unsigned * data = NULL; // data array to be malloc'd
};

typedef struct
{
    int size;
    int used;
    char * buffer;
} DynCharBuf_str;

typedef struct
{
    unsigned __int32 APICID;        // the full x2APIC ID or initial APIC ID of a logical
                                    //  processor assigned by HW
    unsigned __int32 OrdIndexOAMsk; // An ordinal index (zero-based) for each logical
                                    //  processor in the system, 1:1 with "APICID"
    // Next three members are the sub IDs for processor topology enumeration
    unsigned __int32 pkg_IDAPIC;  // Pkg_ID field, subset of APICID bits
                                  //  to distinguish different packages
    unsigned __int32 Core_IDAPIC; // Core_ID field, subset of APICID bits to
                                  //  distinguish different cores in a package
    unsigned __int32 SMT_IDAPIC;  // SMT_ID field, subset of APICID bits to
                                  //  distinguish different logical processors in a core
    // the next three members stores a numbering scheme of ordinal index
    // for each level of the processor topology.
    unsigned __int32 packageORD; // a zero-based numbering scheme for each physical
                                 //  package in the system
    unsigned __int32 coreORD;    // a zero-based numbering scheme for each core in the same package
    unsigned __int32 threadORD;  // a zero-based numbering scheme for each thread in the same core
    // Next two members are the sub IDs for cache topology enumeration
    unsigned __int32 EaCacheSMTIDAPIC[MAX_CACHE_SUBLEAFS]; // SMT_ID field, subset of APICID bits
                                                           // to distinguish different logical processors sharing the same cache level
    unsigned __int32 EaCacheIDAPIC[MAX_CACHE_SUBLEAFS];    // sub ID to enumerate different cache entities
                                                           // of the cache level corresponding to the array index/cpuid leaf 4 subleaf index
    // the next three members stores a numbering scheme of ordinal index
    // for enumerating different cache entities of a cache level, and enumerating
    // logical processors sharing the same cache entity.
    unsigned __int32 EachCacheORD[MAX_CACHE_SUBLEAFS];        // a zero-based numbering scheme
                                                              // for each cache entity of the specified cache level in the system
    unsigned __int32 threadPerEaCacheORD[MAX_CACHE_SUBLEAFS]; // a zero-based numbering scheme
                                                              // for each logical processor sharing the same cache of the specified cache level
} idAffMskOrdMapping_t;

// we are going to put an assortment of global variable, 1D and 2D arrays into
// a data structure. This is the declaration
struct glktsn
{ // for each logical processor we need spaces to store APIC ID,
    // sub IDs, affinity mappings, etc.
    idAffMskOrdMapping_t * pApicAffOrdMapping = NULL;

    // workspace for storing hierarchical counts of each level
    Dyn1Arr_str perPkg_detectedCoresCount;
    Dyn2Arr_str perCore_detectedThreadsCount;
    // workspace for storing hierarchical counts relative to the cache topology
    // of the largest unified cache (may be shared by several cores)
    Dyn1Arr_str perCache_detectedCoreCount;
    Dyn2Arr_str perEachCache_detectedThreadCount;
    // we use an error code to indicate any abnoral situation
    unsigned error;
    // If CPUID full reporting capability has been restricted, we need to be aware of it.
    unsigned Alert_BiosCPUIDmaxLimitSetting;

    unsigned OSProcessorCount = 0; // how many logical processor the OS sees
    unsigned hasLeafB;             // flag to keep track of whether CPUID leaf 0BH is supported
    unsigned maxCacheSubleaf;      // highest CPUID leaf 4 subleaf index in a processor

    // the following global variables are the total counts in the system resulting from software enumeration
    unsigned EnumeratedPkgCount;
    unsigned EnumeratedCoreCount;
    unsigned EnumeratedThreadCount;
    // CPUID ID leaf 4 can report data for several cache levels, we'll keep track of each cache level
    unsigned EnumeratedEachCacheCount[MAX_CACHE_SUBLEAFS];
    // the following global variables are parameters related to
    //  extracting sub IDs from an APIC ID, common to all processors in the system
    unsigned SMTSelectMask;
    unsigned PkgSelectMask;
    unsigned CoreSelectMask;
    unsigned PkgSelectMaskShift;
    unsigned SMTMaskWidth;
    // We'll do sub ID extractions using parameters from each cache level
    unsigned EachCacheSelectMask[MAX_CACHE_SUBLEAFS];
    unsigned EachCacheMaskWidth[MAX_CACHE_SUBLEAFS];

    // the following global variables are used for product capability identification
    unsigned HWMT_SMTperCore;
    unsigned HWMT_SMTperPkg;
    // a data structure that can store simple leaves and complex subleaves of all supported leaf indices of CPUID
    CPUIDinfox * cpuid_values = nullptr;
    // workspace of our generic affinitymask structure to allow iteration over each logical processors in the system
    GenericAffinityMask cpu_generic_processAffinity;
    GenericAffinityMask cpu_generic_systemAffinity;
    // workspeace to assist text display of cache topology information
    cacheDetail_str cacheDetail[MAX_CACHE_SUBLEAFS];

    unsigned isInit = 0;

    void FreeArrays();

    ~glktsn() { FreeArrays(); }
};

[[maybe_unused]] static unsigned long __internal_daal_getBitsFromDWORD(const unsigned int val, const char from, const char to);
[[maybe_unused]] static unsigned __internal_daal_createMask(unsigned numEntries, unsigned * maskLength);
[[maybe_unused]] static unsigned __internal_daal_slectOrdfromPkg(unsigned package, unsigned core, unsigned logical);
[[maybe_unused]] static unsigned __internal_daal_getAPICID(unsigned processor);
[[maybe_unused]] static void __internal_daal_initCpuTopology();
[[maybe_unused]] static int __internal_daal_bindContext(unsigned cpu, void * prevAffinity);
[[maybe_unused]] static void __internal_daal_restoreContext(void * prevAffinity);
[[maybe_unused]] static void __internal_daal_setChkProcessAffinityConsistency(unsigned lcl_OSProcessorCount);
[[maybe_unused]] static void __internal_daal_setGenericAffinityBit(GenericAffinityMask * pAffinityMap, unsigned cpu);
[[maybe_unused]] static void __internal_daal_getCpuidInfo(CPUIDinfo * info, const unsigned int func, const unsigned int subfunc);
[[maybe_unused]] static int __internal_daal_countBits(DWORD_PTR x);

unsigned _internal_daal_GetMaxCPUSupportedByOS();
unsigned _internal_daal_GetOSLogicalProcessorCount();
unsigned _internal_daal_GetSysProcessorPackageCount();
unsigned _internal_daal_GetProcessorCoreCount();
unsigned _internal_daal_GetLogicalProcessorCount();
unsigned _internal_daal_GetCoresPerPackageProcessorCount();
unsigned _internal_daal_GetProcessorPackageCount();
unsigned _internal_daal_GetEnumerateAPICID(unsigned processor);
unsigned _internal_daal_GetLogicalPerCoreProcessorCount();
unsigned _internal_daal_GetCoreCount(unsigned long package_ordinal);
unsigned _internal_daal_GetThreadCount(unsigned long package_ordinal, unsigned long core_ordinal);
unsigned _internal_daal_GetLogicalProcessorQueue(int * queue);
unsigned _internal_daal_GetStatus();

unsigned _internal_daal_GetSysLogicalProcessorCount();
} // namespace internal
} // namespace services
} // namespace daal

void read_topology(int & status, int & nthreads, int & max_threads, int ** cpu_queue);
void delete_topology(void * ptr);

#endif /* #if !defined (DAAL_CPU_TOPO_DISABLED) */
#endif /* __SERVICE_TOPO_H__ */
