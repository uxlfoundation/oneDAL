/* file: service_topo.cpp */
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

/*
//++
//  Implementation of CPU topology reading routines
//--
*/
#include "services/daal_defines.h"

#define MAX_CACHE_LEVELS      4
#define DEFAULT_L1_CACHE_SIZE 32 * 1024
#define DEFAULT_L2_CACHE_SIZE 256 * 1024
#define DEFAULT_LL_CACHE_SIZE 4 * 1024 * 1024

#if !defined(DAAL_CPU_TOPO_DISABLED)

    #include <array>
    #include <limits>

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
            #define LNX_PTR2INT unsigned long long
            #define LNX_MY1CON  1LL
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

        #ifndef BYTE
            #define BYTE unsigned char
        #endif

        #define AFFINITY_MASK unsigned __int64

    #else /* WINDOWS */
        #define NOMINMAX
        #include <windows.h>
        #include <tlhelp32.h> // CreateToolhelp32Snapshot, THREADENTRY32, Thread32First, Thread32Next

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

    #define MAX_LOG_CPU                      (8 * sizeof(DWORD_PTR) * 8)
    #define MAX_WIN7_LOG_CPU                 (4 * sizeof(DWORD_PTR) * 8)
    #define MAX_PACKAGES                     MAX_LOG_CPU
    #define MAX_CORES                        MAX_LOG_CPU
    #define BLOCKSIZE_4K                     4096
    #define MAX_THREAD_GROUPS_WIN7           4
    #define MAX_LOGICAL_PROCESSORS_PER_GROUP 64

    #define MAX_CACHE_SUBLEAFS 16 // max allocation limit of data structure per sub leaf of cpuid leaf 4 enumerated results

    #define ENUM_ALL (0xffffffff)

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

    #define _INTERNAL_DAAL_MALLOC(x)              daal_malloc((x), DAAL_MALLOC_DEFAULT_ALIGNMENT)
    #define _INTERNAL_DAAL_FREE(x)                daal_free((x))
    #define _INTERNAL_DAAL_MEMSET(a1, a2, a3)     __internal_daal_memset((a1), (a2), (a3))
    #define _INTERNAL_DAAL_MEMCPY(a1, a2, a3, a4) daal::services::internal::daal_memcpy_s((a1), (a2), (a3), (a4))

    #define _INTERNAL_DAAL_OVERFLOW_CHECK_BY_ADDING(type, op1, op2)   \
        {                                                             \
            volatile type r = (op1) + (op2);                          \
            r -= (op1);                                               \
            if (!(r == (op2))) error |= _MSGTYP_TOPOLOGY_NOTANALYZED; \
        }
namespace daal
{
namespace services
{
namespace internal
{
struct leaf2_cache_struct
{
    int L1;       // L1Data size
    int L1i;      // L1 instruction size
    int L1i_type; // L1 instruction type (0 == size in bytes, 1= size in trace cache uops)
    int L2;       // L2 size
    int L3;       // L3 size
    int cl;       // cache line size
    int sectored; // cache lines sectored
};

struct CPUIDinfo
{
    unsigned __int32 EAX = 0, EBX = 0, ECX = 0, EDX = 0;

    /*
     * Stores the raw data reported in 4 registers by CPUID into the structure,
     * supports sub-leaf reporting.
     * Note: The CPUID instrinsic in MSVC does not support sub-leaf reporting.
     *
     * \param func       CPUID leaf number
     * \param subfunc    CPUID subleaf number
     */
    void get(unsigned int func, unsigned int subfunc = 0) { get_info(func, subfunc); }

private:
    void get_info(unsigned int eax, unsigned int ecx)
    {
        uint32_t abcd[4];
        run_cpuid(eax, ecx, abcd);
        EAX = abcd[0];
        EBX = abcd[1];
        ECX = abcd[2];
        EDX = abcd[3];
    }
};

struct CPUIDinfox
{
    std::array<CPUIDinfo *, MAX_CACHE_SUBLEAFS> subleaf;
    unsigned __int32 subleaf_max = 0;
};

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
 * __internal_daal_myBitScanReverse
 *
 * Returns of bits [to:from] of DWORD
 * This c-emulation of the BSR instruction is shown here for tool portability
 *
 * \param index      Bit offset of the most significant bit that's not 0 found in mask
 * \param mask       Input data to search the most significant bit
 *
 * \return        1 if a non-zero bit is found, otherwise 0
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

struct GenericAffinityMask
{
    GenericAffinityMask() = default;
    GenericAffinityMask(const unsigned numCpus);
    GenericAffinityMask(const GenericAffinityMask & other);
    GenericAffinityMask & operator=(GenericAffinityMask other)
    {
        swap(*this, other);
        return *this;
    }
    ~GenericAffinityMask();

    friend void swap(GenericAffinityMask & first, GenericAffinityMask & second) // nothrow
    {
        unsigned tmp         = first.maxByteLength;
        first.maxByteLength  = second.maxByteLength;
        second.maxByteLength = tmp;

        unsigned char * tmpMask = first.AffinityMask;
        first.AffinityMask      = second.AffinityMask;
        second.AffinityMask     = tmpMask;
    }

    static constexpr unsigned char N_BITS_IN_BYTE      = 8;
    static constexpr unsigned char LOG2_N_BITS_IN_BYTE = 3; // log2(8) = 3

    static constexpr unsigned char OUT_OF_BOUND_ERROR = 0xff;

    unsigned char test(unsigned cpu) const
    {
        if (cpu < (maxByteLength << LOG2_N_BITS_IN_BYTE))
        {
            if ((AffinityMask[cpu >> LOG2_N_BITS_IN_BYTE] & (1 << (cpu % N_BITS_IN_BYTE))))
                return 1;
            else
                return 0;
        }
        else
        {
            // If cpu is out of range, return 0xff to indicate an error
            return OUT_OF_BOUND_ERROR;
        }
    }

    unsigned char set(unsigned cpu)
    {
        if (cpu < (maxByteLength << LOG2_N_BITS_IN_BYTE))
            AffinityMask[cpu >> LOG2_N_BITS_IN_BYTE] |= 1 << (cpu % N_BITS_IN_BYTE);
        else
        {
            // If cpu is out of range, return 0xff to indicate an error
            return OUT_OF_BOUND_ERROR;
        }
        return 0;
    }

    unsigned maxByteLength       = 0;
    unsigned char * AffinityMask = nullptr;
};
// The width of affinity mask in legacy Windows API is 32 or 64, depending on
// 32-bit or 64-bit OS.
// Linux abstract its equivalent bitmap cpumask_t from direct programmer access,
// cpumask_t can support more than 64 cpu, but is configured at kernel
// compile time by configuration parameter.
// To abstract the size difference of the bitmap used by different OSes
// for topology analysis, we use an unsigned char buffer that can
// map to different OS implementation's affinity mask data structure

struct cacheDetail_str
{
    std::array<char, 256> description;
    std::array<char, 256> descShort;
    unsigned level = 1; // start at 1
    unsigned type;      // cache type (instruction, data, combined)
    unsigned sizeKB;
    unsigned how_many_threads_share_cache;
    unsigned how_many_caches_share_level;
};

struct Dyn2Arr_str
{
    Dyn2Arr_str() = default;
    explicit Dyn2Arr_str(const unsigned xdim, const unsigned ydim, const unsigned value = 0);

    ~Dyn2Arr_str();

    Dyn2Arr_str(const Dyn2Arr_str & other);
    Dyn2Arr_str & operator=(Dyn2Arr_str other)
    {
        swap(*this, other);
        return *this;
    }

    unsigned & operator[](unsigned idx)
    {
        DAAL_ASSERT(idx < size());
        return data[idx];
    }

    const unsigned & operator[](unsigned idx) const
    {
        DAAL_ASSERT(idx < size());
        return data[idx];
    }

    bool isEmpty() const { return (data == nullptr); }

    unsigned size() const { return dim[0] * dim[1]; }

    friend void swap(Dyn2Arr_str & first, Dyn2Arr_str & second) // nothrow
    {
        unsigned tmp  = first.dim[0];
        first.dim[0]  = second.dim[0];
        second.dim[0] = tmp;

        tmp           = first.dim[1];
        first.dim[1]  = second.dim[1];
        second.dim[1] = tmp;

        unsigned * tmpData = first.data;
        first.data         = second.data;
        second.data        = tmpData;
    }

private:
    unsigned dim[2] = { 0, 0 }; // xdim and ydim
    unsigned * data = nullptr;  // data array to be malloc'd
};

struct Dyn1Arr_str
{
    Dyn1Arr_str() = default;
    explicit Dyn1Arr_str(const unsigned xdim, const unsigned value = 0);
    ~Dyn1Arr_str();

    Dyn1Arr_str(const Dyn1Arr_str & other);
    Dyn1Arr_str & operator=(Dyn1Arr_str other)
    {
        swap(*this, other);
        return *this;
    }

    unsigned & operator[](unsigned idx)
    {
        DAAL_ASSERT(idx < dim[0]);
        return data[idx];
    }

    const unsigned & operator[](unsigned idx) const
    {
        DAAL_ASSERT(idx < dim[0]);
        return data[idx];
    }

    bool isEmpty() const { return (data == nullptr); }

    void fill(const unsigned value);

    friend void swap(Dyn1Arr_str & first, Dyn1Arr_str & second) // nothrow
    {
        unsigned tmp  = first.dim[0];
        first.dim[0]  = second.dim[0];
        second.dim[0] = tmp;

        unsigned * tmpData = first.data;
        first.data         = second.data;
        second.data        = tmpData;
    }

private:
    unsigned dim[1] = { 0 };   // xdim
    unsigned * data = nullptr; // data array to be malloc'd
};

struct idAffMskOrdMapping_t
{
    idAffMskOrdMapping_t() = default;
    explicit idAffMskOrdMapping_t(unsigned int cpu, bool hasLeafB, unsigned globalPkgSelectMask, unsigned globalPkgSelectMaskShift,
                                  unsigned globalCoreSelectMask, unsigned globalSMTSelectMask, unsigned globalSMTMaskWidth,
                                  std::array<unsigned, MAX_CACHE_SUBLEAFS> & globalEachCacheSelectMask, unsigned globalmaxCacheSubleaf);
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
    std::array<unsigned __int32, MAX_CACHE_SUBLEAFS> EaCacheSMTIDAPIC; // SMT_ID field, subset of APICID bits
                                                                       // to distinguish different logical processors sharing the same cache level
    std::array<unsigned __int32, MAX_CACHE_SUBLEAFS> EaCacheIDAPIC;    // sub ID to enumerate different cache entities
        // of the cache level corresponding to the array index/cpuid leaf 4 subleaf index
    // the next three members stores a numbering scheme of ordinal index
    // for enumerating different cache entities of a cache level, and enumerating
    // logical processors sharing the same cache entity.
    std::array<unsigned __int32, MAX_CACHE_SUBLEAFS> EachCacheORD;        // a zero-based numbering scheme
                                                                          // for each cache entity of the specified cache level in the system
    std::array<unsigned __int32, MAX_CACHE_SUBLEAFS> threadPerEaCacheORD; // a zero-based numbering scheme
        // for each logical processor sharing the same cache of the specified cache level
private:
    void initApicID(bool hasLeafB);
};

// we are going to put an assortment of global variable, 1D and 2D arrays into
// a data structure. This is the declaration
struct glktsn
{ // for each logical processor we need spaces to store APIC ID,
    // sub IDs, affinity mappings, etc.
    idAffMskOrdMapping_t * pApicAffOrdMapping = nullptr;

    // workspace for storing hierarchical counts of each level
    Dyn1Arr_str perPkg_detectedCoresCount;
    Dyn2Arr_str perCore_detectedThreadsCount;
    // workspace for storing hierarchical counts relative to the cache topology
    // of the largest unified cache (may be shared by several cores)
    Dyn1Arr_str perCache_detectedCoreCount;
    Dyn2Arr_str perEachCache_detectedThreadCount;
    // we use an error code to indicate any abnoral situation
    unsigned error;

    unsigned OSProcessorCount = 0; // how many logical processor the OS sees
    bool hasLeafB;                 // flag to keep track of whether CPUID leaf 0BH is supported
    unsigned maxCacheSubleaf;      // highest CPUID leaf 4 subleaf index in a processor

    // the following global variables are the total counts in the system resulting from software enumeration
    unsigned EnumeratedPkgCount;
    unsigned EnumeratedCoreCount;
    unsigned EnumeratedThreadCount;

    // CPUID ID leaf 4 can report data for several cache levels, we'll keep track of each cache level
    std::array<unsigned, MAX_CACHE_SUBLEAFS> EnumeratedEachCacheCount;
    // the following global variables are parameters related to
    //  extracting sub IDs from an APIC ID, common to all processors in the system
    unsigned SMTSelectMask;
    unsigned PkgSelectMask;
    unsigned CoreSelectMask;
    unsigned PkgSelectMaskShift;
    unsigned SMTMaskWidth;
    // We'll do sub ID extractions using parameters from each cache level
    std::array<unsigned, MAX_CACHE_SUBLEAFS> EachCacheSelectMask;
    std::array<unsigned, MAX_CACHE_SUBLEAFS> EachCacheMaskWidth;

    // the following global variables are used for product capability identification
    unsigned HWMT_SMTperCore;
    unsigned HWMT_SMTperPkg;
    // a data structure that can store simple leaves and complex subleaves of all supported leaf indices of CPUID
    unsigned maxCPUIDLeaf;               // highest CPUID leaf index in a processor
    CPUIDinfox * cpuid_values = nullptr; // CPUID info storage of size [maxCPUIDLeaf + 1]
    // workspace of our generic affinitymask structure to allow iteration over each logical processors in the system
    GenericAffinityMask cpu_generic_processAffinity;
    GenericAffinityMask cpu_generic_systemAffinity;
    // workspeace to assist text display of cache topology information
    std::array<cacheDetail_str, MAX_CACHE_SUBLEAFS> cacheDetail;

    bool isInit = false;

    glktsn();

    ~glktsn() { freeArrays(); }

private:
    int allocArrays(const unsigned cpus);
    void freeArrays();
    unsigned getMaxCPUSupportedByOS();
    int cpuTopologyParams();
    int cpuTopologyLeafBConstants();
    int cpuTopologyLegacyConstants(CPUIDinfo * pinfo);
    int cacheTopologyParams();
    void initStructuredLeafBuffers();
    int findEachCacheIndex(unsigned cache_subleaf);
    int eachCacheTopologyParams(unsigned targ_subleaf);

    void setChkProcessAffinityConsistency();
    int initEnumeratedThreadCountAndParseAPICIDs();
    int analyzeCPUHierarchy();
    int analyzeEachCHierarchy(unsigned subleaf);
    void buildSystemTopologyTables();
};

// Global CPU topology object
static glktsn globalCPUTopology;

static char scratch[BLOCKSIZE_4K]; // scratch space large enough for OS to write SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX

static void * __internal_daal_memset(void * s, int c, size_t nbytes)
{
    const unsigned char value = static_cast<unsigned char>(c);
    unsigned char * p         = static_cast<unsigned char *>(s);
    for (size_t i = 0; i < nbytes; ++i)
    {
        p[i] = value;
    }
    return s;
}

/*
 * Binds the execution context of the process to a specific logical processor on construction.
 * Restores the previous affinity mask on destruction.
 */
struct ScopedThreadContext
{
    // Constructor that binds the execution context to the specified logical processor
    //
    // \param cpu The ordinal index to reference a logical processor in the system
    explicit ScopedThreadContext(unsigned int cpu) { error = bindContext(cpu); }

    ~ScopedThreadContext()
    {
        // Restore the previous affinity mask
        restoreContext();
    }
    int error;

private:
    /*
    * A wrapper function that can compile under two OS environments
    * Linux and Windows, to bind the current executing process context.
    *
    * The size of the bitmap that underlies cpu_set_t is configurable
    * at Linux Kernel Compile time. Each distro may set limit on its own.
    * Some newer Linux distro may support 256 logical processors,
    * For simplicity we don't show the check for range on the ordinal index
    * of the target cpu in Linux, interested reader can check Linux kernel documentation.
    *
    * Starting with Windows OS version 0601H (e.g. Windows 7), it supports up to 4 sets of
    * affinity masks, referred to as "GROUP_AFFINITY".
    * Constext switch within the same group is done by the same API as was done in previous
    * generations of windows (such as Vista). In order to bind the current executing
    * process context to a logical processor in a different group, it must be be binded
    * using a new API to the target processor group.
    * Limitation, New Windows APIs that support GROUP_AFFINITY requires
    * Windows platform SDK 7.0a.
    *
    * \param cpu    The ordinal index to reference a logical processor in the system
    *
    * \return       0 is no error
    */
    int bindContext(unsigned int cpu)
    {
        int ret = -1;
    #if defined(__linux__) || defined(__FreeBSD__)
        cpu_set_t currentCPU;
        // add check for size of cpumask_t.
        MY_CPU_ZERO(&currentCPU);
        // turn on the equivalent bit inside the bitmap corresponding to affinitymask
        MY_CPU_SET(cpu, &currentCPU);
        sched_getaffinity(0, sizeof(prevAffinity), &prevAffinity);
        if (!sched_setaffinity(0, sizeof(currentCPU), &currentCPU)) ret = 0;
    #else // Windows
        if (cpu >= MAX_WIN7_LOG_CPU) return ret;

        // determine the group number and the cpu number within the group
        unsigned int groupId      = cpu / MAX_LOGICAL_PROCESSORS_PER_GROUP;
        unsigned int cpuIdInGroup = cpu - groupId * MAX_LOGICAL_PROCESSORS_PER_GROUP;

        GROUP_AFFINITY grp_affinity;
        _INTERNAL_DAAL_MEMSET(&grp_affinity, 0, sizeof(GROUP_AFFINITY));
        grp_affinity.Group = groupId;
        grp_affinity.Mask  = (KAFFINITY)((DWORD_PTR)(LNX_MY1CON << cpuIdInGroup));
        if (!SetThreadGroupAffinity(GetCurrentThread(), &grp_affinity, &prevAffinity))
        {
            return ret;
        }
        else
        {
            ret = 0;
        }
    #endif
        return ret;
    }

    void restoreContext()
    {
    #if defined(__linux__) || defined(__FreeBSD__)
        sched_setaffinity(0, sizeof(prevAffinity), &prevAffinity);
    #else
        SetThreadGroupAffinity(GetCurrentThread(), &prevAffinity, NULL);
    #endif
    }

    #if defined(__linux__) || defined(__FreeBSD__)
    cpu_set_t prevAffinity;
    #else
    GROUP_AFFINITY prevAffinity;
    #endif
};

glktsn::glktsn()
{
    isInit                = false;
    error                 = 0;
    hasLeafB              = false;
    maxCacheSubleaf       = -1;
    EnumeratedPkgCount    = 0;
    EnumeratedCoreCount   = 0;
    EnumeratedThreadCount = 0;
    HWMT_SMTperCore       = 0;
    HWMT_SMTperPkg        = 0;

    // call CPUID with leaf 0 to get the maximum supported CPUID leaf index
    CPUIDinfo info;
    info.get(0);
    maxCPUIDLeaf = info.EAX;

    // call OS-specific service to find out how many logical processors
    // are supported by the OS
    OSProcessorCount = getMaxCPUSupportedByOS();

    // Allocate memory to store the APIC ID, sub IDs, affinity mappings, etc.
    allocArrays(OSProcessorCount);
    if (error) return;

    for (unsigned i = 0; i < MAX_CACHE_SUBLEAFS; i++)
    {
        EnumeratedEachCacheCount[i] = 0;
        EachCacheSelectMask[i]      = 0;
        EachCacheMaskWidth[i]       = 0;
        cacheDetail[i]              = cacheDetail_str {};
    }

    // Initialize the global CPU topology object
    buildSystemTopologyTables();
}

/*
 * A wrapper function that calls OS specific system API to find out
 * how many logical processor the OS supports
 *
 * \return Number of logical processors enabled by the OS for this process
 */
unsigned int glktsn::getMaxCPUSupportedByOS()
{
    #if defined(__linux__) || defined(__FreeBSD__)

    OSProcessorCount = sysconf(_SC_NPROCESSORS_ONLN); //This will tell us how many CPUs are currently enabled.

    #else

    unsigned short grpCnt;
    DWORD cnt;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX * pSystem_rel_info = NULL;

    // runtime check if os version is greater than 0601h

    OSProcessorCount = 0;
    // if Windows version support processor groups
    // tally actually populated logical processors in each group
    grpCnt = (WORD)GetActiveProcessorGroupCount();
    cnt    = BLOCKSIZE_4K;
    _INTERNAL_DAAL_MEMSET(&scratch[0], 0, cnt);
    pSystem_rel_info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)&scratch[0];

    if (!GetLogicalProcessorInformationEx(RelationGroup, pSystem_rel_info, &cnt))
    {
        error |= _MSGTYP_UNKNOWNERR_OS;
        return 0;
    }
    if (pSystem_rel_info->Relationship != RelationGroup)
    {
        error |= _MSGTYP_UNKNOWNERR_OS;
        return 0;
    }

    for (unsigned int i = 0; i < grpCnt; i++)
    {
        const auto activeProcCount = pSystem_rel_info->Group.GroupInfo[i].ActiveProcessorCount;
        _INTERNAL_DAAL_OVERFLOW_CHECK_BY_ADDING(unsigned __int64, OSProcessorCount, activeProcCount);
        if (error & _MSGTYP_TOPOLOGY_NOTANALYZED) return 0;
        OSProcessorCount += activeProcCount;
    }
    #endif
    return OSProcessorCount;
}

/*
 * A wrapper function that calls OS specific system API to find out
 * which logical processors this process is allowed to run on.
 * And set the corresponding bits in our generic affinity mask construct.
 */
void glktsn::setChkProcessAffinityConsistency()
{
    #if defined(__linux__) || defined(__FreeBSD__)
    cpu_set_t allowedCPUs;

    sched_getaffinity(0, sizeof(allowedCPUs), &allowedCPUs);
    for (unsigned int i = 0; i < OSProcessorCount; i++)
    {
        // check that i-th core belongs to the process affinity mask
        if (MY_CPU_ISSET(i, &allowedCPUs))
        {
            if (cpu_generic_processAffinity.set(i))
            {
                error |= _MSGTYP_USERAFFINITYERR;
            }
            if (cpu_generic_systemAffinity.set(i))
            {
                error |= _MSGTYP_USERAFFINITYERR;
            }
        }
    }

    #else

    if (OSProcessorCount > MAX_WIN7_LOG_CPU)
    {
        error |= _MSGTYP_OSAFFCAP_ERROR; // If the os supports more processors than allowed, make change as required.
    }

    const unsigned short grpCnt = GetActiveProcessorGroupCount();
    if (grpCnt > MAX_THREAD_GROUPS_WIN7)
    {
        error |= _MSGTYP_OSAFFCAP_ERROR; // If the os supports more processor groups than allowed, make change as required.
        return;
    }
    // Take snapshot of all threads in the process

    DWORD processId       = GetCurrentProcessId();
    HANDLE snapshotHandle = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, processId);
    if (snapshotHandle == INVALID_HANDLE_VALUE)
    {
        error |= _MSGTYP_UNKNOWNERR_OS;
        return;
    }

    // Index of the array is the group number, value is the affinity mask
    KAFFINITY groupAffinityMap[MAX_THREAD_GROUPS_WIN7] = {};
    bool groupAffinityInitialized                      = false;

    // Iterate through all threads, get their group affinities and store them in the map
    THREADENTRY32 threadEntry;
    threadEntry.dwSize = sizeof(THREADENTRY32);

    if (Thread32First(snapshotHandle, &threadEntry))
    {
        do
        {
            if (threadEntry.th32OwnerProcessID == processId)
            {
                HANDLE threadHandle = OpenThread(THREAD_QUERY_INFORMATION, FALSE, threadEntry.th32ThreadID);
                if (threadHandle)
                {
                    GROUP_AFFINITY threadAffinity;
                    if (GetThreadGroupAffinity(threadHandle, &threadAffinity))
                    {
                        // Store the affinity mask for this group
                        groupAffinityMap[threadAffinity.Group] = threadAffinity.Mask;
                        groupAffinityInitialized               = true;
                    }
                    CloseHandle(threadHandle);
                }
            }
        } while (Thread32Next(snapshotHandle, &threadEntry));
    }

    CloseHandle(snapshotHandle);

    if (!groupAffinityInitialized)
    {
        error |= _MSGTYP_UNKNOWNERR_OS;
        return;
    }

    unsigned int sum = 0;
    for (unsigned int group = 0; group < grpCnt; group++)
    {
        KAFFINITY groupAffinityMask = groupAffinityMap[group];
        if (groupAffinityMask)
        {
            unsigned int cpu_cnt = __internal_daal_countBits(groupAffinityMask); // count bits on each target affinity group
            sum += cpu_cnt;
            if (sum > OSProcessorCount)
            {
                //throw some exception here, no full affinity for the process
                error |= _MSGTYP_USERAFFINITYERR;
                break;
            }

            for (unsigned int j = 0; j < MAX_LOGICAL_PROCESSORS_PER_GROUP; j++)
            {
                // check that j-th core belongs to the process affinity mask
                if (groupAffinityMask & (KAFFINITY)((DWORD_PTR)(LNX_MY1CON << j)))
                {
                    if (cpu_generic_processAffinity.set(j + group * MAX_LOGICAL_PROCESSORS_PER_GROUP))
                    {
                        error |= _MSGTYP_USERAFFINITYERR;
                        break;
                    }
                }
            }
        }
    }
    #endif
}

GenericAffinityMask::GenericAffinityMask(const unsigned numCpus)
{
    maxByteLength = (numCpus >> 3) + 1;
    AffinityMask  = (unsigned char *)_INTERNAL_DAAL_MALLOC(maxByteLength * sizeof(unsigned char));
    if (!AffinityMask) return;

    _INTERNAL_DAAL_MEMSET(AffinityMask, 0, maxByteLength * sizeof(unsigned char));
}

GenericAffinityMask::GenericAffinityMask(const GenericAffinityMask & other)
{
    maxByteLength      = other.maxByteLength;
    const int maskSize = maxByteLength * sizeof(unsigned char);
    AffinityMask       = (unsigned char *)_INTERNAL_DAAL_MALLOC(maskSize);
    if (!AffinityMask) return;

    _INTERNAL_DAAL_MEMCPY(AffinityMask, maskSize, other.AffinityMask, maskSize);
}

GenericAffinityMask::~GenericAffinityMask()
{
    if (AffinityMask)
    {
        _INTERNAL_DAAL_FREE(AffinityMask);
        AffinityMask = NULL;
    }
    maxByteLength = 0;
}

/*
 * Initialize APIC ID from leaf B if it else from leaf 1
 *
 * Arguments: None
 * Return: APIC ID
 */
void idAffMskOrdMapping_t::initApicID(bool hasLeafB)
{
    CPUIDinfo info;

    if (hasLeafB)
    {
        info.get(0xB);     // query subleaf 0 of leaf B
        APICID = info.EDX; //  x2APIC ID
    }
    else
    {
        info.get(1);
        APICID = (BYTE)(__internal_daal_getBitsFromDWORD(info.EBX, 24, 31)); // zero extend 8-bit initial APIC ID
    }
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
        if (!(package == ENUM_ALL || globalCPUTopology.pApicAffOrdMapping[i].packageORD == package)) continue;

        if (!(core == ENUM_ALL || core == globalCPUTopology.pApicAffOrdMapping[i].coreORD)) continue;

        if (!(logical == ENUM_ALL || logical == globalCPUTopology.pApicAffOrdMapping[i].threadORD)) continue;

        return i;
    }

    return 0;
}

// Derive bitmask extraction parameters used to extract/decompose x2APIC ID.
// The algorithm assumes CPUID feature symmetry across all physical packages.
// Since CPUID reporting by each logical processor in a physical package are identical, we only execute CPUID
// on one logical processor to derive these system-wide parameters
int glktsn::cpuTopologyLeafBConstants()
{
    CPUIDinfo infoB;
    bool wasCoreReported           = false;
    bool wasThreadReported         = false;
    int subLeaf                    = 0, levelType, levelShift;
    unsigned long coreplusSMT_Mask = 0;

    do
    {
        // we already tested CPUID leaf 0BH contain valid sub-leaves
        infoB.get(0xB, subLeaf);
        if (infoB.EBX == 0)
        {
            // if EBX == 0 then this subleaf is not valid, we can exit the loop
            break;
        }

        levelType  = __internal_daal_getBitsFromDWORD(infoB.ECX, 8, 15);
        levelShift = __internal_daal_getBitsFromDWORD(infoB.EAX, 0, 4);

        switch (levelType)
        {
        case 1:
            // level type is SMT, so levelShift is the SMT_Mask_Width
            SMTSelectMask     = ~((std::numeric_limits<decltype(SMTSelectMask)>::max()) << levelShift);
            SMTMaskWidth      = levelShift;
            wasThreadReported = true;
            break;
        case 2:
            // level type is Core, so levelShift is the CorePlsuSMT_Mask_Width
            coreplusSMT_Mask   = ~((std::numeric_limits<decltype(coreplusSMT_Mask)>::max()) << levelShift);
            PkgSelectMaskShift = levelShift;
            PkgSelectMask      = (-1) ^ coreplusSMT_Mask;
            wasCoreReported    = true;
            break;
        default:
            // handle in the future
            break;
        }

        subLeaf++;
    } while (1);

    if (wasThreadReported && wasCoreReported)
    {
        CoreSelectMask = coreplusSMT_Mask ^ SMTSelectMask;
    }
    else if (!wasCoreReported && wasThreadReported)
    {
        CoreSelectMask     = 0;
        PkgSelectMaskShift = SMTMaskWidth;
        PkgSelectMask      = (-1) ^ SMTSelectMask;
    }
    else // (case where !wasThreadReported)
    {
        // throw an error, this should not happen if hardware function normally
        error |= _MSGTYP_GENERAL_ERROR;
    }

    if (error) return error;

    return 0;
}

// Calculate parameters used to extract/decompose Initial APIC ID.
// The algorithm assumes CPUID feature symmetry across all physical packages.
// Since CPUID reporting by each logical processor in a physical package are identical, we only execute CPUID
// on one logical processor to derive these system-wide parameters
/*
 * Derive bitmask extraction parameter using CPUID leaf 1 and leaf 4
 *
 * Arguments:
 *     info - Point to strucrture containing CPIUD instruction leaf 1 data
 * Return: 0 is no error
 */
int glktsn::cpuTopologyLegacyConstants(CPUIDinfo * pinfo)
{
    unsigned corePlusSMTIDMaxCnt;
    unsigned coreIDMaxCnt       = 1;
    unsigned SMTIDPerCoreMaxCnt = 1;

    corePlusSMTIDMaxCnt = __internal_daal_getBitsFromDWORD(pinfo->EBX, 16, 23);
    if (maxCPUIDLeaf >= 4)
    {
        CPUIDinfo info4;
        info4.get(4, 0);
        coreIDMaxCnt       = __internal_daal_getBitsFromDWORD(info4.EAX, 26, 31) + 1; // related to coreMaskWidth
        SMTIDPerCoreMaxCnt = corePlusSMTIDMaxCnt / coreIDMaxCnt;
    }
    else
    {
        coreIDMaxCnt       = 1;
        SMTIDPerCoreMaxCnt = corePlusSMTIDMaxCnt / coreIDMaxCnt;
    }

    SMTSelectMask  = __internal_daal_createMask(SMTIDPerCoreMaxCnt, &SMTMaskWidth);
    CoreSelectMask = __internal_daal_createMask(coreIDMaxCnt, &PkgSelectMaskShift);
    PkgSelectMaskShift += SMTMaskWidth;
    CoreSelectMask <<= SMTMaskWidth;
    PkgSelectMask = (-1) ^ (CoreSelectMask | SMTSelectMask);

    if (error) return error;
    return 0;
}

/*
 * __internal_daal_getCacheTotalLize
 *
 * Caluculates the total capacity (bytes) from the cache parameters reported by CPUID leaf 4
 *          the caller provides the raw data reported by the target sub leaf of cpuid leaf 4
 *
 * Arguments:
 *     cache_type - the cache type encoding recognized by CPIUD instruction leaf 4 for the target cache level
 * Return: the sub-leaf index corresponding to the largest cache of specified cache type
 */
static unsigned long __internal_daal_getCacheTotalLize(const CPUIDinfo & info)
{
    unsigned long LnSz, SectorSz, WaySz, SetSz;

    LnSz     = __internal_daal_getBitsFromDWORD(info.EBX, 0, 11) + 1;
    SectorSz = __internal_daal_getBitsFromDWORD(info.EBX, 12, 21) + 1;
    WaySz    = __internal_daal_getBitsFromDWORD(info.EBX, 22, 31) + 1;
    SetSz    = __internal_daal_getBitsFromDWORD(info.ECX, 0, 31) + 1;

    return (SetSz * WaySz * SectorSz * LnSz);
}

/*
 * Find the subleaf index of CPUID leaf 4 corresponding to the input subleaf
 *
 * Arguments:
 *     cache_subleaf - the cache subleaf encoding recognized by CPIUD instruction leaf 4 for the target cache level
 * Return: the sub-leaf index corresponding to the largest cache of specified cache type
 */
int glktsn::findEachCacheIndex(unsigned cache_subleaf)
{
    unsigned i, type;
    unsigned long cap;
    CPUIDinfo info4;
    int target_index = -1;

    if (maxCPUIDLeaf < 4)
    {
        // if we can't get detailed parameters from cpuid leaf 4, this is stub pointers for older processors
        if (maxCPUIDLeaf >= 2)
        {
            if (!cache_subleaf)
            {
                cacheDetail[cache_subleaf].sizeKB = 0;
                return cache_subleaf;
            }
        }
        return -1;
    }

    info4.get(4, cache_subleaf);
    type = __internal_daal_getBitsFromDWORD(info4.EAX, 0, 4);
    cap  = __internal_daal_getCacheTotalLize(info4);

    if (type > 0)
    {
        cacheDetail[cache_subleaf].level = __internal_daal_getBitsFromDWORD(info4.EAX, 5, 7);
        cacheDetail[cache_subleaf].type  = type;
        for (i = 0; i <= cache_subleaf; i++)
        {
            if (cacheDetail[i].type == type) cacheDetail[i].how_many_caches_share_level++;
        }
        cacheDetail[cache_subleaf].sizeKB = cap / 1024;
        target_index                      = cache_subleaf;
    }

    return target_index;
}

/*
 * __internal_daal_initStructuredLeafBuffers
 *
 * Allocate buffers to store cpuid leaf data including buffers for subleaves within leaf 4 and 11 (0x4 and 0xb)
 *
 * Arguments: none
 * Return: none
 *
 */
void glktsn::initStructuredLeafBuffers()
{
    unsigned j, kk, qeidmsk;
    CPUIDinfo info;
    info.get(0);

    cpuid_values[0].subleaf[0] = (CPUIDinfo *)_INTERNAL_DAAL_MALLOC(sizeof(CPUIDinfo));
    if (!cpuid_values[0].subleaf[0])
    {
        error = -1;
        return;
    }

    cpuid_values[0].subleaf[0][0] = info;

    // Mark this combo of cpu, leaf, subleaf is valid
    cpuid_values[0].subleaf_max = 1;

    for (j = 1; j <= maxCPUIDLeaf; j++)
    {
        info.get(j);
        cpuid_values[j].subleaf[0] = (CPUIDinfo *)_INTERNAL_DAAL_MALLOC(sizeof(CPUIDinfo));
        if (!cpuid_values[j].subleaf[0])
        {
            error = -1;
            return;
        }
        cpuid_values[j].subleaf[0][0] = info;
        cpuid_values[j].subleaf_max   = 1;

        if (j == 0xd)
        {
            int subleaf                 = 2;
            cpuid_values[j].subleaf_max = 1;
            info.get(j, subleaf);
            while (info.EAX && subleaf < MAX_CACHE_SUBLEAFS)
            {
                cpuid_values[j].subleaf_max = subleaf;
                subleaf++;
                info.get(j, subleaf);
            }
        }
        else if (j == 0x10 || j == 0xf)
        {
            int subleaf = 1;
            info.get(j, subleaf);
            if (j == 0xf)
                qeidmsk = info.EDX; // sub-leaf value are derived from valid resource id's
            else
                qeidmsk = info.EBX;
            kk = 1;
            while (kk < 32)
            {
                info.get(j, kk);
                if ((qeidmsk & (1 << kk)) != 0) cpuid_values[j].subleaf_max = kk;
                kk++;
            }
        }
        else if ((j == 0x4 || j == 0xb))
        {
            int subleaf       = 1;
            unsigned int type = 1;
            while (type && (subleaf < MAX_CACHE_SUBLEAFS))
            {
                info.get(j, subleaf);
                if (j == 0x4)
                    type = __internal_daal_getBitsFromDWORD(info.EAX, 0, 4);
                else
                    type = 0xffff & info.EBX;
                cpuid_values[j].subleaf[subleaf] = (CPUIDinfo *)_INTERNAL_DAAL_MALLOC(sizeof(CPUIDinfo));
                if (!cpuid_values[j].subleaf[subleaf])
                {
                    error = -1;
                    return;
                }
                cpuid_values[j].subleaf[subleaf][0] = info;
                subleaf++;
                cpuid_values[j].subleaf_max = subleaf;
            }
        }
    }
}

/*
 * Calculates the select mask that can be used to extract Cache_ID from an APIC ID
 * The caller must specify which target cache level it wishes to extract Cache_IDs
 *
 * \param targ_subleaf  The subleaf index to execute CPIUD instruction leaf 4 for the target cache level, provided by parent
 *
 * \return 0 is no error
 */
int glktsn::eachCacheTopologyParams(unsigned targ_subleaf)
{
    unsigned long SMTMaxCntPerEachCache;
    CPUIDinfo info;

    info.get(1);
    if (maxCPUIDLeaf >= 4)
    {
        // CPUID leaf 4 and HT are supported
        CPUIDinfo info4;
        info4.get(4, targ_subleaf);

        SMTMaxCntPerEachCache = __internal_daal_getBitsFromDWORD(info4.EAX, 14, 25) + 1;
    }
    else if (__internal_daal_getBitsFromDWORD(info.EDX, 28, 28))
    {
        // no support for CPUID leaf 4 but HT is supported
        SMTMaxCntPerEachCache = __internal_daal_getBitsFromDWORD(info.EBX, 16, 23);
    }
    else
    {
        // no support for CPUID leaf 4 and no HT
        SMTMaxCntPerEachCache = 1;
    }

    EachCacheSelectMask[targ_subleaf] = __internal_daal_createMask(SMTMaxCntPerEachCache, &EachCacheMaskWidth[targ_subleaf]);

    return 0;
}

// Calculate parameters used to extract/decompose APIC ID for cache topology
// The algorithm assumes CPUID feature symmetry across all physical packages.
// Since CPUID reporting by each logical processor in a physical package are identical, we only execute CPUID
// on one logical processor to derive these system-wide parameters
// return 0 if successful, non-zero if error occurred
int glktsn::cacheTopologyParams()
{
    int targ_index;
    unsigned subleaf_max = 0;

    // Let's also examine cache topology.
    // As an example choose the largest unified cache as target level
    if (maxCPUIDLeaf >= 4)
    {
        initStructuredLeafBuffers();
        if (error) return -1;

        maxCacheSubleaf = 0;
        subleaf_max     = cpuid_values[4].subleaf_max;
    }
    else if (maxCPUIDLeaf >= 2)
    {
        maxCacheSubleaf = 0;
        subleaf_max     = 4;
    }

    for (unsigned subleaf = 0; subleaf < subleaf_max; subleaf++)
    {
        targ_index = findEachCacheIndex(subleaf);
        if (targ_index >= 0)
        {
            maxCacheSubleaf = targ_index;
            eachCacheTopologyParams(targ_index);
        }
        else
        {
            break;
        }
    }

    if (error) return -1;

    return 0;
}

// Derive parameters used to extract/decompose APIC ID for CPU topology
// The algorithm assumes CPUID feature symmetry across all physical packages.
// Since CPUID reporting by each logical processor in a physical package are
// identical, we only execute CPUID on one logical processor to derive these
// system-wide parameters
// return 0 if successful, non-zero if error occurred
int glktsn::cpuTopologyParams()
{
    CPUIDinfo info; // data structure to store register data reported by CPUID

    // cpuid leaf B detection
    if (maxCPUIDLeaf >= 0xB)
    {
        CPUIDinfo CPUInfoB;
        CPUInfoB.get(0xB, 0);
        //glbl_ptr points to assortment of global data, workspace, etc
        hasLeafB = (CPUInfoB.EBX != 0);
    }

    info.get(1, 0);

    // Use HWMT feature flag CPUID.01:EDX[28] to treat three configurations:
    if (__internal_daal_getBitsFromDWORD(info.EDX, 28, 28))
    {
        // Processors that support Hyper-Threading
        if (hasLeafB)
        {
            // #1, Processors that support CPUID leaf 0BH
            // use CPUID leaf B to derive extraction parameters
            cpuTopologyLeafBConstants();
        }
        else
        {
            //#2, Processors that support legacy parameters
            //  using CPUID leaf 1 and leaf 4
            cpuTopologyLegacyConstants(&info);
        }
    }
    else
    {
        //#3, Prior to HT, there is only one logical processor in a physical package
        CoreSelectMask     = 0;
        SMTMaskWidth       = 0;
        PkgSelectMask      = (unsigned)(-1);
        PkgSelectMaskShift = 0;
        SMTSelectMask      = 0;
    }

    if (error) return -1;

    return 0;
}

Dyn2Arr_str::Dyn2Arr_str(const unsigned xdim, const unsigned ydim, const unsigned value)
{
    dim[0] = xdim;
    dim[1] = ydim;
    data   = (unsigned *)_INTERNAL_DAAL_MALLOC(xdim * ydim * sizeof(unsigned));
    if (!data)
    {
        return;
    }
    _INTERNAL_DAAL_MEMSET(data, value, xdim * ydim * sizeof(unsigned));
}

Dyn2Arr_str::Dyn2Arr_str(const Dyn2Arr_str & other)
{
    if (this == &other) return; // self-assignment check
    dim[0]          = other.dim[0];
    dim[1]          = other.dim[1];
    size_t dataSize = dim[0] * dim[1] * sizeof(unsigned);
    data            = (unsigned *)_INTERNAL_DAAL_MALLOC(dataSize);
    if (!data) return;
    _INTERNAL_DAAL_MEMCPY(data, dataSize, other.data, dataSize);
}

Dyn2Arr_str::~Dyn2Arr_str()
{
    if (data)
    {
        _INTERNAL_DAAL_FREE(data);
        data = NULL;
    }
    dim[0] = 0;
    dim[1] = 0;
}

Dyn1Arr_str::Dyn1Arr_str(const unsigned xdim, const unsigned value)
{
    dim[0] = xdim;
    data   = (unsigned *)_INTERNAL_DAAL_MALLOC(xdim * sizeof(unsigned));
    if (!data)
    {
        return;
    }
    fill(value);
}

Dyn1Arr_str::Dyn1Arr_str(const Dyn1Arr_str & other)
{
    if (this == &other) return; // self-assignment check
    dim[0]          = other.dim[0];
    size_t dataSize = dim[0] * sizeof(unsigned);
    data            = (unsigned *)_INTERNAL_DAAL_MALLOC(dataSize);
    if (!data) return;
    _INTERNAL_DAAL_MEMCPY(data, dataSize, other.data, dataSize);
}

Dyn1Arr_str::~Dyn1Arr_str()
{
    if (data)
    {
        _INTERNAL_DAAL_FREE(data);
        data = NULL;
    }
    dim[0] = 0;
}

void Dyn1Arr_str::fill(const unsigned value)
{
    _INTERNAL_DAAL_MEMSET(data, value, dim[0] * sizeof(unsigned));
}

/*
 * Allocate the dynamic arrays in the global object containing various buffers for analyzing
 * cpu topology and cache topology of a system with N logical processors
 *
 * \param cpus  Number of logical processors
 *
 * \return 0 is no error, -1 is error
 */
int glktsn::allocArrays(const unsigned cpus)
{
    pApicAffOrdMapping = (idAffMskOrdMapping_t *)_INTERNAL_DAAL_MALLOC(cpus * sizeof(idAffMskOrdMapping_t));
    if (!pApicAffOrdMapping)
    {
        error = -1;
        return -1;
    }
    _INTERNAL_DAAL_MEMSET(pApicAffOrdMapping, 0, cpus * sizeof(idAffMskOrdMapping_t));

    perPkg_detectedCoresCount    = Dyn1Arr_str(cpus);
    perCore_detectedThreadsCount = Dyn2Arr_str(cpus, MAX_CORES);
    // workspace for storing hierarchical counts relative to the cache topology
    // of the largest unified cache (may be shared by several cores)
    perCache_detectedCoreCount       = Dyn1Arr_str(cpus);
    perEachCache_detectedThreadCount = Dyn2Arr_str(cpus, MAX_CACHE_SUBLEAFS);
    if (perPkg_detectedCoresCount.isEmpty() || perCore_detectedThreadsCount.isEmpty() || perEachCache_detectedThreadCount.isEmpty()
        || perCache_detectedCoreCount.isEmpty())
    {
        error = -1;
        return -1;
    }

    cpuid_values = (CPUIDinfox *)_INTERNAL_DAAL_MALLOC((maxCPUIDLeaf + 1) * sizeof(CPUIDinfox));
    if (!cpuid_values)
    {
        error = -1;
        return -1;
    }
    _INTERNAL_DAAL_MEMSET(cpuid_values, 0, (maxCPUIDLeaf + 1) * sizeof(CPUIDinfox));

    return 0;
}

/*
 * after execution context has already bound to the target logical processor
 * Query the 32-bit x2APIC ID if the processor supports it, or
 * Query the 8bit initial APIC ID for older processors
 * Apply various system-wide topology constant to parse the APIC ID into various sub IDs
 *
 * Arguments:
 *      cpu - the ordinal index to reference a logical processor in the system
 *      numMappings - running count ot how many processors we've parsed
 */
idAffMskOrdMapping_t::idAffMskOrdMapping_t(unsigned int cpu, bool hasLeafB, unsigned globalPkgSelectMask, unsigned globalPkgSelectMaskShift,
                                           unsigned globalCoreSelectMask, unsigned globalSMTSelectMask, unsigned globalSMTMaskWidth,
                                           std::array<unsigned, MAX_CACHE_SUBLEAFS> & globalEachCacheSelectMask, unsigned globalmaxCacheSubleaf)
{
    initApicID(hasLeafB);

    OrdIndexOAMsk = cpu; // this an ordinal number that can relate to generic affinitymask
    pkg_IDAPIC    = ((APICID & globalPkgSelectMask) >> globalPkgSelectMaskShift);
    Core_IDAPIC   = ((APICID & globalCoreSelectMask) >> globalSMTMaskWidth);
    SMT_IDAPIC    = (APICID & globalSMTSelectMask);

    if (globalmaxCacheSubleaf != -1)
    {
        for (unsigned subleaf = 0; subleaf <= globalmaxCacheSubleaf; subleaf++)
        {
            EaCacheSMTIDAPIC[subleaf] = (APICID & globalEachCacheSelectMask[subleaf]);
            EaCacheIDAPIC[subleaf]    = (APICID & (-1 ^ globalEachCacheSelectMask[subleaf]));
        }
    }
}

/*
 * Use OS specific service to find out how many logical processors can be accessed
 * by this application.
 * Querying CPUID on each logical processor requires using OS-specific API to
 * bind current context to each logical processor first.
 * After gathering the APIC ID's for each logical processor,
 * we can parse APIC ID into sub IDs for each topological levels
 * The thread affnity API to bind the current context limits us
 * in dealing with the limit of specific OS
 * The iteration loop to go through each logical processor managed by the OS can be done
 * in a manner that abstract the OS-specific affinity mask data structure.
 * Here, we construct a generic affinity mask that can handle arbitrary number of logical processors.
 *
 * Return: 0 is no error
 */
int glktsn::initEnumeratedThreadCountAndParseAPICIDs()
{
    EnumeratedThreadCount = 0;

    // we will use our generic affinity bitmap that can be generalized from
    // OS specific affinity mask constructs or the bitmap representation of an OS
    cpu_generic_processAffinity = GenericAffinityMask(OSProcessorCount);
    cpu_generic_systemAffinity  = GenericAffinityMask(OSProcessorCount);
    if (cpu_generic_processAffinity.AffinityMask == NULL || cpu_generic_systemAffinity.AffinityMask == NULL)
    {
        return -1;
    }

    // Set the affinity bits of our generic affinity bitmap according to
    // the system affinity mask and process affinity mask
    setChkProcessAffinityConsistency();
    if (error)
    {
        return -1;
    }

    for (unsigned i = 0; i < OSProcessorCount; i++)
    {
        // can't asume OS affinity bit mask is contiguous,
        // but we are using our generic bitmap representation for affinity
        unsigned char processAffinityBit = cpu_generic_processAffinity.test(i);
        if (processAffinityBit == 1)
        {
            // bind the execution context to the i-th logical processor
            // using OS-specific API
            volatile ScopedThreadContext ctx(i);
            if (ctx.error)
            {
                // Failed to set thread affinity
                error = -1;
                break;
            }

            pApicAffOrdMapping[EnumeratedThreadCount++] = idAffMskOrdMapping_t(i, hasLeafB, PkgSelectMask, PkgSelectMaskShift, CoreSelectMask,
                                                                               SMTSelectMask, SMTMaskWidth, EachCacheSelectMask, maxCacheSubleaf);
        }
        else if (processAffinityBit == 0xff)
        {
            // should never happen
            // i-th bit is out of bounds of the process affinity mask
            error = -1;
            break;
        }
    }

    cpu_generic_processAffinity = GenericAffinityMask();
    cpu_generic_systemAffinity  = GenericAffinityMask();

    if (error) return -1;

    return EnumeratedThreadCount;
}

/*
 * Analyze the Pkg_ID, Core_ID to derive hierarchical ordinal numbering scheme
 *
 * \return 0 is no error
 */
int glktsn::analyzeCPUHierarchy()
{
    // allocate workspace to sort parents and siblings in the topology
    // starting from pkg_ID and work our ways down each inner level
    Dyn1Arr_str pDetectedPkgIDs(EnumeratedThreadCount, 0xff);

    // we got a 1-D array to store unique Pkg_ID as we sort thru
    // each logical processor
    if (pDetectedPkgIDs.isEmpty()) return -1;

    // we got a 2-D array to store unique Core_ID within each Pkg_ID,
    // as we sort thru each logical processor
    Dyn2Arr_str pDetectCoreIDsperPkg(EnumeratedThreadCount, (1 << PkgSelectMaskShift), 0xff);
    if (pDetectCoreIDsperPkg.isEmpty()) return -1;

    // iterate throught each logical processor in the system.
    // mark up each unique physical package with a zero-based numbering scheme
    // Within each distinct package, mark up distinct cores within that package
    // with a zero-based numbering scheme
    unsigned maxPackageDetetcted = 0;
    for (unsigned i = 0; i < EnumeratedThreadCount; i++)
    {
        bool PkgMarked     = false;
        unsigned packageID = pApicAffOrdMapping[i].pkg_IDAPIC;
        unsigned coreID    = pApicAffOrdMapping[i].Core_IDAPIC;

        for (unsigned h = 0; h < maxPackageDetetcted; h++)
        {
            if (pDetectedPkgIDs[h] == packageID)
            {
                bool foundCore = false;
                unsigned k;
                PkgMarked                        = true;
                pApicAffOrdMapping[i].packageORD = h;

                // look for core in marked packages
                for (k = 0; k < perPkg_detectedCoresCount[h]; k++)
                {
                    if (coreID == pDetectCoreIDsperPkg[h * EnumeratedThreadCount + k])
                    {
                        foundCore = true;
                        // add thread - can't be that the thread already exists, breaks uniqe APICID spec
                        pApicAffOrdMapping[i].coreORD   = k;
                        pApicAffOrdMapping[i].threadORD = perCore_detectedThreadsCount[h * MAX_CORES + k];
                        perCore_detectedThreadsCount[h * MAX_CORES + k]++;
                        break;
                    }
                }

                if (!foundCore)
                {
                    // mark up the Core_ID of an unmarked core in a marked package
                    unsigned core                                          = perPkg_detectedCoresCount[h];
                    pDetectCoreIDsperPkg[h * EnumeratedThreadCount + core] = coreID;

                    // keep track of respective hierarchical counts
                    perCore_detectedThreadsCount[h * MAX_CORES + core] = 1;
                    perPkg_detectedCoresCount[h]++;

                    // build a set of numbering system to iterate each topological hierarchy
                    pApicAffOrdMapping[i].coreORD   = core;
                    pApicAffOrdMapping[i].threadORD = 0;
                    EnumeratedCoreCount++; // this is an unmarked core, increment system core count by 1
                }

                break;
            }
        }

        if (!PkgMarked)
        {
            // mark up the pkg_ID and Core_ID of an unmarked package
            pDetectedPkgIDs[maxPackageDetetcted]                                  = packageID;
            pDetectCoreIDsperPkg[maxPackageDetetcted * EnumeratedThreadCount + 0] = coreID;

            // keep track of respective hierarchical counts
            perPkg_detectedCoresCount[maxPackageDetetcted]                    = 1;
            perCore_detectedThreadsCount[maxPackageDetetcted * MAX_CORES + 0] = 1;

            // build a set of zero-based numbering acheme so that
            // each logical processor in the same core can be referenced by a zero-based index
            // each core in the same package can be referenced by another zero-based index
            // each package in the system can be referenced by a third zero-based index scheme.
            // each system wide index i can be mapped to a triplet of zero-based hierarchical indices
            pApicAffOrdMapping[i].packageORD = maxPackageDetetcted;
            pApicAffOrdMapping[i].coreORD    = 0;
            pApicAffOrdMapping[i].threadORD  = 0;

            maxPackageDetetcted++; // this is an unmarked pkg, increment pkg count by 1
            EnumeratedCoreCount++; // there is at least one core in a package
        }
    }

    EnumeratedPkgCount = maxPackageDetetcted;

    return 0;
}

/*
 * This is an example illustrating cache topology analysis of the largest unified cache
 * The largest unified cache may be shared by multiple cores
 *          parse APIC ID into sub IDs for each topological levels
 * This example illustrates several mapping relationships:
 *      1. count distinct target level cache in the system;
 *      2. count distinct cores sharing the same cache;
 *      3. Establish a hierarchical numbering scheme (0-based, ordinal number) for each distinct entity within a hierarchical level
 *
 * \param subleaf    Cache subleaf
 *
 * \return 0 is no error
 */
int glktsn::analyzeEachCHierarchy(unsigned subleaf)
{
    if (EachCacheMaskWidth[subleaf] == 0xffffffff) return -1;

    unsigned maxCacheDetected = 0;
    {
        Dyn1Arr_str pEachCIDs(EnumeratedThreadCount, 0xff);
        if (pEachCIDs.isEmpty()) return -1;

        // enumerate distinct caches of the same subleaf index to get counts of how many caches only
        // mark up each unique cache associated with subleaf index based on the cache_ID
        for (unsigned i = 0; i < EnumeratedThreadCount; i++)
        {
            unsigned j;
            for (j = 0; j < maxCacheDetected; j++)
            {
                if (pEachCIDs[j] == pApicAffOrdMapping[i].EaCacheIDAPIC[subleaf])
                {
                    break;
                }
            }
            if (j >= maxCacheDetected)
            {
                pEachCIDs[maxCacheDetected++] = pApicAffOrdMapping[i].EaCacheIDAPIC[subleaf];
            }
        }
        EnumeratedEachCacheCount[subleaf] = maxCacheDetected;
    }

    {
        Dyn1Arr_str pThreadIDsperEachC(EnumeratedThreadCount, 0xff);
        if (pThreadIDsperEachC.isEmpty()) return -1;

        // enumerate distinct SMT threads within a caches of the subleaf index only without relation to core topology
        // mark up the distinct logical processors sharing a distinct cache level associated subleaf index

        unsigned maxThreadsDetected = 0;
        for (unsigned i = 0; i < EnumeratedThreadCount; i++)
        {
            unsigned j;
            for (j = 0; j < maxThreadsDetected; j++)
            {
                if (pThreadIDsperEachC[j] == pApicAffOrdMapping[i].EaCacheSMTIDAPIC[subleaf])
                {
                    break;
                }
            }

            if (j >= maxThreadsDetected)
            {
                pThreadIDsperEachC[maxThreadsDetected++] = pApicAffOrdMapping[i].EaCacheSMTIDAPIC[subleaf];
            }
        }
    }

    Dyn1Arr_str pDetectedEachCIDs(EnumeratedThreadCount + 1, 0xff);
    Dyn2Arr_str pDetectThreadIDsperEachC(EnumeratedThreadCount, maxCacheDetected + 1, 0xff);
    if (pDetectedEachCIDs.isEmpty() || pDetectThreadIDsperEachC.isEmpty())
    {
        return -1;
    }

    // enumerate distinct SMT threads and cores relative to a cache level of the subleaf index
    // the enumeration below gets the counts and establishes zero-based numbering scheme for cores and SMT threads under each cache
    maxCacheDetected = 0;

    for (unsigned i = 0; i < EnumeratedThreadCount; i++)
    {
        pApicAffOrdMapping[i].EachCacheORD[subleaf] = (unsigned)-1;
    }

    for (unsigned i = 0; i < EnumeratedThreadCount; i++)
    {
        unsigned CacheID  = pApicAffOrdMapping[i].EaCacheIDAPIC[subleaf]; // sub ID to enumerate different caches in the system
        unsigned threadID = pApicAffOrdMapping[i].EaCacheSMTIDAPIC[subleaf];

        bool CacheMarked = false;
        for (unsigned h = 0; h < maxCacheDetected; h++)
        {
            if (pDetectedEachCIDs[h] == CacheID)
            {
                bool foundThread = false;

                CacheMarked                                 = true;
                pApicAffOrdMapping[i].EachCacheORD[subleaf] = h;

                // look for cores sharing the same target cache level
                for (unsigned k = 0; k < perEachCache_detectedThreadCount[h * MAX_CACHE_SUBLEAFS + subleaf]; k++)
                {
                    if (threadID == pDetectThreadIDsperEachC[h * EnumeratedThreadCount + k])
                    {
                        foundThread                                        = true;
                        pApicAffOrdMapping[i].threadPerEaCacheORD[subleaf] = k;
                        break;
                    }
                }

                if (!foundThread)
                {
                    // mark up the thread_ID of an unmarked core in a marked package
                    unsigned thread                                              = perEachCache_detectedThreadCount[h * MAX_CACHE_SUBLEAFS + subleaf];
                    pDetectThreadIDsperEachC[h * EnumeratedThreadCount + thread] = threadID;

                    // keep track of respective hierarchical counts
                    perEachCache_detectedThreadCount[h * MAX_CACHE_SUBLEAFS + subleaf]++;

                    // build a set of numbering system to iterate the child hierarchy below the target cache
                    pApicAffOrdMapping[i].threadPerEaCacheORD[subleaf] = thread;
                }

                break;
            }
        }

        if (!CacheMarked)
        {
            // mark up the pkg_ID and Core_ID of an unmarked package
            pDetectedEachCIDs[maxCacheDetected]                                = CacheID;
            pDetectThreadIDsperEachC[maxCacheDetected * EnumeratedThreadCount] = threadID;

            // keep track of respective hierarchical counts
            perEachCache_detectedThreadCount[maxCacheDetected * MAX_CACHE_SUBLEAFS + subleaf] = 1;

            // build a set of numbering system to iterate each topological hierarchy
            pApicAffOrdMapping[i].EachCacheORD[subleaf]        = maxCacheDetected;
            pApicAffOrdMapping[i].threadPerEaCacheORD[subleaf] = 0;

            maxCacheDetected++; // this is an unmarked cache, increment cache count by 1
        }
    }

    return 0;
}

/*
 * Construct the processor topology tables and values necessary to
 * support the external functions that display CPU topology and/or
 * cache topology derived from system topology enumeration.
 *
 * Return: None, sets globalCPUTopology.error if tables or values can not be calculated.
 */
void glktsn::buildSystemTopologyTables()
{
    // allocated the memory buffers within the global pointer

    // Gather all the system-wide constant parameters needed to derive topology information
    error = cpuTopologyParams();
    if (error) return;
    error = cacheTopologyParams();
    if (error) return;

    // For each logical processor, collect APIC ID and parse sub IDs for each APIC ID
    int numMappings = initEnumeratedThreadCountAndParseAPICIDs();
    if (numMappings < 0)
    {
        error = numMappings;
        return;
    }
    // Derived separate numbering schemes for each level of the cpu topology
    if (analyzeCPUHierarchy() < 0)
    {
        error |= _MSGTYP_TOPOLOGY_NOTANALYZED;
    }

    // an example of building cache topology info for each cache level
    if (maxCacheSubleaf != -1)
    {
        for (unsigned subleaf = 0; subleaf <= maxCacheSubleaf; subleaf++)
        {
            if (EachCacheMaskWidth[subleaf] != 0xffffffff)
            {
                // ensure there is at least one core in the target level cache
                if (analyzeEachCHierarchy(subleaf) < 0) error |= _MSGTYP_TOPOLOGY_NOTANALYZED;
            }
        }
    }
    isInit = true;
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
    if (globalCPUTopology.error)
    {
        return 0xffffffff;
    }

    if (processor >= globalCPUTopology.OSProcessorCount) return 0xffffffff; // allow caller to intercept error

    return globalCPUTopology.pApicAffOrdMapping[processor].APICID;
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
    if (globalCPUTopology.error || package_ordinal >= globalCPUTopology.EnumeratedPkgCount)
    {
        return 0;
    }

    return globalCPUTopology.perPkg_detectedCoresCount[package_ordinal];
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
    if (globalCPUTopology.error || package_ordinal >= globalCPUTopology.EnumeratedPkgCount)
    {
        return 0;
    }

    if (core_ordinal >= globalCPUTopology.perPkg_detectedCoresCount[package_ordinal]) return 0;

    return globalCPUTopology.perCore_detectedThreadsCount[package_ordinal * MAX_CORES + core_ordinal];
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
    if (globalCPUTopology.error)
    {
        return 0;
    }

    return globalCPUTopology.EnumeratedEachCacheCount[subleaf];
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
    if (globalCPUTopology.error) return 0;

    return globalCPUTopology.OSProcessorCount;
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
    if (globalCPUTopology.error) return 0;

    return globalCPUTopology.EnumeratedThreadCount;
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
    if (globalCPUTopology.error) return 0;

    return globalCPUTopology.EnumeratedCoreCount;
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
    if (globalCPUTopology.error) return 0;

    return globalCPUTopology.EnumeratedPkgCount;
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
    if (globalCPUTopology.error || cache_ordinal >= globalCPUTopology.EnumeratedEachCacheCount[subleaf]) return 0;

    return globalCPUTopology.perEachCache_detectedThreadCount[cache_ordinal * MAX_CACHE_SUBLEAFS + subleaf];
}

unsigned _internal_daal_GetLogicalProcessorQueue(int * queue)
{
    const int cpus = _internal_daal_GetSysLogicalProcessorCount();
    int cores      = _internal_daal_GetProcessorCoreCount();

    if (cores == 0) cores = 1;

    int ht = cpus / cores;
    if (ht < 1 || ht >= cpus)
    {
        globalCPUTopology.error |= _MSGTYP_GENERAL_ERROR;
        return globalCPUTopology.error;
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
                    if (globalCPUTopology.pApicAffOrdMapping[j].packageORD == pkg && globalCPUTopology.pApicAffOrdMapping[j].EachCacheORD[0] == i)
                    {
                        int jj = ((q / ht) + (cores * (q % ht))) % cpus;
                        if (jj < cpus) queue[jj] = j;
                        q++;
                    }
                }
            }
        }
    }

    return globalCPUTopology.error;
}

unsigned _internal_daal_GetStatus()
{
    return globalCPUTopology.error;
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

volatile static bool cache_sizes_read                   = false;
volatile static long long cache_sizes[MAX_CACHE_LEVELS] = { DEFAULT_L1_CACHE_SIZE, DEFAULT_L2_CACHE_SIZE, DEFAULT_LL_CACHE_SIZE, 0 };

static __inline void update_cache_sizes()
{
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

void glktsn::freeArrays()
{
    isInit = false;
    _INTERNAL_DAAL_FREE(pApicAffOrdMapping);

    if (cpuid_values)
    {
        for (unsigned int i = 0; i <= maxCPUIDLeaf; i++)
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

#else // DAAL_CPU_TOPO_DISABLED

namespace daal
{
namespace services
{
namespace internal
{
size_t getL1CacheSize()
{
    return DEFAULT_L1_CACHE_SIZE;
}

size_t getL2CacheSize()
{
    return DEFAULT_L2_CACHE_SIZE;
}

size_t getLLCacheSize()
{
    return DEFAULT_LL_CACHE_SIZE; //estimate based on mac pro
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
