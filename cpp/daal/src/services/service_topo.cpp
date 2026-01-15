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
    #define MAX_CORES                        MAX_LOG_CPU
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

    #define _INTERNAL_DAAL_OVERFLOW_CHECK_BY_ADDING(type, op1, op2) \
        {                                                           \
            volatile type r = (op1) + (op2);                        \
            r -= (op1);                                             \
            if (!(r == (op2))) error |= _MSGTYP_INT_OVERFLOW;       \
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

// The width of affinity mask in legacy Windows API is 32 or 64, depending on
// 32-bit or 64-bit OS.
// Linux abstract its equivalent bitmap cpumask_t from direct programmer access,
// cpumask_t can support more than 64 cpu, but is configured at kernel
// compile time by configuration parameter.
// To abstract the size difference of the bitmap used by different OSes
// for topology analysis, we use an unsigned char buffer that can
// map to different OS implementation's affinity mask data structure
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
        unsigned tmp    = first.cpuCount;
        first.cpuCount  = second.cpuCount;
        second.cpuCount = tmp;

        tmp                  = first.maxByteLength;
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
    unsigned cpuCount            = 0;
    unsigned char * AffinityMask = nullptr;
};

struct Dyn2Arr_str
{
    Dyn2Arr_str() : dim0(0), dim1(0), data(nullptr) {}
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

    unsigned size() const { return dim0 * dim1; }

    void fill(const unsigned value);

    friend void swap(Dyn2Arr_str & first, Dyn2Arr_str & second) // nothrow
    {
        unsigned tmp = first.dim0;
        first.dim0   = second.dim0;
        second.dim0  = tmp;

        tmp         = first.dim1;
        first.dim1  = second.dim1;
        second.dim1 = tmp;

        unsigned * tmpData = first.data;
        first.data         = second.data;
        second.data        = tmpData;
    }

private:
    unsigned dim0;
    unsigned dim1;
    unsigned * data; // data array to be malloc'd
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
        DAAL_ASSERT(idx < dim0);
        return data[idx];
    }

    const unsigned & operator[](unsigned idx) const
    {
        DAAL_ASSERT(idx < dim0);
        return data[idx];
    }

    bool isEmpty() const { return (data == nullptr); }

    void fill(const unsigned value);

    void reset(const unsigned xdim)
    {
        if (xdim > dim0)
        {
            _INTERNAL_DAAL_FREE(data);
            data = (unsigned *)_INTERNAL_DAAL_MALLOC(xdim * sizeof(unsigned));
        }
        dim0 = xdim;
    }

    friend void swap(Dyn1Arr_str & first, Dyn1Arr_str & second) // nothrow
    {
        unsigned tmp = first.dim0;
        first.dim0   = second.dim0;
        second.dim0  = tmp;

        unsigned * tmpData = first.data;
        first.data         = second.data;
        second.data        = tmpData;
    }

private:
    unsigned dim0;   // xdim
    unsigned * data; // data array to be malloc'd
};

struct idAffMskOrdMapping_t
{
    idAffMskOrdMapping_t() = default;
    explicit idAffMskOrdMapping_t(unsigned int cpu, bool hasLeafB, unsigned PkgSelectMask, unsigned PkgSelectMaskShift, unsigned CoreSelectMask,
                                  unsigned SMTSelectMask, unsigned SMTMaskWidth);
    unsigned __int32 APICID;        // the full x2APIC ID or initial APIC ID of a logical
                                    //  processor assigned by HW
    unsigned __int32 OrdIndexOAMsk; // An ordinal index (zero-based) for each logical
                                    //  processor in the system
    // Next three members are the sub IDs for processor topology enumeration
    unsigned __int32 pkg_IDAPIC;  // Pkg_ID field, subset of APICID bits
                                  //  to distinguish different packages
    unsigned __int32 Core_IDAPIC; // Core_ID field, subset of APICID bits to
                                  //  distinguish different cores in a package
    unsigned __int32 SMT_IDAPIC;  // SMT_ID field, subset of APICID bits to
                                  //  distinguish different logical processors in a core
private:
    void initApicID(bool hasLeafB);
};

// we are going to put an assortment of global variable, 1D and 2D arrays into
// a data structure. This is the declaration
struct glktsn
{
    // for each logical processor we need spaces to store APIC ID,
    // sub IDs, affinity mappings, etc.
    // Array of size EnumeratedThreadCount
    idAffMskOrdMapping_t * pApicAffOrdMapping = nullptr;

    // workspace for storing hierarchical counts of each level
    Dyn1Arr_str perPkg_detectedCoresCount;
    Dyn2Arr_str perCore_detectedThreadsCount;

    // we use an error code to indicate any abnoral situation
    unsigned error;

    unsigned OSProcessorCount = 0; // how many logical processor the OS sees
    bool hasLeafB;                 // flag to keep track of whether CPUID leaf 0BH is supported

    // the following global variables are the total counts in the system resulting from software enumeration
    unsigned EnumeratedCoreCount;
    unsigned EnumeratedThreadCount;

    unsigned maxCPUIDLeaf; // highest CPUID leaf index in a processor

    bool isInit = false;

    glktsn();

    ~glktsn() { freeArrays(); }

private:
    int allocArrays(const unsigned cpus);
    void freeArrays();
    unsigned getMaxCPUSupportedByOS();

    int parseAPICIDs(const GenericAffinityMask & processAffinity, unsigned & maxPkgSelectMaskShift);
    int analyzeCPUHierarchy(unsigned PkgSelectMaskShift);
    void buildSystemTopologyTables(const GenericAffinityMask & processAffinity);
};

// Global CPU topology object
static glktsn globalCPUTopology;

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
    explicit ScopedThreadContext(unsigned int cpu)
    {
        error = 0;
        error |= bindContext(cpu);
    }

    ~ScopedThreadContext() {}

    void restoreContext()
    {
    #if defined(__linux__) || defined(__FreeBSD__)
        if (sched_setaffinity(0, sizeof(prevAffinity), &prevAffinity))
        {
            error |= _MSGTYP_RESTORE_THREAD_AFFINITY_FAILED;
        }
    #else
        if (!SetThreadGroupAffinity(GetCurrentThread(), &prevAffinity, NULL))
        {
            error |= _MSGTYP_RESTORE_THREAD_AFFINITY_FAILED;
        }
    #endif
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
    #if defined(__linux__) || defined(__FreeBSD__)
        cpu_set_t currentCPU;
        // add check for size of cpumask_t.
        MY_CPU_ZERO(&currentCPU);
        // turn on the equivalent bit inside the bitmap corresponding to affinitymask
        MY_CPU_SET(cpu, &currentCPU);
        if (sched_getaffinity(0, sizeof(prevAffinity), &prevAffinity))
        {
            return _MSGTYP_GET_THREAD_AFFINITY_FAILED;
        }
        if (sched_setaffinity(0, sizeof(currentCPU), &currentCPU))
        {
            return _MSGTYP_SET_THREAD_AFFINITY_FAILED;
        }
    #else // Windows
        if (cpu >= MAX_WIN7_LOG_CPU) return _MSGTYP_OS_PROC_COUNT_EXCEEDED;

        // determine the group number and the cpu number within the group
        unsigned int groupId      = cpu / MAX_LOGICAL_PROCESSORS_PER_GROUP;
        unsigned int cpuIdInGroup = cpu - groupId * MAX_LOGICAL_PROCESSORS_PER_GROUP;

        GROUP_AFFINITY grp_affinity;
        _INTERNAL_DAAL_MEMSET(&grp_affinity, 0, sizeof(GROUP_AFFINITY));
        grp_affinity.Group = groupId;
        grp_affinity.Mask  = (KAFFINITY)((DWORD_PTR)(LNX_MY1CON << cpuIdInGroup));
        if (!SetThreadGroupAffinity(GetCurrentThread(), &grp_affinity, &prevAffinity))
        {
            return _MSGTYP_SET_THREAD_AFFINITY_FAILED;
        }
    #endif
        return 0;
    }

    #if defined(__linux__) || defined(__FreeBSD__)
    cpu_set_t prevAffinity;
    #else
    GROUP_AFFINITY prevAffinity;
    #endif
};

/*
 * A wrapper function that calls OS specific system API to find out
 * which logical processors this process is allowed to run on.
 * And set the corresponding bits in our generic affinity mask construct.
 *
 * \param[in,out] processAffinity  Generic affinity mask to store the process affinity bits.
 *                                 On input, cpuCount member must be set to the number
 *                                 of logical processors available on the OS.
 *
 * \return 0 if no error, otherwise error code.
 */
int setChkProcessAffinityConsistency(GenericAffinityMask & processAffinity)
{
    int error = 0;

    const unsigned OSProcessorCount = processAffinity.cpuCount;

    #if defined(__linux__) || defined(__FreeBSD__)
    cpu_set_t allowedCPUs;

    sched_getaffinity(0, sizeof(allowedCPUs), &allowedCPUs);
    for (unsigned int i = 0; i < OSProcessorCount; i++)
    {
        // check that i-th core belongs to the process affinity mask
        if (MY_CPU_ISSET(i, &allowedCPUs))
        {
            if (processAffinity.set(i))
            {
                error |= _MSGTYP_CANNOT_SET_AFFINITY_BIT;
                break;
            }
        }
    }

    #else

    if (OSProcessorCount > MAX_WIN7_LOG_CPU)
    {
        error |= _MSGTYP_OS_PROC_COUNT_EXCEEDED; // If the os supports more processors than allowed, make change as required.
        return error;
    }

    const unsigned short grpCnt = GetActiveProcessorGroupCount();

    if (grpCnt > MAX_THREAD_GROUPS_WIN7)
    {
        error |= _MSGTYP_OS_GROUP_COUNT_EXCEEDED; // If the os supports more processor groups than allowed, make change as required.
        return error;
    }

    // Take snapshot of all threads in the process
    DWORD processId       = GetCurrentProcessId();
    HANDLE snapshotHandle = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, processId);
    if (snapshotHandle == INVALID_HANDLE_VALUE)
    {
        error |= _MSGTYP_INVALID_SNAPSHOT_HANDLE;
        return error;
    }

    // Index of the array is the group number, value is the affinity mask
    KAFFINITY groupAffinityMap[MAX_THREAD_GROUPS_WIN7] = { 0, 0, 0, 0 };
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
                        groupAffinityMap[threadAffinity.Group] |= threadAffinity.Mask;
                        groupAffinityInitialized = true;
                    }
                    CloseHandle(threadHandle);
                }
            }
        } while (Thread32Next(snapshotHandle, &threadEntry));
    }

    CloseHandle(snapshotHandle);

    if (!groupAffinityInitialized)
    {
        error |= _MSGTYP_FAILED_TO_INIT_PROC_AFFINITY;
        return error;
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
                // should never happen, number of bits set in all groups exceeds number of processors available to the OS
                error |= _MSGTYP_USER_AFFINITY_ERROR;
                break;
            }

            for (unsigned int j = 0; (j < MAX_LOGICAL_PROCESSORS_PER_GROUP) && (group * MAX_LOGICAL_PROCESSORS_PER_GROUP + j < OSProcessorCount); j++)
            {
                // check that j-th core belongs to the process affinity mask
                if (groupAffinityMask & (KAFFINITY)((DWORD_PTR)(LNX_MY1CON << j)))
                {
                    if (processAffinity.set(j + group * MAX_LOGICAL_PROCESSORS_PER_GROUP))
                    {
                        error |= _MSGTYP_CANNOT_SET_AFFINITY_BIT;
                        break;
                    }
                }
            }
        }
    }
    #endif

    return error;
}

/*
 * Use OS specific service to find out how many logical processors can be accessed
 * by this application.
 *
 * \param[in,out] processAffinity  Generic affinity mask to store the process affinity bits.
 *                                 On input, cpuCount member must be set to the size
 *                                 of the full affinity mask of the OS.
 *                                 On output, the affinity bits will be set according to
 *                                 the process affinity mask retrieved from OS.
 *
 * \return Number of logical processors enabled by the OS for this process,
 *         or -1 in case of error.
 */
int initEnumeratedThreadCount(GenericAffinityMask & processAffinity)
{
    unsigned cpuCount              = processAffinity.cpuCount;
    unsigned enumeratedThreadCount = 0;

    // Set the affinity bits of our generic affinity bitmap according to
    // the system affinity mask and process affinity mask
    int error = setChkProcessAffinityConsistency(processAffinity);
    if (error)
    {
        return -1;
    }

    for (unsigned i = 0; i < cpuCount; i++)
    {
        // can't asume OS affinity bit mask is contiguous,
        // but we are using our generic bitmap representation for affinity
        unsigned char processAffinityBit = processAffinity.test(i);
        if (processAffinityBit == 1)
        {
            enumeratedThreadCount++;
        }
        else if (processAffinityBit == 0xff)
        {
            // should never happen
            // i-th bit is out of bounds of the process affinity mask
            error |= _MSGTYP_CANNOT_TEST_AFFINITY_BIT;
            break;
        }
    }

    if (error) return -1;

    return enumeratedThreadCount;
}

glktsn::glktsn()
{
    isInit                = false;
    error                 = 0;
    hasLeafB              = false;
    EnumeratedCoreCount   = 0;
    EnumeratedThreadCount = 0;

    // call CPUID with leaf 0 to get the maximum supported CPUID leaf index
    CPUIDinfo info;
    info.get(0);
    maxCPUIDLeaf = info.EAX;

    // call OS-specific service to find out how many logical processors
    // are supported by the OS
    OSProcessorCount = getMaxCPUSupportedByOS();

    #ifdef __linux__
    GenericAffinityMask processAffinity(OSProcessorCount);
    #else
    // For Windows, we need to consider processor groups.
    // Each group can contain up to 64 logical processors.
    const unsigned short grpCnt = GetActiveProcessorGroupCount();
    GenericAffinityMask processAffinity(grpCnt * MAX_LOGICAL_PROCESSORS_PER_GROUP);
    #endif
    if (processAffinity.AffinityMask == NULL)
    {
        error |= _MSGTYP_MEMORY_ALLOCATION_FAILED;
        return;
    }

    int threadCount = initEnumeratedThreadCount(processAffinity);
    if (threadCount < 0)
    {
        return;
    }
    EnumeratedThreadCount = (unsigned)threadCount;
    if (EnumeratedThreadCount == 0 || EnumeratedThreadCount > OSProcessorCount)
    {
        error |= _MSGTYP_USER_AFFINITY_ERROR;
        return;
    }
    // Allocate memory to store the APIC ID, sub IDs, affinity mappings, etc.
    allocArrays(EnumeratedThreadCount);
    if (error) return;

    // Initialize the global CPU topology object
    buildSystemTopologyTables(processAffinity);
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

    OSProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);

    #endif
    return OSProcessorCount;
}

GenericAffinityMask::GenericAffinityMask(const unsigned numCpus) : cpuCount(numCpus)
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
 * Initialize APIC ID from leaf B if it is available on the CPU, else from leaf 1
 *
 * \param hasLeafB  Flag to indicate whether CPUID leaf 0BH is supported
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

// Derive bitmask extraction parameters used to extract/decompose x2APIC ID.
//
// \param[out] PkgSelectMask        Mask to extract Pkg_ID field from APIC ID.
//                                  Pkg_ID is an index of the physical package in the system,
//                                  i.e. NUMA node index
// \param[out] PkgSelectMaskShift   Shift count to right shift APIC ID before applying PkgSelectMask
// \param[out] CoreSelectMask       Mask to extract Core_ID field from APIC ID.
//                                  Core_ID is an index of the core in a physical package
// \param[out] SMTSelectMask        Mask to extract SMT_ID (simultaneous multithreading)
//                                  field from APIC ID.
//                                  SMT_ID is an index of the logical processor in a core
// \param[out] SMTMaskWidth         Width of the SMT_ID field in bits
//
// \return 0 if successful, non-zero if error occurred
int cpuTopologyLeafBConstants(unsigned & PkgSelectMask, unsigned & PkgSelectMaskShift, unsigned & CoreSelectMask, unsigned & SMTSelectMask,
                              unsigned & SMTMaskWidth)
{
    int error = 0;
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
            SMTSelectMask     = ~((std::numeric_limits<unsigned>::max()) << levelShift);
            SMTMaskWidth      = levelShift;
            wasThreadReported = true;
            break;
        case 2:
            // level type is Core, so levelShift is the CorePlsuSMT_Mask_Width
            coreplusSMT_Mask   = ~((std::numeric_limits<unsigned long>::max()) << levelShift);
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
        error |= _MSGTYP_THREAD_REPORTING_FAILED;
    }

    if (error) return error;

    return 0;
}

// Derive bitmask extraction parameter using CPUID leaf 1 and leaf 4
//
// \param[in] maxCPUIDLeaf          Maximum supported CPUID leaf index
// \param[in] info                  CPUID leaf 1 data structure
// \param[out] PkgSelectMask        Mask to extract Pkg_ID field from APIC ID.
//                                  Pkg_ID is an index of the physical package in the system,
//                                  i.e. NUMA node index
// \param[out] PkgSelectMaskShift   Shift count to right shift APIC ID before applying PkgSelectMask
// \param[out] CoreSelectMask       Mask to extract Core_ID field from APIC ID.
//                                  Core_ID is an index of the core in a physical package
// \param[out] SMTSelectMask        Mask to extract SMT_ID (simultaneous multithreading)
//                                  field from APIC ID.
//                                  SMT_ID is an index of the logical processor in a core
// \param[out] SMTMaskWidth         Width of the SMT_ID field in bits
//
// \return 0 if successful, non-zero if error occurred
int cpuTopologyLegacyConstants(unsigned maxCPUIDLeaf, const CPUIDinfo & info, unsigned & PkgSelectMask, unsigned & PkgSelectMaskShift,
                               unsigned & CoreSelectMask, unsigned & SMTSelectMask, unsigned & SMTMaskWidth)
{
    unsigned corePlusSMTIDMaxCnt;
    unsigned coreIDMaxCnt       = 1;
    unsigned SMTIDPerCoreMaxCnt = 1;

    corePlusSMTIDMaxCnt = __internal_daal_getBitsFromDWORD(info.EBX, 16, 23);
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

// Derive parameters used to extract/decompose APIC ID for CPU topology.
//
// \param[in] maxCPUIDLeaf          Maximum supported CPUID leaf index
// \param[out] hasLeafB             Flag to indicate whether CPUID leaf 0BH is supported
// \param[out] PkgSelectMask        Mask to extract Pkg_ID field from APIC ID.
//                                  Pkg_ID is an index of the physical package in the system,
//                                  i.e. NUMA node index
// \param[out] PkgSelectMaskShift   Shift count to right shift APIC ID before applying PkgSelectMask
// \param[out] CoreSelectMask       Mask to extract Core_ID field from APIC ID.
//                                  Core_ID is an index of the core in a physical package
// \param[out] SMTSelectMask        Mask to extract SMT_ID (simultaneous multithreading)
//                                  field from APIC ID.
//                                  SMT_ID is an index of the logical processor in a core
// \param[out] SMTMaskWidth         Width of the SMT_ID field in bits
//
// \return 0 if successful, non-zero if error occurred
int cpuTopologyParams(unsigned maxCPUIDLeaf, bool & hasLeafB, unsigned & PkgSelectMask, unsigned & PkgSelectMaskShift, unsigned & CoreSelectMask,
                      unsigned & SMTSelectMask, unsigned & SMTMaskWidth)
{
    int error = 0;
    CPUIDinfo info; // data structure to store register data reported by CPUID

    // cpuid leaf B detection
    if (maxCPUIDLeaf >= 0xB)
    {
        CPUIDinfo CPUInfoB;
        CPUInfoB.get(0xB, 0);
        // glbl_ptr points to assortment of global data, workspace, etc
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
            //     use CPUID leaf B to derive extraction parameters
            error = cpuTopologyLeafBConstants(PkgSelectMask, PkgSelectMaskShift, CoreSelectMask, SMTSelectMask, SMTMaskWidth);
        }
        else
        {
            // #2, Processors that support legacy parameters
            //     using CPUID leaf 1 and leaf 4
            error = cpuTopologyLegacyConstants(maxCPUIDLeaf, info, PkgSelectMask, PkgSelectMaskShift, CoreSelectMask, SMTSelectMask, SMTMaskWidth);
        }
    }
    else
    {
        // #3, Prior to HT, there is only one logical processor in a physical package
        CoreSelectMask     = 0;
        SMTMaskWidth       = 0;
        PkgSelectMask      = (unsigned)(-1);
        PkgSelectMaskShift = 0;
        SMTSelectMask      = 0;
    }

    if (error) return error;

    return 0;
}

Dyn2Arr_str::Dyn2Arr_str(const unsigned xdim, const unsigned ydim, const unsigned value)
{
    dim0 = xdim;
    dim1 = ydim;
    data = (unsigned *)_INTERNAL_DAAL_MALLOC(xdim * ydim * sizeof(unsigned));
    if (!data)
    {
        return;
    }
    _INTERNAL_DAAL_MEMSET(data, value, xdim * ydim * sizeof(unsigned));
}

Dyn2Arr_str::Dyn2Arr_str(const Dyn2Arr_str & other)
{
    if (this == &other) return; // self-assignment check
    dim0            = other.dim0;
    dim1            = other.dim1;
    size_t dataSize = dim0 * dim1 * sizeof(unsigned);
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
    dim0 = 0;
    dim1 = 0;
}

void Dyn2Arr_str::fill(const unsigned value)
{
    _INTERNAL_DAAL_MEMSET(data, value, dim0 * dim1 * sizeof(unsigned));
}

Dyn1Arr_str::Dyn1Arr_str(const unsigned xdim, const unsigned value)
{
    dim0 = xdim;
    data = (unsigned *)_INTERNAL_DAAL_MALLOC(xdim * sizeof(unsigned));
    if (!data)
    {
        return;
    }
    fill(value);
}

Dyn1Arr_str::Dyn1Arr_str(const Dyn1Arr_str & other)
{
    if (this == &other) return; // self-assignment check
    dim0            = other.dim0;
    size_t dataSize = dim0 * sizeof(unsigned);
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
    dim0 = 0;
}

void Dyn1Arr_str::fill(const unsigned value)
{
    _INTERNAL_DAAL_MEMSET(data, value, dim0 * sizeof(unsigned));
}

/*
 * Allocate the dynamic arrays in the global object containing various buffers for analyzing
 * cpu topology of a system with N logical processors in the process affinity mask.
 *
 * \param cpus  Number of logical processors
 *
 * \return 0 is no error, error code otherwise
 */
int glktsn::allocArrays(const unsigned cpus)
{
    pApicAffOrdMapping = (idAffMskOrdMapping_t *)_INTERNAL_DAAL_MALLOC(cpus * sizeof(idAffMskOrdMapping_t));
    if (!pApicAffOrdMapping)
    {
        error = _MSGTYP_MEMORY_ALLOCATION_FAILED;
        return error;
    }
    _INTERNAL_DAAL_MEMSET(pApicAffOrdMapping, 0, cpus * sizeof(idAffMskOrdMapping_t));

    perPkg_detectedCoresCount    = Dyn1Arr_str(cpus);
    perCore_detectedThreadsCount = Dyn2Arr_str(cpus, MAX_CORES);
    if (perPkg_detectedCoresCount.isEmpty() || perCore_detectedThreadsCount.isEmpty())
    {
        error = _MSGTYP_MEMORY_ALLOCATION_FAILED;
        return error;
    }
    perPkg_detectedCoresCount.fill(0);
    perCore_detectedThreadsCount.fill(0);

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
idAffMskOrdMapping_t::idAffMskOrdMapping_t(unsigned int cpu, bool hasLeafB, unsigned PkgSelectMask, unsigned PkgSelectMaskShift,
                                           unsigned CoreSelectMask, unsigned SMTSelectMask, unsigned SMTMaskWidth)
{
    initApicID(hasLeafB);

    OrdIndexOAMsk = cpu; // this an ordinal number that can relate to generic affinity mask
    pkg_IDAPIC    = ((APICID & PkgSelectMask) >> PkgSelectMaskShift);
    Core_IDAPIC   = ((APICID & CoreSelectMask) >> SMTMaskWidth);
    SMT_IDAPIC    = (APICID & SMTSelectMask);
}

/*
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
int glktsn::parseAPICIDs(const GenericAffinityMask & processAffinity, unsigned & maxPkgSelectMaskShift)
{
    unsigned threadIndex  = 0;
    maxPkgSelectMaskShift = 0;
    for (unsigned i = 0; i < processAffinity.cpuCount; i++)
    {
        // can't asume OS affinity bit mask is contiguous,
        // but we are using our generic bitmap representation for affinity
        unsigned char processAffinityBit = processAffinity.test(i);
        if (processAffinityBit == 1)
        {
            // bind the execution context to the i-th logical processor
            // using OS-specific API
            ScopedThreadContext ctx(i);
            if (ctx.error)
            {
                // Failed to set thread affinity
                error |= ctx.error;
                break;
            }

            // the following variables are parameters related to
            // extracting sub IDs from an APIC ID, on heterogeneous systems can differ across cores
            unsigned CoreSelectMask;
            unsigned PkgSelectMask;
            unsigned PkgSelectMaskShift;
            unsigned SMTSelectMask;
            unsigned SMTMaskWidth;

            error = cpuTopologyParams(maxCPUIDLeaf, hasLeafB, PkgSelectMask, PkgSelectMaskShift, CoreSelectMask, SMTSelectMask, SMTMaskWidth);
            if (error)
            {
                // Failed to derive CPU topology parameters
                break;
            }

            if (PkgSelectMaskShift > maxPkgSelectMaskShift)
            {
                maxPkgSelectMaskShift = PkgSelectMaskShift;
            }

            if (threadIndex >= EnumeratedThreadCount)
            {
                // should never happen
                error |= _MSGTYP_INVALID_THREAD_INDEX;
                break;
            }
            pApicAffOrdMapping[threadIndex++] =
                idAffMskOrdMapping_t(i, hasLeafB, PkgSelectMask, PkgSelectMaskShift, CoreSelectMask, SMTSelectMask, SMTMaskWidth);
            ctx.restoreContext();
            if (ctx.error)
            {
                // Failed to set thread affinity
                error |= ctx.error;
                break;
            }
        }
        else if (processAffinityBit == 0xff)
        {
            // should never happen
            // i-th bit is out of bounds of the process affinity mask
            error |= _MSGTYP_CANNOT_TEST_AFFINITY_BIT;
            break;
        }
    }

    return error;
}

/*
 * Analyze the Pkg_ID, Core_ID to derive hierarchical ordinal numbering scheme
 *
 * \return 0 is no error
 */
int glktsn::analyzeCPUHierarchy(unsigned PkgSelectMaskShift)
{
    // allocate workspace to sort parents and siblings in the topology
    // starting from pkg_ID and work our ways down each inner level
    Dyn1Arr_str pDetectedPkgIDs(EnumeratedThreadCount, 0xff);

    // we got a 1-D array to store unique Pkg_ID as we sort thru
    // each logical processor
    if (pDetectedPkgIDs.isEmpty()) return _MSGTYP_MEMORY_ALLOCATION_FAILED;

    // we got a 2-D array to store unique Core_ID within each Pkg_ID,
    // as we sort thru each logical processor
    Dyn2Arr_str pDetectCoreIDsperPkg(EnumeratedThreadCount, (1 << PkgSelectMaskShift), 0xff);
    if (pDetectCoreIDsperPkg.isEmpty()) return _MSGTYP_MEMORY_ALLOCATION_FAILED;

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

        if (maxPackageDetetcted > EnumeratedThreadCount)
        {
            // should never happen
            return _MSGTYP_INVALID_PACKAGE_INDEX;
        }
        for (unsigned h = 0; h < maxPackageDetetcted; h++)
        {
            if (pDetectedPkgIDs[h] == packageID)
            {
                bool foundCore = false;
                PkgMarked      = true;

                // look for core in marked packages
                for (unsigned k = 0; k < perPkg_detectedCoresCount[h]; k++)
                {
                    if (h * EnumeratedThreadCount + k >= pDetectCoreIDsperPkg.size())
                    {
                        // should never happen
                        return _MSGTYP_INVALID_CORE_INDEX;
                    }
                    if (coreID == pDetectCoreIDsperPkg[h * EnumeratedThreadCount + k])
                    {
                        foundCore = true;
                        // add thread - can't be that the thread already exists, breaks uniqe APICID spec
                        if (h * MAX_CORES + k >= perCore_detectedThreadsCount.size())
                        {
                            // should never happen
                            return _MSGTYP_INVALID_THREAD_COUNT_INDEX;
                        }
                        perCore_detectedThreadsCount[h * MAX_CORES + k]++;
                        break;
                    }
                }

                if (!foundCore)
                {
                    // mark up the Core_ID of an unmarked core in a marked package
                    unsigned core = perPkg_detectedCoresCount[h];

                    if (h * EnumeratedThreadCount + core >= pDetectCoreIDsperPkg.size())
                    {
                        // should never happen
                        return _MSGTYP_INVALID_CORE_INDEX;
                    }
                    pDetectCoreIDsperPkg[h * EnumeratedThreadCount + core] = coreID;

                    // keep track of respective hierarchical counts
                    if (h * MAX_CORES + core >= perCore_detectedThreadsCount.size())
                    {
                        // should never happen
                        return _MSGTYP_INVALID_THREAD_COUNT_INDEX;
                    }
                    perCore_detectedThreadsCount[h * MAX_CORES + core] = 1;
                    perPkg_detectedCoresCount[h]++;

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

            maxPackageDetetcted++; // this is an unmarked pkg, increment pkg count by 1
            EnumeratedCoreCount++; // there is at least one core in a package
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
void glktsn::buildSystemTopologyTables(const GenericAffinityMask & processAffinity)
{
    // allocated the memory buffers within the global pointer

    // For each logical processor, collect APIC ID and parse sub IDs for each APIC ID
    unsigned maxPkgSelectMaskShift = 0;
    error                          = parseAPICIDs(processAffinity, maxPkgSelectMaskShift);
    if (error) return;
    // Derived separate numbering schemes for each level of the cpu topology
    if (analyzeCPUHierarchy(maxPkgSelectMaskShift) < 0)
    {
        error |= _MSGTYP_TOPOLOGY_NOT_ANALYZED;
        return;
    }

    isInit = true;
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

unsigned _internal_daal_GetLogicalProcessorQueue(int * queue)
{
    // this code runs only when the thread pinning is enabled, which can happen only via DAAL API.
    const int cpus = _internal_daal_GetSysLogicalProcessorCount();

    // pinning j-th logical thread to j-th logical processor in the system.
    for (unsigned j = 0; j < cpus; j++)
    {
        queue[j] = j;
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

unsigned _internal_daal_GetStatus()
{
    return 0;
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
