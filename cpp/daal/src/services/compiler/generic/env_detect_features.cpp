/* file: env_detect_features.cpp */
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
//  CPU features detection.
//--
*/

#include "services/env_detect.h"
#include "services/daal_defines.h"

#if defined(TARGET_X86_64)
    #include <immintrin.h>
#elif defined(TARGET_ARM)
    #include <sys/auxv.h>
    #include <asm/hwcap.h>
#elif defined(TARGET_RISCV64)
// TODO: Include vector if and when we need to use some vector intrinsics in
// here
#endif

#include "src/services/service_defines.h"
#include "src/threading/threading.h"

#include <stdint.h>
#if defined(_MSC_VER)
    #if (_MSC_FULL_VER >= 160040219)
        #include <intrin.h>
    #else
        #error "min VS2010 SP1 compiler is required"
    #endif
#endif

#if defined(__APPLE__)
void __daal_serv_CPUHasAVX512f_enable_it_mac();
#endif

#if defined(TARGET_X86_64)
void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t * abcd)
{
    #if defined(_MSC_VER)
    __cpuidex((int *)abcd, eax, ecx);
    #else
    uint32_t ebx, edx;
        #if defined(__i386__) && defined(__PIC__)
    /* in case of PIC under 32-bit EBX cannot be clobbered */
    __asm__("movl %%ebx, %%edi \n\t cpuid \n\t xchgl %%ebx, %%edi" : "=D"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
        #else
    __asm__("cpuid" : "+b"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
        #endif
    abcd[0] = eax;
    abcd[1] = ebx;
    abcd[2] = ecx;
    abcd[3] = edx;
    #endif
}

uint32_t __daal_internal_get_max_extension_support()
{
    // Running cpuid with a value other than eax=0 and 0x8000000 is an extension
    // To check that a particular eax value is supported we need to check
    // maximum extension that is supported by checking the value returned by
    // cpuid when eax=0x80000000 is given.
    uint32_t abcd[4];
    run_cpuid(0x80000000, 0, abcd);
    return abcd[0];
}

uint32_t daal_get_max_extension_support()
{
    // We cache the result in a static variable here.
    static const uint32_t result = __daal_internal_get_max_extension_support();
    return result;
}

bool __daal_internal_is_intel_cpu()
{
    const uint32_t genu = 0x756e6547, inei = 0x49656e69, ntel = 0x6c65746e;
    uint32_t abcd[4];
    run_cpuid(0, 0, abcd);
    return abcd[1] == genu && abcd[2] == ntel && abcd[3] == inei;
}

DAAL_EXPORT bool daal_check_is_intel_cpu()
{
    static const bool result = __daal_internal_is_intel_cpu();
    return result;
}

static int check_cpuid(uint32_t eax, uint32_t ecx, int abcd_index, uint32_t mask)
{
    if (daal_get_max_extension_support() < eax)
    {
        // need to check that the eax we run here is supported.
        return 0;
    }
    uint32_t abcd[4];

    run_cpuid(eax, ecx, abcd);

    return ((abcd[abcd_index] & mask) == mask);
}

static int check_xgetbv_xcr0_ymm(uint32_t mask)
{
    uint32_t xcr0;
    #if defined(_MSC_VER)
    xcr0 = (uint32_t)_xgetbv(0);
    #else
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
    #endif
    return ((xcr0 & mask) == mask); /* checking if xmm and ymm state are enabled in XCR0 */
}

static int check_avx512_features()
{
    /*
    CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1 &&
    CPUID.(EAX=01H, ECX=0H):ECX.AVX    [bit 28]==1
    */
    uint32_t avx_osxsave_mask = ((1 << 27) | (1 << 28));

    /*
    CPUID.(EAX=07H, ECX=0H):EBX.AVX512F [bit 16]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.AVX512DQ[bit 17]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.AVX512BW[bit 30]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.AVX512VL[bit 31]==1
    */
    uint32_t avx512_mask = (1 << 16) | (1 << 17) | (1 << 30) | (1 << 31);

    /*
    E0H - KMASK state, upper 256-bit of ZMM0-ZMM15 and ZMM16-ZMM31 state are enabled by OS
    06H - XMM state and YMM state are enabled by OS
    */
    uint32_t kmask_ymm_mask = 0xE6;

    if (!check_cpuid(1, 0, 2, avx_osxsave_mask))
    {
        return 0;
    }
    if (!check_xgetbv_xcr0_ymm(kmask_ymm_mask))
    {
        return 0;
    }
    if (!check_cpuid(7, 0, 1, avx512_mask))
    {
        return 0;
    }

    return 1;
}

static int check_avx2_features()
{
    /* CPUID.(EAX=01H, ECX=0H):ECX.FMA[bit 12]==1     &&
       CPUID.(EAX=01H, ECX=0H):ECX.AES[bit 25]==1     &&
       CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1 &&
       CPUID.(EAX=01H, ECX=0H):ECX.AVX[bit 28]==1 */
    uint32_t fma_aes_osxsave_mask = ((1 << 12) | (1 << 25) | (1 << 27) | (1 << 28));

    /*  CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1  &&
        CPUID.(EAX=07H, ECX=0H):EBX.BMI1[bit 3]==1  &&
        CPUID.(EAX=07H, ECX=0H):EBX.BMI2[bit 8]==1  */
    uint32_t avx2_bmi12_mask = (1 << 5) | (1 << 3) | (1 << 8);
    /* CPUID.(EAX=80000001H):ECX.LZCNT[bit 5]==1 */
    uint32_t lzcnt_mask = (1 << 5);

    if (!check_cpuid(1, 0, 2, fma_aes_osxsave_mask))
    {
        return 0;
    }
    if (!check_xgetbv_xcr0_ymm(6))
    {
        return 0;
    }
    if (!check_cpuid(7, 0, 1, avx2_bmi12_mask))
    {
        return 0;
    }
    if (!check_cpuid(0x80000001, 0, 2, lzcnt_mask))
    {
        return 0;
    }

    return 1;
}

static int check_sse42_features()
{
    /* CPUID.(EAX=01H, ECX=0H):ECX.SSE4.2[bit 20]==1 */
    uint32_t sse42_mask = 0x100000;

    if (!check_cpuid(1, 0, 2, sse42_mask))
    {
        return 0;
    }

    return 1;
}

DAAL_EXPORT int __daal_serv_cpu_detect(int enable)
{
    #if defined(__APPLE__)
    __daal_serv_CPUHasAVX512f_enable_it_mac();
    #endif
    if (check_avx512_features() && daal_check_is_intel_cpu())
    {
        return daal::avx512;
    }

    if (check_avx2_features())
    {
        return daal::avx2;
    }

    if (check_sse42_features())
    {
        return daal::sse42;
    }

    return daal::sse2;
}
#elif defined(TARGET_ARM)
static bool check_sve_features()
{
    unsigned long hwcap = getauxval(AT_HWCAP);

    return (hwcap & HWCAP_SVE) != 0;
}

DAAL_EXPORT int __daal_serv_cpu_detect(int enable)
{
    if (check_sve_features())
    {
        return daal::sve;
    }
    return -1;
}

void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t * abcd)
{
    // TODO: ARM implementation for cpuid
}

bool daal_check_is_intel_cpu()
{
    return false;
}
#elif defined(TARGET_RISCV64)
DAAL_EXPORT int __daal_serv_cpu_detect(int enable)
{
    return daal::rv64;
}

void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t * abcd)
{
    // TODO: riscv64 implementation for cpuid
}

bool daal_check_is_intel_cpu()
{
    return false;
}
#endif
