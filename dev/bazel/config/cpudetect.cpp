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

#if defined(__x86_64__) || defined(__x86_64) || defined(__amd64) || defined(_M_AMD64)
    #define TARGET_X86_64
#endif

#if defined(__ARM_ARCH) || defined(__aarch64__)
    #define TARGET_ARM
#endif

#if defined(__riscv) && (__riscv_xlen == 64)
    #define TARGET_RISCV64
#endif

#if defined(TARGET_X86_64)
#include <immintrin.h>
#endif

#if defined(_MSC_VER)
#if (_MSC_FULL_VER >= 160040219)
#include <intrin.h>
#else
#error "min VS2010 SP1 compiler is required"
#endif
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <cstdint>
#include <iostream>
#include <string>

using std::uint32_t;

void try_enable_avx512f_on_macos() {
#if defined(__APPLE__)
    int answer = 0;
    size_t answer_size = sizeof(answer);
    ::sysctlbyname("hw.optional.avx512f", &answer, &answer_size, NULL, 0);
    if (answer)
    {
        asm("kandw %k1, %k2, %k3\t");
    }
#endif
}

void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t* abcd) {
#if defined(_MSC_VER)
    __cpuidex((int *)abcd, eax, ecx);
#else
    uint32_t ebx = 0, edx = 0;
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

int check_cpuid(uint32_t eax, uint32_t ecx, int abcd_index, uint32_t mask) {
    uint32_t abcd[4];
    run_cpuid(eax, ecx, abcd);
    return ((abcd[abcd_index] & mask) == mask);
}

int check_xgetbv_xcr0_ymm(uint32_t mask) {
    uint32_t xcr0;
#if defined(_MSC_VER)
    xcr0 = (uint32_t)_xgetbv(0);
#else
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
    return ((xcr0 & mask) == mask); /* checking if xmm and ymm state are enabled in XCR0 */
}

int check_avx512_features() {
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

    if (!check_cpuid(1, 0, 2, avx_osxsave_mask)) {
        return 0;
    }
    if (!check_xgetbv_xcr0_ymm(kmask_ymm_mask)) {
        return 0;
    }
    if (!check_cpuid(7, 0, 1, avx512_mask)) {
        return 0;
    }
    return 1;
}

int check_avx2_features() {
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

    if (!check_cpuid(1, 0, 2, fma_aes_osxsave_mask)) {
        return 0;
    }
    if (!check_xgetbv_xcr0_ymm(6)) {
        return 0;
    }
    if (!check_cpuid(7, 0, 1, avx2_bmi12_mask)) {
        return 0;
    }
    if (!check_cpuid(0x80000001, 0, 2, lzcnt_mask)) {
        return 0;
    }
    return 1;
}

int check_sse42_features() {
    /* CPUID.(EAX=01H, ECX=0H):ECX.SSE4.2[bit 20]==1 */
    uint32_t sse42_mask = 0x100000;

    if (!check_cpuid(1, 0, 2, sse42_mask)) {
        return 0;
    }
    return 1;
}

std::string detect_cpu() {

    #if defined(TARGET_X86_64)
        try_enable_avx512f_on_macos();

        if (check_avx512_features()) {
            return "avx512";
        }
        else if (check_avx2_features()) {
            return "avx2";
        }
        else if (check_sse42_features()) {
            return "sse42";
        }
        else {
            return "sse2";
        }
    #elif defined(TARGET_ARM)
        return "sve";
    #elif defined(TARGET_RISCV64)
        return "rv64";
    #endif
}

int main(int argc, char const *argv[]) {
    std::cout << detect_cpu() << std::endl;
    return 0;
}
