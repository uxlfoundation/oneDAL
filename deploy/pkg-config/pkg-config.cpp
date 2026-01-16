//===============================================================================
// Copyright contributors to the oneDAL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===============================================================================

#define ONEDAL_URL "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal.html"

#if defined(__linux__) || defined(__linux)

    #if defined(__x86_64__) || defined(__x86_64) || defined(__amd64) || defined(_M_AMD64)
        #define LIBDIR lib/intel64
    #elif defined(__ARM_ARCH) || defined(__aarch64__)
        #define LIBDIR lib/arm
    #elif defined(__riscv) && (__riscv_xlen == 64)
        #define LIBDIR lib/riscv64
    #else
        #error Unknown CPU architecture
    #endif

    #define OTHER_LIBS -lmkl_core -lmkl_intel_lp64 -lmkl_tbb_thread -ltbb -ltbbmalloc -lpthread -ldl
    #define ONEDAL_LIBS PATH(onedal) PATH(onedal_core) PATH(onedal_thread) PATH(onedal_parameters)

    #ifdef STATIC
        #define SUFFIX a
    #else
        #define SUFFIX so
    #endif

    #define PATH(inp) ${libdir}/lib##inp.SUFFIX

    #define OPTS -std=c++17 -Wno-deprecated-declarations

#elif defined(__APPLE__) && defined(__MACH__)
    #define LIBDIR lib

    #define OTHER_LIBS -lmkl_core -lmkl_intel_lp64 -lmkl_tbb_thread -ltbb -ltbbmalloc -ldl
    #define ONEDAL_LIBS PATH(onedal) PATH(onedal_core) PATH(onedal_thread) PATH(onedal_parameters)

    #ifdef STATIC
        #define SUFFIX a
    #else
        #define SUFFIX dylib
    #endif

    #define PATH(inp) ${libdir}/lib##inp.SUFFIX

    #define OPTS -std=c++17 -Wno-deprecated-declarations -diag-disable=10441

#elif defined(_WIN32) || defined(_WIN64)
    #define LIBDIR lib/intel64

    #define OTHER_LIBS mkl_core.lib mkl_intel_lp64.lib mkl_tbb_thread.lib tbb12.lib tbbmalloc.lib

    #ifdef STATIC
        #define ONEDAL_LIBS PATH(onedal) PATH(onedal_core) PATH(onedal_thread)
        #define PATH(inp) ${libdir}/inp.lib
    #else
        #define ONEDAL_LIBS PATH(onedal) PATH(onedal_core)
        #define PATH(inp) ${libdir}/inp##_dll.lib
    #endif

    #define OPTS /std:c++17 /MD /wd4996 /EHsc

#else
    #error Not a supported OS
#endif

prefix=${pcfiledir}/../../
exec_prefix=${prefix}
libdir=${exec_prefix}/LIBDIR
includedir=${prefix}/include

Name: oneDAL
Description: oneAPI Data Analytics Library
Version: 2026.0
URL: ONEDAL_URL
Libs: ONEDAL_LIBS OTHER_LIBS
Cflags: OPTS -I${includedir}
