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

    // `libonedal_parameters` exists only when oneDAL is built with
    // `BUILD_PARAMETERS_LIB=yes`; the folded layout links the parameter objects
    // into `libonedal` itself. `makefile` passes `-DPARAMETERS_LIB` to match.
    #ifdef PARAMETERS_LIB
        #define ONEDAL_LIBS PATH(onedal) PATH(onedal_core) PATH(onedal_thread) PATH(onedal_parameters)
    #else
        #define ONEDAL_LIBS PATH(onedal) PATH(onedal_core) PATH(onedal_thread)
    #endif

    // The static libraries are built against the ILP64 oneMKL interface, the
    // dynamic ones against LP64 (see `dev/make/deps.mkl.mk`).
    //
    // A static consumer needs the oneMKL *archives*, named with `-l:` and
    // resolved inside a link group:
    //
    //   * `-lmkl_core` finds `libmkl_core.so`, whose internals live in the
    //     kernel libraries oneMKL dlopen's at runtime (`libmkl_def.so`,
    //     `libmkl_avx*.so`), so a static link against the shared libraries
    //     leaves ~9600 undefined references. oneDAL itself links the archives
    //     (`daaldep.$(PLAT).mkl.*` in `dev/make/deps.mkl.mk`) and localizes
    //     them with `--exclude-libs`, so a static consumer must supply them.
    //   * naming the archives without a group still leaves ~400 undefined
    //     references, and repeating them once leaves ~30: oneMKL's archives
    //     are mutually recursive and need `--start-group`.
    //   * the oneDAL archives are inside the group too, which makes the link
    //     independent of their order, matching what `oneDALConfig.cmake`
    //     already asks CMake consumers to do.
    #ifdef STATIC
        #define SUFFIX a
        #define MATH_LIBS -l:libmkl_intel_ilp64.a -l:libmkl_tbb_thread.a -l:libmkl_core.a
        #define OTHER_LIBS -ltbb -ltbbmalloc -lpthread -ldl
        #define LIBS_LINE -Wl,--start-group ONEDAL_LIBS MATH_LIBS -Wl,--end-group OTHER_LIBS
    #else
        #define SUFFIX so
        #define OTHER_LIBS -lmkl_core -lmkl_intel_lp64 -lmkl_tbb_thread -ltbb -ltbbmalloc -lpthread -ldl
        #define LIBS_LINE ONEDAL_LIBS OTHER_LIBS
    #endif

    #define PATH(inp) ${libdir}/lib##inp.SUFFIX

    #define OPTS -std=c++17 -Wno-deprecated-declarations

#elif defined(__APPLE__) && defined(__MACH__)
    #define LIBDIR lib

    #define OTHER_LIBS -lmkl_core -lmkl_intel_lp64 -lmkl_tbb_thread -ltbb -ltbbmalloc -ldl
    #ifdef PARAMETERS_LIB
        #define ONEDAL_LIBS PATH(onedal) PATH(onedal_core) PATH(onedal_thread) PATH(onedal_parameters)
    #else
        #define ONEDAL_LIBS PATH(onedal) PATH(onedal_core) PATH(onedal_thread)
    #endif

    #ifdef STATIC
        #define SUFFIX a
    #else
        #define SUFFIX dylib
    #endif

    #define PATH(inp) ${libdir}/lib##inp.SUFFIX

    // Apple's linker has no `--start-group` and no `-l:`, so the static
    // package's oneMKL naming is left as it is here; see the `__linux__`
    // branch for why that shape is not linkable.
    #define LIBS_LINE ONEDAL_LIBS OTHER_LIBS

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

    #define LIBS_LINE ONEDAL_LIBS OTHER_LIBS

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
Version: 2026.3
URL: ONEDAL_URL
Libs: LIBS_LINE
Cflags: OPTS -I${includedir}
