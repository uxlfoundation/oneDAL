/* file: service_defines.h */
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
//  Declaration of service constants
//--
*/

#ifndef __SERVICE_DEFINES_H__
#define __SERVICE_DEFINES_H__

#include <stdint.h>
#include "services/env_detect.h"

DAAL_EXPORT bool __daal_serv_cpu_extensions_available();
DAAL_EXPORT int __daal_serv_cpu_detect(int);

void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t * abcd);
DAAL_EXPORT bool daal_check_is_intel_cpu();

#if defined(TARGET_X86_64)
    #define DAAL_BASE_CPU daal::sse2
#elif defined(TARGET_ARM)
    #define DAAL_BASE_CPU daal::sve
#elif defined(TARGET_RISCV64)
    #define DAAL_BASE_CPU daal::rv64
#endif

#define DAAL_CHECK_CPU_ENVIRONMENT (__daal_serv_cpu_extensions_available())

#if defined(__INTEL_COMPILER)
    #define PRAGMA_IVDEP            _Pragma("ivdep")
    #define PRAGMA_NOVECTOR         _Pragma("novector")
    #define PRAGMA_VECTOR_ALIGNED   _Pragma("vector aligned")
    #define PRAGMA_VECTOR_UNALIGNED _Pragma("vector unaligned")
    #define PRAGMA_VECTOR_ALWAYS    _Pragma("vector always")
    #define PRAGMA_ICC_TO_STR(ARGS) _Pragma(#ARGS)
    #define PRAGMA_ICC_OMP(ARGS)    PRAGMA_ICC_TO_STR(omp ARGS)
    #define PRAGMA_ICC_NO16(ARGS)   PRAGMA_ICC_TO_STR(ARGS)
    #define DAAL_TYPENAME           typename
#elif defined(__GNUC__)
    #define PRAGMA_IVDEP
    #define PRAGMA_NOVECTOR
    #define PRAGMA_VECTOR_ALIGNED
    #define PRAGMA_VECTOR_UNALIGNED
    #define PRAGMA_VECTOR_ALWAYS
    #define PRAGMA_ICC_TO_STR(ARGS)
    #define PRAGMA_ICC_OMP(ARGS)
    #define PRAGMA_ICC_NO16(ARGS)
    #define DAAL_TYPENAME typename
#elif defined(_MSC_VER)
    #define PRAGMA_IVDEP
    #define PRAGMA_NOVECTOR
    #define PRAGMA_VECTOR_ALIGNED
    #define PRAGMA_VECTOR_UNALIGNED
    #define PRAGMA_VECTOR_ALWAYS
    #define PRAGMA_ICC_TO_STR(ARGS)
    #define PRAGMA_ICC_OMP(ARGS)
    #define PRAGMA_ICC_NO16(ARGS)
    #define DAAL_TYPENAME typename
#else
    #define PRAGMA_IVDEP
    #define PRAGMA_NOVECTOR
    #define PRAGMA_VECTOR_ALIGNED
    #define PRAGMA_VECTOR_UNALIGNED
    #define PRAGMA_VECTOR_ALWAYS
    #define PRAGMA_ICC_OMP(ARGS)
    #define PRAGMA_ICC_NO16(ARGS)
    #define DAAL_TYPENAME typename
#endif

#if defined __APPLE__ && defined __INTEL_COMPILER && (__INTEL_COMPILER == 1600)
    #undef PRAGMA_ICC_TO_STR
    #define PRAGMA_ICC_TO_STR(ARGS) _Pragma(#ARGS)
    #undef PRAGMA_ICC_OMP
    #define PRAGMA_ICC_OMP(ARGS) PRAGMA_ICC_TO_STR(ARGS)
    #undef PRAGMA_ICC_NO16
    #define PRAGMA_ICC_NO16(ARGS)
#endif

#ifdef DEBUG_ASSERT
    #include <assert.h>
    #define DAAL_ASSERT(cond) assert(cond);
#else
    #define DAAL_ASSERT(cond)
#endif

namespace daal
{
/* Execution if(!this->_errors->isEmpty()) */
enum ServiceStatus
{
    SERV_ERR_OK = 0,
    SERV_ERR_MKL_SVD_ITH_PARAM_ILLEGAL_VALUE,
    SERV_ERR_MKL_SVD_XBDSQR_DID_NOT_CONVERGE,
    SERV_ERR_MKL_QR_ITH_PARAM_ILLEGAL_VALUE,
    SERV_ERR_MKL_QR_XBDSQR_DID_NOT_CONVERGE,
    SERV_ERR_MALLOC
};

/** Data storage format */
enum DataFormat
{
    dense = 0, /*!< Dense storage format */
    CSR   = 1  /*!< Compressed Sparse Rows (CSR) storage format */
};

} // namespace daal

/* CPU comparison macro */
#if defined(TARGET_X86_64)
    #define __sse2__   (0)
    #define __sse42__  (2)
    #define __avx2__   (4)
    #define __avx512__ (6)
#elif defined(TARGET_ARM)
    #define __sve__ (0)
#elif defined(TARGET_RISCV64)
    #define __rv64__ (0)
#endif

#define __float__  (0)
#define __double__ (1)

#if defined(TARGET_X86_64)
    #define CPU_sse2   __sse2__
    #define CPU_sse42  __sse42__
    #define CPU_avx2   __avx2__
    #define CPU_avx512 __avx512__
#elif defined(TARGET_ARM)
    #define CPU_sve __sve__
#elif defined(TARGET_RISCV64)
    #define CPU_rv64 __rv64__
#endif

#define FPTYPE_float  __float__
#define FPTYPE_double __double__

#define __GLUE__(a, b)   a##b
#define __CPUID__(cpu)   __GLUE__(CPU_, cpu)
#define __FPTYPE__(type) __GLUE__(FPTYPE_, type)

/*
//  Set of macro definitions
//  for FP values bit fields easy access
*/

#define DAAL_LITTLE_ENDIAN

#if defined DAAL_BIG_ENDIAN
    #undef DAAL_LITTLE_ENDIAN
#endif

/*
//  Bit fields in IEEE-754 representation of single
//      sign:1 exponent:8 significand:23 (implied leading 1)
//  and double precision number
//      sign:1 exponent:11 significand:52 (implied leading 1)
*/
#if defined(DAAL_LITTLE_ENDIAN)

    #if !defined(DAAL_NO_BITFIELDS)
/* Little endian float */
typedef struct tag_daal_spbits_t
{
    uint32_t significand : 23;
    uint32_t exponent : 8;
    uint32_t sign : 1;
} _daal_spbits_t;

/* Little endian double */
typedef struct tag_daal_dpbits_t
{
    uint32_t lo_significand : 32;
    uint32_t hi_significand : 20;
    uint32_t exponent : 11;
    uint32_t sign : 1;
} _daal_dpbits_t;

typedef _daal_spbits_t daal_fp32;
typedef _daal_dpbits_t daal_fp64;

    #endif //!defined (DAAL_NO_BITFIELDS)

/* Little endian double */
typedef struct tag_daal_dpdwords_t
{
    uint32_t lo_dword;
    uint32_t hi_dword;
} _daal_dpdwords_t;

#else /* !defined (DAAL_LITTLE_ENDIAN) */

    #if !defined(DAAL_NO_BITFIELDS)
/* Big endian float */
typedef struct tag_daal_spbits_t
{
    uint32_t sign : 1;
    uint32_t exponent : 8;
    uint32_t significand : 23;
} _daal_spbits_t;

/* Big endian double */
typedef struct tag_daal_dpbits_t
{
    uint32_t sign : 1;
    uint32_t exponent : 11;
    uint32_t hi_significand : 20;
    uint32_t lo_significand : 32;
} _daal_dpbits_t;

typedef _daal_spbits_t daal_fp32;
typedef _daal_dpbits_t daal_fp64;

    #endif //!defined (DAAL_NO_BITFIELDS)

/* Big endian double */
typedef struct tag_daal_dpdwords_t
{
    uint32_t hi_dword;
    uint32_t lo_dword;
} _daal_dpdwords_t;

#endif /* defined (DAAL_LITTLE_ENDIAN) */

/*
//  This union is created to simplify use of IEEE-754 single and double
//  precision numbers in different contexts. Most typical ways of use of
//  different fields of _daal_sp_union_t and _daal_dp_union_t:
//
//  hex is used primarily for initialization of double precision entries in
//  table look-up.
//
//  bits is used for more convenient access to sign, exponent, and significand
//  of double precision number.
//
//  fp is used when floating point operations required.
//
*/

typedef union
{
    uint32_t hex[1];
#if !defined(DAAL_NO_BITFIELDS)
    _daal_spbits_t bits;
#endif
    float fp;
} _daal_sp_union_t;

typedef union
{
    uint32_t hex[2];
#if !defined(DAAL_NO_BITFIELDS)
    _daal_dpbits_t bits;
#endif
    _daal_dpdwords_t dwords;
    double fp;
} _daal_dp_union_t;

#define IMPLEMENT_SERIALIZABLE_TAG(Class, Tag) \
    int Class::serializationTag()              \
    {                                          \
        return Tag;                            \
    }                                          \
    int Class::getSerializationTag() const     \
    {                                          \
        return Class::serializationTag();      \
    }

#define IMPLEMENT_SERIALIZABLE_TAG1T(Class, T1, Tag)            \
    template <>                                                 \
    int Class<T1>::serializationTag()                           \
    {                                                           \
        return features::internal::getIndexNumType<T1>() + Tag; \
    }                                                           \
    template <>                                                 \
    int Class<T1>::getSerializationTag() const                  \
    {                                                           \
        return Class<T1>::serializationTag();                   \
    }

#define IMPLEMENT_SERIALIZABLE_TAG1T_SPECIALIZATION(Class, TemplateClass, Tag) \
    template <>                                                                \
    int Class<TemplateClass>::serializationTag()                               \
    {                                                                          \
        return Tag;                                                            \
    }                                                                          \
    template <>                                                                \
    int Class<TemplateClass>::getSerializationTag() const                      \
    {                                                                          \
        return Class<TemplateClass>::serializationTag();                       \
    }

#define IMPLEMENT_SERIALIZABLE_TAG2T(Class, T1, T2, Tag)        \
    template <>                                                 \
    int Class<T1, T2>::serializationTag()                       \
    {                                                           \
        return features::internal::getIndexNumType<T2>() + Tag; \
    }                                                           \
    template <>                                                 \
    int Class<T1, T2>::getSerializationTag() const              \
    {                                                           \
        return Class<T1, T2>::serializationTag();               \
    }

#define IMPLEMENT_SERIALIZABLE_TAG22(Class, T1, Tag) \
    int Class<T1, Tag>::serializationTag()           \
    {                                                \
        return Tag;                                  \
    }                                                \
    int Class<T1, Tag>::getSerializationTag() const  \
    {                                                \
        return Class<T1, Tag>::serializationTag();   \
    }

/* Maximal size of services::String */
#define DAAL_MAX_STRING_SIZE 4096

#define COMPUTE_DAAL_VERSION(majorVersion, minorVersion, updateVersion) (majorVersion * 10000 + minorVersion * 100 + updateVersion)

#if defined(_MSC_VER) || defined(DAAL_INTEL_CPP_COMPILER)
    #include <immintrin.h>
    #define DAAL_PREFETCH_READ_T0(addr) _mm_prefetch((char *)addr, _MM_HINT_T0)
#else
    #define DAAL_PREFETCH_READ_T0(addr) __builtin_prefetch((char *)addr, 0, 3)
#endif

#if defined(_MSC_VER)
    #define DAAL_FORCEINLINE   __forceinline
    #define DAAL_FORCENOINLINE __declspec(noinline)
#else
    #define DAAL_FORCEINLINE   inline __attribute__((always_inline))
    #define DAAL_FORCENOINLINE __attribute__((noinline))
#endif

#if defined(_MSC_VER) && (_MSC_VER < 1900)
    #define DAAL_ALIGNAS(n) __declspec(align(n))
#else
    #define DAAL_ALIGNAS(n) alignas(n)
#endif

#if defined(_MSC_VER)
    #define DAAL_PACKED_STRUCT(...) __pragma(pack(push, 1)) __VA_ARGS__ __pragma(pack(pop))
#else
    #define DAAL_PACKED_STRUCT(...) __VA_ARGS__ __attribute__((packed))
#endif

#define DAAL_SAFE_CPU_CALL(base, safe) \
    if (DAAL_CHECK_CPU_ENVIRONMENT)    \
    {                                  \
        (base);                        \
    }                                  \
    else                               \
    {                                  \
        (safe);                        \
    }

#endif
