/* file: service_rng_mkl.h */
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
//  Template wrappers for common Intel(R) MKL functions.
//--
*/

#ifndef __SERVICE_RNG_MKL_H__
#define __SERVICE_RNG_MKL_H__

#include <mkl.h>
#include "src/externals/service_stat_rng_mkl.h"
#include "src/externals/service_rng_common.h"

// RNGs
#define __DAAL_BRNG_MT2203                    VSL_BRNG_MT2203
#define __DAAL_BRNG_MT19937                   VSL_BRNG_MT19937
#define __DAAL_BRNG_MCG59                     VSL_BRNG_MCG59
#define __DAAL_BRNG_MRG32K3A                  VSL_BRNG_MRG32K3A
#define __DAAL_BRNG_PHILOX4X32X10             VSL_BRNG_PHILOX4X32X10
#define __DAAL_RNG_METHOD_UNIFORM_STD         VSL_RNG_METHOD_UNIFORM_STD
#define __DAAL_RNG_METHOD_UNIFORMBITS32_STD   0
#define __DAAL_RNG_METHOD_BERNOULLI_ICDF      VSL_RNG_METHOD_BERNOULLI_ICDF
#define __DAAL_RNG_METHOD_GAUSSIAN_BOXMULLER  VSL_RNG_METHOD_GAUSSIAN_BOXMULLER
#define __DAAL_RNG_METHOD_GAUSSIAN_BOXMULLER2 VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2
#define __DAAL_RNG_METHOD_GAUSSIAN_ICDF       VSL_RNG_METHOD_GAUSSIAN_ICDF

// Errors
#define __DAAL_RNG_ERROR_LEAPFROG_UNSUPPORTED  -1002
#define __DAAL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED -1003

namespace daal
{
namespace internal
{
namespace mkl
{
/* Uniform distribution generator functions */
template <typename T, CpuType cpu>
int uniformRNG(const size_t n, T * r, void * stream, const T a, const T b, const int method);

template <CpuType cpu>
int uniformRNG(const size_t cn, size_t * r, void * stream, const size_t a, const size_t b, const int method)
{
    size_t n    = cn;
    int errcode = 0;

    if (a >= b)
    {
        return -3;
    }

    if (sizeof(size_t) == sizeof(unsigned int))
    {
        size_t len = b - a;
        int na     = -(len / 2 + len % 2);
        int nb     = len / 2;
        int nn     = (int)n;
        int * rr   = (int *)r;
        __DAAL_VSLFN_CALL_NR_WHILE(iRngUniform, ((const MKL_INT)method, stream, (const MKL_INT)nn, rr, na, nb), errcode);

        if (errcode != 0)
        {
            return errcode;
        }

        size_t shift = a - na;
        for (size_t i = 0; i < n; i++)
        {
            r[i] = r[i] + shift;
        }
    }
    else
    {
        // works only for case when sizeof(size_t) > sizeof(unsigned int)
        if (b - a < ((size_t)1 << sizeof(size_t) * 8 / 2))
        {
            size_t len = b - a;
            int na     = -(len / 2 + len % 2);
            int nb     = len / 2;
            int nn     = (int)n;
            int * rr   = (int *)r + n;
            __DAAL_VSLFN_CALL_NR_WHILE(iRngUniform, ((const MKL_INT)method, stream, (const MKL_INT)nn, rr, na, nb), errcode);

            if (errcode != 0)
            {
                return errcode;
            }

            rr           = (int *)r + n;
            size_t shift = a - na;
            for (size_t i = 0; i < n; i++)
            {
                r[i] = rr[i] + shift;
            }
        }
        else
        {
            unsigned __int64 * cr = (unsigned __int64 *)r;

            size_t len     = b - a;
            size_t rem     = (size_t)(-1) % len;
            rem            = (rem + 1) % len;
            size_t MAX     = -rem;
            long double dv = 0.0;

            if (MAX == 0)
            {
                dv = len;
                for (int i = 0; i < 64; i++) dv /= 2.0;
                int nn                = (int)n;
                unsigned __int64 * rr = cr;
                __DAAL_VSLFN_CALL_NR_WHILE(iRngUniformBits64, ((const MKL_INT)method, stream, (const MKL_INT)nn, (unsigned MKL_INT64 *)rr), errcode);

                if (errcode != 0)
                {
                    return errcode;
                }
            }
            else
            {
                dv         = (long double)len / MAX;
                size_t pos = 0;
                while (pos < cn)
                {
                    n                     = cn - pos;
                    int nn                = (int)n;
                    unsigned __int64 * rr = cr + pos;
                    __DAAL_VSLFN_CALL_NR_WHILE(iRngUniformBits64, ((const MKL_INT)method, stream, (const MKL_INT)nn, (unsigned MKL_INT64 *)rr),
                                               errcode);

                    if (errcode != 0)
                    {
                        return errcode;
                    }

                    for (size_t i = pos; i < cn; i++)
                    {
                        if (cr[i] < MAX)
                        {
                            cr[pos++] = cr[i];
                        }
                    }
                }
            }

            for (size_t i = 0; i < cn; i++)
            {
                cr[i] = a + cr[i] * dv;
            }
        }
    }

    return errcode;
}

template <CpuType cpu>
int uniformRNG(const size_t n, int * r, void * stream, const int a, const int b, const int method)
{
    int errcode = 0;
    int nn      = (int)n;
    int * rr    = r;
    __DAAL_VSLFN_CALL_NR_WHILE(iRngUniform, ((const MKL_INT)method, stream, (const MKL_INT)nn, rr, a, b), errcode);
    return errcode;
}

template <CpuType cpu>
int uniformRNG(const size_t n, float * r, void * stream, const float a, const float b, const int method)
{
    int errcode = 0;
    int nn      = (int)n;
    float * rr  = r;
    __DAAL_VSLFN_CALL_NR_WHILE(sRngUniform, ((const MKL_INT)method, stream, (const MKL_INT)nn, rr, a, b), errcode);
    return errcode;
}

template <CpuType cpu>
int uniformRNG(const size_t n, double * r, void * stream, const double a, const double b, const int method)
{
    int errcode = 0;
    int nn      = (int)n;
    double * rr = r;
    __DAAL_VSLFN_CALL_NR_WHILE(dRngUniform, ((const MKL_INT)method, stream, (const MKL_INT)nn, rr, a, b), errcode);
    return errcode;
}

template <CpuType cpu>
int uniformBits32RNG(const size_t n, unsigned int * r, void * stream, const int method)
{
    int errcode       = 0;
    int nn            = (int)n;
    unsigned int * rr = r;
    __DAAL_VSLFN_CALL_NR_WHILE(iRngUniformBits32, ((const MKL_INT)method, stream, (const MKL_INT)nn, rr), errcode);
    return errcode;
}

/* Gaussian distribution generator functions */
template <typename T, CpuType cpu>
int gaussianRNG(const size_t n, T * r, void * stream, const T a, const T sigma, const int method);

template <CpuType cpu>
int gaussianRNG(const size_t n, float * r, void * stream, const float a, const float sigma, const int method)
{
    int errcode = 0;
    int nn      = (int)n;
    float * rr  = r;
    __DAAL_VSLFN_CALL_NR_WHILE(sRngGaussian, ((const MKL_INT)method, stream, (const MKL_INT)nn, rr, a, sigma), errcode);
    return errcode;
}

template <CpuType cpu>
int gaussianRNG(const size_t n, double * r, void * stream, const double a, const double sigma, const int method)
{
    int errcode = 0;
    int nn      = (int)n;
    double * rr = r;
    __DAAL_VSLFN_CALL_NR_WHILE(dRngGaussian, ((const MKL_INT)method, stream, (const MKL_INT)nn, rr, a, sigma), errcode);
    return errcode;
}

/* Bernoulli distribution generator functions */
template <typename T, CpuType cpu>
int bernoulliRNG(const size_t n, T * r, void * stream, const double p, const int method);

template <CpuType cpu>
int bernoulliRNG(const size_t n, int * r, void * stream, const double p, const int method)
{
    int errcode = 0;
    int nn      = (int)n;
    int * rr    = r;
    __DAAL_VSLFN_CALL_NR_WHILE(iRngBernoulli, ((const MKL_INT)method, stream, (const MKL_INT)nn, rr, p), errcode);
    return errcode;
}

template <CpuType cpu>
class BaseRNG : public BaseRNGIface<cpu>
{
public:
    BaseRNG(const unsigned int seed, const int brngId) : _stream(0), _seed(nullptr), _seedSize(0), _brngId(brngId)
    {
        services::Status s = allocSeeds(1);
        if (s)
        {
            _seed[0]    = seed;
            int errcode = 0;
            __DAAL_VSLFN_CALL_NR(vslNewStreamEx, (&_stream, (const MKL_INT)brngId, (const MKL_INT)1, &seed), errcode);
        }
    }

    BaseRNG(const size_t n, const unsigned int * seed, const int brngId = __DAAL_BRNG_MT19937)
        : _stream(0), _seed(nullptr), _seedSize(0), _brngId(brngId)
    {
        services::Status s = allocSeeds(n);
        if (s)
        {
            if (seed)
            {
                for (size_t i = 0; i < n; i++)
                {
                    _seed[i] = seed[i];
                }
            }
            int errcode = 0;
            __DAAL_VSLFN_CALL_NR(vslNewStreamEx, (&_stream, (const MKL_INT)brngId, (const MKL_INT)n, seed), errcode);
        }
    }

    BaseRNG(const BaseRNG<cpu> & other) : _stream(0), _seed(nullptr), _seedSize(other._seedSize), _brngId(other._brngId)
    {
        services::Status s = allocSeeds(_seedSize);
        if (s)
        {
            for (size_t i = 0; i < _seedSize; i++)
            {
                _seed[i] = other._seed[i];
            }
            int errcode = 0;
            __DAAL_VSLFN_CALL_NR(vslNewStreamEx, (&_stream, (const MKL_INT)_brngId, (const MKL_INT)_seedSize, _seed), errcode);
            if (!errcode) __DAAL_VSLFN_CALL_NR(vslCopyStreamState, (_stream, other._stream), errcode);
        }
    }

    ~BaseRNG()
    {
        daal::services::daal_free((void *)_seed);
        int errcode = 0;
        __DAAL_VSLFN_CALL_NR(vslDeleteStream, (&_stream), errcode);
    }

    int getStateSize() const
    {
        int res = 0;
        __DAAL_VSLFN_CALL_NR(vslGetStreamSize, (_stream), res);
        return res;
    }

    int saveState(void * dest) const
    {
        int errcode = 0;
        __DAAL_VSLFN_CALL_NR(vslSaveStreamM, (_stream, (char *)dest), errcode);
        return errcode;
    }

    int loadState(const void * src)
    {
        int errcode = 0;
        __DAAL_VSLFN_CALL_NR(vslDeleteStream, (&_stream), errcode);
        if (!errcode) __DAAL_VSLFN_CALL_NR(vslLoadStreamM, (&_stream, (const char *)src), errcode);
        return errcode;
    }

    int leapfrog(size_t threadNum, size_t nThreads)
    {
        int errcode = 0;
        __DAAL_VSLFN_CALL_NR(vslLeapfrogStream, (_stream, (const MKL_INT)threadNum, (const MKL_INT)nThreads), errcode);
        return errcode;
    }

    int skipAhead(size_t nSkip)
    {
        int errcode = 0;
        __DAAL_VSLFN_CALL_NR(vslSkipAheadStream, (_stream, nSkip), errcode);
        return errcode;
    }

    void * getState() { return _stream; }

protected:
    services::Status allocSeeds(const size_t n)
    {
        _seedSize = n;
        _seed     = (unsigned int *)daal::services::daal_malloc(sizeof(unsigned int) * n);
        DAAL_CHECK_MALLOC(_seed);
        return services::Status();
    }

private:
    void * _stream;
    unsigned int * _seed;
    size_t _seedSize;
    const int _brngId;
};

/*
// Generator functions definition
*/
template <typename Type, CpuType cpu>
class RNGs
{
public:
    typedef DAAL_INT SizeType;
    typedef BaseRNG<cpu> BaseType;

    RNGs() {}

    int uniform(const SizeType n, Type * r, BaseType & brng, const Type a, const Type b, const int method)
    {
        return uniformRNG<cpu>(n, r, brng.getState(), a, b, method);
    }

    int uniform(const SizeType n, Type * r, void * state, const Type a, const Type b, const int method)
    {
        return uniformRNG<cpu>(n, r, state, a, b, method);
    }

    int uniformBits32(const SizeType n, Type * r, void * state, const int method) { return uniformBits32RNG<cpu>(n, r, state, method); }

    int bernoulli(const SizeType n, Type * r, BaseType & brng, const double p, const int method)
    {
        return bernoulliRNG<cpu>(n, r, brng.getState(), p, method);
    }

    int bernoulli(const SizeType n, Type * r, void * state, const double p, const int method) { return bernoulliRNG<cpu>(n, r, state, p, method); }

    int gaussian(const SizeType n, Type * r, BaseType & brng, const Type a, const Type sigma, const int method)
    {
        return gaussianRNG<cpu>(n, r, brng.getState(), a, sigma, method);
    }

    int gaussian(const SizeType n, Type * r, void * state, const Type a, const Type sigma, const int method)
    {
        return gaussianRNG<cpu>(n, r, state, a, sigma, method);
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
