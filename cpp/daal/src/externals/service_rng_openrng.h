/* file: service_rng_openrng.h */
/*******************************************************************************
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

#ifndef __SERVICE_RNG_OPENRNG_H__
#define __SERVICE_RNG_OPENRNG_H__

#include "openrng.h"
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

namespace daal
{
namespace internal
{
namespace ref
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
        size_t len       = b - a;
        int na           = -(len / 2 + len % 2);
        int nb           = len / 2;
        openrng_int_t nn = (openrng_int_t)n;
        int * rr         = (int *)r;
        errcode          = viRngUniform((openrng_int_t)method, stream, nn, rr, na, nb);

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
            size_t len       = b - a;
            int na           = -(len / 2 + len % 2);
            int nb           = len / 2;
            openrng_int_t nn = (openrng_int_t)n;
            int * rr         = (int *)r + n;
            errcode          = viRngUniform((openrng_int_t)method, stream, nn, rr, na, nb);

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
                openrng_int_t nn      = (openrng_int_t)n;
                unsigned __int64 * rr = cr;
                errcode               = viRngUniformBits64((openrng_int_t)method, stream, nn, rr);

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
                    openrng_int_t nn      = (openrng_int_t)n;
                    unsigned __int64 * rr = cr + pos;
                    errcode               = viRngUniformBits64((openrng_int_t)method, stream, nn, rr);

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
    int errcode      = 0;
    openrng_int_t nn = (openrng_int_t)n;
    int * rr         = r;
    errcode          = viRngUniform((openrng_int_t)method, stream, nn, rr, a, b);
    return errcode;
}

template <CpuType cpu>
int uniformRNG(const size_t n, float * r, void * stream, const float a, const float b, const int method)
{
    int errcode      = 0;
    openrng_int_t nn = (openrng_int_t)n;
    float * rr       = r;
    errcode          = vsRngUniform((openrng_int_t)method, stream, nn, rr, a, b);
    return errcode;
}

template <CpuType cpu>
int uniformRNG(const size_t n, double * r, void * stream, const double a, const double b, const int method)
{
    int errcode      = 0;
    openrng_int_t nn = (openrng_int_t)n;
    double * rr      = r;
    errcode          = vdRngUniform((openrng_int_t)method, stream, nn, rr, a, b);
    return errcode;
}

template <CpuType cpu>
int uniformBits32RNG(const size_t n, unsigned int * r, void * stream, const int method)
{
    int errcode       = 0;
    openrng_int_t nn  = (openrng_int_t)n;
    unsigned int * rr = r;
    errcode           = viRngUniformBits32((openrng_int_t)method, stream, nn, rr);
    return errcode;
}

/* Gaussian distribution generator functions */
template <typename T, CpuType cpu>
int gaussianRNG(const size_t n, T * r, void * stream, const T a, const T sigma, const int method);

template <CpuType cpu>
int gaussianRNG(const size_t n, float * r, void * stream, const float a, const float sigma, const int method)
{
    int errcode      = 0;
    openrng_int_t nn = (openrng_int_t)n;
    float * rr       = r;
    errcode          = vsRngGaussian((openrng_int_t)method, stream, nn, rr, a, sigma);
    return errcode;
}

template <CpuType cpu>
int gaussianRNG(const size_t n, double * r, void * stream, const double a, const double sigma, const int method)
{
    int errcode      = 0;
    openrng_int_t nn = (openrng_int_t)n;
    double * rr      = r;
    errcode          = vdRngGaussian((openrng_int_t)method, stream, nn, rr, a, sigma);
    return errcode;
}

/* Bernoulli distribution generator functions */
template <typename T, CpuType cpu>
int bernoulliRNG(const size_t n, T * r, void * stream, const double p, const int method);

template <CpuType cpu>
int bernoulliRNG(const size_t n, int * r, void * stream, const double p, const int method)
{
    int errcode      = 0;
    openrng_int_t nn = (openrng_int_t)n;
    int * rr         = r;
    errcode          = viRngBernoulli((openrng_int_t)method, stream, nn, rr, p);
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
            errcode     = vslNewStreamEx(&_stream, (openrng_int_t)brngId, 1, &seed);
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
            errcode     = vslNewStreamEx(&_stream, (openrng_int_t)brngId, (openrng_int_t)n, seed);
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
            errcode     = vslNewStreamEx(&_stream, _brngId, _seedSize, _seed);
            if (!errcode) errcode = vslCopyStreamState(_stream, other._stream);
        }
    }

    ~BaseRNG()
    {
        daal::services::daal_free((void *)_seed);
        int errcode = 0;
        errcode     = vslDeleteStream(&_stream);
    }

    int getStateSize() const
    {
        int res = 0;
        res     = vslGetStreamSize(_stream);
        return res;
    }

    int saveState(void * dest) const
    {
        int errcode = 0;
        errcode     = vslSaveStreamM(_stream, (char *)dest);
        return errcode;
    }

    int loadState(const void * src)
    {
        int errcode = 0;
        errcode     = vslDeleteStream(&_stream);
        if (!errcode) errcode = vslLoadStreamM(&_stream, (const char *)src);
        return errcode;
    }

    int leapfrog(size_t threadNum, size_t nThreads)
    {
        int errcode = 0;
        errcode     = vslLeapfrogStream(_stream, (openrng_int_t)threadNum, (openrng_int_t)nThreads);
        return errcode;
    }

    int skipAhead(size_t nSkip)
    {
        int errcode = 0;
        errcode     = vslSkipAheadStream(_stream, (long long int)nSkip);
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

} // namespace ref
} // namespace internal
} // namespace daal

#endif
