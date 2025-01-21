/* file: service_rng_ref.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
//  Template wrappers for common a random generation functions.
//--
*/

#ifndef __SERVICE_RNG_REF_H__
#define __SERVICE_RNG_REF_H__

#ifdef OPENRNG_BACKEND

    #include "service_rng_openrng.h"

#else

    #include "src/externals/service_rng_common.h"
    #include "services/error_indexes.h"
    #include <random>

    // RNGs
    #define __DAAL_BRNG_MT2203        (1 << 20) * 9  //VSL_BRNG_MT2203
    #define __DAAL_BRNG_MT19937       (1 << 20) * 8  //VSL_BRNG_MT19937
    #define __DAAL_BRNG_MCG59         (1 << 20) * 4  //VSL_BRNG_MCG59
    #define __DAAL_BRNG_MRG32K3A      (1 << 20) * 3  //VSL_BRNG_MRG32K3A
    #define __DAAL_BRNG_PHILOX4X32X10 (1 << 20) * 16 //VSL_BRNG_PHILOX4X32X10

    #define __DAAL_RNG_METHOD_UNIFORM_STD         0 //VSL_RNG_METHOD_UNIFORM_STD
    #define __DAAL_RNG_METHOD_UNIFORMBITS32_STD   4
    #define __DAAL_RNG_METHOD_BERNOULLI_ICDF      0 //VSL_RNG_METHOD_BERNOULLI_ICDF
    #define __DAAL_RNG_METHOD_GAUSSIAN_BOXMULLER  0 //VSL_RNG_METHOD_GAUSSIAN_BOXMULLER
    #define __DAAL_RNG_METHOD_GAUSSIAN_BOXMULLER2 1 //VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2
    #define __DAAL_RNG_METHOD_GAUSSIAN_ICDF       2 //VSL_RNG_METHOD_GAUSSIAN_ICDF

namespace daal
{
namespace internal
{
namespace ref
{
DAAL_FORCEINLINE bool genTypeIsSame(int brngId, int genTypeId)
{
    return (brngId >> 20) == (genTypeId >> 20);
}

class StateIface
{
public:
    virtual ~StateIface() {}
    virtual int uniformRNG(const size_t n, size_t * r, const size_t a, const size_t b, const int method) = 0;
    virtual int uniformRNG(const size_t n, int * r, const int a, const int b, const int method) = 0;
    virtual int uniformRNG(const size_t n, float * r, const float a, const float b, const int method) = 0;
    virtual int uniformRNG(const size_t n, double * r, const double a, const double b, const int method) = 0;
    virtual int gaussianRNG(const size_t n, float * r, const float a, const float sigma, const int method) = 0;
    virtual int gaussianRNG(const size_t n, double * r, const double a, const double sigma, const int method) = 0;
    virtual int bernoulliRNG(const size_t n, int * r, const double p, const int method) = 0;
    virtual int getSize() = 0;
    virtual void clone(void * dest) const = 0;
    virtual StateIface * clone() const = 0;
    virtual void assign(const void * src) = 0;
    virtual void discard(size_t nSkip) = 0;
};

template <typename Gen = std::mt19937>
class State : public StateIface
{
    using ThisType = State<Gen>;
    static constexpr unsigned int elSize = sizeof(unsigned int);

public:
    int uniformRNG(const size_t n, size_t * r, const size_t a, const size_t b, const int method) final { return iUniform(n, r, a, b, method); }

    int uniformRNG(const size_t n, int * r, const int a, const int b, const int method) final { return iUniform(n, r, a, b, method); }

    int uniformRNG(const size_t n, float * r, const float a, const float b, const int method) final { return fUniform(n, r, a, b, method); }

    int uniformRNG(const size_t n, double * r, const double a, const double b, const int method) final { return fUniform(n, r, a, b, method); }

    int gaussianRNG(const size_t n, float * r, const float a, const float sigma, const int method) final { return fGaussian(n, r, a, sigma, method); }

    int gaussianRNG(const size_t n, double * r, const double a, const double sigma, const int method) final
    {
        return fGaussian(n, r, a, sigma, method);
    }

    int bernoulliRNG(const size_t n, int * r, const double p, const int method) final { return iBernoulli(n, r, p, method); }

    State(const int brngId, const size_t seedSize, const unsigned int * seed)
        : _gen(0), _brngId(brngId), _seedSize(seedSize), _seed(nullptr), _nSkip(0)
    {
        _seed = (unsigned int *)daal::services::daal_malloc(sizeof(unsigned int) * _seedSize);
        if (_seedSize > 0 && seed != nullptr)
        {
            daal::services::daal_memcpy_s(_seed, _seedSize * elSize, seed, seedSize * elSize);
            _gen.seed(_seed[0]);
        }
    }
    State(const State & other) : _gen(0), _brngId(other._brngId), _seedSize(other._seedSize), _seed(nullptr), _nSkip(other._nSkip)
    {
        if (_seedSize > 0 && other._seed != nullptr)
        {
            _seed = (unsigned int *)daal::services::daal_malloc(sizeof(unsigned int) * _seedSize);
            daal::services::daal_memcpy_s(_seed, _seedSize * elSize, other._seed, other._seedSize * elSize);
            _gen.seed(_seed[0]);
            _gen.discard(_nSkip);
        }
    }
    ~State()
    {
        if (_seed)
        {
            daal::services::daal_free(_seed);
            _seed = nullptr;
        }
    }
    int getSize() final { return sizeof(ThisType); }
    void clone(void * dest) const final
    {
        State * destState = static_cast<State *>(dest);
        destState->_seedSize = _seedSize;
        destState->_brngId = _brngId;
        destState->_nSkip = _nSkip;
        destState->_seed = (unsigned int *)daal::services::daal_malloc(sizeof(unsigned int) * destState->_seedSize);
        daal::services::daal_memcpy_s(destState->_seed, destState->_seedSize * elSize, _seed, _seedSize * elSize);
    }
    StateIface * clone() const final { return new ThisType(*this); }
    void assign(const void * src) final
    {
        const State * srcState = static_cast<const State *>(src);
        _seedSize = srcState->_seedSize;
        _brngId = srcState->_brngId;
        _nSkip = srcState->_nSkip;
        if (_seed)
        {
            daal::services::daal_free((void *)_seed);
            _seed = nullptr;
        }
        if (srcState->_seedSize > 0 && srcState->_seed != nullptr)
        {
            _seed = (unsigned int *)daal::services::daal_malloc(sizeof(unsigned int) * _seedSize);
            daal::services::daal_memcpy_s(_seed, _seedSize * elSize, srcState->_seed, srcState->_seedSize * elSize);
            _gen.seed(_seed[0]);
            _gen.discard(_nSkip);
        }
    }
    void discard(size_t nSkip) final
    {
        _nSkip += nSkip;
        _gen.discard(nSkip);
    }
    size_t rngType() { return _brngId; }
    unsigned int * rngSeed() { return _seed; }
    size_t rngSeedSize() { return _seedSize; }
    size_t skipSize() { return _nSkip; }

    template <typename T>
    int iUniform(const size_t n, T * r, const T a, const T b, const int method)
    {
        int errcode = 0;
        if (rngSeedSize() == 0 || rngSeed() == nullptr) return services::ErrorNullInput;
        std::uniform_int_distribution<T> distrib(a, b - 1);
        for (size_t i = 0; i < n; i++)
        {
            r[i] = distrib(_gen);
        }
        _nSkip += n;
        return errcode;
    }
    template <typename T>
    int fUniform(const size_t n, T * r, const T a, const T b, const int method)
    {
        int errcode = 0;
        if (rngSeedSize() == 0 || rngSeed() == nullptr) return services::ErrorNullInput;
        std::uniform_real_distribution<T> distrib(a, b);
        for (size_t i = 0; i < n; i++)
        {
            r[i] = distrib(_gen);
        }
        _nSkip += n;
        return errcode;
    }
    template <typename T>
    int fGaussian(const size_t n, T * r, const T a, const T sigma, const int method)
    {
        int errcode = 0;
        if (rngSeedSize() == 0 || rngSeed() == nullptr) return services::ErrorNullInput;
        if (!std::is_floating_point<T>::value) return services::ErrorDataTypeNotSupported;
        std::normal_distribution<T> distrib(a, sigma);
        for (size_t i = 0; i < n; i++)
        {
            r[i] = distrib(_gen);
        }
        _nSkip += n;
        return errcode;
    }
    template <typename T>
    int iBernoulli(const size_t n, T * r, const double p, const int method)
    {
        int errcode = 0;
        if (rngSeedSize() == 0 || rngSeed() == nullptr) return services::ErrorNullInput;
        if (!std::is_integral<T>::value) return services::ErrorDataTypeNotSupported;
        std::bernoulli_distribution distrib(p);
        for (size_t i = 0; i < n; i++)
        {
            r[i] = distrib(_gen);
        }
        _nSkip += n;
        return errcode;
    }

private:
    Gen _gen;
    int _brngId;
    size_t _seedSize;
    unsigned int * _seed;
    size_t _nSkip;
};

template <CpuType cpu>
class BaseRNG : public BaseRNGIface<cpu>
{
public:
    BaseRNG(const unsigned int seed, const int brngId) : _stream(0)
    {
        unsigned int tempSeed = seed;
        if (genTypeIsSame(brngId, __DAAL_BRNG_MT2203))
        {
            // TODO Replace workaround
            tempSeed += brngId - __DAAL_BRNG_MT2203;
        }
        _stream = new State<>(brngId, 1, &tempSeed);
    }

    BaseRNG(const size_t n, const unsigned int * seed, const int brngId = __DAAL_BRNG_MT19937) : _stream(0)
    {
        _stream = new State<>(brngId, n, seed);
    }

    BaseRNG(const BaseRNG<cpu> & other) : _stream(0) { _stream = other._stream->clone(); }

    ~BaseRNG()
    {
        if (_stream != nullptr)
        {
            delete _stream;
            _stream = nullptr;
        }
    }

    int getStateSize() const { return _stream->getSize(); }

    int saveState(void * dest) const
    {
        int errcode = 0;
        _stream->clone((char *)dest);
        return errcode;
    }

    int loadState(const void * src)
    {
        int errcode = 0;
        _stream->assign(src);
        return errcode;
    }

    // Not implemented
    int leapfrog(size_t threadNum, size_t nThreads)
    {
        int errcode = 0;
        // TODO add call for (vslLeapfrogStream, (_stream, threadNum, nThreads), errcode);
        return errcode;
    }

    int skipAhead(size_t nSkip)
    {
        int errcode = 0;
        _stream->discard(nSkip);
        return errcode;
    }

    void * getState() { return _stream; }

private:
    StateIface * _stream;
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
        return uniform(n, r, brng.getState(), a, b, method);
    }

    int uniform(const SizeType n, Type * r, void * stream, const Type a, const Type b, const int method)
    {
        StateIface * state = static_cast<StateIface *>(stream);
        return state->uniformRNG(n, r, a, b, method);
    }

    int uniformBits32(const SizeType n, Type * r, void * stream, const int method) { return services::ErrorMethodNotSupported; }

    int bernoulli(const SizeType n, Type * r, BaseType & brng, const double p, const int method)
    {
        return bernoulli(n, r, brng.getState(), p, method);
    }

    int bernoulli(const SizeType n, Type * r, void * stream, const double p, const int method)
    {
        StateIface * state = static_cast<StateIface *>(stream);
        return state->bernoulliRNG(n, r, p, method);
    }

    int gaussian(const SizeType n, Type * r, BaseType & brng, const Type a, const Type sigma, const int method)
    {
        return gaussian(n, r, brng.getState(), a, sigma, method);
    }

    int gaussian(const SizeType n, Type * r, void * stream, const Type a, const Type sigma, const int method)
    {
        StateIface * state = static_cast<StateIface *>(stream);
        return state->gaussianRNG(n, r, a, sigma, method);
    }
};

} // namespace ref
} // namespace internal
} // namespace daal

#endif // openrng
#endif
