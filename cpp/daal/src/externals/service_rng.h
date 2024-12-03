/* file: service_rng.h */
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
//  Template wrappers for RNG functions.
//--
*/

#ifndef __SERVICE_RNG_H__
#define __SERVICE_RNG_H__

#include "services/daal_defines.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_rng_common.h"

#include "src/externals/config.h"

using namespace daal::services;

namespace daal
{
namespace internal
{
template <CpuType cpu, typename _BaseGenerators>
class BaseRNGs : public BaseRNGIface<cpu>
{
public:
    BaseRNGs(const unsigned int _seed = 777, const int _brngId = __DAAL_BRNG_MT19937) : _baseRNG(_seed, _brngId) {}
    ~BaseRNGs() {}

    int getStateSize() const { return _baseRNG.getStateSize(); }

    int saveState(void * dest) const { return _baseRNG.saveState(dest); }

    int loadState(const void * src) { return _baseRNG.loadState(src); }

    int leapfrog(size_t threadNum, size_t nThreads) { return _baseRNG.leapfrog(threadNum, nThreads); }

    int skipAhead(size_t nSkip) { return _baseRNG.skipAhead(nSkip); }

    void * getState() { return _baseRNG.getState(); }

    _BaseGenerators & getBrng() { return _baseRNG; }

    BaseRNGs(const BaseRNGs<cpu, _BaseGenerators> & other) : _baseRNG(other._baseRNG) {}

private:
    _BaseGenerators _baseRNG;
};

template <typename Type, CpuType cpu, typename _RNGs>
class RNGs
{
public:
    typedef typename _RNGs::SizeType SizeType;
    typedef typename _RNGs::BaseType BaseType;

    RNGs() {}
    ~RNGs() {}

    /* Generates random numbers uniformly distributed on the interval [a, b) */
    int uniform(const SizeType n, Type * r, BaseRNGs<cpu, BaseType> & brng, const Type a, const Type b,
                const int method = __DAAL_RNG_METHOD_UNIFORM_STD)
    {
        return _generators.uniform(n, r, brng.getBrng(), a, b, method);
    }

    int uniform(const SizeType n, Type * r, void * state, const Type a, const Type b, const int method = __DAAL_RNG_METHOD_UNIFORM_STD)
    {
        return _generators.uniform(n, r, state, a, b, method);
    }

    int uniformBits32(const SizeType n, Type * r, void * state, const int method = __DAAL_RNG_METHOD_UNIFORMBITS32_STD)
    {
        return _generators.uniformBits32(n, r, state, method);
    }

    int bernoulli(const SizeType n, Type * r, BaseRNGs<cpu, BaseType> & brng, const double p, const int method = __DAAL_RNG_METHOD_BERNOULLI_ICDF)
    {
        return _generators.bernoulli(n, r, brng.getBrng(), p, method);
    }

    int bernoulli(const SizeType n, Type * r, void * state, const double p, const int method = __DAAL_RNG_METHOD_BERNOULLI_ICDF)
    {
        return _generators.bernoulli(n, r, state, p, method);
    }

    int gaussian(const SizeType n, Type * r, BaseRNGs<cpu, BaseType> & brng, const Type a, const Type sigma,
                 const int method = __DAAL_RNG_METHOD_GAUSSIAN_ICDF)
    {
        return _generators.gaussian(n, r, brng.getBrng(), a, sigma, method);
    }

    int gaussian(const SizeType n, Type * r, void * state, const Type a, const Type sigma, const int method = __DAAL_RNG_METHOD_GAUSSIAN_ICDF)
    {
        return _generators.gaussian(n, r, state, a, sigma, method);
    }

    template <typename DstType = Type>
    int uniformWithoutReplacement(const SizeType n, DstType * r, void * state, const Type a, const Type b,
                                  const int method = __DAAL_RNG_METHOD_UNIFORM_STD)
    {
        Type * buffer = (Type *)daal_malloc(sizeof(Type) * 1);
        int errorcode = uniformWithoutReplacement(n, r, buffer, state, a, b, method);
        daal_free(buffer);
        return errorcode;
    }

    template <typename DstType = Type>
    int uniformWithoutReplacement(const SizeType n, DstType * r, Type * buffer, void * state, const Type a, const Type b,
                                  const int method = __DAAL_RNG_METHOD_UNIFORM_STD)
    {
        int errorcode          = 0;
        SizeType sequence_size = abs(b - a);
        if (sequence_size < n)
        {
            return -1;
        }
        Type * buffer_ = (Type *)daal_malloc(sizeof(Type) * sequence_size);
        for (SizeType i = 0; i < sequence_size; i++)
        {
            buffer_[i] = i;
        }
        Type swapIdx;
        for (SizeType i = 0; i < n; i++)
        {
            errorcode = uniform(1, &swapIdx, state, i, sequence_size, method);
            int index = int(swapIdx);

            std::swap(buffer_[i], buffer_[index]);
        }
        for (SizeType i = 0; i < n; i++)
        {
            r[i] = buffer_[i];
        }
        return errorcode;
    }

    /* Draw a random sample of length k from the numbers that are provided in buffer of length n
    * \param[in]  k       The length of the sample to be drawn
    * \param[out] r       A pointer to the result buffer
    * \param[in]  buffer  A pointer to the buffer containing the numbers
    * \param[in]  n       Length of the buffer
    * \param[in]  method  Method handed to the uniform random number generator
    *
    * This method is based on the Fisher Yates sampling technique, but since we are re-using the provided buffer, there
    * is no need to initialize it to [0, 1, 2, ..., n-1] first, providing us with a speed-up from O(n) -> O(k) runtime
    */
    template <typename DstType = Type>
    int drawKFromBufferWithoutReplacement(const SizeType k, DstType * r, Type * buffer, void * state, const Type n,
                                          const int method = __DAAL_RNG_METHOD_UNIFORM_STD)
    {
        int errorcode = 0;
        Type swapIdx;

        for (SizeType i = 0; i < k; ++i)
        {
            errorcode         = uniform(1, &swapIdx, state, 0, n - i, method);
            r[i]              = (DstType)buffer[swapIdx];
            buffer[swapIdx]   = buffer[n - 1 - i];
            buffer[n - 1 - i] = (Type)r[i];
        }

        return errorcode;
    }

    template <typename DstType = Type>
    int uniformWithoutReplacement(const SizeType n, Type * r, BaseRNGs<cpu, BaseType> & brng, const Type a, const Type b,
                                  const int method = __DAAL_RNG_METHOD_UNIFORM_STD)
    {
        return uniformWithoutReplacement(n, r, brng.getState(), a, b, method);
    }

    template <typename DstType = Type>
    int uniformWithoutReplacement(const SizeType n, Type * r, Type * buffer, BaseRNGs<cpu, BaseType> & brng, const Type a, const Type b,
                                  const int method = __DAAL_RNG_METHOD_UNIFORM_STD)
    {
        return uniformWithoutReplacement(n, r, buffer, brng.getState(), a, b, method);
    }

private:
    _RNGs _generators;
};

} // namespace internal
} // namespace daal

namespace daal
{
namespace internal
{
template <CpuType cpu>
using BaseRNGsInst = BaseRNGs<cpu, BaseRngBackend<cpu> >;

template <typename Type, CpuType cpu>
using RNGsInst = RNGs<Type, cpu, RNGsBackend<Type, cpu> >;

} // namespace internal
} // namespace daal

#endif
