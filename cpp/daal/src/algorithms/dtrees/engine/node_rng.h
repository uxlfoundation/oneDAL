/* file: node_rng.h */
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

/*
//++
//  Draws addressed by node identity rather than by consumption order, so they do
//  not depend on which thread takes them or when (design 4.13).
//--
*/

#ifndef __DTREES_ENGINE_NODE_RNG_H__
#define __DTREES_ENGINE_NODE_RNG_H__

#include <cstddef>
#include <cstdint>

#include "src/externals/service_rng.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace engine
{
namespace internal
{
using daal::internal::CpuType;

// Golden-ratio increment and finalizer multipliers of splitmix64 (Steele et al., OOPSLA'14).
constexpr uint64_t splitmix64Gamma = 0x9E3779B97F4A7C15ull;
constexpr uint64_t splitmix64Mul0  = 0xBF58476D1CE4E5B9ull;
constexpr uint64_t splitmix64Mul1  = 0x94D049BB133111EBull;

/// splitmix64 finalizer, a bijection on 64-bit values.
template <CpuType cpu>
inline uint64_t mix64(uint64_t z)
{
    z = (z ^ (z >> 30)) * splitmix64Mul0;
    z = (z ^ (z >> 27)) * splitmix64Mul1;
    return z ^ (z >> 31);
}

/// Structural key of a node's child, side 0 (left) or 1 (right). It depends only on the parent's key,
/// so a node's draws do not depend on when the node was created.
template <CpuType cpu>
inline uint64_t childNodeKey(uint64_t parentKey, size_t side)
{
    return mix64<cpu>(parentKey + uint64_t(side + 1) * splitmix64Gamma);
}

// philox4x32x10 emits 4 words per value of its 128-bit counter.
constexpr size_t philoxBlockWords = 4;

/**
 *  \brief philox4x32x10 words addressed by (stream id, position), for one 64-bit key
 *
 *  A node's draws are a function of (treeSeed, nodeKey, index) only, so a forest trained with
 *  philox4x32x10 is bit-identical for any thread count and schedule (design 4.13).
 *
 *  Word w of stream s is output s * 2^66 + w of the sequence keyed by key: its counter holds w / 4 in
 *  the low 64 bits and s in the high 64 bits. Stream ids are 64-bit node keys, hashes that stay rare to
 *  collide at any depth, so positions reach 2^130 and need the multi-word skipAheadEx. With key = seed,
 *  stream 0 is the output of engines::philox4x32x10::Batch(seed).
 *
 *  Draws go through service_rng so that MKL and OpenRNG maintain the generator. A read costs about 5x a
 *  direct philox round function, mostly the generate call (design 4.13). Each read copies the keyed base
 *  state into the calling thread's own stream and skips there, so concurrent reads share nothing
 *  mutable. Backends that cannot skip past 2^64 (the reference backend, whose philox4x32x10 is not
 *  philox) fail init with ErrorMethodNotSupported.
 */
template <CpuType cpu>
class CounterRng
{
    using Stream = daal::internal::BaseRNGsInst<cpu>;

public:
    CounterRng() : _streams([this]() -> Stream * { return _base ? new Stream(*_base) : nullptr; }) {}
    ~CounterRng()
    {
        _streams.reduce([](Stream * s) { delete s; });
        delete _base;
    }
    CounterRng(const CounterRng &)             = delete;
    CounterRng & operator=(const CounterRng &) = delete;

    services::Status init(uint64_t key)
    {
        const unsigned int seed[2] = { unsigned(key), unsigned(key >> 32) };
        delete _base;
        _base = new Stream(2, seed, __DAAL_BRNG_PHILOX4X32X10);
        DAAL_CHECK_MALLOC(_base && _base->getState());
        // A skip with a nonzero high word tells whether the backend can address the whole counter.
        Stream probe(*_base);
        const DAAL_UINT64 skip[3] = { 0, 1, 0 };
        if (probe.skipAheadEx(3, skip)) return services::Status(services::ErrorMethodNotSupported);
        return services::Status();
    }

    /// Writes words [position, position + n) of stream streamId to out.
    services::Status read(uint64_t streamId, uint64_t position, size_t n, uint32_t * out) const
    {
        Stream * s = _streams.local();
        DAAL_CHECK_MALLOC(s && s->getState());
        // Skip of streamId * 2^66 + position words, as three 64-bit words.
        const DAAL_UINT64 skip[3] = { position, streamId << 2, streamId >> 62 };
        if (s->copyState(_base->getState()) || s->skipAheadEx(3, skip)) return services::Status(services::ErrorIncorrectErrorcodeFromGenerator);
        daal::internal::RNGsInst<unsigned int, cpu> rng;
        if (rng.uniformBits32(n, out, s->getState())) return services::Status(services::ErrorIncorrectErrorcodeFromGenerator);
        return services::Status();
    }

private:
    Stream * _base = nullptr;
    mutable daal::tls<Stream *> _streams;
};

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
