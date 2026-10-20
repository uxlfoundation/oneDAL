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
/// Per-thread working streams of philox4x32x10, shared by every CounterRng of a fit. A thread's
/// stream is reset from a CounterRng's base state before each read, so one stream per thread serves
/// all keys.
template <CpuType cpu>
class CounterStreams
{
    using Stream = daal::internal::BaseRNGsInst<cpu>;

public:
    CounterStreams()
        : _streams([]() -> Stream * {
              const unsigned int seed[2] = { 0, 0 };
              return new Stream(2, seed, __DAAL_BRNG_PHILOX4X32X10);
          })
    {}
    ~CounterStreams()
    {
        _streams.reduce([](Stream * s) { delete s; });
    }
    CounterStreams(const CounterStreams &)             = delete;
    CounterStreams & operator=(const CounterStreams &) = delete;

    Stream * local() const { return _streams.local(); }

private:
    mutable daal::tls<Stream *> _streams;
};

template <CpuType cpu>
class CounterRng
{
    using Stream = daal::internal::BaseRNGsInst<cpu>;

public:
    /// Reads through shared's per-thread streams, or through its own when shared is null.
    explicit CounterRng(const CounterStreams<cpu> * shared = nullptr) : _streams(shared ? shared : &_own) {}
    ~CounterRng() { delete _base; }

    /// Reads through shared's per-thread streams from now on; shared must outlive this object.
    void useStreams(const CounterStreams<cpu> * shared) { _streams = shared ? shared : &_own; }
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
        Stream * s = _streams->local();
        DAAL_CHECK_MALLOC(s && s->getState());
        // Skip of streamId * 2^66 + position words, as three 64-bit words.
        const DAAL_UINT64 skip[3] = { position, streamId << 2, streamId >> 62 };
        if (s->copyState(_base->getState()) || s->skipAheadEx(3, skip)) return services::Status(services::ErrorIncorrectErrorcodeFromGenerator);
        daal::internal::RNGsInst<unsigned int, cpu> rng;
        if (rng.uniformBits32(n, out, s->getState())) return services::Status(services::ErrorIncorrectErrorcodeFromGenerator);
        return services::Status();
    }

private:
    CounterStreams<cpu> _own;
    const CounterStreams<cpu> * _streams;
    Stream * _base = nullptr;
};

/// Unbiased integer in [0, range) from 32-bit words (Lemire, ACM TOMACS 2019), range > 0.
template <CpuType cpu, typename NextWord>
inline uint32_t boundedWord(uint32_t range, NextWord && next)
{
    uint64_t m = uint64_t(next()) * range;
    if (uint32_t(m) < range)
    {
        const uint32_t t = uint32_t(-range) % range;
        while (uint32_t(m) < t) m = uint64_t(next()) * range;
    }
    return uint32_t(m >> 32);
}

// Words read per CounterRng call for one node: a feature subset of sqrt(p) features and its random
// thresholds usually fit, and the read cost is mostly per call (design 4.13).
constexpr size_t nodeDrawBlock = 16;

// A node's feature draws use words [0, 2^62) of its stream and its splitter draws (random thresholds and
// categories) words from 2^62, so the two kinds do not depend on how a phase interleaves them: the
// histogram phase draws its features before any threshold, the exact phase one feature at a time.
constexpr uint64_t nodeThresholdWords = uint64_t(1) << 62;

// 2^53 and 2^26: a unit draw takes 26 + 27 bits from two words, as for a 53-bit double.
constexpr double twoPow53 = 9007199254740992.0;
constexpr double twoPow26 = 67108864.0;

/**
 *  \brief Draws of one node with philox4x32x10: word w is word w of stream nodeKey of the tree's CounterRng
 *
 *  The draws depend only on (treeSeed, nodeKey) and their order within the node.
 */
template <CpuType cpu>
class CounterNodeDraws
{
public:
    /// Draws from word firstWord of the node's stream; the threshold draws of a node start at nodeThresholdWords.
    CounterNodeDraws(const CounterRng<cpu> & rng, uint64_t nodeKey, uint64_t firstWord = 0) : _rng(rng), _nodeKey(nodeKey), _position(firstWord) {}

    /// Uniform integer in [0, range), range > 0.
    uint32_t below(uint32_t range)
    {
        return boundedWord<cpu>(range, [this]() { return next(); });
    }

    /// Uniform double in [0, 1).
    double unit()
    {
        const double hi = double(next() >> 6);
        const double lo = double(next() >> 5);
        return (hi * twoPow26 * 2 + lo) / twoPow53;
    }

    const services::Status & status() const { return _status; }

private:
    uint32_t next()
    {
        if (_used == _filled)
        {
            _status |= _rng.read(_nodeKey, _position, nodeDrawBlock, _words);
            _position += nodeDrawBlock;
            _filled = nodeDrawBlock;
            _used   = 0;
        }
        return _words[_used++];
    }

    const CounterRng<cpu> & _rng;
    uint64_t _nodeKey;
    uint64_t _position;
    size_t _used   = 0;
    size_t _filled = 0;
    uint32_t _words[nodeDrawBlock] {};
    services::Status _status;
};

/**
 *  \brief Draws of one node from a stream of the user's engine owned by the calling thread
 *
 *  For engines other than philox4x32x10 (design 4.13): draws are taken in execution order, so they
 *  depend on which thread builds the node and on what it drew before.
 */
template <CpuType cpu>
class StreamNodeDraws
{
public:
    explicit StreamNodeDraws(void * streamState) : _state(streamState) {}

    uint32_t below(uint32_t range)
    {
        int v = 0;
        daal::internal::RNGsInst<int, cpu> rng;
        if (rng.uniform(1, &v, _state, 0, int(range))) _status |= services::Status(services::ErrorIncorrectErrorcodeFromGenerator);
        return uint32_t(v);
    }

    double unit()
    {
        double v = 0;
        daal::internal::RNGsInst<double, cpu> rng;
        if (rng.uniform(1, &v, _state, 0.0, 1.0)) _status |= services::Status(services::ErrorIncorrectErrorcodeFromGenerator);
        return v < 1.0 ? v : 0.0;
    }

    const services::Status & status() const { return _status; }

private:
    void * _state;
    services::Status _status;
};

// A node's permutation is a full identity array, reset per node, below this many features or when the
// node may draw at least 1/8 of them; otherwise a table of the swapped positions, so a node with few
// draws over many features costs O(draws^2) with small draws rather than O(p).
constexpr size_t densePermutationFeatures  = 4096;
constexpr size_t densePermutationDrawShare = 8;

/**
 *  \brief The features of a node in draw order: a Fisher-Yates permutation of [0, p) drawn one element at a time
 *
 *  Draw j is uniform over the p - j features not yet drawn, so the first k draws are a uniform
 *  k-subset in a uniform order (design 4.13).
 */
template <CpuType cpu>
class FeaturePermutation
{
public:
    /// scratch holds 2 * nFeatures entries; expectedDraws selects the representation.
    FeaturePermutation(uint32_t nFeatures, size_t expectedDraws, uint32_t * scratch)
        : _p(nFeatures),
          _dense(nFeatures <= densePermutationFeatures || expectedDraws * densePermutationDrawShare >= nFeatures),
          _keys(scratch),
          _values(scratch + nFeatures)
    {
        if (_dense)
            for (uint32_t i = 0; i < _p; ++i) _keys[i] = i;
    }

    bool exhausted() const { return _drawn == _p; }

    template <typename Draws>
    uint32_t next(Draws & draws)
    {
        const uint32_t j    = _drawn++;
        const uint32_t pick = j + draws.below(_p - j);
        if (_dense)
        {
            const uint32_t v = _keys[pick];
            _keys[pick]      = _keys[j];
            _keys[j]         = v;
            return v;
        }
        const uint32_t v = lookup(pick);
        set(pick, lookup(j));
        return v;
    }

private:
    uint32_t lookup(uint32_t pos) const
    {
        for (size_t i = 0; i < _n; ++i)
            if (_keys[i] == pos) return _values[i];
        return pos;
    }
    void set(uint32_t pos, uint32_t value)
    {
        for (size_t i = 0; i < _n; ++i)
            if (_keys[i] == pos)
            {
                _values[i] = value;
                return;
            }
        _keys[_n]   = pos;
        _values[_n] = value;
        ++_n;
    }

    uint32_t _p;
    bool _dense;
    uint32_t * _keys;
    uint32_t * _values;
    size_t _n       = 0;
    uint32_t _drawn = 0;
};

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
