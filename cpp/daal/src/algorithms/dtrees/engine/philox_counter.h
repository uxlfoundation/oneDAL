/* file: philox_counter.h */
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
//  Counter-based philox4x32x10 draws for the decision tree engine. A draw is a
//  function of (key, stream, position), so it does not depend on the order in
//  which threads consume it.
//--
*/

#ifndef __DTREES_ENGINE_PHILOX_COUNTER_H__
#define __DTREES_ENGINE_PHILOX_COUNTER_H__

#include <cstddef>
#include <cstdint>

#include "src/services/service_defines.h"

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

// Round multipliers and Weyl key increments of Philox4x32 (Salmon et al., SC'11, Random123).
constexpr uint32_t philoxM0     = 0xD2511F53u;
constexpr uint32_t philoxM1     = 0xCD9E8D57u;
constexpr uint32_t philoxW0     = 0x9E3779B9u;
constexpr uint32_t philoxW1     = 0xBB67AE85u;
constexpr int philoxRounds      = 10;
constexpr size_t philoxBlockLen = 4;

// Golden-ratio increment and finalizer multipliers of splitmix64 (Steele et al., OOPSLA'14).
constexpr uint64_t splitmix64Gamma = 0x9E3779B97F4A7C15ull;
constexpr uint64_t splitmix64Mul0  = 0xBF58476D1CE4E5B9ull;
constexpr uint64_t splitmix64Mul1  = 0x94D049BB133111EBull;

/**
 *  \brief Computes one philox4x32x10 block: four 32-bit words for a 64-bit key and a 128-bit counter
 *
 *  \param key[in]     Key words, key[0] least significant
 *  \param counter[in] Counter words, counter[0] least significant
 *  \param out[out]    Output words, in the order engines::philox4x32x10 emits them
 */
template <CpuType cpu>
inline void philox4x32x10Block(const uint32_t key[2], const uint32_t counter[philoxBlockLen], uint32_t out[philoxBlockLen])
{
    uint32_t c0 = counter[0];
    uint32_t c1 = counter[1];
    uint32_t c2 = counter[2];
    uint32_t c3 = counter[3];
    uint32_t k0 = key[0];
    uint32_t k1 = key[1];
    for (int r = 0; r < philoxRounds; ++r)
    {
        const uint64_t p0 = uint64_t(philoxM0) * c0;
        const uint64_t p1 = uint64_t(philoxM1) * c2;
        c0                = uint32_t(p1 >> 32) ^ c1 ^ k0;
        c1                = uint32_t(p1);
        c2                = uint32_t(p0 >> 32) ^ c3 ^ k1;
        c3                = uint32_t(p0);
        k0 += philoxW0;
        k1 += philoxW1;
    }
    out[0] = c0;
    out[1] = c1;
    out[2] = c2;
    out[3] = c3;
}

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

/**
 *  \brief Sequence of 32-bit philox4x32x10 words addressed by (key, stream id, position)
 *
 *  Word w is lane w % 4 of the block whose counter holds w / 4 in its low 64 bits and the stream id in
 *  its high 64 bits. With key = seed and stream id 0 the sequence equals the output of
 *  engines::philox4x32x10::Batch(seed) on the MKL and OpenRNG backends.
 */
template <CpuType cpu>
class PhiloxCounterStream
{
public:
    PhiloxCounterStream(uint64_t key, uint64_t streamId, uint64_t position = 0)
        : _block(position / philoxBlockLen), _lane(size_t(position % philoxBlockLen))
    {
        _key[0]     = uint32_t(key);
        _key[1]     = uint32_t(key >> 32);
        _counter[2] = uint32_t(streamId);
        _counter[3] = uint32_t(streamId >> 32);
        refill();
    }

    uint32_t next()
    {
        if (_lane == philoxBlockLen)
        {
            ++_block;
            refill();
            _lane = 0;
        }
        return _words[_lane++];
    }

    /// Index of the word the next call to next() returns.
    uint64_t position() const { return _block * philoxBlockLen + _lane; }

private:
    void refill()
    {
        _counter[0] = uint32_t(_block);
        _counter[1] = uint32_t(_block >> 32);
        philox4x32x10Block<cpu>(_key, _counter, _words);
    }

    uint32_t _key[2];
    uint32_t _counter[philoxBlockLen];
    uint32_t _words[philoxBlockLen];
    uint64_t _block;
    size_t _lane;
};

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
