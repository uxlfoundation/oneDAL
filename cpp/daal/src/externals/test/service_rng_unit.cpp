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

/// Unit tests for the multi-word skip-ahead, the in-place state copy and the draw accounting of the RNG backends.

#include <cstdint>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "src/externals/service_rng.h"

namespace daal::test
{
constexpr auto cpu = DAAL_BASE_CPU;
using Stream       = daal::internal::BaseRNGsInst<cpu>;
using Bits         = daal::internal::RNGsInst<unsigned int, cpu>;

std::vector<unsigned int> draw(Stream & s, size_t n)
{
    std::vector<unsigned int> out(n);
    Bits rng;
    REQUIRE(rng.uniformBits32(n, out.data(), s.getState()) == 0);
    return out;
}

/// Whether the backend generates raw bits and skips past 2^64 outputs: MKL and OpenRNG do, the reference
/// backend does neither.
bool fullSkipSupported()
{
    const unsigned int seed[2] = { 1, 0 };
    Stream s(2, seed, __DAAL_BRNG_PHILOX4X32X10);
    const DAAL_UINT64 skip[2] = { 0, 1 };
    unsigned int out;
    Bits rng;
    return s.skipAheadEx(2, skip) == 0 && rng.uniformBits32(1, &out, s.getState()) == 0;
}

TEST("philox4x32x10 with a two-word key and a three-word skip matches the Random123 known-answer vectors", "[unit]")
{
    SKIP_IF(!fullSkipSupported());
    struct Kat
    {
        unsigned int key[2];
        uint32_t counter[4];
        uint32_t expected[4];
    };
    // kat_vectors from the Random123 distribution, philox4x32 with 10 rounds.
    const Kat kats[] = {
        { { 0x00000000, 0x00000000 }, { 0x00000000, 0x00000000, 0x00000000, 0x00000000 }, { 0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8 } },
        { { 0xffffffff, 0xffffffff }, { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff }, { 0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd } },
        { { 0xa4093822, 0x299f31d0 }, { 0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344 }, { 0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1 } },
    };
    for (const auto & kat : kats)
    {
        Stream s(2, kat.key, __DAAL_BRNG_PHILOX4X32X10);
        // The block at counter c starts at output 4 c, a 130-bit number in three 64-bit words.
        const uint64_t lo         = uint64_t(kat.counter[0]) | (uint64_t(kat.counter[1]) << 32);
        const uint64_t hi         = uint64_t(kat.counter[2]) | (uint64_t(kat.counter[3]) << 32);
        const DAAL_UINT64 skip[3] = { lo << 2, (hi << 2) | (lo >> 62), hi >> 62 };
        REQUIRE(s.skipAheadEx(3, skip) == 0);
        const auto out = draw(s, 4);
        for (size_t j = 0; j < 4; ++j) REQUIRE(out[j] == kat.expected[j]);
    }
}

TEST("a one-word key and a two-word key with a zero high word seed philox4x32x10 alike", "[unit]")
{
    SKIP_IF(!fullSkipSupported());
    const unsigned int words[2] = { 777, 0 };
    Stream one(777, __DAAL_BRNG_PHILOX4X32X10);
    Stream two(2, words, __DAAL_BRNG_PHILOX4X32X10);
    REQUIRE(draw(one, 64) == draw(two, 64));
}

TEST("skipAheadEx with one word equals skipAhead", "[unit]")
{
    for (const int brng : { __DAAL_BRNG_PHILOX4X32X10, __DAAL_BRNG_MT19937, __DAAL_BRNG_MCG59, __DAAL_BRNG_MRG32K3A })
    {
        INFO("brng " << brng);
        Stream a(5, brng);
        Stream b(5, brng);
        for (const size_t n : { size_t(0), size_t(1), size_t(4), size_t(1000) })
        {
            REQUIRE(a.skipAhead(n) == 0);
            const DAAL_UINT64 skip[1] = { n };
            REQUIRE(b.skipAheadEx(1, skip) == 0);
            std::vector<int> x(16), y(16);
            daal::internal::RNGsInst<int, cpu> rng;
            REQUIRE(rng.uniform(16, x.data(), a.getState(), 0, 1 << 30) == 0);
            REQUIRE(rng.uniform(16, y.data(), b.getState(), 0, 1 << 30) == 0);
            REQUIRE(x == y);
        }
    }
}

TEST("skips past 2^64 succeed or report the skip-ahead error code", "[unit]")
{
    const unsigned int seed[2] = { 3, 0 };
    Stream s(2, seed, __DAAL_BRNG_PHILOX4X32X10);
    const DAAL_UINT64 skip[3] = { 7, 1, 0 };
    const int errcode         = s.skipAheadEx(3, skip);
    REQUIRE((errcode == 0 || errcode == __DAAL_SKIP_AHEAD_METHOD_ERRCODE));
}

TEST("copyState continues the source stream without allocating a new one", "[unit]")
{
    for (const int brng : { __DAAL_BRNG_PHILOX4X32X10, __DAAL_BRNG_MT19937, __DAAL_BRNG_MCG59 })
    {
        Stream a(11, brng);
        Stream b(11, brng);
        std::vector<int> skipped(100), fromA(16), fromB(16);
        daal::internal::RNGsInst<int, cpu> rng;
        REQUIRE(rng.uniform(skipped.size(), skipped.data(), a.getState(), 0, 1 << 30) == 0);
        void * const bState = b.getState();
        REQUIRE(b.copyState(a.getState()) == 0);
        REQUIRE(b.getState() == bState);
        REQUIRE(rng.uniform(fromA.size(), fromA.data(), a.getState(), 0, 1 << 30) == 0);
        REQUIRE(rng.uniform(fromB.size(), fromB.data(), b.getState(), 0, 1 << 30) == 0);
        REQUIRE(fromA == fromB);
    }
}

/// The next k integers in [0, 1 << 30) of s.
std::vector<int> nextInts(Stream & s, size_t k)
{
    std::vector<int> out(k);
    daal::internal::RNGsInst<int, cpu> rng;
    REQUIRE(rng.uniform(k, out.data(), s.getState(), 0, 1 << 30) == 0);
    return out;
}

TEST("a copy of a used stream continues it", "[unit]")
{
    // Doubles, Gaussians and Bernoulli draws can take more than one engine value each; the copy must follow.
    for (const int brng : { __DAAL_BRNG_MT19937, __DAAL_BRNG_MCG59, __DAAL_BRNG_PHILOX4X32X10 })
    {
        CAPTURE(brng);
        Stream a(5, brng);
        std::vector<double> d(1000);
        std::vector<int> b(1000);
        daal::internal::RNGsInst<double, cpu> real;
        daal::internal::RNGsInst<int, cpu> integer;
        REQUIRE(real.uniform(d.size(), d.data(), a.getState(), 0.0, 1.0) == 0);
        REQUIRE(real.gaussian(d.size(), d.data(), a.getState(), 0.0, 1.0) == 0);
        REQUIRE(integer.bernoulli(b.size(), b.data(), a.getState(), 0.3) == 0);
        Stream copy(a);
        REQUIRE(nextInts(copy, 16) == nextInts(a, 16));
    }
}

TEST("integer draws take one engine value each", "[unit]")
{
    // The forest's stream budgets count one value per integer (df_forest_trainer.i).
    for (const int brng : { __DAAL_BRNG_MT19937, __DAAL_BRNG_MCG59, __DAAL_BRNG_PHILOX4X32X10 })
        for (const int range : { 11000000, 1 << 30, 7 })
        {
            CAPTURE(brng, range);
            Stream a(9, brng);
            std::vector<int> draws(100000);
            daal::internal::RNGsInst<int, cpu> rng;
            REQUIRE(rng.uniform(draws.size(), draws.data(), a.getState(), 0, range) == 0);
            for (int v : draws) REQUIRE((v >= 0 && v < range));
            Stream skipped(9, brng);
            REQUIRE(skipped.skipAhead(draws.size()) == 0);
            REQUIRE(nextInts(skipped, 16) == nextInts(a, 16));
        }
}

} // namespace daal::test
