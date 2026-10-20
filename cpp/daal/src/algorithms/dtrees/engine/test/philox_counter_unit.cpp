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

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "algorithms/engines/philox4x32x10/philox4x32x10.h"
#include "src/algorithms/engines/engine_batch_impl.h"
#include "src/externals/service_rng.h"
#include "src/algorithms/dtrees/engine/philox_counter.h"

namespace daal::algorithms::dtrees::engine::internal::test
{
constexpr auto cpu = DAAL_BASE_CPU;

// Reads n raw 32-bit words from a DAAL philox4x32x10 engine. Returns false where the RNG backend
// cannot produce raw bits (the reference backend, which is not philox).
bool batchWords(size_t seed, size_t skip, size_t n, std::vector<uint32_t> & out)
{
    auto engine = engines::philox4x32x10::Batch<>::create(seed);
    if (skip) REQUIRE(engine->skipAhead(skip).ok());
    auto impl = dynamic_cast<engines::internal::BatchBaseImpl *>(engine.get());
    REQUIRE(impl);
    out.resize(n);
    daal::internal::RNGsInst<unsigned int, cpu> rng;
    return rng.uniformBits32(n, out.data(), impl->getState()) == 0;
}

std::vector<uint32_t> streamWords(uint64_t key, uint64_t streamId, uint64_t position, size_t n)
{
    PhiloxCounterStream<cpu> stream(key, streamId, position);
    std::vector<uint32_t> out(n);
    for (auto & w : out) w = stream.next();
    return out;
}

TEST("philox4x32x10 block matches the Random123 known-answer vectors", "[unit]")
{
    struct Kat
    {
        uint32_t key[2];
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
        uint32_t out[4];
        philox4x32x10Block<cpu>(kat.key, kat.counter, out);
        for (size_t j = 0; j < 4; ++j) REQUIRE(out[j] == kat.expected[j]);
    }
}

TEST("stream 0 reproduces engines::philox4x32x10::Batch", "[unit]")
{
    constexpr size_t n = 64;
    for (size_t seed : { size_t(0), size_t(777), size_t(0xdeadbeef) })
    {
        std::vector<uint32_t> expected;
        SKIP_IF(!batchWords(seed, 0, n, expected));
        REQUIRE(streamWords(seed, 0, 0, n) == expected);

        // Past 2^32 blocks the counter carries into its second word; the position is mid-block.
        const uint64_t skip = (uint64_t(1) << 35) + 2;
        REQUIRE(batchWords(seed, skip, n, expected));
        REQUIRE(streamWords(seed, 0, skip, n) == expected);
    }
}

TEST("a stream started at a position equals a stream advanced to it", "[unit]")
{
    constexpr uint64_t key               = 0x0123456789abcdefull;
    constexpr uint64_t streamId          = 42;
    const std::vector<uint32_t> fromZero = streamWords(key, streamId, 0, 32);
    for (uint64_t position : { 0, 1, 3, 4, 5, 13 })
    {
        PhiloxCounterStream<cpu> stream(key, streamId, position);
        REQUIRE(stream.position() == position);
        for (uint64_t w = position; w < fromZero.size(); ++w) REQUIRE(stream.next() == fromZero[w]);
        REQUIRE(stream.position() == fromZero.size());
    }
}

TEST("key and stream id both select the sequence", "[unit]")
{
    constexpr size_t n  = 16;
    const auto base     = streamWords(1, 1, 0, n);
    const auto otherKey = streamWords(2, 1, 0, n);
    const auto otherId  = streamWords(1, 2, 0, n);
    const auto highId   = streamWords(1, uint64_t(1) << 32, 0, n);
    REQUIRE(base != otherKey);
    REQUIRE(base != otherId);
    REQUIRE(base != highId);
    REQUIRE(base == streamWords(1, 1, 0, n));
}

TEST("child node keys of a full tree are distinct", "[unit]")
{
    // Depth 16 holds 2^17 - 1 nodes, more than any node count a collision test could reach otherwise.
    constexpr size_t depth = 16;
    std::unordered_set<uint64_t> seen;
    std::vector<uint64_t> level = { 777 };
    seen.insert(level[0]);
    for (size_t d = 0; d < depth; ++d)
    {
        std::vector<uint64_t> next;
        next.reserve(level.size() * 2);
        for (uint64_t parent : level)
        {
            for (size_t side = 0; side < 2; ++side)
            {
                const uint64_t child = childNodeKey<cpu>(parent, side);
                REQUIRE(child == childNodeKey<cpu>(parent, side));
                REQUIRE(seen.insert(child).second);
                next.push_back(child);
            }
        }
        level.swap(next);
    }
    REQUIRE(seen.size() == (size_t(1) << (depth + 1)) - 1);
}

} // namespace daal::algorithms::dtrees::engine::internal::test
