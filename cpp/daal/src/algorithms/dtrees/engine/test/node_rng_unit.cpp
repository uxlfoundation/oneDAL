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
#include "src/algorithms/dtrees/engine/node_rng.h"

namespace daal::algorithms::dtrees::engine::internal::test
{
constexpr auto cpu = DAAL_BASE_CPU;

/// Whether the RNG backend addresses philox4x32x10 by its full counter (MKL and OpenRNG do).
bool counterAddressable()
{
    CounterRng<cpu> rng;
    return rng.init(1).ok();
}

std::vector<uint32_t> words(const CounterRng<cpu> & rng, uint64_t streamId, uint64_t position, size_t n)
{
    std::vector<uint32_t> out(n);
    REQUIRE(rng.read(streamId, position, n, out.data()).ok());
    return out;
}

TEST("stream 0 reproduces engines::philox4x32x10::Batch", "[unit]")
{
    SKIP_IF(!counterAddressable());
    constexpr size_t n = 64;
    for (size_t seed : { size_t(0), size_t(777), size_t(0xdeadbeef) })
    {
        auto engine = engines::philox4x32x10::Batch<>::create(seed);
        auto impl   = dynamic_cast<engines::internal::BatchBaseImpl *>(engine.get());
        REQUIRE(impl);
        std::vector<uint32_t> expected(n);
        daal::internal::RNGsInst<unsigned int, cpu> rng;
        REQUIRE(rng.uniformBits32(n, expected.data(), impl->getState()) == 0);
        CounterRng<cpu> counter;
        REQUIRE(counter.init(seed).ok());
        REQUIRE(words(counter, 0, 0, n) == expected);
    }
}

TEST("a read at a position equals the same words of a read from 0", "[unit]")
{
    SKIP_IF(!counterAddressable());
    CounterRng<cpu> rng;
    REQUIRE(rng.init(0x0123456789abcdefull).ok());
    for (uint64_t streamId : { uint64_t(0), uint64_t(42), uint64_t(1) << 40, ~uint64_t(0) })
    {
        const auto all = words(rng, streamId, 0, 64);
        for (uint64_t position : { 0, 1, 3, 4, 5, 13 })
        {
            const auto part = words(rng, streamId, position, size_t(64 - position));
            REQUIRE(std::vector<uint32_t>(all.begin() + position, all.end()) == part);
        }
    }
}

TEST("key and stream id both select the sequence", "[unit]")
{
    SKIP_IF(!counterAddressable());
    CounterRng<cpu> a, b;
    REQUIRE(a.init(1).ok());
    REQUIRE(b.init(2).ok());
    const auto base = words(a, 1, 0, 16);
    REQUIRE(base != words(b, 1, 0, 16));
    REQUIRE(base != words(a, 2, 0, 16));
    REQUIRE(base != words(a, uint64_t(1) << 32, 0, 16));
    REQUIRE(base != words(a, uint64_t(1) << 63, 0, 16));
    REQUIRE(base == words(a, 1, 0, 16));
}

TEST("reads from many threads see the same words", "[unit]")
{
    SKIP_IF(!counterAddressable());
    CounterRng<cpu> rng;
    REQUIRE(rng.init(99).ok());
    constexpr size_t nStreams = 2000, n = 8;
    std::vector<uint32_t> serial(nStreams * n), parallel(nStreams * n);
    for (size_t i = 0; i < nStreams; ++i) REQUIRE(rng.read(mix64<cpu>(i), 0, n, serial.data() + i * n).ok());
    std::vector<int> failed(nStreams, 0);
    daal::threader_for(nStreams, 1, [&](size_t i) { failed[i] = !rng.read(mix64<cpu>(i), 0, n, parallel.data() + i * n).ok(); });
    for (int f : failed) REQUIRE(!f);
    REQUIRE(serial == parallel);
}

TEST("child node keys of a full tree are distinct", "[unit]")
{
    // Depth 16 holds 2^17 - 1 nodes.
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

TEST("draws from a stream that cannot be read end and report the error", "[unit]")
{
    CounterRng<cpu> rng; // never initialized: every read fails
    CounterNodeDraws<cpu> draws(rng, 1);
    REQUIRE(draws.below(3) < 3);
    REQUIRE(!draws.status().ok());
}

} // namespace daal::algorithms::dtrees::engine::internal::test
