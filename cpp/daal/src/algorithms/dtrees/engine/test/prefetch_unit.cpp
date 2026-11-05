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
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "src/algorithms/dtrees/engine/prefetch.h"

namespace daal::algorithms::dtrees::engine::internal::test
{
constexpr auto cpu = DAAL_BASE_CPU;

TEST("prefetchedLoop processes every row once, in order, each prefetched once beforehand", "[unit]")
{
    for (size_t n : { 0, 1, 5, 10, 11, 100 })
    {
        for (size_t distance : { 0, 1, 10, 200 })
        {
            std::vector<size_t> processed;
            std::vector<size_t> prefetchCount(n, 0);
            std::vector<size_t> prefetchedBefore(n, 0); // rows processed when row i was prefetched
            prefetchedLoop<cpu>(
                n, distance,
                [&](size_t i) {
                    REQUIRE(i < n);
                    ++prefetchCount[i];
                    prefetchedBefore[i] = processed.size();
                },
                [&](size_t i) { processed.push_back(i); });

            REQUIRE(processed.size() == n);
            for (size_t i = 0; i < n; ++i)
            {
                REQUIRE(processed[i] == i);
                REQUIRE(prefetchCount[i] == (distance ? 1 : 0));
                // Row i is prefetched before it is processed and no more than distance rows ahead.
                if (distance) REQUIRE(prefetchedBefore[i] <= i);
                if (distance && i >= distance) REQUIRE(prefetchedBefore[i] == i - distance);
            }
        }
    }
}

TEST("PrefetchBudget keeps the lines in flight, not the rows, and skips passes that fit half of L2", "[unit]")
{
    constexpr size_t l2 = size_t(2) << 20;
    const PrefetchBudget<cpu> budget { 10, l2 };
    // One line per row (Higgs bins): 10 rows ahead, as the row-count rule gave.
    REQUIRE(budget.rows(1000000, 1) == 10);
    // About 70 lines per row (gisette, k = sqrt(5000) drawn features): one row ahead, about 70 lines in flight.
    REQUIRE(budget.rows(1000000, 70) == 1);
    REQUIRE(budget.rows(1000000, 3) == 3);
    // 10000 rows of one line fit half of L2; of 70 lines each they do not.
    REQUIRE(budget.rows(10000, 1) == 0);
    REQUIRE(budget.rows(10000, 70) == 1);
    REQUIRE(budget.rows(1000000, 0) == 10);
    REQUIRE((PrefetchBudget<cpu> { 0, l2 }).rows(1000000, 1) == 0);
}

TEST("prefetchRowLines accepts any line of a row", "[unit]")
{
    constexpr size_t nLines                                      = 4;
    alignas(cacheLineBytes) uint8_t row[nLines * cacheLineBytes] = {};
    const uint32_t lines[]                                       = { 0, 2, 3 };
    prefetchRowLines<cpu>(row, lines, 3);
    prefetchRowLines<cpu>(row, lines, 0);
    SUCCEED();
}

} // namespace daal::algorithms::dtrees::engine::internal::test
