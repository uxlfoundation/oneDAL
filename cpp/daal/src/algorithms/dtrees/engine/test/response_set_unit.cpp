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

#include <cmath>
#include <cstdint>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "data_management/data/homogen_numeric_table.h"
#include "src/algorithms/dtrees/engine/response_set.h"

namespace daal::algorithms::dtrees::engine::internal::test
{
using namespace daal::data_management;
constexpr auto cpu = DAAL_BASE_CPU;

NumericTablePtr column(std::vector<double> & v)
{
    return HomogenNumericTable<double>::create(v.data(), 1, v.size());
}

TEST("row counts above 2^31 - 1 are rejected", "[unit]")
{
    REQUIRE(checkRowCount(maxTrainingRows).ok());
    REQUIRE(!checkRowCount(maxTrainingRows + 1).ok());
}

TEST("unweighted classification records keep in-bag rows in order with their counts", "[unit]")
{
    std::vector<double> y = { 0, 1, 2, 1, 0 };
    ResponseSource<double, cpu> src;
    REQUIRE(src.build(*column(y), nullptr, true, NumaPlacement()).ok());
    REQUIRE(!src.weighted());

    const std::vector<uint16_t> counts = { 2, 0, 1, 3, 0 };
    ResponseSet<ClsCountRecord, cpu> set;
    REQUIRE(set.allocate(y.size()).ok());
    double total = 0;
    REQUIRE(set.fill(counts.data(), src, 0, total).ok());
    REQUIRE(set.size() == 3);
    REQUIRE(total == 6);
    const ClsCountRecord * r = set.buffer(0);
    REQUIRE((r[0].row == 0 && r[0].cls == 0 && r[0].count == 2));
    REQUIRE((r[1].row == 2 && r[1].cls == 2 && r[1].count == 1));
    REQUIRE((r[2].row == 3 && r[2].cls == 1 && r[2].count == 3));
    REQUIRE(set.buffer(2) == set.buffer(0));
    REQUIRE(set.buffer(1) != set.buffer(0));

    // Without bootstrap every row is in the bag once.
    REQUIRE(set.fill(nullptr, src, 0, total).ok());
    REQUIRE(set.size() == y.size());
    REQUIRE(total == double(y.size()));
}

TEST("weights are normalized to a maximum of 1, scaled by the count, and zero weights dropped", "[unit]")
{
    std::vector<double> y = { 1.5, -2, 3, 4 };
    std::vector<double> w = { 2, 0, 4, 1 };
    ResponseSource<double, cpu> src;
    REQUIRE(src.build(*column(y), column(w).get(), false, NumaPlacement()).ok());
    REQUIRE(src.weighted());

    const std::vector<uint16_t> counts = { 1, 5, 2, 0 };
    ResponseSet<RegWeightRecord<double>, cpu> set;
    REQUIRE(set.allocate(y.size()).ok());
    double total = 0;
    REQUIRE(set.fill(counts.data(), src, 0, total).ok());
    REQUIRE(set.size() == 2);
    const auto * r = set.buffer(0);
    REQUIRE((r[0].row == 0 && r[0].y == 1.5 && r[0].w == 0.5));
    REQUIRE((r[1].row == 2 && r[1].y == 3 && r[1].w == 2.0));
    REQUIRE(total == 2.5);
}

TEST("the mean response is weighted by the normalized weights", "[unit]")
{
    // Weights normalize to { 0.5, 0, 1, 0.25 }: (1.5 * 0.5 + 3 + 4 * 0.25) / 1.75.
    std::vector<double> y = { 1.5, -2, 3, 4 };
    std::vector<double> w = { 2, 0, 4, 1 };
    ResponseSource<double, cpu> src;
    REQUIRE(src.build(*column(y), column(w).get(), false, NumaPlacement()).ok());
    REQUIRE(std::abs(src.responseMean() - 4.75 / 1.75) < 1e-15);
    REQUIRE(src.build(*column(y), nullptr, false, NumaPlacement()).ok());
    REQUIRE(src.responseMean() == 6.5 / 4);
}

TEST("all-zero weights make the fit unweighted, as legacy", "[unit]")
{
    std::vector<double> y = { 0, 1 };
    std::vector<double> w = { 0, 0 };
    ResponseSource<double, cpu> src;
    REQUIRE(src.build(*column(y), column(w).get(), true, NumaPlacement()).ok());
    REQUIRE(!src.weighted());
}

TEST("filling more records than allocated fails", "[unit]")
{
    std::vector<double> y = { 0, 1, 1 };
    ResponseSource<double, cpu> src;
    REQUIRE(src.build(*column(y), nullptr, true, NumaPlacement()).ok());
    ResponseSet<ClsCountRecord, cpu> set;
    REQUIRE(set.allocate(2).ok());
    double total = 0;
    REQUIRE(!set.fill(nullptr, src, 0, total).ok());
}

} // namespace daal::algorithms::dtrees::engine::internal::test
