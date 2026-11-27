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

#include "oneapi/dal/test/engine/common.hpp"
#include "src/algorithms/service_sort.h"

#include <random>
#include <vector>

namespace daal::algorithms::internal
{
TEST("Indexed qSort", "[unit]")
{
    const size_t n = GENERATE(100, 20000000);
    std::vector<float> x(n);
    std::vector<int> idx(n);
    std::mt19937 gen(777);
    std::uniform_real_distribution<float> dist(0, 1);
    for (size_t i = 0; i < n; ++i)
    {
        x[i]   = dist(gen);
        idx[i] = int(i);
    }
    qSort<float, int, DAAL_BASE_CPU>(n, x.data(), idx.data());
    size_t bad = 0;
    for (size_t i = 1; i < n; ++i) bad += x[i - 1] > x[i];
    REQUIRE(bad == 0);
}

TEST("Indexed qSort of up to two elements", "[unit]")
{
    std::vector<float> x { 2.f, 1.f };
    std::vector<int> idx { 10, 20 };
    qSort<float, int, DAAL_BASE_CPU>(2, x.data(), idx.data());
    REQUIRE(x == std::vector<float> { 1.f, 2.f });
    REQUIRE(idx == std::vector<int> { 20, 10 });

    qSort<float, int, DAAL_BASE_CPU>(2, x.data(), idx.data());
    REQUIRE(x == std::vector<float> { 1.f, 2.f });
    REQUIRE(idx == std::vector<int> { 20, 10 });

    qSort<float, int, DAAL_BASE_CPU>(1, x.data() + 1, idx.data() + 1);
    qSort<float, int, DAAL_BASE_CPU>(0, x.data(), idx.data());
    REQUIRE(x == std::vector<float> { 1.f, 2.f });
    REQUIRE(idx == std::vector<int> { 20, 10 });
}
} // namespace daal::algorithms::internal
