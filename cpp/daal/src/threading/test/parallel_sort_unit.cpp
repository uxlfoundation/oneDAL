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

/// Unit tests for `daal::parallel_sort(n, data, index)`, which must behave like the indexed
/// `qSort`: the (data[k], index[k]) pairs are reordered so that `data` is non-decreasing.

#include <algorithm>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "src/threading/threading.h"

namespace daal::test
{
enum class IndexKind
{
    identity,     // 0, 1, ..., n - 1
    shuffled,     // a permutation of 0, ..., n - 1
    sparseRowIds, // distinct row ids from a larger range, as decision forest passes them
};

template <typename FPType>
void checkIndexedSort(std::size_t n, IndexKind kind, int valueRange, unsigned seed)
{
    std::mt19937 gen(seed);
    std::vector<FPType> data(n);
    std::vector<int> index(n);

    // valueRange > 0 draws integers from [0, valueRange) to force ties.
    std::uniform_real_distribution<FPType> realDist(-1, 1);
    std::uniform_int_distribution<int> intDist(0, valueRange > 0 ? valueRange - 1 : 0);
    for (std::size_t i = 0; i < n; ++i)
    {
        data[i] = valueRange > 0 ? FPType(intDist(gen)) : realDist(gen);
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        index[i] = kind == IndexKind::sparseRowIds ? int(3 * i + 7) : int(i);
    }
    if (kind != IndexKind::identity)
    {
        std::shuffle(index.begin(), index.end(), gen);
    }

    std::vector<std::pair<FPType, int> > expected(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        expected[i] = { data[i], index[i] };
    }

    daal::parallel_sort<FPType, int>(n, data.data(), index.data());

    for (std::size_t i = 1; i < n; ++i)
    {
        INFO("n = " << n << ", position " << i);
        REQUIRE(data[i - 1] <= data[i]);
    }

    // The sort need not be stable, so compare the pairs as multisets.
    std::vector<std::pair<FPType, int> > actual(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        actual[i] = { data[i], index[i] };
    }
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    REQUIRE(actual == expected);
}

TEMPLATE_TEST("parallel_sort keeps data and index paired", "[unit][parallel_sort]", float, double)
{
    const std::size_t n  = GENERATE(0, 1, 2, 7, 16, 100, 1000, 10000, 100000);
    const IndexKind kind = GENERATE(IndexKind::identity, IndexKind::shuffled, IndexKind::sparseRowIds);
    const int valueRange = GENERATE(0, 1, 10);
    checkIndexedSort<TestType>(n, kind, valueRange, 777u + unsigned(n));
}

TEMPLATE_TEST("parallel_sort on already sorted and reversed data", "[unit][parallel_sort]", float, double)
{
    const std::size_t n = GENERATE(2, 100, 10000);
    const bool reversed = GENERATE(false, true);

    std::vector<TestType> data(n);
    std::vector<int> index(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        data[i]  = TestType(reversed ? n - i : i);
        index[i] = int(5 * i + 1);
    }

    daal::parallel_sort<TestType, int>(n, data.data(), index.data());

    for (std::size_t i = 0; i < n; ++i)
    {
        INFO("n = " << n << ", position " << i);
        REQUIRE(data[i] == TestType(reversed ? i + 1 : i));
        REQUIRE(index[i] == int(5 * (reversed ? n - 1 - i : i) + 1));
    }
}
} // namespace daal::test
