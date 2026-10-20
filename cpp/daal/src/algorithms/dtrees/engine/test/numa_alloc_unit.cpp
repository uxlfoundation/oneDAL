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

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "data_management/data/homogen_numeric_table.h"
#include "src/algorithms/dtrees/engine/numa_alloc.h"

namespace daal::algorithms::dtrees::engine::internal::test
{
using namespace daal::data_management;
constexpr auto cpu = DAAL_BASE_CPU;

TEST("forEachBlock visits every block of every copy once", "[unit]")
{
    for (bool replicate : { false, true })
    {
        NumaReplicated<int, cpu> a;
        REQUIRE(a.allocate(1000, replicate).ok());
        REQUIRE(a.copies() == ((replicate && threader_numa_count() > 1) ? size_t(threader_numa_count()) : 1));
        constexpr size_t nBlocks = 37;
        for (size_t c = 0; c < a.copies(); ++c)
        {
            std::unique_ptr<std::atomic<int>[]> visits(new std::atomic<int>[nBlocks]);
            for (size_t b = 0; b < nBlocks; ++b) visits[b] = 0;
            REQUIRE(a.forEachBlock(c, nBlocks, [&](size_t b) {
                         ++visits[b];
                         return services::Status();
                     }).ok());
            for (size_t b = 0; b < nBlocks; ++b) REQUIRE(visits[b] == 1);
        }
    }
}

TEST("forEachBlock returns a block's error", "[unit]")
{
    NumaReplicated<int, cpu> a;
    REQUIRE(a.allocate(10, false).ok());
    REQUIRE(!a.forEachBlock(0, 8, [](size_t b) { return b == 5 ? services::Status(services::ErrorIncorrectParameter) : services::Status(); }).ok());
}

TEST("copyTableRows copies the table row-major into every copy", "[unit]")
{
    constexpr size_t nRows = 10000, nCols = 3;
    std::vector<double> data(nRows * nCols);
    for (size_t i = 0; i < data.size(); ++i) data[i] = double(i) * 0.5;
    NumericTablePtr x = HomogenNumericTable<double>::create(data.data(), nCols, nRows);
    for (bool replicate : { false, true })
    {
        NumaReplicated<float, cpu> copy;
        REQUIRE((copyTableRows<float, cpu>(*x, replicate, copy)).ok());
        for (size_t c = 0; c < copy.copies(); ++c)
        {
            const float * d = copy.data(int(c));
            for (size_t i = 0; i < data.size(); ++i) REQUIRE(d[i] == float(data[i]));
        }
    }
}

TEST("replicate copies copy 0 into the other copies", "[unit]")
{
    NumaReplicated<uint8_t, cpu> a;
    REQUIRE(a.allocate(5000, true).ok());
    uint8_t * src = a.data(0);
    for (size_t i = 0; i < a.size(); ++i) src[i] = uint8_t(i * 7);
    REQUIRE(a.replicate(1024).ok());
    for (size_t c = 0; c < a.copies(); ++c)
        for (size_t i = 0; i < a.size(); ++i) REQUIRE(a.data(int(c))[i] == uint8_t(i * 7));
}

} // namespace daal::algorithms::dtrees::engine::internal::test
