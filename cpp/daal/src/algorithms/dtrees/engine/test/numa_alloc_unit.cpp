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

TEST("forEachBlock visits every block once on one node and spread over the host", "[unit]")
{
    for (const NumaPlacement & placement : { NumaPlacement(), NumaPlacement::host<cpu>() })
    {
        NumaArray<int, cpu> a;
        REQUIRE(a.allocate(1000, placement).ok());
        REQUIRE(a.spansNodes() == (placement.nodes > 1));
        constexpr size_t nBlocks = 37;
        std::unique_ptr<std::atomic<int>[]> visits(new std::atomic<int>[nBlocks]);
        for (size_t b = 0; b < nBlocks; ++b) visits[b] = 0;
        REQUIRE(a.forEachBlock(nBlocks, [&](size_t b) {
                     ++visits[b];
                     return services::Status();
                 }).ok());
        for (size_t b = 0; b < nBlocks; ++b) REQUIRE(visits[b] == 1);
    }
}

TEST("forEachBlock returns a block's error", "[unit]")
{
    NumaArray<int, cpu> a;
    REQUIRE(a.allocate(10, NumaPlacement()).ok());
    REQUIRE(!a.forEachBlock(8, [](size_t b) { return b == 5 ? services::Status(services::ErrorIncorrectParameter) : services::Status(); }).ok());
}

TEST("forEachTile visits every tile of every block once", "[unit]")
{
    for (const NumaPlacement & placement : { NumaPlacement(), NumaPlacement::host<cpu>() })
    {
        NumaArray<int, cpu> a;
        REQUIRE(a.allocate(1000, placement).ok());
        constexpr size_t nBlocks = 37, nTiles = 5;
        std::unique_ptr<std::atomic<int>[]> visits(new std::atomic<int>[nBlocks * nTiles]);
        for (size_t i = 0; i < nBlocks * nTiles; ++i) visits[i] = 0;
        REQUIRE(a.forEachTile(nBlocks, nTiles, [&](size_t b, size_t t) {
                     ++visits[b * nTiles + t];
                     return services::Status();
                 }).ok());
        for (size_t i = 0; i < nBlocks * nTiles; ++i) REQUIRE(visits[i] == 1);
    }
}

TEST("passBlockRows splits short tables over the threads and keeps tall ones at tableBlockRows", "[unit]")
{
    // gisette (6000 x 5000) binned on 224 threads: 79 tiles of 64 u8 bins per block, rows of 5056 bytes.
    const size_t binRows = passBlockRows<cpu>(6000, 224, 79, 64 * sizeof(double), 5056, false);
    REQUIRE((6000 + binRows - 1) / binRows * 79 >= 224 * passTasksPerThread);
    // Its X copy spread over two nodes: whole pages per block.
    const size_t xRows = passBlockRows<cpu>(6000, 224, 1, 5000 * sizeof(double), 5000 * sizeof(double), true);
    REQUIRE(xRows * 5000 * sizeof(double) % pageBytes == 0);
    REQUIRE((6000 + xRows - 1) / xRows >= 64);
    // Higgs (10.5M x 28) has blocks enough at tableBlockRows.
    REQUIRE(passBlockRows<cpu>(10500000, 224, 1, 28 * sizeof(double), 32, false) == tableBlockRows);
    // A task reads at least minTaskBytes when the table has them.
    REQUIRE(passBlockRows<cpu>(100000, 224, 1, 16, 16, false) * 16 >= minTaskBytes);

    for (size_t nRows : { 0, 1, 100, 6000, 1000000 })
        for (size_t nThreads : { 0, 1, 7, 224 })
            for (size_t writeRowBytes : { 1, 2, 24, 64, 5056, 40000 })
                for (bool pageAligned : { false, true })
                {
                    CAPTURE(nRows, nThreads, writeRowBytes, pageAligned);
                    const size_t rows = passBlockRows<cpu>(nRows, nThreads, 3, writeRowBytes, writeRowBytes, pageAligned);
                    REQUIRE(rows >= 1);
                    REQUIRE(rows <= tableBlockRows);
                    REQUIRE(rows * writeRowBytes % (pageAligned ? pageBytes : cacheLineBytes) == 0);
                }
}

TEST("copyTableRows copies a table of another type row-major, on one node and spread over the host", "[unit]")
{
    constexpr size_t nRows = 10000, nCols = 3;
    std::vector<double> data(nRows * nCols);
    for (size_t i = 0; i < data.size(); ++i) data[i] = double(i) * 0.5;
    NumericTablePtr x = HomogenNumericTable<double>::create(data.data(), nCols, nRows);
    for (const NumaPlacement & placement : { NumaPlacement(), NumaPlacement::host<cpu>() })
    {
        NumaArray<float, cpu> copy;
        REQUIRE((copyTableRows<float, cpu>(*x, placement, copy)).ok());
        REQUIRE(!copy.borrowed());
        const float * d = copy.data();
        for (size_t i = 0; i < data.size(); ++i) REQUIRE(d[i] == float(data[i]));
    }
}

TEST("copyTableRows borrows a homogen table of the training type on any placement", "[unit]")
{
    constexpr size_t nRows = 1000, nCols = 3;
    std::vector<float> data(nRows * nCols);
    for (size_t i = 0; i < data.size(); ++i) data[i] = float(i) * 0.5f;
    NumericTablePtr x = HomogenNumericTable<float>::create(data.data(), nCols, nRows);
    for (const NumaPlacement & placement : { NumaPlacement(), NumaPlacement { 2 } })
    {
        NumaArray<float, cpu> one;
        REQUIRE((copyTableRows<float, cpu>(*x, placement, one)).ok());
        REQUIRE(one.borrowed());
        const NumaArray<float, cpu> & view = one;
        REQUIRE(view.data() == data.data());
    }
}

} // namespace daal::algorithms::dtrees::engine::internal::test
