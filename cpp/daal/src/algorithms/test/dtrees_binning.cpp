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
#include "data_management/data/homogen_numeric_table.h"
#include "src/algorithms/dtrees/dtrees_train_data_helper.i"

#include <limits>
#include <vector>

namespace daal::algorithms::dtrees::internal
{
// Every quantile bin border here falls inside a run of equal values; the bins must still keep those values together.
TEST("IndexedFeatures quantile bins do not split equal values", "[unit][dtrees]")
{
    const size_t nRows     = 2000;
    const size_t nDistinct = 100;
    std::vector<float> x(nRows);
    for (size_t i = 0; i < nRows; ++i)
    {
        x[i] = float((i * 37) % nDistinct);
    }
    auto nt = data_management::HomogenNumericTable<float>::create(x.data(), 1, nRows);
    REQUIRE(nt);

    IndexedFeatures indexed;
    const BinParams prm(8, 5);
    REQUIRE(indexed.init<float, DAAL_BASE_CPU>(*nt, nullptr, &prm).ok());

    const IndexedFeatures::IndexType * bins = indexed.data(0);
    std::vector<IndexedFeatures::IndexType> binOfValue(nDistinct, -1);
    for (size_t i = 0; i < nRows; ++i)
    {
        const size_t v = size_t(x[i]);
        if (binOfValue[v] < 0) binOfValue[v] = bins[i];
        INFO("row " << i << ", value " << v);
        REQUIRE(bins[i] == binOfValue[v]);
    }

    REQUIRE(indexed.numIndices(0) > 1);
    for (IndexedFeatures::IndexType k = 1; k < indexed.numIndices(0); ++k)
    {
        INFO("bin " << k);
        REQUIRE(indexed.binRightBorder(0, k - 1) < indexed.binRightBorder(0, k));
    }
}

// NaN keys must sort before every finite value and share one index; [1, 0, NaN, 0] was marked constant.
TEST("IndexedFeatures orders NaN before finite values", "[unit][dtrees]")
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<std::vector<float> > columns { { 1.f, 0.f, nan, 0.f }, { 1.f, 0.f, nan, 0.f, nan } };
    const BinParams prm(8, 1);
    for (const auto & column : columns)
    {
        std::vector<float> x(column);
        const size_t nRows = x.size();
        auto nt            = data_management::HomogenNumericTable<float>::create(x.data(), 1, nRows);
        REQUIRE(nt);

        for (const BinParams * pPrm : { static_cast<const BinParams *>(nullptr), &prm })
        {
            INFO("rows " << nRows << ", binning " << (pPrm ? "on" : "off"));
            IndexedFeatures indexed;
            REQUIRE(indexed.init<float, DAAL_BASE_CPU>(*nt, nullptr, pPrm).ok());

            const IndexedFeatures::IndexType * idx = indexed.data(0);
            REQUIRE(indexed.numIndices(0) == 3);
            for (size_t i = 0; i < nRows; ++i)
            {
                INFO("row " << i);
                const IndexedFeatures::IndexType expected = (x[i] != x[i]) ? 0 : IndexedFeatures::IndexType(x[i]) + 1;
                REQUIRE(idx[i] == expected);
            }
        }
    }
}

// Quantile and averages binning keep NaN rows together in the lowest bin.
TEST("IndexedFeatures bins NaN rows into the lowest bin", "[unit][dtrees]")
{
    const float nan    = std::numeric_limits<float>::quiet_NaN();
    const size_t nRows = 1000;
    std::vector<float> x(nRows);
    for (size_t i = 0; i < nRows; ++i)
    {
        x[i] = (i % 10 == 3) ? nan : float((i * 37) % 101);
    }
    auto nt = data_management::HomogenNumericTable<float>::create(x.data(), 1, nRows);
    REQUIRE(nt);

    for (const auto strategy : { BinningStrategy::quantiles, BinningStrategy::averages })
    {
        INFO("strategy " << int(strategy));
        IndexedFeatures indexed;
        const BinParams prm(16, 5, strategy);
        REQUIRE(indexed.init<float, DAAL_BASE_CPU>(*nt, nullptr, &prm).ok());

        const IndexedFeatures::IndexType * bins = indexed.data(0);
        const IndexedFeatures::IndexType nBins  = indexed.numIndices(0);
        REQUIRE(nBins > 1);
        REQUIRE(nBins <= 16);
        for (size_t i = 0; i < nRows; ++i)
        {
            INFO("row " << i);
            if (x[i] != x[i])
                REQUIRE(bins[i] == 0);
            else
                REQUIRE((bins[i] > 0 && bins[i] < nBins));
        }
    }
}
} // namespace daal::algorithms::dtrees::internal
