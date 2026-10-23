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
#include <type_traits>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "data_management/data/homogen_numeric_table.h"
#include "src/algorithms/dtrees/engine/phase_hist.i"
#include "src/algorithms/dtrees/engine/quantile_binner.h"

namespace daal::algorithms::dtrees::engine::internal::test
{
using namespace daal::data_management;

constexpr auto cpu = DAAL_BASE_CPU;

// Rows alternate between the two values of one feature; within a bin the weights alternate between 1 and 1e-3,
// which float sums round from the second add on.
constexpr size_t nRows = 16384;

inline double weightOf(size_t i)
{
    return (i / 2) % 2 ? 1e-3 : 1.;
}

/// The bins of one feature holding i % 2 in row i.
void binTwoValues(BinMatrix<cpu> & bins)
{
    std::vector<double> x(nRows);
    for (size_t i = 0; i < nRows; ++i) x[i] = double(i % 2);
    NumericTablePtr t = HomogenNumericTable<double>::create(x.data(), 1, nRows);
    bool binned       = false;
    REQUIRE((QuantileBinner<double, cpu>::build(*t, BinningParams(), nullptr, bins, binned)).ok());
    REQUIRE(binned);
    REQUIRE(bins.nBins(0) == 2);
}

/// Each bin's weight sum, in row order.
std::vector<double> exactSums()
{
    std::vector<double> sums(2, 0);
    for (size_t i = 0; i < nRows; ++i) sums[i % 2] += weightOf(i);
    return sums;
}

TEST("weighted histograms sum weights in double", "[unit]")
{
    BinMatrix<cpu> bins;
    binTwoValues(bins);
    const uint32_t lane         = 0;
    const std::vector<double> e = exactSums();
    for (size_t nChunks : { size_t(1), size_t(4) })
    {
        CAPTURE(nChunks);
        // Chunks add in another order; the bound is far below float rounding (about 1e-5 here).
        const double tol = nChunks == 1 ? 0 : 1e-12;
        std::vector<RegWeightRecord<double> > reg(nRows);
        std::vector<ClsWeightRecord<double> > cls(nRows);
        for (size_t i = 0; i < nRows; ++i)
        {
            reg[i]     = RegWeightRecord<double> {};
            reg[i].row = uint32_t(i);
            reg[i].y   = 100;
            reg[i].w   = weightOf(i);
            cls[i]     = { uint32_t(i), 0, weightOf(i) };
        }
        hist::Planes<RegWeightRecord<double>, cpu> hr;
        hist::Planes<ClsWeightRecord<double>, cpu> hc;
        REQUIRE((hist::build(reg.data(), nRows, bins, 0, &lane, 1, 0, nChunks, 0, hr)).ok());
        REQUIRE((hist::build(cls.data(), nRows, bins, 0, &lane, 1, 1, nChunks, 0, hc)).ok());
        for (size_t b = 0; b < 2; ++b)
        {
            REQUIRE(std::abs(double(hr.stat[hr.at(b, 0)]) - e[b]) <= tol * e[b]);
            REQUIRE(std::abs(double(hc.classStat(0, b, 0)) - e[b]) <= tol * e[b]);
        }
    }
}

TEST("count histograms sum counts as integers", "[unit]")
{
    REQUIRE(std::is_integral<hist::Layout<ClsCountRecord>::Stat>::value);
    REQUIRE(std::is_integral<hist::Layout<RegCountRecord<float> >::Stat>::value);
    REQUIRE(std::is_integral<hist::Layout<RegCountRecord<double> >::Stat>::value);
}

} // namespace daal::algorithms::dtrees::engine::internal::test
