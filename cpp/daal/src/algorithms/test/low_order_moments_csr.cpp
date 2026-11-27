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
#include "algorithms/moments/low_order_moments_batch.h"
#include "data_management/data/csr_numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

#include <cmath>
#include <random>
#include <vector>

namespace daal::algorithms::low_order_moments
{
using namespace daal::data_management;

template <typename FPType, Method method>
static void checkCsrMoments()
{
    // Columns far from zero, where a variance taken from the raw sum of squares cancels, and
    // enough rows for several chunks; about a third of the entries are implicit zeros.
    const size_t nRows     = 6000;
    const size_t nFeatures = 7;
    std::mt19937 gen(777);
    std::uniform_real_distribution<double> noise(-1.0, 1.0);
    std::uniform_real_distribution<double> keep(0.0, 1.0);

    std::vector<FPType> values;
    std::vector<size_t> colIndices;
    std::vector<size_t> rowOffsets(nRows + 1);
    std::vector<double> dense(nRows * nFeatures, 0.0);
    for (size_t i = 0; i < nRows; ++i)
    {
        rowOffsets[i] = values.size() + 1;
        for (size_t j = 0; j < nFeatures; ++j)
        {
            if (j > 0 && keep(gen) < 0.35) continue;
            const FPType x = FPType(100.0 * double(j) + noise(gen));
            if (x == FPType(0)) continue;
            values.push_back(x);
            colIndices.push_back(j + 1);
            dense[i * nFeatures + j] = double(x);
        }
    }
    rowOffsets[nRows] = values.size() + 1;

    services::Status status;
    auto table =
        CSRNumericTable::create(values.data(), colIndices.data(), rowOffsets.data(), nFeatures, nRows, CSRNumericTableIface::oneBased, &status);
    REQUIRE(status.ok());

    const double n = double(nRows);
    std::vector<double> refSum(nFeatures, 0.0), refSum2(nFeatures, 0.0), refSum2c(nFeatures, 0.0), refMin(nFeatures), refMax(nFeatures);
    for (size_t j = 0; j < nFeatures; ++j)
    {
        long double s = 0, s2 = 0;
        refMin[j] = refMax[j] = dense[j];
        for (size_t i = 0; i < nRows; ++i)
        {
            const double x = dense[i * nFeatures + j];
            s += x;
            s2 += (long double)x * x;
            refMin[j] = x < refMin[j] ? x : refMin[j];
            refMax[j] = x > refMax[j] ? x : refMax[j];
        }
        long double s2c = 0;
        for (size_t i = 0; i < nRows; ++i)
        {
            const long double d = dense[i * nFeatures + j] - s / n;
            s2c += d * d;
        }
        refSum[j]   = double(s);
        refSum2[j]  = double(s2);
        refSum2c[j] = double(s2c);
    }

    if (method == sumCSR)
    {
        std::vector<FPType> sums(refSum.begin(), refSum.end());
        auto sumsTable = HomogenNumericTable<FPType>::create(nFeatures, 1, NumericTable::doAllocate, &status);
        REQUIRE(status.ok());
        for (size_t j = 0; j < nFeatures; ++j) sumsTable->getArray()[j] = sums[j];
        table->basicStatistics.set(NumericTable::sum, sumsTable);
    }

    Batch<FPType, method> algorithm;
    algorithm.input.set(data, table);
    REQUIRE(algorithm.compute().ok());
    const auto result = algorithm.getResult();

    // The kernels accumulate in double; what is left is the rounding of the values and results to FPType.
    const double tol = (sizeof(FPType) == 4) ? 2e-5 : 1e-10;
    const auto check = [&](ResultId id, const std::vector<double> & expected) {
        BlockDescriptor<double> block;
        result->get(id)->getBlockOfRows(0, 1, readOnly, block);
        for (size_t j = 0; j < nFeatures; ++j)
        {
            INFO("result " << int(id) << ", column " << j << ": " << block.getBlockPtr()[j] << " vs " << expected[j]);
            REQUIRE(std::abs(block.getBlockPtr()[j] - expected[j]) <= tol * std::max(1.0, std::abs(expected[j])));
        }
        result->get(id)->releaseBlockOfRows(block);
    };

    std::vector<double> refMean(nFeatures), refSorm(nFeatures), refVar(nFeatures), refStd(nFeatures), refVart(nFeatures);
    for (size_t j = 0; j < nFeatures; ++j)
    {
        refMean[j] = refSum[j] / n;
        refSorm[j] = refSum2[j] / n;
        refVar[j]  = refSum2c[j] / (n - 1.0);
        refStd[j]  = std::sqrt(refVar[j]);
        refVart[j] = refStd[j] / refMean[j];
    }
    check(minimum, refMin);
    check(maximum, refMax);
    check(sum, refSum);
    check(sumSquares, refSum2);
    check(sumSquaresCentered, refSum2c);
    check(mean, refMean);
    check(secondOrderRawMoment, refSorm);
    check(variance, refVar);
    check(standardDeviation, refStd);
    // Column 0 is a full column of noise around zero: its mean is near zero and its variation is not meaningful.
    refVart.erase(refVart.begin());
    BlockDescriptor<double> block;
    result->get(variation)->getBlockOfRows(0, 1, readOnly, block);
    for (size_t j = 1; j < nFeatures; ++j)
    {
        REQUIRE(std::abs(block.getBlockPtr()[j] - refVart[j - 1]) <= tol * std::max(1.0, std::abs(refVart[j - 1])));
    }
    result->get(variation)->releaseBlockOfRows(block);
}

// oneAPI basic statistics reaches fastCSR only, so the other CSR kernels are run from here.
TEMPLATE_TEST("low order moments of a CSR table by each CSR method", "[unit][low_order_moments]", float, double)
{
    SECTION("fastCSR")
    {
        checkCsrMoments<TestType, fastCSR>();
    }
    SECTION("singlePassCSR")
    {
        checkCsrMoments<TestType, singlePassCSR>();
    }
    SECTION("sumCSR")
    {
        checkCsrMoments<TestType, sumCSR>();
    }
}
} // namespace daal::algorithms::low_order_moments
