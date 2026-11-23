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
#include "algorithms/covariance/covariance_batch.h"
#include "algorithms/covariance/covariance_online.h"
#include "data_management/data/csr_numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace daal::algorithms::covariance::test
{
using namespace daal::data_management;

/// A CSR copy of a dense row-major matrix, one-based, that keeps its arrays alive.
template <typename FPType>
struct CsrData
{
    std::vector<FPType> values;
    std::vector<size_t> cols;
    std::vector<size_t> rows;
};

template <typename FPType>
CsrData<FPType> toCsr(const std::vector<FPType> & dense, size_t nRows, size_t nCols, size_t rowBegin, size_t rowEnd)
{
    CsrData<FPType> csr;
    csr.rows.push_back(1);
    for (size_t i = rowBegin; i < rowEnd; ++i)
    {
        for (size_t j = 0; j < nCols; ++j)
        {
            if (dense[i * nCols + j] != FPType(0))
            {
                csr.values.push_back(dense[i * nCols + j]);
                csr.cols.push_back(j + 1);
            }
        }
        csr.rows.push_back(csr.values.size() + 1);
    }
    return csr;
}

template <typename FPType>
NumericTablePtr wrapCsr(CsrData<FPType> & csr, size_t nCols)
{
    return CSRNumericTable::create(csr.values.data(), csr.cols.data(), csr.rows.data(), nCols, csr.rows.size() - 1, CSRNumericTableIface::oneBased);
}

template <typename FPType>
std::vector<FPType> pull(const NumericTablePtr & table)
{
    BlockDescriptor<FPType> block;
    table->getBlockOfRows(0, table->getNumberOfRows(), readOnly, block);
    const size_t count = table->getNumberOfRows() * table->getNumberOfColumns();
    std::vector<FPType> out(block.getBlockPtr(), block.getBlockPtr() + count);
    table->releaseBlockOfRows(block);
    return out;
}

template <typename FPType>
void requireClose(const std::vector<FPType> & result, const std::vector<FPType> & reference)
{
    const double tol = std::is_same_v<FPType, float> ? 1e-4 : 1e-10;
    REQUIRE(result.size() == reference.size());
    for (size_t i = 0; i < result.size(); ++i)
    {
        CAPTURE(i, result[i], reference[i]);
        REQUIRE(std::abs(double(result[i]) - double(reference[i])) <= tol * std::max(1.0, std::abs(double(reference[i]))));
    }
}

/// Column 0 is dense and far from zero, column 3 is empty, every eleventh row is empty.
template <typename FPType>
std::vector<FPType> makeDense(size_t nRows, size_t nCols)
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<FPType> dense(nRows * nCols, FPType(0));
    for (size_t i = 0; i < nRows; ++i)
    {
        if (i % 11 == 0) continue;
        dense[i * nCols] = FPType(1e3 + uniform(rng));
        for (size_t j = 1; j < nCols; ++j)
        {
            const double x = uniform(rng);
            if (j != 3 && std::abs(x) < 0.3) dense[i * nCols + j] = FPType(x * 10);
        }
    }
    return dense;
}

template <typename FPType, Method csrMethod>
void checkBatchMatchesDense()
{
    constexpr size_t nRows = 300, nCols = 7;
    std::vector<FPType> dense = makeDense<FPType>(nRows, nCols);
    CsrData<FPType> csr       = toCsr(dense, nRows, nCols, 0, nRows);

    Batch<FPType, defaultDense> denseAlgorithm;
    denseAlgorithm.input.set(data, HomogenNumericTable<FPType>::create(dense.data(), nCols, nRows));
    REQUIRE(denseAlgorithm.compute().ok());

    Batch<FPType, csrMethod> csrAlgorithm;
    csrAlgorithm.input.set(data, wrapCsr(csr, nCols));
    REQUIRE(csrAlgorithm.compute().ok());

    requireClose(pull<FPType>(csrAlgorithm.getResult()->get(covariance)), pull<FPType>(denseAlgorithm.getResult()->get(covariance)));
    requireClose(pull<FPType>(csrAlgorithm.getResult()->get(mean)), pull<FPType>(denseAlgorithm.getResult()->get(mean)));
}

TEST("covariance fastCSR batch matches the dense result", "[unit]")
{
    checkBatchMatchesDense<double, fastCSR>();
    checkBatchMatchesDense<float, fastCSR>();
}

TEST("covariance singlePassCSR batch matches the dense result", "[unit]")
{
    checkBatchMatchesDense<double, singlePassCSR>();
}

TEST("covariance fastCSR online matches the dense batch result", "[unit]")
{
    using FPType           = double;
    constexpr size_t nRows = 300, nCols = 7, nBlocks = 3;
    std::vector<FPType> dense = makeDense<FPType>(nRows, nCols);

    Batch<FPType, defaultDense> denseAlgorithm;
    denseAlgorithm.input.set(data, HomogenNumericTable<FPType>::create(dense.data(), nCols, nRows));
    REQUIRE(denseAlgorithm.compute().ok());

    Online<FPType, fastCSR> online;
    std::vector<CsrData<FPType> > blocks;
    for (size_t b = 0; b < nBlocks; ++b)
    {
        blocks.push_back(toCsr(dense, nRows, nCols, b * nRows / nBlocks, (b + 1) * nRows / nBlocks));
    }
    for (auto & block : blocks)
    {
        online.input.set(data, wrapCsr(block, nCols));
        REQUIRE(online.compute().ok());
    }
    REQUIRE(online.finalizeCompute().ok());

    requireClose(pull<FPType>(online.getResult()->get(covariance)), pull<FPType>(denseAlgorithm.getResult()->get(covariance)));
    requireClose(pull<FPType>(online.getResult()->get(mean)), pull<FPType>(denseAlgorithm.getResult()->get(mean)));
}

// Large enough for MKL to run the cross-product in parallel if it were allowed to.
TEST("covariance fastCSR batch is the same from run to run", "[unit]")
{
    using FPType           = double;
    constexpr size_t nRows = 20000, nCols = 200, nnzPerRow = 10;
    std::mt19937 rng(3);
    std::uniform_int_distribution<size_t> column(0, nCols - 1);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<FPType> dense(nRows * nCols, FPType(0));
    for (size_t i = 0; i < nRows; ++i)
    {
        for (size_t k = 0; k < nnzPerRow; ++k)
        {
            dense[i * nCols + column(rng)] = FPType(uniform(rng));
        }
    }
    CsrData<FPType> csr = toCsr(dense, nRows, nCols, 0, nRows);

    const auto run = [&]() {
        Batch<FPType, fastCSR> algorithm;
        algorithm.input.set(data, wrapCsr(csr, nCols));
        REQUIRE(algorithm.compute().ok());
        return pull<FPType>(algorithm.getResult()->get(covariance));
    };
    const auto first = run();
    for (int r = 0; r < 5; ++r)
    {
        CAPTURE(r);
        REQUIRE(run() == first);
    }
}

} // namespace daal::algorithms::covariance::test
