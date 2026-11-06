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

/// Unit tests for the QR method of linear regression training (no intercept) on inputs where the
/// triangular factor is singular: fewer observations than coefficients (in one block or in
/// partial results) and exactly collinear columns. In both cases the expected result is the
/// minimum-norm least-squares solution. Expected values are derived by hand in the comments.

#include <cmath>
#include <cstddef>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "algorithms/linear_regression/linear_regression_training_batch.h"
#include "algorithms/linear_regression/linear_regression_training_online.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal::test
{
namespace lr = daal::algorithms::linear_regression;
using daal::data_management::BlockDescriptor;
using daal::data_management::HomogenNumericTable;
using daal::data_management::NumericTablePtr;
using daal::data_management::readOnly;

/// Reads the coefficients of the first response from a trained model. The intercept slot is
/// returned first when `withIntercept` is set and dropped otherwise.
template <typename FPType>
std::vector<FPType> readBeta(const lr::ModelPtr & model, size_t nCols, bool withIntercept = false)
{
    NumericTablePtr betaTable = model->getBeta();
    BlockDescriptor<FPType> block;
    REQUIRE(betaTable->getBlockOfRows(0, 1, readOnly, block).ok());
    const FPType * beta = block.getBlockPtr();
    /* beta[0] is the intercept slot, it is zero when interceptFlag is false */
    std::vector<FPType> result(withIntercept ? beta : beta + 1, beta + 1 + nCols);
    REQUIRE(betaTable->releaseBlockOfRows(block).ok());
    return result;
}

/// Trains in a single batch call.
template <typename FPType>
std::vector<FPType> trainBatch(const std::vector<FPType> & x, const std::vector<FPType> & y, size_t nRows, size_t nCols, bool intercept = false)
{
    NumericTablePtr xTable = HomogenNumericTable<FPType>::create(const_cast<FPType *>(x.data()), nCols, nRows);
    NumericTablePtr yTable = HomogenNumericTable<FPType>::create(const_cast<FPType *>(y.data()), 1, nRows);

    lr::training::Batch<FPType, lr::training::qrDense> algorithm;
    algorithm.parameter.interceptFlag = intercept;
    algorithm.input.set(lr::training::data, xTable);
    algorithm.input.set(lr::training::dependentVariables, yTable);
    REQUIRE(algorithm.compute().ok());
    return readBeta<FPType>(algorithm.getResult()->get(lr::training::model), nCols, intercept);
}

/// Trains in the online mode, feeding consecutive blocks of `chunkRows[i]` rows.
template <typename FPType>
std::vector<FPType> trainOnline(const std::vector<FPType> & x, const std::vector<FPType> & y, size_t nCols, const std::vector<size_t> & chunkRows)
{
    lr::training::Online<FPType, lr::training::qrDense> algorithm;
    algorithm.parameter.interceptFlag = false;

    size_t firstRow = 0;
    for (size_t rows : chunkRows)
    {
        NumericTablePtr xTable = HomogenNumericTable<FPType>::create(const_cast<FPType *>(x.data()) + firstRow * nCols, nCols, rows);
        NumericTablePtr yTable = HomogenNumericTable<FPType>::create(const_cast<FPType *>(y.data()) + firstRow, 1, rows);
        algorithm.input.set(lr::training::data, xTable);
        algorithm.input.set(lr::training::dependentVariables, yTable);
        REQUIRE(algorithm.compute().ok());
        firstRow += rows;
    }
    REQUIRE(algorithm.finalizeCompute().ok());
    return readBeta<FPType>(algorithm.getResult()->get(lr::training::model), nCols);
}

template <typename FPType>
void requireClose(const std::vector<FPType> & actual, const std::vector<double> & expected, double tolerance)
{
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        INFO("coefficient " << i << ": got " << actual[i] << ", expected " << expected[i]);
        REQUIRE(std::abs(static_cast<double>(actual[i]) - expected[i]) < tolerance);
    }
}

/// X = [1 1], y = 2 has infinitely many solutions b0 + b1 = 2; the minimum-norm one is (1, 1).
template <typename FPType>
void checkWideSingleObservation(double tolerance)
{
    const std::vector<FPType> x = { 1, 1 };
    const std::vector<FPType> y = { 2 };
    requireClose(trainBatch<FPType>(x, y, 1, 2), { 1.0, 1.0 }, tolerance);
}

/// X = [1 1 0; 0 1 1], y = (2, 2). With X*X' = [2 1; 1 2], the minimum-norm solution is
/// X' * inv(X*X') * y = X' * (2/3, 2/3) = (2/3, 4/3, 2/3).
template <typename FPType>
void checkWideTwoObservations(double tolerance)
{
    const std::vector<FPType> x        = { 1, 1, 0, 0, 1, 1 };
    const std::vector<FPType> y        = { 2, 2 };
    const std::vector<double> expected = { 2.0 / 3.0, 4.0 / 3.0, 2.0 / 3.0 };

    requireClose(trainBatch<FPType>(x, y, 2, 3), expected, tolerance);
    /* Every partial result is a block with fewer observations than coefficients */
    requireClose(trainOnline<FPType>(x, y, 3, { 1, 1 }), expected, tolerance);
}

/// X = [1 1; 2 2], y = (2, 4): the columns are equal, and both rows say b0 + b1 = 2. Among all
/// such solutions the minimum-norm one is (1, 1).
template <typename FPType>
void checkExactlyCollinearColumns(double tolerance)
{
    const std::vector<FPType> x = { 1, 1, 2, 2 };
    const std::vector<FPType> y = { 2, 4 };
    requireClose(trainBatch<FPType>(x, y, 2, 2), { 1.0, 1.0 }, tolerance);
}

/// X = [1 1; 2 2; 3 3], y = 1 + 2 * x with an intercept: the intercept is 1 and the two equal
/// columns share a total slope of 2, so the minimum-norm coefficients are (1, 1, 1).
template <typename FPType>
void checkCollinearColumnsWithIntercept(double tolerance)
{
    const std::vector<FPType> x = { 1, 1, 2, 2, 3, 3 };
    const std::vector<FPType> y = { 3, 5, 7 };
    requireClose(trainBatch<FPType>(x, y, 3, 2, true), { 1.0, 1.0, 1.0 }, tolerance);
}

/// Full-rank data with y = X * (1, -1) must give the same exact answer however it is split into
/// blocks, including blocks with fewer rows than columns.
template <typename FPType>
void checkOnlineBlocksOfAnySize(double tolerance)
{
    const std::vector<FPType> x        = { 1, 2, 3, 5, 4, 1, 2, 7 };
    const std::vector<FPType> y        = { -1, -2, 3, -5 };
    const std::vector<double> expected = { 1.0, -1.0 };

    requireClose(trainBatch<FPType>(x, y, 4, 2), expected, tolerance);
    requireClose(trainOnline<FPType>(x, y, 2, { 2, 2 }), expected, tolerance);
    requireClose(trainOnline<FPType>(x, y, 2, { 1, 3 }), expected, tolerance);
    requireClose(trainOnline<FPType>(x, y, 2, { 3, 1 }), expected, tolerance);
    requireClose(trainOnline<FPType>(x, y, 2, { 1, 1, 2 }), expected, tolerance);
    requireClose(trainOnline<FPType>(x, y, 2, { 1, 1, 1, 1 }), expected, tolerance);
}

TEST("QR linear regression: fewer observations than coefficients gives the minimum-norm solution", "[unit][linear_regression]")
{
    checkWideSingleObservation<double>(1e-10);
    checkWideSingleObservation<float>(1e-4);
    checkWideTwoObservations<double>(1e-10);
    checkWideTwoObservations<float>(1e-4);
}

TEST("QR linear regression: exactly collinear columns give the minimum-norm solution", "[unit][linear_regression]")
{
    checkExactlyCollinearColumns<double>(1e-10);
}

TEST("QR linear regression: exactly collinear columns with an intercept give the minimum-norm solution", "[unit][linear_regression]")
{
    checkCollinearColumnsWithIntercept<double>(1e-10);
}

TEST("QR linear regression: online blocks with fewer rows than columns agree with batch", "[unit][linear_regression]")
{
    checkOnlineBlocksOfAnySize<double>(1e-10);
    checkOnlineBlocksOfAnySize<float>(1e-4);
}

} // namespace daal::test
