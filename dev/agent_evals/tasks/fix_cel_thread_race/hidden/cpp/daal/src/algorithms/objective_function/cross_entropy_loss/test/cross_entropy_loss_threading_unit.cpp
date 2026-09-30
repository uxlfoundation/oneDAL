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
#include "algorithms/optimization_solver/objective_function/cross_entropy_loss_batch.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/env_detect.h"

#include <cmath>
#include <random>
#include <vector>

namespace daal::algorithms::optimization_solver::cross_entropy_loss::test
{
using daal::data_management::HomogenNumericTable;
using daal::data_management::NumericTablePtr;

struct problem
{
    std::size_t n, p, k;
    std::vector<double> x, y, beta;
};

static problem make_problem(std::size_t n, std::size_t p, std::size_t k)
{
    problem pr { n, p, k, std::vector<double>(n * p), std::vector<double>(n), std::vector<double>(k * (p + 1)) };
    std::mt19937_64 gen(42);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> label(0, k - 1);
    for (auto & v : pr.x) v = normal(gen);
    for (auto & v : pr.y) v = double(label(gen));
    for (auto & v : pr.beta) v = 0.1 * normal(gen);
    return pr;
}

// Value and gradient (intercept first in each class block), in the layout the kernel uses.
static std::vector<double> compute(problem & pr)
{
    auto x    = HomogenNumericTable<double>::create(pr.x.data(), pr.p, pr.n);
    auto y    = HomogenNumericTable<double>::create(pr.y.data(), 1, pr.n);
    auto beta = HomogenNumericTable<double>::create(pr.beta.data(), 1, pr.beta.size());

    auto loss = Batch<double>::create(pr.k, pr.n);
    loss->input.set(cross_entropy_loss::data, x);
    loss->input.set(cross_entropy_loss::dependentVariables, y);
    loss->input.set(cross_entropy_loss::argument, beta);
    loss->parameter().interceptFlag    = true;
    loss->parameter().resultsToCompute = objective_function::value | objective_function::gradient;
    REQUIRE(loss->compute().ok());

    std::vector<double> out;
    for (auto id : { objective_function::valueIdx, objective_function::gradientIdx })
    {
        NumericTablePtr t = loss->getResult()->get(id);
        daal::data_management::BlockDescriptor<double> block;
        t->getBlockOfRows(0, t->getNumberOfRows(), daal::data_management::readOnly, block);
        const double * const ptr = block.getBlockPtr();
        out.insert(out.end(), ptr, ptr + t->getNumberOfRows() * t->getNumberOfColumns());
        t->releaseBlockOfRows(block);
    }
    return out;
}

// Plain serial reference: value = -mean log softmax(x b + b0)[y], gradient = mean (softmax - onehot) [1, x].
static std::vector<double> reference(const problem & pr)
{
    const std::size_t nb = pr.p + 1;
    std::vector<double> out(1 + pr.k * nb, 0.0), f(pr.k);
    for (std::size_t i = 0; i < pr.n; ++i)
    {
        const double * xi = pr.x.data() + i * pr.p;
        double fmax       = -INFINITY;
        for (std::size_t c = 0; c < pr.k; ++c)
        {
            f[c] = pr.beta[c * nb];
            for (std::size_t j = 0; j < pr.p; ++j) f[c] += xi[j] * pr.beta[c * nb + 1 + j];
            fmax = std::max(fmax, f[c]);
        }
        double sum = 0;
        for (std::size_t c = 0; c < pr.k; ++c) sum += std::exp(f[c] - fmax);
        const std::size_t yi = std::size_t(pr.y[i]);
        out[0] -= (f[yi] - fmax - std::log(sum)) / double(pr.n);
        for (std::size_t c = 0; c < pr.k; ++c)
        {
            const double r = std::exp(f[c] - fmax) / sum - (c == yi ? 1.0 : 0.0);
            out[1 + c * nb] += r / double(pr.n);
            for (std::size_t j = 0; j < pr.p; ++j) out[1 + c * nb + 1 + j] += r * xi[j] / double(pr.n);
        }
    }
    return out;
}

static void require_close(const std::vector<double> & a, const std::vector<double> & b, double tol)
{
    REQUIRE(a.size() == b.size());
    double scale = 0;
    for (double v : b) scale = std::max(scale, std::abs(v));
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        CAPTURE(i, a[i], b[i]);
        REQUIRE(std::abs(a[i] - b[i]) <= tol * std::max(1.0, scale));
    }
}

TEST("cross_entropy_loss value and gradient do not depend on the thread count", "[unit]")
{
    const std::size_t n = GENERATE(20000, 50000);
    const std::size_t k = GENERATE(7, 64);
    problem pr          = make_problem(n, 16, k);
    const auto ref      = reference(pr);

    auto & env              = *daal::services::Environment::getInstance();
    const std::size_t saved = env.getNumberOfThreads();

    env.setNumberOfThreads(1);
    const auto serial = compute(pr);
    env.setNumberOfThreads(saved);
    require_close(serial, ref, 1e-10);

    // The block partition is fixed and the per-block partial results are reduced serially,
    // so repeated threaded calls have to reproduce the single-threaded result exactly.
    for (int rep = 0; rep < 10; ++rep)
    {
        CAPTURE(n, k, rep, saved);
        const auto threaded = compute(pr);
        require_close(threaded, ref, 1e-10);
        REQUIRE(threaded == serial);
    }
}

} // namespace daal::algorithms::optimization_solver::cross_entropy_loss::test
