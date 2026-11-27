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

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/tables.hpp"

#include "src/algorithms/kmeans/kmeans_init_kernel.h"
#include "src/algorithms/kmeans/kmeans_plusplus_init_impl.i"

namespace oneapi::dal::kmeans_init::test {

namespace te = dal::test::engine;
namespace interop = dal::backend::interop;
namespace daal_init = daal::algorithms::kmeans::init::internal;

constexpr auto daal_cpu = daal::internal::firstCpuType;

// Each trial's slot must be scored with that trial's own candidate, the previous best trial's
// slot included, and every slot must start from the previous best trial's distances.
TEMPLATE_TEST("kmeans++ updateMinDistInBlock scores each trial with its own candidate",
              "[kmeans_init][internal][plus_plus]",
              float,
              double) {
    using float_t = TestType;
    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 2;
    constexpr std::size_t trial_count = 3;

    const std::vector<float_t> data = { 0, 0, 3, 0, 0, 4, 6, 8 };
    const auto dense_table = interop::copy_to_daal_homogen_table<float_t>(
        homogen_table::wrap(data.data(), row_count, column_count));
    const auto csr_table = interop::copy_to_daal_csr_table<float_t>(
        te::dense_to_explicit_csr(data.data(),
                                  row_count,
                                  column_count,
                                  sparse_indexing::one_based));
    daal_init::DataHelperDense<float_t, daal_cpu> dense(dense_table.get());
    daal_init::DataHelperCSR<float_t, daal_cpu> csr(csr_table.get());

    // Trial t's candidate is row t.
    const std::vector<float_t> candidates(data.begin(), data.begin() + trial_count * column_count);
    const std::vector<float_t> previous_min_dist = { 1000, 1000, 1000, 50 };

    // min(previous_min_dist, squared distance to row t) for each trial t and row.
    const float_t expected_dist[trial_count][row_count] = { { 0, 9, 16, 50 },
                                                            { 9, 0, 25, 50 },
                                                            { 16, 25, 0, 50 } };
    const float_t expected_potential[trial_count] = { 75, 84, 91 };

    auto check = [&](auto& data_helper) {
        for (std::size_t best_trial = 0; best_trial < trial_count; ++best_trial) {
            CAPTURE(best_trial);
            std::vector<float_t> min_dist(trial_count * row_count, float_t(-1));
            std::copy(previous_min_dist.begin(),
                      previous_min_dist.end(),
                      min_dist.begin() + best_trial * row_count);
            std::vector<float_t> potential(trial_count, float_t(-1));

            const auto status = data_helper.updateMinDistInBlock(potential.data(),
                                                                 1,
                                                                 0,
                                                                 trial_count,
                                                                 best_trial,
                                                                 nullptr,
                                                                 candidates.data(),
                                                                 min_dist.data());
            REQUIRE(status.ok());

            for (std::size_t t = 0; t < trial_count; ++t) {
                CAPTURE(t);
                REQUIRE(potential[t] == expected_potential[t]);
                for (std::int64_t r = 0; r < row_count; ++r) {
                    CAPTURE(r);
                    REQUIRE(min_dist[t * row_count + r] == expected_dist[t][r]);
                }
            }
        }
    };
    check(dense);
    check(csr);
}

// Enough blocks for every thread, so a result that depends on the thread schedule would show.
TEMPLATE_TEST("kmeans|| CSR update assigns every row to its nearest candidate in every run",
              "[kmeans_init][internal][parallel_plus]",
              float,
              double) {
    using float_t = TestType;
    constexpr std::int64_t row_count = 20000;
    constexpr std::int64_t column_count = 10;
    constexpr std::size_t candidate_count = 5;
    constexpr int run_count = 3;

    std::mt19937 rng(777);
    std::normal_distribution<float_t> normal;
    std::vector<float_t> data(row_count * column_count);
    for (auto& value : data) {
        const float_t x = normal(rng);
        value = x > 0 ? x : float_t(0);
    }
    const auto csr_table = interop::copy_to_daal_csr_table<float_t>(
        te::dense_to_explicit_csr(data.data(),
                                  row_count,
                                  column_count,
                                  sparse_indexing::one_based));
    daal_init::DataHelperCSR<float_t, daal_cpu> csr(csr_table.get());

    // Candidate c is row c; the kernel ranks candidates by |c|^2 / 2 - <x, c>.
    const std::vector<float_t> candidates(data.begin(),
                                          data.begin() + candidate_count * column_count);
    std::vector<float_t> half_norms(candidate_count);
    for (std::size_t c = 0; c < candidate_count; ++c) {
        double norm = 0;
        for (std::int64_t j = 0; j < column_count; ++j) {
            norm += double(candidates[c * column_count + j]) * candidates[c * column_count + j];
        }
        half_norms[c] = float_t(0.5 * norm);
    }
    const auto score = [&](std::int64_t row, std::size_t c) {
        double product = 0;
        for (std::int64_t j = 0; j < column_count; ++j) {
            product += double(data[row * column_count + j]) * candidates[c * column_count + j];
        }
        return double(half_norms[c]) - product;
    };
    // Near-ties may go either way, so a row only has to be as close as its nearest candidate.
    const double tie_tol = std::is_same_v<float_t, double> ? 1e-10 : 1e-4;

    const std::size_t block_count =
        (row_count + daal_init::_nRowsInBlock - 1) / daal_init::_nRowsInBlock;
    std::vector<int> first_nearest;
    std::vector<float_t> first_min_dist;
    for (int run = 0; run < run_count; ++run) {
        CAPTURE(run);
        std::vector<int> rating(candidate_count, 0);
        std::vector<int> nearest(row_count, 0);
        std::vector<float_t> min_dist(row_count, std::numeric_limits<float_t>::max());
        std::vector<float_t> min_dist_acc(block_count);
        float_t overall_error = 0;
        daal_init::TaskParallelPlusUpdateDist<float_t, daal_cpu, decltype(csr)> task(
            block_count,
            rating.data(),
            nearest.data(),
            overall_error,
            csr,
            candidates.data(),
            half_norms.data(),
            min_dist.data(),
            min_dist_acc.data());
        REQUIRE(task.updateMinDist(0, candidate_count).ok());

        std::int64_t not_nearest = 0;
        for (std::int64_t i = 0; i < row_count; ++i) {
            double best = score(i, 0);
            for (std::size_t c = 1; c < candidate_count; ++c) {
                best = std::min(best, score(i, c));
            }
            not_nearest += score(i, nearest[i]) > best + tie_tol * std::max(1.0, std::abs(best));
        }
        REQUIRE(not_nearest == 0);
        if (run == 0) {
            first_nearest = nearest;
            first_min_dist = min_dist;
        }
        REQUIRE(nearest == first_nearest);
        REQUIRE(min_dist == first_min_dist);
    }
}

} // namespace oneapi::dal::kmeans_init::test
