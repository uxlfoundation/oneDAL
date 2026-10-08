/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include <unordered_set>
#include <vector>

#include "oneapi/dal/algo/kmeans_init/compute.hpp"
#include "oneapi/dal/backend/primitives/rng/host_engine.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

#include "oneapi/dal/algo/kmeans_init/test/fixture.hpp"

namespace oneapi::dal::kmeans_init::test {

namespace te = dal::test::engine;

template <typename TestType>
class kmeans_init_batch_test : public kmeans_init_test<TestType, kmeans_init_batch_test<TestType>> {
};

using kmeans_init_types = _TE_COMBINE_TYPES_2((float, double),
                                              (kmeans_init::method::dense,
                                               kmeans_init::method::random_dense,
                                               kmeans_init::method::plus_plus_dense,
                                               kmeans_init::method::parallel_plus_dense));

TEMPLATE_LIST_TEST_M(kmeans_init_batch_test,
                     "kmeans init dense test random",
                     "[kmeans_init][batch][random]",
                     kmeans_init_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t cluster_count = 4;

    const double data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                            -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };
    const auto data_table = homogen_table::wrap(data, row_count, column_count);

    this->dense_checks(cluster_count, data_table);
}

/// Picks of the reference k-means++ and how often they hinge on the best trial's own candidate.
struct plus_plus_reference {
    /// Row indices of the picked centers, in pick order
    std::vector<std::int64_t> rows;
    /// Steps where only the candidate of the previous step's winning trial gives the pick, when that
    /// trial is not trial 0
    std::int64_t own_candidate_picks = 0;
};

/// Reference k-means++ with local trials that replays the kernel's random draws.
///
/// A small copy of the DAAL kernel's sampling (draw layout, first center, cumulative search),
/// so it has to be updated if that sampling changes.
///
/// @tparam Float        Floating-point type of the draws and distances
/// @param data          Input  `row_count x column_count` row-major buffer
/// @param row_count     Input  Number of rows in `data`
/// @param column_count  Input  Number of columns in `data`
/// @param cluster_count Input  Number of centers to pick
/// @param trial_count   Input  Number of candidates scored per center, at least 2
/// @param seed          Input  Seed of the mt19937 engine the kernel uses
/// @return              The picks and the count of steps that need the best trial's candidate
template <typename Float>
static plus_plus_reference reference_plus_plus(const std::vector<Float>& data,
                                               std::int64_t row_count,
                                               std::int64_t column_count,
                                               std::int64_t cluster_count,
                                               std::int64_t trial_count,
                                               std::int64_t seed) {
    namespace pr = dal::backend::primitives;
    pr::host_engine engine(seed, pr::engine_type_internal::mt19937);
    std::vector<Float> probabilities(cluster_count * trial_count);
    pr::uniform<Float>(probabilities.size(), probabilities.data(), engine, Float(0), Float(1));

    const auto sq_dist = [&](std::int64_t a, std::int64_t b) {
        Float d = 0;
        for (std::int64_t j = 0; j < column_count; ++j) {
            const Float diff = data[a * column_count + j] - data[b * column_count + j];
            d += diff * diff;
        }
        return d;
    };

    const std::int64_t first =
        std::min<std::int64_t>(std::int64_t(probabilities[0] * Float(row_count)), row_count - 1);
    plus_plus_reference result;
    result.rows = { first };
    std::vector<Float> min_dist(row_count);
    Float potential = 0;
    for (std::int64_t r = 0; r < row_count; ++r) {
        min_dist[r] = sq_dist(r, first);
        potential += min_dist[r];
    }

    std::int64_t previous_best_trial = 0;
    for (std::int64_t c = 1; c < cluster_count; ++c) {
        std::int64_t best_trial = -1;
        std::int64_t best_row = -1;
        Float best_potential = std::numeric_limits<Float>::max();
        std::vector<Float> best_dist;
        std::vector<std::int64_t> candidates(trial_count);
        for (std::int64_t t = 0; t < trial_count; ++t) {
            Float sample = potential * probabilities[t * cluster_count + c];
            std::int64_t candidate = 0;
            for (; candidate + 1 < row_count && sample >= min_dist[candidate]; ++candidate) {
                sample -= min_dist[candidate];
            }
            candidates[t] = candidate;
            std::vector<Float> dist(row_count);
            Float trial_potential = 0;
            for (std::int64_t r = 0; r < row_count; ++r) {
                dist[r] = std::min(min_dist[r], sq_dist(r, candidate));
                trial_potential += dist[r];
            }
            if (trial_potential < best_potential) {
                best_potential = trial_potential;
                best_trial = t;
                best_row = candidate;
                best_dist = std::move(dist);
            }
        }
        if (best_trial != 0 && best_trial == previous_best_trial &&
            std::count(candidates.begin(), candidates.end(), best_row) == 1) {
            ++result.own_candidate_picks;
        }
        result.rows.push_back(best_row);
        min_dist = std::move(best_dist);
        potential = best_potential;
        previous_best_trial = best_trial;
    }
    return result;
}

using kmeans_init_plus_plus_types = _TE_COMBINE_TYPES_2((float, double),
                                                        (kmeans_init::method::plus_plus_dense,
                                                         kmeans_init::method::plus_plus_csr));

// Integer coordinates keep every distance and potential exact, so the kernel must pick the
// same rows as the reference whatever its summation order.
TEMPLATE_LIST_TEST_M(kmeans_init_batch_test,
                     "kmeans init plus plus scores every local trial",
                     "[kmeans_init][batch][plus_plus]",
                     kmeans_init_plus_plus_types) {
    SKIP_IF(this->get_policy().is_gpu());
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    constexpr std::int64_t row_count = 60;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t cluster_count = 5;
    constexpr std::int64_t trial_count = 3;

    std::mt19937 gen(42);
    std::vector<float_t> data(row_count * column_count);
    for (auto& v : data) {
        v = float_t(gen() % 16);
    }
    // One-based CSR that stores every value, zeros included.
    auto csr_cols = dal::array<std::int64_t>::empty(row_count * column_count);
    auto csr_offsets = dal::array<std::int64_t>::empty(row_count + 1);
    for (std::int64_t i = 0; i < row_count * column_count; ++i) {
        csr_cols.get_mutable_data()[i] = i % column_count + 1;
    }
    for (std::int64_t r = 0; r <= row_count; ++r) {
        csr_offsets.get_mutable_data()[r] = r * column_count + 1;
    }
    const auto csr_values = dal::array<float_t>::wrap(data.data(), row_count * column_count);
    const table data_table =
        std::is_same_v<method_t, kmeans_init::method::plus_plus_csr>
            ? table{ csr_table::wrap(csr_values,
                                     csr_cols,
                                     csr_offsets,
                                     column_count,
                                     sparse_indexing::one_based) }
            : table{ homogen_table::wrap(data.data(), row_count, column_count) };

    std::int64_t own_candidate_picks = 0;
    for (std::int64_t seed = 0; seed < 20; ++seed) {
        CAPTURE(seed);
        const auto desc =
            this->get_descriptor(cluster_count).set_local_trials_count(trial_count).set_seed(seed);
        const auto centroids = this->compute(desc, data_table).get_centroids();
        const auto actual = row_accessor<const float_t>(centroids).pull();
        const auto expected =
            reference_plus_plus(data, row_count, column_count, cluster_count, trial_count, seed);
        own_candidate_picks += expected.own_candidate_picks;
        for (std::int64_t c = 0; c < cluster_count; ++c) {
            CAPTURE(c, expected.rows[c]);
            for (std::int64_t j = 0; j < column_count; ++j) {
                REQUIRE(actual[c * column_count + j] == data[expected.rows[c] * column_count + j]);
            }
        }
    }
    // The seeds must reach a step where only the previous winner's own candidate gives the
    // reference pick; otherwise they cannot tell a kernel that skips that candidate apart.
    REQUIRE(own_candidate_picks > 0);
}

} // namespace oneapi::dal::kmeans_init::test
