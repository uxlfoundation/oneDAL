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

#include "oneapi/dal/algo/hdbscan/test/fixture.hpp"

#include <map>
#include <set>

namespace oneapi::dal::hdbscan::test {

template <typename TestType>
class hdbscan_batch_test : public hdbscan_test<TestType, hdbscan_batch_test<TestType>> {};

// =========================================================================
// brute_force method tests
// =========================================================================

using hdbscan_bf_types = COMBINE_TYPES((float, double), (hdbscan::method::brute_force));
using hdbscan_bf_only = COMBINE_TYPES((double), (hdbscan::method::brute_force));

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: compute mode check",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2,  0.15, 0.15, 5.0,
                                 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2, 5.15, 5.15, 10.0, 0.0 };
    const auto x = homogen_table::wrap(data, 11, 2);

    constexpr std::int64_t min_cluster_size = 5;
    constexpr std::int64_t min_samples = 5;

    result_option_id res_all = result_option_id(dal::result_option_id_base(mask_full));

    const result_option_id compute_mode = GENERATE_COPY(result_options::responses,
                                                        result_options::core_flags,
                                                        result_options::core_observations,
                                                        result_options::core_observation_indices,
                                                        res_all);

    this->mode_checks(compute_mode, x, min_cluster_size, min_samples);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: all noise when min_cluster_size > n",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    const auto x = homogen_table::wrap(data, 5, 1);

    constexpr std::int64_t min_cluster_size = 10;
    constexpr std::int64_t min_samples = 2;

    this->run_checks(x, min_cluster_size, min_samples, 0);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: two well-separated clusters",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    this->run_checks(x, 5, 5, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: single cluster with noise",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        100.0, 100.0, //
    };
    const auto x = homogen_table::wrap(data, 6, 2);

    this->run_checks(x, 5, 5, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: gold data test",
                     "[hdbscan][batch][gold]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());

    std::int64_t min_cluster_size = gold_dataset::get_min_cluster_size();
    std::int64_t min_samples = gold_dataset::get_min_samples();

    this->run_checks(x, min_cluster_size, min_samples, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: small min_cluster_size",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0, 0.0, //
        0.1, 0.1, //
        0.2, 0.0, //
        5.0, 5.0, //
        5.1, 5.1, //
        5.2, 5.0, //
    };
    const auto x = homogen_table::wrap(data, 6, 2);

    this->run_checks(x, 2, 2, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: two points",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 0.0, 1.0, 1.0 };
    const auto x = homogen_table::wrap(data, 2, 2);

    // min_cluster_size=2: both points should form a cluster
    this->run_checks(x, 2, 2, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: all identical points",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        1.0, 1.0, //
        1.0, 1.0, //
        1.0, 1.0, //
        1.0, 1.0, //
        1.0, 1.0, //
    };
    const auto x = homogen_table::wrap(data, 5, 2);

    // All identical points with zero distances
    this->run_checks(x, 2, 2, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: 1D data three clusters",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    // Three clusters in 1D, well-separated
    constexpr float_t data[] = { 0.0,  0.1,  0.2,  0.15,  0.05, //
                                 5.0,  5.1,  5.2,  5.15,  5.05, //
                                 10.0, 10.1, 10.2, 10.15, 10.05 };
    const auto x = homogen_table::wrap(data, 15, 1);

    this->run_checks(x, 5, 5, 3);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: high-dimensional data",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    // 10-dimensional data with 2 clusters
    constexpr std::int64_t n = 10;
    constexpr std::int64_t d = 10;
    float_t data[n * d];

    // Cluster 0: near origin
    for (std::int64_t i = 0; i < 5; i++)
        for (std::int64_t j = 0; j < d; j++)
            data[i * d + j] = static_cast<float_t>(0.01 * (i + j));

    // Cluster 1: far from origin
    for (std::int64_t i = 5; i < 10; i++)
        for (std::int64_t j = 0; j < d; j++)
            data[i * d + j] = static_cast<float_t>(10.0 + 0.01 * (i + j));

    const auto x = homogen_table::wrap(data, n, d);

    this->run_checks(x, 5, 5, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: min_samples=1",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0, 0.0, //
        0.1, 0.1, //
        0.2, 0.0, //
        5.0, 5.0, //
        5.1, 5.1, //
        5.2, 5.0, //
    };
    const auto x = homogen_table::wrap(data, 6, 2);

    // min_samples=1 means core distance is 0 for every point
    this->run_checks(x, 2, 1, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: varying min_cluster_size",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    // Three clusters with varying sizes: 3, 5, 7 points
    constexpr float_t data[] = { // Cluster A: 3 points
                                 0.0,
                                 0.0,
                                 0.1,
                                 0.1,
                                 0.2,
                                 0.0,
                                 // Cluster B: 5 points
                                 5.0,
                                 5.0,
                                 5.1,
                                 5.1,
                                 5.2,
                                 5.0,
                                 5.0,
                                 5.2,
                                 5.15,
                                 5.15,
                                 // Cluster C: 7 points
                                 10.0,
                                 10.0,
                                 10.1,
                                 10.1,
                                 10.2,
                                 10.0,
                                 10.0,
                                 10.2,
                                 10.15,
                                 10.15,
                                 10.05,
                                 10.05,
                                 10.1,
                                 10.0
    };
    const auto x = homogen_table::wrap(data, 15, 2);

    // With min_cluster_size=2, should find up to 3 clusters
    this->run_checks(x, 2, 2, -1);

    // With min_cluster_size=4, cluster A (3 pts) is too small
    this->run_checks(x, 4, 2, -1);

    // With min_cluster_size=6, only cluster C (7 pts) is large enough
    this->run_checks(x, 6, 2, -1);
}

// =========================================================================
// brute_force metric tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: manhattan two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    this->run_checks(x, 5, 5, distance_metric::manhattan, 2.0, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: chebyshev two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    this->run_checks(x, 5, 5, distance_metric::chebyshev, 2.0, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: minkowski p=3 two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    this->run_checks(x, 5, 5, distance_metric::minkowski, 3.0, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: cosine two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    // For cosine, clusters must differ in direction, not just magnitude.
    // 8 points per cluster with wider angular spread for stability.
    // Cluster 0: points near direction (1, 0)
    // Cluster 1: points near direction (0, 1)
    constexpr float_t data[] = {
        10.0, 0.5, //
        10.0, 1.0, //
        10.0, 0.0, //
        10.0, 0.8, //
        10.0, 0.3, //
        10.0, 0.6, //
        10.0, 0.4, //
        10.0, 0.9, //
        0.5,  10.0, //
        1.0,  10.0, //
        0.0,  10.0, //
        0.8,  10.0, //
        0.3,  10.0, //
        0.6,  10.0, //
        0.4,  10.0, //
        0.9,  10.0, //
    };
    const auto x = homogen_table::wrap(data, 16, 2);

    this->run_checks(x, 5, 5, distance_metric::cosine, 2.0, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: minkowski(p=1) equals manhattan",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
        5.0,   5.0, //
    };
    const std::int64_t row_count = 11;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    const auto mink_desc = hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
                               .set_result_options(result_options::responses)
                               .set_metric(distance_metric::minkowski)
                               .set_degree(1.0);
    const auto manh_desc = hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
                               .set_result_options(result_options::responses)
                               .set_metric(distance_metric::manhattan);

    const auto mink_result = dal::compute(mink_desc, x);
    const auto manh_result = dal::compute(manh_desc, x);

    REQUIRE(mink_result.get_cluster_count() == manh_result.get_cluster_count());

    const auto mink_rows = row_accessor<const float_t>(mink_result.get_responses()).pull({ 0, -1 });
    const auto manh_rows = row_accessor<const float_t>(manh_result.get_responses()).pull({ 0, -1 });

    check_same_partition(mink_rows, manh_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: minkowski(p=2) equals euclidean",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
        5.0,   5.0, //
    };
    const std::int64_t row_count = 11;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    const auto mink_desc = hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
                               .set_result_options(result_options::responses)
                               .set_metric(distance_metric::minkowski)
                               .set_degree(2.0);
    const auto eucl_desc = hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
                               .set_result_options(result_options::responses)
                               .set_metric(distance_metric::euclidean);

    const auto mink_result = dal::compute(mink_desc, x);
    const auto eucl_result = dal::compute(eucl_desc, x);

    REQUIRE(mink_result.get_cluster_count() == eucl_result.get_cluster_count());

    const auto mink_rows = row_accessor<const float_t>(mink_result.get_responses()).pull({ 0, -1 });
    const auto eucl_rows = row_accessor<const float_t>(eucl_result.get_responses()).pull({ 0, -1 });

    check_same_partition(mink_rows, eucl_rows, row_count);
}

// =========================================================================
// kd_tree method tests
// =========================================================================

using hdbscan_kd_types = COMBINE_TYPES((float, double), (hdbscan::method::kd_tree));

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: two well-separated clusters",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    this->run_checks(x, 5, 5, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: gold data test",
                     "[hdbscan][batch][gold]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());

    std::int64_t min_cluster_size = gold_dataset::get_min_cluster_size();
    std::int64_t min_samples = gold_dataset::get_min_samples();

    this->run_checks(x, min_cluster_size, min_samples, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: all noise when min_cluster_size > n",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    const auto x = homogen_table::wrap(data, 5, 1);

    this->run_checks(x, 10, 2, 0);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: small min_cluster_size",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0, 0.0, //
        0.1, 0.1, //
        0.2, 0.0, //
        5.0, 5.0, //
        5.1, 5.1, //
        5.2, 5.0, //
    };
    const auto x = homogen_table::wrap(data, 6, 2);

    this->run_checks(x, 2, 2, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: 1D data three clusters",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0,  0.1,  0.2,  0.15,  0.05, //
                                 5.0,  5.1,  5.2,  5.15,  5.05, //
                                 10.0, 10.1, 10.2, 10.15, 10.05 };
    const auto x = homogen_table::wrap(data, 15, 1);

    this->run_checks(x, 5, 5, 3);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: high-dimensional data",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr std::int64_t n = 10;
    constexpr std::int64_t d = 10;
    float_t data[n * d];

    for (std::int64_t i = 0; i < 5; i++)
        for (std::int64_t j = 0; j < d; j++)
            data[i * d + j] = static_cast<float_t>(0.01 * (i + j));

    for (std::int64_t i = 5; i < 10; i++)
        for (std::int64_t j = 0; j < d; j++)
            data[i * d + j] = static_cast<float_t>(10.0 + 0.01 * (i + j));

    const auto x = homogen_table::wrap(data, n, d);

    this->run_checks(x, 5, 5, 2);
}

// =========================================================================
// kd_tree metric tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: manhattan two clusters",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    this->run_checks(x, 5, 5, distance_metric::manhattan, 2.0, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: chebyshev two clusters",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    this->run_checks(x, 5, 5, distance_metric::chebyshev, 2.0, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: minkowski p=3 two clusters",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    this->run_checks(x, 5, 5, distance_metric::minkowski, 3.0, 2);
}

// =========================================================================
// Cross-method consistency tests: brute_force vs kd_tree
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: same partition on gold data",
                     "[hdbscan][batch][gold]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    const std::int64_t row_count = gold_dataset::get_row_count();
    std::int64_t min_cluster_size = gold_dataset::get_min_cluster_size();
    std::int64_t min_samples = gold_dataset::get_min_samples();

    const auto bf_desc =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);
    const auto kd_desc =
        hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    INFO("run brute_force");
    const auto bf_result = dal::compute(bf_desc, x);

    INFO("run kd_tree");
    const auto kd_result = dal::compute(kd_desc, x);

    INFO("compare cluster counts");
    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    INFO("compare partitions (permutation-invariant)");
    const auto bf_rows = row_accessor<const float_t>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const float_t>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: same partition on two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
        5.0,   5.0, //
    };
    const std::int64_t row_count = 11;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t min_cluster_size = 5;
    constexpr std::int64_t min_samples = 5;

    const auto bf_desc =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);
    const auto kd_desc =
        hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    const auto bf_result = dal::compute(bf_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bf_rows = row_accessor<const float_t>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const float_t>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: three clusters 1D",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0,  0.1,  0.2,  0.15,  0.05, //
                                 5.0,  5.1,  5.2,  5.15,  5.05, //
                                 10.0, 10.1, 10.2, 10.15, 10.05 };
    const std::int64_t row_count = 15;
    const auto x = homogen_table::wrap(data, row_count, 1);

    const auto bf_desc =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(5, 5).set_result_options(
            result_options::responses);
    const auto kd_desc =
        hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(5, 5).set_result_options(
            result_options::responses);

    const auto bf_result = dal::compute(bf_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bf_rows = row_accessor<const float_t>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const float_t>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

// =========================================================================
// Cross-method consistency: non-Euclidean metrics
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: manhattan consistency",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
        5.0,   5.0, //
    };
    const std::int64_t row_count = 11;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    const auto bf_desc = hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::manhattan);
    const auto kd_desc = hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::manhattan);

    const auto bf_result = dal::compute(bf_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bf_rows = row_accessor<const float_t>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const float_t>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: chebyshev consistency",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
        5.0,   5.0, //
    };
    const std::int64_t row_count = 11;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    const auto bf_desc = hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::chebyshev);
    const auto kd_desc = hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::chebyshev);

    const auto bf_result = dal::compute(bf_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bf_rows = row_accessor<const float_t>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const float_t>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: minkowski p=3 consistency",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
        5.0,   5.0, //
    };
    const std::int64_t row_count = 11;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    const auto bf_desc = hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::minkowski)
                             .set_degree(3.0);
    const auto kd_desc = hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::minkowski)
                             .set_degree(3.0);

    const auto bf_result = dal::compute(bf_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bf_rows = row_accessor<const float_t>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const float_t>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

// =========================================================================
// cluster_selection_epsilon tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: epsilon merges close clusters",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    // Three clusters: A and B are close (distance ~1.0), C is far (distance ~10.0)
    constexpr float_t data[] = {
        0.0,  0.0, //
        0.05, 0.05, //
        0.1,  0.0, //
        0.0,  0.1, //
        0.08, 0.08, //
        1.0,  1.0, //
        1.05, 1.05, //
        1.1,  1.0, //
        1.0,  1.1, //
        1.08, 1.08, //
        10.0, 10.0, //
        10.05, 10.05, //
        10.1, 10.0, //
        10.0, 10.1, //
        10.08, 10.08, //
    };
    const std::int64_t row_count = 15;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    // Without epsilon: should find 3 clusters
    const auto desc_no_eps =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
            .set_result_options(result_options::responses)
            .set_cluster_selection_epsilon(0.0);
    const auto result_no_eps = dal::compute(desc_no_eps, x);
    const auto count_no_eps = result_no_eps.get_cluster_count();

    // With large epsilon: should merge close clusters, resulting in fewer clusters
    const auto desc_eps =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
            .set_result_options(result_options::responses)
            .set_cluster_selection_epsilon(5.0);
    const auto result_eps = dal::compute(desc_eps, x);
    const auto count_eps = result_eps.get_cluster_count();

    INFO("epsilon should merge clusters, resulting in <= original count");
    REQUIRE(count_eps <= count_no_eps);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: epsilon merges close clusters",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,  0.0, //
        0.05, 0.05, //
        0.1,  0.0, //
        0.0,  0.1, //
        0.08, 0.08, //
        1.0,  1.0, //
        1.05, 1.05, //
        1.1,  1.0, //
        1.0,  1.1, //
        1.08, 1.08, //
        10.0, 10.0, //
        10.05, 10.05, //
        10.1, 10.0, //
        10.0, 10.1, //
        10.08, 10.08, //
    };
    const std::int64_t row_count = 15;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    const auto desc_no_eps =
        hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(mcs, ms)
            .set_result_options(result_options::responses)
            .set_cluster_selection_epsilon(0.0);
    const auto result_no_eps = dal::compute(desc_no_eps, x);

    const auto desc_eps =
        hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(mcs, ms)
            .set_result_options(result_options::responses)
            .set_cluster_selection_epsilon(5.0);
    const auto result_eps = dal::compute(desc_eps, x);

    REQUIRE(result_eps.get_cluster_count() <= result_no_eps.get_cluster_count());
}

// =========================================================================
// alpha tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: alpha=1 is default behavior",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const std::int64_t row_count = 10;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    // Default (alpha=1.0) and explicit alpha=1.0 should give same result
    const auto desc_default =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
            .set_result_options(result_options::responses);
    const auto desc_alpha1 =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
            .set_result_options(result_options::responses)
            .set_alpha(1.0);

    const auto result_default = dal::compute(desc_default, x);
    const auto result_alpha1 = dal::compute(desc_alpha1, x);

    REQUIRE(result_default.get_cluster_count() == result_alpha1.get_cluster_count());

    const auto rows_default =
        row_accessor<const float_t>(result_default.get_responses()).pull({ 0, -1 });
    const auto rows_alpha1 =
        row_accessor<const float_t>(result_alpha1.get_responses()).pull({ 0, -1 });

    check_same_partition(rows_default, rows_alpha1, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: alpha > 1 runs without error",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    const auto desc =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(5, 5)
            .set_result_options(result_options::responses)
            .set_alpha(1.5);
    REQUIRE_NOTHROW(dal::compute(desc, x));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: alpha > 1 runs without error",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    const auto desc =
        hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(5, 5)
            .set_result_options(result_options::responses)
            .set_alpha(1.5);
    REQUIRE_NOTHROW(dal::compute(desc, x));
}

// =========================================================================
// leaf_size tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: different leaf_size gives same partition",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const std::int64_t row_count = 10;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    const auto desc_leaf10 =
        hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(mcs, ms)
            .set_result_options(result_options::responses)
            .set_leaf_size(10);
    const auto desc_leaf40 =
        hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(mcs, ms)
            .set_result_options(result_options::responses)
            .set_leaf_size(40);

    const auto result_leaf10 = dal::compute(desc_leaf10, x);
    const auto result_leaf40 = dal::compute(desc_leaf40, x);

    REQUIRE(result_leaf10.get_cluster_count() == result_leaf40.get_cluster_count());

    const auto rows_10 =
        row_accessor<const float_t>(result_leaf10.get_responses()).pull({ 0, -1 });
    const auto rows_40 =
        row_accessor<const float_t>(result_leaf40.get_responses()).pull({ 0, -1 });

    check_same_partition(rows_10, rows_40, row_count);
}

// =========================================================================
// max_cluster_size tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: max_cluster_size=0 is no limit",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const std::int64_t row_count = 10;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    // max_cluster_size=0 should behave same as default
    const auto desc_default =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
            .set_result_options(result_options::responses);
    const auto desc_max0 =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms)
            .set_result_options(result_options::responses)
            .set_max_cluster_size(0);

    const auto result_default = dal::compute(desc_default, x);
    const auto result_max0 = dal::compute(desc_max0, x);

    REQUIRE(result_default.get_cluster_count() == result_max0.get_cluster_count());

    const auto rows_default =
        row_accessor<const float_t>(result_default.get_responses()).pull({ 0, -1 });
    const auto rows_max0 =
        row_accessor<const float_t>(result_max0.get_responses()).pull({ 0, -1 });

    check_same_partition(rows_default, rows_max0, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: max_cluster_size limits cluster size",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    // max_cluster_size should run without error (functional check)
    const auto desc =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(5, 5)
            .set_result_options(result_options::responses)
            .set_max_cluster_size(3);
    REQUIRE_NOTHROW(dal::compute(desc, x));
}

// =========================================================================
// Nightly tests: external datasets
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: susy 500K samples",
                     "[hdbscan][nightly][batch][external-dataset][susy]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe data =
        te::dataframe_builder{ "workloads/susy/dataset/susy_test.csv" }.build();
    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    // SUSY: 500K x 18, min_cluster_size=50, min_samples=25
    this->run_checks(x, 50, 25, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: susy 500K samples",
                     "[hdbscan][nightly][batch][external-dataset][susy]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe data =
        te::dataframe_builder{ "workloads/susy/dataset/susy_test.csv" }.build();
    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    // Use kd_tree method explicitly via descriptor
    using float_t = std::tuple_element_t<0, TestType>;

    const auto desc =
        hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(50, 25).set_result_options(
            result_options::responses);

    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    INFO("check cluster count >= 0");
    REQUIRE(result.get_cluster_count() >= 0);

    INFO("check responses shape");
    const auto responses = result.get_responses();
    REQUIRE(responses.get_row_count() == x.get_row_count());
    REQUIRE(responses.get_column_count() == 1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: susy consistency",
                     "[hdbscan][nightly][batch][external-dataset][susy]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    const te::dataframe data =
        te::dataframe_builder{ "workloads/susy/dataset/susy_test.csv" }.build();
    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    const std::int64_t row_count = x.get_row_count();
    constexpr std::int64_t mcs = 50;
    constexpr std::int64_t ms = 25;

    const auto bf_desc =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(mcs, ms).set_result_options(
            result_options::responses);
    const auto kd_desc =
        hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(mcs, ms).set_result_options(
            result_options::responses);

    const auto bf_result = oneapi::dal::test::engine::compute(this->get_policy(), bf_desc, x);
    const auto kd_result = oneapi::dal::test::engine::compute(this->get_policy(), kd_desc, x);

    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bf_rows = row_accessor<const float_t>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const float_t>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

// =========================================================================
// ball_tree method tests
// =========================================================================

using hdbscan_bt_types = COMBINE_TYPES((float, double), (hdbscan::method::ball_tree));
using hdbscan_bt_only = COMBINE_TYPES((double), (hdbscan::method::ball_tree));

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: two well-separated clusters",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    this->run_checks(x, 5, 5, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: gold data test",
                     "[hdbscan][batch][gold]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());

    std::int64_t min_cluster_size = gold_dataset::get_min_cluster_size();
    std::int64_t min_samples = gold_dataset::get_min_samples();

    this->run_checks(x, min_cluster_size, min_samples, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: all noise when min_cluster_size > n",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    const auto x = homogen_table::wrap(data, 5, 1);

    this->run_checks(x, 10, 2, 0);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: small min_cluster_size",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0, 0.0, //
        0.1, 0.1, //
        0.2, 0.0, //
        5.0, 5.0, //
        5.1, 5.1, //
        5.2, 5.0, //
    };
    const auto x = homogen_table::wrap(data, 6, 2);

    this->run_checks(x, 2, 2, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: 1D data three clusters",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0,  0.1,  0.2,  0.15,  0.05, //
                                 5.0,  5.1,  5.2,  5.15,  5.05, //
                                 10.0, 10.1, 10.2, 10.15, 10.05 };
    const auto x = homogen_table::wrap(data, 15, 1);

    this->run_checks(x, 5, 5, 3);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: high-dimensional data",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr std::int64_t n = 10;
    constexpr std::int64_t d = 10;
    float_t data[n * d];

    for (std::int64_t i = 0; i < 5; i++)
        for (std::int64_t j = 0; j < d; j++)
            data[i * d + j] = static_cast<float_t>(0.01 * (i + j));

    for (std::int64_t i = 5; i < 10; i++)
        for (std::int64_t j = 0; j < d; j++)
            data[i * d + j] = static_cast<float_t>(10.0 + 0.01 * (i + j));

    const auto x = homogen_table::wrap(data, n, d);

    this->run_checks(x, 5, 5, 2);
}

// =========================================================================
// ball_tree metric tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: manhattan two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    this->run_checks(x, 5, 5, distance_metric::manhattan, 2.0, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: chebyshev two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    this->run_checks(x, 5, 5, distance_metric::chebyshev, 2.0, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: minkowski p=3 two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    this->run_checks(x, 5, 5, distance_metric::minkowski, 3.0, 2);
}

// =========================================================================
// Cross-method consistency: ball_tree vs brute_force and kd_tree
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree vs brute_force: same partition on gold data",
                     "[hdbscan][batch][gold]",
                     hdbscan_bt_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    const std::int64_t row_count = gold_dataset::get_row_count();
    std::int64_t min_cluster_size = gold_dataset::get_min_cluster_size();
    std::int64_t min_samples = gold_dataset::get_min_samples();

    const auto bt_desc =
        hdbscan::descriptor<float_t, hdbscan::method::ball_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);
    const auto bf_desc =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    const auto bt_result = dal::compute(bt_desc, x);
    const auto bf_result = dal::compute(bf_desc, x);

    REQUIRE(bt_result.get_cluster_count() == bf_result.get_cluster_count());

    const auto bt_rows = row_accessor<const float_t>(bt_result.get_responses()).pull({ 0, -1 });
    const auto bf_rows = row_accessor<const float_t>(bf_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bt_rows, bf_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree vs kd_tree: same partition on two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bt_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
        5.0,   5.0, //
    };
    const std::int64_t row_count = 11;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t min_cluster_size = 5;
    constexpr std::int64_t min_samples = 5;

    const auto bt_desc =
        hdbscan::descriptor<float_t, hdbscan::method::ball_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);
    const auto kd_desc =
        hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    const auto bt_result = dal::compute(bt_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bt_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bt_rows = row_accessor<const float_t>(bt_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const float_t>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bt_rows, kd_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree vs kd_tree: manhattan consistency",
                     "[hdbscan][batch]",
                     hdbscan_bt_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
        5.0,   5.0, //
    };
    const std::int64_t row_count = 11;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    const auto bt_desc = hdbscan::descriptor<float_t, hdbscan::method::ball_tree>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::manhattan);
    const auto kd_desc = hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::manhattan);

    const auto bt_result = dal::compute(bt_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bt_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bt_rows = row_accessor<const float_t>(bt_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const float_t>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bt_rows, kd_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: alpha > 1 runs without error",
                     "[hdbscan][batch]",
                     hdbscan_bt_only) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    const auto desc =
        hdbscan::descriptor<float_t, hdbscan::method::ball_tree>(5, 5)
            .set_result_options(result_options::responses)
            .set_alpha(1.5);
    REQUIRE_NOTHROW(dal::compute(desc, x));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: different leaf_size gives same partition",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const std::int64_t row_count = 10;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    const auto desc_leaf10 =
        hdbscan::descriptor<float_t, hdbscan::method::ball_tree>(mcs, ms)
            .set_result_options(result_options::responses)
            .set_leaf_size(10);
    const auto desc_leaf40 =
        hdbscan::descriptor<float_t, hdbscan::method::ball_tree>(mcs, ms)
            .set_result_options(result_options::responses)
            .set_leaf_size(40);

    const auto result_leaf10 = dal::compute(desc_leaf10, x);
    const auto result_leaf40 = dal::compute(desc_leaf40, x);

    REQUIRE(result_leaf10.get_cluster_count() == result_leaf40.get_cluster_count());

    const auto rows_10 =
        row_accessor<const float_t>(result_leaf10.get_responses()).pull({ 0, -1 });
    const auto rows_40 =
        row_accessor<const float_t>(result_leaf40.get_responses()).pull({ 0, -1 });

    check_same_partition(rows_10, rows_40, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: epsilon merges close clusters",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,  0.0, //
        0.05, 0.05, //
        0.1,  0.0, //
        0.0,  0.1, //
        0.08, 0.08, //
        1.0,  1.0, //
        1.05, 1.05, //
        1.1,  1.0, //
        1.0,  1.1, //
        1.08, 1.08, //
        10.0, 10.0, //
        10.05, 10.05, //
        10.1, 10.0, //
        10.0, 10.1, //
        10.08, 10.08, //
    };
    const std::int64_t row_count = 15;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    const auto desc_no_eps =
        hdbscan::descriptor<float_t, hdbscan::method::ball_tree>(mcs, ms)
            .set_result_options(result_options::responses)
            .set_cluster_selection_epsilon(0.0);
    const auto result_no_eps = dal::compute(desc_no_eps, x);

    const auto desc_eps =
        hdbscan::descriptor<float_t, hdbscan::method::ball_tree>(mcs, ms)
            .set_result_options(result_options::responses)
            .set_cluster_selection_epsilon(5.0);
    const auto result_eps = dal::compute(desc_eps, x);

    REQUIRE(result_eps.get_cluster_count() <= result_no_eps.get_cluster_count());
}

// =========================================================================
// GPU tests (conditional on ONEDAL_DATA_PARALLEL)
// =========================================================================

#ifdef ONEDAL_DATA_PARALLEL

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: cpu and gpu results match",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
        5.0,   5.0, //
    };
    const std::int64_t row_count = 11;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t min_cluster_size = 5;
    constexpr std::int64_t min_samples = 5;

    const auto desc =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    INFO("run on CPU (no queue)");
    const auto cpu_result = dal::compute(desc, x);

    INFO("run on GPU (with queue)");
    const auto gpu_result = dal::compute(this->get_policy().get_queue(), desc, x);

    INFO("compare CPU vs GPU responses (permutation-invariant)");
    REQUIRE(cpu_result.get_cluster_count() == gpu_result.get_cluster_count());

    const auto cpu_rows = row_accessor<const float_t>(cpu_result.get_responses()).pull({ 0, -1 });
    const auto gpu_rows = row_accessor<const float_t>(gpu_result.get_responses()).pull({ 0, -1 });

    check_same_partition(cpu_rows, gpu_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: cpu and gpu results match on gold data",
                     "[hdbscan][batch][gold]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    const std::int64_t row_count = gold_dataset::get_row_count();

    std::int64_t min_cluster_size = gold_dataset::get_min_cluster_size();
    std::int64_t min_samples = gold_dataset::get_min_samples();

    using float_t = std::tuple_element_t<0, TestType>;
    const auto desc =
        hdbscan::descriptor<float_t, hdbscan::method::brute_force>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    INFO("run on CPU (no queue)");
    const auto cpu_result = dal::compute(desc, x);

    INFO("run on GPU (with queue)");
    const auto gpu_result = dal::compute(this->get_policy().get_queue(), desc, x);

    REQUIRE(cpu_result.get_cluster_count() == gpu_result.get_cluster_count());

    const auto cpu_rows = row_accessor<const float_t>(cpu_result.get_responses()).pull({ 0, -1 });
    const auto gpu_rows = row_accessor<const float_t>(gpu_result.get_responses()).pull({ 0, -1 });

    check_same_partition(cpu_rows, gpu_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: cpu and gpu results match on gold data",
                     "[hdbscan][batch][gold]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    const std::int64_t row_count = gold_dataset::get_row_count();

    std::int64_t min_cluster_size = gold_dataset::get_min_cluster_size();
    std::int64_t min_samples = gold_dataset::get_min_samples();

    using float_t = std::tuple_element_t<0, TestType>;
    const auto desc =
        hdbscan::descriptor<float_t, hdbscan::method::kd_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    INFO("run on CPU (no queue)");
    const auto cpu_result = dal::compute(desc, x);

    INFO("run on GPU (with queue)");
    const auto gpu_result = dal::compute(this->get_policy().get_queue(), desc, x);

    REQUIRE(cpu_result.get_cluster_count() == gpu_result.get_cluster_count());

    const auto cpu_rows = row_accessor<const float_t>(cpu_result.get_responses()).pull({ 0, -1 });
    const auto gpu_rows = row_accessor<const float_t>(gpu_result.get_responses()).pull({ 0, -1 });

    check_same_partition(cpu_rows, gpu_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: cpu and gpu results match on gold data",
                     "[hdbscan][batch][gold]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    const std::int64_t row_count = gold_dataset::get_row_count();

    std::int64_t min_cluster_size = gold_dataset::get_min_cluster_size();
    std::int64_t min_samples = gold_dataset::get_min_samples();

    using float_t = std::tuple_element_t<0, TestType>;
    const auto desc =
        hdbscan::descriptor<float_t, hdbscan::method::ball_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    INFO("run on CPU (no queue)");
    const auto cpu_result = dal::compute(desc, x);

    INFO("run on GPU (with queue)");
    const auto gpu_result = dal::compute(this->get_policy().get_queue(), desc, x);

    REQUIRE(cpu_result.get_cluster_count() == gpu_result.get_cluster_count());

    const auto cpu_rows = row_accessor<const float_t>(cpu_result.get_responses()).pull({ 0, -1 });
    const auto gpu_rows = row_accessor<const float_t>(gpu_result.get_responses()).pull({ 0, -1 });

    check_same_partition(cpu_rows, gpu_rows, row_count);
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::hdbscan::test
