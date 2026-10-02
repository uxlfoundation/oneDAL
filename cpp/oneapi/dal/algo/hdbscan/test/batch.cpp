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

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: compute mode check",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = { 0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2,  0.15, 0.15, 5.0,
                               5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2, 5.15, 5.15, 10.0, 0.0 };
    const auto x = homogen_table::wrap(data, 11, 2);

    constexpr std::int64_t min_cluster_size = 5;
    constexpr std::int64_t min_samples = 5;

    result_option_id res_all = result_option_id(dal::result_option_id_base(mask_full));

    const result_option_id compute_mode = GENERATE_COPY(result_options::responses,
                                                        result_options::core_flags,
                                                        result_options::core_observations,
                                                        result_options::core_observation_indices,
                                                        result_options::probabilities,
                                                        res_all);

    this->mode_checks(compute_mode, x, min_cluster_size, min_samples);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: all noise when min_cluster_size > n",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = { 0.0, 0.0, 1.0, 1.0 };
    const auto x = homogen_table::wrap(data, 2, 2);

    // min_cluster_size=2: both points should form a cluster
    this->run_checks(x, 2, 2, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: all identical points",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    // Three clusters in 1D, well-separated
    constexpr Float data[] = { 0.0,  0.1,  0.2,  0.15,  0.05, //
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
    using Float = std::tuple_element_t<0, TestType>;

    // 10-dimensional data with 2 clusters
    constexpr std::int64_t n = 10;
    constexpr std::int64_t d = 10;
    Float data[n * d];

    // Cluster 0: near origin
    for (std::int64_t i = 0; i < 5; i++)
        for (std::int64_t j = 0; j < d; j++)
            data[i * d + j] = static_cast<Float>(0.01 * (i + j));

    // Cluster 1: far from origin
    for (std::int64_t i = 5; i < 10; i++)
        for (std::int64_t j = 0; j < d; j++)
            data[i * d + j] = static_cast<Float>(10.0 + 0.01 * (i + j));

    const auto x = homogen_table::wrap(data, n, d);

    this->run_checks(x, 5, 5, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: min_samples=1",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    // Three clusters with varying sizes: 3, 5, 7 points
    constexpr Float data[] = { // Cluster A: 3 points
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // For cosine, clusters must differ in direction, not just magnitude.
    // 8 points per cluster with wider angular spread for stability.
    // Cluster 0: points near direction (1, 0)
    // Cluster 1: points near direction (0, 1)
    constexpr Float data[] = {
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
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    const auto mink_desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms)
                               .set_result_options(result_options::responses)
                               .set_metric(distance_metric::minkowski)
                               .set_degree(1.0);
    const auto manh_desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms)
                               .set_result_options(result_options::responses)
                               .set_metric(distance_metric::manhattan);

    const auto mink_result = dal::compute(mink_desc, x);
    const auto manh_result = dal::compute(manh_desc, x);

    REQUIRE(mink_result.get_cluster_count() == manh_result.get_cluster_count());

    const auto mink_rows = row_accessor<const Float>(mink_result.get_responses()).pull({ 0, -1 });
    const auto manh_rows = row_accessor<const Float>(manh_result.get_responses()).pull({ 0, -1 });

    check_same_partition(mink_rows, manh_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: minkowski(p=2) equals euclidean",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    const auto mink_desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms)
                               .set_result_options(result_options::responses)
                               .set_metric(distance_metric::minkowski)
                               .set_degree(2.0);
    const auto eucl_desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms)
                               .set_result_options(result_options::responses)
                               .set_metric(distance_metric::euclidean);

    const auto mink_result = dal::compute(mink_desc, x);
    const auto eucl_result = dal::compute(eucl_desc, x);

    REQUIRE(mink_result.get_cluster_count() == eucl_result.get_cluster_count());

    const auto mink_rows = row_accessor<const Float>(mink_result.get_responses()).pull({ 0, -1 });
    const auto eucl_rows = row_accessor<const Float>(eucl_result.get_responses()).pull({ 0, -1 });

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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    const auto x = homogen_table::wrap(data, 5, 1);

    this->run_checks(x, 10, 2, 0);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: small min_cluster_size",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = { 0.0,  0.1,  0.2,  0.15,  0.05, //
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr std::int64_t n = 10;
    constexpr std::int64_t d = 10;
    Float data[n * d];

    for (std::int64_t i = 0; i < 5; i++)
        for (std::int64_t j = 0; j < d; j++)
            data[i * d + j] = static_cast<Float>(0.01 * (i + j));

    for (std::int64_t i = 5; i < 10; i++)
        for (std::int64_t j = 0; j < d; j++)
            data[i * d + j] = static_cast<Float>(10.0 + 0.01 * (i + j));

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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    const std::int64_t row_count = gold_dataset::get_row_count();
    std::int64_t min_cluster_size = gold_dataset::get_min_cluster_size();
    std::int64_t min_samples = gold_dataset::get_min_samples();

    const auto bf_desc =
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);
    const auto kd_desc =
        hdbscan::descriptor<Float, hdbscan::method::kd_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    INFO("run brute_force");
    const auto bf_result = dal::compute(bf_desc, x);

    INFO("run kd_tree");
    const auto kd_result = dal::compute(kd_desc, x);

    INFO("compare cluster counts");
    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    INFO("compare partitions (permutation-invariant)");
    const auto bf_rows = row_accessor<const Float>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const Float>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: same partition on two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);
    const auto kd_desc =
        hdbscan::descriptor<Float, hdbscan::method::kd_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    const auto bf_result = dal::compute(bf_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bf_rows = row_accessor<const Float>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const Float>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: three clusters 1D",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = { 0.0,  0.1,  0.2,  0.15,  0.05, //
                               5.0,  5.1,  5.2,  5.15,  5.05, //
                               10.0, 10.1, 10.2, 10.15, 10.05 };
    const std::int64_t row_count = 15;
    const auto x = homogen_table::wrap(data, row_count, 1);

    const auto bf_desc =
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(5, 5).set_result_options(
            result_options::responses);
    const auto kd_desc =
        hdbscan::descriptor<Float, hdbscan::method::kd_tree>(5, 5).set_result_options(
            result_options::responses);

    const auto bf_result = dal::compute(bf_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bf_rows = row_accessor<const Float>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const Float>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

// =========================================================================
// Cross-method consistency: non-Euclidean metrics
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: manhattan consistency",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    const auto bf_desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::manhattan);
    const auto kd_desc = hdbscan::descriptor<Float, hdbscan::method::kd_tree>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::manhattan);

    const auto bf_result = dal::compute(bf_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bf_rows = row_accessor<const Float>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const Float>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: chebyshev consistency",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    const auto bf_desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::chebyshev);
    const auto kd_desc = hdbscan::descriptor<Float, hdbscan::method::kd_tree>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::chebyshev);

    const auto bf_result = dal::compute(bf_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bf_rows = row_accessor<const Float>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const Float>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: minkowski p=3 consistency",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    const auto bf_desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::minkowski)
                             .set_degree(3.0);
    const auto kd_desc = hdbscan::descriptor<Float, hdbscan::method::kd_tree>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::minkowski)
                             .set_degree(3.0);

    const auto bf_result = dal::compute(bf_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bf_rows = row_accessor<const Float>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const Float>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bf_rows, kd_rows, row_count);
}

// =========================================================================
// cluster_selection_epsilon tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: epsilon merges close clusters",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Three clusters: A and B are close (distance ~1.0), C is far (distance ~10.0)
    constexpr Float data[] = {
        0.0,   0.0, //
        0.05,  0.05, //
        0.1,   0.0, //
        0.0,   0.1, //
        0.08,  0.08, //
        1.0,   1.0, //
        1.05,  1.05, //
        1.1,   1.0, //
        1.0,   1.1, //
        1.08,  1.08, //
        10.0,  10.0, //
        10.05, 10.05, //
        10.1,  10.0, //
        10.0,  10.1, //
        10.08, 10.08, //
    };
    const std::int64_t row_count = 15;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    // Without epsilon: should find 3 clusters
    const auto desc_no_eps = hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms)
                                 .set_result_options(result_options::responses)
                                 .set_cluster_selection_epsilon(0.0);
    const auto result_no_eps = dal::compute(desc_no_eps, x);
    const auto count_no_eps = result_no_eps.get_cluster_count();

    // With large epsilon: should merge close clusters, resulting in fewer clusters
    const auto desc_eps = hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms)
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
        0.0,   0.0, //
        0.05,  0.05, //
        0.1,   0.0, //
        0.0,   0.1, //
        0.08,  0.08, //
        1.0,   1.0, //
        1.05,  1.05, //
        1.1,   1.0, //
        1.0,   1.1, //
        1.08,  1.08, //
        10.0,  10.0, //
        10.05, 10.05, //
        10.1,  10.0, //
        10.0,  10.1, //
        10.08, 10.08, //
    };
    const std::int64_t row_count = 15;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    const auto desc_no_eps = hdbscan::descriptor<Float, hdbscan::method::kd_tree>(mcs, ms)
                                 .set_result_options(result_options::responses)
                                 .set_cluster_selection_epsilon(0.0);
    const auto result_no_eps = dal::compute(desc_no_eps, x);

    const auto desc_eps = hdbscan::descriptor<Float, hdbscan::method::kd_tree>(mcs, ms)
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
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms).set_result_options(
            result_options::responses);
    const auto desc_alpha1 = hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms)
                                 .set_result_options(result_options::responses)
                                 .set_alpha(1.0);

    const auto result_default = dal::compute(desc_default, x);
    const auto result_alpha1 = dal::compute(desc_alpha1, x);

    REQUIRE(result_default.get_cluster_count() == result_alpha1.get_cluster_count());

    const auto rows_default =
        row_accessor<const Float>(result_default.get_responses()).pull({ 0, -1 });
    const auto rows_alpha1 =
        row_accessor<const Float>(result_alpha1.get_responses()).pull({ 0, -1 });

    check_same_partition(rows_default, rows_alpha1, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: alpha > 1 runs without error",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    const auto desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(5, 5)
                          .set_result_options(result_options::responses)
                          .set_alpha(1.5);
    REQUIRE_NOTHROW(dal::compute(desc, x));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: alpha > 1 runs without error",
                     "[hdbscan][batch]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    const auto desc = hdbscan::descriptor<Float, hdbscan::method::kd_tree>(5, 5)
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    const auto desc_leaf10 = hdbscan::descriptor<Float, hdbscan::method::kd_tree>(mcs, ms)
                                 .set_result_options(result_options::responses)
                                 .set_leaf_size(10);
    const auto desc_leaf40 = hdbscan::descriptor<Float, hdbscan::method::kd_tree>(mcs, ms)
                                 .set_result_options(result_options::responses)
                                 .set_leaf_size(40);

    const auto result_leaf10 = dal::compute(desc_leaf10, x);
    const auto result_leaf40 = dal::compute(desc_leaf40, x);

    REQUIRE(result_leaf10.get_cluster_count() == result_leaf40.get_cluster_count());

    const auto rows_10 = row_accessor<const Float>(result_leaf10.get_responses()).pull({ 0, -1 });
    const auto rows_40 = row_accessor<const Float>(result_leaf40.get_responses()).pull({ 0, -1 });

    check_same_partition(rows_10, rows_40, row_count);
}

// =========================================================================
// max_cluster_size tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: max_cluster_size=0 is no limit",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms).set_result_options(
            result_options::responses);
    const auto desc_max0 = hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms)
                               .set_result_options(result_options::responses)
                               .set_max_cluster_size(0);

    const auto result_default = dal::compute(desc_default, x);
    const auto result_max0 = dal::compute(desc_max0, x);

    REQUIRE(result_default.get_cluster_count() == result_max0.get_cluster_count());

    const auto rows_default =
        row_accessor<const Float>(result_default.get_responses()).pull({ 0, -1 });
    const auto rows_max0 = row_accessor<const Float>(result_max0.get_responses()).pull({ 0, -1 });

    check_same_partition(rows_default, rows_max0, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: max_cluster_size limits cluster size",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    const auto desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(5, 5)
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
                     hdbscan_bf_types) {
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
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe data =
        te::dataframe_builder{ "workloads/susy/dataset/susy_test.csv" }.build();
    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    // Use kd_tree method explicitly via descriptor
    using Float = std::tuple_element_t<0, TestType>;

    const auto desc =
        hdbscan::descriptor<Float, hdbscan::method::kd_tree>(50, 25).set_result_options(
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
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const te::dataframe data =
        te::dataframe_builder{ "workloads/susy/dataset/susy_test.csv" }.build();
    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    const std::int64_t row_count = x.get_row_count();
    constexpr std::int64_t mcs = 50;
    constexpr std::int64_t ms = 25;

    const auto bf_desc =
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(mcs, ms).set_result_options(
            result_options::responses);
    const auto kd_desc =
        hdbscan::descriptor<Float, hdbscan::method::kd_tree>(mcs, ms).set_result_options(
            result_options::responses);

    const auto bf_result = oneapi::dal::test::engine::compute(this->get_policy(), bf_desc, x);
    const auto kd_result = oneapi::dal::test::engine::compute(this->get_policy(), kd_desc, x);

    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bf_rows = row_accessor<const Float>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const Float>(kd_result.get_responses()).pull({ 0, -1 });

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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    const auto x = homogen_table::wrap(data, 5, 1);

    this->run_checks(x, 10, 2, 0);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: small min_cluster_size",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = { 0.0,  0.1,  0.2,  0.15,  0.05, //
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr std::int64_t n = 10;
    constexpr std::int64_t d = 10;
    Float data[n * d];

    for (std::int64_t i = 0; i < 5; i++)
        for (std::int64_t j = 0; j < d; j++)
            data[i * d + j] = static_cast<Float>(0.01 * (i + j));

    for (std::int64_t i = 5; i < 10; i++)
        for (std::int64_t j = 0; j < d; j++)
            data[i * d + j] = static_cast<Float>(10.0 + 0.01 * (i + j));

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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    const std::int64_t row_count = gold_dataset::get_row_count();
    std::int64_t min_cluster_size = gold_dataset::get_min_cluster_size();
    std::int64_t min_samples = gold_dataset::get_min_samples();

    const auto bt_desc =
        hdbscan::descriptor<Float, hdbscan::method::ball_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);
    const auto bf_desc =
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    const auto bt_result = dal::compute(bt_desc, x);
    const auto bf_result = dal::compute(bf_desc, x);

    REQUIRE(bt_result.get_cluster_count() == bf_result.get_cluster_count());

    const auto bt_rows = row_accessor<const Float>(bt_result.get_responses()).pull({ 0, -1 });
    const auto bf_rows = row_accessor<const Float>(bf_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bt_rows, bf_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree vs kd_tree: same partition on two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bt_only) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
        hdbscan::descriptor<Float, hdbscan::method::ball_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);
    const auto kd_desc =
        hdbscan::descriptor<Float, hdbscan::method::kd_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    const auto bt_result = dal::compute(bt_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bt_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bt_rows = row_accessor<const Float>(bt_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const Float>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bt_rows, kd_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree vs kd_tree: manhattan consistency",
                     "[hdbscan][batch]",
                     hdbscan_bt_only) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    const auto bt_desc = hdbscan::descriptor<Float, hdbscan::method::ball_tree>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::manhattan);
    const auto kd_desc = hdbscan::descriptor<Float, hdbscan::method::kd_tree>(mcs, ms)
                             .set_result_options(result_options::responses)
                             .set_metric(distance_metric::manhattan);

    const auto bt_result = dal::compute(bt_desc, x);
    const auto kd_result = dal::compute(kd_desc, x);

    REQUIRE(bt_result.get_cluster_count() == kd_result.get_cluster_count());

    const auto bt_rows = row_accessor<const Float>(bt_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const Float>(kd_result.get_responses()).pull({ 0, -1 });

    check_same_partition(bt_rows, kd_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: alpha > 1 runs without error",
                     "[hdbscan][batch]",
                     hdbscan_bt_only) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    const auto desc = hdbscan::descriptor<Float, hdbscan::method::ball_tree>(5, 5)
                          .set_result_options(result_options::responses)
                          .set_alpha(1.5);
    REQUIRE_NOTHROW(dal::compute(desc, x));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: different leaf_size gives same partition",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    const auto desc_leaf10 = hdbscan::descriptor<Float, hdbscan::method::ball_tree>(mcs, ms)
                                 .set_result_options(result_options::responses)
                                 .set_leaf_size(10);
    const auto desc_leaf40 = hdbscan::descriptor<Float, hdbscan::method::ball_tree>(mcs, ms)
                                 .set_result_options(result_options::responses)
                                 .set_leaf_size(40);

    const auto result_leaf10 = dal::compute(desc_leaf10, x);
    const auto result_leaf40 = dal::compute(desc_leaf40, x);

    REQUIRE(result_leaf10.get_cluster_count() == result_leaf40.get_cluster_count());

    const auto rows_10 = row_accessor<const Float>(result_leaf10.get_responses()).pull({ 0, -1 });
    const auto rows_40 = row_accessor<const Float>(result_leaf40.get_responses()).pull({ 0, -1 });

    check_same_partition(rows_10, rows_40, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: epsilon merges close clusters",
                     "[hdbscan][batch]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
        0.0,   0.0, //
        0.05,  0.05, //
        0.1,   0.0, //
        0.0,   0.1, //
        0.08,  0.08, //
        1.0,   1.0, //
        1.05,  1.05, //
        1.1,   1.0, //
        1.0,   1.1, //
        1.08,  1.08, //
        10.0,  10.0, //
        10.05, 10.05, //
        10.1,  10.0, //
        10.0,  10.1, //
        10.08, 10.08, //
    };
    const std::int64_t row_count = 15;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr std::int64_t mcs = 5;
    constexpr std::int64_t ms = 5;

    const auto desc_no_eps = hdbscan::descriptor<Float, hdbscan::method::ball_tree>(mcs, ms)
                                 .set_result_options(result_options::responses)
                                 .set_cluster_selection_epsilon(0.0);
    const auto result_no_eps = dal::compute(desc_no_eps, x);

    const auto desc_eps = hdbscan::descriptor<Float, hdbscan::method::ball_tree>(mcs, ms)
                              .set_result_options(result_options::responses)
                              .set_cluster_selection_epsilon(5.0);
    const auto result_eps = dal::compute(desc_eps, x);

    REQUIRE(result_eps.get_cluster_count() <= result_no_eps.get_cluster_count());
}

// =========================================================================
// membership probability tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: probabilities on two well-separated clusters",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    this->run_probability_checks(x, 5, 5);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: probabilities on two well-separated clusters",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    this->run_probability_checks(x, 5, 5);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: probabilities on two well-separated clusters",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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

    this->run_probability_checks(x, 5, 5);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: noise points get zero probability",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // The last two points sit far away from the tight blob and from each other,
    // so no assignment of theirs can survive `min_cluster_size = 5`.
    constexpr Float data[] = {
        0.0,    0.0, //
        0.1,    0.1, //
        0.2,    0.0, //
        0.0,    0.2, //
        0.15,   0.15, //
        100.0,  100.0, //
        -100.0, -100.0, //
    };
    const auto x = homogen_table::wrap(data, 7, 2);

    const auto desc =
        hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(5, 5).set_result_options(
            result_options::responses | result_options::probabilities);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    this->check_probabilities(result, 7);

    const auto responses = row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
    const auto probs = row_accessor<const Float>(result.get_probabilities()).pull({ 0, -1 });
    for (std::int64_t i = 5; i < 7; i++) {
        CAPTURE(i, responses[i], probs[i]);
        REQUIRE(static_cast<std::int32_t>(responses[i]) == -1);
        REQUIRE(probs[i] == Float(0));
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: probabilities are all zero when everything is noise",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    const auto x = homogen_table::wrap(data, 5, 1);

    // `min_cluster_size > n` short-circuits before the MST is built, which is the
    // path that has to zero-fill the probability output on its own.
    const auto desc =
        hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(10, 2).set_result_options(
            result_options::responses | result_options::probabilities);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 0);
    this->check_probabilities(result, 5);

    const auto probs = row_accessor<const Float>(result.get_probabilities()).pull({ 0, -1 });
    for (std::int64_t i = 0; i < 5; i++) {
        CAPTURE(i, probs[i]);
        REQUIRE(probs[i] == Float(0));
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: duplicate points get full membership",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Coincident points give zero-weight MST edges, hence an infinite-in-spirit
    // lambda, so every member persists to the death of its cluster.
    constexpr Float data[] = {
        1.0, 1.0, //
        1.0, 1.0, //
        1.0, 1.0, //
        1.0, 1.0, //
        1.0, 1.0, //
        1.0, 1.0, //
    };
    const auto x = homogen_table::wrap(data, 6, 2);

    const auto desc =
        hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(3, 3).set_result_options(
            result_options::responses | result_options::probabilities);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    this->check_probabilities(result, 6);

    const auto responses = row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
    const auto probs = row_accessor<const Float>(result.get_probabilities()).pull({ 0, -1 });
    for (std::int64_t i = 0; i < 6; i++) {
        CAPTURE(i, responses[i], probs[i]);
        if (static_cast<std::int32_t>(responses[i]) >= 0) {
            REQUIRE(probs[i] == Float(1));
        }
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: probabilities on gold data",
                     "[hdbscan][batch][probabilities][gold]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());

    this->run_probability_checks(x,
                                 gold_dataset::get_min_cluster_size(),
                                 gold_dataset::get_min_samples());
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: probabilities match the reference implementation",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Two blobs of different density, one point that only barely joins the dense
    // blob, and one pure noise point. The expected values come from
    // `sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=5).probabilities_`
    // with the same defaults (euclidean, excess of mass, alpha = 1).
    constexpr Float data[] = {
        0.5917,  -0.1631, //
        0.0115,  0.1426, //
        -0.2761, 0.0007, //
        -0.0003, -0.6142, //
        0.3562,  0.2102, //
        -0.2189, -0.06, //
        0.1769,  -0.0915, //
        -0.085,  -0.5086, //
        0.1941,  0.0434, //
        6.151,   5.1604, //
        6.9079,  6.0849, //
        5.7871,  7.116, //
        5.975,   5.2021, //
        5.7771,  4.7414, //
        6.5772,  5.7709, //
        5.5916,  6.5899, //
        5.0919,  6.2945, //
        4.8646,  5.6358, //
        3.0,     -4.0, //
        -5.0,    7.0, //
    };
    constexpr std::int64_t row_count = 20;
    const auto x = homogen_table::wrap(data, row_count, 2);

    constexpr double ref_probabilities[] = {
        0.6730265741, 1.0, 0.9348524093, 0.6549392602, 1.0,          1.0, 1.0,
        0.8113952439, 1.0, 1.0,          0.9089429453, 0.8214639762, 1.0, 0.9828347746,
        1.0,          1.0, 0.912268997,  1.0,          0.0915445561, 0.0,
    };
    // The reference assigns the two blobs and treats only the last point as noise.
    constexpr std::int32_t ref_noise[] = { 19 };

    const auto desc =
        hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(5, 5).set_result_options(
            result_options::responses | result_options::probabilities);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 2);
    this->check_probabilities(result, row_count);

    const auto responses = row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
    for (const auto i : ref_noise) {
        CAPTURE(i, responses[i]);
        REQUIRE(static_cast<std::int32_t>(responses[i]) == -1);
    }

    // The probability is invariant under a relabeling of the clusters, so unlike
    // the responses it can be compared against the reference entry by entry.
    const auto probs = row_accessor<const Float>(result.get_probabilities()).pull({ 0, -1 });
    const double tol = te::get_tolerance<Float>(1e-4, 1e-7);
    for (std::int64_t i = 0; i < row_count; i++) {
        CAPTURE(i, probs[i], ref_probabilities[i]);
        REQUIRE(std::abs(double(probs[i]) - ref_probabilities[i]) < tol);
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: probabilities can be requested alone",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
        hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(5, 5).set_result_options(
            result_options::probabilities);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE_THROWS_AS(result.get_responses(), domain_error);
    this->check_probabilities(result, 10);
}

// =========================================================================
// single-cluster (root-only) selection tests
// =========================================================================

/// One dense blob plus a ladder of increasingly distant points. The condensed
/// tree of this dataset never splits into two clusters of `min_cluster_size`
/// points, so the root cluster is the only candidate and the outcome is decided
/// entirely by `allow_single_cluster`, `cluster_selection_epsilon` and the
/// cluster selection method. Every reference below comes from
/// `sklearn.cluster.HDBSCAN(min_cluster_size=3, min_samples=3, ...)`.
constexpr std::int64_t single_root_row_count = 16;
constexpr double single_root_data[] = {
    0.5366,  0.131, //
    0.0289,  -0.559, //
    -0.0832, -0.1064, //
    -0.0248, -0.1881, //
    -0.0131, -0.1432, //
    -0.3942, 0.2654, //
    0.2644,  0.5129, //
    0.015,   -0.1214, //
    -0.1636, -0.4639, //
    0.2947,  -0.3303, //
    -0.3555, -0.0617, //
    0.4458,  0.071, //
    1.5,     0.0, //
    2.5,     0.0, //
    4.0,     0.0, //
    7.0,     0.0, //
};

/// Materialize `single_root_data` in the floating-point type under test.
///
/// @tparam Float Floating-point type of the test instantiation
///
/// @return An owning array of `2 * single_root_row_count` feature values
template <typename Float>
static dal::array<Float> make_single_root_data() {
    auto arr = dal::array<Float>::empty(single_root_row_count * 2);
    auto* const dst = arr.get_mutable_data();
    for (std::int64_t i = 0; i < single_root_row_count * 2; i++) {
        dst[i] = static_cast<Float>(single_root_data[i]);
    }
    return arr;
}

/// Compare responses against a pinned reference label vector entry by entry.
///
/// Unlike `check_same_partition` this is not permutation-invariant, which is
/// exactly what a single-cluster reference needs: there is at most one label, so
/// the only question is which points carry it.
///
/// @tparam Float Floating-point type of the response table
///
/// @param[in] responses Response table under test
/// @param[in] ref       Reference labels, length `row_count`
/// @param[in] row_count Number of observations
template <typename Float>
static void check_exact_labels(const table& responses,
                               const std::int32_t* ref,
                               std::int64_t row_count) {
    const auto rows = row_accessor<const Float>(responses).pull({ 0, -1 });
    for (std::int64_t i = 0; i < row_count; i++) {
        CAPTURE(i, rows[i], ref[i]);
        REQUIRE(static_cast<std::int32_t>(rows[i]) == ref[i]);
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: single-cluster selection demotes weak members",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x =
        homogen_table::wrap(make_single_root_data<Float>(), single_root_row_count, std::int64_t(2));

    // Without an epsilon the threshold is the root's own death lambda, so only
    // the three points that persist to the very end of the tree stay clustered.
    // They all drop out at that same lambda, so their membership is 1 and the
    // comparison does not depend on how equal distances are broken.
    constexpr std::int32_t ref_labels[] = { -1, -1, -1, 0,  0,  -1, -1, 0,
                                            -1, -1, -1, -1, -1, -1, -1, -1 };

    const auto desc =
        hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(3, 3)
            .set_allow_single_cluster(true)
            .set_result_options(result_options::responses | result_options::probabilities);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 1);
    check_exact_labels<Float>(result.get_responses(), ref_labels, single_root_row_count);

    const auto probs = row_accessor<const Float>(result.get_probabilities()).pull({ 0, -1 });
    const double tol = te::get_tolerance<Float>(1e-4, 1e-7);
    for (std::int64_t i = 0; i < single_root_row_count; i++) {
        CAPTURE(i, probs[i], ref_labels[i]);
        REQUIRE(std::abs(double(probs[i]) - (ref_labels[i] < 0 ? 0.0 : 1.0)) < tol);
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: cluster_selection_epsilon sets the single-cluster "
                     "threshold",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x =
        homogen_table::wrap(make_single_root_data<Float>(), single_root_row_count, std::int64_t(2));

    // With an epsilon the threshold becomes `1 / epsilon`, a lambda that does not
    // depend on the tree, so both the labels and the membership strengths are
    // fully determined and can be pinned against scikit-learn.
    constexpr std::int32_t ref_labels_eps1[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1
    };
    constexpr double ref_probabilities_eps1[] = {
        0.16561955, 0.22151034, 0.78188754, 1.0,        1.0,        0.16023989, 0.16260124, 1.0,
        0.25156307, 0.22210022, 0.23581156, 0.18113621, 0.07767194, 0.0,        0.0,        0.0,
    };
    constexpr std::int32_t ref_labels_eps3[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 };
    constexpr double ref_probabilities_eps3[] = {
        0.16561955, 0.22151034, 0.78188754, 1.0,        1.0,        0.16023989, 0.16260124, 1.0,
        0.25156307, 0.22210022, 0.23581156, 0.18113621, 0.07767194, 0.05178129, 0.03106878, 0.0,
    };

    const double epsilon = GENERATE(1.0, 3.0);
    const std::int32_t* const ref_labels = (epsilon == 1.0) ? ref_labels_eps1 : ref_labels_eps3;
    const double* const ref_probabilities =
        (epsilon == 1.0) ? ref_probabilities_eps1 : ref_probabilities_eps3;
    CAPTURE(epsilon);

    const auto desc =
        hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(3, 3)
            .set_allow_single_cluster(true)
            .set_cluster_selection_epsilon(epsilon)
            .set_result_options(result_options::responses | result_options::probabilities);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 1);
    check_exact_labels<Float>(result.get_responses(), ref_labels, single_root_row_count);

    const auto probs = row_accessor<const Float>(result.get_probabilities()).pull({ 0, -1 });
    const double tol = te::get_tolerance<Float>(1e-4, 1e-7);
    for (std::int64_t i = 0; i < single_root_row_count; i++) {
        CAPTURE(i, probs[i], ref_probabilities[i]);
        REQUIRE(std::abs(double(probs[i]) - ref_probabilities[i]) < tol);
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: a root-only tree is all noise by default",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x =
        homogen_table::wrap(make_single_root_data<Float>(), single_root_row_count, std::int64_t(2));

    const auto desc =
        hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(3, 3).set_result_options(
            result_options::responses);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 0);
    const auto rows = row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
    for (std::int64_t i = 0; i < single_root_row_count; i++) {
        CAPTURE(i, rows[i]);
        REQUIRE(static_cast<std::int32_t>(rows[i]) == -1);
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: leaf selection never picks the root cluster",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x =
        homogen_table::wrap(make_single_root_data<Float>(), single_root_row_count, std::int64_t(2));

    // Leaf selection picks the leaves of the cluster tree, and the root is not
    // one of them; a tree that never splits has no candidate at all. This holds
    // with `allow_single_cluster` too, which in leaf mode only relaxes the
    // labeling threshold of an already-selected root.
    const bool allow_single_cluster = GENERATE(false, true);
    CAPTURE(allow_single_cluster);

    const auto desc = hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(3, 3)
                          .set_cluster_selection(cluster_selection_method::leaf)
                          .set_allow_single_cluster(allow_single_cluster)
                          .set_result_options(result_options::responses);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 0);
    const auto rows = row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
    for (std::int64_t i = 0; i < single_root_row_count; i++) {
        CAPTURE(i, rows[i]);
        REQUIRE(static_cast<std::int32_t>(rows[i]) == -1);
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: leaf selection still splits a two-cluster dataset",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Guards the leaf-mode change above against over-reach: as soon as the
    // condensed tree does split, leaf selection must still return its leaves.
    constexpr Float data[] = {
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

    const auto desc = hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(3, 3)
                          .set_cluster_selection(cluster_selection_method::leaf)
                          .set_result_options(result_options::responses);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 2);
    const auto rows = row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
    REQUIRE(static_cast<std::int32_t>(rows[0]) >= 0);
    REQUIRE(static_cast<std::int32_t>(rows[5]) >= 0);
    REQUIRE(static_cast<std::int32_t>(rows[0]) != static_cast<std::int32_t>(rows[5]));
    for (std::int64_t i = 0; i < 10; i++) {
        CAPTURE(i, rows[i]);
        REQUIRE(static_cast<std::int32_t>(rows[i]) ==
                static_cast<std::int32_t>(i < 5 ? rows[0] : rows[5]));
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: leaf selection never picks the root cluster",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x =
        homogen_table::wrap(make_single_root_data<Float>(), single_root_row_count, std::int64_t(2));

    const auto desc = hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(3, 3)
                          .set_cluster_selection(cluster_selection_method::leaf)
                          .set_result_options(result_options::responses);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 0);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: leaf selection never picks the root cluster",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x =
        homogen_table::wrap(make_single_root_data<Float>(), single_root_row_count, std::int64_t(2));

    const auto desc = hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(3, 3)
                          .set_cluster_selection(cluster_selection_method::leaf)
                          .set_result_options(result_options::responses);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 0);
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
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    INFO("run on CPU (no queue)");
    const auto cpu_result = dal::compute(desc, x);

    INFO("run on GPU (with queue)");
    const auto gpu_result = dal::compute(this->get_policy().get_queue(), desc, x);

    INFO("compare CPU vs GPU responses (permutation-invariant)");
    REQUIRE(cpu_result.get_cluster_count() == gpu_result.get_cluster_count());

    const auto cpu_rows = row_accessor<const Float>(cpu_result.get_responses()).pull({ 0, -1 });
    const auto gpu_rows = row_accessor<const Float>(gpu_result.get_responses()).pull({ 0, -1 });

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

    using Float = std::tuple_element_t<0, TestType>;
    const auto desc =
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    INFO("run on CPU (no queue)");
    const auto cpu_result = dal::compute(desc, x);

    INFO("run on GPU (with queue)");
    const auto gpu_result = dal::compute(this->get_policy().get_queue(), desc, x);

    REQUIRE(cpu_result.get_cluster_count() == gpu_result.get_cluster_count());

    const auto cpu_rows = row_accessor<const Float>(cpu_result.get_responses()).pull({ 0, -1 });
    const auto gpu_rows = row_accessor<const Float>(gpu_result.get_responses()).pull({ 0, -1 });

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

    using Float = std::tuple_element_t<0, TestType>;
    const auto desc =
        hdbscan::descriptor<Float, hdbscan::method::kd_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    INFO("run on CPU (no queue)");
    const auto cpu_result = dal::compute(desc, x);

    INFO("run on GPU (with queue)");
    const auto gpu_result = dal::compute(this->get_policy().get_queue(), desc, x);

    REQUIRE(cpu_result.get_cluster_count() == gpu_result.get_cluster_count());

    const auto cpu_rows = row_accessor<const Float>(cpu_result.get_responses()).pull({ 0, -1 });
    const auto gpu_rows = row_accessor<const Float>(gpu_result.get_responses()).pull({ 0, -1 });

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

    using Float = std::tuple_element_t<0, TestType>;
    const auto desc =
        hdbscan::descriptor<Float, hdbscan::method::ball_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    INFO("run on CPU (no queue)");
    const auto cpu_result = dal::compute(desc, x);

    INFO("run on GPU (with queue)");
    const auto gpu_result = dal::compute(this->get_policy().get_queue(), desc, x);

    REQUIRE(cpu_result.get_cluster_count() == gpu_result.get_cluster_count());

    const auto cpu_rows = row_accessor<const Float>(cpu_result.get_responses()).pull({ 0, -1 });
    const auto gpu_rows = row_accessor<const Float>(gpu_result.get_responses()).pull({ 0, -1 });

    check_same_partition(cpu_rows, gpu_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: cpu and gpu probabilities match",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr Float data[] = {
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
    constexpr std::int64_t row_count = 10;
    const auto x = homogen_table::wrap(data, row_count, 2);

    const auto desc =
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(5, 5).set_result_options(
            result_options::responses | result_options::probabilities);

    INFO("run on CPU (no queue)");
    const auto cpu_result = dal::compute(desc, x);

    INFO("run on GPU (with queue)");
    const auto gpu_result = dal::compute(this->get_policy().get_queue(), desc, x);

    REQUIRE(cpu_result.get_cluster_count() == gpu_result.get_cluster_count());

    const auto cpu_labels = row_accessor<const Float>(cpu_result.get_responses()).pull({ 0, -1 });
    const auto gpu_labels = row_accessor<const Float>(gpu_result.get_responses()).pull({ 0, -1 });
    check_same_partition(cpu_labels, gpu_labels, row_count);

    // The probability is a per-point quantity, so unlike the labels it needs no
    // permutation matching: the two backends have to agree entry by entry.
    const auto cpu_probs =
        row_accessor<const Float>(cpu_result.get_probabilities()).pull({ 0, -1 });
    const auto gpu_probs =
        row_accessor<const Float>(gpu_result.get_probabilities()).pull({ 0, -1 });
    const double tol = te::get_tolerance<Float>(1e-4, 1e-10);
    for (std::int64_t i = 0; i < row_count; i++) {
        CAPTURE(i, cpu_probs[i], gpu_probs[i]);
        REQUIRE(std::abs(double(cpu_probs[i]) - double(gpu_probs[i])) < tol);
    }
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::hdbscan::test
