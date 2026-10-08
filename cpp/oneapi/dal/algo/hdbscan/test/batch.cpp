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

#ifdef ONEDAL_DATA_PARALLEL
#include "oneapi/dal/algo/hdbscan/backend/gpu/kernel_impl.hpp"
#endif

#include <algorithm>
#include <map>
#include <set>
#include <type_traits>
#include <vector>

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
// Larger-scale brute_force MST tests
//
// The literal-array cases above are below every threaded threshold in the backend. These are
// sized so the parallel Boruvka scan is the path under test.
// =========================================================================

/// Deterministic well-separated blobs. The fixed LCG only has to be reproducible and free of
/// exact coordinate ties, not statistically sound.
template <typename Float>
static std::vector<Float> make_blobs(std::int64_t per_cluster,
                                     std::int64_t cluster_count,
                                     std::int64_t column_count,
                                     Float separation,
                                     Float spread) {
    std::vector<Float> data(per_cluster * cluster_count * column_count);

    std::uint32_t state = 777u;
    const auto next_unit = [&]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<Float>(state >> 8) / static_cast<Float>(1u << 24);
    };

    std::int64_t pos = 0;
    for (std::int64_t c = 0; c < cluster_count; c++) {
        for (std::int64_t i = 0; i < per_cluster; i++) {
            for (std::int64_t j = 0; j < column_count; j++) {
                const Float center = separation * static_cast<Float>(c + 1) *
                                     static_cast<Float>(j % 2 == 0 ? 1 : -1);
                data[pos++] = center + spread * (next_unit() - Float(0.5));
            }
        }
    }
    return data;
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs tree methods: same partition at thousands of rows",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // 4 x 1000 points: enough rows for a genuinely threaded Boruvka round.
    constexpr std::int64_t per_cluster = 1000;
    constexpr std::int64_t cluster_count = 4;
    constexpr std::int64_t column_count = 3;
    constexpr std::int64_t row_count = per_cluster * cluster_count;

    const auto data = make_blobs<Float>(per_cluster,
                                        cluster_count,
                                        column_count,
                                        /*separation=*/Float(20.0),
                                        /*spread=*/Float(1.0));
    const auto x = homogen_table::wrap(data.data(), row_count, column_count);

    constexpr std::int64_t min_cluster_size = 25;
    const std::int64_t min_samples = GENERATE(5, 50);
    CAPTURE(min_samples);

    const auto bf_desc =
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);
    const auto kd_desc =
        hdbscan::descriptor<Float, hdbscan::method::kd_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);
    const auto bt_desc =
        hdbscan::descriptor<Float, hdbscan::method::ball_tree>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);

    INFO("run brute_force");
    const auto bf_result = dal::compute(bf_desc, x);

    INFO("run kd_tree");
    const auto kd_result = dal::compute(kd_desc, x);

    INFO("run ball_tree");
    const auto bt_result = dal::compute(bt_desc, x);

    INFO("the blobs are separated by 20x their spread, so all methods must recover them");
    REQUIRE(bf_result.get_cluster_count() == cluster_count);

    INFO("compare cluster counts");
    REQUIRE(bf_result.get_cluster_count() == kd_result.get_cluster_count());
    REQUIRE(bf_result.get_cluster_count() == bt_result.get_cluster_count());

    const auto bf_rows = row_accessor<const Float>(bf_result.get_responses()).pull({ 0, -1 });
    const auto kd_rows = row_accessor<const Float>(kd_result.get_responses()).pull({ 0, -1 });
    const auto bt_rows = row_accessor<const Float>(bt_result.get_responses()).pull({ 0, -1 });

    INFO("compare partitions (permutation-invariant)");
    check_same_partition(bf_rows, kd_rows, row_count);
    check_same_partition(bf_rows, bt_rows, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: threaded MST is stable across runs",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // A single diffuse blob: no density gap, so the labels are decided by the MST edge order
    // alone, and the input is dense in exact weight ties for the sort to break reproducibly.
    constexpr std::int64_t row_count = 3000;
    constexpr std::int64_t column_count = 2;

    const auto data = make_blobs<Float>(row_count,
                                        /*cluster_count=*/1,
                                        column_count,
                                        /*separation=*/Float(0.0),
                                        /*spread=*/Float(1.0));
    const auto x = homogen_table::wrap(data.data(), row_count, column_count);

    const auto desc =
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(15, 5).set_result_options(
            result_options::responses);

    const auto first = dal::compute(desc, x);
    const auto first_rows = row_accessor<const Float>(first.get_responses()).pull({ 0, -1 });

    for (int run = 1; run < 3; run++) {
        CAPTURE(run);
        const auto again = dal::compute(desc, x);
        REQUIRE(again.get_cluster_count() == first.get_cluster_count());

        const auto again_rows = row_accessor<const Float>(again.get_responses()).pull({ 0, -1 });
        for (std::int64_t i = 0; i < row_count; i++) {
            CAPTURE(i);
            REQUIRE(again_rows[i] == first_rows[i]);
        }
    }
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
    const std::int64_t row_count = 10;
    const auto x = homogen_table::wrap(data, row_count, 2);

    // Sizes of every reported (non-noise) cluster, keyed by label.
    const auto cluster_sizes = [&](std::int64_t max_cluster_size) {
        const auto desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(5, 5)
                              .set_result_options(result_options::responses)
                              .set_max_cluster_size(max_cluster_size);
        const auto responses = dal::compute(desc, x).get_responses();
        const auto rows = row_accessor<const Float>(responses).pull({ 0, -1 });

        std::map<std::int64_t, std::int64_t> sizes;
        for (std::int64_t i = 0; i < row_count; i++) {
            const auto label = static_cast<std::int64_t>(rows[i]);
            if (label >= 0) {
                sizes[label]++;
            }
        }
        return sizes;
    };

    // Uncapped: two clusters of 5 points. Both are condensed-tree leaves, so only the size cap
    // can unselect them.
    const auto uncapped = cluster_sizes(0);
    REQUIRE(uncapped.size() == 2);
    for (const auto& [label, size] : uncapped) {
        INFO("label = " << label);
        REQUIRE(size == 5);
    }

    // Capped below the blob size: neither blob may be reported as a cluster.
    constexpr std::int64_t max_cluster_size = 3;
    for (const auto& [label, size] : cluster_sizes(max_cluster_size)) {
        INFO("label = " << label);
        REQUIRE(size <= max_cluster_size);
    }
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
// single linkage tree tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: single linkage tree on gold data",
                     "[hdbscan][batch][single_linkage_tree][gold]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());

    this->run_single_linkage_tree_checks(x,
                                         gold_dataset::get_min_cluster_size(),
                                         gold_dataset::get_min_samples());
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: single linkage tree on gold data",
                     "[hdbscan][batch][single_linkage_tree][gold]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());

    this->run_single_linkage_tree_checks(x,
                                         gold_dataset::get_min_cluster_size(),
                                         gold_dataset::get_min_samples());
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: single linkage tree on gold data",
                     "[hdbscan][batch][single_linkage_tree][gold]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());

    this->run_single_linkage_tree_checks(x,
                                         gold_dataset::get_min_cluster_size(),
                                         gold_dataset::get_min_samples());
}

TEMPLATE_LIST_TEST_M(
    hdbscan_batch_test,
    "hdbscan brute_force: single linkage tree matches the reference implementation",
    "[hdbscan][batch][single_linkage_tree]",
    hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Same data as the probabilities reference test above. The expected rows come
    // from `sklearn.cluster.HDBSCAN(min_cluster_size=5,
    // min_samples=5)._single_linkage_tree_` with the same defaults.
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

    constexpr double ref_tree[] = {
        1,  6,  0.3970514954, 2, //
        20, 5,  0.4257470611, 3, //
        21, 8,  0.4257470611, 4, //
        22, 4,  0.4413764153, 5, //
        23, 2,  0.4721348642, 6, //
        24, 7,  0.5439721500, 7, //
        0,  25, 0.6558083028, 8, //
        26, 3,  0.6739196169, 9, //
        12, 9,  1.1948212670, 2, //
        17, 28, 1.2777353443, 3, //
        29, 15, 1.2814711702, 4, //
        30, 14, 1.2814711702, 5, //
        31, 13, 1.3038520852, 6, //
        32, 16, 1.4047075746, 7, //
        33, 10, 1.4098477542, 8, //
        34, 11, 1.5599846217, 9, //
        27, 18, 4.8214381527, 10, //
        36, 35, 7.0542757190, 19, //
        37, 19, 8.6481363588, 20, //
    };

    const auto desc =
        hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(5, 5).set_result_options(
            result_options::responses | result_options::single_linkage_tree);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    this->check_single_linkage_tree(result, row_count);

    const auto tree = row_accessor<const Float>(result.get_single_linkage_tree()).pull({ 0, -1 });
    const auto ref_table = homogen_table::wrap(ref_tree, row_count - 1, 4);
    const auto ref_rows = row_accessor<const double>(ref_table).pull({ 0, -1 });

    check_same_hierarchy(tree, ref_rows, row_count, te::get_tolerance<Float>(1e-4, 1e-7));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: single linkage tree re-cuts like dbscan_clustering",
                     "[hdbscan][batch][single_linkage_tree]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // This is the feature the dendrogram is exported for: a caller cuts it at an
    // arbitrary distance and gets the DBSCAN clustering at that epsilon without
    // recomputing anything. The references come from
    // `sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=5)
    //     .dbscan_clustering(cut_distance=cut)` on the same data.
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

    // One row per cut distance: only the dense blob survives at 1.0, both blobs at
    // 2.0, the far-away point joins the dense blob at 5.0, and everything is one
    // cluster at 10.0.
    constexpr double cut_distances[] = { 1.0, 2.0, 5.0, 10.0 };
    constexpr std::int32_t ref_labels[][row_count] = {
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    };

    const auto desc =
        hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(5, 5).set_result_options(
            result_options::single_linkage_tree);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    this->check_single_linkage_tree(result, row_count);
    const auto tree = row_accessor<const Float>(result.get_single_linkage_tree()).pull({ 0, -1 });

    for (std::int64_t c = 0; c < 4; c++) {
        const double cut = cut_distances[c];
        CAPTURE(c, cut);

        const auto labels = labelling_at_cut(tree, row_count, cut, 5);
        for (std::int64_t i = 0; i < row_count; i++) {
            CAPTURE(i, labels[i], ref_labels[c][i]);
            REQUIRE(labels[i] == ref_labels[c][i]);
        }
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: single linkage tree can be requested alone",
                     "[hdbscan][batch][single_linkage_tree]",
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
            result_options::single_linkage_tree);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE_THROWS_AS(result.get_responses(), domain_error);
    this->check_single_linkage_tree(result, 10);

    // The two tight blobs are ten units apart, so every merge but the last one is
    // within a blob and a cut below ten has to split them.
    const auto tree = row_accessor<const Float>(result.get_single_linkage_tree()).pull({ 0, -1 });
    const auto labels = labelling_at_cut(tree, 10, 1.0, 5);
    REQUIRE(labels[0] != labels[5]);
    for (std::int64_t i = 0; i < 10; i++) {
        CAPTURE(i, labels[i]);
        REQUIRE(labels[i] >= 0);
        REQUIRE(labels[i] == labels[(i < 5) ? 0 : 5]);
    }
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

/// scikit-learn's `test_hdbscan_allow_single_cluster_with_epsilon` dataset:
/// `np.random.RandomState(0).rand(150, 2)`. The condensed tree of this
/// unstructured cloud has a root with 16 child clusters, none of which is
/// stable enough to beat the root, so excess-of-mass selection returns the root
/// alone and the outcome is decided entirely by the single-cluster labeling
/// threshold.
constexpr std::int64_t unstructured_row_count = 150;
constexpr double unstructured_data[] = {
    0.54881350392732475,  0.71518936637241948,  0.60276337607164387,  0.54488318299689686, //
    0.42365479933890471,  0.64589411306665612,  0.43758721126269251,  0.89177300078207977, //
    0.96366276050102928,  0.38344151882577771,  0.79172503808266459,  0.52889491975290448, //
    0.56804456109393231,  0.92559663829266103,  0.071036058197886942, 0.087129299701540708, //
    0.020218397440325719, 0.832619845547938,    0.77815675094985048,  0.87001214824681916, //
    0.978618342232764,    0.7991585642167236,   0.46147936225293185,  0.78052917628645546, //
    0.11827442586893322,  0.63992102132752382,  0.1433532874090464,   0.94466891704958389, //
    0.52184832175007168,  0.41466193999052359,  0.26455561210462697,  0.77423368943421667, //
    0.45615033221654855,  0.56843394886864851,  0.018789800436355142, 0.61763549707587706, //
    0.61209572272242141,  0.61693399687475692,  0.94374807851462417,  0.68182029910348341, //
    0.35950790057378601,  0.43703195379934145,  0.69763119592726486,  0.060225471629269833, //
    0.66676671544566768,  0.67063786961815941,  0.2103825610738409,   0.12892629765485331, //
    0.31542835092418386,  0.36371077094262261,  0.57019677041787964,  0.43860151346232035, //
    0.98837383805922618,  0.10204481074802807,  0.20887675609483469,  0.16130951788499626, //
    0.65310832546539843,  0.25329160253978211,  0.46631077285630629,  0.24442559200160274, //
    0.15896958364551972,  0.11037514116430513,  0.65632958946527342,  0.1381829513486138, //
    0.1965823616800535,   0.36872517066096411,  0.8209932298479351,   0.09710127579306127, //
    0.8379449074988039,   0.096098407893963067, 0.97645946501339576,  0.46865120164770158, //
    0.97676108819033713,  0.60484551974504597,  0.73926357939830167,  0.039187792254320675, //
    0.28280696257640958,  0.12019656121316891,  0.29614019752214493,  0.11872771895424405, //
    0.31798317939397602,  0.41426299451466997,  0.064147496348784361, 0.69247211937001985, //
    0.56660145420657515,  0.26538949093944542,  0.52324805346669967,  0.093940510758441675, //
    0.57594649555617927,  0.92929619757621407,  0.31856895245132366,  0.66741037996368169, //
    0.13179786240439217,  0.71632720411856554,  0.2894060929472011,   0.18319136200711683, //
    0.58651293481008315,  0.020107546187493552, 0.82894002921736309,  0.0046954761925470656, //
    0.67781653679623011,  0.27000797319216485,  0.73519402212259488,  0.96218854511743823, //
    0.24875314351995803,  0.5761573344178369,   0.59204193127183902,  0.57225190579087337, //
    0.22308163264061831,  0.95274901151698499,  0.44712537861762736,  0.84640867247112783, //
    0.69947927531750431,  0.29743695085513366,  0.81379781970247722,  0.39650574084698464, //
    0.88110319711116158,  0.5812728726358587,   0.88173536185485279,  0.69253159007776588, //
    0.72525427981964052,  0.50132438192670226,  0.95608363472322389,  0.64399019922963741, //
    0.42385504855817968,  0.60639321412792435,  0.019193198309333526, 0.30157481667454933, //
    0.66017353749268504,  0.29007760721044407,  0.61801542899884154,  0.42876870094576613, //
    0.13547406422245023,  0.29828232595603077,  0.56996491070126487,  0.59087276124817323, //
    0.57432524884957881,  0.65320081985713363,  0.65210327000168888,  0.43141843543397396, //
    0.896546595851063,    0.36756187004789653,  0.43586492526562681,  0.8919233550156721, //
    0.80619398904608575,  0.70388858354036632,  0.10022688731230112,  0.91948261374467355, //
    0.71424129954911142,  0.99884700656786651,  0.14944830465799375,  0.86812605736821424, //
    0.16249293467637482,  0.61555956428384417,  0.12381998284944151,  0.84800822932223441, //
    0.80731895872501069,  0.56910073861459332,  0.40718329722599966,  0.069166995455138047, //
    0.69742877314456364,  0.45354268267806885,  0.72205559947034792,  0.86638232592862918, //
    0.97552150500288581,  0.85580334239261102,  0.011714084185001972, 0.35997806447836389, //
    0.72999056242405802,  0.17162967726144052,  0.52103660620412928,  0.054337988339253629, //
    0.19999652489640007,  0.018521794460613972, 0.79369770335742063,  0.22392468806038013, //
    0.3453516806969027,   0.92808129346559087,  0.70441440192353277,  0.031838929531307847, //
    0.16469415649791275,  0.62147840149976352,  0.57722858860416759,  0.23789282137450862, //
    0.93421399792479376,  0.61396595596589598,  0.5356328030249583,   0.58990997635457099, //
    0.73012202951676963,  0.31194499547960186,  0.39822106221609188,  0.20984374897512215, //
    0.18619300588033616,  0.94437238998393358,  0.73955079504928756,  0.49045880861756708, //
    0.22741462797332324,  0.25435648177039294,  0.058029160323875617, 0.43441662555812077, //
    0.31179588199410257,  0.69634348881545949,  0.37775183929248091,  0.1796036775596348, //
    0.024678728391331228, 0.067249631463248583, 0.6793927734985673,   0.45369684455604531, //
    0.5365792111087222,   0.8966712930403421,   0.99033894739670436,  0.21689698439847394, //
    0.66307820310010079,  0.26332237673715064,  0.020650999465728681, 0.75837865383614145, //
    0.32001715082246784,  0.38346389417189797,  0.58831711355360572,  0.83104845523619042, //
    0.62898184359114873,  0.87265065544739528,  0.27354203481563577,  0.7980468339125637, //
    0.18563594430595221,  0.95279165697194457,  0.68748827638781529,  0.21550767711355845, //
    0.94737059048892425,  0.73085580677015782,  0.25394164259502583,  0.21331197736748198, //
    0.51820071393066325,  0.025662718054531575, 0.2074700754411094,   0.42468546875150626, //
    0.37416998033422555,  0.4635754243648107,   0.27762870629473191,  0.58678434645816879, //
    0.86385560592323141,  0.11753185596203308,  0.51737910715411417,  0.13206810634515331, //
    0.71685968119259369,  0.39605970280729375,  0.56542131185850897,  0.18327983621407862, //
    0.14484775934337724,  0.48805628064895457,  0.35561273784995562,  0.94043194525281304, //
    0.76532525380696526,  0.7486636198505473,   0.90371973974593345,  0.08342243544201855, //
    0.55219246992240656,  0.58447606895576887,  0.96193637854722902,  0.29214752679254885, //
    0.24082877991544682,  0.10029394226549782,  0.016429629591474204, 0.92952931679219053, //
    0.66991654659091004,  0.78515291202313775,  0.28173010575394908,  0.58641016618632669, //
    0.063955266120981125, 0.4856275959346229,   0.97749513974444679,  0.8765052453165908, //
    0.33815895183684563,  0.96157015454149852,  0.23170162647120451,  0.94931882241568144, //
    0.94137770470649862,  0.79920258735239169,  0.63044793686679113,  0.87428796662494701, //
    0.29302028450779671,  0.84894355531291821,  0.61787669191752381,  0.01323685775889949, //
    0.34723351793221957,  0.14814086094816503,  0.98182938981825318,  0.47837030703998806, //
    0.4973913654986627,   0.6394725163987236,   0.36858460612961752,  0.13690027168559893, //
    0.82211773319424553,  0.18984791190275796,  0.51131898254645602,  0.22431702897473926, //
    0.097844484494034045, 0.86219151742168332,  0.9729194890231303,   0.96083465806300017, //
};

/// Reference labels of `unstructured_data` from
/// `sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=5,
/// cluster_selection_method="eom", allow_single_cluster=True)` with
/// `cluster_selection_epsilon=0.0`: the threshold is the largest lambda at
/// which anything leaves the root, so every sample that falls out earlier is
/// noise.
constexpr std::int32_t unstructured_ref_eps0[] = {
    0, 0,  0,  0, -1, 0,  0,  -1, 0,  -1, 0,  0,  0, 0,  -1, -1, 0,  -1, 0,  0, 0,  0, -1, 0, 0,
    0, -1, 0,  0, 0,  0,  0,  0,  0,  0,  -1, 0,  0, 0,  0,  0,  0,  0,  0,  0, 0,  0, 0,  0, 0,
    0, -1, 0,  0, 0,  0,  0,  -1, 0,  0,  0,  0,  0, -1, 0,  0,  -1, 0,  0,  0, -1, 0, -1, 0, -1,
    0, 0,  0,  0, 0,  0,  0,  0,  -1, 0,  0,  -1, 0, 0,  0,  0,  0,  0,  0,  0, 0,  0, 0,  0, -1,
    0, 0,  -1, 0, 0,  -1, 0,  -1, 0,  0,  0,  -1, 0, 0,  0,  0,  0,  -1, 0,  0, 0,  0, 0,  0, -1,
    0, -1, 0,  0, -1, 0,  -1, 0,  0,  -1, 0,  0,  0, 0,  0,  0,  0,  0,  -1, 0, 0,  0, 0,  0, -1,
};

/// The same dataset with `cluster_selection_epsilon=0.18`, where the threshold
/// becomes `1 / 0.18` and only two samples fall out before it.
constexpr std::int32_t unstructured_ref_eps018[] = {
    0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

/// Materialize `unstructured_data` in the floating-point type under test.
///
/// @tparam Float Floating-point type of the test instantiation
///
/// @return An owning array of `2 * unstructured_row_count` feature values
template <typename Float>
static dal::array<Float> make_unstructured_data() {
    auto arr = dal::array<Float>::empty(unstructured_row_count * 2);
    auto* const dst = arr.get_mutable_data();
    for (std::int64_t i = 0; i < unstructured_row_count * 2; i++) {
        dst[i] = static_cast<Float>(unstructured_data[i]);
    }
    return arr;
}

/// Reproduces scikit-learn's `test_hdbscan_allow_single_cluster_with_epsilon`.
/// The two configurations exercise both single-cluster labeling thresholds on a
/// tree whose root does have cluster children: the root's own largest outgoing
/// lambda when there is no epsilon, and `1 / epsilon` when there is one. The
/// samples that fall out of the root before that lambda must come back as
/// noise, not as members of the root cluster.
///
/// @tparam Method oneDAL neighbors-search method under test
/// @tparam Float  Floating-point type of the test instantiation
///
/// @param[in] policy Compute policy to run on
template <typename Method, typename Float, typename Policy>
static void check_unstructured_single_cluster(Policy&& policy) {
    const auto x = homogen_table::wrap(make_unstructured_data<Float>(),
                                       unstructured_row_count,
                                       std::int64_t(2));

    const double epsilon = GENERATE(0.0, 0.18);
    const std::int32_t* const ref_labels =
        (epsilon == 0.0) ? unstructured_ref_eps0 : unstructured_ref_eps018;
    const std::int64_t ref_noise_count = (epsilon == 0.0) ? 31 : 2;
    CAPTURE(epsilon, ref_noise_count);

    const auto desc = hdbscan::descriptor<Float, Method>(5, 5)
                          .set_allow_single_cluster(true)
                          .set_cluster_selection_epsilon(epsilon)
                          .set_result_options(result_options::responses);
    const auto result = oneapi::dal::test::engine::compute(policy, desc, x);

    REQUIRE(result.get_cluster_count() == 1);

    const auto rows = row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
    std::int64_t noise_count = 0;
    for (std::int64_t i = 0; i < unstructured_row_count; i++) {
        if (static_cast<std::int32_t>(rows[i]) < 0)
            noise_count++;
    }
    REQUIRE(noise_count == ref_noise_count);
    check_exact_labels<Float>(result.get_responses(), ref_labels, unstructured_row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: single-cluster epsilon leaves early fall-outs as noise",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    check_unstructured_single_cluster<std::tuple_element_t<1, TestType>,
                                      std::tuple_element_t<0, TestType>>(this->get_policy());
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan kd_tree: single-cluster epsilon leaves early fall-outs as noise",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_kd_types) {
    SKIP_IF(this->not_float64_friendly());
    check_unstructured_single_cluster<std::tuple_element_t<1, TestType>,
                                      std::tuple_element_t<0, TestType>>(this->get_policy());
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan ball_tree: single-cluster epsilon leaves early fall-outs as noise",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bt_types) {
    SKIP_IF(this->not_float64_friendly());
    check_unstructured_single_cluster<std::tuple_element_t<1, TestType>,
                                      std::tuple_element_t<0, TestType>>(this->get_policy());
}

/// Coincident points, so every condensed-tree edge carries the zero-distance
/// lambda scikit-learn stores as infinity, and the same set with one point a
/// hair away, so one edge carries a finite 1e31 instead. References from
/// `sklearn.cluster.HDBSCAN(min_cluster_size=3, min_samples=2,
/// allow_single_cluster=True)`.
constexpr std::int64_t zero_lambda_row_count = 4;
constexpr double zero_lambda_data[] = {
    0.0, 0.0, //
    0.0, 0.0, //
    0.0, 0.0, //
    0.0, 0.0, //
};
constexpr double split_lambda_data[] = {
    0.0,     0.0, //
    0.0,     0.0, //
    0.0,     0.0, //
    1.0e-31, 0.0, //
};

/// Materialize a row-major double literal in the floating-point type under test.
///
/// @tparam Float Floating-point type of the test instantiation
///
/// @param[in] src   Source values
/// @param[in] count Number of values
///
/// @return An owning array of `count` feature values
template <typename Float>
static dal::array<Float> make_feature_array(const double* src, std::int64_t count) {
    auto arr = dal::array<Float>::empty(count);
    auto* const dst = arr.get_mutable_data();
    for (std::int64_t i = 0; i < count; i++) {
        dst[i] = static_cast<Float>(src[i]);
    }
    return arr;
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: a zero-distance lambda outranks every epsilon",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = homogen_table::wrap(make_feature_array<Float>(zero_lambda_data, 8),
                                       zero_lambda_row_count,
                                       std::int64_t(2));

    constexpr std::int32_t ref_labels[] = { 0, 0, 0, 0 };

    const double epsilon = GENERATE(0.0, 1e-30, 1e-31, 1e-40, 1e-310);
    CAPTURE(epsilon);

    const auto desc =
        hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(3, 2)
            .set_allow_single_cluster(true)
            .set_cluster_selection_epsilon(epsilon)
            .set_result_options(result_options::responses | result_options::probabilities);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 1);
    check_exact_labels<Float>(result.get_responses(), ref_labels, zero_lambda_row_count);

    const auto probs = row_accessor<const Float>(result.get_probabilities()).pull({ 0, -1 });
    const double tol = te::get_tolerance<Float>(1e-4, 1e-7);
    for (std::int64_t i = 0; i < zero_lambda_row_count; i++) {
        CAPTURE(i, probs[i]);
        REQUIRE(std::abs(double(probs[i]) - 1.0) < tol);
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: coincident points outlive a finite-lambda outlier",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    // 1e-31 squares to zero in float32, which makes all four points coincident
    // and removes the finite lambda this case is about.
    SKIP_IF((!std::is_same_v<Float, double>));

    const auto x = homogen_table::wrap(make_feature_array<Float>(split_lambda_data, 8),
                                       zero_lambda_row_count,
                                       std::int64_t(2));

    // The coincident points drop out at the zero-distance lambda, the fourth at
    // 1e31. Without an epsilon the threshold is the root's death lambda, which
    // only the coincident points reach; an epsilon of 1e-31 lowers it to 1e31
    // and admits the outlier, whose membership stays 0 against an infinite
    // maximum.
    constexpr std::int32_t ref_labels_no_eps[] = { 0, 0, 0, -1 };
    constexpr std::int32_t ref_labels_eps[] = { 0, 0, 0, 0 };
    constexpr double ref_probabilities[] = { 1.0, 1.0, 1.0, 0.0 };

    const double epsilon = GENERATE(0.0, 1e-31, 1e-30);
    const std::int32_t* const ref_labels = (epsilon > 0.0) ? ref_labels_eps : ref_labels_no_eps;
    CAPTURE(epsilon);

    const auto desc =
        hdbscan::descriptor<Float, std::tuple_element_t<1, TestType>>(3, 2)
            .set_allow_single_cluster(true)
            .set_cluster_selection_epsilon(epsilon)
            .set_result_options(result_options::responses | result_options::probabilities);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 1);
    check_exact_labels<Float>(result.get_responses(), ref_labels, zero_lambda_row_count);

    const auto probs = row_accessor<const Float>(result.get_probabilities()).pull({ 0, -1 });
    const double tol = te::get_tolerance<Float>(1e-4, 1e-7);
    for (std::int64_t i = 0; i < zero_lambda_row_count; i++) {
        CAPTURE(i, probs[i], ref_probabilities[i]);
        REQUIRE(std::abs(double(probs[i]) - ref_probabilities[i]) < tol);
    }
}

using hdbscan_tree_methods_f64 = COMBINE_TYPES((double),
                                               (hdbscan::method::kd_tree,
                                                hdbscan::method::ball_tree));

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan trees: non-finite input is rejected instead of truncating the MST",
                     "[hdbscan][batch]",
                     hdbscan_tree_methods_f64) {
    // The GPU kernels take the MRD with `sycl::fmax`, which drops a NaN operand, so the graph
    // stays connected there and the CPU is the only place the truncation can occur.
    SKIP_IF(!this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    const Float bad =
        GENERATE(std::numeric_limits<Float>::quiet_NaN(), std::numeric_limits<Float>::infinity());
    CAPTURE(bad);

    // One poisoned row: no MRD comparison against it holds, so no Boruvka round can join it.
    // Finiteness is the caller's to check, as for every algorithm; this only pins that the
    // tree methods fail instead of reading an incomplete MST.
    Float data[] = { 0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 0.15, 0.15,
                     5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2, bad,  5.15 };
    const auto x = homogen_table::wrap(data, 10, 2);

    const auto desc =
        hdbscan::descriptor<Float, Method>(3, 3).set_result_options(result_options::responses);
    REQUIRE_THROWS_AS(oneapi::dal::test::engine::compute(this->get_policy(), desc, x),
                      domain_error);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: a zero row is at cosine distance 1 like in scikit-learn",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Two direction clusters plus two all-zero rows, whose norm used to turn their cosine
    // distances into NaN. The reference is sklearn.cluster.HDBSCAN(3, 3, metric="cosine").
    constexpr std::int64_t row_count = 14;
    constexpr Float data[] = { 1.0,  0.05, 1.0,  0.1, 1.0, 0.0, 1.0, 0.08, 1.0,  0.03,
                               1.0,  0.12, 0.05, 1.0, 0.1, 1.0, 0.0, 1.0,  0.08, 1.0,
                               0.03, 1.0,  0.12, 1.0, 0.0, 0.0, 0.0, 0.0 };
    constexpr Float expected[] = { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1 };
    const auto x = homogen_table::wrap(data, row_count, 2);

    const auto desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(3, 3)
                          .set_metric(distance_metric::cosine)
                          .set_result_options(result_options::responses);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 2);
    const auto rows = row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
    check_same_partition(rows, dal::array<Float>::wrap(expected, row_count), row_count);
}

// Two blobs with fringe points, so membership probabilities vary. The references are
// sklearn.cluster.HDBSCAN(4, 3, store_centers="both") with the metric below.
static constexpr std::int64_t centers_row_count = 20;
static constexpr double centers_data[] = { 0.845,  -0.233, 0.016, 0.204,  -0.394, 0.001,  -0.0,
                                           -0.877, 0.509,  0.3,   -0.313, -0.086, 0.253,  -0.131,
                                           -0.121, -0.727, 0.277, 0.062,  0.137,  -0.763, 7.321,
                                           6.123,  5.69,   7.623, 5.964,  4.839,  5.676,  4.169,
                                           6.84,   5.667,  5.406, 6.858,  4.679,  6.428,  4.348,
                                           5.47,   5.037,  7.17,  7.413,  5.736 };

using hdbscan_center_types = COMBINE_TYPES((float, double),
                                           (hdbscan::method::brute_force,
                                            hdbscan::method::kd_tree,
                                            hdbscan::method::ball_tree));

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: centers are probability-weighted like scikit-learn's",
                     "[hdbscan][batch]",
                     hdbscan_center_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    const bool manhattan = GENERATE(false, true);
    CAPTURE(manhattan);
    // A plain mean gives (0.1209, -0.225) for the first cluster, and the point nearest to it
    // in squared L2 is not sklearn's medoid either.
    const double expected_centroids[2][4] = {
        { 0.1048961814, -0.1708402313, 5.8384860213, 6.0206762016 },
        { 0.1074746881, -0.1861634668, 5.8397797591, 6.0354195229 },
    };
    const double expected_medoids[] = { 0.253, -0.131, 5.406, 6.858 };

    std::vector<Float> data(centers_data, centers_data + 2 * centers_row_count);
    const auto x = homogen_table::wrap(data.data(), centers_row_count, 2);
    const auto desc =
        hdbscan::descriptor<Float, Method>(4, 3)
            .set_metric(manhattan ? distance_metric::manhattan : distance_metric::euclidean)
            .set_store_centers(store_centers_method::both)
            .set_result_options(result_options::responses | result_options::cluster_centers |
                                result_options::medoid_centers);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);
    REQUIRE(result.get_cluster_count() == 2);

    // Cluster ids follow first appearance, and the first rows belong to the blob at the origin.
    const auto centroids = row_accessor<const Float>(result.get_cluster_centers()).pull({ 0, -1 });
    const auto medoids = row_accessor<const Float>(result.get_medoid_centers()).pull({ 0, -1 });
    const double tol = te::get_tolerance<Float>(1e-4, 1e-8);
    for (std::int64_t i = 0; i < 4; ++i) {
        CAPTURE(i);
        REQUIRE(std::abs(double(centroids[i]) - expected_centroids[manhattan][i]) < tol);
        REQUIRE(std::abs(double(medoids[i]) - expected_medoids[i]) < tol);
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: medoids use the fitted metric, like scikit-learn's",
                     "[hdbscan][batch]",
                     hdbscan_center_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    // sklearn.cluster.HDBSCAN(4, 3, store_centers="medoid", metric=...), minkowski with p=3.
    // Chebyshev picks a different medoid for the second blob than the L_p metrics.
    struct metric_case {
        distance_metric metric;
        double expected[4];
    };
    const metric_case c =
        GENERATE(metric_case{ distance_metric::chebyshev, { 0.253, -0.131, 6.84, 5.667 } },
                 metric_case{ distance_metric::minkowski, { 0.253, -0.131, 5.406, 6.858 } },
                 metric_case{ distance_metric::cosine, { 5.964, 4.839, 5.69, 7.623 } });
    CAPTURE(int(c.metric));
    // The trees need an L_p metric.
    SKIP_IF((c.metric == distance_metric::cosine &&
             !std::is_same_v<Method, hdbscan::method::brute_force>));

    std::vector<Float> data(centers_data, centers_data + 2 * centers_row_count);
    const auto x = homogen_table::wrap(data.data(), centers_row_count, 2);
    const auto desc =
        hdbscan::descriptor<Float, Method>(4, 3)
            .set_metric(c.metric)
            .set_degree(3.0)
            .set_store_centers(store_centers_method::medoid)
            .set_result_options(result_options::responses | result_options::medoid_centers);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);
    REQUIRE(result.get_cluster_count() == 2);

    const auto medoids = row_accessor<const Float>(result.get_medoid_centers()).pull({ 0, -1 });
    const double tol = te::get_tolerance<Float>(1e-4, 1e-8);
    for (std::int64_t i = 0; i < 4; ++i) {
        CAPTURE(i);
        REQUIRE(std::abs(double(medoids[i]) - c.expected[i]) < tol);
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: medoids of clusters larger than one distance tile",
                     "[hdbscan][batch]",
                     hdbscan_center_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    const distance_metric metric = GENERATE(distance_metric::euclidean,
                                            distance_metric::manhattan,
                                            distance_metric::minkowski,
                                            distance_metric::chebyshev);
    CAPTURE(int(metric));

    // Two 21 x 21 integer lattices, 441 rows each, so every cluster spans several 128 x 256
    // distance tiles. By symmetry the lattice center is the medoid, as scikit-learn agrees.
    constexpr std::int64_t side = 21;
    constexpr std::int64_t row_count = 2 * side * side;
    std::vector<Float> data;
    data.reserve(row_count * 2);
    for (const Float shift : { Float(0), Float(100) }) {
        for (std::int64_t i = 0; i < side; ++i) {
            for (std::int64_t j = 0; j < side; ++j) {
                data.push_back(Float(i - side / 2) + shift);
                data.push_back(Float(j - side / 2));
            }
        }
    }
    const auto x = homogen_table::wrap(data.data(), row_count, 2);
    const auto desc =
        hdbscan::descriptor<Float, Method>(50, 5)
            .set_metric(metric)
            .set_degree(3.0)
            .set_store_centers(store_centers_method::medoid)
            .set_result_options(result_options::responses | result_options::medoid_centers);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);
    REQUIRE(result.get_cluster_count() == 2);

    // The two lattices tie at every merge, so which one gets id 0 is not pinned down.
    const auto labels = row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
    const std::int64_t origin = std::int64_t(labels[0]);
    REQUIRE(labels[row_count - 1] == Float(1 - origin));
    const auto medoids = row_accessor<const Float>(result.get_medoid_centers()).pull({ 0, -1 });
    const Float expected[2][2] = { { 0, 0 }, { 100, 0 } };
    for (std::int64_t k = 0; k < 2; ++k) {
        const std::int64_t blob = (k == origin) ? 0 : 1;
        CAPTURE(k, blob);
        REQUIRE(medoids[2 * k] == expected[blob][0]);
        REQUIRE(medoids[2 * k + 1] == expected[blob][1]);
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: alpha divides the core distances too, like scikit-learn",
                     "[hdbscan][batch][single_linkage_tree]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // scikit-learn's brute path divides the whole distance matrix by alpha before taking the core
    // distances. Reference: sklearn.cluster.HDBSCAN(4, 3, algorithm="brute", alpha=2.0).
    const double expected[] = { 0.1302497601, 0.1302497601, 0.20517919,   0.20517919,
                                0.2192834923, 0.2287515027, 0.2506476611, 0.3150337284,
                                0.3212786952, 0.33139742,   0.33139742,   0.4080076592,
                                0.4080076592, 0.4223236318, 0.6026939522, 0.8674123875,
                                0.8674123875, 0.9295408813, 3.2197368293 };

    std::vector<Float> data(centers_data, centers_data + 2 * centers_row_count);
    const auto x = homogen_table::wrap(data.data(), centers_row_count, 2);
    const auto desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(4, 3)
                          .set_alpha(2.0)
                          .set_result_options(result_options::single_linkage_tree);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);
    const auto tree = row_accessor<const Float>(result.get_single_linkage_tree()).pull({ 0, -1 });

    std::vector<double> merges;
    for (std::int64_t e = 0; e < centers_row_count - 1; ++e)
        merges.push_back(double(tree[4 * e + 2]));
    std::sort(merges.begin(), merges.end());
    for (std::int64_t e = 0; e < centers_row_count - 1; ++e) {
        CAPTURE(e);
        REQUIRE(std::abs(merges[e] - expected[e]) < te::get_tolerance<Float>(1e-5, 1e-9));
    }
}

TEMPLATE_LIST_TEST_M(
    hdbscan_batch_test,
    "hdbscan: max_cluster_size caps the root by its child clusters like scikit-learn",
    "[hdbscan][batch]",
    hdbscan_center_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    // With allow_single_cluster, scikit-learn sizes the root as the points its child clusters
    // hold, not every point, so a cap of 7 on 17 points still lets the root win here. Reference:
    // sklearn.cluster.HDBSCAN(3, 3, allow_single_cluster=True, max_cluster_size=7).
    constexpr std::int64_t row_count = 17;
    constexpr double source[] = { 0.118,  0.114,  0.37,   1.041,  -1.517, -0.866, -0.055,
                                  -0.107, 1.365,  -0.098, -2.426, -0.453, -0.471, 0.973,
                                  -1.278, 1.437,  -0.078, 1.09,   0.097,  1.419,  1.168,
                                  0.947,  1.085,  2.382,  -0.406, 0.266,  -0.422, -5.019,
                                  3.791,  -4.535, -5.837, -2.814, -5.605, -4.443 };
    constexpr Float expected[] = { 0, 0, -1, 0, -1, -1, 0, -1, 0, 0, -1, -1, 0, -1, -1, -1, -1 };

    std::vector<Float> data(source, source + 2 * row_count);
    const auto x = homogen_table::wrap(data.data(), row_count, 2);
    const auto desc = hdbscan::descriptor<Float, Method>(3, 3)
                          .set_allow_single_cluster(true)
                          .set_max_cluster_size(7)
                          .set_result_options(result_options::responses);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);

    REQUIRE(result.get_cluster_count() == 1);
    const auto rows = row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
    check_same_partition(rows, dal::array<Float>::wrap(expected, row_count), row_count);
}

using hdbscan_tree_types = COMBINE_TYPES((float, double),
                                         (hdbscan::method::kd_tree, hdbscan::method::ball_tree));

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan trees: small leaves keep the exact core distances",
                     "[hdbscan][batch][single_linkage_tree]",
                     hdbscan_tree_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    // The k-NN heap used to report its partial maximum as the pruning radius before it held
    // `min_samples` points, so a query could skip the subtrees holding its true neighbours and
    // return a core distance that was too small. Leaves smaller than `min_samples` exposed it.
    constexpr std::int64_t row_count = 20;
    const bool on_line = GENERATE(true, false);
    CAPTURE(on_line);
    std::vector<Float> data(row_count * 2);
    for (std::int64_t i = 0; i < row_count; ++i) {
        // Growing gaps, so no two merge distances tie.
        data[2 * i] = Float(i * i);
        data[2 * i + 1] = on_line ? Float(0) : Float(0.37) * Float(i % 3);
    }
    const auto x = homogen_table::wrap(data.data(), row_count, 2);

    const auto options = result_options::responses | result_options::single_linkage_tree;
    const auto ref_desc =
        hdbscan::descriptor<Float, hdbscan::method::brute_force>(3, 5).set_result_options(options);
    const auto ref = oneapi::dal::test::engine::compute(this->get_policy(), ref_desc, x);
    const auto ref_tree = row_accessor<const Float>(ref.get_single_linkage_tree()).pull({ 0, -1 });

    const std::int64_t leaf_size = GENERATE(1, 2, 3);
    CAPTURE(leaf_size);
    const auto desc =
        hdbscan::descriptor<Float, Method>(3, 5).set_leaf_size(leaf_size).set_result_options(
            options);
    const auto result = oneapi::dal::test::engine::compute(this->get_policy(), desc, x);
    const auto tree = row_accessor<const Float>(result.get_single_linkage_tree()).pull({ 0, -1 });

    check_same_hierarchy(tree, ref_tree, row_count, te::get_tolerance<Float>(1e-4, 1e-9));
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

// The two sizing helpers below guard row counts whose n x n matrix is several
// gigabytes, so they cannot be reached through `dal::compute` in a test. Drive
// them directly instead.

using hdbscan_gpu_otf_types = COMBINE_TYPES((float, double),
                                            (hdbscan::method::kd_tree, hdbscan::method::ball_tree));

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan gpu: the blocked core distances do not depend on the block size",
                     "[hdbscan][batch][gpu]",
                     hdbscan_gpu_otf_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    // Three blobs on a fixed grid. 64 rows with blocks of 7 leave a 1-row tail block.
    constexpr std::int64_t row_count = 64;
    std::vector<Float> data(row_count * 2);
    for (std::int64_t i = 0; i < row_count; ++i) {
        const Float center = Float(10 * (i % 3));
        data[2 * i] = center + Float(0.01) * Float(i % 11);
        data[2 * i + 1] = center + Float(0.013) * Float(i % 7);
    }
    const auto x = homogen_table::wrap(data.data(), row_count, 2);

    const auto compute_with_block = [&](std::int64_t block_size) {
        const auto desc = hdbscan::descriptor<Float, Method>(5, 4)
                              .set_distance_block_size(block_size)
                              .set_result_options(result_options::responses);
        return oneapi::dal::test::engine::compute(this->get_policy(), desc, x);
    };

    const auto reference = compute_with_block(row_count);
    REQUIRE(reference.get_cluster_count() == 3);
    const auto reference_rows =
        row_accessor<const Float>(reference.get_responses()).pull({ 0, -1 });

    const std::int64_t block_size = GENERATE(1, 7, 63);
    CAPTURE(block_size);
    const auto result = compute_with_block(block_size);
    REQUIRE(result.get_cluster_count() == reference.get_cluster_count());
    const auto rows = row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
    check_same_partition(reference_rows, rows, row_count);
}

TEST("hdbscan gpu: the square launches stay inside the int32 range limit",
     "[hdbscan][batch][gpu]") {
    constexpr std::int64_t int32_max = 2147483647;

    // Below the limit a single launch still covers the whole matrix.
    REQUIRE(dal::backend::max_range_2d_rows(1) >= 1);
    REQUIRE(dal::backend::max_range_2d_rows(46340) * 46340 <= int32_max);
    REQUIRE(dal::backend::max_range_2d_rows(46340) >= 46340);

    // 46341^2 is the first square past int32, so blocking must kick in there.
    REQUIRE(dal::backend::max_range_2d_rows(46341) < 46341);

    for (const std::int64_t n : { std::int64_t(46341),
                                  std::int64_t(50000),
                                  std::int64_t(100000),
                                  std::int64_t(1000000),
                                  int32_max }) {
        const std::int64_t rows = dal::backend::max_range_2d_rows(n);
        REQUIRE(rows >= 1);
        REQUIRE(rows * n <= int32_max);
    }
}

TEST("hdbscan gpu: brute_force rejects a matrix larger than device memory",
     "[hdbscan][batch][gpu]") {
    constexpr std::int64_t gib = 1024 * 1024 * 1024;
    constexpr std::int64_t f64 = 8;
    // The simd k-selection path, which keeps no device scratch of its own.
    constexpr std::int64_t k = 5;
    constexpr std::int64_t no_scratch = 0;

    // 16 GiB global, 4 GiB per allocation: 10k rows need 800 MB and fit.
    REQUIRE(backend::mrd_matrix_fits_on_device(10000, 8, f64, 4 * gib, 16 * gib, k, no_scratch));

    // 30k rows need 7.2 GB, past the single-allocation limit.
    REQUIRE(!backend::mrd_matrix_fits_on_device(30000, 8, f64, 4 * gib, 16 * gib, k, no_scratch));

    // A 32k x 32k matrix is 7.63 GiB and fits both one allocation and the 9 GiB
    // budget, but the input table pushes the total over once it gets wide.
    REQUIRE(backend::mrd_matrix_fits_on_device(32000, 8, f64, 16 * gib, 18 * gib, k, no_scratch));
    REQUIRE(
        !backend::mrd_matrix_fits_on_device(32000, 8000, f64, 16 * gib, 18 * gib, k, no_scratch));

    // The reported 100k-row `std::bad_alloc`: 80 GB in float64 on a 48 GB card.
    REQUIRE(!backend::mrd_matrix_fits_on_device(100000, 8, f64, 48 * gib, 48 * gib, k, no_scratch));

    // Only part of global memory is usable: a 32.5 GiB matrix allocates on a
    // 48 GiB card and then faults on first access, so it has to be rejected
    // even though it is under the 45 GiB single-allocation limit.
    REQUIRE(!backend::mrd_matrix_fits_on_device(66000, 8, f64, 45 * gib, 48 * gib, k, no_scratch));
    REQUIRE(backend::mrd_matrix_fits_on_device(50000, 8, f64, 45 * gib, 48 * gib, k, no_scratch));

    // Halving the element size halves the footprint, so float32 is checked
    // against its own size rather than the widest one.
    REQUIRE(backend::mrd_matrix_fits_on_device(20000, 8, 4, 4 * gib, 16 * gib, k, no_scratch));
    REQUIRE(!backend::mrd_matrix_fits_on_device(20000, 8, f64, 2 * gib, 16 * gib, k, no_scratch));

    // A footprint that leaves int64 saturates instead of wrapping negative:
    // `8 * 2^30 * 2^30` is exactly 2^63, which used to compare as "fits".
    constexpr std::int64_t pow30 = std::int64_t(1) << 30;
    REQUIRE(!backend::mrd_matrix_fits_on_device(pow30, 8, f64, 45 * gib, 48 * gib, k, no_scratch));
    REQUIRE(!backend::mrd_matrix_fits_on_device(std::int64_t(1) << 40,
                                                8,
                                                f64,
                                                45 * gib,
                                                48 * gib,
                                                k,
                                                no_scratch));
    // The column count, `min_samples` and the reported scratch each overflow on
    // their own.
    REQUIRE(!backend::mrd_matrix_fits_on_device(1000,
                                                std::int64_t(1) << 50,
                                                f64,
                                                45 * gib,
                                                48 * gib,
                                                k,
                                                no_scratch));
    REQUIRE(
        !backend::mrd_matrix_fits_on_device(50000, 8, f64, 45 * gib, 48 * gib, pow30, no_scratch));
    REQUIRE(!backend::mrd_matrix_fits_on_device(50000,
                                                8,
                                                f64,
                                                45 * gib,
                                                48 * gib,
                                                k,
                                                backend::mrd_bytes_saturated));

    // 55k float64 rows fit the matrix and the budget on a 48 GiB card, but not
    // once quick select's two n x n scratch matrices are counted.
    constexpr std::int64_t quick_rows = 55000;
    constexpr std::int64_t quick_scratch = quick_rows * quick_rows * (f64 + 4) + 1024 * f64;
    REQUIRE(
        backend::mrd_matrix_fits_on_device(quick_rows, 8, f64, 45 * gib, 48 * gib, k, no_scratch));
    REQUIRE(!backend::mrd_matrix_fits_on_device(quick_rows,
                                                8,
                                                f64,
                                                45 * gib,
                                                48 * gib,
                                                1023,
                                                quick_scratch));

    // The `n x min_samples` selection output counts on every path, quick or not.
    REQUIRE(
        !backend::mrd_matrix_fits_on_device(50000, 8, f64, 45 * gib, 48 * gib, 20000, no_scratch));
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

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: cpu and gpu single linkage trees match",
                     "[hdbscan][batch][single_linkage_tree]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    const std::int64_t row_count = gold_dataset::get_row_count();

    const auto desc = hdbscan::descriptor<Float, hdbscan::method::brute_force>(
                          gold_dataset::get_min_cluster_size(),
                          gold_dataset::get_min_samples())
                          .set_result_options(result_options::single_linkage_tree);

    INFO("run on CPU (no queue)");
    const auto cpu_result = dal::compute(desc, x);

    INFO("run on GPU (with queue)");
    const auto gpu_result = dal::compute(this->get_policy().get_queue(), desc, x);

    this->check_single_linkage_tree(cpu_result, row_count);
    this->check_single_linkage_tree(gpu_result, row_count);

    // The dendrogram is built from the same sorted MST on both backends, so the
    // hierarchy has to agree. The gold data has two merges at an equal distance,
    // which the two backends take in the opposite order, so the rows themselves
    // do not line up -- see `check_same_hierarchy`.
    const auto cpu_tree =
        row_accessor<const Float>(cpu_result.get_single_linkage_tree()).pull({ 0, -1 });
    const auto gpu_tree =
        row_accessor<const Float>(gpu_result.get_single_linkage_tree()).pull({ 0, -1 });

    check_same_hierarchy(gpu_tree, cpu_tree, row_count, te::get_tolerance<Float>(1e-4, 1e-10));
}

TEST("hdbscan gpu: compute_core_distances returns a completed event", "[hdbscan][batch][gpu]") {
    namespace pr = dal::backend::primitives;
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t n = 64;
    constexpr std::int64_t min_samples = 5;

    auto [dist, dist_event] =
        pr::ndarray<float, 2>::full(q, { n, n }, 1.0f, sycl::usm::alloc::device);
    auto [core, core_event] = pr::ndarray<float, 1>::zeros(q, n, sycl::usm::alloc::device);

    pr::ndview<float, 1> core_view = core;
    const auto event = backend::compute_core_distances<float>(q,
                                                              dist,
                                                              core_view,
                                                              min_samples,
                                                              n,
                                                              distance_metric::manhattan,
                                                              { dist_event, core_event });
    // The k-selection scratch the kernel reads is freed on return, so the event must be done.
    const auto status = event.get_info<sycl::info::event::command_execution_status>();
    REQUIRE(status == sycl::info::event_command_status::complete);

    const auto core_host = core.to_host(q);
    for (std::int64_t i = 0; i < n; ++i) {
        REQUIRE(core_host.get_data()[i] == 1.0f);
    }
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::hdbscan::test
