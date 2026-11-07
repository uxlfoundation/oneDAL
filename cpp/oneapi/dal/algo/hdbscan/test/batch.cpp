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
#include "oneapi/dal/test/engine/blobs.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace oneapi::dal::hdbscan::test {

template <typename TestType>
class hdbscan_batch_test : public hdbscan_test<TestType, hdbscan_batch_test<TestType>> {};

using hdbscan_types = COMBINE_TYPES((float, double),
                                    (hdbscan::method::brute_force,
                                     hdbscan::method::kd_tree,
                                     hdbscan::method::ball_tree));
using hdbscan_bf_types = COMBINE_TYPES((float, double), (hdbscan::method::brute_force));
using hdbscan_bf_only = COMBINE_TYPES((double), (hdbscan::method::brute_force));
using hdbscan_bf_kd_types = COMBINE_TYPES((float, double),
                                          (hdbscan::method::brute_force, hdbscan::method::kd_tree));
using hdbscan_tree_types = COMBINE_TYPES((float, double),
                                         (hdbscan::method::kd_tree, hdbscan::method::ball_tree));
using hdbscan_tree_methods_f64 = COMBINE_TYPES((double),
                                               (hdbscan::method::kd_tree,
                                                hdbscan::method::ball_tree));

// =========================================================================
// Tests shared by all methods
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: two well-separated clusters",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    this->run_checks(x, 5, 5, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: clusters numbered as scikit-learn numbers them",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // scikit-learn gives label 0 to the cluster of point 0, which here is the cluster near 10.
    constexpr std::int64_t row_count = 12;
    constexpr double data[] = { 10.044, 10.957, 0.253,  0.435,  0.779,  0.198,  10.863, 10.983,
                                10.786, 10.866, 10.164, 10.597, 10.009, 10.387, 0.698,  0.216,
                                0.976,  0.006,  0.967,  0.547,  0.973,  0.715,  10.436, 10.949 };
    constexpr std::int32_t labels[] = { 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0 };

    const auto x = make_table<Float>(data, row_count, 2);
    const auto result = this->compute(this->get_descriptor(3, 3), x);
    this->check_responses_against_ref(result.get_responses(), labels, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: clusters at different depths numbered breadth-first",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Five 7-point blobs, no two pairwise distances within 1e-5 of each other. The root splits into
    // G1 and G2, G1 into A (which holds point 0) and H, G2 into C and D, and H into H1 and H2.
    // Breadth-first numbering gives A, C, D, H1, H2 = 0 .. 4; either depth-first order differs.
    constexpr std::int64_t row_count = 35;
    constexpr double data[] = { 0.41,  -0.35, 4.88,  0.39,  19.89, -0.05, 20.07, 3.69,  4.98,
                                2.36,  20.42, -0.2,  20.11, -0.1,  0.73,  0.4,   4.16,  2.62,
                                19.93, 3.9,   5.09,  0.12,  20.09, 4.02,  5.44,  0.02,  20.0,
                                0.08,  20.28, 3.87,  5.41,  3.02,  19.8,  0.4,   5.13,  2.56,
                                6.01,  0.16,  0.26,  0.17,  19.72, 3.57,  19.75, -0.14, 19.99,
                                0.07,  -0.21, -0.23, 0.29,  -0.09, 5.4,   2.37,  0.2,   0.54,
                                -0.04, -0.25, 4.9,   0.08,  20.23, 3.9,   4.97,  2.32,  4.56,
                                2.94,  19.97, 4.03,  5.28,  0.37,  4.72,  0.43 };
    // Reference: sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=4).labels_.
    constexpr std::int32_t labels[] = { 0, 3, 1, 2, 4, 1, 1, 0, 4, 2, 3, 2, 3, 1, 2, 4, 1, 4,
                                        3, 0, 2, 1, 1, 0, 0, 4, 0, 0, 3, 2, 4, 4, 2, 3, 3 };

    const auto x = make_table<Float>(data, row_count, 2);
    const auto result = this->compute(this->get_descriptor(5, 4), x);
    REQUIRE(result.get_cluster_count() == 5);
    this->check_responses_against_ref(result.get_responses(), labels, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: gold data test",
                     "[hdbscan][batch][gold]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    this->run_checks(x, gold_dataset::get_min_cluster_size(), gold_dataset::get_min_samples(), -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: all noise when min_cluster_size > n",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr double data[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    const auto x = make_table<Float>(data, 5, 1);
    this->run_checks(x, 10, 2, 0);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: min_samples > n is clamped to n",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    this->check_descriptors_agree(this->get_descriptor(5, two_blob_row_count + 5),
                                  this->get_descriptor(5, two_blob_row_count),
                                  x);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: small min_cluster_size",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_small_groups_data, two_small_groups_row_count, 2);
    this->run_checks(x, 2, 2, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: 1D data three clusters",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(three_blob_1d_data, three_blob_1d_row_count, 1);
    this->run_checks(x, 5, 5, 3);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: high-dimensional data",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Two 5-point clusters in 10 dimensions: one near the origin, one shifted by 10.
    constexpr std::int64_t n = 10;
    constexpr std::int64_t d = 10;
    Float data[n * d];
    for (std::int64_t i = 0; i < n; i++) {
        for (std::int64_t j = 0; j < d; j++) {
            data[i * d + j] = static_cast<Float>((i < 5 ? 0.0 : 10.0) + 0.01 * (i + j));
        }
    }
    const auto x = homogen_table::wrap(data, n, d);

    this->run_checks(x, 5, 5, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: two clusters under the manhattan, chebyshev and minkowski metrics",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    struct metric_case {
        distance_metric metric;
        double degree;
    };
    const metric_case c = GENERATE(metric_case{ distance_metric::manhattan, 2.0 },
                                   metric_case{ distance_metric::chebyshev, 2.0 },
                                   metric_case{ distance_metric::minkowski, 3.0 });
    CAPTURE(int(c.metric), c.degree);

    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    this->run_checks(x, 5, 5, c.metric, c.degree, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: epsilon merges close clusters",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Three clusters: A and B are about 1.0 apart, C is about 10.0 away.
    constexpr double data[] = {
        0.0,  0.0,  0.05,  0.05,  0.1,  0.0,  0.0,  0.1,  0.08,  0.08, //
        1.0,  1.0,  1.05,  1.05,  1.1,  1.0,  1.0,  1.1,  1.08,  1.08, //
        10.0, 10.0, 10.05, 10.05, 10.1, 10.0, 10.0, 10.1, 10.08, 10.08,
    };
    constexpr std::int64_t row_count = 15;
    const auto x = make_table<Float>(data, row_count, 2);

    // Reference: sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=5,
    // cluster_selection_epsilon=epsilon).labels_.
    constexpr std::int32_t ref_labels_eps0[] = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0 };
    constexpr std::int32_t ref_labels_eps5[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };

    const auto result_no_eps =
        this->compute(this->get_descriptor(5, 5).set_cluster_selection_epsilon(0.0), x);
    REQUIRE(result_no_eps.get_cluster_count() == 3);
    this->check_responses_against_ref(result_no_eps.get_responses(), ref_labels_eps0, row_count);

    const auto result_eps =
        this->compute(this->get_descriptor(5, 5).set_cluster_selection_epsilon(5.0), x);
    REQUIRE(result_eps.get_cluster_count() == 2);
    this->check_responses_against_ref(result_eps.get_responses(), ref_labels_eps5, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: epsilon search stops only above epsilon",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Five 5-point groups on a line. Leaves L1, L2 (gap 2) form P, leaves Y1, Y2 (gap 2.5) form
    // Y, P and Y (gap 4) form G, and G and X (gap 15.5) form the root. Multiples of 1/8, so every
    // merge distance is exact in float32 and lands exactly on the epsilons 2 and 4.
    constexpr std::int64_t row_count = 25;
    constexpr double data[] = {
        0.0,    0.125,  0.375,  0.5,    0.875, //
        2.875,  3.0,    3.25,   3.375,  3.75, //
        7.75,   7.875,  8.125,  8.25,   8.625, //
        11.125, 11.25,  11.5,   11.625, 12.0, //
        27.5,   27.625, 27.875, 28.0,   28.375,
    };
    const auto x = make_table<Float>(data, row_count, 1);

    struct epsilon_case {
        double epsilon;
        std::int64_t cluster_count;
        std::int32_t labels[row_count];
    };
    // Reference: sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=5,
    // cluster_selection_epsilon=epsilon).labels_. At 2.0 and 2.5 a leaf born exactly at epsilon
    // is kept; at 4.0 the search passes P and Y, born exactly at epsilon, and stops at G.
    const epsilon_case c =
        GENERATE(epsilon_case{ 0.0, 5, { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3,
                                         3, 3, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0 } },
                 epsilon_case{ 2.0, 5, { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3,
                                         3, 3, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0 } },
                 epsilon_case{ 2.5, 4, { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                                         2, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0 } },
                 epsilon_case{ 3.0, 3, { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                                         2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0 } },
                 epsilon_case{ 4.0, 2, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 } },
                 epsilon_case{ 16.0, 2, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 } });
    CAPTURE(c.epsilon);

    const auto result =
        this->compute(this->get_descriptor(5, 5).set_cluster_selection_epsilon(c.epsilon), x);
    REQUIRE(result.get_cluster_count() == c.cluster_count);
    this->check_responses_against_ref(result.get_responses(), c.labels, row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: alpha > 1 runs without error",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    REQUIRE_NOTHROW(this->compute(this->get_descriptor(5, 5).set_alpha(1.5), x));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: probabilities on two well-separated clusters",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    this->run_probability_checks(x, 5, 5);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: single linkage tree on gold data",
                     "[hdbscan][batch][single_linkage_tree][gold]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    this->run_single_linkage_tree_checks(x,
                                         gold_dataset::get_min_cluster_size(),
                                         gold_dataset::get_min_samples());
}

// =========================================================================
// Tree methods
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan trees: different leaf_size gives same partition",
                     "[hdbscan][batch]",
                     hdbscan_tree_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    this->check_descriptors_agree(this->get_descriptor(5, 5).set_leaf_size(10),
                                  this->get_descriptor(5, 5).set_leaf_size(40),
                                  x);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan trees: small leaves keep the exact core distances",
                     "[hdbscan][batch][single_linkage_tree]",
                     hdbscan_tree_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    // Leaves smaller than `min_samples` must still give the exact core distances, so the
    // hierarchy matches brute force.
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
    const auto ref_tree = this->get_single_linkage_tree(this->compute(ref_desc, x));

    const std::int64_t leaf_size = GENERATE(1, 2, 3);
    CAPTURE(leaf_size);
    const auto desc =
        hdbscan::descriptor<Float, Method>(3, 5).set_leaf_size(leaf_size).set_result_options(
            options);
    const auto tree = this->get_single_linkage_tree(this->compute(desc, x));

    check_same_hierarchy(tree, ref_tree, row_count, te::get_tolerance<Float>(1e-4, 1e-9));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: non-finite input is rejected instead of truncating the MST",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // The GPU takes the MRD with `sycl::fmax`, which drops a NaN operand and keeps the graph
    // connected, so it only rejects infinity.
    const Float bad =
        GENERATE(std::numeric_limits<Float>::quiet_NaN(), std::numeric_limits<Float>::infinity());
    CAPTURE(bad);
    SKIP_IF(!this->get_policy().is_cpu() && std::isnan(bad));

    // No Boruvka round can join the poisoned row, so the MST is incomplete and the call must fail.
    // Checking finiteness is the caller's job, as for every algorithm.
    Float data[] = { 0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 0.15, 0.15,
                     5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2, bad,  5.15 };
    const auto x = homogen_table::wrap(data, 10, 2);

    const std::string expected = dal::detail::error_messages::hdbscan_input_data_is_not_finite();
    std::string message;
    try {
        this->compute(this->get_descriptor(3, 3), x);
    }
    catch (const domain_error& e) {
        message = e.what();
    }
    REQUIRE(message == expected);
}

// =========================================================================
// Cross-method consistency
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: kd_tree and ball_tree match brute_force on gold data",
                     "[hdbscan][batch][gold]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    this->template check_methods_agree<hdbscan::method::kd_tree, hdbscan::method::ball_tree>(
        x,
        gold_dataset::get_min_cluster_size(),
        gold_dataset::get_min_samples());
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: kd_tree and ball_tree match brute_force on two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    struct metric_case {
        distance_metric metric;
        double degree;
    };
    const metric_case c = GENERATE(metric_case{ distance_metric::euclidean, 2.0 },
                                   metric_case{ distance_metric::manhattan, 2.0 },
                                   metric_case{ distance_metric::chebyshev, 2.0 },
                                   metric_case{ distance_metric::minkowski, 3.0 });
    CAPTURE(int(c.metric), c.degree);

    const auto x = make_table<Float>(two_blob_noise_data, two_blob_noise_row_count, 2);
    this->template check_methods_agree<hdbscan::method::kd_tree, hdbscan::method::ball_tree>(
        x,
        5,
        5,
        c.metric,
        c.degree);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: kd_tree matches brute_force on 1D data",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(three_blob_1d_data, three_blob_1d_row_count, 1);
    this->template check_methods_agree<hdbscan::method::kd_tree>(x, 5, 5);
}

// The literal-array cases are below every threaded threshold of the backend; these are sized so
// the parallel Boruvka scan is the path under test.

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs tree methods: same partition at thousands of rows",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr std::int64_t per_cluster = 1000;
    constexpr std::int64_t cluster_count = 4;
    constexpr std::int64_t column_count = 3;
    constexpr std::int64_t row_count = per_cluster * cluster_count;

    const auto data = te::make_blobs<Float>(per_cluster,
                                            cluster_count,
                                            column_count,
                                            /*separation=*/Float(20.0),
                                            /*spread=*/Float(1.0));
    const auto x = homogen_table::wrap(data.data(), row_count, column_count);

    constexpr std::int64_t min_cluster_size = 25;
    const std::int64_t min_samples = GENERATE(5, 50);
    CAPTURE(min_samples);

    const auto bf_result = this->compute(this->get_descriptor(min_cluster_size, min_samples), x);

    INFO("the blobs are separated by 20x their spread, so all methods must recover them");
    REQUIRE(bf_result.get_cluster_count() == cluster_count);

    this->check_same_result(
        bf_result,
        this->compute(
            this->template get_method_descriptor<hdbscan::method::kd_tree>(min_cluster_size,
                                                                           min_samples),
            x));
    this->check_same_result(
        bf_result,
        this->compute(
            this->template get_method_descriptor<hdbscan::method::ball_tree>(min_cluster_size,
                                                                             min_samples),
            x));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: threaded MST is stable across runs",
                     "[hdbscan][batch]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // A single diffuse blob: the labels are decided by the MST edge order alone, and the weights
    // tie often, so the sort has to break ties the same way every run.
    constexpr std::int64_t row_count = 3000;
    constexpr std::int64_t column_count = 2;

    const auto data = te::make_blobs<Float>(row_count,
                                            /*cluster_count=*/1,
                                            column_count,
                                            /*separation=*/Float(0.0),
                                            /*spread=*/Float(1.0));
    const auto x = homogen_table::wrap(data.data(), row_count, column_count);

    const auto desc = this->get_descriptor(15, 5);
    const auto first = this->compute(desc, x);

    for (int run = 1; run < 3; run++) {
        CAPTURE(run);
        const auto again = this->compute(desc, x);
        REQUIRE(again.get_cluster_count() == first.get_cluster_count());
        this->check_responses_against_ref(again.get_responses(), first.get_responses());
    }
}

// =========================================================================
// brute_force method tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: compute mode check",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr double data[] = { 0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2,  0.15, 0.15, 5.0,
                                5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2, 5.15, 5.15, 10.0, 0.0 };
    const auto x = make_table<Float>(data, 11, 2);

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
                     "hdbscan brute_force: single cluster with noise",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr double data[] = {
        0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 0.15, 0.15, 100.0, 100.0,
    };
    const auto x = make_table<Float>(data, 6, 2);
    this->run_checks(x, 5, 5, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: two points",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr double data[] = { 0.0, 0.0, 1.0, 1.0 };
    const auto x = make_table<Float>(data, 2, 2);
    this->run_checks(x, 2, 2, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: all identical points",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr double data[] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    const auto x = make_table<Float>(data, 5, 2);
    this->run_checks(x, 2, 2, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: min_samples=1",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // min_samples=1 makes the core distance of every point 0.
    const auto x = make_table<Float>(two_small_groups_data, two_small_groups_row_count, 2);
    this->run_checks(x, 2, 1, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: varying min_cluster_size",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Clusters of 3, 5 and 7 points; each cluster disappears once min_cluster_size exceeds it.
    constexpr double data[] = {
        0.0,  0.0,  0.1,  0.1,  0.2,  0.0, //
        5.0,  5.0,  5.1,  5.1,  5.2,  5.0,  5.0,  5.2,  5.15,  5.15, //
        10.0, 10.0, 10.1, 10.1, 10.2, 10.0, 10.0, 10.2, 10.15, 10.15, 10.05, 10.05, 10.1, 10.0,
    };
    const auto x = make_table<Float>(data, 15, 2);

    this->run_checks(x, 2, 2, -1);
    this->run_checks(x, 4, 2, -1);
    this->run_checks(x, 6, 2, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: cosine two clusters",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Cosine clusters differ in direction: eight points near (1, 0) and eight near (0, 1).
    constexpr double data[] = {
        10.0, 0.5, 10.0, 1.0, 10.0, 0.0, 10.0, 0.8, 10.0, 0.3, 10.0, 0.6, 10.0, 0.4, 10.0, 0.9, 0.5,
        10.0, 1.0, 10.0, 0.0, 10.0, 0.8, 10.0, 0.3, 10.0, 0.6, 10.0, 0.4, 10.0, 0.9, 10.0,
    };
    const auto x = make_table<Float>(data, 16, 2);
    this->run_checks(x, 5, 5, distance_metric::cosine, 2.0, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: minkowski(p=1) equals manhattan",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_noise_data, two_blob_noise_row_count, 2);
    this->check_descriptors_agree(this->get_descriptor(5, 5, distance_metric::minkowski, 1.0),
                                  this->get_descriptor(5, 5, distance_metric::manhattan),
                                  x);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: minkowski(p=2) equals euclidean",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_noise_data, two_blob_noise_row_count, 2);
    this->check_descriptors_agree(this->get_descriptor(5, 5, distance_metric::minkowski, 2.0),
                                  this->get_descriptor(5, 5, distance_metric::euclidean),
                                  x);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: alpha=1 is default behavior",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    this->check_descriptors_agree(this->get_descriptor(5, 5),
                                  this->get_descriptor(5, 5).set_alpha(1.0),
                                  x);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: max_cluster_size=0 is no limit",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    this->check_descriptors_agree(this->get_descriptor(5, 5),
                                  this->get_descriptor(5, 5).set_max_cluster_size(0),
                                  x);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: max_cluster_size limits cluster size",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    const auto cluster_sizes = [&](std::int64_t max_cluster_size) {
        const auto desc = this->get_descriptor(5, 5).set_max_cluster_size(max_cluster_size);
        return this->cluster_sizes(this->get_labels(this->compute(desc, x)));
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

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: a zero row is at cosine distance 1 like in scikit-learn",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Two direction clusters plus two all-zero rows, which scikit-learn puts at cosine distance 1
    // from everything. Reference: sklearn.cluster.HDBSCAN(3, 3, metric="cosine").
    constexpr std::int64_t row_count = 14;
    constexpr double data[] = { 1.0,  0.05, 1.0,  0.1, 1.0, 0.0, 1.0, 0.08, 1.0,  0.03,
                                1.0,  0.12, 0.05, 1.0, 0.1, 1.0, 0.0, 1.0,  0.08, 1.0,
                                0.03, 1.0,  0.12, 1.0, 0.0, 0.0, 0.0, 0.0 };
    constexpr Float expected[] = { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1 };
    const auto x = make_table<Float>(data, row_count, 2);

    const auto result = this->compute(this->get_descriptor(3, 3, distance_metric::cosine), x);

    REQUIRE(result.get_cluster_count() == 2);
    check_same_partition(this->get_labels(result),
                         dal::array<Float>::wrap(expected, row_count),
                         row_count);
}

// =========================================================================
// Nightly tests: external datasets
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: susy 500K samples",
                     "[hdbscan][nightly][batch][external-dataset][susy]",
                     hdbscan_bf_kd_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe data =
        te::dataframe_builder{ "workloads/susy/dataset/susy_test.csv" }.build();
    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    // SUSY: 500K x 18, min_cluster_size=50, min_samples=25
    this->run_checks(x, 50, 25, -1);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force vs kd_tree: susy consistency",
                     "[hdbscan][nightly][batch][external-dataset][susy]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe data =
        te::dataframe_builder{ "workloads/susy/dataset/susy_test.csv" }.build();
    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    this->template check_methods_agree<hdbscan::method::kd_tree>(x, 50, 25);
}

// =========================================================================
// membership probability tests
// =========================================================================

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: noise points get zero probability",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // The last two points are far from the blob and from each other, so neither can belong to a
    // cluster of `min_cluster_size = 5`.
    constexpr double data[] = {
        0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 0.15, 0.15, 100.0, 100.0, -100.0, -100.0,
    };
    const auto x = make_table<Float>(data, 7, 2);

    const auto result = this->compute(
        this->get_descriptor(5, 5, result_options::responses | result_options::probabilities),
        x);

    this->check_probabilities(result, 7);

    const auto responses = this->get_labels(result);
    const auto probs = this->get_probabilities(result);
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

    // `min_cluster_size > n` returns before the MST is built and must still zero the
    // probabilities.
    constexpr double data[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    const auto x = make_table<Float>(data, 5, 1);

    const auto result = this->compute(
        this->get_descriptor(10, 2, result_options::responses | result_options::probabilities),
        x);

    REQUIRE(result.get_cluster_count() == 0);
    this->check_probabilities(result, 5);

    const auto probs = this->get_probabilities(result);
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

    // Coincident points persist to the death of their cluster, so every member gets 1.
    constexpr double data[] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    const auto x = make_table<Float>(data, 6, 2);

    const auto result = this->compute(
        this->get_descriptor(3, 3, result_options::responses | result_options::probabilities),
        x);

    this->check_probabilities(result, 6);

    const auto responses = this->get_labels(result);
    const auto probs = this->get_probabilities(result);
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

    // Reference: sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=5).probabilities_,
    // which treats only the last point as noise.
    constexpr double ref_probabilities[] = {
        0.6730265741, 1.0, 0.9348524093, 0.6549392602, 1.0,          1.0, 1.0,
        0.8113952439, 1.0, 1.0,          0.9089429453, 0.8214639762, 1.0, 0.9828347746,
        1.0,          1.0, 0.912268997,  1.0,          0.0915445561, 0.0,
    };
    constexpr std::int32_t ref_noise[] = { 19 };
    const auto x = make_table<Float>(reference_data, reference_row_count, 2);

    const auto result = this->compute(
        this->get_descriptor(5, 5, result_options::responses | result_options::probabilities),
        x);

    REQUIRE(result.get_cluster_count() == 2);
    this->check_probabilities(result, reference_row_count);

    const auto responses = this->get_labels(result);
    for (const auto i : ref_noise) {
        CAPTURE(i, responses[i]);
        REQUIRE(static_cast<std::int32_t>(responses[i]) == -1);
    }

    // Membership strengths do not depend on how clusters are numbered, so they compare directly.
    this->check_close(this->get_probabilities(result),
                      ref_probabilities,
                      reference_row_count,
                      te::get_tolerance<Float>(1e-4, 1e-7));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: probabilities can be requested alone",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    const auto result = this->compute(this->get_descriptor(5, 5, result_options::probabilities), x);

    REQUIRE_THROWS_AS(result.get_responses(), domain_error);
    this->check_probabilities(result, two_blob_row_count);
}

// =========================================================================
// single linkage tree tests
// =========================================================================

TEMPLATE_LIST_TEST_M(
    hdbscan_batch_test,
    "hdbscan brute_force: single linkage tree matches the reference implementation",
    "[hdbscan][batch][single_linkage_tree]",
    hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Reference: sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=5)._single_linkage_tree_.
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
    constexpr std::int64_t row_count = reference_row_count;
    const auto x = make_table<Float>(reference_data, row_count, 2);

    const auto result = this->compute(
        this->get_descriptor(5, 5, result_options::responses | result_options::single_linkage_tree),
        x);

    this->check_single_linkage_tree(result, row_count);

    const auto ref_table = homogen_table::wrap(ref_tree, row_count - 1, 4);
    const auto ref_rows = row_accessor<const double>(ref_table).pull({ 0, -1 });
    const auto tree = this->get_single_linkage_tree(result);
    check_same_hierarchy(tree, ref_rows, row_count, te::get_tolerance<Float>(1e-4, 1e-7));
    check_same_children(tree, ref_rows, row_count, te::get_tolerance<Float>(1e-4, 1e-7));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: single linkage tree re-cuts like dbscan_clustering",
                     "[hdbscan][batch][single_linkage_tree]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // A cut of the dendrogram gives the DBSCAN clustering at that distance. References:
    // sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=5).dbscan_clustering(cut).
    // Only the dense blob survives at 1.0, both blobs at 2.0, the far point joins the dense blob
    // at 5.0, and everything is one cluster at 10.0.
    constexpr std::int64_t row_count = reference_row_count;
    constexpr double cut_distances[] = { 1.0, 2.0, 5.0, 10.0 };
    constexpr std::int32_t ref_labels[][row_count] = {
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    };
    const auto x = make_table<Float>(reference_data, row_count, 2);

    const auto result =
        this->compute(this->get_descriptor(5, 5, result_options::single_linkage_tree), x);

    this->check_single_linkage_tree(result, row_count);
    const auto tree = this->get_single_linkage_tree(result);

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

    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    const auto result =
        this->compute(this->get_descriptor(5, 5, result_options::single_linkage_tree), x);

    REQUIRE_THROWS_AS(result.get_responses(), domain_error);
    this->check_single_linkage_tree(result, two_blob_row_count);

    // The blobs are ten units apart, so a cut at 1.0 splits them and keeps each one whole.
    const auto labels =
        labelling_at_cut(this->get_single_linkage_tree(result), two_blob_row_count, 1.0, 5);
    REQUIRE(labels[0] != labels[5]);
    for (std::int64_t i = 0; i < two_blob_row_count; i++) {
        CAPTURE(i, labels[i]);
        REQUIRE(labels[i] >= 0);
        REQUIRE(labels[i] == labels[(i < 5) ? 0 : 5]);
    }
}

// =========================================================================
// single-cluster (root-only) selection tests
// =========================================================================

/// One dense blob plus a ladder of increasingly distant points. The condensed tree never splits
/// into two clusters of `min_cluster_size` points, so `allow_single_cluster`, the epsilon and
/// the selection method alone decide the outcome. References come from
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

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: single-cluster selection demotes weak members",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(single_root_data, single_root_row_count, 2);

    // Without an epsilon the threshold is the root's death lambda, so only the three points that
    // reach it stay clustered, all with membership 1.
    constexpr std::int32_t ref_labels[] = { -1, -1, -1, 0,  0,  -1, -1, 0,
                                            -1, -1, -1, -1, -1, -1, -1, -1 };
    std::vector<double> ref_probabilities(single_root_row_count);
    for (std::int64_t i = 0; i < single_root_row_count; i++) {
        ref_probabilities[i] = ref_labels[i] < 0 ? 0.0 : 1.0;
    }

    const auto desc =
        this->get_descriptor(3, 3, result_options::responses | result_options::probabilities)
            .set_allow_single_cluster(true);
    const auto result = this->compute(desc, x);

    REQUIRE(result.get_cluster_count() == 1);
    this->check_responses_against_ref(result.get_responses(), ref_labels, single_root_row_count);
    this->check_close(this->get_probabilities(result),
                      ref_probabilities.data(),
                      single_root_row_count,
                      te::get_tolerance<Float>(1e-4, 1e-7));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: cluster_selection_epsilon sets the single-cluster "
                     "threshold",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(single_root_data, single_root_row_count, 2);

    // With an epsilon the threshold is `1 / epsilon`, independent of the tree, so the labels and
    // the membership strengths are pinned against scikit-learn.
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
        this->get_descriptor(3, 3, result_options::responses | result_options::probabilities)
            .set_allow_single_cluster(true)
            .set_cluster_selection_epsilon(epsilon);
    const auto result = this->compute(desc, x);

    REQUIRE(result.get_cluster_count() == 1);
    this->check_responses_against_ref(result.get_responses(), ref_labels, single_root_row_count);
    this->check_close(this->get_probabilities(result),
                      ref_probabilities,
                      single_root_row_count,
                      te::get_tolerance<Float>(1e-4, 1e-7));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: a root-only tree is all noise by default",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(single_root_data, single_root_row_count, 2);
    const auto result = this->compute(this->get_descriptor(3, 3), x);

    REQUIRE(result.get_cluster_count() == 0);
    REQUIRE(this->count_noise(this->get_labels(result)) == single_root_row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: allow_single_cluster selects the root when min_cluster_size "
                     "exceeds the row count",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(single_root_data, single_root_row_count, 2);

    // Every point falls out of the root at the same lambda, so scikit-learn keeps them all in
    // one cluster with membership 1.
    const std::int64_t min_cluster_size = GENERATE(17, 100);
    CAPTURE(min_cluster_size);
    const std::vector<std::int32_t> ref_labels(single_root_row_count, 0);
    const std::vector<double> ref_probabilities(single_root_row_count, 1.0);

    const auto desc =
        this->get_descriptor(min_cluster_size,
                             3,
                             result_options::responses | result_options::probabilities)
            .set_allow_single_cluster(true);
    const auto result = this->compute(desc, x);

    REQUIRE(result.get_cluster_count() == 1);
    this->check_responses_against_ref(result.get_responses(),
                                      ref_labels.data(),
                                      single_root_row_count);
    this->check_close(this->get_probabilities(result),
                      ref_probabilities.data(),
                      single_root_row_count,
                      te::get_tolerance<Float>(1e-4, 1e-7));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: leaf selection never picks the root cluster",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(single_root_data, single_root_row_count, 2);

    // The root is not a leaf of the cluster tree, so a tree that never splits has no candidate,
    // with or without `allow_single_cluster`.
    const bool allow_single_cluster = GENERATE(false, true);
    CAPTURE(allow_single_cluster);

    const auto desc = this->get_descriptor(3, 3)
                          .set_cluster_selection(cluster_selection_method::leaf)
                          .set_allow_single_cluster(allow_single_cluster);
    const auto result = this->compute(desc, x);

    REQUIRE(result.get_cluster_count() == 0);
    REQUIRE(this->count_noise(this->get_labels(result)) == single_root_row_count);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: leaf selection still splits a two-cluster dataset",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Once the condensed tree splits, leaf selection returns its leaves.
    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    const auto desc =
        this->get_descriptor(3, 3).set_cluster_selection(cluster_selection_method::leaf);
    const auto result = this->compute(desc, x);

    REQUIRE(result.get_cluster_count() == 2);
    const auto rows = this->get_labels(result);
    REQUIRE(static_cast<std::int32_t>(rows[0]) >= 0);
    REQUIRE(static_cast<std::int32_t>(rows[5]) >= 0);
    REQUIRE(static_cast<std::int32_t>(rows[0]) != static_cast<std::int32_t>(rows[5]));
    for (std::int64_t i = 0; i < two_blob_row_count; i++) {
        CAPTURE(i, rows[i]);
        REQUIRE(static_cast<std::int32_t>(rows[i]) ==
                static_cast<std::int32_t>(i < 5 ? rows[0] : rows[5]));
    }
}

/// scikit-learn's `test_hdbscan_allow_single_cluster_with_epsilon` dataset,
/// `np.random.RandomState(0).rand(150, 2)`. Excess-of-mass selection returns the root alone, so
/// the single-cluster labeling threshold decides which samples are noise.
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

/// Reference labels from `sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=5,
/// allow_single_cluster=True)` with `cluster_selection_epsilon=0.0`.
constexpr std::int32_t unstructured_ref_eps0[] = {
    0, 0,  0,  0, -1, 0,  0,  -1, 0,  -1, 0,  0,  0, 0,  -1, -1, 0,  -1, 0,  0, 0,  0, -1, 0, 0,
    0, -1, 0,  0, 0,  0,  0,  0,  0,  0,  -1, 0,  0, 0,  0,  0,  0,  0,  0,  0, 0,  0, 0,  0, 0,
    0, -1, 0,  0, 0,  0,  0,  -1, 0,  0,  0,  0,  0, -1, 0,  0,  -1, 0,  0,  0, -1, 0, -1, 0, -1,
    0, 0,  0,  0, 0,  0,  0,  0,  -1, 0,  0,  -1, 0, 0,  0,  0,  0,  0,  0,  0, 0,  0, 0,  0, -1,
    0, 0,  -1, 0, 0,  -1, 0,  -1, 0,  0,  0,  -1, 0, 0,  0,  0,  0,  -1, 0,  0, 0,  0, 0,  0, -1,
    0, -1, 0,  0, -1, 0,  -1, 0,  0,  -1, 0,  0,  0, 0,  0,  0,  0,  0,  -1, 0, 0,  0, 0,  0, -1,
};

/// The same with `cluster_selection_epsilon=0.18`.
constexpr std::int32_t unstructured_ref_eps018[] = {
    0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: single-cluster epsilon leaves early fall-outs as noise",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Samples that leave the root before the threshold lambda are noise, not root members.
    const auto x = make_table<Float>(unstructured_data, unstructured_row_count, 2);

    const double epsilon = GENERATE(0.0, 0.18);
    const std::int32_t* const ref_labels =
        (epsilon == 0.0) ? unstructured_ref_eps0 : unstructured_ref_eps018;
    const std::int64_t ref_noise_count = (epsilon == 0.0) ? 31 : 2;
    CAPTURE(epsilon, ref_noise_count);

    const auto desc =
        this->get_descriptor(5, 5).set_allow_single_cluster(true).set_cluster_selection_epsilon(
            epsilon);
    const auto result = this->compute(desc, x);

    REQUIRE(result.get_cluster_count() == 1);
    REQUIRE(this->count_noise(this->get_labels(result)) == ref_noise_count);
    this->check_responses_against_ref(result.get_responses(), ref_labels, unstructured_row_count);
}

/// Coincident points, where every condensed-tree edge carries the zero-distance lambda
/// scikit-learn stores as infinity, and the same set with one point 1e-31 away, which carries a
/// finite lambda of 1e31. References: `sklearn.cluster.HDBSCAN(min_cluster_size=3,
/// min_samples=2, allow_single_cluster=True)`.
constexpr std::int64_t zero_lambda_row_count = 4;
constexpr double zero_lambda_data[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
constexpr double split_lambda_data[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-31, 0.0 };

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: a zero-distance lambda outranks every epsilon",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(zero_lambda_data, zero_lambda_row_count, 2);

    constexpr std::int32_t ref_labels[] = { 0, 0, 0, 0 };
    constexpr double ref_probabilities[] = { 1.0, 1.0, 1.0, 1.0 };

    const double epsilon = GENERATE(0.0, 1e-30, 1e-31, 1e-40, 1e-310);
    CAPTURE(epsilon);

    const auto desc =
        this->get_descriptor(3, 2, result_options::responses | result_options::probabilities)
            .set_allow_single_cluster(true)
            .set_cluster_selection_epsilon(epsilon);
    const auto result = this->compute(desc, x);

    REQUIRE(result.get_cluster_count() == 1);
    this->check_responses_against_ref(result.get_responses(), ref_labels, zero_lambda_row_count);
    this->check_close(this->get_probabilities(result),
                      ref_probabilities,
                      zero_lambda_row_count,
                      te::get_tolerance<Float>(1e-4, 1e-7));
}

// Double only: 1e-31 squares to zero in float32, which makes all four points coincident.
TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: coincident points outlive a finite-lambda outlier",
                     "[hdbscan][batch][single-cluster]",
                     hdbscan_bf_only) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(split_lambda_data, zero_lambda_row_count, 2);

    // The coincident points drop out at the zero-distance lambda and the fourth at 1e31. Without
    // an epsilon only the coincident points reach the threshold; an epsilon of 1e-31 lowers it to
    // 1e31 and admits the outlier, whose membership stays 0.
    constexpr std::int32_t ref_labels_no_eps[] = { 0, 0, 0, -1 };
    constexpr std::int32_t ref_labels_eps[] = { 0, 0, 0, 0 };
    constexpr double ref_probabilities[] = { 1.0, 1.0, 1.0, 0.0 };

    const double epsilon = GENERATE(0.0, 1e-31, 1e-30);
    const std::int32_t* const ref_labels = (epsilon > 0.0) ? ref_labels_eps : ref_labels_no_eps;
    CAPTURE(epsilon);

    const auto desc =
        this->get_descriptor(3, 2, result_options::responses | result_options::probabilities)
            .set_allow_single_cluster(true)
            .set_cluster_selection_epsilon(epsilon);
    const auto result = this->compute(desc, x);

    REQUIRE(result.get_cluster_count() == 1);
    this->check_responses_against_ref(result.get_responses(), ref_labels, zero_lambda_row_count);
    this->check_close(this->get_probabilities(result),
                      ref_probabilities,
                      zero_lambda_row_count,
                      te::get_tolerance<Float>(1e-4, 1e-7));
}

// =========================================================================
// cluster centers and max_cluster_size against scikit-learn
// =========================================================================

// Two blobs with fringe points, so membership probabilities vary. The references are
// sklearn.cluster.HDBSCAN(4, 3, store_centers="both") with the metric below.
static constexpr std::int64_t centers_row_count = 20;
static constexpr double centers_data[] = { 0.845,  -0.233, 0.016, 0.204,  -0.394, 0.001,  -0.0,
                                           -0.877, 0.509,  0.3,   -0.313, -0.086, 0.253,  -0.131,
                                           -0.121, -0.727, 0.277, 0.062,  0.137,  -0.763, 7.321,
                                           6.123,  5.69,   7.623, 5.964,  4.839,  5.676,  4.169,
                                           6.84,   5.667,  5.406, 6.858,  4.679,  6.428,  4.348,
                                           5.47,   5.037,  7.17,  7.413,  5.736 };

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

    const auto x = make_table<Float>(centers_data, centers_row_count, 2);
    const auto desc =
        this->get_descriptor(4, 3, result_options::single_linkage_tree).set_alpha(2.0);
    const auto tree = this->get_single_linkage_tree(this->compute(desc, x));

    std::vector<double> merges;
    for (std::int64_t e = 0; e < centers_row_count - 1; ++e)
        merges.push_back(double(tree[4 * e + 2]));
    std::sort(merges.begin(), merges.end());
    for (std::int64_t e = 0; e < centers_row_count - 1; ++e) {
        CAPTURE(e);
        REQUIRE(std::abs(merges[e] - expected[e]) < te::get_tolerance<Float>(1e-5, 1e-9));
    }
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: centers are probability-weighted like scikit-learn's",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const bool manhattan = GENERATE(false, true);
    CAPTURE(manhattan);
    // A plain mean gives (0.1209, -0.225) for the first cluster, and the point nearest to it in
    // squared L2 is not scikit-learn's medoid either.
    const double expected_centroids[2][4] = {
        { 0.1048961814, -0.1708402313, 5.8384860213, 6.0206762016 },
        { 0.1074746881, -0.1861634668, 5.8397797591, 6.0354195229 },
    };
    const double expected_medoids[] = { 0.253, -0.131, 5.406, 6.858 };

    const auto x = make_table<Float>(centers_data, centers_row_count, 2);
    const auto desc =
        this->get_descriptor(4,
                             3,
                             result_options::responses | result_options::cluster_centers |
                                 result_options::medoid_centers)
            .set_metric(manhattan ? distance_metric::manhattan : distance_metric::euclidean)
            .set_store_centers(store_centers_method::both);
    const auto result = this->compute(desc, x);
    REQUIRE(result.get_cluster_count() == 2);

    // Cluster ids follow first appearance, and the first rows belong to the blob at the origin.
    const double tol = te::get_tolerance<Float>(1e-4, 1e-8);
    this->check_close(row_accessor<const Float>(result.get_cluster_centers()).pull({ 0, -1 }),
                      expected_centroids[manhattan],
                      4,
                      tol);
    this->check_close(row_accessor<const Float>(result.get_medoid_centers()).pull({ 0, -1 }),
                      expected_medoids,
                      4,
                      tol);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: medoids use the fitted metric, like scikit-learn's",
                     "[hdbscan][batch]",
                     hdbscan_types) {
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

    const auto x = make_table<Float>(centers_data, centers_row_count, 2);
    const auto desc =
        this->get_descriptor(4, 3, result_options::responses | result_options::medoid_centers)
            .set_metric(c.metric)
            .set_degree(3.0)
            .set_store_centers(store_centers_method::medoid);
    const auto result = this->compute(desc, x);
    REQUIRE(result.get_cluster_count() == 2);

    this->check_close(row_accessor<const Float>(result.get_medoid_centers()).pull({ 0, -1 }),
                      c.expected,
                      4,
                      te::get_tolerance<Float>(1e-4, 1e-8));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: medoids of clusters larger than one distance tile",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

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
        this->get_descriptor(50, 5, result_options::responses | result_options::medoid_centers)
            .set_metric(metric)
            .set_degree(3.0)
            .set_store_centers(store_centers_method::medoid);
    const auto result = this->compute(desc, x);
    REQUIRE(result.get_cluster_count() == 2);

    // The two lattices tie at every merge, so which one gets id 0 is not pinned down.
    const auto labels = this->get_labels(result);
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
                     "hdbscan: medoids are the same on every compute",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Eight 9 x 9 lattices, each with its center as the medoid.
    constexpr std::int64_t side = 9;
    constexpr std::int64_t cluster_count = 8;
    constexpr std::int64_t row_count = cluster_count * side * side;
    std::vector<Float> data;
    data.reserve(row_count * 2);
    for (std::int64_t c = 0; c < cluster_count; ++c) {
        for (std::int64_t i = 0; i < side; ++i) {
            for (std::int64_t j = 0; j < side; ++j) {
                data.push_back(Float(i - side / 2 + 100 * c));
                data.push_back(Float(j - side / 2));
            }
        }
    }
    const auto x = homogen_table::wrap(data.data(), row_count, 2);
    const auto desc =
        this->get_descriptor(20, 5, result_options::responses | result_options::medoid_centers)
            .set_store_centers(store_centers_method::medoid);

    for (int run = 0; run < 10; ++run) {
        CAPTURE(run);
        const auto result = this->compute(desc, x);
        REQUIRE(result.get_cluster_count() == cluster_count);
        const auto labels = this->get_labels(result);
        const auto medoids = row_accessor<const Float>(result.get_medoid_centers()).pull({ 0, -1 });
        for (std::int64_t c = 0; c < cluster_count; ++c) {
            const std::int64_t k = std::int64_t(labels[c * side * side]);
            CAPTURE(c, k);
            REQUIRE(medoids[2 * k] == Float(100 * c));
            REQUIRE(medoids[2 * k + 1] == Float(0));
        }
    }
}

TEMPLATE_LIST_TEST_M(
    hdbscan_batch_test,
    "hdbscan: max_cluster_size caps the root by its child clusters like scikit-learn",
    "[hdbscan][batch]",
    hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

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

    const auto x = make_table<Float>(source, row_count, 2);
    const auto desc =
        this->get_descriptor(3, 3).set_allow_single_cluster(true).set_max_cluster_size(7);
    const auto result = this->compute(desc, x);

    REQUIRE(result.get_cluster_count() == 1);
    check_same_partition(this->get_labels(result),
                         dal::array<Float>::wrap(expected, row_count),
                         row_count);
}

// =========================================================================
// GPU tests
// =========================================================================

#ifdef ONEDAL_DATA_PARALLEL

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: cpu and gpu results match",
                     "[hdbscan][batch]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_noise_data, two_blob_noise_row_count, 2);
    this->compare_host_and_device(this->get_descriptor(5, 5), x);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan: cpu and gpu results match on gold data",
                     "[hdbscan][batch][gold]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    this->compare_host_and_device(
        this->get_descriptor(gold_dataset::get_min_cluster_size(), gold_dataset::get_min_samples()),
        x);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan brute_force: cpu and gpu probabilities match",
                     "[hdbscan][batch][probabilities]",
                     hdbscan_bf_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    using Float = std::tuple_element_t<0, TestType>;

    const auto x = make_table<Float>(two_blob_data, two_blob_row_count, 2);
    const auto [host, device] = this->compare_host_and_device(
        this->get_descriptor(5, 5, result_options::responses | result_options::probabilities),
        x);

    // Membership strengths are per point, so the two backends must agree entry by entry.
    this->check_close(this->get_probabilities(device),
                      this->get_probabilities(host),
                      te::get_tolerance<Float>(1e-4, 1e-10));
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

    const auto [host, device] =
        this->compare_host_and_device(this->get_descriptor(gold_dataset::get_min_cluster_size(),
                                                           gold_dataset::get_min_samples(),
                                                           result_options::single_linkage_tree),
                                      x);

    this->check_single_linkage_tree(host, row_count);
    this->check_single_linkage_tree(device, row_count);

    // The gold data has two merges at an equal distance, which the backends may take in either
    // order, so the hierarchies are compared rather than the rows; see `check_same_hierarchy`.
    // The CPU forms squared L2 from norms and a dot product, which loses about eps * |x|^2 in
    // float32; the GPU subtracts coordinates.
    check_same_hierarchy(this->get_single_linkage_tree(device),
                         this->get_single_linkage_tree(host),
                         row_count,
                         te::get_tolerance<Float>(5e-4, 1e-10));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan gpu: the blocked core distances do not depend on the block size",
                     "[hdbscan][batch][gpu]",
                     hdbscan_tree_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // More than 256 neighbors take the blocked search. Blocks of 7 and 899 rows leave a tail.
    constexpr std::int64_t per_cluster = 300;
    constexpr std::int64_t cluster_count = 3;
    constexpr std::int64_t row_count = per_cluster * cluster_count;
    const auto data = te::make_blobs<Float>(per_cluster,
                                            cluster_count,
                                            2,
                                            /*separation=*/Float(20.0),
                                            /*spread=*/Float(1.0));
    const auto x = homogen_table::wrap(data.data(), row_count, 2);

    const auto compute_with_block = [&](std::int64_t block_size) {
        return this->compute(this->get_descriptor(50, 257).set_distance_block_size(block_size), x);
    };

    const auto reference = compute_with_block(row_count);
    REQUIRE(reference.get_cluster_count() == cluster_count);

    const std::int64_t block_size = GENERATE(1, 7, 899);
    CAPTURE(block_size);
    this->check_same_result(reference, compute_with_block(block_size));
}

// Thousands of rows give the GPU block tree three levels and a partial last leaf.
TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan gpu: the block tree search matches the cpu for every Lp metric",
                     "[hdbscan][batch][gpu]",
                     hdbscan_bf_kd_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    constexpr std::int64_t per_cluster = 1501;
    constexpr std::int64_t cluster_count = 4;
    constexpr std::int64_t column_count = 5;
    constexpr std::int64_t row_count = per_cluster * cluster_count;
    const auto data = te::make_blobs<Float>(per_cluster,
                                            cluster_count,
                                            column_count,
                                            /*separation=*/Float(20.0),
                                            /*spread=*/Float(1.0));
    const auto x = homogen_table::wrap(data.data(), row_count, column_count);

    const auto metric = GENERATE(distance_metric::euclidean,
                                 distance_metric::manhattan,
                                 distance_metric::minkowski,
                                 distance_metric::chebyshev);
    const double alpha = GENERATE(1.0, 2.0);
    CAPTURE(metric, alpha);

    const auto desc =
        this->template get_method_descriptor<Method>(25, 10, metric, 3.0)
            .set_alpha(alpha)
            .set_result_options(result_options::responses | result_options::single_linkage_tree);
    const auto [host, device] = this->compare_device_with_float64_host(desc, x);
    REQUIRE(device.get_cluster_count() == cluster_count);

    // Both build an exact MST, so the merge distances agree up to the device rounding.
    check_same_hierarchy(this->get_single_linkage_tree(device),
                         this->get_single_linkage_tree(host),
                         row_count,
                         te::get_tolerance<Float>(1e-4, 1e-9));
}

// From 16 columns up the GPU takes the matrix path whenever the matrix fits.
TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan gpu: the matrix path matches the cpu for every metric",
                     "[hdbscan][batch][gpu]",
                     hdbscan_bf_kd_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    constexpr std::int64_t per_cluster = 501;
    constexpr std::int64_t cluster_count = 4;
    constexpr std::int64_t column_count = 24;
    constexpr std::int64_t row_count = per_cluster * cluster_count;
    const auto data = te::make_blobs<Float>(per_cluster,
                                            cluster_count,
                                            column_count,
                                            /*separation=*/Float(20.0),
                                            /*spread=*/Float(1.0));
    const auto x = homogen_table::wrap(data.data(), row_count, column_count);

    const auto metric = GENERATE(distance_metric::euclidean,
                                 distance_metric::manhattan,
                                 distance_metric::minkowski,
                                 distance_metric::chebyshev);
    const double alpha = GENERATE(1.0, 2.0);
    CAPTURE(metric, alpha);

    const auto desc =
        this->template get_method_descriptor<Method>(25, 10, metric, 3.0)
            .set_alpha(alpha)
            .set_result_options(result_options::responses | result_options::single_linkage_tree);
    const auto [host, device] = this->compare_device_with_float64_host(desc, x);
    REQUIRE(device.get_cluster_count() == cluster_count);

    check_same_hierarchy(this->get_single_linkage_tree(device),
                         this->get_single_linkage_tree(host),
                         row_count,
                         te::get_tolerance<Float>(1e-4, 1e-9));
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan gpu: the block tree breaks distance ties like the cpu brute force",
                     "[hdbscan][batch][gpu]",
                     hdbscan_bf_kd_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    // Integer grid points, every third one duplicated, so most MRD values tie exactly.
    constexpr std::int64_t side = 40;
    std::vector<Float> data;
    for (std::int64_t i = 0; i < side; i++) {
        for (std::int64_t j = 0; j < side; j++) {
            const Float shift = Float((i < side / 2) ? 0 : 3);
            const std::int64_t copies = ((i * side + j) % 3 == 0) ? 2 : 1;
            for (std::int64_t c = 0; c < copies; c++) {
                data.push_back(Float(i) + shift);
                data.push_back(Float(j));
            }
        }
    }
    const std::int64_t row_count = static_cast<std::int64_t>(data.size()) / 2;
    const auto x = homogen_table::wrap(data.data(), row_count, 2);

    const auto metric = GENERATE(distance_metric::euclidean,
                                 distance_metric::manhattan,
                                 distance_metric::chebyshev);
    CAPTURE(metric);

    const auto gpu_desc =
        this->template get_method_descriptor<Method>(20, 6, metric)
            .set_result_options(result_options::responses | result_options::single_linkage_tree);
    const auto cpu_desc =
        this->template get_method_descriptor<hdbscan::method::brute_force>(20, 6, metric)
            .set_result_options(result_options::responses | result_options::single_linkage_tree);
    const auto host = dal::compute(cpu_desc, x);
    const auto device = dal::compute(this->get_policy().get_queue(), gpu_desc, x);

    // The grid distances are exact in either precision, so even the tied merges agree.
    this->check_same_result(host, device);
    check_same_hierarchy(this->get_single_linkage_tree(device),
                         this->get_single_linkage_tree(host),
                         row_count,
                         1e-6);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan gpu: core distances past the tree k-NN capacity match the cpu",
                     "[hdbscan][batch][gpu]",
                     hdbscan_bf_kd_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    constexpr std::int64_t per_cluster = 700;
    constexpr std::int64_t cluster_count = 3;
    constexpr std::int64_t row_count = per_cluster * cluster_count;
    const auto data = te::make_blobs<Float>(per_cluster,
                                            cluster_count,
                                            3,
                                            /*separation=*/Float(20.0),
                                            /*spread=*/Float(1.0));
    const auto x = homogen_table::wrap(data.data(), row_count, 3);

    // Up to 256 neighbors take the tree k-NN, more the blocked search.
    const std::int64_t min_samples = GENERATE(16, 33, 65, 129, 256, 257);
    CAPTURE(min_samples);
    const auto desc =
        this->template get_method_descriptor<Method>(50, min_samples)
            .set_result_options(result_options::responses | result_options::single_linkage_tree);
    const auto [host, device] = this->compare_device_with_float64_host(desc, x);
    // The blocked search forms squared L2 from norms and a dot product, which loses about
    // eps * |x|^2 in float32.
    const double float_tol = (min_samples > 256) ? 5e-3 : 1e-4;
    check_same_hierarchy(this->get_single_linkage_tree(device),
                         this->get_single_linkage_tree(host),
                         row_count,
                         te::get_tolerance<Float>(float_tol, 1e-9));
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::hdbscan::test
