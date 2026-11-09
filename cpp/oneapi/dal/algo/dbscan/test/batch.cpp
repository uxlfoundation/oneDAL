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

#include <array>
#include <limits>
#include <random>

#include "oneapi/dal/algo/dbscan/test/fixture.hpp"
#include "oneapi/dal/test/engine/blobs.hpp"

namespace oneapi::dal::dbscan::test {

template <typename TestType>
class dbscan_batch_test : public dbscan_test<TestType, dbscan_batch_test<TestType>> {};

using dbscan_types = COMBINE_TYPES((float, double), (dbscan::method::brute_force));

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan compute mode check",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 1.0, 4.0,
                                 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 1.0 };
    const auto x = homogen_table::wrap(data, 3, 5);

    constexpr double epsilon = 0.01;
    constexpr std::int64_t min_observations = 1;

    result_option_id res_all = result_option_id(dal::result_option_id_base(mask_full));

    const result_option_id compute_mode = GENERATE_COPY(result_options::responses,
                                                        result_options::core_flags,
                                                        result_options::core_observations,
                                                        result_options::core_observation_indices,
                                                        res_all);

    this->mode_checks(compute_mode, x, table{}, epsilon, min_observations);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan degenerated test",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 1.0, 4.0,
                                 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 1.0 };
    const auto x = homogen_table::wrap(data, 3, 5);

    constexpr double epsilon = 0.01;
    constexpr std::int64_t min_observations = 1;

    constexpr float_t weights[] = { 1.0, 1.1, 1, 2 };
    const auto w = homogen_table::wrap(weights, 3, 1);

    constexpr std::int32_t responses[] = { 0, 1, 2 };
    const auto r = homogen_table::wrap(responses, 3, 1);

    this->run_checks(x, w, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test, "dbscan boundary test", "[dbscan][batch]", dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr std::int64_t min_observations = 2;
    constexpr float_t data1[] = { 0.0, 1.0 };
    constexpr std::int32_t responses1[] = { 0, 0 };
    const auto x1 = homogen_table::wrap(data1, 2, 1);
    const auto r1 = homogen_table::wrap(responses1, 2, 1);
    constexpr double epsilon1 = 2.0;
    this->run_checks(x1, table{}, epsilon1, min_observations, r1);

    constexpr float_t data2[] = { 0.0, 1.0, 1.0 };
    constexpr std::int32_t responses2[] = { 0, 0, 0 };
    const auto x2 = homogen_table::wrap(data2, 3, 1);
    const auto r2 = homogen_table::wrap(responses2, 3, 1);
    constexpr double epsilon2 = 1.0;
    this->run_checks(x2, table{}, epsilon2, min_observations, r2);

    constexpr std::int32_t responses3[] = { -1, 0, 0 };
    const auto r3 = homogen_table::wrap(responses3, 3, 1);
    constexpr double epsilon3 = 0.999;
    this->run_checks(x2, table{}, epsilon3, min_observations, r3);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test, "dbscan weight test", "[dbscan][batch]", dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 1.0 };
    const auto x = homogen_table::wrap(data, 2, 1);

    constexpr std::int64_t min_observations = 6;

    constexpr std::int32_t responses1[] = { -1, -1 };
    const auto r_none = homogen_table::wrap(responses1, 2, 1);

    constexpr std::int32_t responses2[] = { 0, -1 };
    const auto r_first = homogen_table::wrap(responses2, 2, 1);

    constexpr std::int32_t responses3[] = { 0, 1 };
    const auto r_both = homogen_table::wrap(responses3, 2, 1);

    constexpr float_t weights1[] = { 5, 5 };
    const auto w1 = homogen_table::wrap(weights1, 2, 1);

    constexpr float_t weights2[] = { 6, 5 };
    const auto w2 = homogen_table::wrap(weights2, 2, 1);

    constexpr float_t weights3[] = { 6, 6 };
    const auto w3 = homogen_table::wrap(weights3, 2, 1);

    constexpr double epsilon1 = 0.5;
    this->run_checks(x, table{}, epsilon1, min_observations, r_none);
    this->run_checks(x, w1, epsilon1, min_observations, r_none);
    this->run_checks(x, w2, epsilon1, min_observations, r_first);
    this->run_checks(x, w3, epsilon1, min_observations, r_both);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan simple core observations test #1",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 2.0, 2.0 };
    const auto x = homogen_table::wrap(data, 8, 1);

    constexpr double epsilon = 1;
    constexpr std::int64_t min_observations = 1;

    constexpr std::int32_t responses[] = { 0, 1, 1, 1, 2, 3, 1, 1 };
    const auto r = homogen_table::wrap(responses, 8, 1);

    this->run_checks(x, table{}, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan simple core observations test #2",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0 };
    const auto x = homogen_table::wrap(data, 7, 1);

    constexpr double epsilon = 1;
    constexpr std::int64_t min_observations = 2;

    constexpr std::int32_t responses[] = { -1, 0, 0, 0, -1, -1, -1 };
    const auto r = homogen_table::wrap(responses, 7, 1);

    this->run_checks(x, table{}, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan simple core observations test #3",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0 };
    const auto x = homogen_table::wrap(data, 7, 1);

    constexpr double epsilon = 1;
    constexpr std::int64_t min_observations = 3;

    constexpr std::int32_t responses[] = { -1, 0, 0, 0, -1, -1, -1 };
    const auto r = homogen_table::wrap(responses, 7, 1);

    this->run_checks(x, table{}, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan simple core observations test #4",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0 };
    const auto x = homogen_table::wrap(data, 7, 1);

    constexpr double epsilon = 1;
    constexpr std::int64_t min_observations = 4;

    constexpr std::int32_t responses[] = { -1, -1, -1, -1, -1, -1, -1 };
    const auto r = homogen_table::wrap(responses, 7, 1);

    this->run_checks(x, table{}, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan euclidean matches the definition on data far from the origin",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;

    // Norm-expanded squared distances lose the spread to cancellation here unless the data are centered.
    constexpr std::int64_t row_count = 120;
    constexpr std::int64_t column_count = 4;
    constexpr std::int64_t min_observations = 5;
    auto x = te::make_blobs<float_t>(40, 3, column_count, 4.0, 3.0);
    for (auto& v : x) {
        v += float_t(1.0e4);
    }
    const std::vector<float_t> w(row_count, 1);
    const auto dist = this->reference_distances(x, column_count, distance_metric::euclidean, 2.0);
    const double epsilon = this->pick_epsilon(dist, row_count, 0.03);

    const bool mem_save_mode = GENERATE(false, true);
    CAPTURE(mem_save_mode);
    const auto desc =
        dbscan::descriptor<float_t, method_t>(epsilon, min_observations)
            .set_mem_save_mode(mem_save_mode)
            .set_result_options(result_options::responses | result_options::core_flags |
                                result_options::core_observation_indices |
                                result_options::core_observations);
    const auto result = te::compute(this->get_policy(),
                                    desc,
                                    homogen_table::wrap(x.data(), row_count, column_count));
    this->check_against_definition(result, dist, w, epsilon, min_observations);

    // The core observations are rows of the input, not of the centered copy.
    const auto indices =
        row_accessor<const std::int32_t>(result.get_core_observation_indices()).pull();
    const auto observations = row_accessor<const float_t>(result.get_core_observations()).pull();
    for (std::int64_t k = 0; k < indices.get_count(); k++) {
        for (std::int64_t j = 0; j < column_count; j++) {
            REQUIRE(observations[k * column_count + j] == x[indices[k] * column_count + j]);
        }
    }
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan euclidean confines a non-finite value to its row",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    // The GPU kernels can count a NaN distance as within epsilon.
    SKIP_IF(this->get_policy().is_gpu());
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;

    constexpr std::int64_t row_count = 120;
    constexpr std::int64_t column_count = 4;
    constexpr std::int64_t min_observations = 5;
    auto x = te::make_blobs<float_t>(40, 3, column_count, 4.0, 3.0);
    const std::vector<float_t> w(row_count, 1);
    const double epsilon = this->pick_epsilon(
        this->reference_distances(x, column_count, distance_metric::euclidean, 2.0),
        row_count,
        0.03);

    const float_t bad = GENERATE(std::numeric_limits<float_t>::quiet_NaN(),
                                 std::numeric_limits<float_t>::infinity());
    CAPTURE(bad);
    x[5 * column_count + 1] = bad;
    const auto dist = this->reference_distances(x, column_count, distance_metric::euclidean, 2.0);

    const auto desc =
        dbscan::descriptor<float_t, method_t>(epsilon, min_observations)
            .set_result_options(result_options::responses | result_options::core_flags |
                                result_options::core_observation_indices);
    const auto result = te::compute(this->get_policy(),
                                    desc,
                                    homogen_table::wrap(x.data(), row_count, column_count));
    REQUIRE(result.get_cluster_count() > 0);
    this->check_against_definition(result, dist, w, epsilon, min_observations);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan wide data with neighbors at exactly epsilon match the definition",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;

    // Pairs of points exactly `epsilon` apart, far from the origin and from the other pairs: each point
    // is a core only through its partner, which GEMM rounding on its own can push past `epsilon`.
    constexpr std::int64_t pair_count = 150;
    constexpr std::int64_t row_count = 2 * pair_count;
    constexpr std::int64_t column_count = 8;
    constexpr std::int64_t min_observations = 2;
    constexpr double epsilon = 1.0;
    std::vector<float_t> x(row_count * column_count);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> cell(0, 9);
    std::uniform_int_distribution<int> axis(0, column_count - 1);
    for (std::int64_t p = 0; p < pair_count; p++) {
        const int k = axis(rng);
        for (std::int64_t j = 0; j < column_count; j++) {
            const float_t base = float_t(1000 + 3 * cell(rng));
            x[(2 * p) * column_count + j] = base;
            x[(2 * p + 1) * column_count + j] = base + (j == k ? float_t(1) : float_t(0));
        }
    }
    const std::vector<float_t> w(row_count, 1);
    const auto dist = this->reference_distances(x, column_count, distance_metric::euclidean, 2.0);

    const auto desc =
        dbscan::descriptor<float_t, method_t>(epsilon, min_observations)
            .set_result_options(result_options::responses | result_options::core_flags |
                                result_options::core_observation_indices);
    const auto result = te::compute(this->get_policy(),
                                    desc,
                                    homogen_table::wrap(x.data(), row_count, column_count));
    this->check_against_definition(result, dist, w, epsilon, min_observations);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan metrics match the definition on reference distances",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;

    struct metric_case {
        distance_metric metric;
        double degree;
    };
    const metric_case c = GENERATE(metric_case{ distance_metric::euclidean, 2.0 },
                                   metric_case{ distance_metric::manhattan, 2.0 },
                                   metric_case{ distance_metric::minkowski, 3.0 },
                                   metric_case{ distance_metric::minkowski, 0.5 },
                                   metric_case{ distance_metric::chebyshev, 2.0 },
                                   metric_case{ distance_metric::cosine, 2.0 });
    // 2 and 5 columns take the narrow and the wide GPU kernels; 20 also takes the early exit.
    const std::int64_t column_count = GENERATE(2, 5, 20);
    const bool weighted = GENERATE(false, true);
    const bool mem_save_mode = GENERATE(false, true);
    CAPTURE(static_cast<int>(c.metric), c.degree, column_count, weighted, mem_save_mode);

    constexpr std::int64_t row_count = 120;
    constexpr std::int64_t min_observations = 5;
    const auto x = te::make_blobs<float_t>(40, 3, column_count, 4.0, 3.0);
    std::vector<float_t> w(row_count, 1);
    for (std::int64_t i = 0; weighted && i < row_count; i++) {
        w[i] = float_t(1 + i % 3);
    }

    const auto dist = this->reference_distances(x, column_count, c.metric, c.degree);
    const double epsilon = this->pick_epsilon(dist, row_count, 0.03);

    const auto desc =
        dbscan::descriptor<float_t, method_t>(epsilon, min_observations)
            .set_mem_save_mode(mem_save_mode)
            .set_metric(c.metric)
            .set_degree(c.degree)
            // The CPU fills the core flags only when the core indices are requested too.
            .set_result_options(result_options::responses | result_options::core_flags |
                                result_options::core_observation_indices);
    const auto result =
        te::compute(this->get_policy(),
                    desc,
                    homogen_table::wrap(x.data(), row_count, column_count),
                    weighted ? table{ homogen_table::wrap(w.data(), row_count, 1) } : table{});
    this->check_against_definition(result, dist, w, epsilon, min_observations);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan cosine metric handles zero rows like scikit-learn",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    // Two zero rows, at cosine distance 1 from every other row including each other, then two
    // collinear rows. Reference: sklearn.cluster.DBSCAN(eps, min_samples=m, metric="cosine") on
    // its unchunked path, where the self-distance is forced to 0 and a zero row is its own neighbor.
    constexpr float_t narrow_data[] = { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0 };
    const auto x_narrow = homogen_table::wrap(narrow_data, 4, 2);
    constexpr float_t wide_data[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0 };
    const auto x_wide = homogen_table::wrap(wide_data, 4, 5);

    struct zero_row_case {
        double epsilon;
        std::int64_t min_observations;
        std::array<std::int32_t, 4> responses;
    };
    const zero_row_case c = GENERATE(zero_row_case{ 0.1, 1, { 0, 1, 2, 2 } },
                                     zero_row_case{ 0.1, 2, { -1, -1, 0, 0 } },
                                     zero_row_case{ 1.0, 4, { 0, 0, 0, 0 } });
    const auto r = homogen_table::wrap(c.responses.data(), 4, 1);
    // On CPU the two memory modes are separate distance code paths.
    for (const auto& x : { x_narrow, x_wide }) {
        for (bool mem_save_mode : { false, true }) {
            CAPTURE(c.epsilon, c.min_observations, x.get_column_count(), mem_save_mode);
            const auto desc = this->get_descriptor(c.epsilon, c.min_observations)
                                  .set_mem_save_mode(mem_save_mode)
                                  .set_metric(distance_metric::cosine);
            const auto result = te::compute(this->get_policy(), desc, x, table{});
            this->check_responses_against_ref(result.get_responses(), r);
        }
    }
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan gold data clusters test",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());

    double epsilon = gold_dataset::get_epsilon();
    std::int64_t min_observations = gold_dataset::get_min_observations();

    const auto r = gold_dataset::get_expected_responses().get_table(this->get_homogen_table_id());

    this->run_checks(x, table{}, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan gold data clusters weights test",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    const auto weights = gold_dataset::get_weights().get_table(this->get_homogen_table_id());
    double epsilon = gold_dataset::get_epsilon();
    std::int64_t min_observations = gold_dataset::get_min_observations();

    const auto r =
        gold_dataset::get_expected_responses_with_weights().get_table(this->get_homogen_table_id());

    this->run_checks(x, weights, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan gold data dbi test",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;
    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());

    double epsilon = gold_dataset::get_epsilon();
    std::int64_t min_observations = gold_dataset::get_min_observations();

    float_t ref_dbi = gold_dataset::get_expected_dbi();

    this->dbi_determenistic_checks(x, epsilon, min_observations, ref_dbi, 1.0e-3);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "mnist: samples=10K, epsilon=1.7e3, min_observations=3",
                     "[dbscan][nightly][batch][external-dataset]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    const te::dataframe data =
        te::dataframe_builder{ "workloads/mnist/dataset/mnist_test.csv" }.build();

    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    constexpr double epsilon = 1.7e3;
    constexpr std::int64_t min_observations = 3;
    constexpr float_t ref_dbi = 1.584515;

    this->dbi_determenistic_checks(x, epsilon, min_observations, ref_dbi, 1.0e-3);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "hepmass: samples=10K, epsilon=5, min_observations=3",
                     "[dbscan][nightly][batch][external-dataset]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    const te::dataframe data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });
    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    constexpr double epsilon = 5;
    constexpr std::int64_t min_observations = 3;
    constexpr float_t ref_dbi = 0.78373;

    this->dbi_determenistic_checks(x, epsilon, min_observations, ref_dbi, 1.0e-3);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "road_network: samples=20K, epsilon=1.0e3, min_observations=220",
                     "[dbscan][nightly][batch][external-dataset]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    const te::dataframe data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/road_network/dataset/road_network_20t_cluster.csv" });
    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    constexpr double epsilon = 1.0e3;
    constexpr std::int64_t min_observations = 220;
    constexpr float_t ref_dbi = float_t(0.00036);

    this->dbi_determenistic_checks(x, epsilon, min_observations, ref_dbi, 1.0e-1);
}

} // namespace oneapi::dal::dbscan::test
