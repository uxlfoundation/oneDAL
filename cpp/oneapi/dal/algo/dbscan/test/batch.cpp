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

#include "oneapi/dal/algo/dbscan/test/fixture.hpp"

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

// The metric tests use 2 and 5 columns, on both sides of the GPU narrow/wide kernel threshold.

struct metric_case {
    distance_metric metric;
    double degree;
    double epsilon;
    const std::int32_t* responses;
};

constexpr std::int32_t pair_and_noise[] = { 0, 0, -1 };
constexpr std::int32_t all_noise[] = { -1, -1, -1 };
constexpr std::int32_t one_cluster[] = { 0, 0, 0 };

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan narrow metric test",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    // Distances (p0, p1), (p1, p2), (p0, p2): euclidean 1.41, 2.83, 4.24; manhattan 2, 4, 6;
    // chebyshev 1, 2, 3; minkowski(3) 1.26, 2.52, 3.78.
    constexpr float_t data[] = { 0.0, 0.0, 1.0, 1.0, 3.0, 3.0 };
    const auto x = homogen_table::wrap(data, 3, 2);

    const metric_case c =
        GENERATE(metric_case{ distance_metric::euclidean, 2.0, 1.5, pair_and_noise },
                 metric_case{ distance_metric::chebyshev, 2.0, 1.5, pair_and_noise },
                 metric_case{ distance_metric::minkowski, 3.0, 1.5, pair_and_noise },
                 metric_case{ distance_metric::manhattan, 2.0, 1.5, all_noise },
                 metric_case{ distance_metric::euclidean, 2.0, 2.5, pair_and_noise },
                 metric_case{ distance_metric::manhattan, 2.0, 2.5, pair_and_noise },
                 metric_case{ distance_metric::minkowski, 3.0, 2.5, pair_and_noise },
                 metric_case{ distance_metric::chebyshev, 2.0, 2.5, one_cluster });

    this->run_metric_checks(x,
                            c.epsilon,
                            2,
                            c.metric,
                            c.degree,
                            homogen_table::wrap(c.responses, 3, 1));
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan wide metric test",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    // Distances (p0, p1), (p1, p2), (p0, p2): euclidean 2.24, 3.61, 4; manhattan 5, 7, 4;
    // chebyshev 1, 3, 4; minkowski(3) 1.71, 3.14, 4.
    constexpr float_t data[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 1.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0 };
    const auto x = homogen_table::wrap(data, 3, 5);

    const metric_case c =
        GENERATE(metric_case{ distance_metric::euclidean, 2.0, 2.5, pair_and_noise },
                 metric_case{ distance_metric::chebyshev, 2.0, 2.5, pair_and_noise },
                 metric_case{ distance_metric::minkowski, 3.0, 2.5, pair_and_noise },
                 metric_case{ distance_metric::manhattan, 2.0, 2.5, all_noise },
                 metric_case{ distance_metric::euclidean, 2.0, 3.5, pair_and_noise },
                 metric_case{ distance_metric::manhattan, 2.0, 3.5, all_noise },
                 metric_case{ distance_metric::chebyshev, 2.0, 3.5, one_cluster },
                 metric_case{ distance_metric::minkowski, 3.0, 3.5, one_cluster });

    this->run_metric_checks(x,
                            c.epsilon,
                            2,
                            c.metric,
                            c.degree,
                            homogen_table::wrap(c.responses, 3, 1));
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan minkowski degree matches the dedicated metrics",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                 1.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0 };
    const auto x = homogen_table::wrap(data, 3, 5);

    constexpr std::int64_t min_observations = 2;
    const double epsilon = GENERATE(1.5, 2.5, 3.5, 4.5);
    struct degree_case {
        double degree;
        distance_metric dedicated;
    };
    const degree_case c = GENERATE(degree_case{ 2.0, distance_metric::euclidean },
                                   degree_case{ 1.0, distance_metric::manhattan });

    const auto reference = oneapi::dal::test::engine::compute(
        this->get_policy(),
        this->get_metric_descriptor(epsilon, min_observations, c.dedicated, 2.0, true),
        x,
        table{});
    this->run_metric_checks(x,
                            epsilon,
                            min_observations,
                            distance_metric::minkowski,
                            c.degree,
                            reference.get_responses());
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan cosine metric test",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    // The first two rows are collinear, so far apart in the euclidean sense but at cosine
    // distance 0; the third row is orthogonal to both, at cosine distance 1.
    constexpr float_t narrow_data[] = { 1.0, 0.0, 2.0, 0.0, 0.0, 1.0 };
    const auto x_narrow = homogen_table::wrap(narrow_data, 3, 2);
    constexpr float_t wide_data[] = { 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0 };
    const auto x_wide = homogen_table::wrap(wide_data, 3, 5);

    const auto r = homogen_table::wrap(pair_and_noise, 3, 1);
    this->run_metric_checks(x_narrow, 0.1, 2, distance_metric::cosine, 2.0, r);
    this->run_metric_checks(x_wide, 0.1, 2, distance_metric::cosine, 2.0, r);
}

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan cosine metric handles zero rows like scikit-learn",
                     "[dbscan][batch]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    // Two zero rows, at cosine distance 1 from every other row including each other, then two
    // collinear rows. Reference: sklearn.cluster.DBSCAN(eps, min_samples=m, metric="cosine").
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
    this->run_metric_checks(x_narrow,
                            c.epsilon,
                            c.min_observations,
                            distance_metric::cosine,
                            2.0,
                            r);
    this->run_metric_checks(x_wide, c.epsilon, c.min_observations, distance_metric::cosine, 2.0, r);
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
