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

#include "oneapi/dal/algo/dbscan/test/fixture.hpp"

namespace oneapi::dal::dbscan::test {

template <typename TestType>
class dbscan_batch_test : public dbscan_test<TestType, dbscan_batch_test<TestType>> {};

using dbscan_types = COMBINE_TYPES((float, double),
                                   (dbscan::method::brute_force,
                                    dbscan::method::kd_tree,
                                    dbscan::method::ball_tree));

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

using dbscan_cross_method_types = COMBINE_TYPES((float, double), (dbscan::method::brute_force));

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan cross-method comparison",
                     "[dbscan][batch]",
                     dbscan_cross_method_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 5.0, 5.1, 5.2,
                                 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 5.0, 5.1, 5.2 };
    const auto x = homogen_table::wrap(data, 9, 2);

    constexpr double epsilon = 0.5;
    constexpr std::int64_t min_observations = 2;

    const auto desc_bf =
        dbscan::descriptor<float_t, dbscan::method::brute_force>(epsilon, min_observations)
            .set_result_options(result_options::responses);
    const auto desc_kd =
        dbscan::descriptor<float_t, dbscan::method::kd_tree>(epsilon, min_observations)
            .set_result_options(result_options::responses);
    const auto desc_bt =
        dbscan::descriptor<float_t, dbscan::method::ball_tree>(epsilon, min_observations)
            .set_result_options(result_options::responses);

    const auto result_bf = te::compute(this->get_policy(), desc_bf, x, table{});
    const auto result_kd = te::compute(this->get_policy(), desc_kd, x, table{});
    const auto result_bt = te::compute(this->get_policy(), desc_bt, x, table{});

    REQUIRE(result_bf.get_cluster_count() == result_kd.get_cluster_count());
    REQUIRE(result_bf.get_cluster_count() == result_bt.get_cluster_count());

    const auto resp_bf = result_bf.get_responses();
    const auto resp_kd = result_kd.get_responses();
    const auto resp_bt = result_bt.get_responses();

    row_accessor<const float_t> acc_bf(resp_bf);
    row_accessor<const float_t> acc_kd(resp_kd);
    row_accessor<const float_t> acc_bt(resp_bt);

    const auto arr_bf = acc_bf.pull({ 0, -1 });
    const auto arr_kd = acc_kd.pull({ 0, -1 });
    const auto arr_bt = acc_bt.pull({ 0, -1 });

    for (std::int64_t i = 0; i < x.get_row_count(); ++i) {
        CAPTURE(i, arr_bf[i], arr_kd[i], arr_bt[i]);
        REQUIRE(arr_bf[i] == arr_kd[i]);
        REQUIRE(arr_bf[i] == arr_bt[i]);
    }
}

} // namespace oneapi::dal::dbscan::test
