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
#include "oneapi/dal/algo/dbscan/test/data.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::dbscan::test {

namespace te = dal::test::engine;

template <typename TestType>
class dbscan_spmd_test : public dbscan_test<TestType, dbscan_spmd_test<TestType>> {
public:
    using base_t = dbscan_test<TestType, dbscan_spmd_test<TestType>>;
    using float_t = typename base_t::float_t;
    using method_t = typename base_t::method_t;
    using result_t = typename base_t::result_t;
    using input_t = typename base_t::input_t;

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    template <typename... Args>
    result_t compute_override(Args &&...args) {
        return this->compute_via_spmd_threads_and_merge(rank_count_, std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<input_t> split_compute_input_override(std::int64_t split_count, Args &&...args) {
        // Data table is distributed across the ranks
        const input_t input{ std::forward<Args>(args)... };
        const auto split_data =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), split_count);

        std::vector<input_t> split_input;
        split_input.reserve(split_count);
        if (input.get_weights().has_data()) {
            const auto split_weights = te::split_table_by_rows<float_t>(this->get_policy(),
                                                                        input.get_weights(),
                                                                        split_count);

            for (std::int64_t i = 0; i < split_count; i++) {
                split_input.push_back( //
                    input_t{ split_data[i], split_weights[i] });
            }
        }
        else {
            for (std::int64_t i = 0; i < split_count; i++) {
                split_input.push_back( //
                    input_t{ split_data[i] });
            }
        }

        return split_input;
    }

    result_t merge_compute_result_override(const std::vector<result_t> &results) {
        // cluster count is the same for all ranks
        // responses are distributed accross the ranks, need to combine them into single table;
        std::vector<table> response_tables;
        std::vector<table> cores_tables;
        for (const auto &r : results) {
            if (r.get_responses().has_data()) {
                response_tables.push_back(r.get_responses());
            }
            if (r.get_core_flags().has_data()) {
                cores_tables.push_back(r.get_core_flags());
            }
        }
        const auto full_responses = te::stack_tables_by_rows<float_t>(response_tables);
        const auto full_cores = te::stack_tables_by_rows<float_t>(cores_tables);
        return result_t{}
            .set_cluster_count(results[0].get_cluster_count())
            .set_responses(full_responses)
            .set_core_flags(full_cores)
            .set_result_options(result_options::responses | result_options::core_flags);
    }

    void run_spmd_vs_batch_checks(const table &data,
                                  float_t epsilon,
                                  std::int64_t min_observations,
                                  const table &ref_responses) {
        auto dbscan_desc = dbscan::descriptor<float_t, method_t>(epsilon, min_observations)
                               .set_mem_save_mode(true);
        dbscan_desc.set_result_options(result_options::responses | result_options::core_flags);
        INFO("run computation");

        const auto compute_results = this->compute_via_spmd_threads(rank_count_, dbscan_desc, data);

        auto joined_result = this->merge_compute_result(compute_results);

        const auto compute_result_batch =
            oneapi::dal::test::engine::compute(this->get_policy(), dbscan_desc, data);
        INFO("check references");
        this->check_if_close(joined_result.get_core_flags(),
                             compute_result_batch.get_core_flags(),
                             "Cores");
        INFO("check references");
        this->check_if_close(joined_result.get_responses(),
                             compute_result_batch.get_responses(),
                             "responses");
        base_t::check_responses_against_ref(joined_result.get_responses(), ref_responses);
    }

    void run_spmd_response_checks(const table &data,
                                  const table &weights,
                                  float_t epsilon,
                                  std::int64_t min_observations,
                                  const table &ref_responses) {
        auto dbscan_desc = dbscan::descriptor<float_t, method_t>(epsilon, min_observations)
                               .set_mem_save_mode(true);
        dbscan_desc.set_result_options(result_options::responses | result_options::core_flags);

        INFO("run computation");
        const auto compute_results =
            this->compute_via_spmd_threads(rank_count_, dbscan_desc, data, weights);

        auto joined_result = this->merge_compute_result(compute_results);

        INFO("check references");
        base_t::check_responses_against_ref(joined_result.get_responses(), ref_responses);
    }
    void run_spmd_dbi_checks(const table &data,
                             float_t epsilon,
                             std::int64_t min_observations,
                             float_t ref_dbi,
                             float_t dbi_ref_tol = 1.0e-4) {
        INFO("create descriptor");
        const auto dbscan_desc =
            dbscan::descriptor<float_t, method_t>(epsilon, min_observations)
                .set_mem_save_mode(true)
                .set_result_options(result_options::responses | result_options::core_flags);

        INFO("run computation");
        const auto compute_results =
            this->compute_via_spmd_threads(rank_count_, dbscan_desc, data, table{});

        auto joined_result = this->merge_compute_result(compute_results);
        const auto cluster_count = joined_result.get_cluster_count();
        REQUIRE(cluster_count > 0);

        const auto responses = joined_result.get_responses();

        auto dbi = te::davies_bouldin_index(data, responses);
        REQUIRE(base_t::check_value_with_ref_tol(dbi, ref_dbi, dbi_ref_tol));
    }

private:
    std::int64_t rank_count_ = 1;
};

using dbscan_types = COMBINE_TYPES((float, double),
                                   (dbscan::method::brute_force,
                                    dbscan::method::kd_tree,
                                    dbscan::method::ball_tree));

TEMPLATE_LIST_TEST_M(dbscan_spmd_test, "dbscan degenerated test", "[dbscan][spmd]", dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

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

    this->set_rank_count(GENERATE(2, 3));
    this->run_spmd_response_checks(x, w, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test, "dbscan boundary test", "[dbscan][spmd]", dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    this->set_rank_count(2);
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr std::int64_t min_observations = 2;
    constexpr float_t data1[] = { 0.0, 1.0 };
    constexpr std::int32_t responses1[] = { 0, 0 };
    const auto x1 = homogen_table::wrap(data1, 2, 1);
    const auto r1 = homogen_table::wrap(responses1, 2, 1);
    constexpr double epsilon1 = 2.0;
    this->run_spmd_vs_batch_checks(x1, epsilon1, min_observations, r1);

    constexpr float_t data2[] = { 0.0, 1.0, 1.0 };
    constexpr std::int32_t responses2[] = { 0, 0, 0 };
    const auto x2 = homogen_table::wrap(data2, 3, 1);
    const auto r2 = homogen_table::wrap(responses2, 3, 1);
    constexpr double epsilon2 = 1.0;
    this->run_spmd_vs_batch_checks(x2, epsilon2, min_observations, r2);

    constexpr std::int32_t responses3[] = { -1, 0, 0 };
    const auto r3 = homogen_table::wrap(responses3, 3, 1);
    constexpr double epsilon3 = 0.999;
    this->run_spmd_vs_batch_checks(x2, epsilon3, min_observations, r3);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test, "dbscan weight test #1", "[dbscan][spmd]", dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

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

    constexpr double epsilon = 0.5;

    this->set_rank_count(2);
    this->run_spmd_response_checks(x, table{}, epsilon, min_observations, r_none);
    this->run_spmd_response_checks(x, w1, epsilon, min_observations, r_none);
    this->run_spmd_response_checks(x, w2, epsilon, min_observations, r_first);
    this->run_spmd_response_checks(x, w3, epsilon, min_observations, r_both);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test, "dbscan weight test #2", "[dbscan][spmd]", dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 1.0 };
    const auto x = homogen_table::wrap(data, 2, 1);

    constexpr std::int64_t min_observations = 6;

    constexpr std::int32_t responses1[] = { -1, -1 };
    const auto r_none = homogen_table::wrap(responses1, 2, 1);

    constexpr std::int32_t responses2[] = { 0, 0 };
    const auto r_both = homogen_table::wrap(responses2, 2, 1);

    constexpr float_t weights1[] = { 2, 2 };
    const auto w1 = homogen_table::wrap(weights1, 2, 1);

    constexpr float_t weights2[] = { 3, 3 };
    const auto w2 = homogen_table::wrap(weights2, 2, 1);

    constexpr float_t weights3[] = { 1, 5 };
    const auto w3 = homogen_table::wrap(weights3, 2, 1);

    constexpr float_t weights4[] = { 5, 1 };
    const auto w4 = homogen_table::wrap(weights4, 2, 1);
    constexpr double epsilon = 2;

    this->set_rank_count(2);
    this->run_spmd_response_checks(x, table{}, epsilon, min_observations, r_none);
    this->run_spmd_response_checks(x, w1, epsilon, min_observations, r_none);
    this->run_spmd_response_checks(x, w2, epsilon, min_observations, r_both);
    this->run_spmd_response_checks(x, w3, epsilon, min_observations, r_both);
    this->run_spmd_response_checks(x, w4, epsilon, min_observations, r_both);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test,
                     "dbscan simple core observations test #1",
                     "[dbscan][spmd]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 2.0, 2.0 };
    const auto x = homogen_table::wrap(data, 8, 1);

    constexpr double epsilon = 1;
    constexpr std::int64_t min_observations = 1;

    constexpr std::int32_t responses[] = { 0, 1, 1, 1, 2, 3, 1, 1 };
    const auto r = homogen_table::wrap(responses, 8, 1);

    this->set_rank_count(GENERATE(2, 3, 4));
    this->run_spmd_vs_batch_checks(x, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test,
                     "dbscan simple core observations test #2",
                     "[dbscan][spmd]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0 };
    const auto x = homogen_table::wrap(data, 7, 1);

    constexpr double epsilon = 1;
    constexpr std::int64_t min_observations = 2;

    constexpr std::int32_t responses[] = { -1, 0, 0, 0, -1, -1, -1 };
    const auto r = homogen_table::wrap(responses, 7, 1);

    this->set_rank_count(GENERATE(2, 3, 4));
    this->run_spmd_vs_batch_checks(x, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test,
                     "dbscan simple core observations test #3",
                     "[dbscan][spmd]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0 };
    const auto x = homogen_table::wrap(data, 7, 1);

    constexpr double epsilon = 1;
    constexpr std::int64_t min_observations = 3;

    constexpr std::int32_t responses[] = { -1, 0, 0, 0, -1, -1, -1 };
    const auto r = homogen_table::wrap(responses, 7, 1);

    this->set_rank_count(GENERATE(2, 3, 4));
    this->run_spmd_vs_batch_checks(x, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test,
                     "dbscan simple core observations test #4",
                     "[dbscan][spmd]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0 };
    const auto x = homogen_table::wrap(data, 7, 1);

    constexpr double epsilon = 1;
    constexpr std::int64_t min_observations = 4;

    constexpr std::int32_t responses[] = { -1, -1, -1, -1, -1, -1, -1 };
    const auto r = homogen_table::wrap(responses, 7, 1);

    this->set_rank_count(GENERATE(2, 3, 4));
    this->run_spmd_vs_batch_checks(x, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test,
                     "dbscan gold data clusters weights test",
                     "[dbscan][spmd]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());
    const auto weights = gold_dataset::get_weights().get_table(this->get_homogen_table_id());
    double epsilon = gold_dataset::get_epsilon();
    std::int64_t min_observations = gold_dataset::get_min_observations();

    const auto r =
        gold_dataset::get_expected_responses_with_weights().get_table(this->get_homogen_table_id());
    this->run_spmd_response_checks(x, weights, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test,
                     "dbscan gold data clusters test",
                     "[dbscan][spmd]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());

    double epsilon = gold_dataset::get_epsilon();
    std::int64_t min_observations = gold_dataset::get_min_observations();

    const auto r = gold_dataset::get_expected_responses().get_table(this->get_homogen_table_id());
    this->set_rank_count(GENERATE(2, 3, 4, 5));
    this->run_spmd_vs_batch_checks(x, epsilon, min_observations, r);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test,
                     "dbscan gold data dbi test",
                     "[dbscan][spmd]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    using float_t = std::tuple_element_t<0, TestType>;
    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());

    double epsilon = gold_dataset::get_epsilon();
    std::int64_t min_observations = gold_dataset::get_min_observations();

    float_t ref_dbi = gold_dataset::get_expected_dbi();

    this->set_rank_count(GENERATE(2, 3, 4, 5));
    this->run_spmd_dbi_checks(x, epsilon, min_observations, ref_dbi, 1.0e-3);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test,
                     "mnist: samples=10K, epsilon=1.7e3, min_observations=3",
                     "[dbscan][nightly][spmd][external-dataset]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    using float_t = std::tuple_element_t<0, TestType>;

    const te::dataframe data =
        te::dataframe_builder{ "workloads/mnist/dataset/mnist_test.csv" }.build();

    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    constexpr double epsilon = 1.7e3;
    constexpr std::int64_t min_observations = 3;
    constexpr float_t ref_dbi = 1.584515;
    this->set_rank_count(GENERATE(2, 3, 4));
    this->run_spmd_dbi_checks(x, epsilon, min_observations, ref_dbi, 1.0e-3);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test,
                     "hepmass: samples=10K, epsilon=5, min_observations=3",
                     "[dbscan][nightly][spmd][external-dataset]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    using float_t = std::tuple_element_t<0, TestType>;

    const te::dataframe data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });
    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    constexpr double epsilon = 5;
    constexpr std::int64_t min_observations = 3;
    constexpr float_t ref_dbi = 0.78373;
    this->set_rank_count(GENERATE(2, 3, 4));
    this->run_spmd_dbi_checks(x, epsilon, min_observations, ref_dbi, 1.0e-3);
}

TEMPLATE_LIST_TEST_M(dbscan_spmd_test,
                     "road_network: samples=20K, epsilon=1.0e3, min_observations=220",
                     "[dbscan][nightly][spmd][external-dataset]",
                     dbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    using float_t = std::tuple_element_t<0, TestType>;

    const te::dataframe data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/road_network/dataset/road_network_20t_cluster.csv" });
    const table x = data.get_table(this->get_policy(), this->get_homogen_table_id());

    constexpr double epsilon = 1.0e3;
    constexpr std::int64_t min_observations = 220;
    constexpr float_t ref_dbi = float_t(0.00036);
    this->set_rank_count(GENERATE(2, 3, 4));
    this->run_spmd_dbi_checks(x, epsilon, min_observations, ref_dbi, 1.0e-1);
}

} // namespace oneapi::dal::dbscan::test
