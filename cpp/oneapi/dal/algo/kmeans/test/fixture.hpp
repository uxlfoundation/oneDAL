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

#pragma once

#include <limits>
#include <cmath>

#include "oneapi/dal/algo/kmeans/train.hpp"
#include "oneapi/dal/algo/kmeans/infer.hpp"
#include "oneapi/dal/algo/kmeans/test/data.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/csr_table_builder.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/metrics/clustering.hpp"

namespace oneapi::dal::kmeans::test {

namespace te = dal::test::engine;
namespace la = dal::test::engine::linalg;

using kmeans_types = COMBINE_TYPES((float, double),
                                   (kmeans::method::lloyd_dense, kmeans::method::lloyd_csr));

using kmeans_types_csr = COMBINE_TYPES((float, double), (kmeans::method::lloyd_csr));

template <typename TestType, typename Derived>
class kmeans_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    sparse_indexing data_indexing_; // for sparse data testing
    using base_t = te::crtp_algo_fixture<TestType, Derived>;
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using task_t = kmeans::task::clustering;
    using descriptor_t = kmeans::descriptor<float_t, method_t, task_t>;
    using train_input_t = kmeans::train_input<task_t>;
    using train_result_t = kmeans::train_result<task_t>;
    using infer_input_t = kmeans::infer_input<task_t>;
    using infer_result_t = kmeans::infer_result<task_t>;
    using model_t = kmeans::model<task_t>;

    descriptor_t get_descriptor(std::int64_t cluster_count,
                                std::int64_t max_iteration_count,
                                float_t accuracy_threshold) const {
        return descriptor_t{}
            .set_cluster_count(cluster_count)
            .set_max_iteration_count(max_iteration_count)
            .set_accuracy_threshold(accuracy_threshold);
    }

    kmeans::descriptor<float_t, kmeans::method::lloyd_dense, task_t> get_dense_descriptor(
        std::int64_t cluster_count,
        std::int64_t max_iteration_count,
        float_t accuracy_threshold) const {
        return kmeans::descriptor<float_t, kmeans::method::lloyd_dense, task_t>{}
            .set_cluster_count(cluster_count)
            .set_max_iteration_count(max_iteration_count)
            .set_accuracy_threshold(accuracy_threshold);
    }

    descriptor_t get_descriptor(std::int64_t cluster_count) const {
        return descriptor_t{ cluster_count };
    }

    bool is_sparse_method() {
        return std::is_same_v<method_t, kmeans::method::lloyd_csr>;
    }

    void exact_checks(const table& data,
                      const table& initial_centroids,
                      const table& ref_centroids,
                      const table& ref_responses,
                      std::int64_t cluster_count,
                      std::int64_t max_iteration_count,
                      float_t accuracy_threshold,
                      float_t ref_objective_function = -1.0,
                      bool test_convergence = false) {
        CAPTURE(cluster_count);

        INFO("create descriptor");
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = this->train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();
        check_train_result(kmeans_desc,
                           train_result,
                           ref_centroids,
                           ref_responses,
                           test_convergence);

        INFO("run inference");
        const auto kmeans_infer_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold)
                .set_result_options(kmeans::result_options::compute_exact_objective_function);
        const auto infer_result = this->infer(kmeans_infer_desc, model, data);
        check_infer_result(kmeans_infer_desc, infer_result, ref_responses, ref_objective_function);
    }

    void exact_checks_with_reordering(const table& data,
                                      const table& initial_centroids,
                                      const table& ref_centroids,
                                      const table& ref_responses,
                                      std::int64_t cluster_count,
                                      std::int64_t max_iteration_count,
                                      float_t accuracy_threshold,
                                      float_t ref_objective_function = -1.0,
                                      bool test_convergence = false) {
        CAPTURE(cluster_count);

        INFO("create descriptor");
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = this->train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();

        auto match_map = array<float_t>::zeros(cluster_count);
        find_match_centroids(ref_centroids,
                             model.get_centroids(),
                             ref_centroids.get_column_count(),
                             match_map);
        check_train_result(kmeans_desc,
                           train_result,
                           match_map,
                           ref_centroids,
                           ref_responses,
                           test_convergence);
        INFO("run inference");
        const auto infer_result = this->infer(kmeans_desc, model, data);
        check_infer_result(kmeans_desc,
                           infer_result,
                           match_map,
                           ref_responses,
                           ref_objective_function);
    }

    void check_on_gold_data() {
        const auto table_id = this->get_homogen_table_id();
        const auto data = gold_dataset::get_data().get_table(table_id);
        const auto initial_centroids = gold_dataset::get_initial_centroids().get_table(table_id);
        const auto expected_centroids = gold_dataset::get_expected_centroids().get_table(table_id);
        const auto expected_responses = gold_dataset::get_expected_responses().get_table(table_id);

        const std::int64_t cluster_count = gold_dataset::get_cluster_count();
        const std::int64_t max_iteration_count = 100;
        const float_t accuracy_threshold = 0.0;

        INFO("create descriptor");
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = this->train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();
        const auto centroids = model.get_centroids();

        auto match_map = array<float_t>::zeros(cluster_count);
        find_match_centroids(expected_centroids,
                             centroids,
                             expected_centroids.get_column_count(),
                             match_map);

        SECTION("check if centroids close to gold") {
            const double rel_tol = 1e-7;
            check_centroid_match_with_rel_tol(match_map, rel_tol, expected_centroids, centroids);
        }

        INFO("run inference");
        const auto infer_result = this->infer(kmeans_desc, model, data);
        const auto responses = infer_result.get_responses();

        SECTION("check if responses are expected") {
            check_response_match(match_map, expected_responses, responses);
        }

        SECTION("check if objective function value is expected") {
            const double objective = train_result.get_objective_function_value();
            const double expected_objective = gold_dataset::get_expected_objective();
            CAPTURE(objective, expected_objective);

            const double rel_tol = 1e-7;
            check_value_with_ref_tol(objective, expected_objective, rel_tol);
        }
    }

    void check_empty_clusters() {
        float_t data[] = { -10, -9.5, -9, -8.5, -8, -1, 1, 9, 9.5, 10 };
        const auto x = homogen_table::wrap(data, 10, 1);

        float_t initial_centroids[] = { -10, -10, -10 };
        const auto c_init = homogen_table::wrap(initial_centroids, 3, 1);

        float_t final_centroids[] = { -4.5, 10, 9.5 };
        const auto c_final = homogen_table::wrap(final_centroids, 3, 1);

        float_t responses[] = { 0, 0, 0, 0, 0, 0, 0, 2, 2, 1 };
        const auto y = homogen_table::wrap(responses, 10, 1);

        this->exact_checks(x, c_init, c_final, y, 3, 1, 0.0);
    }

    void check_empty_clusters_distinct_inits() {
        // Three distinct initial centroids (not the repeated -10 of
        // check_empty_clusters), all to the left of the data: every
        // point is closest to -10, so on iter 0 all six points land in
        // cluster 0 and clusters 1, 2 are empty. The two-pass merge
        // then fills cluster 1 with the farthest-from-own-centroid
        // candidate (10) and cluster 2 with the next farthest (9),
        // decrementing cluster 0's cS0 / cS1 on the way so its new
        // centroid becomes (-10 -9 +0 +1) / 4 = -4.5.
        float_t data[] = { -10, -9, 0, 1, 9, 10 };
        const auto x = homogen_table::wrap(data, 6, 1);

        float_t initial_centroids[] = { -10, -10.5, -11 };
        const auto c_init = homogen_table::wrap(initial_centroids, 3, 1);

        float_t final_centroids[] = { -4.5, 10, 9 };
        const auto c_final = homogen_table::wrap(final_centroids, 3, 1);

        float_t responses[] = { 0, 0, 0, 0, 2, 1 };
        const auto y = homogen_table::wrap(responses, 6, 1);

        this->exact_checks(x, c_init, c_final, y, 3, 1, 0.0);
    }

    void check_empty_clusters_all_duplicates() {
        // Degenerate input: every point is a duplicate, so the data holds
        // fewer distinct values than there are clusters and no partition
        // with k non-empty clusters exists. Relocation cannot invent one --
        // the only point it could move into the empty cluster already sits
        // exactly on the centroid it is assigned to, so moving it would
        // leave the source cluster's mean where it is and merely plant a
        // second centroid on top of the first. The kernel declines to move
        // the row and instead duplicates the centroid of the cluster that
        // holds the most observations, which is what scikit-learn returns
        // for the same input: both centroids on the single distinct value,
        // a zero objective function and every point assigned to cluster 0.
        const std::int64_t cluster_count = 2;
        const std::int64_t row_count = 4;
        const std::int64_t max_iteration_count = 10;

        float_t data[] = { 5, 5, 5, 5 };
        const auto x = homogen_table::wrap(data, row_count, 1);

        float_t initial_centroids[] = { 5, 100 };
        const auto c_init = homogen_table::wrap(initial_centroids, cluster_count, 1);

        // A zero threshold means "stop once the objective function stops
        // decreasing", which both the CPU and the GPU kernels now honour, so
        // these cases converge well inside `max_iteration_count`.
        const auto desc = get_descriptor(cluster_count, max_iteration_count, 0.0);
        const auto train_result = this->train(desc, x, c_init);

        const auto centroids =
            row_accessor<const float_t>(train_result.get_model().get_centroids()).pull({ 0, -1 });
        REQUIRE(centroids.get_count() == cluster_count);
        CAPTURE(centroids[0], centroids[1]);
        REQUIRE(centroids[0] == float_t(5));
        REQUIRE(centroids[1] == float_t(5));

        REQUIRE(train_result.get_objective_function_value() == float_t(0));

        // A duplicated centroid is a fixed point, so the run converges instead
        // of spending every allowed iteration on it.
        CAPTURE(train_result.get_iteration_count());
        REQUIRE(train_result.get_iteration_count() < max_iteration_count);

        const auto responses =
            row_accessor<const int>(train_result.get_responses()).pull({ 0, -1 });
        REQUIRE(responses.get_count() == row_count);
        for (std::int64_t i = 0; i < row_count; i++) {
            CAPTURE(i, responses[i]);
            REQUIRE(responses[i] == 0);
        }
    }

    void check_empty_clusters_drained_source() {
        // The globally farthest-from-its-centroid row is the only row of its
        // cluster: 100 is alone in the cluster it is assigned to, and it wins
        // the candidate selection (which goes purely by distance, as in
        // scikit-learn's `_relocate_empty_clusters`) for the one empty
        // cluster. Whether the source cluster keeps the row or is drained by
        // the theft differs between the CPU and the GPU kernels, so the run
        // reaches the same fixed point after a different number of
        // iterations; either way the surviving empty cluster gets a duplicate
        // of the biggest cluster's centroid, and the final state has both
        // distinct values represented and every point exactly on its centroid.
        const std::int64_t cluster_count = 3;
        const std::int64_t row_count = 3;
        const std::int64_t max_iteration_count = 10;

        float_t data[] = { 0, 0, 100 };
        const auto x = homogen_table::wrap(data, row_count, 1);

        float_t initial_centroids[] = { 1, 2, 3 };
        const auto c_init = homogen_table::wrap(initial_centroids, cluster_count, 1);

        // A zero threshold means "stop once the objective function stops
        // decreasing", which both the CPU and the GPU kernels now honour, so
        // these cases converge well inside `max_iteration_count`.
        const auto desc = get_descriptor(cluster_count, max_iteration_count, 0.0);
        const auto train_result = this->train(desc, x, c_init);

        const auto centroids =
            row_accessor<const float_t>(train_result.get_model().get_centroids()).pull({ 0, -1 });
        REQUIRE(centroids.get_count() == cluster_count);

        const auto responses =
            row_accessor<const int>(train_result.get_responses()).pull({ 0, -1 });
        REQUIRE(responses.get_count() == row_count);

        for (std::int64_t i = 0; i < row_count; i++) {
            const int response = responses[i];
            CAPTURE(i, response, data[i]);
            REQUIRE(response >= 0);
            REQUIRE(response < cluster_count);
            REQUIRE(centroids[response] == data[i]);
        }

        // Every centroid is one of the two distinct data values, and both of
        // them are used: nothing was left behind on an initial position.
        bool zero_found = false;
        bool hundred_found = false;
        for (std::int64_t i = 0; i < cluster_count; i++) {
            CAPTURE(i, centroids[i]);
            REQUIRE((centroids[i] == float_t(0) || centroids[i] == float_t(100)));
            zero_found = zero_found || (centroids[i] == float_t(0));
            hundred_found = hundred_found || (centroids[i] == float_t(100));
        }
        REQUIRE(zero_found);
        REQUIRE(hundred_found);

        REQUIRE(train_result.get_objective_function_value() == float_t(0));

        CAPTURE(train_result.get_iteration_count());
        REQUIRE(train_result.get_iteration_count() < max_iteration_count);
    }

    void check_empty_clusters_duplicate_groups() {
        // Four groups of three duplicated points and six clusters, so at
        // most four clusters can be non-empty. Only the points of one group
        // are farther than zero from the centroid they are assigned to on
        // the first iteration, and there are exactly three of them -- enough
        // to seed every empty cluster once. After that seeding the state is a
        // fixed point: every group sits on a centroid of its own, the extra
        // clusters duplicate one of those centroids and hold no points, and
        // the objective function is zero. So regardless of how the backend
        // breaks the distance ties, the run has to end with every point
        // exactly on its centroid and all four distinct values represented.
        const std::int64_t cluster_count = 6;
        const std::int64_t row_count = 12;
        const std::int64_t max_iteration_count = 100;

        float_t data[] = { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4 };
        const auto x = homogen_table::wrap(data, row_count, 1);

        float_t initial_centroids[] = { 0, 2, 3, 4, 5, 7 };
        const auto c_init = homogen_table::wrap(initial_centroids, cluster_count, 1);

        // A zero threshold means "stop once the objective function stops
        // decreasing", which both the CPU and the GPU kernels now honour, so
        // these cases converge well inside `max_iteration_count`.
        const auto desc = get_descriptor(cluster_count, max_iteration_count, 0.0);
        const auto train_result = this->train(desc, x, c_init);

        const auto centroids =
            row_accessor<const float_t>(train_result.get_model().get_centroids()).pull({ 0, -1 });
        REQUIRE(centroids.get_count() == cluster_count);

        const auto responses =
            row_accessor<const int>(train_result.get_responses()).pull({ 0, -1 });
        REQUIRE(responses.get_count() == row_count);

        // Every point ended up exactly on the centroid it is assigned to.
        for (std::int64_t i = 0; i < row_count; i++) {
            const int response = responses[i];
            CAPTURE(i, response, data[i]);
            REQUIRE(response >= 0);
            REQUIRE(response < cluster_count);
            REQUIRE(centroids[response] == data[i]);
        }

        // No centroid was left behind on its initial position: every one of
        // them is one of the four group values, the clusters that hold no
        // points duplicating one of the group centroids.
        for (std::int64_t i = 0; i < cluster_count; i++) {
            CAPTURE(i, centroids[i]);
            REQUIRE(centroids[i] >= float_t(1));
            REQUIRE(centroids[i] <= float_t(4));
            REQUIRE(centroids[i] == std::floor(centroids[i]));
        }

        // All four groups are represented, so no group shares a centroid.
        for (std::int64_t value = 1; value <= 4; value++) {
            bool found = false;
            for (std::int64_t i = 0; i < cluster_count; i++) {
                found = found || (centroids[i] == float_t(value));
            }
            CAPTURE(value);
            REQUIRE(found);
        }

        REQUIRE(train_result.get_objective_function_value() == float_t(0));

        // The state reached above is a fixed point, so the run converges well
        // before the iteration limit.
        CAPTURE(train_result.get_iteration_count());
        REQUIRE(train_result.get_iteration_count() < max_iteration_count);
    }

    void check_on_smoke_data() {
        const float_t data[] = { 1.0,  1.0, //
                                 2.0,  2.0, //
                                 1.0,  2.0, //
                                 2.0,  1.0, //
                                 -1.0, -1.0, //
                                 -1.0, -2.0, //
                                 -2.0, -1.0, //
                                 -2.0, -2.0 };
        const auto x = homogen_table::wrap(data, 8, 2);

        const float_t final_centroids[] = { -1.5, -1.5, 1.5, 1.5 };
        const auto c_final = homogen_table::wrap(final_centroids, 2, 2);

        const int responses[] = { 1, 1, 1, 1, 0, 0, 0, 0 };
        const auto y = homogen_table::wrap(responses, 8, 1);

        const auto model = this->train_with_initialization_checks(x, c_final, y, 2, 4, 0.001);

        const float_t data_infer[] = { 1.0,  1.0, //
                                       0.0,  1.0, //
                                       1.0,  0.0, //
                                       2.0,  2.0, //
                                       7.0,  0.0, //
                                       -1.0, 0.0, //
                                       -5.0, -5.0, //
                                       -5.0, 0.0, //
                                       -2.0, 1.0 };
        const auto x2 = homogen_table::wrap(data_infer, 9, 2);
        float_t expected_obj_function = 4;
        this->infer_checks(x, model, y, expected_obj_function);
    }

    void check_on_large_data_with_one_cluster() {
        constexpr std::int64_t row_count = 1024 * 1024;
        constexpr std::int64_t column_count = 1024 / sizeof(float_t);
        constexpr std::int64_t cluster_count = 1;
        constexpr std::int64_t max_iteration_count = 1;

        const auto x_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ row_count, column_count }.fill_uniform(-0.2, 0.5));
        const table x_table =
            x_dataframe.get_table(this->get_policy(), this->get_homogen_table_id());

        const auto first_row = row_accessor<const float_t>(x_table).pull({ 0, 1 });
        const auto c_init = homogen_table::wrap(first_row, 1, column_count);

        auto responses = array<std::int32_t>::zeros(row_count);
        const auto y = homogen_table::wrap(responses, row_count, 1);

        auto stat = te::compute_basic_statistics<float_t>(x_dataframe);
        const auto c_final = homogen_table::wrap(stat.get_means(), 1, column_count);
        auto variance = stat.get_variances().get_data();
        double obj_function = 0.0;
        for (std::int64_t i = 0; i < column_count; ++i) {
            obj_function += variance[i];
        }
        obj_function *= column_count - 1;

        this->exact_checks(x_table,
                           c_init,
                           c_final,
                           y,
                           cluster_count,
                           max_iteration_count,
                           obj_function);
    }

    void partial_centroids_stress_test() {
        constexpr std::int64_t row_count = 8 * 1024;
        constexpr std::int64_t column_count = 2 * 1024;
        constexpr std::int64_t cluster_count = 8 * 1024;

        const auto x_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ row_count, column_count }.fill_uniform(-0.2, 0.5));
        const table x = x_dataframe.get_table(this->get_policy(), this->get_homogen_table_id());

        const auto first_row = row_accessor<const float_t>(x).pull({ 0, 1 });
        const auto c_init = homogen_table::wrap(first_row.get_data(), 1, column_count);

        auto responses = array<std::int32_t>::zeros(1 * cluster_count);
        auto response_ptr = responses.get_mutable_data();
        auto first_response = &response_ptr[0];
        std::iota(first_response, first_response + row_count, std::int32_t(0));
        const auto y = homogen_table::wrap(responses.get_data(), row_count, 1);

        this->exact_checks(x, x, x, y, cluster_count, 1, 0.0);
    }

    void test_on_sparse_data(const oneapi::dal::test::engine::csr_make_blobs<float_t>& input,
                             std::int64_t max_iter_count,
                             float_t accuracy_threshold,
                             bool init_centroids) {
        const table data = input.get_data(this->get_policy());
        const table dense_data = input.get_dense_data(this->get_policy());

        const auto cluster_count = input.cluster_count_;
        REQUIRE(data.get_kind() == csr_table::kind());
        auto desc = this->get_descriptor(cluster_count, max_iter_count, accuracy_threshold);
        INFO("KMeans sparse training");
        if (init_centroids) {
            const table initial_centroids = input.get_initial_centroids();
            const auto train_result = this->train(desc, data, initial_centroids);
            check_response_match(input.get_responses(), train_result.get_responses());
        }
        else {
            const auto train_result = this->train(desc, data);
            const auto model = train_result.get_model();
            auto match_map = array<float_t>::zeros(cluster_count);
            find_match_centroids(input.get_result_centroids(),
                                 model.get_centroids(),
                                 input.column_count_,
                                 match_map);
            check_response_match(match_map, input.get_responses(), train_result.get_responses());
        }
    }

    void test_on_dataset(const std::string& dataset_path,
                         std::int64_t cluster_count,
                         std::int64_t max_iteration_count,
                         float_t expected_dbi,
                         float_t expected_obj,
                         float_t obj_ref_tol = 1.0e-4,
                         float_t dbi_ref_tol = 1.0e-3) {
        const te::dataframe data = te::dataframe_builder{ dataset_path }.build();
        const table x = data.get_table(this->get_homogen_table_id());
        this->dbi_deterministic_checks(x,
                                       cluster_count,
                                       max_iteration_count,
                                       0.0,
                                       expected_dbi,
                                       expected_obj,
                                       obj_ref_tol,
                                       dbi_ref_tol);
    }

    void test_optional_results_on_dataset(const std::string& dataset_path,
                                          std::int64_t cluster_count,
                                          std::int64_t max_iteration_count,
                                          float_t expected_dbi,
                                          float_t expected_obj,
                                          float_t obj_ref_tol = 1.0e-4,
                                          float_t dbi_ref_tol = 1.0e-3) {
        const te::dataframe data = te::dataframe_builder{ dataset_path }.build();
        const table x = data.get_table(this->get_homogen_table_id());
        this->optional_results_dbi_deterministic_checks(x,
                                                        cluster_count,
                                                        max_iteration_count,
                                                        0.0,
                                                        expected_dbi,
                                                        expected_obj,
                                                        obj_ref_tol,
                                                        dbi_ref_tol);
    }

    void dbi_deterministic_checks(const table& data,
                                  std::int64_t cluster_count,
                                  std::int64_t max_iteration_count,
                                  float_t accuracy_threshold,
                                  float_t ref_dbi,
                                  float_t ref_obj_func,
                                  float_t obj_ref_tol = 1.0e-4,
                                  float_t dbi_ref_tol = 1.0e-4) {
        CAPTURE(cluster_count);

        INFO("create descriptor");
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        const auto data_rows = row_accessor<const float_t>(data).pull({ 0, cluster_count });
        const auto initial_centroids =
            homogen_table::wrap(data_rows.get_data(), cluster_count, data.get_column_count());

        INFO("run training");
        const auto train_result = this->train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();
        REQUIRE(te::has_no_nans(model.get_centroids()));

        INFO("run inference");
        const auto infer_result = this->infer(kmeans_desc, model, data);
        REQUIRE(te::has_no_nans(infer_result.get_responses()));

        auto dbi =
            te::davies_bouldin_index(data, model.get_centroids(), infer_result.get_responses());
        CAPTURE(dbi, ref_dbi);
        CAPTURE(infer_result.get_objective_function_value(), ref_obj_func);
        REQUIRE(check_value_with_ref_tol(dbi, ref_dbi, dbi_ref_tol));
        REQUIRE(check_value_with_ref_tol(infer_result.get_objective_function_value(),
                                         ref_obj_func,
                                         obj_ref_tol));
    }

    void optional_results_dbi_deterministic_checks(const table& data,
                                                   std::int64_t cluster_count,
                                                   std::int64_t max_iteration_count,
                                                   float_t accuracy_threshold,
                                                   float_t ref_dbi,
                                                   float_t ref_obj_func,
                                                   float_t obj_ref_tol = 1.0e-4,
                                                   float_t dbi_ref_tol = 1.0e-4) {
        CAPTURE(cluster_count);

        INFO("create descriptor");
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        const auto data_rows = row_accessor<const float_t>(data).pull({ 0, cluster_count });
        const auto initial_centroids =
            homogen_table::wrap(data_rows.get_data(), cluster_count, data.get_column_count());

        INFO("run training");
        const auto train_result = this->train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();
        REQUIRE(te::has_no_nans(model.get_centroids()));

        const auto kmeans_desc_infer_assignments =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        const auto kmeans_desc_infer_obj_func =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold)
                .set_result_options(dal::kmeans::result_options::compute_exact_objective_function);

        INFO("run inference assignments");
        const auto infer_result_assignments =
            this->infer(kmeans_desc_infer_assignments, model, data);
        REQUIRE(te::has_no_nans(infer_result_assignments.get_responses()));

        auto dbi = te::davies_bouldin_index(data,
                                            model.get_centroids(),
                                            infer_result_assignments.get_responses());
        CAPTURE(dbi, ref_dbi);
        REQUIRE(check_value_with_ref_tol(dbi, ref_dbi, dbi_ref_tol));

        INFO("run inference just objective function");
        const auto infer_result_obj_func = this->infer(kmeans_desc_infer_obj_func, model, data);
        CAPTURE(infer_result_obj_func.get_objective_function_value(), ref_obj_func);
        REQUIRE(check_value_with_ref_tol(infer_result_obj_func.get_objective_function_value(),
                                         ref_obj_func,
                                         obj_ref_tol));
    }

    void dbi_deterministic_checks_with_centroids(const table& data,
                                                 const table& initial_centroids,
                                                 std::int64_t cluster_count,
                                                 std::int64_t max_iteration_count,
                                                 float_t accuracy_threshold,
                                                 float_t ref_dbi,
                                                 float_t ref_obj_func,
                                                 float_t obj_ref_tol = 1.0e-4,
                                                 float_t dbi_ref_tol = 1.0e-4) {
        CAPTURE(cluster_count);

        INFO("create descriptor");
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = this->train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();
        REQUIRE(te::has_no_nans(model.get_centroids()));

        INFO("run inference");
        const auto infer_result = this->infer(kmeans_desc, model, data);
        REQUIRE(te::has_no_nans(infer_result.get_responses()));

        auto dbi =
            te::davies_bouldin_index(data, model.get_centroids(), infer_result.get_responses());
        CAPTURE(dbi, ref_dbi);
        CAPTURE(infer_result.get_objective_function_value(), ref_obj_func);
        REQUIRE(check_value_with_ref_tol(dbi, ref_dbi, dbi_ref_tol));
        REQUIRE(check_value_with_ref_tol(infer_result.get_objective_function_value(),
                                         ref_obj_func,
                                         obj_ref_tol));
    }

    model_t train_with_initialization_checks(const table& data,
                                             const table& ref_centroids,
                                             const table& ref_responses,
                                             std::int64_t cluster_count,
                                             std::int64_t max_iteration_count,
                                             float_t accuracy_threshold) {
        CAPTURE(cluster_count);

        INFO("create descriptor");
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = this->train(kmeans_desc, data);
        check_train_result(kmeans_desc, train_result, ref_centroids, ref_responses, false);
        return train_result.get_model();
    }

    void infer_checks(const table& data,
                      const model_t& model,
                      const table& ref_responses,
                      float_t ref_objective_function = -1.0) {
        CAPTURE(model.get_cluster_count());

        INFO("create descriptor");
        const auto kmeans_desc =
            get_descriptor(model.get_cluster_count())
                .set_result_options(kmeans::result_options::compute_exact_objective_function);

        INFO("run inference");
        const auto infer_result = this->infer(kmeans_desc, model, data);
        check_infer_result(kmeans_desc, infer_result, ref_responses, ref_objective_function);
    }

    void check_train_result(const descriptor_t& desc,
                            const train_result_t& result,
                            const table& ref_centroids,
                            const table& ref_responses,
                            bool test_convergence = false) {
        const auto [centroids, responses, iteration_count] = unpack_result(result);

        check_nans(result);
        const float_t strict_rel_tol =
            5.f * std::numeric_limits<float_t>::epsilon() * iteration_count * 100;
        check_centroid_match_with_rel_tol(strict_rel_tol, ref_centroids, centroids);
        check_response_match(ref_responses, responses);
        if (test_convergence) {
            INFO("check convergence");
            REQUIRE(iteration_count < desc.get_max_iteration_count());
        }
    }

    void check_train_result(const descriptor_t& desc,
                            const train_result_t& result,
                            const array<float_t>& match_map,
                            const table& ref_centroids,
                            const table& ref_responses,
                            bool test_convergence = false) {
        const auto [centroids, responses, iteration_count] = unpack_result(result);

        check_nans(result);
        const float_t strict_rel_tol =
            std::numeric_limits<float_t>::epsilon() * iteration_count * 10;
        check_centroid_match_with_rel_tol(match_map, strict_rel_tol, ref_centroids, centroids);
        check_response_match(match_map, ref_responses, responses);

        if (test_convergence) {
            INFO("check convergence");
            REQUIRE(iteration_count < desc.get_max_iteration_count());
        }
    }

    bool check_value_with_ref_tol(float_t val, float_t ref_val, float_t ref_tol) {
        float_t max_abs = std::max(fabs(val), fabs(ref_val));
        if (max_abs == 0.0)
            return true;
        CAPTURE(val, ref_val, fabs(val - ref_val) / max_abs, ref_tol);
        return fabs(val - ref_val) / max_abs < ref_tol;
    }

    void check_base_infer_result(const descriptor_t& desc,
                                 const infer_result_t& result,
                                 float_t ref_objective_function) {
        const auto [responses, objective_function] = unpack_result(result);

        check_nans(result);

        INFO("check if non-negative objective function value is expected");
        REQUIRE(objective_function >= 0.0);

        float_t rel_tol = 1.0e-5;
        if (!(ref_objective_function < 0.0)) {
            CAPTURE(objective_function, ref_objective_function, rel_tol);
            REQUIRE(check_value_with_ref_tol(objective_function, ref_objective_function, rel_tol));
        }
    }

    void check_infer_result(const descriptor_t& desc,
                            const infer_result_t& result,
                            const table& ref_responses,
                            float_t ref_objective_function) {
        const auto [responses, objective_function] = unpack_result(result);

        check_base_infer_result(desc, result, ref_objective_function);
        check_response_match(ref_responses, responses);
    }

    void check_infer_result(const descriptor_t& desc,
                            const infer_result_t& result,
                            const array<float_t>& match_map,
                            const table& ref_responses,
                            float_t ref_objective_function) {
        const auto [responses, objective_function] = unpack_result(result);
        check_base_infer_result(desc, result, ref_objective_function);
        check_response_match(match_map, ref_responses, responses);
    }

    void check_centroid_match_with_rel_tol(float_t rel_tol, const table& left, const table& right) {
        INFO("check if centroid shape is expected");
        REQUIRE(left.get_row_count() == right.get_row_count());
        REQUIRE(left.get_column_count() == right.get_column_count());

        INFO("check if centroid match is expected");
        const auto left_rows = row_accessor<const float_t>(left).pull();
        const auto right_rows = row_accessor<const float_t>(right).pull();
        const float_t alpha = std::numeric_limits<float_t>::min();
        for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
            const float_t l = left_rows[i];
            const float_t r = right_rows[i];
            if (fabs(l - r) < alpha)
                continue;
            const float_t denom = fabs(l) + fabs(r) + alpha;
            if (fabs(l - r) / denom > rel_tol) {
                CAPTURE(l, r, l - r, rel_tol, (l - r) / denom / rel_tol);
                FAIL("Centroid feature mismatch");
            }
        }
    }

    void check_centroid_match_with_rel_tol(const array<float_t>& match_map,
                                           float_t rel_tol,
                                           const table& left,
                                           const table& right) {
        INFO("check if centroid shape is expected");
        REQUIRE(left.get_row_count() == right.get_row_count());
        REQUIRE(left.get_column_count() == right.get_column_count());

        INFO("check if centroid match is expected");
        const auto left_rows = row_accessor<const float_t>(left).pull();
        const auto right_rows = row_accessor<const float_t>(right).pull();
        const float_t alpha = std::numeric_limits<float_t>::min();
        std::int64_t cluster_count = left.get_row_count();
        std::int64_t feature_count = left.get_column_count();
        for (std::int64_t i = 0; i < cluster_count; i++) {
            for (std::int64_t j = 0; j < feature_count; j++) {
                const float_t l = left_rows[match_map[i] * feature_count + j];
                const float_t r = right_rows[i * feature_count + j];
                if (fabs(l - r) < alpha)
                    continue;
                const float_t denom = fabs(l) + fabs(r) + alpha;
                if (fabs(l - r) / denom > rel_tol) {
                    CAPTURE(l, r, l - r, rel_tol, (l - r) / denom / rel_tol);
                    FAIL("Centroid feature mismatch for mapped centroids");
                }
            }
        }
    }

    float_t squared_euclidian_distance(std::int64_t x_offset,
                                       const array<float_t>& x,
                                       std::int64_t y_offset,
                                       const array<float_t>& y,
                                       std::int64_t feature_count) {
        float_t sum = 0.0;
        for (std::int64_t i = 0; i < feature_count; i++) {
            float_t val = x[x_offset * feature_count + i] - y[y_offset * feature_count + i];
            sum += val * val;
        }
        return sum;
    }

    void find_match_centroids(const table& ref_centroids,
                              const table& centroids,
                              std::int64_t feature_count,
                              array<float_t>& match_map) {
        REQUIRE(ref_centroids.get_row_count() == centroids.get_row_count());
        REQUIRE(ref_centroids.get_column_count() == centroids.get_column_count());
        const auto ref_rows = row_accessor<const float_t>(ref_centroids).pull({ 0, -1 });
        const auto cur_rows = row_accessor<const float_t>(centroids).pull({ 0, -1 });
        std::int64_t cluster_count = centroids.get_row_count();
        auto match_counters = array<std::int64_t>::zeros(cluster_count);
        for (std::int64_t i = 0; i < cluster_count; i++) {
            float_t min_distance =
                squared_euclidian_distance(0, ref_rows, i, cur_rows, feature_count);
            for (std::int64_t j = 1; j < cluster_count; j++) {
                float_t probe_distance =
                    squared_euclidian_distance(j, ref_rows, i, cur_rows, feature_count);
                if (probe_distance < min_distance) {
                    match_map.get_mutable_data()[i] = j;
                    min_distance = probe_distance;
                }
            }
        }
        for (std::int64_t i = 0; i < cluster_count; i++) {
            std::int64_t match_count = 0;
            for (std::int64_t j = 0; j < cluster_count; j++) {
                match_count += match_map[i] == j ? 1 : 0;
            }
            REQUIRE(match_count == 1);
        }
    }

    void check_response_match(const table& left, const table& right) {
        INFO("check if response shape is expected");
        REQUIRE(left.get_row_count() == right.get_row_count());
        REQUIRE(left.get_column_count() == right.get_column_count());
        REQUIRE(left.get_column_count() == 1);
        INFO("check if response match is expected");
        const auto left_rows = row_accessor<const float_t>(left).pull({ 0, -1 });
        const auto right_rows = row_accessor<const float_t>(right).pull({ 0, -1 });
        for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
            const float_t l = left_rows[i];
            const float_t r = right_rows[i];
            if (l != r) {
                CAPTURE(l, r);
                FAIL("response mismatch");
            }
        }
    }

    void check_response_match(const array<float_t>& match_map,
                              const table& left,
                              const table& right) {
        INFO("check if response shape is expected");
        REQUIRE(left.get_row_count() == right.get_row_count());
        REQUIRE(left.get_column_count() == right.get_column_count());
        REQUIRE(left.get_column_count() == 1);

        INFO("check if response match is expected");
        const auto left_rows = row_accessor<const float_t>(left).pull({ 0, -1 });
        const auto right_rows = row_accessor<const float_t>(right).pull({ 0, -1 });
        for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
            const float_t l = left_rows[i];
            const float_t r = right_rows[i];
            if (l != match_map[r]) {
                CAPTURE(l, r, match_map[r]);
                FAIL("response mismatch for mapped centroids");
            }
        }
    }

    void check_nans(const train_result_t& result) {
        const auto [centroids, responses, iteration_count] = unpack_result(result);

        INFO("check if there is no NaN in centroids");
        REQUIRE(te::has_no_nans(centroids));

        INFO("check if there is no NaN in responses");
        REQUIRE(te::has_no_nans(responses));
    }

    void check_nans(const infer_result_t& result) {
        const auto [responses, objective_function] = unpack_result(result);

        INFO("check if there is no NaN in objective function values");
        REQUIRE(te::has_no_nans(homogen_table::wrap(&objective_function, 1, 1)));

        INFO("check if there is no NaN in responses");
        REQUIRE(te::has_no_nans(responses));
    }

    static auto unpack_result(const train_result_t& result) {
        const auto centroids = result.get_model().get_centroids();
        const auto responses = result.get_responses();
        const auto iteration_count = result.get_iteration_count();
        return std::make_tuple(centroids, responses, iteration_count);
    }

    static auto unpack_result(const infer_result_t& result) {
        const auto responses = result.get_responses();
        const auto objective_function = result.get_objective_function_value();
        return std::make_tuple(responses, objective_function);
    }
};

} // namespace oneapi::dal::kmeans::test
