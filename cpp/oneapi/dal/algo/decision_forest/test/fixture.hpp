/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/train.hpp"
#include "oneapi/dal/algo/decision_forest/infer.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

#include <list>

#define sizeofa(p) sizeof(p) / sizeof(*p)

namespace oneapi::dal::decision_forest::test {

namespace df = dal::decision_forest;
namespace te = dal::test::engine;

template <typename T>
struct checker_info {
    typedef T (*checker_func)(const dal::v1::table& infer_responses,
                              const dal::v1::table& ground_truth);

    std::string name;
    checker_func check;
    double required_accuracy;
};

template <typename TestType, typename Derived>
class df_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using base_t = te::crtp_algo_fixture<TestType, Derived>;
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using task_t = std::tuple_element_t<2, TestType>;
    using descriptor_t = df::descriptor<float_t, method_t, task_t>;
    using train_input_t = df::train_input<task_t>;
    using train_result_t = df::train_result<task_t>;
    using infer_input_t = df::infer_input<task_t>;
    using infer_result_t = df::infer_result<task_t>;
    using model_t = df::model<task_t>;
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using Task = std::tuple_element_t<2, TestType>;

    bool is_gpu() {
        return this->get_policy().is_gpu();
    }

    bool not_available_on_device() {
        constexpr bool is_dense = std::is_same_v<Method, decision_forest::method::dense>;
        return this->get_policy().is_gpu() && is_dense;
    }

    auto get_default_descriptor() {
        return df::descriptor<Float, Method, Task>{};
    }

    auto get_cls_dataframe_base() {
        constexpr double required_accuracy = 0.95;
        constexpr std::int64_t row_count_train = 6;
        constexpr std::int64_t row_count_test = 3;
        constexpr std::int64_t column_count = 3;
        constexpr std::int64_t class_count = 2;

        static const float train_arr[] = { -2.f, -1.f, 0.f, -1.f, -1.f, 0.f, -1.f, -2.f, 0.f,
                                           +1.f, +1.f, 1.f, +1.f, +2.f, 1.f, +2.f, +1.f, 1.f };

        static const float test_arr[] = { -1.f, -1.f, 0.f, +2.f, +2.f, 1.f, +3.f, +2.f, 1.f };

        te::dataframe data{ array<float>::wrap(train_arr, row_count_train * column_count),
                            row_count_train,
                            column_count };
        te::dataframe data_test{ array<float>::wrap(test_arr, row_count_test * column_count),
                                 row_count_test,
                                 column_count };

        const std::list<checker_info<double>> checker_list = { this->get_cls_checker(
            1 - required_accuracy) };

        return std::make_tuple(data, data_test, class_count, checker_list);
    }

    auto get_cls_dataframe_weighted_base() {
        constexpr double required_accuracy = 0.95;
        constexpr std::int64_t row_count_train = 10;
        constexpr std::int64_t row_count_test = 3;
        constexpr std::int64_t column_count_train = 4;
        constexpr std::int64_t column_count_test = 3;
        constexpr std::int64_t class_count = 2;

        static const float train_arr[] = {
            -2.2408f, 0.5005f,  0.f,  0.1f, -1.6002f, -0.512f,  0.f,  0.2f,
            -2.3748f, 0.8277f,  0.f,  0.1f, -1.4341f, 1.5004f,  1.f,  0.1f,
            -0.9874f, 0.9996f,  1.f,  0.6f, -1.1954f, 1.2781f,  1.f,  0.1f,
            0.29484f, -0.7925f, 0.f,  0.1f, 0.6476f,  -0.8175f, 0.0f, 0.44f,
            2.3836f,  1.5691f,  1.0f, 0.1f, 1.51783f, 1.2214f,  1.0f, 0.1f,
        };

        static const float test_arr[] = { -1.f, -1.f, 0.f, +2.f, +2.f, 1.f, +3.f, +2.f, 1.f };

        te::dataframe data{ array<float>::wrap(train_arr, row_count_train * column_count_train),
                            row_count_train,
                            column_count_train };
        te::dataframe data_test{ array<float>::wrap(test_arr, row_count_test * column_count_test),
                                 row_count_test,
                                 column_count_test };

        const std::list<checker_info<double>> checker_list = { this->get_cls_checker(
            1 - required_accuracy) };

        return std::make_tuple(data, data_test, class_count, checker_list);
    }

    auto get_cls_dataframe(std::string ds_name, double required_accuracy) {
        const te::dataframe data =
            GENERATE_DATAFRAME(te::dataframe_builder{ ds_name + ".train.csv" });
        const te::dataframe data_test =
            GENERATE_DATAFRAME(te::dataframe_builder{ ds_name + ".test.csv" });

        const std::list<checker_info<double>> checker_list = { this->get_cls_checker(
            1 - required_accuracy) };
        return std::make_tuple(data, data_test, checker_list);
    }

    auto get_reg_dataframe_base(const double required_mse = 0.05) {
        constexpr std::int64_t row_count_train = 10;
        constexpr std::int64_t row_count_test = 5;
        constexpr std::int64_t column_count = 3;

        static const float train_arr[] = {
            0.1f,    0.25f,   0.0079f, 0.15f,   0.35f,   0.0160f, 0.25f,   0.55f,
            0.0407f, 0.3f,    0.65f,   0.0573f, 0.4f,    0.85f,   0.0989f, 0.45f,
            0.95f,   0.1240f, 0.55f,   1.15f,   0.1827f, 0.6f,    1.25f,   0.2163f,
            0.7f,    1.45f,   0.2919f, 0.8f,    1.65f,   0.3789f,
        };

        static const float test_arr[] = {
            0.2f,    0.45f, 0.0269f, 0.35f,   0.75f, 0.0767f, 0.5f,    1.05f,
            0.1519f, 0.65f, 1.35f,   0.2527f, 0.75f, 1.55f,   0.3340f,
        };

        te::dataframe data{ array<float>::wrap(train_arr, row_count_train * column_count),
                            row_count_train,
                            column_count };
        te::dataframe data_test{ array<float>::wrap(test_arr, row_count_test * column_count),
                                 row_count_test,
                                 column_count };

        const std::list<checker_info<double>> checker_list = { this->get_mse_checker(
            required_mse) };

        return std::make_tuple(data, data_test, checker_list);
    }

    auto get_reg_dataframe_weighted_base() {
        const double required_mse = 0.05;
        constexpr std::int64_t row_count_train = 10;
        constexpr std::int64_t row_count_test = 5;
        constexpr std::int64_t column_count_train = 4;
        constexpr std::int64_t column_count_test = 3;

        static const float train_arr[] = {
            0.1f,    0.25f, 0.0079f, 0.1f,  0.15f,   0.35f, 0.0160f, 0.2f,  0.25f,   0.55f,
            0.0407f, 0.1f,  0.3f,    0.65f, 0.0573f, 0.1f,  0.4f,    0.85f, 0.0989f, 0.6f,
            0.45f,   0.95f, 0.1240f, 0.1f,  0.55f,   1.15f, 0.1827f, 0.1f,  0.6f,    1.25f,
            0.2163f, 0.44f, 0.7f,    1.45f, 0.2919f, 0.1f,  0.8f,    1.65f, 0.3789f, 0.1f,
        };

        static const float test_arr[] = {
            0.6f,    0.45f, 0.0262f, 0.6f,    0.75f, 0.0671f, 0.6f,    1.05f,
            0.1281f, 0.6f,  1.35f,   0.2315f, 0.6f,  1.55f,   0.3113f,
        };

        te::dataframe data{ array<float>::wrap(train_arr, row_count_train * column_count_train),
                            row_count_train,
                            column_count_train };
        te::dataframe data_test{ array<float>::wrap(test_arr, row_count_test * column_count_test),
                                 row_count_test,
                                 column_count_test };

        const std::list<checker_info<double>> checker_list = { this->get_mse_checker(
            required_mse) };

        return std::make_tuple(data, data_test, checker_list);
    }

    auto get_reg_dataframe(std::string ds_name, double required_mse, double required_mae) {
        const te::dataframe data =
            GENERATE_DATAFRAME(te::dataframe_builder{ ds_name + ".train.csv" });
        const te::dataframe data_test =
            GENERATE_DATAFRAME(te::dataframe_builder{ ds_name + ".test.csv" });

        const std::list<checker_info<double>> checker_list = { this->get_mse_checker(required_mse),
                                                               this->get_mae_checker(
                                                                   required_mae) };

        return std::make_tuple(data, data_test, checker_list);
    }

    auto train_base_checks(const df::descriptor<Float, Method, Task>& desc,
                           const te::dataframe& data,
                           const te::table_id& data_table_id) {
        const auto x = data.get_table(data_table_id, range(0, -1));
        const auto y = data.get_table(data_table_id,
                                      range(data.get_column_count() - 1, data.get_column_count()));
        INFO("run training");
        const auto train_result = this->train(desc, x, y);
        check_train_shapes(desc, data, train_result);
        return train_result;
    }

    auto train_weighted_base_checks(const df::descriptor<Float, Method, Task>& desc,
                                    const te::dataframe& data,
                                    const te::table_id& data_table_id) {
        const auto x = data.get_table(data_table_id, range(0, -2));
        const auto y =
            data.get_table(data_table_id,
                           range(data.get_column_count() - 2, data.get_column_count() - 1));
        const auto w = data.get_table(data_table_id,
                                      range(data.get_column_count() - 1, data.get_column_count()));
        INFO("run training");
        const auto train_result = this->train(desc, x, y, w);
        check_train_shapes(desc, data, train_result);
        return train_result;
    }

    auto infer_base_checks(const df::descriptor<Float, Method, Task>& desc,
                           const te::dataframe& data,
                           const te::table_id& data_table_id,
                           const df::model<Task>& model,
                           const std::list<checker_info<double>>& checker_list) {
        const auto x_test = data.get_table(data_table_id, range(0, -1));
        const auto y_test =
            data.get_table(data_table_id,
                           range(data.get_column_count() - 1, data.get_column_count()));
        INFO("run inference");
        const auto infer_result = this->infer(desc, model, x_test);
        check_infer_shapes(desc, data, infer_result);

        INFO("check if infer accuracy is expected");
        for (auto ch : checker_list) {
            CAPTURE(desc.get_features_per_node());
            CAPTURE(desc.get_max_tree_depth());
            CAPTURE(ch.name);
            REQUIRE(ch.check(infer_result.get_responses(), y_test) < ch.required_accuracy + eps);
        }

        return infer_result;
    }

    template <typename Checker>
    void model_traverse_check(const df::model<Task>& model, Checker&& check) {
        INFO("run model check");
        for (std::int64_t tree_idx = 0; tree_idx < model.get_tree_count(); ++tree_idx) {
            CAPTURE(tree_idx);
            model.traverse_depth_first(tree_idx, std::forward<Checker>(check));
        }
    }

    void check_trees_node_min_sample_count(const df::model<Task>& model,
                                           std::int64_t min_observations_in_leaf_node) {
        INFO("run check trees' node min sample count");
        model_traverse_check(model, [&](const node_info<Task>& node) {
            CAPTURE(node.get_level());
            REQUIRE(node.get_sample_count() >= min_observations_in_leaf_node);
            return true;
        });
    }

    void check_train_shapes(const df::descriptor<Float, Method, Task>& desc,
                            const te::dataframe& data,
                            const df::train_result<Task>& result) {
        constexpr bool is_cls = std::is_same_v<Task, decision_forest::task::classification>;

        INFO("check if model shape is expected");
        REQUIRE(result.get_model().get_tree_count() == desc.get_tree_count());
        if constexpr (is_cls) {
            REQUIRE(result.get_model().get_class_count() == desc.get_class_count());
        }

        if (check_mask_flag(desc.get_error_metric_mode(), error_metric_mode::out_of_bag_error)) {
            INFO("check if oob error shape is expected");
            REQUIRE(result.get_oob_err().has_data());
            REQUIRE(result.get_oob_err().get_row_count() == 1);
            REQUIRE(result.get_oob_err().get_column_count() == 1);
        }

        if (check_mask_flag(desc.get_error_metric_mode(),
                            error_metric_mode::out_of_bag_error_per_observation)) {
            INFO("check if oob error per observation shape is expected");
            REQUIRE(result.get_oob_err_per_observation().has_data());
            REQUIRE(result.get_oob_err_per_observation().get_row_count() == data.get_row_count());
            REQUIRE(result.get_oob_err_per_observation().get_column_count() == 1);
        }

        if (variable_importance_mode::none != desc.get_variable_importance_mode()) {
            INFO("check if variable improtance shape is expected");
            REQUIRE(result.get_var_importance().has_data());
            REQUIRE(result.get_var_importance().get_row_count() == 1);
            REQUIRE(result.get_var_importance().get_column_count() == data.get_column_count() - 1);
        }
    }

    void check_infer_shapes(const df::descriptor<Float, Method, Task>& desc,
                            const te::dataframe& data,
                            const df::infer_result<Task>& result) {
        constexpr bool is_cls = std::is_same_v<Task, decision_forest::task::classification>;
        if constexpr (is_cls) {
            if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_responses)) {
                INFO("check if infer responses shape is expected");
                REQUIRE(result.get_responses().has_data());
                REQUIRE(result.get_responses().get_row_count() == data.get_row_count());
                REQUIRE(result.get_responses().get_column_count() == 1);
            }

            if (check_mask_flag(desc.get_infer_mode(), infer_mode::class_probabilities)) {
                INFO("check if infer probabilities shape is expected");
                REQUIRE(result.get_probabilities().has_data());
                REQUIRE(result.get_probabilities().get_row_count() == data.get_row_count());
                REQUIRE(result.get_probabilities().get_column_count() == desc.get_class_count());
            }
        }
        else {
            INFO("check if infer responses shape is expected");
            REQUIRE(result.get_responses().has_data());
            REQUIRE(result.get_responses().get_row_count() == data.get_row_count());
            REQUIRE(result.get_responses().get_column_count() == 1);
        }
    }

    void check_var_importance_matches_required(const df::descriptor<Float, Method, Task>& desc,
                                               const df::train_result<Task>& train_result,
                                               const te::dataframe& var_imp_data,
                                               const te::table_id& data_table_id,
                                               double accuracy_threshold) {
        if (variable_importance_mode::none != desc.get_variable_importance_mode()) {
            INFO("check if match of variable importance vs required one is expected");
            const auto required_var_imp = var_imp_data.get_table(data_table_id);
            std::int64_t row_ind = 0;
            switch (desc.get_variable_importance_mode()) {
                case variable_importance_mode::mda_raw: row_ind = 1; break;
                case variable_importance_mode::mda_scaled: row_ind = 2; break;
                default: row_ind = 0; break;
            };

            const auto var_imp_val =
                dal::row_accessor<const Float>(train_result.get_var_importance()).pull();
            const auto required_var_imp_val =
                dal::row_accessor<const float>(required_var_imp).pull({ row_ind, row_ind + 1 });

            for (std::int32_t i = 0; i < var_imp_val.get_count(); i++) {
                if (required_var_imp_val[i] > 0.0) {
                    REQUIRE(((required_var_imp_val[i] - var_imp_val[i]) / required_var_imp_val[i]) <
                            accuracy_threshold + eps);
                }
            }
        }
    }

    void check_oob_err_matches_required(const df::descriptor<Float, Method, Task>& desc,
                                        const df::train_result<Task>& train_result,
                                        double required_oob_error,
                                        double accuracy_threshold) {
        if (check_mask_flag(desc.get_error_metric_mode(), error_metric_mode::out_of_bag_error)) {
            INFO("check if match of oob error vs required one is expected");
            const auto oob_err_val =
                dal::row_accessor<const double>(train_result.get_oob_err()).pull();
            if (required_oob_error > 0.0) {
                REQUIRE(std::abs((required_oob_error - oob_err_val[0]) / required_oob_error) <
                        accuracy_threshold + eps);
            }
        }
    }

    void check_oob_err_matches_oob_err_per_observation(
        const df::descriptor<Float, Method, Task>& desc,
        const df::train_result<Task>& train_result,
        double accuracy_threshold) {
        if (check_mask_flag(desc.get_error_metric_mode(),
                            error_metric_mode::out_of_bag_error_per_observation)) {
            INFO("check if match of oob error vs cumulative oob error per observation is expected");
            const auto oob_err_val =
                dal::row_accessor<const double>(train_result.get_oob_err()).pull();
            const auto oob_err_per_obs_arr =
                dal::row_accessor<const double>(train_result.get_oob_err_per_observation()).pull();

            double ref_oob_err = 0.0;
            for (std::int64_t i = 0; i < oob_err_per_obs_arr.get_count(); i++) {
                if (oob_err_per_obs_arr[i] >= 0.0) {
                    ref_oob_err += oob_err_per_obs_arr[i];
                }
                else {
                    REQUIRE(oob_err_per_obs_arr[i] >= -1.0);
                }
            }

            if (oob_err_val[0] > 0.0) {
                REQUIRE(((oob_err_val[0] - ref_oob_err) / oob_err_val[0]) <
                        accuracy_threshold + eps);
            }
        }
    }

    checker_info<double> get_cls_checker(double required_accuracy) {
        return checker_info<double>{ "cls_checker",
                                     &calculate_classification_error,
                                     required_accuracy };
    }

    checker_info<double> get_mse_checker(double required_accuracy) {
        return checker_info<double>{ "mse_checker", &calculate_mse, required_accuracy };
    }

    checker_info<double> get_mae_checker(double required_accuracy) {
        return checker_info<double>{ "mae_checker", &calculate_mae, required_accuracy };
    }

    static double calculate_classification_error(const dal::table& infer_responses,
                                                 const dal::table& ground_truth) {
        const auto responses = dal::row_accessor<const Float>(infer_responses).pull();
        const auto truth_responses = dal::row_accessor<const Float>(ground_truth).pull();
        std::int64_t incorrect_response_count = 0;

        for (std::int64_t i = 0; i < responses.get_count(); i++) {
            incorrect_response_count +=
                (static_cast<int>(responses[i]) != static_cast<int>(truth_responses[i]));
        }
        return static_cast<double>(incorrect_response_count) / responses.get_count();
    }

    static double calculate_mse(const dal::v1::table& infer_responses,
                                const dal::v1::table& ground_truth) {
        double mean = 0.0;
        const auto responses = dal::row_accessor<const Float>(infer_responses).pull();
        const auto truth_responses = dal::row_accessor<const Float>(ground_truth).pull();
        for (std::int64_t i = 0; i < responses.get_count(); i++) {
            mean += (responses[i] - truth_responses[i]) * (responses[i] - truth_responses[i]);
        }

        return mean / responses.get_count();
    }

    static double calculate_mae(const dal::v1::table& infer_responses,
                                const dal::v1::table& ground_truth) {
        double mae = 0.0;
        const auto responses = dal::row_accessor<const Float>(infer_responses).pull();
        const auto truth_responses = dal::row_accessor<const Float>(ground_truth).pull();

        for (std::int64_t i = 0; i < responses.get_count(); i++) {
            mae += std::abs(responses[i] - truth_responses[i]);
        }

        return mae / responses.get_count();
    }

private:
    double eps = 1e-10;
};

struct dataset_info {
    std::string name;
    std::int64_t class_count = 0;
    std::int64_t categ_feature_count = 0;
    const std::int64_t* categ_feature_list = nullptr;
};

struct workload_cls {
    dataset_info ds_info;
    double required_accuracy = 0.0;
};

struct workload_reg {
    dataset_info ds_info;
    double required_mse = 0.0;
    double required_mae = 0.0;
};

} // namespace oneapi::dal::decision_forest::test
