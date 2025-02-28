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

#include "oneapi/dal/algo/decision_forest/test/fixture.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::decision_forest::test {

namespace df = dal::decision_forest;
namespace te = dal::test::engine;

template <typename TestType>
class df_spmd_test : public df_test<TestType, df_spmd_test<TestType>> {
public:
    using base_t = df_test<TestType, df_spmd_test<TestType>>;
    using Float = typename base_t::Float;
    using Method = typename base_t::Method;
    using Task = typename base_t::Task;
    using float_t = typename base_t::float_t;
    using method_t = typename base_t::method_t;
    using task_t = typename base_t::task_t;
    using descriptor_t = df::descriptor<float_t, method_t, task_t>;
    using train_input_t = typename base_t::train_input_t;
    using train_result_t = typename base_t::train_result_t;

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    template <typename... Args>
    train_result_t train_override(Args&&... args) {
        return this->train_via_spmd_threads_and_merge(rank_count_, std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<train_input_t> split_train_input_override(std::int64_t split_count,
                                                          Args&&... args) {
        // Data table is distributed across the ranks, but
        // initial centroids are common for all the ranks
        const train_input_t input{ std::forward<Args>(args)... };

        const auto split_data =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), split_count);
        const auto split_responses = te::split_table_by_rows<float_t>(this->get_policy(),
                                                                      input.get_responses(),
                                                                      split_count);
        bool is_weighted = input.get_weights().get_row_count() > 0;

        std::vector<table> split_weights;
        if (is_weighted) {
            split_weights = te::split_table_by_rows<float_t>(this->get_policy(),
                                                             input.get_weights(),
                                                             split_count);
        }

        std::vector<train_input_t> split_input;
        split_input.reserve(split_count);

        for (std::int64_t i = 0; i < split_count; i++) {
            if (is_weighted) {
                split_input.push_back( //
                    train_input_t{ split_data[i], split_responses[i], split_weights[i] });
            }
            else {
                split_input.push_back( //
                    train_input_t{ split_data[i], split_responses[i] });
            }
        }

        return split_input;
    }

    //train_result_t merge_train_result_override(const descriptor_t& desc, const std::vector<train_result_t>& results) {
    train_result_t merge_train_result_override(const std::vector<train_result_t>& results) {
        // oob error is the same for all ranks
        // oob error per observation is distributed accross the ranks, need to combine them into one table;
        // variable importance is the same for all ranks
        // Model is the same for all ranks
        std::vector<table> oob_err_per_observation_tables;
        for (const auto& r : results) {
            if (r.get_oob_err_per_observation().has_data()) {
                oob_err_per_observation_tables.push_back(r.get_oob_err_per_observation());
            }
        }
        const auto full_oob_err_per_observation =
            te::stack_tables_by_rows<float_t>(oob_err_per_observation_tables);

        return train_result_t{}
            .set_model(results[0].get_model())
            .set_oob_err(results[0].get_oob_err())
            .set_oob_err_per_observation(full_oob_err_per_observation)
            .set_var_importance(results[0].get_var_importance());
    }

    train_result_t train_spmd_base_checks(const descriptor_t& desc,
                                          const te::dataframe& data,
                                          const te::table_id& data_table_id) {
        const auto x = data.get_table(data_table_id, range(0, -1));
        const auto y = data.get_table(data_table_id,
                                      range(data.get_column_count() - 1, data.get_column_count()));
        INFO("run training");
        const auto train_result = this->train(desc, x, y);

        base_t::check_train_shapes(desc, data, train_result);
        return train_result;
    }

    train_result_t train_spmd_weighted_base_checks(const descriptor_t& desc,
                                                   const te::dataframe& data,
                                                   const te::table_id& data_table_id) {
        const auto x = data.get_table(data_table_id, range(0, -1));
        const auto y =
            data.get_table(data_table_id,
                           range(data.get_column_count() - 2, data.get_column_count() - 1));
        const auto w = data.get_table(data_table_id,
                                      range(data.get_column_count() - 1, data.get_column_count()));
        INFO("run training");
        const auto train_result = this->train(desc, x, y, w);

        base_t::check_train_shapes(desc, data, train_result);
        return train_result;
    }

private:
    std::int64_t rank_count_ = 1;
};

const std::int64_t df_ds_ion_ftrs_list[] = { 0 };
const dataset_info df_ds_ion = { "workloads/ionosphere/dataset/ionosphere",
                                 2 /* class count */,
                                 sizeofa(df_ds_ion_ftrs_list),
                                 df_ds_ion_ftrs_list };
const dataset_info df_ds_segment = { "workloads/segment/dataset/segment", 7 /* class count */ };
const dataset_info df_ds_classification = { "workloads/classification/dataset/df_classification",
                                            2 /* class count */ };
const dataset_info df_ds_white_wine = { "workloads/white_wine/dataset/white_wine" };

using df_cls_types = _TE_COMBINE_TYPES_3((float, double),
                                         (df::method::dense, df::method::hist),
                                         (df::task::classification));
using df_reg_types = _TE_COMBINE_TYPES_3((float, double),
                                         (df::method::dense, df::method::hist),
                                         (df::task::regression));

#define DF_SPMD_CLS_TEST(name) \
    TEMPLATE_LIST_TEST_M(df_spmd_test, name, "[df][integration][spmd]", df_cls_types)
#define DF_SPMD_CLS_TEST_EXT(name)                                    \
    TEMPLATE_LIST_TEST_M(df_spmd_test,                                \
                         name,                                        \
                         "[df][integration][spmd][external-dataset]", \
                         df_cls_types)
#define DF_SPMD_CLS_TEST_NIGHTLY_EXT(name)                                     \
    TEMPLATE_LIST_TEST_M(df_spmd_test,                                         \
                         name,                                                 \
                         "[df][integration][spmd][nightly][external-dataset]", \
                         df_cls_types)

#define DF_SPMD_REG_TEST(name) \
    TEMPLATE_LIST_TEST_M(df_spmd_test, name, "[df][integration][spmd]", df_reg_types)
#define DF_SPMD_REG_TEST_EXT(name)                                    \
    TEMPLATE_LIST_TEST_M(df_spmd_test,                                \
                         name,                                        \
                         "[df][integration][spmd][external-dataset]", \
                         df_reg_types)
#define DF_SPMD_REG_TEST_NIGHTLY_EXT(name)                                     \
    TEMPLATE_LIST_TEST_M(df_spmd_test,                                         \
                         name,                                                 \
                         "[df][integration][spmd][nightly][external-dataset]", \
                         df_reg_types)

DF_SPMD_CLS_TEST_NIGHTLY_EXT("df cls default flow") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_cls wl =
        GENERATE_COPY(workload_cls{ df_ds_ion, 0.95 }, workload_cls{ df_ds_segment, 0.938 });

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const std::int64_t features_per_node_val = GENERATE_COPY(0, 4);
    const std::int64_t max_tree_depth_val = GENERATE_COPY(0, 10);
    const bool memory_saving_mode_val = this->is_gpu() ? false : GENERATE_COPY(true, false);

    const auto error_metric_mode_val = error_metric_mode::out_of_bag_error;
    const auto variable_importance_mode_val = variable_importance_mode::mdi;

    auto desc = this->get_default_descriptor();

    desc.set_memory_saving_mode(memory_saving_mode_val);
    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_features_per_node(features_per_node_val);
    desc.set_max_tree_depth(max_tree_depth_val);
    desc.set_class_count(wl.ds_info.class_count);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_SPMD_CLS_TEST_EXT("df cls corner flow") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_cls wl = { df_ds_classification, 0.95 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(10);
    desc.set_error_metric_mode(error_metric_mode::out_of_bag_error);
    desc.set_min_observations_in_leaf_node(8);
    desc.set_class_count(wl.ds_info.class_count);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_SPMD_CLS_TEST_EXT("df cls small flow") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_cls wl = { df_ds_segment, 0.738 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const std::int64_t tree_count = GENERATE_COPY(1, 2);

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(tree_count);
    desc.set_class_count(wl.ds_info.class_count);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_SPMD_CLS_TEST_NIGHTLY_EXT("df cls impurity flow") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_cls wl = { df_ds_segment, 0.738 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const auto error_metric_mode_val = error_metric_mode::out_of_bag_error;
    const auto variable_importance_mode_val = variable_importance_mode::mdi;
    const double impurity_threshold_val = GENERATE_COPY(0.0, 0.1);
    const std::int64_t min_observations_in_leaf_node = 30;

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(500);
    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_min_observations_in_leaf_node(min_observations_in_leaf_node);
    desc.set_impurity_threshold(impurity_threshold_val);
    desc.set_class_count(wl.ds_info.class_count);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
    this->check_trees_node_min_sample_count(model, min_observations_in_leaf_node);
}

DF_SPMD_CLS_TEST_NIGHTLY_EXT("df cls all features flow") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_cls wl = { df_ds_segment, 0.738 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const auto error_metric_mode_val = error_metric_mode::out_of_bag_error;
    const auto variable_importance_mode_val = variable_importance_mode::mdi;

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(30);
    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_features_per_node(data.get_column_count() - 1); // skip responses column
    desc.set_class_count(wl.ds_info.class_count);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_SPMD_CLS_TEST_NIGHTLY_EXT("df cls bootstrap flow") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_cls wl = { df_ds_ion, 0.95 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const bool bootstrap_val = GENERATE_COPY(false, true);

    auto desc = this->get_default_descriptor();

    desc.set_bootstrap(bootstrap_val);
    desc.set_max_tree_depth(50);
    desc.set_class_count(wl.ds_info.class_count);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_SPMD_CLS_TEST_NIGHTLY_EXT("df cls oob per observation flow") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_cls wl = { df_ds_ion, 0.95 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const auto error_metric_mode_val =
        error_metric_mode::out_of_bag_error | error_metric_mode::out_of_bag_error_per_observation;
    const std::int64_t features_per_node_val = GENERATE_COPY(0, 4);
    const double observations_per_tree_fraction_val = GENERATE_COPY(1.0, 0.5);

    auto desc = this->get_default_descriptor();

    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_features_per_node(features_per_node_val);
    desc.set_max_tree_depth(10);
    desc.set_observations_per_tree_fraction(observations_per_tree_fraction_val);
    desc.set_class_count(wl.ds_info.class_count);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
    this->check_oob_err_matches_oob_err_per_observation(desc,
                                                        train_result,
                                                        1 - wl.required_accuracy);
}

DF_SPMD_CLS_TEST("df cls base check with default params") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const auto [data, data_test, class_count, checker_list] = this->get_cls_dataframe_base();

    auto desc = this->get_default_descriptor();

    desc.set_class_count(class_count);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

// DF_SPMD_CLS_TEST("df cls base check with default params and train weights") {
//     SKIP_IF(this->get_policy().is_cpu());
//     SKIP_IF(this->not_available_on_device());
//     SKIP_IF(this->not_float64_friendly());
//     const auto [data, data_test, class_count, checker_list] =
//         this->get_cls_dataframe_weighted_base();

//     auto desc = this->get_default_descriptor();

//     desc.set_class_count(class_count);

//     this->set_rank_count(2);
//     const auto train_result =
//         this->train_spmd_weighted_base_checks(desc, data, this->get_homogen_table_id());
//     const auto model = train_result.get_model();
//     this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
// }

DF_SPMD_CLS_TEST("df cls base check with non default params") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    const auto [data, data_test, class_count, checker_list] = this->get_cls_dataframe_base();

    const std::int64_t tree_count_val = GENERATE_COPY(10, 50);
    const auto error_metric_mode_val = GENERATE_COPY(
        error_metric_mode::out_of_bag_error,
        error_metric_mode::out_of_bag_error | error_metric_mode::out_of_bag_error_per_observation);
    const auto variable_importance_mode_val =
        GENERATE_COPY(variable_importance_mode::none, variable_importance_mode::mdi);
    const auto infer_mode_val =
        GENERATE_COPY(df::infer_mode::class_responses,
                      df::infer_mode::class_responses | df::infer_mode::class_probabilities);

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(tree_count_val);
    desc.set_min_observations_in_leaf_node(1);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_infer_mode(infer_mode_val);
    desc.set_voting_mode(df::voting_mode::unweighted);
    desc.set_class_count(class_count);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

// regression tests

DF_SPMD_REG_TEST("df reg base check with default params") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const auto [data, data_test, checker_list] = this->get_reg_dataframe_base();

    auto desc = this->get_default_descriptor();

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_SPMD_REG_TEST("df reg base check with default params and train weights") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const auto [data, data_test, checker_list] = this->get_reg_dataframe_weighted_base();

    auto desc = this->get_default_descriptor();

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_weighted_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_SPMD_REG_TEST("df reg base check with non default params") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const auto [data, data_test, checker_list] = this->get_reg_dataframe_base();

    const std::int64_t tree_count_val = GENERATE_COPY(10, 50);
    const auto error_metric_mode_val = GENERATE_COPY(
        error_metric_mode::out_of_bag_error,
        error_metric_mode::out_of_bag_error | error_metric_mode::out_of_bag_error_per_observation);
    const auto variable_importance_mode_val =
        GENERATE_COPY(variable_importance_mode::none, variable_importance_mode::mdi);
    auto desc = this->get_default_descriptor();

    desc.set_tree_count(tree_count_val);
    desc.set_min_observations_in_leaf_node(2);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_error_metric_mode(error_metric_mode_val);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_SPMD_REG_TEST_NIGHTLY_EXT("df reg default flow") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_reg wl = { df_ds_white_wine, 0.45, 0.5 };

    const auto [data, data_test, checker_list] =
        this->get_reg_dataframe(wl.ds_info.name, wl.required_mse, wl.required_mae);

    auto desc = this->get_default_descriptor();

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_SPMD_REG_TEST_EXT("df reg small flow") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_reg wl = { df_ds_white_wine, 0.94, 0.62 };

    const auto [data, data_test, checker_list] =
        this->get_reg_dataframe(wl.ds_info.name, wl.required_mse, wl.required_mae);

    const std::int64_t tree_count = GENERATE_COPY(1, 2);

    auto desc = this->get_default_descriptor();
    desc.set_tree_count(tree_count);
    desc.set_min_observations_in_leaf_node(1);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_SPMD_REG_TEST_NIGHTLY_EXT("df reg impurity flow") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_reg wl = { df_ds_white_wine, 0.94, 0.62 };

    const auto [data, data_test, checker_list] =
        this->get_reg_dataframe(wl.ds_info.name, wl.required_mse, wl.required_mae);

    const double impurity_threshold_val = GENERATE_COPY(0.0, 0.1);
    const std::int64_t min_observations_in_leaf_node = 30;

    auto desc = this->get_default_descriptor();
    desc.set_tree_count(500);
    desc.set_min_observations_in_leaf_node(min_observations_in_leaf_node);
    desc.set_impurity_threshold(impurity_threshold_val);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
    this->check_trees_node_min_sample_count(model, min_observations_in_leaf_node);
}

DF_SPMD_REG_TEST_NIGHTLY_EXT("df reg bootstrap flow") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_reg wl = { df_ds_white_wine, 0.94, 0.62 };

    const auto [data, data_test, checker_list] =
        this->get_reg_dataframe(wl.ds_info.name, wl.required_mse, wl.required_mae);

    const double impurity_threshold_val = GENERATE_COPY(0.0, 0.1);
    const std::int64_t max_tree_depth_val = GENERATE_COPY(0, 50);
    const bool bootstrap_val = GENERATE_COPY(false, true);

    auto desc = this->get_default_descriptor();
    desc.set_impurity_threshold(impurity_threshold_val);
    desc.set_max_tree_depth(max_tree_depth_val);
    desc.set_bootstrap(bootstrap_val);

    this->set_rank_count(2);
    const auto train_result =
        this->train_spmd_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}
} // namespace oneapi::dal::decision_forest::test
