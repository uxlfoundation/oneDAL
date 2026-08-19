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

#include "oneapi/dal/algo/decision_forest/test/fixture.hpp"

#include <iostream>

namespace oneapi::dal::decision_forest::test {

template <typename TestType>
class df_batch_test : public df_test<TestType, df_batch_test<TestType>> {};

// dataset configuration
const std::int64_t df_ds_ion_ftrs_list[] = { 0 };
const dataset_info df_ds_ion = { "workloads/ionosphere/dataset/ionosphere",
                                 2 /* class count */,
                                 sizeofa(df_ds_ion_ftrs_list),
                                 df_ds_ion_ftrs_list };
const dataset_info df_ds_segment = { "workloads/segment/dataset/segment", 7 /* class count */ };
const dataset_info df_ds_classification = { "workloads/classification/dataset/df_classification",
                                            2 /* class count */ };
const dataset_info df_ds_pendigits = { "workloads/pendigits/dataset/pendigits",
                                       10 /* class count */ };

const dataset_info df_ds_white_wine = { "workloads/white_wine/dataset/white_wine" };

using df_cls_types = _TE_COMBINE_TYPES_3((float, double),
                                         (df::method::dense, df::method::hist),
                                         (df::task::classification));
using df_reg_types = _TE_COMBINE_TYPES_3((float, double),
                                         (df::method::dense, df::method::hist),
                                         (df::task::regression));

#define DF_BATCH_CLS_TEST(name) \
    TEMPLATE_LIST_TEST_M(df_batch_test, name, "[df][integration][batch]", df_cls_types)
#define DF_BATCH_CLS_TEST_EXT(name)                                    \
    TEMPLATE_LIST_TEST_M(df_batch_test,                                \
                         name,                                         \
                         "[df][integration][batch][external-dataset]", \
                         df_cls_types)
#define DF_BATCH_CLS_TEST_NIGHTLY_EXT(name)                                     \
    TEMPLATE_LIST_TEST_M(df_batch_test,                                         \
                         name,                                                  \
                         "[df][integration][batch][nightly][external-dataset]", \
                         df_cls_types)

#define DF_BATCH_REG_TEST(name) \
    TEMPLATE_LIST_TEST_M(df_batch_test, name, "[df][integration][batch]", df_reg_types)
#define DF_BATCH_REG_TEST_EXT(name)                                    \
    TEMPLATE_LIST_TEST_M(df_batch_test,                                \
                         name,                                         \
                         "[df][integration][batch][external-dataset]", \
                         df_reg_types)
#define DF_BATCH_REG_TEST_NIGHTLY_EXT(name)                                     \
    TEMPLATE_LIST_TEST_M(df_batch_test,                                         \
                         name,                                                  \
                         "[df][integration][batch][nightly][external-dataset]", \
                         df_reg_types)

DF_BATCH_CLS_TEST_NIGHTLY_EXT("df cls default flow") {
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

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST_EXT("df cls corner flow") {
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

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST_NIGHTLY_EXT("df var importance flow") {
    SKIP_IF(this->is_gpu()); // var importance differs on GPU due to difference in built model
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_cls wl = { df_ds_pendigits };
    const double oob_required_accuracy = 0.65;
    const double oob_required_error = 0.00867361;

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".train.csv" });
    const te::dataframe var_imp_test_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ wl.ds_info.name + ".train.var_imp.csv" });

    const double accuracy_threshold = 1 - oob_required_accuracy;

    const auto error_metric_mode_val = GENERATE_COPY(
        error_metric_mode::out_of_bag_error,
        error_metric_mode::out_of_bag_error | error_metric_mode::out_of_bag_error_per_observation);
    const auto variable_importance_mode_val = GENERATE_COPY(variable_importance_mode::none,
                                                            variable_importance_mode::mdi,
                                                            variable_importance_mode::mda_raw,
                                                            variable_importance_mode::mda_scaled);

    auto desc = this->get_default_descriptor();

    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());

    this->check_oob_err_matches_required(desc,
                                         train_result,
                                         oob_required_error,
                                         accuracy_threshold);
    this->check_oob_err_matches_oob_err_per_observation(desc, train_result, accuracy_threshold);
    this->check_var_importance_matches_required(desc,
                                                train_result,
                                                var_imp_test_data,
                                                this->get_homogen_table_id(),
                                                accuracy_threshold);
}

DF_BATCH_CLS_TEST_EXT("df cls small flow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_cls wl = { df_ds_segment, 0.738 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const std::int64_t tree_count = GENERATE_COPY(1, 2);
    const splitter_mode splitter_mode_val =
        GENERATE_COPY(splitter_mode::best, splitter_mode::random);
    const bool bootstrap_val = GENERATE_COPY(true, false);

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(tree_count);
    desc.set_class_count(wl.ds_info.class_count);
    desc.set_splitter_mode(splitter_mode_val);
    desc.set_bootstrap(bootstrap_val);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST_NIGHTLY_EXT("df cls impurity flow") {
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

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
    this->check_trees_node_min_sample_count(model, min_observations_in_leaf_node);
}

DF_BATCH_CLS_TEST_NIGHTLY_EXT("df cls all features flow") {
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

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST_NIGHTLY_EXT("df cls bootstrap flow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_cls wl = { df_ds_ion, 0.95 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const bool bootstrap_val = GENERATE_COPY(false, true);
    const splitter_mode splitter_mode_val =
        GENERATE_COPY(splitter_mode::best, splitter_mode::random);

    auto desc = this->get_default_descriptor();

    desc.set_bootstrap(bootstrap_val);
    desc.set_splitter_mode(splitter_mode_val);
    desc.set_max_tree_depth(50);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST_NIGHTLY_EXT("df cls oob per observation flow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_cls wl = { df_ds_ion, 0.95 };

    const auto [data, data_test, checker_list] =
        this->get_cls_dataframe(wl.ds_info.name, wl.required_accuracy);

    const auto error_metric_mode_val =
        error_metric_mode::out_of_bag_error | error_metric_mode::out_of_bag_error_per_observation;
    const std::int64_t features_per_node_val = GENERATE_COPY(0, 4);
    const double observations_per_tree_fraction_val = GENERATE_COPY(1.0, 0.5, 1.2);

    std::cout << "observation per tree fraction: " << observations_per_tree_fraction_val
              << std::endl;

    auto desc = this->get_default_descriptor();

    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_features_per_node(features_per_node_val);
    desc.set_max_tree_depth(10);
    desc.set_observations_per_tree_fraction(observations_per_tree_fraction_val);
    desc.set_class_count(wl.ds_info.class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
    this->check_oob_err_matches_oob_err_per_observation(desc,
                                                        train_result,
                                                        1 - wl.required_accuracy);
}

DF_BATCH_CLS_TEST("df cls base check with default params") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const auto [data, data_test, class_count, checker_list] = this->get_cls_dataframe_base();

    auto desc = this->get_default_descriptor();

    desc.set_class_count(class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_CLS_TEST("df cls base check with default params and train weights") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_gpu());

    const auto [data, data_test, class_count, checker_list] =
        this->get_cls_dataframe_weighted_base();

    const double observations_per_tree_fraction_val = GENERATE_COPY(1.0, 0.5, 1.2);

    auto desc = this->get_default_descriptor();

    desc.set_class_count(class_count);
    desc.set_observations_per_tree_fraction(observations_per_tree_fraction_val);

    const auto train_result =
        this->train_weighted_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

// `sample_weights[i] / sum(sample_weights)` is the probability of the training row `i`, so an
// integer weight `w` must act as the multiplicity of that row. Training on weighted data must
// therefore match training on data where every row is physically repeated `w` times.
// The weights below sum to 20, well above 1.0.
DF_BATCH_CLS_TEST("df cls train weights equivalent to repeated rows") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_gpu());

    constexpr std::int64_t column_count = 3;
    constexpr std::int64_t class_count = 2;

    // The two classes are drawn from different distributions: class 0 is centered near
    // (-1.2, -0.9) and class 1 near (+1.2, +0.9). The last sample of each class lies inside the
    // other class' cluster and carries a large weight, so the weighted majority of a region
    // differs from its unweighted majority. Training is limited to a single split below, which
    // keeps both clusters in one leaf each and makes the leaf labels depend on the weights.
    static const float train_arr[] = {
        // class 0 cluster
        -1.42f, -0.87f, 0.f, // w = 1
        -0.95f, -1.13f, 0.f, // w = 1
        -1.31f, -0.62f, 0.f, // w = 1
        -0.78f, -1.05f, 0.f, // w = 1
        // class 1 cluster
        1.28f, 1.02f, 1.f, // w = 1
        0.86f, 0.74f, 1.f, // w = 1
        1.41f, 1.19f, 1.f, // w = 1
        0.97f, 0.88f, 1.f, // w = 1
        // class 1 sample inside the class 0 cluster, outweighing it
        -1.10f, -0.95f, 1.f, // w = 10
        // class 0 sample inside the class 1 cluster, outweighing it
        1.20f, 0.91f, 0.f, // w = 10
    };
    const std::vector<std::int64_t> weights = { 1, 1, 1, 1, 1, 1, 1, 1, 10, 10 };

    const te::dataframe weighted_data =
        this->make_weighted_dataframe(train_arr, weights, column_count);
    const te::dataframe expanded_data =
        this->make_expanded_dataframe(train_arr, weights, column_count);
    // The same rows with no weights at all, used below to show the weights change the fit.
    const te::dataframe unweighted_data =
        this->make_expanded_dataframe(train_arr,
                                      std::vector<std::int64_t>(weights.size(), 1),
                                      column_count);

    // One test point in the middle of each cluster.
    static const float test_arr[] = {
        -1.2f, -0.9f, 1.f, //
        1.2f,  0.9f,  0.f, //
    };
    constexpr std::int64_t row_count_test = 2;
    const te::dataframe data_test{ array<float>::wrap(test_arr, row_count_test * column_count),
                                   row_count_test,
                                   column_count };

    // The two inputs have different row counts, so the training must not depend on row sampling
    // or on the random feature subsetting for the resulting models to be comparable. A single
    // split keeps each cluster in one leaf, so its label is the weighted majority of that leaf.
    auto desc = this->get_default_descriptor();
    desc.set_class_count(class_count);
    desc.set_tree_count(1);
    desc.set_bootstrap(false);
    desc.set_splitter_mode(splitter_mode::best);
    desc.set_features_per_node(column_count - 1);
    desc.set_max_tree_depth(1);
    desc.set_min_observations_in_leaf_node(1);
    desc.set_min_bin_size(1);

    const auto table_id = this->get_homogen_table_id();
    const auto x_test = data_test.get_table(table_id, range(0, -1));

    const auto weighted_model =
        this->train_weighted_base_checks(desc, weighted_data, table_id).get_model();
    const auto expanded_model = this->train_base_checks(desc, expanded_data, table_id).get_model();

    const auto weighted_responses = this->infer(desc, weighted_model, x_test).get_responses();
    const auto expanded_responses = this->infer(desc, expanded_model, x_test).get_responses();

    const auto weighted_arr = dal::row_accessor<const float_t>(weighted_responses).pull();
    const auto expanded_arr = dal::row_accessor<const float_t>(expanded_responses).pull();
    for (std::int64_t i = 0; i < row_count_test; ++i) {
        std::cout << "row " << i << " weighted response = " << weighted_arr[i]
                  << ", expanded response = " << expanded_arr[i] << std::endl;
    }
    te::check_if_tables_equal<float_t>(weighted_responses, expanded_responses);

    // Guard against the check above passing trivially: the weights must actually change the fit,
    // otherwise a model that ignores them entirely would satisfy the equivalence.
    const auto unweighted_model =
        this->train_base_checks(desc, unweighted_data, table_id).get_model();
    const auto unweighted_responses = this->infer(desc, unweighted_model, x_test).get_responses();

    const auto unweighted_arr = dal::row_accessor<const float_t>(unweighted_responses).pull();

    for (std::int64_t i = 0; i < row_count_test; ++i) {
        REQUIRE(weighted_arr[i] != unweighted_arr[i]);
    }
}

DF_BATCH_CLS_TEST("df cls base check with non default params") {
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
    desc.set_min_observations_in_leaf_node(2);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_infer_mode(infer_mode_val);
    desc.set_voting_mode(df::voting_mode::unweighted);
    desc.set_class_count(class_count);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

// regression tests

DF_BATCH_REG_TEST("df reg base check with default params") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const auto [data, data_test, checker_list] = this->get_reg_dataframe_base();

    auto desc = this->get_default_descriptor();

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

// Reproduces the bug described in this issue:
// https://github.com/uxlfoundation/oneDAL/issues/3648
DF_BATCH_REG_TEST("df reg random splitter fits training data with large max_bins parameter") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    // Require MSE == 0.0 on the training data.
    [[maybe_unused]] const auto [data, data_test, checker_list] = this->get_reg_dataframe_base(0.0);

    const splitter_mode splitter_mode_val =
        GENERATE_COPY(splitter_mode::best, splitter_mode::random);
    auto desc = this->get_default_descriptor();

    desc.set_tree_count(1);
    desc.set_bootstrap(false);
    desc.set_min_observations_in_leaf_node(1);
    desc.set_min_bin_size(1);
    desc.set_max_bins(data.get_row_count());
    desc.set_splitter_mode(splitter_mode_val);

    INFO("splitter mode = " +
         std::string(splitter_mode_val == splitter_mode::best ? "best" : "random"));

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_REG_TEST("df reg base check with default params and train weights") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_gpu());

    const auto [data, data_test, checker_list] = this->get_reg_dataframe_weighted_base();

    auto desc = this->get_default_descriptor();

    const auto train_result =
        this->train_weighted_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

// Regression counterpart of "df cls train weights equivalent to repeated rows".
// The weights below sum to 20, well above 1.0.
DF_BATCH_REG_TEST("df reg train weights equivalent to repeated rows") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_gpu());

    constexpr std::int64_t column_count = 3;

    // The samples come from two groups with different response distributions: the low-x group
    // responds around 0.2 and the high-x group around 0.8. The last sample of each group is an
    // outlier from the other group's response range and carries a large weight, so the weighted
    // mean of a region differs from its unweighted mean. Training is limited to a single split
    // below, which keeps each group in one leaf and makes the leaf value depend on the weights.
    static const float train_arr[] = {
        // low-x group, responses around 0.2
        0.10f, 0.22f, 0.18f, // w = 1
        0.15f, 0.31f, 0.23f, // w = 1
        0.20f, 0.44f, 0.19f, // w = 1
        0.25f, 0.51f, 0.21f, // w = 1
        // high-x group, responses around 0.8
        0.70f, 1.42f, 0.79f, // w = 1
        0.75f, 1.51f, 0.83f, // w = 1
        0.80f, 1.63f, 0.78f, // w = 1
        0.85f, 1.71f, 0.82f, // w = 1
        // heavily weighted outliers pulling each group's mean toward the other group
        0.18f, 0.39f, 0.90f, // w = 10
        0.78f, 1.58f, 0.10f, // w = 10
    };
    const std::vector<std::int64_t> weights = { 1, 1, 1, 1, 1, 1, 1, 1, 10, 10 };

    const te::dataframe weighted_data =
        this->make_weighted_dataframe(train_arr, weights, column_count);
    const te::dataframe expanded_data =
        this->make_expanded_dataframe(train_arr, weights, column_count);
    // The same rows with no weights at all, used below to show the weights change the fit.
    const te::dataframe unweighted_data =
        this->make_expanded_dataframe(train_arr,
                                      std::vector<std::int64_t>(weights.size(), 1),
                                      column_count);

    // One test point inside each group.
    static const float test_arr[] = {
        0.18f, 0.37f, 0.8f, //
        0.78f, 1.57f, 0.2f, //
    };
    constexpr std::int64_t row_count_test = 2;
    const te::dataframe data_test{ array<float>::wrap(test_arr, row_count_test * column_count),
                                   row_count_test,
                                   column_count };

    // The two inputs have different row counts, so the training must not depend on row sampling
    // or on the random feature subsetting for the resulting models to be comparable. A single
    // split keeps each group in one leaf, so its value is the weighted mean of that leaf.
    auto desc = this->get_default_descriptor();
    desc.set_tree_count(1);
    desc.set_bootstrap(true);
    desc.set_splitter_mode(splitter_mode::best);
    desc.set_features_per_node(column_count - 1); // skip responses column
    desc.set_max_tree_depth(1);
    desc.set_min_observations_in_leaf_node(1);
    desc.set_min_bin_size(1);

    const auto table_id = this->get_homogen_table_id();
    const auto x_test = data_test.get_table(table_id, range(0, -1));

    const auto weighted_model =
        this->train_weighted_base_checks(desc, weighted_data, table_id).get_model();
    const auto expanded_model = this->train_base_checks(desc, expanded_data, table_id).get_model();

    const auto weighted_responses = this->infer(desc, weighted_model, x_test).get_responses();
    const auto expanded_responses = this->infer(desc, expanded_model, x_test).get_responses();

    const auto weighted_arr = dal::row_accessor<const float_t>(weighted_responses).pull();
    const auto expanded_arr = dal::row_accessor<const float_t>(expanded_responses).pull();
    for (std::int64_t i = 0; i < row_count_test; ++i) {
        CAPTURE("row " + std::to_string(i) + " weighted response = " +
             std::to_string(weighted_arr[i]) +
             ", expanded response = " +
             std::to_string(expanded_arr[i]));
    }
    const double tolerance = te::get_tolerance<float_t>(1e-4, 1e-10);
    te::check_if_tables_equal_approx<float_t>(weighted_responses, expanded_responses, tolerance);

    // Guard against the check above passing trivially: the weights must actually change the fit,
    // otherwise a model that ignores them entirely would satisfy the equivalence.
    const auto unweighted_model =
        this->train_base_checks(desc, unweighted_data, table_id).get_model();
    const auto unweighted_responses = this->infer(desc, unweighted_model, x_test).get_responses();

    const auto unweighted_arr = dal::row_accessor<const float_t>(unweighted_responses).pull();

    for (std::int64_t i = 0; i < row_count_test; ++i) {
        REQUIRE(std::abs(double(weighted_arr[i]) - double(unweighted_arr[i])) > tolerance);
    }
}

DF_BATCH_REG_TEST("df reg base check with non default params") {
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

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_REG_TEST_NIGHTLY_EXT("df reg default flow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_reg wl = { df_ds_white_wine, 0.45, 0.5 };

    const auto [data, data_test, checker_list] =
        this->get_reg_dataframe(wl.ds_info.name, wl.required_mse, wl.required_mae);

    const splitter_mode splitter_mode_val =
        GENERATE_COPY(splitter_mode::best, splitter_mode::random);
    const bool bootstrap_val = GENERATE_COPY(true, false);

    auto desc = this->get_default_descriptor();
    desc.set_splitter_mode(splitter_mode_val);
    desc.set_bootstrap(bootstrap_val);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_REG_TEST_EXT("df reg small flow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_reg wl = { df_ds_white_wine, 0.94, 0.62 };

    const auto [data, data_test, checker_list] =
        this->get_reg_dataframe(wl.ds_info.name, wl.required_mse, wl.required_mae);

    const std::int64_t tree_count = GENERATE_COPY(1, 2);

    auto desc = this->get_default_descriptor();
    desc.set_tree_count(tree_count);
    desc.set_min_observations_in_leaf_node(1);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

DF_BATCH_REG_TEST_NIGHTLY_EXT("df reg impurity flow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_reg wl = { df_ds_white_wine, 0.94, 0.62 };

    const auto [data, data_test, checker_list] =
        this->get_reg_dataframe(wl.ds_info.name, wl.required_mse, wl.required_mae);

    const double impurity_threshold_val = GENERATE_COPY(0.0, 0.1);
    const std::int64_t min_observations_in_leaf_node = 30;
    const splitter_mode splitter_mode_val =
        GENERATE_COPY(splitter_mode::best, splitter_mode::random);
    const bool bootstrap_val = GENERATE_COPY(true, false);

    auto desc = this->get_default_descriptor();
    desc.set_tree_count(500);
    desc.set_min_observations_in_leaf_node(min_observations_in_leaf_node);
    desc.set_impurity_threshold(impurity_threshold_val);
    desc.set_splitter_mode(splitter_mode_val);
    desc.set_bootstrap(bootstrap_val);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
    this->check_trees_node_min_sample_count(model, min_observations_in_leaf_node);
}

DF_BATCH_REG_TEST_NIGHTLY_EXT("df reg bootstrap flow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const workload_reg wl = { df_ds_white_wine, 0.94, 0.62 };

    const auto [data, data_test, checker_list] =
        this->get_reg_dataframe(wl.ds_info.name, wl.required_mse, wl.required_mae);

    const double impurity_threshold_val = GENERATE_COPY(0.0, 0.1);
    const std::int64_t max_tree_depth_val = GENERATE_COPY(0, 50);
    const bool bootstrap_val = GENERATE_COPY(false, true);
    const splitter_mode splitter_mode_val =
        GENERATE_COPY(splitter_mode::best, splitter_mode::random);

    auto desc = this->get_default_descriptor();
    desc.set_impurity_threshold(impurity_threshold_val);
    desc.set_max_tree_depth(max_tree_depth_val);
    desc.set_bootstrap(bootstrap_val);
    desc.set_splitter_mode(splitter_mode_val);

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();
    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

} // namespace oneapi::dal::decision_forest::test
