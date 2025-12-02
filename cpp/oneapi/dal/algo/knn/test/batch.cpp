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

#include "oneapi/dal/algo/knn/test/fixture.hpp"

namespace oneapi::dal::knn::test {

template <typename TestType>
class knn_batch_test : public knn_test<TestType, knn_batch_test<TestType>> {};

#define KNN_SMALL_TEST(name)                                               \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                   \
                         name,                                             \
                         "[small-dataset][knn][integration][batch][test]", \
                         knn_cls_types)

#define KNN_CLS_SYNTHETIC_TEST(name)                                           \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                       \
                         name,                                                 \
                         "[synthetic-dataset][knn][integration][batch][test]", \
                         knn_cls_types)

#define KNN_CLS_EXTERNAL_TEST(name)                                           \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                      \
                         name,                                                \
                         "[external-dataset][knn][integration][batch][test]", \
                         knn_cls_types)

#define KNN_CLS_BF_EXTERNAL_TEST(name)                                        \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                      \
                         name,                                                \
                         "[external-dataset][knn][integration][batch][test]", \
                         knn_cls_bf_types)

#define KNN_REG_SYNTHETIC_TEST(name)                                           \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                       \
                         name,                                                 \
                         "[synthetic-dataset][knn][integration][batch][test]", \
                         knn_reg_types)

#define KNN_REG_EXTERNAL_TEST(name)                                           \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                      \
                         name,                                                \
                         "[external-dataset][knn][integration][batch][test]", \
                         knn_reg_types)

#define KNN_REG_BF_EXTERNAL_TEST(name)                                        \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                      \
                         name,                                                \
                         "[external-dataset][knn][integration][batch][test]", \
                         knn_reg_bf_types)

#define KNN_DIST_VOTING_TRAIN_IN_TEST(name)                                \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                   \
                         name,                                             \
                         "[small-dataset][knn][integration][batch][test]", \
                         knn_cls_types)

#define KNN_KDTREE_SINGLE_RUN_TEST(name)                                       \
    TEMPLATE_LIST_TEST_M(knn_batch_test,                                       \
                         name,                                                 \
                         "[synthetic-dataset][knn][integration][batch][test][nightly]", \
                         knn_kdtree_cls_float_only)

KNN_SMALL_TEST("knn nearest points test predefined 7x5x2") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    constexpr std::int64_t train_row_count = 7;
    constexpr std::int64_t infer_row_count = 5;
    constexpr std::int64_t column_count = 2;

    CAPTURE(train_row_count, infer_row_count, column_count);

    constexpr std::int64_t train_element_count = train_row_count * column_count;
    constexpr std::int64_t infer_element_count = infer_row_count * column_count;

    constexpr std::array<float, train_element_count> train = { -2.f, -1.f, -1.f,   -1.f,   -1.f,
                                                               -2.f, +1.f, +1.f,   +1.f,   +2.f,
                                                               +2.f, +1.f, +100.f, -1024.f };

    constexpr std::array<float, infer_element_count> infer = { +2.f, +1.f, -1.f, +3.f, -1.f,
                                                               -1.f, +1.f, +2.f, +1.f, +2.f };

    const auto x_train_table = homogen_table::wrap(train.data(), train_row_count, column_count);
    const auto x_infer_table = homogen_table::wrap(infer.data(), infer_row_count, column_count);
    const auto y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_CLS_SYNTHETIC_TEST("knn nearest points test random uniform 513x301x17") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_kd_tree);

    constexpr std::int64_t train_row_count = 513;
    constexpr std::int64_t infer_row_count = 301;
    constexpr std::int64_t column_count = 17;

    CAPTURE(train_row_count, infer_row_count, column_count);

    const auto train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ train_row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_train_table = train_dataframe.get_table(this->get_homogen_table_id());
    const auto infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ infer_row_count, column_count }.fill_uniform(-0.3, 1.));
    const table x_infer_table = infer_dataframe.get_table(this->get_homogen_table_id());

    const table y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_REG_SYNTHETIC_TEST("knn nearest points test random uniform using regression 513x301x17") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    constexpr std::int64_t train_row_count = 513;
    constexpr std::int64_t infer_row_count = 301;
    constexpr std::int64_t column_count = 17;

    CAPTURE(train_row_count, infer_row_count, column_count);

    const auto train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ train_row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_train_table = train_dataframe.get_table(this->get_homogen_table_id());
    const auto infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ infer_row_count, column_count }.fill_uniform(-0.3, 1.));
    const table x_infer_table = infer_dataframe.get_table(this->get_homogen_table_id());

    const table y_train_table = this->arange(train_row_count);

    using distance_t = oneapi::dal::minkowski_distance::descriptor<>;
    constexpr double minkowski_degree = 2.0;
    const auto distance_desc = distance_t(minkowski_degree);
    constexpr auto voting = oneapi::dal::knn::voting_mode::uniform;
    constexpr knn::task::regression task{};

    const auto knn_desc = this->get_descriptor(train_row_count, 1, distance_desc, voting, task);

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_CLS_SYNTHETIC_TEST("knn nearest points test random uniform 16390x20x5") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_kd_tree);

    constexpr std::int64_t train_row_count = 16390;
    constexpr std::int64_t infer_row_count = 20;
    constexpr std::int64_t column_count = 5;

    CAPTURE(train_row_count, infer_row_count, column_count);

    const auto train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ train_row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_train_table = train_dataframe.get_table(this->get_homogen_table_id());
    const auto infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ infer_row_count, column_count }.fill_uniform(-0.3, 1.));
    const table x_infer_table = infer_dataframe.get_table(this->get_homogen_table_id());

    const table y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_KDTREE_SINGLE_RUN_TEST("knn nearest points test random uniform 1'000'000x20x10") {
    // this test triggers issue with KD-Tree training on large datasets
    // that leads to crash due to use-after-free in multithreaded execution
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    constexpr std::int64_t train_row_count = 1'000'000;
    constexpr std::int64_t infer_row_count = 20;
    constexpr std::int64_t column_count = 10;

    CAPTURE(train_row_count, infer_row_count, column_count);

    const auto train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ train_row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_train_table = train_dataframe.get_table(this->get_homogen_table_id());
    const auto infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ infer_row_count, column_count }.fill_uniform(-0.3, 1.));
    const table x_infer_table = infer_dataframe.get_table(this->get_homogen_table_id());

    const table y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_CLS_EXTERNAL_TEST("knn classification hepmass 50kx10k") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    constexpr double target_score = 0.8;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    const auto score = this->classification(x_train_table,
                                            y_train_table,
                                            x_infer_table,
                                            y_infer_table,
                                            n_classes,
                                            n_neighbors);
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_REG_EXTERNAL_TEST("knn distance regression hepmass 50kx10k") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    constexpr double target_score = 0.072;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    using distance_t = oneapi::dal::minkowski_distance::descriptor<>;
    constexpr double minkowski_degree = 2.0;
    const auto distance_desc = distance_t(minkowski_degree);
    constexpr auto voting = oneapi::dal::knn::voting_mode::distance;

    const auto score = this->regression(x_train_table,
                                        y_train_table,
                                        x_infer_table,
                                        y_infer_table,
                                        n_neighbors,
                                        distance_desc,
                                        voting);

    CAPTURE(score, target_score);
    REQUIRE(score < target_score);
}

KNN_REG_EXTERNAL_TEST("knn uniform regression hepmass 50kx10k") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    constexpr double target_score = 0.072;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    const auto score =
        this->regression(x_train_table, y_train_table, x_infer_table, y_infer_table, n_neighbors);

    CAPTURE(score, target_score);
    REQUIRE(score < target_score);
}

KNN_CLS_BF_EXTERNAL_TEST("knn classification hepmass 50kx10k with distance voting)") {
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    // TODO: Investigate low accuracy on CPU
    const double target_score = this->get_policy().is_gpu() ? 0.8 : 0.6;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    using distance_t = oneapi::dal::minkowski_distance::descriptor<>;
    constexpr double minkowski_degree = 2.0;
    const auto distance_desc = distance_t(minkowski_degree);
    constexpr auto voting = oneapi::dal::knn::voting_mode::distance;
    const auto score = this->classification(x_train_table,
                                            y_train_table,
                                            x_infer_table,
                                            y_infer_table,
                                            n_classes,
                                            n_neighbors,
                                            distance_desc,
                                            voting);
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_CLS_BF_EXTERNAL_TEST("knn classification hepmass 50kx10k with Minkowski distance (p = 2.5)") {
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    constexpr double target_score = 0.8;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    using distance_t = oneapi::dal::minkowski_distance::descriptor<>;
    const double minkowski_degree = 2.5;
    const auto distance_desc = distance_t(minkowski_degree);
    const auto score = this->classification(x_train_table,
                                            y_train_table,
                                            x_infer_table,
                                            y_infer_table,
                                            n_classes,
                                            n_neighbors,
                                            distance_desc);
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_CLS_BF_EXTERNAL_TEST("knn classification hepmass 50kx10k with Cosine distance") {
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    constexpr double target_score = 0.78;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    using distance_t = oneapi::dal::cosine_distance::descriptor<>;
    const auto distance_desc = distance_t();
    const auto score = this->classification(x_train_table,
                                            y_train_table,
                                            x_infer_table,
                                            y_infer_table,
                                            n_classes,
                                            n_neighbors,
                                            distance_desc);
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_CLS_BF_EXTERNAL_TEST("knn classification hepmass 50kx10k with Chebyshev distance") {
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->not_available_on_device());

    constexpr double target_score = 0.8;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    using distance_t = oneapi::dal::chebyshev_distance::descriptor<>;
    const auto score = this->classification(x_train_table,
                                            y_train_table,
                                            x_infer_table,
                                            y_infer_table,
                                            n_classes,
                                            n_neighbors,
                                            distance_t{});
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_DIST_VOTING_TRAIN_IN_TEST("knn classification predefined data with distance voting") {
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    constexpr std::int64_t feature_count = 5;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;
    constexpr std::int64_t train_row_count = 7;
    constexpr std::int64_t infer_row_count = 3;
    constexpr double target_score = 1.0;
    CAPTURE(feature_count, n_classes, n_neighbors, train_row_count, infer_row_count);

    constexpr std::int64_t train_element_count = train_row_count * feature_count;
    constexpr std::int64_t infer_element_count = infer_row_count * feature_count;

    constexpr std::array<float, train_element_count> train = {
        0.451, 0.666, 0.154, 0.496, 0.521, 0.446, 0.844, 0.3,   0.682, 0.695, 0.64,  0.835,
        0.355, 0.707, 0.673, 0.201, 0.704, 0.145, 0.341, 0.486, 0.276, 0.423, 0.927, 0.402,
        0.256, 0.738, 0.384, 0.232, 0.325, 0.996, 0.644, 0.936, 0.296, 0.54,  0.719
    };

    constexpr std::array<float, infer_element_count> infer = { 0.451, 0.666, 0.154, 0.496, 0.521,
                                                               0.086, 0.17,  0.721, 0.383, 0.028,
                                                               0.216, 0.255, 0.672, 0.303, 0.466 };

    constexpr std::array<float, train_row_count> train_label = { 0, 1, 1, 1, 1, 1, 1 };
    constexpr std::array<float, infer_row_count> infer_label = { 0, 1, 1 };

    const auto x_train_table = homogen_table::wrap(train.data(), train_row_count, feature_count);
    const auto x_infer_table = homogen_table::wrap(infer.data(), infer_row_count, feature_count);
    const auto y_train_table = homogen_table::wrap(train_label.data(), train_row_count, 1);
    const auto y_infer_table = homogen_table::wrap(infer_label.data(), infer_row_count, 1);

    using distance_t = oneapi::dal::minkowski_distance::descriptor<>;
    constexpr double minkowski_degree = 2.0;
    const auto distance_desc = distance_t(minkowski_degree);
    constexpr auto voting = oneapi::dal::knn::voting_mode::distance;
    const auto score = this->classification(x_train_table,
                                            y_train_table,
                                            x_infer_table,
                                            y_infer_table,
                                            n_classes,
                                            n_neighbors,
                                            distance_desc,
                                            voting);
    CAPTURE(score, target_score);
    REQUIRE(score == target_score);
}
} // namespace oneapi::dal::knn::test
