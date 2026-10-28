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

#include <cmath>
#include <list>
#include <vector>

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

    std::int64_t count_model_nodes(const df::model<Task>& model) {
        std::int64_t node_count = 0;
        model_traverse_check(model, [&](const node_info<Task>& node) {
            ++node_count;
            return true;
        });
        return node_count;
    }

    /// Trains a bootstrapped forest of `tree_count` trees with `engine` and returns the leaf sizes
    /// of each tree in traversal order, which identify its bootstrap sample and splits.
    ///
    /// @param[in] engine     The engine type of the forest.
    /// @param[in] tree_count The number of trees.
    std::vector<std::vector<std::int64_t>> train_leaf_sizes(engine_type engine,
                                                            std::int64_t tree_count) {
        constexpr std::int64_t row_count = 256;

        auto x_arr = array<float>::empty(row_count);
        auto y_arr = array<float>::empty(row_count);
        float* x_ptr = x_arr.get_mutable_data();
        float* y_ptr = y_arr.get_mutable_data();
        for (std::int64_t i = 0; i < row_count; ++i) {
            x_ptr[i] = static_cast<float>(i);
            y_ptr[i] = static_cast<float>(i % 2);
        }
        const auto x = dal::homogen_table::wrap(x_ptr, row_count, 1);
        const auto y = dal::homogen_table::wrap(y_ptr, row_count, 1);

        auto desc = this->get_default_descriptor();
        if constexpr (std::is_same_v<Task, decision_forest::task::classification>) {
            desc.set_class_count(2);
        }
        desc.set_tree_count(tree_count);
        desc.set_features_per_node(1);
        desc.set_bootstrap(true);
        desc.set_engine_type(engine);
        const auto model = this->train(desc, x, y).get_model();

        struct leaf_size_collector {
            std::vector<std::int64_t>* out;
            bool operator()(const leaf_node_info<Task>& info) {
                out->push_back(info.get_sample_count());
                return true;
            }
            bool operator()(const split_node_info<Task>&) {
                return true;
            }
        };

        std::vector<std::vector<std::int64_t>> trees(tree_count);
        for (std::int64_t t = 0; t < tree_count; ++t) {
            model.traverse_depth_first(t, leaf_size_collector{ &trees[t] });
        }
        return trees;
    }

    /// Checks that the trees of a bootstrapped forest are not all the same, which they are when
    /// every tree draws from the same random stream.
    ///
    /// @param[in] engine The engine type of the forest.
    void check_trees_draw_different_streams(engine_type engine) {
        const auto trees = train_leaf_sizes(engine, 4);
        bool all_same = true;
        for (std::size_t t = 1; t < trees.size(); ++t) {
            all_same = all_same && (trees[t] == trees[0]);
        }
        REQUIRE(!all_same);
    }

    /// Checks that two engines with the same seed train different forests, so the descriptor's
    /// engine type reaches the training kernel.
    ///
    /// @param[in] first  One engine type.
    /// @param[in] second Another engine type.
    void check_engine_type_changes_forest(engine_type first, engine_type second) {
        REQUIRE(train_leaf_sizes(first, 4) != train_leaf_sizes(second, 4));
    }

    /// Trains one tree with the random splitter on `row_count` rows whose only feature is the row
    /// index, and returns the leaves from left to right. Leaf `k` then holds the rows right after
    /// those of leaf `k - 1`, which lets the caller compute what every leaf must contain.
    ///
    /// @param[in] row_count           The number of training rows.
    /// @param[in] column_count        The number of identical feature columns.
    /// @param[in] min_weight_fraction The minimum weight fraction of a leaf, or 0 for none.
    /// @param[in] max_tree_depth      The maximum depth of the tree.
    /// @param[in] min_leaf_rows       The minimum number of rows in a leaf.
    /// @param[in] response_of         The response of row `i`.
    /// @param[in] weight_of           The weight of row `i`, or `nullptr` to train unweighted.
    /// @param[in] check_leaf          Called with each leaf, its first row and its row count.
    template <typename LeafCheck>
    void check_random_split_leaves(std::int64_t row_count,
                                   std::int64_t column_count,
                                   double min_weight_fraction,
                                   std::int64_t max_tree_depth,
                                   std::int64_t min_leaf_rows,
                                   float (*response_of)(std::int64_t),
                                   float (*weight_of)(std::int64_t),
                                   LeafCheck&& check_leaf) {
        constexpr bool is_cls = std::is_same_v<Task, decision_forest::task::classification>;

        auto x_arr = array<float>::empty(row_count * column_count);
        auto y_arr = array<float>::empty(row_count);
        auto w_arr = array<float>::empty(row_count);
        float* x_ptr = x_arr.get_mutable_data();
        float* y_ptr = y_arr.get_mutable_data();
        float* w_ptr = w_arr.get_mutable_data();
        for (std::int64_t i = 0; i < row_count; ++i) {
            for (std::int64_t j = 0; j < column_count; ++j) {
                x_ptr[i * column_count + j] = static_cast<float>(i);
            }
            y_ptr[i] = response_of(i);
            w_ptr[i] = weight_of ? weight_of(i) : 1.0f;
        }
        const auto x = dal::homogen_table::wrap(x_ptr, row_count, column_count);
        const auto y = dal::homogen_table::wrap(y_ptr, row_count, 1);

        auto desc = this->get_default_descriptor();
        if constexpr (is_cls) {
            desc.set_class_count(cls_class_count);
        }
        desc.set_tree_count(1);
        desc.set_features_per_node(column_count);
        desc.set_bootstrap(false);
        desc.set_max_tree_depth(max_tree_depth);
        desc.set_min_observations_in_leaf_node(min_leaf_rows);
        desc.set_splitter_mode(splitter_mode::random);

        struct leaf_collector {
            std::vector<leaf_node_info<Task>>* out;
            bool operator()(const leaf_node_info<Task>& info) {
                out->push_back(info);
                return true;
            }
            bool operator()(const split_node_info<Task>&) {
                return true;
            }
        };

        if (min_weight_fraction > 0.0) {
            desc.set_min_weight_fraction_in_leaf_node(min_weight_fraction);
        }
        model_t model;
        if (weight_of) {
            const auto w = dal::homogen_table::wrap(w_ptr, row_count, 1);
            model = this->train(desc, x, y, w).get_model();
        }
        else {
            model = this->train(desc, x, y).get_model();
        }

        std::vector<leaf_node_info<Task>> leaves;
        model.traverse_depth_first(0, leaf_collector{ &leaves });
        REQUIRE(leaves.size() >= 2);

        std::int64_t start = 0;
        for (const auto& leaf : leaves) {
            const std::int64_t count = leaf.get_sample_count();
            CAPTURE(row_count, leaves.size(), start, count);
            REQUIRE(count > 0);
            check_leaf(leaf, start, count);
            start += count;
        }
        REQUIRE(start == row_count);
    }

    static constexpr std::int64_t cls_class_count = 3;

    /// The class of row `i` in the random splitter checks: runs of three rows per class.
    static std::int64_t cls_label(std::int64_t i) {
        return (i / 3) % cls_class_count;
    }

    /// The response of row `i` in the random splitter checks: its class or its row index.
    static float row_response(std::int64_t i) {
        constexpr bool is_cls = std::is_same_v<Task, decision_forest::task::classification>;
        return is_cls ? static_cast<float>(cls_label(i)) : static_cast<float>(i);
    }

    static float unit_weight(std::int64_t) {
        return 1.0f;
    }

    /// Checks that every leaf of a random split describes exactly the rows it holds, including
    /// for nodes with more rows than the GPU splitter has work items in a work group.
    ///
    /// @param[in] row_count The number of training rows.
    void check_random_split_covers_all_rows(std::int64_t row_count) {
        // A shallow tree keeps the split nodes far larger than a work group.
        check_random_split_leaves(
            row_count,
            1,
            0.0,
            2,
            1,
            row_response,
            nullptr,
            [](const leaf_node_info<Task>& leaf, std::int64_t start, std::int64_t count) {
                if constexpr (std::is_same_v<Task, decision_forest::task::classification>) {
                    for (std::int64_t c = 0; c < cls_class_count; ++c) {
                        std::int64_t class_rows = 0;
                        for (std::int64_t i = start; i < start + count; ++i) {
                            class_rows += (cls_label(i) == c);
                        }
                        CAPTURE(c, class_rows);
                        REQUIRE(std::fabs(leaf.get_probability(c) -
                                          double(class_rows) / double(count)) < 1e-6);
                    }
                }
                else {
                    // The response of a leaf is the mean of its rows' row indices.
                    const double expected = 0.5 * double(2 * start + count - 1);
                    CAPTURE(leaf.get_response(), expected);
                    REQUIRE(std::fabs(leaf.get_response() - expected) < 1.0);
                }
            });
    }

    /// Checks that the weight of every row of a node counts towards the minimum leaf weight,
    /// including for nodes with more rows than the GPU splitter has work items in a work group.
    ///
    /// @param[in] row_count The number of training rows, each of unit weight.
    void check_random_split_weights_cover_all_rows(std::int64_t row_count) {
        // Every leaf needs a quarter of the rows, more than any work group has work items, so a
        // left part that misses rows can never be accepted. Several identical columns give each
        // node more than one random threshold to try.
        constexpr double min_weight_fraction = 0.25;
        const auto min_rows = static_cast<std::int64_t>(std::ceil(min_weight_fraction * row_count));
        check_random_split_leaves(
            row_count,
            8,
            min_weight_fraction,
            4,
            1,
            row_response,
            unit_weight,
            [=](const leaf_node_info<Task>&, std::int64_t, std::int64_t count) {
                REQUIRE(count >= min_rows);
            });
    }

    /// The response of row `i` in the leaf statistics check. Row `i` is drawn from one of 16
    /// distributions chosen by `i % 16`: uniform, exponential, Bernoulli and normal families with
    /// different means and scales. 16 divides the work-group size, so every GPU work item sees a
    /// single distribution, and the per-item means and variances all differ.
    static float mixed_family_response(std::int64_t i) {
        const auto uniform01 = [](std::uint64_t x) {
            // splitmix64
            std::uint64_t z = (x + 1) * 0x9E3779B97F4A7C15ull;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z ^= z >> 31;
            return double(z >> 11) * 0x1.0p-53;
        };
        const std::int64_t k = i % 16;
        const double u = uniform01(2 * i);
        const double v = uniform01(2 * i + 1);
        double x = 0.0;
        switch (k % 4) {
            case 0: x = u; break;
            case 1: x = -std::log1p(-u); break;
            case 2: x = (u < 0.2) ? 1.0 : 0.0; break;
            default: x = std::sqrt(-2.0 * std::log1p(-u)) * std::cos(6.283185307179586 * v); break;
        }
        return static_cast<float>(100.0 + 25.0 * double(k) + 4.0 * double(1 + 3 * k) * x);
    }

    /// Checks that the mean and the variance of every regression leaf match a sequential
    /// double-precision pass over its rows, including for nodes with more rows than the GPU
    /// splitter has work items in a work group. Leaves hold at least an eighth of the rows, so
    /// the tolerances stay far below the error of an inexact combination of the per-item
    /// statistics: weighing every item equally misses the variance by about 5% on this data and
    /// dropping the within-item term by more than 10%.
    ///
    /// @param[in] row_count The number of training rows.
    /// @param[in] weighted  Whether to train with non-uniform weights.
    void check_random_split_leaf_statistics(std::int64_t row_count, bool weighted) {
        const auto weight_of = [](std::int64_t i) {
            return static_cast<float>(1 + i % 5);
        };
        // A leaf's variance can come from subtracting a sibling's from the parent's, which loses
        // a few ulp of the node's sum of squared deviations.
        double total_sum2cent = 0.0;
        {
            double mean = 0.0;
            for (std::int64_t i = 0; i < row_count; ++i) {
                const double delta = mixed_family_response(i) - mean;
                mean += delta / double(i + 1);
                total_sum2cent += delta * (mixed_family_response(i) - mean);
            }
        }
        const double eps = std::is_same_v<Float, float> ? 1.2e-7 : 2.3e-16;
        const double rel_tol = std::is_same_v<Float, float> ? 1e-4 : 1e-10;
        check_random_split_leaves(
            row_count,
            8,
            0.0,
            3,
            row_count / 8,
            mixed_family_response,
            weighted ? +weight_of : nullptr,
            [&](const leaf_node_info<Task>& leaf, std::int64_t start, std::int64_t count) {
                double mean = 0.0;
                double sum2cent = 0.0;
                for (std::int64_t i = start; i < start + count; ++i) {
                    const double y = mixed_family_response(i);
                    const double delta = y - mean;
                    mean += delta / double(i - start + 1);
                    sum2cent += delta * (y - mean);
                }
                const double variance = sum2cent / double(count);
                CAPTURE(leaf.get_response(), mean, leaf.get_impurity(), variance);
                REQUIRE(std::fabs(leaf.get_response() - mean) < rel_tol * std::sqrt(variance));
                REQUIRE(std::fabs(leaf.get_impurity() - variance) <
                        rel_tol * variance + 10 * eps * total_sum2cent / double(count));
            });
    }

    // Trains a single fully-grown tree with and without a min_weight_fraction
    // constraint and checks that the constrained model has significantly fewer
    // nodes and that every leaf respects the effective minimum sample count.
    // Unit weights are used so the weighted path is exercised while the effect
    // mimics a plain sample-count limit.
    void check_min_weight_fraction_reduces_node_count(splitter_mode splitter_mode_val) {
        constexpr bool is_cls = std::is_same_v<Task, decision_forest::task::classification>;
        constexpr std::int64_t row_count = 256;
        constexpr std::int64_t class_count = 2;
        constexpr double min_weight_fraction = 0.1;

        static float x_arr[row_count];
        static float y_arr[row_count];
        static float w_arr[row_count];
        for (std::int64_t i = 0; i < row_count; ++i) {
            x_arr[i] = static_cast<float>(i);
            // Both patterns force many beneficial splits so a fully-grown tree
            // is deep: staircase labels for classification, distinct monotonic
            // responses for regression.
            y_arr[i] = is_cls ? static_cast<float>((i / 8) % class_count) : static_cast<float>(i);
            w_arr[i] = 1.0f;
        }
        const auto x = dal::homogen_table::wrap(x_arr, row_count, 1);
        const auto y = dal::homogen_table::wrap(y_arr, row_count, 1);
        const auto w = dal::homogen_table::wrap(w_arr, row_count, 1);

        INFO("splitter mode = " +
             std::string(splitter_mode_val == splitter_mode::best ? "best" : "random"));

        const auto make_desc = [&](double fraction) {
            auto desc = this->get_default_descriptor();
            if constexpr (is_cls) {
                desc.set_class_count(class_count);
            }
            desc.set_tree_count(1);
            desc.set_features_per_node(1);
            desc.set_bootstrap(false);
            desc.set_min_observations_in_leaf_node(1);
            desc.set_max_bins(row_count);
            desc.set_min_bin_size(1);
            desc.set_splitter_mode(splitter_mode_val);
            desc.set_min_weight_fraction_in_leaf_node(fraction);
            return desc;
        };

        const auto baseline_model = this->train(make_desc(0.001), x, y, w).get_model();
        const auto constrained_model =
            this->train(make_desc(min_weight_fraction), x, y, w).get_model();

        const std::int64_t baseline_nodes = count_model_nodes(baseline_model);
        const std::int64_t constrained_nodes = count_model_nodes(constrained_model);

        CAPTURE(baseline_nodes, constrained_nodes);
        INFO("constrained tree must have fewer nodes than the unconstrained one");
        REQUIRE(constrained_nodes < baseline_nodes);

        INFO("every leaf must hold at least ceil(fraction * n_samples) samples");
        const std::int64_t min_samples =
            static_cast<std::int64_t>(std::ceil(min_weight_fraction * row_count));
        check_trees_node_min_sample_count(constrained_model, min_samples);
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
