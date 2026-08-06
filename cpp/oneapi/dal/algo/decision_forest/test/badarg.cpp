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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::decision_forest::test {

namespace df = dal::decision_forest;
namespace te = dal::test::engine;

template <typename TestType>
class df_badarg_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using Task = std::tuple_element_t<2, TestType>;

    bool is_gpu() {
        return get_policy().is_gpu();
    }

    bool not_available_on_device() {
        constexpr bool is_dense = std::is_same_v<Method, decision_forest::method::dense>;
        return get_policy().is_gpu() && is_dense;
    }

    auto get_default_descriptor() {
        return df::descriptor<Float, Method, Task>{};
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    auto get_train_data() {
        constexpr std::int64_t row_count_train = 6;
        constexpr std::int64_t column_count = 2;

        static const float x_train[] = { -2.f, -1.f, -1.f, -1.f, -1.f, -2.f,
                                         +1.f, +1.f, +1.f, +2.f, +2.f, +1.f };
        static const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

        const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
        const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

        return std::make_tuple(x_train_table, y_train_table);
    }

    auto get_incorrect_train_data() {
        constexpr std::int64_t row_count_train = 6;
        constexpr std::int64_t row_count_train_less = 5;
        constexpr std::int64_t column_count = 2;

        static const float x_train[] = { -2.f, -1.f, -1.f, -1.f, -1.f, -2.f,
                                         +1.f, +1.f, +1.f, +2.f, +2.f, +1.f };
        static const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

        const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
        const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train_less, 1);

        return std::make_tuple(x_train_table, y_train_table);
    }

    auto get_incorrect_train_data2() {
        constexpr std::int64_t row_count_train = 6;
        constexpr std::int64_t row_count_train_less = 5;
        constexpr std::int64_t column_count = 2;

        static const float x_train[] = { -2.f, -1.f, -1.f, -1.f, -1.f, -2.f,
                                         +1.f, +1.f, +1.f, +2.f, +2.f, +1.f };
        static const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };
        static const float z_train[] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };

        const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
        const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);
        const auto z_train_table = dal::homogen_table::wrap(z_train, row_count_train_less, 1);

        return std::make_tuple(x_train_table, y_train_table, z_train_table);
    }
};

using df_cls_types = _TE_COMBINE_TYPES_3((float),
                                         (df::method::dense, df::method::hist),
                                         (df::task::classification));
using df_common_types = _TE_COMBINE_TYPES_3((float),
                                            (df::method::dense, df::method::hist),
                                            (df::task::classification, df::task::regression));

#define DF_BADARG_CLS_TEST(name) \
    TEMPLATE_LIST_TEST_M(df_badarg_test, name, "[df][badarg]", df_cls_types)

#define DF_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(df_badarg_test, name, "[df][badarg]", df_common_types)

DF_BADARG_CLS_TEST("throws if class_count is le 1") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_class_count(0), domain_error);
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_class_count(-1), domain_error);
}

DF_BADARG_TEST("throws if tree_count is le 0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_tree_count(0), domain_error);
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_tree_count(-1), domain_error);
}

DF_BADARG_TEST("throws if min_observations_in_leaf_node is le 0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_min_observations_in_leaf_node(0),
                      domain_error);
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_min_observations_in_leaf_node(-1),
                      domain_error);
}

DF_BADARG_TEST("throws if min_observations_in_split_node is le 0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_min_observations_in_split_node(0),
                      domain_error);
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_min_observations_in_split_node(-1),
                      domain_error);
}

DF_BADARG_TEST("throws if min_weight_fraction_in_leaf_node is less than 0 or over 0.5") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_min_weight_fraction_in_leaf_node(-0.1),
                      domain_error);
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_min_weight_fraction_in_leaf_node(0.6),
                      domain_error);
}

DF_BADARG_TEST("accepts if min_weight_fraction_in_leaf_node in [0.0, 0.5]") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_default_descriptor().set_min_weight_fraction_in_leaf_node(0.0));
    REQUIRE_NOTHROW(this->get_default_descriptor().set_min_weight_fraction_in_leaf_node(0.5));
}

DF_BADARG_TEST("throws if min_impurity_decrease_in_split_node is less than 0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_min_impurity_decrease_in_split_node(-0.1),
                      domain_error);
}

DF_BADARG_TEST("accepts if min_impurity_decrease_in_split_node eq 0.0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_default_descriptor().set_min_impurity_decrease_in_split_node(0.0));
}

DF_BADARG_TEST("throws if observations_per_tree_fraction is outside of (0.0, 1.0]") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_observations_per_tree_fraction(0.0),
                      domain_error);
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_observations_per_tree_fraction(-0.5),
                      domain_error);
}

DF_BADARG_TEST("accepts if observations_per_tree_fraction is in (0.0, 1.0]") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_default_descriptor().set_observations_per_tree_fraction(1.0));
    REQUIRE_NOTHROW(this->get_default_descriptor().set_observations_per_tree_fraction(0.5));
}

DF_BADARG_TEST("throws if features_per_node is less than 0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_features_per_node(-1), domain_error);
}

DF_BADARG_TEST("accepts if features_per_node is ge 0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_default_descriptor().set_features_per_node(0));
    REQUIRE_NOTHROW(this->get_default_descriptor().set_features_per_node(2));
}

DF_BADARG_TEST("throws if impurity_threshold is less than 0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_impurity_threshold(-0.1), domain_error);
}

DF_BADARG_TEST("accepts if impurity_threshold is ge 0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_default_descriptor().set_impurity_threshold(0.0));
    REQUIRE_NOTHROW(this->get_default_descriptor().set_impurity_threshold(0.5));
}

DF_BADARG_TEST("throws if max_tree_depth is less than 0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_max_tree_depth(-1), domain_error);
}

DF_BADARG_TEST("accepts if max_tree_depth is ge 0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_default_descriptor().set_max_tree_depth(0));
    REQUIRE_NOTHROW(this->get_default_descriptor().set_max_tree_depth(5));
}

DF_BADARG_TEST("throws if max_leaf_nodes is less than 0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_max_leaf_nodes(-1), domain_error);
}

DF_BADARG_TEST("accepts if max_leaf_nodes is ge 0") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_default_descriptor().set_max_leaf_nodes(0));
    REQUIRE_NOTHROW(this->get_default_descriptor().set_max_leaf_nodes(5));
}

DF_BADARG_TEST("throws if max_bins is less than 2") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_max_bins(-1), domain_error);
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_max_bins(1), domain_error);
}

DF_BADARG_TEST("accepts if max_bins is ge 2") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_default_descriptor().set_max_bins(2));
    REQUIRE_NOTHROW(this->get_default_descriptor().set_max_bins(64));
}

DF_BADARG_TEST("throws if min_bin_size is less than 1") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_min_bin_size(-1), domain_error);
    REQUIRE_THROWS_AS(this->get_default_descriptor().set_min_bin_size(0), domain_error);
}

DF_BADARG_TEST("accepts if min_bin_size is ge 1") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_default_descriptor().set_min_bin_size(1));
    REQUIRE_NOTHROW(this->get_default_descriptor().set_min_bin_size(8));
}

DF_BADARG_TEST("throws if train input data is empty") {
    SKIP_IF(this->not_available_on_device());

    dal::homogen_table x_empty;
    const auto [x, y] = this->get_train_data();
    REQUIRE_THROWS_AS(this->train(this->get_default_descriptor(), x_empty, y), domain_error);
}

DF_BADARG_TEST("throws if train input responses is empty") {
    SKIP_IF(this->not_available_on_device());

    dal::homogen_table y_empty;
    const auto [x, y] = this->get_train_data();
    REQUIRE_THROWS_AS(this->train(this->get_default_descriptor(), x, y_empty), domain_error);
}

DF_BADARG_TEST("throws if infer input data is empty") {
    SKIP_IF(this->not_available_on_device());

    dal::homogen_table x_empty;
    const auto [x, y] = this->get_train_data();
    const auto desc = this->get_default_descriptor();
    REQUIRE_THROWS_AS(this->infer(desc, this->train(desc, x, y).get_model(), x_empty),
                      domain_error);
}

DF_BADARG_TEST("throws if bootstrap false and variable importance is mda_raw or mda_scaled") {
    SKIP_IF(this->not_available_on_device());

    const auto [x, y] = this->get_train_data();
    const auto variable_importance_mode_val =
        GENERATE_COPY(variable_importance_mode::mda_raw, variable_importance_mode::mda_scaled);
    auto desc = this->get_default_descriptor();
    desc.set_bootstrap(false);
    desc.set_variable_importance_mode(variable_importance_mode_val);

    REQUIRE_THROWS_AS(this->train(desc, x, y), invalid_argument);
}

DF_BADARG_TEST(
    "throws if bootstrap false and oob_error or oob_error_per_observation was requested") {
    SKIP_IF(this->not_available_on_device());

    const auto [x, y] = this->get_train_data();
    const auto error_metric_mode_val =
        GENERATE_COPY(error_metric_mode::out_of_bag_error,
                      error_metric_mode::out_of_bag_error_per_observation);
    auto desc = this->get_default_descriptor();
    desc.set_bootstrap(false);
    desc.set_error_metric_mode(error_metric_mode_val);

    REQUIRE_THROWS_AS(this->train(desc, x, y), invalid_argument);
}

DF_BADARG_TEST("throws if train input data row count doesn't match responses row count") {
    SKIP_IF(this->not_available_on_device());

    const auto [x, y] = this->get_incorrect_train_data();
    REQUIRE_THROWS_AS(this->train(this->get_default_descriptor(), x, y), invalid_argument);
}

DF_BADARG_TEST("throws if train input data row count doesn't match weights row count") {
    SKIP_IF(this->not_available_on_device());

    const auto [x, y, z] = this->get_incorrect_train_data2();
    REQUIRE_THROWS_AS(this->train(this->get_default_descriptor(), x, y, z), invalid_argument);
}

} // namespace oneapi::dal::decision_forest::test
