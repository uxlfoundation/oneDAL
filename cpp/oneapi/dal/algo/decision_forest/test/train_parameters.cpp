/*******************************************************************************
* Copyright contributors to the oneDAL project
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

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

template <typename TestType>
class df_train_params_test : public df_test<TestType, df_train_params_test<TestType>> {
public:
    using base_t = df_test<TestType, df_train_params_test<TestType>>;

    using task_t = typename base_t::task_t;
    using float_t = typename base_t::float_t;
    using method_t = typename base_t::method_t;
    using train_input_t = typename base_t::train_input_t;
    using train_result_t = typename base_t::train_result_t;
    using model_t = typename base_t::model_t;

    void generate_parameters() {
        if constexpr (std::is_same_v<task_t, task::classification>) {
            this->small_classes_threshold_ = GENERATE(4, 8);
        }
        this->min_part_coefficient_ = GENERATE(4, 6);
        this->min_size_coefficient_ = GENERATE(24000, 32000);
        this->pack_as_struct_ = GENERATE(0, 1);
    }

    auto get_current_parameters() const {
        detail::train_parameters<task_t> res{};
        if constexpr (std::is_same_v<task_t, task::classification>) {
            res.set_small_classes_threshold(this->small_classes_threshold_);
        }
        res.set_min_part_coefficient(this->min_part_coefficient_);
        res.set_min_size_coefficient(this->min_size_coefficient_);

        return res;
    }

    template <typename Desc, typename... Args>
    train_result_t train_override(Desc&& desc, Args&&... args) {
        if constexpr (std::is_same_v<task_t, task::classification>) {
            REQUIRE(0 < this->small_classes_threshold_);
            REQUIRE(this->small_classes_threshold_ <= 8);
        }
        REQUIRE(0 < this->min_part_coefficient_);
        REQUIRE(0 < this->min_size_coefficient_);
        const auto params = this->get_current_parameters();
        if (this->pack_as_struct_) {
            return te::float_algo_fixture<float_t>::train(
                std::forward<Desc>(desc),
                params,
                train_input_t{ std::forward<Args>(args)... });
        }
        else {
            return te::float_algo_fixture<float_t>::train(std::forward<Desc>(desc),
                                                          params,
                                                          std::forward<Args>(args)...);
        }
    }

private:
    std::int64_t small_classes_threshold_;
    std::int64_t min_part_coefficient_;
    std::int64_t min_size_coefficient_;
    bool pack_as_struct_;
};

using df_cls_types = _TE_COMBINE_TYPES_3((float, double),
                                         (df::method::hist),
                                         (df::task::classification));

TEMPLATE_LIST_TEST_M(df_train_params_test,
                     "DF classification train params",
                     "[df][cls][train][params]",
                     df_cls_types) {
    SKIP_IF(this->is_gpu());
    SKIP_IF(this->not_available_on_device());

    const auto [data, data_test, class_count, checker_list] = this->get_cls_dataframe_base();

    const std::int64_t tree_count_val = GENERATE_COPY(10, 50);
    const bool memory_saving_mode_val = GENERATE_COPY(true, false);

    const auto error_metric_mode_val = error_metric_mode::out_of_bag_error;
    const auto variable_importance_mode_val = variable_importance_mode::mdi;

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(tree_count_val);
    desc.set_min_observations_in_leaf_node(2);
    desc.set_memory_saving_mode(memory_saving_mode_val);
    desc.set_error_metric_mode(error_metric_mode_val);
    desc.set_variable_importance_mode(variable_importance_mode_val);
    desc.set_voting_mode(df::voting_mode::unweighted);
    desc.set_class_count(class_count);

    this->generate_parameters();

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();

    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

using df_reg_types = _TE_COMBINE_TYPES_3((float, double),
                                         (df::method::hist),
                                         (df::task::regression));

TEMPLATE_LIST_TEST_M(df_train_params_test,
                     "DF regression train params",
                     "[df][reg][train][params]",
                     df_reg_types) {
    SKIP_IF(this->is_gpu());
    SKIP_IF(this->not_available_on_device());

    const auto [data, data_test, checker_list] = this->get_reg_dataframe_base();

    const std::int64_t tree_count_val = GENERATE_COPY(10, 50);

    auto desc = this->get_default_descriptor();

    desc.set_tree_count(tree_count_val);

    this->generate_parameters();

    const auto train_result = this->train_base_checks(desc, data, this->get_homogen_table_id());
    const auto model = train_result.get_model();

    this->infer_base_checks(desc, data_test, this->get_homogen_table_id(), model, checker_list);
}

} // namespace oneapi::dal::decision_forest::test
