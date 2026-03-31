/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/algo/pca/test/fixture.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::pca::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace pca = oneapi::dal::pca;

template <typename TestType>
class pca_spmd_test : public pca_test<TestType, pca_spmd_test<TestType>> {
public:
    using base_t = pca_test<TestType, pca_spmd_test<TestType>>;
    using float_t = typename base_t::float_t;
    using input_t = typename base_t::input_t;
    using result_t = typename base_t::result_t;

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    template <typename... Args>
    result_t train_override(Args&&... args) {
        return this->train_via_spmd_threads_and_merge(rank_count_, std::forward<Args>(args)...);
    }

    result_t merge_train_result_override(const std::vector<result_t>& results) {
        return results[0];
    }

    template <typename... Args>
    std::vector<input_t> split_train_input_override(std::int64_t split_count, Args&&... args) {
        const input_t input{ std::forward<Args>(args)... };

        const auto split_data =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), split_count);

        std::vector<input_t> split_input;
        split_input.reserve(split_count);

        for (std::int64_t i = 0; i < split_count; i++) {
            split_input.push_back( //
                input_t{ split_data[i] });
        }

        return split_input;
    }

    void spmd_general_checks(const te::dataframe& data_fr, const te::table_id& data_table_id) {
        const table data = data_fr.get_table(this->get_policy(), data_table_id);
        const std::int64_t component_count = 0;
        const bool deterministic = true;
        const auto pca_desc = base_t::get_descriptor(component_count, deterministic);
        const auto train_result = this->train(pca_desc, data);
        INFO("run training");
        base_t::check_train_result(pca_desc, data_fr, train_result);
    }

private:
    std::int64_t rank_count_;
};

using pca_types = COMBINE_TYPES((float, double), (pca::method::cov));

TEMPLATE_LIST_TEST_M(pca_spmd_test, "pca common flow", "[pca][integration][spmd]", pca_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 10, 10 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 20, 20 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 1000, 100 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 2000, 20 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 2500, 20 }.fill_normal(-30, 30, 7777));
    this->set_rank_count(GENERATE(2, 4));

    const auto data_table_id = this->get_homogen_table_id();

    this->spmd_general_checks(data, data_table_id);
}

} // namespace oneapi::dal::pca::test
