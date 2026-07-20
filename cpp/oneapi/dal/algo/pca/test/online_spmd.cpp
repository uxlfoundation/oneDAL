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

#include "oneapi/dal/algo/pca/test/fixture.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::pca::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace pca = oneapi::dal::pca;

template <typename TestType>
class pca_online_spmd_test : public pca_test<TestType, pca_online_spmd_test<TestType>> {
public:
    using base_t = pca_test<TestType, pca_online_spmd_test<TestType>>;
    using float_t = typename base_t::float_t;
    using input_t = typename base_t::input_t;
    using partial_input_t = typename base_t::partial_input_t;
    using partial_result_t = typename base_t::partial_result_t;
    using result_t = typename base_t::result_t;

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    void set_blocks_count(std::int64_t blocks_count) {
        blocks_count_ = blocks_count;
    }

    template <typename... Args>
    result_t finalize_train_override(Args&&... args) {
        return this->finalize_train_via_spmd_threads_and_merge(rank_count_,
                                                               std::forward<Args>(args)...);
    }

    result_t merge_finalize_train_result_override(const std::vector<result_t>& results) {
        return results[0];
    }

    template <typename... Args>
    std::vector<partial_result_t> split_finalize_train_input_override(std::int64_t split_count,
                                                                      Args&&... args) {
        ONEDAL_ASSERT(split_count == rank_count_);
        const std::vector<partial_result_t> input{ std::forward<Args>(args)... };

        return input;
    }

    void online_spmd_general_checks(const te::dataframe& data_fr,
                                    std::int64_t component_count,
                                    const te::table_id& data_table_id) {
        const table data = data_fr.get_table(this->get_policy(), data_table_id);

        const auto pca_desc = base_t::get_descriptor(component_count);
        std::vector<partial_result_t> partial_results;
        auto input_table = te::split_table_by_rows<float_t>(this->get_policy(), data, rank_count_);
        for (int64_t i = 0; i < rank_count_; i++) {
            dal::pca::partial_train_result<> partial_result;
            auto input_table_blocks =
                te::split_table_by_rows<float_t>(this->get_policy(), input_table[i], blocks_count_);
            for (int64_t j = 0; j < blocks_count_; j++) {
                partial_result =
                    this->partial_train(pca_desc, partial_result, input_table_blocks[j]);
            }
            partial_results.push_back(partial_result);
        }
        auto train_result = this->finalize_train_override(pca_desc, partial_results);
        base_t::check_train_result(pca_desc, data_fr, train_result);

        train_result = this->finalize_train_override(pca_desc, partial_results);
        base_t::check_train_result(pca_desc, data_fr, train_result);
    }

private:
    std::int64_t rank_count_;
    std::int64_t blocks_count_;
};

using pca_types = COMBINE_TYPES((float, double), (pca::method::cov));

TEMPLATE_LIST_TEST_M(pca_online_spmd_test,
                     "pca common flow",
                     "[pca][integration][spmd]",
                     pca_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(GENERATE(1, 2, 4));
    this->set_blocks_count(GENERATE(1, 3, 10));

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_uniform(0.2, 0.5),
                           te::dataframe_builder{ 100000, 10 }.fill_uniform(-0.2, 1.5));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto data_table_id = this->get_homogen_table_id();

    const std::int64_t component_count = GENERATE_COPY(0, 1, data.get_column_count());

    this->online_spmd_general_checks(data, component_count, data_table_id);
}

} // namespace oneapi::dal::pca::test
