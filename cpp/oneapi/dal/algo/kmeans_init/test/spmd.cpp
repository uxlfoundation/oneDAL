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

#include "oneapi/dal/algo/kmeans_init/test/fixture.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::kmeans_init::test {

template <typename TestType>
class kmeans_init_spmd_test : public kmeans_init_test<TestType, kmeans_init_spmd_test<TestType>> {
public:
    using base_t = kmeans_init_test<TestType, kmeans_init_spmd_test<TestType>>;
    using desc_t = typename base_t::desc_t;
    using float_t = typename base_t::float_t;
    using task_t = kmeans_init::task::by_default;
    using input_t = kmeans_init::compute_input<task_t>;
    using result_t = kmeans_init::compute_result<task_t>;

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    template <typename... Args>
    auto compute_override(Args&&... args) {
        return this->compute_via_spmd_threads_and_merge(rank_count_, std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<input_t> split_compute_input_override(std::int64_t split_count, Args&&... args) {
        const input_t input{ std::forward<Args>(args)... };

        const auto split_data =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), split_count);

        std::vector<input_t> split_input;
        split_input.reserve(split_count);

        for (std::int64_t i = 0; i < split_count; i++) {
            split_input.push_back(input_t{ split_data[i] });
        }

        return split_input;
    }

    template <typename... Args>
    result_t compute_as_single_node(Args&&... args) {
        return dal::test::engine::float_algo_fixture<float_t>::compute(std::forward<Args>(args)...);
    }

    void check_consistency(std::int64_t cluster_count, const table& data) {
        const auto desc = base_t::get_descriptor(cluster_count);

        const auto multi_node_results = this->compute_override(desc, data);

        const auto single_node_results = this->compute_as_single_node(desc, data);

        const auto multi_table = multi_node_results.get_centroids();
        const auto single_table = single_node_results.get_centroids();

        const bool are_the_same = !are_different(multi_table, single_table);

        CAPTURE(cluster_count, data.get_row_count(), data.get_column_count());
        REQUIRE(are_the_same);
    }

    bool are_different(const table& a, const table& b) const {
        const std::int64_t row_count = a.get_row_count();
        const std::int64_t column_count = b.get_column_count();
        const std::int64_t count = row_count * column_count;

        if (row_count != b.get_row_count())
            return true;
        if (column_count != b.get_column_count())
            return true;

        const auto aarray = row_accessor<const float_t>(a).pull();
        const auto barray = row_accessor<const float_t>(b).pull();

        REQUIRE(aarray.get_count() == count);
        REQUIRE(barray.get_count() == count);

        for (std::int64_t i = 0; i < count; ++i) {
            if (aarray[i] != barray[i])
                return true;
        }

        return false;
    }

    bool are_different(const std::vector<result_t>& results) const {
        for (std::size_t i = 0; i < results.size(); ++i) {
            for (std::size_t j = i + 1; j < results.size(); ++j) {
                const auto& a = results[i].get_centroids();
                const auto& b = results[j].get_centroids();
                if (!are_different(a, b))
                    return false;
            }
        }
        return true;
    }

    result_t merge_compute_result_override(const std::vector<result_t>& results) {
        REQUIRE(!are_different(results));

        return results[0];
    }

private:
    std::int64_t rank_count_ = -1;
};

// TODO: Add tests for other methods
using kmeans_init_types = _TE_COMBINE_TYPES_2((float, double),
                                              (kmeans_init::method::plus_plus_dense));

TEMPLATE_LIST_TEST_M(kmeans_init_spmd_test,
                     "kmeans init dense test",
                     "[kmeans_init][batch]",
                     kmeans_init_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    constexpr std::int64_t row_count = 12;
    constexpr std::int64_t column_count = 3;
    constexpr std::int64_t cluster_count = 3;
    this->set_rank_count(GENERATE(2, 3));

    const double data[] = { 1.0,  1.0,  2.0,  2.0,  1.0, 2.0,  2.0, 1.0,  -1.0, -1.0, -1.0, -2.0,
                            -2.0, -1.0, -2.0, -2.0, 7.0, -7.0, 8.0, -8.0, 9.0,  -9.0, 1.0,  2.0,
                            3.0,  -2.0, 3.0,  0.1,  0.2, 0.3,  0.3, 0.4,  0.5,  7.0,  -1.1, 4.2 };
    const auto data_table = homogen_table::wrap(data, row_count, column_count);

    this->dense_checks(cluster_count, data_table);

    this->check_consistency(cluster_count, data_table);
}

TEMPLATE_LIST_TEST_M(kmeans_init_spmd_test,
                     "kmeans init dense test random",
                     "[kmeans_init][batch][random]",
                     kmeans_init_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t row_count = GENERATE(513, 5007);
    const std::int64_t column_count = GENERATE(3, 7);
    const std::int64_t cluster_count = GENERATE(7, 15);

    this->set_rank_count(GENERATE(2));

    const auto dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ row_count, column_count }.fill_uniform(-10.1, +10.1));

    const auto data_table = dataframe.get_table(te::table_id::homogen<double>());

    this->check_consistency(cluster_count, data_table);
}

} // namespace oneapi::dal::kmeans_init::test
