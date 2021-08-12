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

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/search.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/table/detail/table_builder.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

template <typename Float>
class search_test : public te::float_algo_fixture<Float> {
    using idx_t = ndview<std::int32_t, 2>;

public:
    void generate() {
        m_ = GENERATE(1, 15, 17, 32, 1025);
        n_ = GENERATE(1, 15, 17, 32, 1026);
        k_ = GENERATE(1, 15, 32, 96, 1027);
        d_ = GENERATE(2, 16, 28, 41, 1029);
        generate_data();
    }

    void generate_data() {
        const auto train_df = GENERATE_DATAFRAME(
            te::dataframe_builder{ m_, d_ }.fill_uniform(-0.2, 0.5));
        this->train_ = train_df.get_table(this->get_homogen_table_id());
        const auto query_df = GENERATE_DATAFRAME(
            te::dataframe_builder{ n_, d_ }.fill_uniform(-0.5, 1.0));
        this->query_ = query_df.get_table(this->get_homogen_table_id());
    }

    auto get_train_view() {
        const auto acc = row_accessor<const Float>(train_).pull({ 0, n_ });
        return ndview<Float, 2>::wrap(acc.get_data(), {m_, d_});
    }

    auto get_query_view() {
        const auto acc = row_accessor<const Float>(query_).pull({ 0, n_ });
        return ndview<Float, 2>::wrap(acc.get_data(), {m_, d_});
    }

    void exact_nearest_indices_check(const table& train_data,
                                     const table& infer_data,
                                     const idx_t& result_arr) {

        const auto gtruth = naive_knn_search(train_data, infer_data);

        INFO("check if data shape is expected");
        REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
        REQUIRE(train_data.get_row_count() == gtruth.get_column_count());
        REQUIRE(infer_data.get_row_count() == gtruth.get_row_count());

        const auto indices = naive_knn_search(train_data, infer_data);
        const auto ind_arr = idx_t::wrap(indices.get_data(), {n_, m_});

        for(std::int64_t j = 0; j < n_; ++j) {
            for(std::int64_t i = 0; i < k_; ++i) {
                const auto gtr_val = ind_arr.at(j, i);
                const auto res_val = result_arr.at(j, i);
                CAPTURE(i, j, m_, n_, k_, d_, gtr_val, res_val);
                REQUIRE(gtr_val == res_val);
            }
        }
    }

    static auto naive_knn_search(const table& train_data, const table& infer_data) {
        const auto distances_matrix = distances(train_data, infer_data);
        const auto indices_matrix = argsort(distances_matrix);

        return indices_matrix;
    }

    static auto distances(const table& train_data, const table& infer_data) {
        const auto m = train_data.get_row_count();
        const auto n = infer_data.get_row_count();
        const auto d = infer_data.get_column_count();

        auto distances_arr = array<Float>::zeros(m * n);
        auto* distances_ptr = distances_arr.get_mutable_data();

        for (std::int64_t j = 0; j < n; ++j) {
            const auto queue_row = row_accessor<const Float>(infer_data).pull({ j, j + 1 });
            for (std::int64_t i = 0; i < m; ++i) {
                const auto train_row = row_accessor<const Float>(train_data).pull({ i, i + 1 });
                for (std::int64_t s = 0; s < d; ++s) {
                    const auto diff = queue_row[s] - train_row[s];
                    distances_ptr[j * m + i] += diff * diff;
                }
            }
        }
        return de::homogen_table_builder{}.reset(distances_arr, n, m).build();
    }

    static auto argsort(const table& distances) {
        const auto n = distances.get_row_count();
        const auto m = distances.get_column_count();

        auto indices = array<std::int32_t>::zeros(m * n);
        auto indices_ptr = indices.get_mutable_data();
        for (std::int64_t j = 0; j < n; ++j) {
            const auto dist_row = row_accessor<const Float>(distances).pull({ j, j + 1 });
            auto idcs_row = &indices_ptr[j * m];
            std::iota(idcs_row, idcs_row + m, std::int32_t(0));
            const auto compare = [&](std::int32_t x, std::int32_t y) -> bool {
                return dist_row[x] < dist_row[y];
            };
            std::sort(idcs_row, idcs_row + m, compare);
        }
        return de::homogen_table_builder{}.reset(indices, n, m).build();
    }

    static auto arange(std::int64_t from, std::int64_t to) {
        auto indices_arr = array<std::int32_t>::zeros(to - from);
        auto* indices_ptr = indices_arr.get_mutable_data();
        std::iota(indices_ptr, indices_ptr + to - from, std::int32_t(from));
        return de::homogen_table_builder{}.reset(indices_arr, to - from, 1).build();
    }

    static auto arange(std::int64_t to) {
        return arange(0, to);
    }

private:
    table train_, query_;
    std::int64_t m_, n_, k_, d_;
};

using search_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(search_test,
                     "Randomly filled L2-distance search",
                     "[l2][search][small]",
                     search_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    //this->test_l1_distance();
}

} // namespace oneapi::dal::backend::primitives::test
