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

class c_order {};
class f_order {};

template <typename order>
struct order_map {};

template <>
struct order_map<c_order> {
    constexpr static auto value = ndorder::c;
};

template <>
struct order_map<f_order> {
    constexpr static auto value = ndorder::f;
};

template <typename order>
constexpr auto order_v = order_map<order>::value;

template <typename T, ndorder order>
auto table_to_ndarray(sycl::queue& q, const table& t) {
    const auto rc = t.get_row_count();
    const auto cc = t.get_column_count();
    auto res = ndarray<T, 2, order>::empty(q, { rc, cc }, sycl::usm::alloc::host);
    for (std::int64_t r = 0; r < rc; ++r) {
        const auto row = row_accessor<const T>(t).pull(q, { r, r + 1 });
        for (std::int64_t c = 0; c < cc; ++c) {
            res.at(r, c) = row[c];
        }
    }
    return res;
}

template <typename TestType>
class search_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
    using float_t = std::tuple_element_t<0, TestType>;
    using torder_t = std::tuple_element_t<1, TestType>;
    using qorder_t = std::tuple_element_t<2, TestType>;

    using idx_t = ndview<std::int32_t, 2>;
    using dst_t = ndview<float_t, 2>;

    static constexpr auto torder = order_v<torder_t>;
    static constexpr auto qorder = order_v<qorder_t>;
    using distance_t = chebyshev_distance<float_t>;
    using search_t = search_engine<float_t, distance_t, torder>;

public:
    void generate() {
        m_ = GENERATE(2, 11, 17, 32, 127);
        n_ = GENERATE(3, 10, 17, 32, 127);
        k_ = GENERATE(1, 16, 32, 64, 127);
        d_ = GENERATE(2, 28, 41, 77, 133);
        generate_data();
    }

    void generate_data() {
        const auto train_df =
            GENERATE_DATAFRAME(te::dataframe_builder{ m_, d_ }.fill_uniform(-0.2, 0.5));
        this->train_ = train_df.get_table(this->get_policy(), this->get_homogen_table_id());
        const auto query_df =
            GENERATE_DATAFRAME(te::dataframe_builder{ n_, d_ }.fill_uniform(-0.5, 1.0));
        this->query_ = query_df.get_table(this->get_policy(), this->get_homogen_table_id());
    }

    bool is_initialized() const {
        return m_ > 0 && n_ > 0 && k_ > 0 && d_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    auto get_train_view() {
        auto& queue = this->get_queue();
        return table_to_ndarray<float_t, torder>(queue, this->train_);
    }

    auto get_query_view() {
        auto& queue = this->get_queue();
        return table_to_ndarray<float_t, qorder>(queue, this->query_);
    }

    auto get_temp_indices() {
        return ndarray<std::int32_t, 2>::empty(this->get_queue(), { n_, k_ });
    }

    auto get_temp_distances() {
        return ndarray<float_t, 2>::empty(this->get_queue(), { n_, k_ });
    }

    void exact_nearest_indices_check(const table& train_data,
                                     const table& infer_data,
                                     const idx_t& result_ids,
                                     const dst_t& result_dst,
                                     const float_t threshold) {
        const auto gtruth = naive_knn_search(train_data, infer_data);

        INFO("check if data shape is expected");
        REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
        REQUIRE(train_data.get_row_count() == gtruth.get_column_count());
        REQUIRE(infer_data.get_row_count() == gtruth.get_row_count());

        auto ind_arr = row_accessor<const std::int32_t>(gtruth).pull({ 0, n_ });
        const auto ind_ndarr = idx_t::wrap(ind_arr.get_data(), { n_, m_ });

        auto dst_table = distances(train_data, infer_data);
        auto dst_arr = row_accessor<const float_t>(dst_table).pull({ 0, n_ });
        const auto dst_ndarr = dst_t::wrap(dst_arr.get_data(), { n_, m_ });

        for (std::int64_t j = 0; j < n_; ++j) {
            for (std::int64_t i = 0; i < k_; ++i) {
                const auto gtr_val = ind_ndarr.at(j, i);
                const auto gtr_dst = dst_ndarr.at(j, gtr_val);
                const auto res_val = result_ids.at(j, i);
                const auto res_dst = result_dst.at(j, i);
                const bool close_dst = std::abs((gtr_dst / res_dst) - 1) < threshold;
                CAPTURE(i, j, m_, n_, k_, d_, gtr_val, gtr_dst, res_val, res_dst, close_dst);
                const bool is_valid = close_dst || (gtr_val == res_val);
                REQUIRE(is_valid);
            }
        }
    }

    void test_correctness(const float_t threshold = 1e-5) {
        check_if_initialized();
        auto& queue = this->get_queue();
        if (m_ > k_) {
            const auto train = get_train_view();
            const auto query = get_query_view();

            auto indices = get_temp_indices();
            auto distances = get_temp_distances();

            constexpr std::int64_t qblock = 32;
            constexpr std::int64_t tblock = 64;

            const search_t engine(queue, train, tblock);
            copy_callback<float_t, true, true> callbk(queue, qblock, indices, distances);

            engine(query, callbk, qblock, k_).wait_and_throw();

            exact_nearest_indices_check(train_, query_, indices, distances, threshold);
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

        auto distances_arr = array<float_t>::zeros(m * n);
        auto* distances_ptr = distances_arr.get_mutable_data();

        for (std::int64_t j = 0; j < n; ++j) {
            const auto query_row = row_accessor<const float_t>(infer_data).pull({ j, j + 1 });
            for (std::int64_t i = 0; i < m; ++i) {
                const auto train_row = row_accessor<const float_t>(train_data).pull({ i, i + 1 });

                float_t gtv = 0.0;
                for (std::int64_t k = 0; k < d; ++k) {
                    const auto q = train_row[k];
                    const auto t = query_row[k];
                    const auto d = std::abs(q - t);
                    gtv = std::max(gtv, d);
                }

                distances_ptr[j * m + i] = gtv;
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
            const auto dist_row = row_accessor<const float_t>(distances).pull({ j, j + 1 });
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

using search_types = COMBINE_TYPES((float, double), (c_order, f_order), (c_order, f_order));

TEMPLATE_LIST_TEST_M(search_test,
                     "Randomly filled cosine-distance search",
                     "[chebyshev][search][small]",
                     search_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_correctness();
}

} // namespace oneapi::dal::backend::primitives::test
