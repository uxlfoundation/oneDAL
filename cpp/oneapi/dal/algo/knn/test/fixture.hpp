/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "oneapi/dal/algo/knn/train.hpp"
#include "oneapi/dal/algo/knn/infer.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/table/detail/table_builder.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

#include "oneapi/dal/test/engine/metrics/classification.hpp"
#include "oneapi/dal/test/engine/metrics/regression.hpp"

namespace oneapi::dal::knn::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace de = dal::detail;
namespace knn = oneapi::dal::knn;

template <typename TestType, typename Derived>
class knn_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using base_t = te::crtp_algo_fixture<TestType, Derived>;
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using task_t = std::tuple_element_t<2, TestType>;
    using train_result_t = train_result<task_t>;
    using train_input_t = train_input<task_t>;
    using infer_result_t = infer_result<task_t>;
    using infer_input_t = infer_input<task_t>;

    using default_distance_t = oneapi::dal::minkowski_distance::descriptor<>;

    using voting_t = oneapi::dal::knn::voting_mode;
    constexpr static inline voting_t default_voting = voting_t::uniform;

    template <typename D = default_distance_t, typename T = knn::task::by_default>
    auto get_descriptor(std::int64_t override_class_count,
                        std::int64_t override_neighbor_count,
                        D distance = D{},
                        voting_t voting = default_voting,
                        T task = T{}) const {
        auto desc =
            knn::descriptor<float_t, method_t, T, D>(override_class_count, override_neighbor_count)
                .set_result_options(knn::result_options::responses | knn::result_options::indices |
                                    knn::result_options::distances)
                .set_voting_mode(voting);

        if constexpr (is_brute_force) {
            desc.set_distance(distance);
        }

        return desc;
    }

    static constexpr bool is_kd_tree = std::is_same_v<method_t, knn::method::kd_tree>;
    static constexpr bool is_brute_force = std::is_same_v<method_t, knn::method::brute_force>;

    bool not_available_on_device() {
        return (this->get_policy().is_gpu() && is_kd_tree);
    }

    template <typename Distance = default_distance_t>
    float_t classification(const table& train_data,
                           const table& train_responses,
                           const table& infer_data,
                           const table& infer_responses,
                           const std::int64_t n_classes,
                           const std::int64_t n_neighbors,
                           const Distance distance = Distance{},
                           const voting_t voting = default_voting,
                           const float_t tolerance = float_t(1.e-5)) {
        INFO("check if data shape is expected");
        REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
        REQUIRE(train_responses.get_column_count() == 1);
        REQUIRE(infer_responses.get_column_count() == 1);
        REQUIRE(infer_data.get_row_count() == infer_responses.get_row_count());
        REQUIRE(train_data.get_row_count() == train_responses.get_row_count());

        const auto knn_desc = this->get_descriptor(n_classes, n_neighbors, distance, voting);

        auto train_result = this->train(knn_desc, train_data, train_responses);
        auto train_model = train_result.get_model();
        auto infer_result = this->infer(knn_desc, infer_data, train_model);
        auto [prediction] = this->unpack_result(infer_result);

        const auto score_table =
            te::accuracy_score<float_t>(infer_responses, prediction, tolerance);
        const auto score = row_accessor<const float_t>(score_table).pull({ 0, -1 })[0];
        return score;
    }

    template <typename Distance = default_distance_t>
    float_t regression(const table& train_data,
                       const table& train_responses,
                       const table& infer_data,
                       const table& infer_responses,
                       const std::int64_t n_neighbors,
                       const Distance distance = Distance{},
                       const voting_t voting = default_voting) {
        INFO("check if data shape is expected");
        REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
        REQUIRE(train_responses.get_column_count() == 1);
        REQUIRE(infer_responses.get_column_count() == 1);
        REQUIRE(infer_data.get_row_count() == infer_responses.get_row_count());
        REQUIRE(train_data.get_row_count() == train_responses.get_row_count());

        constexpr knn::task::regression task{};

        const auto knn_desc = this->get_descriptor(42, n_neighbors, distance, voting, task);

        auto train_result = this->train(knn_desc, train_data, train_responses);
        auto train_model = train_result.get_model();
        auto infer_result = this->infer(knn_desc, infer_data, train_model);
        auto [prediction] = this->unpack_result(infer_result);

        const auto score_table = te::mse_score<float_t>(infer_responses, prediction);
        const auto score = row_accessor<const float_t>(score_table).pull({ 0, -1 })[0];
        return score;
    }

    template <typename Task>
    void exact_nearest_indices_check(const table& train_data,
                                     const table& infer_data,
                                     const knn::infer_result<Task>& result) {
        check_nans(result);

        const auto [responses] = unpack_result(result);

        const auto gtruth = naive_knn_search(train_data, infer_data);

        INFO("check if data shape is expected");
        REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
        REQUIRE(infer_data.get_row_count() == responses.get_row_count());
        REQUIRE(responses.get_column_count() == 1);
        REQUIRE(infer_data.get_row_count() == gtruth.get_row_count());
        REQUIRE(train_data.get_row_count() == gtruth.get_column_count());

        const auto m = infer_data.get_row_count();

        const auto indices = naive_knn_search(train_data, infer_data);

        for (std::int64_t j = 0; j < m; ++j) {
            const auto gt_indices_row = row_accessor<const float_t>(indices).pull({ j, j + 1 });
            const auto te_indices_row = row_accessor<const float_t>(responses).pull({ j, j + 1 });
            const auto l = gt_indices_row[0];
            const auto r = te_indices_row[0];
            if (l != r) {
                CAPTURE(j, l, r);
                FAIL("Indices of nearest neighbors are unequal");
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

        auto distances_arr = array<float_t>::zeros(m * n);
        auto* distances_ptr = distances_arr.get_mutable_data();

        for (std::int64_t j = 0; j < n; ++j) {
            const auto queue_row = row_accessor<const float_t>(infer_data).pull({ j, j + 1 });
            for (std::int64_t i = 0; i < m; ++i) {
                const auto train_row = row_accessor<const float_t>(train_data).pull({ i, i + 1 });
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

    template <typename Task>
    void check_nans(const knn::infer_result<Task>& result) {
        const auto [responses] = unpack_result(result);

        INFO("check if there is no NaN in responses");
        REQUIRE(te::has_no_nans(responses));
    }

    template <typename Task>
    static auto unpack_result(const knn::infer_result<Task>& result) {
        const auto responses = result.get_responses();
        return std::make_tuple(responses);
    }
};

using knn_cls_types = COMBINE_TYPES((float, double),
                                    (knn::method::brute_force, knn::method::kd_tree),
                                    (knn::task::classification));
using knn_cls_bf_types = COMBINE_TYPES((float, double),
                                       (knn::method::brute_force),
                                       (knn::task::classification));
using knn_reg_types = COMBINE_TYPES((float, double),
                                    (knn::method::brute_force, knn::method::kd_tree),
                                    (knn::task::regression));
using knn_reg_bf_types = COMBINE_TYPES((float, double),
                                       (knn::method::brute_force),
                                       (knn::task::regression));
using knn_kdtree_cls_float_only = COMBINE_TYPES((float),
                                                (knn::method::kd_tree),
                                                (knn::task::classification));

} // namespace oneapi::dal::knn::test
