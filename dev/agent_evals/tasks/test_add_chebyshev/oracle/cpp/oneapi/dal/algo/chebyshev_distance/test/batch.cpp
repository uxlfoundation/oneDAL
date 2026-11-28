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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "oneapi/dal/algo/chebyshev_distance/common.hpp"
#include "oneapi/dal/algo/knn/infer.hpp"
#include "oneapi/dal/algo/knn/train.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::chebyshev_distance::test {

namespace te = dal::test::engine;

template <typename Float>
class chebyshev_distance_knn_test : public te::float_algo_fixture<Float> {
public:
    using distance_t = chebyshev_distance::descriptor<Float>;
    using knn_desc_t =
        knn::descriptor<Float, knn::method::brute_force, knn::task::search, distance_t>;

    static std::vector<Float> random_matrix(std::int64_t rows, std::int64_t cols, int seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(-5.0, 5.0);
        std::vector<Float> data(rows * cols);
        for (auto& v : data) {
            v = Float(dist(gen));
        }
        return data;
    }

    static double reference_distance(const Float* x, const Float* y, std::int64_t d) {
        double res = 0.0;
        for (std::int64_t k = 0; k < d; ++k) {
            res = std::max(res, std::abs(double(x[k]) - double(y[k])));
        }
        return res;
    }

    // Runs brute-force kNN search with the Chebyshev metric and checks returned neighbors and
    // distances against a naive reference computed in double precision.
    void check_search(const std::vector<Float>& train,
                      const std::vector<Float>& query,
                      std::int64_t train_count,
                      std::int64_t query_count,
                      std::int64_t d,
                      std::int64_t k) {
        const auto x_train = homogen_table::wrap(train.data(), train_count, d);
        const auto x_query = homogen_table::wrap(query.data(), query_count, d);

        const auto desc =
            knn_desc_t(k, distance_t{})
                .set_result_options(knn::result_options::indices | knn::result_options::distances);

        const auto train_result = this->train(desc, x_train);
        const auto infer_result = this->infer(desc, x_query, train_result.get_model());

        const auto indices = infer_result.get_indices();
        const auto distances = infer_result.get_distances();
        REQUIRE(indices.get_row_count() == query_count);
        REQUIRE(indices.get_column_count() == k);
        REQUIRE(distances.get_row_count() == query_count);
        REQUIRE(distances.get_column_count() == k);

        const auto idx_arr = row_accessor<const double>(indices).pull();
        const auto dst_arr = row_accessor<const double>(distances).pull();
        const double tol = std::is_same_v<Float, float> ? 1e-5 : 1e-10;

        for (std::int64_t i = 0; i < query_count; ++i) {
            std::vector<double> ref(train_count);
            for (std::int64_t j = 0; j < train_count; ++j) {
                ref[j] = reference_distance(&query[i * d], &train[j * d], d);
            }
            std::vector<double> sorted_ref = ref;
            std::sort(sorted_ref.begin(), sorted_ref.end());

            for (std::int64_t kk = 0; kk < k; ++kk) {
                const auto idx = static_cast<std::int64_t>(idx_arr[i * k + kk]);
                const double dst = dst_arr[i * k + kk];
                CAPTURE(i, kk, idx, dst, sorted_ref[kk]);
                REQUIRE(idx >= 0);
                REQUIRE(idx < train_count);
                REQUIRE(std::abs(dst - ref[idx]) <= tol * std::max(1.0, ref[idx]));
                REQUIRE(std::abs(dst - sorted_ref[kk]) <= tol * std::max(1.0, sorted_ref[kk]));
                if (kk > 0) {
                    REQUIRE(dst_arr[i * k + kk - 1] <= dst);
                }
            }
        }
    }
};

using chebyshev_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(chebyshev_distance_knn_test,
                     "chebyshev distance: hand-computed values",
                     "[chebyshev_distance][batch]",
                     chebyshev_types) {
    using Float = TestType;
    SKIP_IF(this->not_float64_friendly());

    // 4 train points in 3D; the largest coordinate difference sits in different columns and
    // has different signs, so sum/abs/first-or-last-column errors change the answer.
    const std::vector<Float> train = {
        0.0,  0.0,  0.0, //
        1.0,  -2.0, 0.5, //
        -3.0, 1.0,  1.0, //
        0.5,  0.5,  -4.0, //
    };
    const std::vector<Float> query = {
        0.0, 0.0, 0.0, //
        1.0, 1.0, 1.0, //
    };

    const auto x_train = homogen_table::wrap(train.data(), 4, 3);
    const auto x_query = homogen_table::wrap(query.data(), 2, 3);
    using fixture_t = chebyshev_distance_knn_test<Float>;
    const auto desc =
        typename fixture_t::knn_desc_t(4, typename fixture_t::distance_t{})
            .set_result_options(knn::result_options::indices | knn::result_options::distances);
    const auto train_result = this->train(desc, x_train);
    const auto infer_result = this->infer(desc, x_query, train_result.get_model());

    const auto idx = row_accessor<const double>(infer_result.get_indices()).pull();
    const auto dst = row_accessor<const double>(infer_result.get_distances()).pull();

    // query 0 -> distances to train: 0, 2, 3, 4
    const std::int64_t expected_idx0[] = { 0, 1, 2, 3 };
    const double expected_dst0[] = { 0.0, 2.0, 3.0, 4.0 };
    // query 1 -> distances to train: 1, 3, 4, 5
    const std::int64_t expected_idx1[] = { 0, 1, 2, 3 };
    const double expected_dst1[] = { 1.0, 3.0, 4.0, 5.0 };
    for (std::int64_t kk = 0; kk < 4; ++kk) {
        CAPTURE(kk);
        REQUIRE(idx[kk] == expected_idx0[kk]);
        REQUIRE(std::abs(dst[kk] - expected_dst0[kk]) < 1e-6);
        REQUIRE(idx[4 + kk] == expected_idx1[kk]);
        REQUIRE(std::abs(dst[4 + kk] - expected_dst1[kk]) < 1e-6);
    }
}

TEMPLATE_LIST_TEST_M(chebyshev_distance_knn_test,
                     "chebyshev distance: random data vs naive reference",
                     "[chebyshev_distance][batch]",
                     chebyshev_types) {
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t train_count = GENERATE(1, 17, 200);
    const std::int64_t query_count = GENERATE(1, 9, 150);
    const std::int64_t d = GENERATE(1, 5, 33);
    const std::int64_t k = std::min<std::int64_t>(train_count, 7);
    CAPTURE(train_count, query_count, d, k);

    const auto train = this->random_matrix(train_count, d, 7777);
    const auto query = this->random_matrix(query_count, d, 8888);
    this->check_search(train, query, train_count, query_count, d, k);
}

TEMPLATE_LIST_TEST_M(chebyshev_distance_knn_test,
                     "chebyshev distance: all neighbors, more queries than one block",
                     "[chebyshev_distance][batch]",
                     chebyshev_types) {
    SKIP_IF(this->not_float64_friendly());

    constexpr std::int64_t train_count = 40;
    constexpr std::int64_t query_count = 300;
    constexpr std::int64_t d = 6;
    const auto train = this->random_matrix(train_count, d, 1234);
    const auto query = this->random_matrix(query_count, d, 4321);
    this->check_search(train, query, train_count, query_count, d, train_count);
}

} // namespace oneapi::dal::chebyshev_distance::test
