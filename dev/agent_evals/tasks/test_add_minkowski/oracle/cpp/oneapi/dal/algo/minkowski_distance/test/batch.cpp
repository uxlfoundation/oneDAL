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
#include <random>
#include <vector>

#include "oneapi/dal/algo/minkowski_distance/common.hpp"
#include "oneapi/dal/algo/knn/infer.hpp"
#include "oneapi/dal/algo/knn/train.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::minkowski_distance::test {

namespace te = dal::test::engine;

template <typename Float>
class minkowski_distance_knn_test : public te::float_algo_fixture<Float> {
public:
    using distance_t = minkowski_distance::descriptor<Float>;
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

    static double reference_distance(const Float* x, const Float* y, std::int64_t d, double p) {
        double res = 0.0;
        for (std::int64_t k = 0; k < d; ++k) {
            res += std::pow(std::abs(double(x[k]) - double(y[k])), p);
        }
        return std::pow(res, 1.0 / p);
    }

    struct search_result {
        std::vector<std::int64_t> indices;
        std::vector<double> distances;
    };

    search_result search(const std::vector<Float>& train,
                         const std::vector<Float>& query,
                         std::int64_t train_count,
                         std::int64_t query_count,
                         std::int64_t d,
                         std::int64_t k,
                         double p) {
        const auto x_train = homogen_table::wrap(train.data(), train_count, d);
        const auto x_query = homogen_table::wrap(query.data(), query_count, d);

        const auto desc =
            knn_desc_t(k, distance_t(p))
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
        search_result res;
        for (std::int64_t i = 0; i < query_count * k; ++i) {
            res.indices.push_back(static_cast<std::int64_t>(idx_arr[i]));
            res.distances.push_back(dst_arr[i]);
        }
        return res;
    }

    // Checks brute-force kNN search with the Minkowski metric of degree p against a naive
    // reference computed in double precision.
    void check_search(const std::vector<Float>& train,
                      const std::vector<Float>& query,
                      std::int64_t train_count,
                      std::int64_t query_count,
                      std::int64_t d,
                      std::int64_t k,
                      double p) {
        const auto res = search(train, query, train_count, query_count, d, k, p);
        // degree 2 goes through the |x|^2 + |y|^2 - 2<x, y> expansion, which is lossy in float
        const double tol = std::is_same_v<Float, float> ? 5e-3 : 1e-9;

        for (std::int64_t i = 0; i < query_count; ++i) {
            std::vector<double> ref(train_count);
            for (std::int64_t j = 0; j < train_count; ++j) {
                ref[j] = reference_distance(&query[i * d], &train[j * d], d, p);
            }
            std::vector<double> sorted_ref = ref;
            std::sort(sorted_ref.begin(), sorted_ref.end());

            for (std::int64_t kk = 0; kk < k; ++kk) {
                const auto idx = res.indices[i * k + kk];
                const double dst = res.distances[i * k + kk];
                CAPTURE(p, i, kk, idx, dst, sorted_ref[kk]);
                REQUIRE(idx >= 0);
                REQUIRE(idx < train_count);
                REQUIRE(std::abs(dst - ref[idx]) <= tol * std::max(1.0, ref[idx]));
                REQUIRE(std::abs(dst - sorted_ref[kk]) <= tol * std::max(1.0, sorted_ref[kk]));
            }
        }
    }
};

using minkowski_types = std::tuple<float, double>;

TEST("minkowski distance descriptor: degree property", "[minkowski_distance][descriptor]") {
    REQUIRE(minkowski_distance::descriptor<>{}.get_degree() == 2.0);
    REQUIRE(minkowski_distance::descriptor<>{ 1.5 }.get_degree() == 1.5);
    REQUIRE(minkowski_distance::descriptor<double>{}.set_degree(3.0).get_degree() == 3.0);
    REQUIRE(minkowski_distance::descriptor<>{}.set_degree(1e-3).get_degree() == 1e-3);
}

TEST("minkowski distance descriptor: non-positive degree throws",
     "[minkowski_distance][descriptor]") {
    REQUIRE_THROWS(minkowski_distance::descriptor<>{}.set_degree(0.0));
    REQUIRE_THROWS(minkowski_distance::descriptor<>{}.set_degree(-1.0));
    REQUIRE_THROWS(minkowski_distance::descriptor<>{ 0.0 });
}

TEMPLATE_LIST_TEST_M(minkowski_distance_knn_test,
                     "minkowski distance: hand-computed values",
                     "[minkowski_distance][batch]",
                     minkowski_types) {
    using Float = TestType;
    SKIP_IF(this->not_float64_friendly());

    const std::vector<Float> train = {
        0.0,  0.0, //
        3.0,  4.0, //
        -1.0, -1.0, //
        2.0,  -2.0, //
    };
    const std::vector<Float> query = { 0.0, 0.0 };

    // distances from the origin: p=1 -> 0, 7, 2, 4; p=3 -> 0, cbrt(91), cbrt(2), cbrt(16)
    const double p = GENERATE(1.0, 3.0);
    const std::int64_t expected_idx[] = { 0, 2, 3, 1 };
    const double expected_p1[] = { 0.0, 2.0, 4.0, 7.0 };
    const double expected_p3[] = { 0.0, std::cbrt(2.0), std::cbrt(16.0), std::cbrt(91.0) };

    const auto res = this->search(train, query, 4, 1, 2, 4, p);
    for (std::int64_t kk = 0; kk < 4; ++kk) {
        const double expected = p == 1.0 ? expected_p1[kk] : expected_p3[kk];
        CAPTURE(p, kk, res.indices[kk], res.distances[kk], expected);
        REQUIRE(res.indices[kk] == expected_idx[kk]);
        REQUIRE(std::abs(res.distances[kk] - expected) < 1e-5);
    }
}

TEMPLATE_LIST_TEST_M(minkowski_distance_knn_test,
                     "minkowski distance: random data vs naive reference",
                     "[minkowski_distance][batch]",
                     minkowski_types) {
    SKIP_IF(this->not_float64_friendly());

    const double p = GENERATE(1.0, 1.5, 2.0, 3.0, 4.5);
    const std::int64_t train_count = GENERATE(1, 23, 200);
    const std::int64_t query_count = GENERATE(1, 150);
    const std::int64_t d = GENERATE(1, 7, 20);
    const std::int64_t k = std::min<std::int64_t>(train_count, 5);
    CAPTURE(p, train_count, query_count, d, k);

    const auto train = this->random_matrix(train_count, d, 7777);
    const auto query = this->random_matrix(query_count, d, 8888);
    this->check_search(train, query, train_count, query_count, d, k, p);
}

} // namespace oneapi::dal::minkowski_distance::test
