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

#include "oneapi/dal/algo/dbscan/compute.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/metrics/clustering.hpp"

#include "oneapi/dal/algo/dbscan/test/data.hpp"

namespace oneapi::dal::dbscan::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

constexpr inline std::uint64_t mask_full = 0xffffffffffffffff;

template <typename TestType, typename Derived>
class dbscan_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using base_t = te::crtp_algo_fixture<TestType, Derived>;
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using result_t = compute_result<task::clustering>;
    using input_t = compute_input<task::clustering>;

    auto get_descriptor(float_t epsilon, std::int64_t min_observations) const {
        return dbscan::descriptor<float_t, method_t>(epsilon, min_observations)
            .set_mem_save_mode(true)
            .set_result_options(result_options::responses);
    }
    /// Distances between all pairs of rows of `x`, in double, from the host reference metrics of
    /// the distance primitives.
    static std::vector<double> reference_distances(const std::vector<float_t> &x,
                                                   std::int64_t column_count,
                                                   distance_metric metric,
                                                   double degree) {
        namespace pr = dal::backend::primitives;
        const std::vector<double> data(x.begin(), x.end());
        const std::int64_t row_count = data.size() / column_count;
        std::vector<double> dist(row_count * row_count);
        for (std::int64_t i = 0; i < row_count; i++) {
            for (std::int64_t j = 0; j < row_count; j++) {
                const double *a = data.data() + i * column_count;
                const double *b = data.data() + j * column_count;
                const double *a_end = a + column_count;
                double &d = dist[i * row_count + j];
                switch (metric) {
                    case distance_metric::euclidean:
                        d = pr::lp_metric<double>(2.0)(a, a_end, b);
                        break;
                    case distance_metric::manhattan:
                        d = pr::lp_metric<double>(1.0)(a, a_end, b);
                        break;
                    case distance_metric::minkowski:
                        d = pr::lp_metric<double>(degree)(a, a_end, b);
                        break;
                    case distance_metric::chebyshev:
                        d = pr::chebyshev_metric<double>()(a, a_end, b);
                        break;
                    case distance_metric::cosine:
                        d = pr::cosine_metric<double>()(a, a_end, b);
                        break;
                }
            }
        }
        return dist;
    }

    /// An `epsilon` in the middle of the widest relative gap between the sorted pairwise
    /// distances near `quantile`, so rounding cannot move a pair across it.
    static double pick_epsilon(const std::vector<double> &dist,
                               std::int64_t row_count,
                               double quantile) {
        std::vector<double> sorted;
        for (std::int64_t i = 0; i < row_count; i++) {
            for (std::int64_t j = i + 1; j < row_count; j++) {
                sorted.push_back(dist[i * row_count + j]);
            }
        }
        std::sort(sorted.begin(), sorted.end());
        const std::size_t center = static_cast<std::size_t>(quantile * sorted.size());
        const std::size_t first = center / 2;
        const std::size_t last = std::min(sorted.size() - 1, center + center / 2);
        std::size_t best = first;
        for (std::size_t k = first; k < last; k++) {
            if ((sorted[k + 1] - sorted[k]) / sorted[k + 1] >
                (sorted[best + 1] - sorted[best]) / sorted[best + 1]) {
                best = k;
            }
        }
        const double eps = (sorted[best] + sorted[best + 1]) / 2;
        CAPTURE(quantile, sorted[best], sorted[best + 1]);
        REQUIRE((sorted[best + 1] - sorted[best]) / eps > 1e-3);
        return eps;
    }

    /// Check `result` against the definition of DBSCAN on the distances `dist`: the core flags,
    /// that core rows within `epsilon` share a cluster and each cluster is one connected set of
    /// core rows, and that a border row joins the cluster of a core row within `epsilon` while
    /// every other row is noise.
    static void check_against_definition(const result_t &result,
                                         const std::vector<double> &dist,
                                         const std::vector<float_t> &weights,
                                         double epsilon,
                                         std::int64_t min_observations) {
        const std::int64_t row_count = weights.size();
        const auto labels = row_accessor<const std::int32_t>(result.get_responses()).pull();
        const auto core_flags = row_accessor<const std::int32_t>(result.get_core_flags()).pull();
        const auto near = [&](std::int64_t i, std::int64_t j) {
            return i == j || dist[i * row_count + j] <= epsilon;
        };

        std::vector<bool> is_core(row_count);
        for (std::int64_t i = 0; i < row_count; i++) {
            double count = 0;
            for (std::int64_t j = 0; j < row_count; j++) {
                count += near(i, j) ? weights[j] : 0.0;
            }
            is_core[i] = count >= min_observations;
            CAPTURE(i, count);
            REQUIRE(core_flags[i] == std::int32_t(is_core[i]));
        }

        // Components of the graph of core rows within `epsilon` of each other.
        std::vector<std::int64_t> component(row_count);
        std::iota(component.begin(), component.end(), 0);
        const auto find = [&](std::int64_t i) {
            while (component[i] != i) {
                i = component[i] = component[component[i]];
            }
            return i;
        };
        for (std::int64_t i = 0; i < row_count; i++) {
            for (std::int64_t j = 0; j < row_count; j++) {
                if (is_core[i] && is_core[j] && near(i, j)) {
                    component[find(i)] = find(j);
                }
            }
        }

        const std::int64_t cluster_count = result.get_cluster_count();
        std::vector<std::int64_t> cluster_component(cluster_count, -1);
        for (std::int64_t i = 0; i < row_count; i++) {
            CAPTURE(i, labels[i], is_core[i]);
            REQUIRE(labels[i] >= -1);
            REQUIRE(labels[i] < cluster_count);
            if (is_core[i]) {
                REQUIRE(labels[i] >= 0);
                auto &c = cluster_component[labels[i]];
                c = c < 0 ? find(i) : c;
                REQUIRE(c == find(i));
                for (std::int64_t j = 0; j < row_count; j++) {
                    REQUIRE((!is_core[j] || !near(i, j) || labels[j] == labels[i]));
                }
                continue;
            }
            bool joins_core_cluster = false;
            bool has_core_neighbor = false;
            for (std::int64_t j = 0; j < row_count; j++) {
                if (is_core[j] && near(i, j)) {
                    has_core_neighbor = true;
                    joins_core_cluster = joins_core_cluster || labels[j] == labels[i];
                }
            }
            REQUIRE(has_core_neighbor == (labels[i] >= 0));
            REQUIRE(joins_core_cluster == has_core_neighbor);
        }
        for (std::int64_t c = 0; c < cluster_count; c++) {
            REQUIRE(cluster_component[c] >= 0);
        }
    }

    void check_if_close(const table &left,
                        const table &right,
                        std::string name = "",
                        double tol = 1e-2) {
        constexpr auto eps = std::numeric_limits<float_t>::epsilon();

        const auto c_count = left.get_column_count();
        const auto r_count = left.get_row_count();

        REQUIRE(right.get_column_count() == c_count);
        REQUIRE(right.get_row_count() == r_count);

        row_accessor<const float_t> lacc(left);
        row_accessor<const float_t> racc(right);

        const auto larr = lacc.pull({ 0, -1 });
        const auto rarr = racc.pull({ 0, -1 });

        for (std::int64_t r = 0; r < r_count; ++r) {
            for (std::int64_t c = 0; c < c_count; ++c) {
                const auto lval = larr[r * c_count + c];
                const auto rval = rarr[r * c_count + c];

                CAPTURE(name, r_count, c_count, r, c, lval, rval);

                const auto aerr = std::abs(lval - rval);
                if (aerr < tol || (!std::isfinite(lval) && !std::isfinite(rval)))
                    continue;

                const auto den = std::max({ eps, //
                                            std::abs(lval),
                                            std::abs(rval) });

                const auto rerr = aerr / den;
                CAPTURE(aerr, rerr, den, r, c, lval, rval);
                REQUIRE(rerr < tol);
            }
        }
    }
    void run_checks(const table &data,
                    const table &weights,
                    float_t epsilon,
                    std::int64_t min_observations,
                    const table &ref_responses) {
        CAPTURE(epsilon, min_observations);

        INFO("create descriptor");
        const auto dbscan_desc = get_descriptor(epsilon, min_observations);

        INFO("run compute");
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), dbscan_desc, data, weights);
        check_responses_against_ref(compute_result.get_responses(), ref_responses);
    }

    void check_responses_against_ref(const table &responses, const table &ref_responses) {
        ONEDAL_ASSERT(responses.get_row_count() == ref_responses.get_row_count());
        ONEDAL_ASSERT(responses.get_column_count() == ref_responses.get_column_count());
        ONEDAL_ASSERT(responses.get_column_count() == 1);
        const auto row_count = responses.get_row_count();
        const auto rows = row_accessor<const float_t>(responses).pull({ 0, -1 });
        const auto ref_rows = row_accessor<const float_t>(ref_responses).pull({ 0, -1 });
        for (std::int64_t i = 0; i < row_count; i++) {
            REQUIRE(ref_rows[i] == rows[i]);
        }
    }
    void dbi_determenistic_checks(const table &data,
                                  double epsilon,
                                  std::int64_t min_observations,
                                  float_t ref_dbi,
                                  float_t dbi_ref_tol = 1.0e-4) {
        INFO("create descriptor");
        const auto dbscan_desc = get_descriptor(epsilon, min_observations);

        INFO("run compute");
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), dbscan_desc, data);

        const auto cluster_count = compute_result.get_cluster_count();
        REQUIRE(cluster_count > 0);

        const auto responses = compute_result.get_responses();

        auto dbi = te::davies_bouldin_index(data, responses);
        CAPTURE(dbi, ref_dbi);
        REQUIRE(check_value_with_ref_tol(dbi, ref_dbi, dbi_ref_tol));
    }

    bool check_value_with_ref_tol(float_t val, float_t ref_val, float_t ref_tol) {
        float_t max_abs = std::max(fabs(val), fabs(ref_val));
        if (max_abs == 0.0)
            return true;
        CAPTURE(val, ref_val, fabs(val - ref_val) / max_abs, ref_tol);
        return fabs(val - ref_val) / max_abs < ref_tol;
    }

    void mode_checks(result_option_id compute_mode,
                     const table &data,
                     const table &weights,
                     float_t epsilon,
                     std::int64_t min_observations) {
        CAPTURE(epsilon, min_observations);

        INFO("create descriptor");
        const auto dbscan_desc =
            get_descriptor(epsilon, min_observations).set_result_options(compute_mode);

        INFO("run compute");
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), dbscan_desc, data, weights);

        INFO("check mode");
        check_for_exception_for_non_requested_results(compute_mode, compute_result);
    }

    void check_for_exception_for_non_requested_results(result_option_id compute_mode,
                                                       const result_t &result) {
        if (!compute_mode.test(result_options::responses)) {
            REQUIRE_THROWS_AS(result.get_responses(), domain_error);
        }
        if (!compute_mode.test(result_options::core_flags)) {
            REQUIRE_THROWS_AS(result.get_core_flags(), domain_error);
        }
        if (!compute_mode.test(result_options::core_observations)) {
            REQUIRE_THROWS_AS(result.get_core_observations(), domain_error);
        }
        if (!compute_mode.test(result_options::core_observation_indices)) {
            REQUIRE_THROWS_AS(result.get_core_observation_indices(), domain_error);
        }
    }
};

} // namespace oneapi::dal::dbscan::test
