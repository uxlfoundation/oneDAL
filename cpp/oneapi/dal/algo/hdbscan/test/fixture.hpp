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

#include <limits>
#include <cmath>
#include <map>

#include "oneapi/dal/algo/hdbscan/compute.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

#include "oneapi/dal/algo/hdbscan/test/data.hpp"

namespace oneapi::dal::hdbscan::test {

namespace te = dal::test::engine;

constexpr inline std::uint64_t mask_full = 0xffffffffffffffff;

/// Check that two label arrays define the same partition (permutation-invariant).
/// Different implementations may assign different integer labels to the same clusters
/// because floating-point differences can reorder MST edges.
template <typename Float>
static void check_same_partition(const dal::array<Float>& a_rows,
                                 const dal::array<Float>& b_rows,
                                 std::int64_t row_count) {
    std::map<std::int32_t, std::int32_t> a_to_b;
    std::map<std::int32_t, std::int32_t> b_to_a;

    for (std::int64_t i = 0; i < row_count; i++) {
        const auto a = static_cast<std::int32_t>(a_rows[i]);
        const auto b = static_cast<std::int32_t>(b_rows[i]);

        auto it = a_to_b.find(a);
        if (it == a_to_b.end()) {
            CAPTURE(i, a, b);
            REQUIRE(b_to_a.find(b) == b_to_a.end());
            a_to_b[a] = b;
            b_to_a[b] = a;
        }
        else {
            CAPTURE(i, a, b, it->second);
            REQUIRE(it->second == b);
        }
    }
}

template <typename TestType, typename Derived>
class hdbscan_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using base_t = te::crtp_algo_fixture<TestType, Derived>;
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using result_t = compute_result<task::clustering>;
    using input_t = compute_input<task::clustering>;

    auto get_descriptor(std::int64_t min_cluster_size, std::int64_t min_samples) const {
        return hdbscan::descriptor<float_t, method_t>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);
    }

    auto get_descriptor(std::int64_t min_cluster_size,
                        std::int64_t min_samples,
                        distance_metric metric,
                        double degree = 2.0) const {
        return hdbscan::descriptor<float_t, method_t>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses)
            .set_metric(metric)
            .set_degree(degree);
    }

    void run_checks(const table& data,
                    std::int64_t min_cluster_size,
                    std::int64_t min_samples,
                    std::int64_t expected_cluster_count) {
        CAPTURE(min_cluster_size, min_samples);

        INFO("create descriptor");
        const auto hdbscan_desc = get_descriptor(min_cluster_size, min_samples);

        INFO("run compute");
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), hdbscan_desc, data);

        check_compute_result(compute_result, data, expected_cluster_count);
    }

    void run_checks(const table& data,
                    std::int64_t min_cluster_size,
                    std::int64_t min_samples,
                    distance_metric metric,
                    double degree,
                    std::int64_t expected_cluster_count) {
        CAPTURE(min_cluster_size, min_samples, static_cast<int>(metric), degree);

        INFO("create descriptor");
        const auto hdbscan_desc = get_descriptor(min_cluster_size, min_samples, metric, degree);

        INFO("run compute");
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), hdbscan_desc, data);

        check_compute_result(compute_result, data, expected_cluster_count);
    }

    void check_compute_result(const result_t& compute_result,
                              const table& data,
                              std::int64_t expected_cluster_count) {
        INFO("check cluster count");
        REQUIRE(compute_result.get_cluster_count() >= 0);

        if (expected_cluster_count >= 0) {
            REQUIRE(compute_result.get_cluster_count() == expected_cluster_count);
        }

        INFO("check responses shape");
        const auto responses = compute_result.get_responses();
        REQUIRE(responses.get_row_count() == data.get_row_count());
        REQUIRE(responses.get_column_count() == 1);

        INFO("check response values");
        const auto rows = row_accessor<const float_t>(responses).pull({ 0, -1 });
        for (std::int64_t i = 0; i < data.get_row_count(); i++) {
            const auto label = static_cast<std::int32_t>(rows[i]);
            // Labels should be >= -1 (noise) and < cluster_count
            REQUIRE(label >= -1);
            if (compute_result.get_cluster_count() > 0) {
                REQUIRE(label < compute_result.get_cluster_count());
            }
        }
    }

    void run_checks_with_responses(const table& data,
                                   std::int64_t min_cluster_size,
                                   std::int64_t min_samples,
                                   const table& ref_responses) {
        CAPTURE(min_cluster_size, min_samples);

        INFO("create descriptor");
        const auto hdbscan_desc = get_descriptor(min_cluster_size, min_samples);

        INFO("run compute");
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), hdbscan_desc, data);

        INFO("check responses match reference");
        check_responses_against_ref(compute_result.get_responses(), ref_responses);
    }

    void check_responses_against_ref(const table& responses, const table& ref_responses) {
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

    void mode_checks(result_option_id compute_mode,
                     const table& data,
                     std::int64_t min_cluster_size,
                     std::int64_t min_samples) {
        CAPTURE(min_cluster_size, min_samples);

        INFO("create descriptor");
        const auto hdbscan_desc =
            get_descriptor(min_cluster_size, min_samples).set_result_options(compute_mode);

        INFO("run compute");
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), hdbscan_desc, data);

        INFO("check mode");
        check_for_exception_for_non_requested_results(compute_mode, compute_result);
    }

    void check_for_exception_for_non_requested_results(result_option_id compute_mode,
                                                       const result_t& result) {
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

} // namespace oneapi::dal::hdbscan::test
