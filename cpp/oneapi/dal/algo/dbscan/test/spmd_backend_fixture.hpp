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

#include <limits>
#include <cmath>

#include "oneapi/dal/algo/dbscan/compute.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/metrics/clustering.hpp"
#include "oneapi/dal/spmd/communicator.hpp"

namespace oneapi::dal::dbscan::test {

#ifdef ONEDAL_DATA_PARALLEL

namespace te = dal::test::engine;
namespace la = te::linalg;

constexpr inline std::uint64_t mask_full = 0xffffffffffffffff;

template <typename Backend, typename TestType, typename Derived>
class dbscan_spmd_backend_fixture : public te::crtp_spmd_backend_algo_fixture<TestType, Derived> {
public:
    using base_t = te::crtp_spmd_backend_algo_fixture<TestType, Derived>;
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using result_t = compute_result<task::clustering>;
    using input_t = compute_input<task::clustering>;
    using base_t::get_policy;

    auto get_comm() {
        return dal::preview::spmd::make_communicator<Backend>(this->get_queue());
    }

    auto get_descriptor(float_t epsilon, std::int64_t min_observations) const {
        return dbscan::descriptor<float_t, method_t>(epsilon, min_observations)
            .set_mem_save_mode(true)
            .set_result_options(result_options::responses);
    }

    void run_checks(const table& data,
                    const table& weights,
                    float_t epsilon,
                    std::int64_t min_observations,
                    const table& ref_responses) {
        CAPTURE(epsilon, min_observations);

        INFO("create descriptor");
        const auto dbscan_desc = get_descriptor(epsilon, min_observations);

        INFO("run compute");
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), dbscan_desc, data, weights);

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
    void dbi_determenistic_checks(const table& data,
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
        if (max_abs == float_t(0))
            return true;
        CAPTURE(val, ref_val, fabs(val - ref_val) / max_abs, ref_tol);
        return fabs(val - ref_val) / max_abs < ref_tol;
    }

    void mode_checks(result_option_id compute_mode,
                     const table& data,
                     const table& weights,
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

#endif
} // namespace oneapi::dal::dbscan::test
