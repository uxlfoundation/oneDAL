/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/algo/hdbscan/test/fixture.hpp"

namespace oneapi::dal::hdbscan::test {

template <typename TestType>
class hdbscan_batch_test : public hdbscan_test<TestType, hdbscan_batch_test<TestType>> {};

using hdbscan_types = COMBINE_TYPES((float, double), (hdbscan::method::brute_force));

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan compute mode check",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2,  0.15, 0.15, 5.0,
                                 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2, 5.15, 5.15, 10.0, 0.0 };
    const auto x = homogen_table::wrap(data, 11, 2);

    constexpr std::int64_t min_cluster_size = 5;
    constexpr std::int64_t min_samples = 5;

    result_option_id res_all = result_option_id(dal::result_option_id_base(mask_full));

    const result_option_id compute_mode = GENERATE_COPY(result_options::responses,
                                                        result_options::core_flags,
                                                        result_options::core_observations,
                                                        result_options::core_observation_indices,
                                                        res_all);

    this->mode_checks(compute_mode, x, min_cluster_size, min_samples);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan all noise when min_cluster_size > n",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    const auto x = homogen_table::wrap(data, 5, 1);

    constexpr std::int64_t min_cluster_size = 10;
    constexpr std::int64_t min_samples = 2;

    // All points should be noise since min_cluster_size > row_count
    this->run_checks(x, min_cluster_size, min_samples, 0);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan two well-separated clusters",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    // Two tight clusters far apart
    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        10.0,  10.0, //
        10.1,  10.1, //
        10.2,  10.0, //
        10.0,  10.2, //
        10.15, 10.15, //
    };
    const auto x = homogen_table::wrap(data, 10, 2);

    constexpr std::int64_t min_cluster_size = 5;
    constexpr std::int64_t min_samples = 5;

    // Should find 2 clusters
    this->run_checks(x, min_cluster_size, min_samples, 2);
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan single cluster with noise",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    // One tight cluster plus an outlier
    constexpr float_t data[] = {
        0.0,   0.0, //
        0.1,   0.1, //
        0.2,   0.0, //
        0.0,   0.2, //
        0.15,  0.15, //
        100.0, 100.0, //
    };
    const auto x = homogen_table::wrap(data, 6, 2);

    constexpr std::int64_t min_cluster_size = 5;
    constexpr std::int64_t min_samples = 5;

    this->run_checks(x, min_cluster_size, min_samples, -1); // -1 = don't check exact count
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan gold data test",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto x = gold_dataset::get_data().get_table(this->get_homogen_table_id());

    std::int64_t min_cluster_size = gold_dataset::get_min_cluster_size();
    std::int64_t min_samples = gold_dataset::get_min_samples();

    // The gold dataset has 4 well-separated clusters + 1 noise point
    this->run_checks(x, min_cluster_size, min_samples, -1); // -1 = don't check exact count
}

TEMPLATE_LIST_TEST_M(hdbscan_batch_test,
                     "hdbscan small min_cluster_size",
                     "[hdbscan][batch]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = std::tuple_element_t<0, TestType>;

    constexpr float_t data[] = {
        0.0, 0.0, //
        0.1, 0.1, //
        0.2, 0.0, //
        5.0, 5.0, //
        5.1, 5.1, //
        5.2, 5.0, //
    };
    const auto x = homogen_table::wrap(data, 6, 2);

    constexpr std::int64_t min_cluster_size = 2;
    constexpr std::int64_t min_samples = 2;

    // With small min_cluster_size, should find clusters
    this->run_checks(x, min_cluster_size, min_samples, -1);
}

} // namespace oneapi::dal::hdbscan::test
