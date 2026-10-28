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

#include "oneapi/dal/test/engine/common.hpp"

#include "oneapi/dal/algo/hdbscan/backend/cpu/cluster_utils.hpp"

namespace oneapi::dal::hdbscan::backend::test {

inline auto make_descriptor() {
    return hdbscan::descriptor<double, method::brute_force>(2, 1).set_result_options(
        result_options::responses);
}

TEST("hdbscan: host extraction reads an immutable MST", "[hdbscan][cpu]") {
    // The long edge splits the path 0-1-2-3 into {0, 1} and {2, 3}.
    constexpr std::int64_t row_count = 4;
    const std::int32_t from[] = { 0, 2, 1 };
    const std::int32_t to[] = { 1, 3, 2 };
    const double weights[] = { 0.1, 0.1, 5.0 };
    const auto clusters =
        backend::extract_clusters_from_mst<double>(dal::backend::context_cpu{},
                                                   make_descriptor(),
                                                   array<std::int32_t>::wrap(from, row_count - 1),
                                                   array<std::int32_t>::wrap(to, row_count - 1),
                                                   array<double>::wrap(weights, row_count - 1),
                                                   row_count,
                                                   false,
                                                   false);

    REQUIRE(clusters.cluster_count == 2);
    const std::int32_t labels[] = { 0, 0, 1, 1 };
    for (std::int64_t i = 0; i < row_count; i++) {
        CAPTURE(i);
        REQUIRE(clusters.responses[i] == labels[i]);
    }
}

TEST("hdbscan: host extraction rejects an invalid MST", "[hdbscan][cpu]") {
    constexpr std::int64_t row_count = 4;
    const auto extract = [](std::int32_t last_to, double last_weight) {
        const std::int32_t from[] = { 0, 2, 1 };
        const std::int32_t to[] = { 1, 3, last_to };
        const double weights[] = { 0.1, 0.1, last_weight };
        return backend::extract_clusters_from_mst<double>(
            dal::backend::context_cpu{},
            make_descriptor(),
            array<std::int32_t>::wrap(from, row_count - 1),
            array<std::int32_t>::wrap(to, row_count - 1),
            array<double>::wrap(weights, row_count - 1),
            row_count,
            false,
            false);
    };

    REQUIRE_THROWS_AS(extract(row_count, 5.0), out_of_range);
    REQUIRE_THROWS_AS(extract(-1, 5.0), out_of_range);
    REQUIRE_THROWS_AS(extract(2, std::numeric_limits<double>::quiet_NaN()), domain_error);
}

} // namespace oneapi::dal::hdbscan::backend::test
