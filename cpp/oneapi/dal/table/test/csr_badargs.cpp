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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/linalg.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

TEST("create table with invalid row or column count") {
    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };

    REQUIRE_NOTHROW(csr_table::wrap(data, column_indices, row_offsets, row_count, column_count));
    REQUIRE_THROWS_AS(csr_table::wrap(data, column_indices, row_offsets, 0, column_count),
                      dal::domain_error);
    REQUIRE_THROWS_AS(csr_table::wrap(data, column_indices, row_offsets, row_count, 0),
                      dal::domain_error);
}

TEST("create tables that have one-based indexing and various types of incorrect indices") {
    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t column_indices_lt_min[] = { 1, 2, 4, 3, 0, 4, 2 };
    const std::int64_t column_indices_gt_max[] = { 1, 2, 5, 3, 2, 4, 2 };
    const std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };
    const std::int64_t row_offsets_lt_min[] = { 0, 4, 5, 7, 8 };
    const std::int64_t row_offsets_gt_max[] = { 1, 4, 9, 7, 8 };
    const std::int64_t row_offsets_not_ascending[] = { 1, 4, 7, 5, 8 };
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };

    REQUIRE_THROWS_AS(
        csr_table::wrap(data, column_indices_lt_min, row_offsets, row_count, column_count),
        dal::domain_error);
    REQUIRE_THROWS_AS(
        csr_table::wrap(data, column_indices_gt_max, row_offsets, row_count, column_count),
        dal::domain_error);
    REQUIRE_THROWS_AS(
        csr_table::wrap(data, column_indices, row_offsets_lt_min, row_count, column_count),
        dal::domain_error);
    REQUIRE_THROWS_AS(
        csr_table::wrap(data, column_indices, row_offsets_gt_max, row_count, column_count),
        dal::domain_error);
    REQUIRE_THROWS_AS(
        csr_table::wrap(data, column_indices, row_offsets_not_ascending, row_count, column_count),
        dal::domain_error);
}

#ifdef ONEDAL_DATA_PARALLEL

TEST((std::string("cannot construct table from arrays allocated on different devices")).c_str()) {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 7 };

    const float data_host[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices_host[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets_host[] = { 1, 4, 5, 7, 8 };

    auto* const column_indices = sycl::malloc_device<std::int64_t>(element_count, q);
    auto* const row_offsets = sycl::malloc_device<std::int64_t>(row_count + 1, q);

    auto column_indices_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(column_indices, column_indices_host, element_count * sizeof(std::int64_t));
    });

    auto row_offsets_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(row_offsets, row_offsets_host, (row_count + 1) * sizeof(std::int64_t));
    });

    const auto data_array = array<float>::wrap(data_host, element_count);
    const auto column_indices_array =
        array<std::int64_t>::wrap(q, column_indices, element_count, { column_indices_event });
    const auto row_offsets_array =
        array<std::int64_t>::wrap(q, row_offsets, row_count + 1, { row_offsets_event });

    REQUIRE_THROWS_AS(
        csr_table::wrap(data_array, column_indices_array, row_offsets_array, column_count),
        dal::domain_error);

    sycl::free(column_indices, q);
    sycl::free(row_offsets, q);
}

TEST((std::string("cannot construct table from arrays from different queues")).c_str()) {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    sycl::queue q2 = sycl::queue{ sycl::gpu_selector{} };
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 7 };

    const float data_host[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices_host[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets_host[] = { 1, 4, 5, 7, 8 };

    auto* const data = sycl::malloc_device<float>(element_count, q);
    auto* const column_indices = sycl::malloc_device<std::int64_t>(element_count, q2);
    auto* const row_offsets = sycl::malloc_device<std::int64_t>(row_count + 1, q);

    auto data_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(data, data_host, element_count * sizeof(float));
    });

    auto column_indices_event = q2.submit([&](sycl::handler& cgh) {
        cgh.memcpy(column_indices, column_indices_host, element_count * sizeof(std::int64_t));
    });

    auto row_offsets_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(row_offsets, row_offsets_host, (row_count + 1) * sizeof(std::int64_t));
    });

    const auto data_array = array<float>::wrap(q, data, element_count, { data_event });
    const auto column_indices_array =
        array<std::int64_t>::wrap(q2, column_indices, element_count, { column_indices_event });
    const auto row_offsets_array =
        array<std::int64_t>::wrap(q, row_offsets, row_count + 1, { row_offsets_event });

    REQUIRE_THROWS_AS(
        csr_table::wrap(data_array, column_indices_array, row_offsets_array, column_count),
        dal::invalid_argument);

    sycl::free(data, q);
    sycl::free(column_indices, q2);
    sycl::free(row_offsets, q);
}

#endif

} // namespace oneapi::dal::test
