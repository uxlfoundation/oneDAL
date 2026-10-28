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
#include "oneapi/dal/backend/atomic.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::backend::test {

TEST("parallel_for_2d_by_row_blocks visits every cell once", "[parallel_for_2d_by_row_blocks]") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr std::int64_t row_count = 10;
    constexpr std::int64_t col_count = 7;
    const std::int64_t block_rows = GENERATE(1, 3, 4, 9, 10, 16);
    CAPTURE(block_rows);

    auto visits = array<std::int32_t>::zeros(q, row_count * col_count);
    std::int32_t* const ptr = visits.get_mutable_data();
    parallel_for_2d_by_row_blocks(
        q,
        row_count,
        col_count,
        [=](std::int64_t row, std::int64_t col) {
            if (0 <= row && row < row_count && 0 <= col && col < col_count) {
                atomic_global_add(ptr + row * col_count + col, std::int32_t(1));
            }
            else {
                // An out-of-grid index lands on cell 0, which then counts more than one visit.
                atomic_global_add(ptr, std::int32_t(1));
            }
        },
        {},
        block_rows)
        .wait_and_throw();

    for (std::int64_t i = 0; i < row_count * col_count; i++) {
        CAPTURE(i / col_count, i % col_count);
        REQUIRE(visits[i] == 1);
    }
}

} // namespace oneapi::dal::backend::test
