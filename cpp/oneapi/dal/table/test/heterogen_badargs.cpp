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
#include "oneapi/dal/chunked_array.hpp"

#include "oneapi/dal/table/heterogen.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test {

TEST("Dummy. Needed to produce non-empty test suite on host") {}

#ifdef ONEDAL_DATA_PARALLEL

TEST("Cannot create heterogen table from columns with different alloc kinds") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr auto device = sycl::usm::alloc::device;
    constexpr auto shared = sycl::usm::alloc::shared;

    auto arr0 = array<float>::empty(q, 8l, shared);
    chunked_array<float> chunked0(arr0);

    auto arr1 = array<std::int32_t>::empty(q, 8l, device);
    chunked_array<std::int32_t> chunked1(arr1);

    REQUIRE_THROWS_AS(heterogen_table::wrap(chunked0, chunked1), invalid_argument);
}

TEST("Cannot build chunked array from chunks with different alloc kinds") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr auto device = sycl::usm::alloc::device;
    constexpr auto shared = sycl::usm::alloc::shared;

    auto arr0 = array<float>::empty(q, 4l, shared);
    auto arr1 = array<float>::empty(q, 4l, device);

    chunked_array<float> chunked(2);
    chunked.set_chunk(0l, arr0);
    REQUIRE_THROWS_AS(chunked.set_chunk(1l, arr1), invalid_argument);
}

#endif

} // namespace oneapi::dal::test
