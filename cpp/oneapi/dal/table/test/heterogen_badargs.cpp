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

TEST("Cannot create heterogen table from columns with different alloc kinds") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr auto device = sycl::usm::alloc::device;
    constexpr auto shared = sycl::usm::alloc::shared;

    auto arr0 = array<float>::empty(q, 8l, shared);
    std::iota(begin(arr0), end(arr0), float(0));
    chunked_array<float> chunked0(arr0);

    auto arr1 = array<std::int32_t>::empty(q, 8l, device);
    chunked_array<std::int32_t> chunked1(arr1);

    REQUIRE_THROWS_AS(heterogen_table::wrap(chunked0, chunked1), invalid_argument);
}

} // namespace oneapi::dal::test
