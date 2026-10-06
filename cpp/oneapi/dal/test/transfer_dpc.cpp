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

#include "oneapi/dal/backend/transfer.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test {

#ifdef ONEDAL_DATA_PARALLEL
namespace bk = dal::backend;

static sycl::event fill_with_index(sycl::queue& q, array<float>& arr) {
    float* const ptr = arr.get_mutable_data();
    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(arr.get_count()), [=](sycl::id<1> idx) {
            ptr[idx] = float(idx[0]);
        });
    });
}

TEST("to_host aliases host-readable USM and orders after the dependencies") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t count = 1024;

    const auto alloc = GENERATE(sycl::usm::alloc::shared, sycl::usm::alloc::host);
    auto arr = array<float>::empty(q, count, alloc);
    const auto fill_event = fill_with_index(q, arr);

    auto [arr_host, event] = bk::to_host(arr, { fill_event });
    event.wait_and_throw();

    REQUIRE(arr_host.get_data() == arr.get_data());
    for (std::int64_t i = 0; i < count; ++i) {
        REQUIRE(arr_host[i] == float(i));
    }
}

TEST("to_host copies device USM after the dependencies") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t count = 1024;

    auto arr = array<float>::empty(q, count, sycl::usm::alloc::device);
    const auto fill_event = fill_with_index(q, arr);

    const auto arr_host = bk::to_host_sync(arr, { fill_event });

    REQUIRE(arr_host.get_data() != arr.get_data());
    REQUIRE(sycl::get_pointer_type(arr_host.get_data(), q.get_context()) == sycl::usm::alloc::host);
    for (std::int64_t i = 0; i < count; ++i) {
        REQUIRE(arr_host[i] == float(i));
    }
}
#endif

} // namespace oneapi::dal::test
