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

#include "oneapi/dal/chunked_array.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test {

#ifdef ONEDAL_DATA_PARALLEL

TEST("throw if allocation kinds of chunks are different") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t count = 3;

    auto deleter = detail::make_default_delete<const float>(q);

    auto* const data0 = sycl::malloc_shared<float>(count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
             data0[idx] = static_cast<float>(idx);
         });
     }).wait_and_throw();
    auto arr0 = array<float>(q, data0, count, deleter);

    constexpr float data1[count] = { 0.f, 1.f, 2.f };
    auto arr1 = array<float>::wrap(data1, count);

    chunked_array<float> chunked;

    REQUIRE_THROWS_AS(chunked.append(arr0, arr1), invalid_argument);
}

#endif

} // namespace oneapi::dal::test
