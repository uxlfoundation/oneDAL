/*******************************************************************************
* Copyright 2023 Intel Corporation
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

TEST("can construct an empty chunked_array") {
    chunked_array<float> chunked;

    REQUIRE(chunked.get_count() == 0);
}

TEST("can set chunks into chunked_array") {
    constexpr float src1[] = { 4.f, 5.f };
    constexpr float src2[] = { 1.f, 2.f, 3.f };

    auto arr1 = array<float>::wrap(src1, 2l);
    auto arr2 = array<float>::wrap(src2, 3l);

    chunked_array<float> chunked(2);

    chunked.set_chunk(0, arr1);
    chunked.set_chunk(1, arr2);

    REQUIRE(chunked.get_count() == 5l);
}

TEST("can create copy chunked_array") {
    constexpr float src1[] = { 4.f, 5.f };
    constexpr float src2[] = { 1.f, 2.f, 3.f };

    auto arr1 = array<float>::wrap(src1, 2l);
    auto arr2 = array<float>::wrap(src2, 3l);

    chunked_array<float> source(2);

    source.set_chunk(0, arr1);
    source.set_chunk(1, arr2);

    chunked_array<float> destination = source;

    REQUIRE(source.get_count() == 5l);
    REQUIRE(destination.get_count() == 5l);

    chunked_array<float> further{ destination };

    REQUIRE(destination.get_count() == 5l);
    REQUIRE(further.get_count() == 5l);
}

TEST("can create move of chunked_array") {
    constexpr float src1[] = { 4.f, 5.f };
    constexpr float src2[] = { 1.f, 2.f, 3.f };

    auto arr1 = array<float>::wrap(src1, 2l);
    auto arr2 = array<float>::wrap(src2, 3l);

    chunked_array<float> source(2);

    source.set_chunk(0, arr1);
    source.set_chunk(1, arr2);

    chunked_array<float> destination = std::move(source);

    REQUIRE(source.get_count() == 0l);
    REQUIRE(destination.get_count() == 5l);

    chunked_array<float> further{ std::move(destination) };

    REQUIRE(destination.get_count() == 0l);
    REQUIRE(further.get_count() == 5l);
}

TEST("can set another chunk") {
    constexpr float src1[] = { 4.f, 5.f };
    constexpr float src2[] = { 1.f, 2.f, 3.f };
    constexpr float src3[] = { 6.f, 7.f, 8.f, 9.f };

    auto arr1 = array<float>::wrap(src1, 2l);
    auto arr2 = array<float>::wrap(src2, 3l);
    auto arr3 = array<float>::wrap(src3, 4l);

    chunked_array<float> chunked(2);

    chunked.set_chunk(0, arr1);
    chunked.set_chunk(1, arr2);

    chunked.set_chunk(0, arr3);

    REQUIRE(chunked.get_chunk_count() == 2l);
    REQUIRE(chunked.get_count() == 7l);
}

TEST("can construct chunked_array from 2 arrays") {
    constexpr float src1[] = { 1.f, 2.f, 3.f };
    constexpr float src2[] = { 4.f, 5.f };

    auto arr1 = array<float>::wrap(src1, 3l);
    auto arr2 = array<float>::wrap(src2, 2l);

    chunked_array<float> chunked(arr1, arr2);

    REQUIRE(chunked.get_count() == 5l);
}

TEST("elements are correct in order after flatten") {
    constexpr std::int64_t count1 = 3;
    constexpr std::int64_t count2 = 2;

    constexpr float src1[count1] = { 1.f, 2.f, 3.f };
    constexpr float src2[count2] = { 4.f, 5.f };

    auto arr1 = array<float>::wrap(src1, count1);
    auto arr2 = array<float>::wrap(src2, count2);

    chunked_array<float> chunked(arr1, arr2);

    auto flt = chunked.flatten();

    for (std::int64_t i = 0; i < count1; ++i) {
        const auto val = flt[i];
        const auto gtr = arr1[i];

        CAPTURE(i, val, gtr);
        REQUIRE(val == gtr);
    }

    for (std::int64_t i = 0; i < count2; ++i) {
        const auto val = flt[i + count1];
        const auto gtr = arr2[i];

        CAPTURE(i, val, gtr);
        REQUIRE(val == gtr);
    }
}

TEST("check is not contiguous") {
    constexpr std::int64_t count = 6l;

    constexpr float src[count] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    auto arr1 = array<float>::wrap(src + 0l, 2l);
    auto arr2 = array<float>::wrap(src + 3l, 3l);

    chunked_array<float> chunked(arr1, arr2);

    REQUIRE(!chunked.is_contiguous());
}

TEST("check is contiguous") {
    constexpr std::int64_t count = 6l;

    constexpr float src[count] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    auto arr1 = array<float>::wrap(src + 0l, 2l);
    auto arr2 = array<float>::wrap(src + 2l, 4l);

    chunked_array<float> chunked1(arr1, arr2);

    REQUIRE(chunked1.is_contiguous());

    auto arr = array<float>::wrap(src, count);
    chunked_array<float> chunked2(arr);

    REQUIRE(chunked2.is_contiguous());
}

TEST("can append new chunks") {
    constexpr std::int64_t count = 6l;

    constexpr float src[count] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    auto arr1 = array<float>::wrap(src + 0l, 2l);
    auto arr2 = array<float>::wrap(src + 2l, 4l);

    chunked_array<float> chunked{};

    chunked.append(arr1);
    chunked.append(arr2);

    REQUIRE(chunked.validate());
    REQUIRE(chunked.is_contiguous());
    REQUIRE(chunked.get_count() == count);

    auto flt = chunked.flatten();

    for (std::int64_t i = 0; i < count; ++i) {
        const auto val = src[i];
        const auto gtr = flt[i];

        CAPTURE(i, val, gtr);
        REQUIRE(val == gtr);
    }

    chunked.append(arr1);

    REQUIRE(chunked.validate());
    REQUIRE(!chunked.is_contiguous());
    auto new_count = count + arr1.get_count();
    REQUIRE(chunked.get_count() == new_count);
}

TEST("can append chunked array") {
    constexpr std::int64_t count = 6l;

    constexpr float src[count] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    auto arr1 = array<float>::wrap(src + 0l, 2l);
    auto arr2 = array<float>::wrap(src + 2l, 4l);

    chunked_array<float> chunked(arr1);
    chunked_array<float> chunked1(arr1, arr2);
    chunked_array<float> chunked2(arr1, arr1, arr2);

    chunked.append(chunked1);

    REQUIRE(chunked.validate());
    REQUIRE(chunked.get_count() == 8l);
    REQUIRE(chunked.have_same_policies());

    auto val_arr = chunked.flatten();
    auto gtr_arr = chunked2.flatten();

    for (std::int64_t i = 0; i < count; ++i) {
        const auto val = val_arr[i];
        const auto gtr = gtr_arr[i];

        CAPTURE(i, val, gtr);
        REQUIRE(val == gtr);
    }
}

TEST("can get data from a chunked_array") {
    constexpr std::int64_t count = 6l;

    constexpr float src[count] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    auto arr1 = array<float>::wrap(src + 0l, 2l);
    auto arr2 = array<float>::wrap(src + 2l, 4l);

    chunked_array<float> chunked(arr1, arr1, arr2);

    REQUIRE(chunked.have_same_policies());

    auto data = chunked.get_data();

    const auto cc = chunked.get_chunk_count();
    for (std::int64_t c = 0l; c < cc; ++c) {
        const auto chunk = chunked.get_chunk(c);

        const auto gtr = chunk.get_data();
        const auto val = data[c];

        CAPTURE(c, cc, gtr, val);
        REQUIRE(gtr == val);
    }
}

template <typename Type>
void check_if_equal(const array<Type>& lhs, const array<Type>& rhs) {
    const auto count = lhs.get_count();
    REQUIRE(count == rhs.get_count());

    for (std::int64_t i = 0; i < count; ++i) {
        const auto l_val = lhs[i];
        const auto r_val = rhs[i];

        CAPTURE(i, l_val, r_val);
        REQUIRE(l_val == r_val);
    }
}

TEST("can get slice of chunked_array on host") {
    constexpr std::int64_t count = 6l;

    constexpr float src[count] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    auto arr1 = array<float>::wrap(src + 0l, 2l);
    auto arr2 = array<float>::wrap(src + 2l, 4l);

    chunked_array<float> chunked(arr1, arr1, arr2);
    auto flattened = chunked.flatten();

    REQUIRE(chunked.get_count() == 8l);

    auto check_slice = [&](auto lhs, auto rhs) {
        auto slice = chunked.get_slice(lhs, rhs);
        REQUIRE(slice.get_count() == rhs - lhs);
        auto gtr = flattened.get_slice(lhs, rhs);
        check_if_equal(slice.flatten(), gtr);
    };

    check_slice(0l, 5l);
    check_slice(2l, 5l);
    check_slice(0l, 8l);
    check_slice(2l, 8l);
    check_slice(2l, 4l);
    check_slice(0l, 2l);
    check_slice(1l, 4l);
}

#ifdef ONEDAL_DATA_PARALLEL

TEST("can flatten array on device") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t count = 4;

    auto deleter = detail::make_default_delete<const float>(q);

    auto* const data1 = sycl::malloc_device<float>(count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
             data1[idx] = static_cast<float>(idx);
         });
     }).wait_and_throw();
    auto arr1 = array<float>(q, data1, count, deleter);

    auto* const data2 = sycl::malloc_device<float>(count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
             data2[idx] = static_cast<float>(idx);
         });
     }).wait_and_throw();
    auto arr2 = array<float>(q, data2, count, deleter);

    chunked_array<float> chunked(arr1, arr2);

    chunked.append(arr1, arr2);

    REQUIRE(chunked.have_same_policies());
    REQUIRE(chunked.get_count() == 4l * count);

    auto shared_arr = chunked.flatten(q, sycl::usm::alloc::shared);

    for (std::int64_t i = 0l; i < (4l * count); ++i) {
        const auto gtr = i % count;
        const auto val = shared_arr[i];

        CAPTURE(i, gtr, val);
        REQUIRE(gtr == val);
    }
}

TEST("can get data array on device") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t count = 4;

    auto deleter = detail::make_default_delete<const float>(q);

    auto* const data1 = sycl::malloc_device<float>(count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
             data1[idx] = static_cast<float>(idx);
         });
     }).wait_and_throw();
    auto arr1 = array<float>(q, data1, count, deleter);

    auto* const data2 = sycl::malloc_device<float>(count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
             data2[idx] = static_cast<float>(idx);
         });
     }).wait_and_throw();
    auto arr2 = array<float>(q, data2, count, deleter);

    chunked_array<float> chunked(arr1, arr2);

    REQUIRE(chunked.have_same_policies());

    auto data_host = chunked.get_data(q, sycl::usm::alloc::host);

    REQUIRE(data_host[0] == arr1.get_data());
    REQUIRE(data_host[1] == arr2.get_data());

    auto data_shared = chunked.get_data(q, sycl::usm::alloc::shared);

    REQUIRE(data_shared[0] == arr1.get_data());
    REQUIRE(data_shared[1] == arr2.get_data());
}

TEST("can get slice of chunked_array on device") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t arr2_count = 4;
    constexpr std::int64_t arr1_count = 3;

    auto deleter = detail::make_default_delete<const float>(q);

    auto* const data1 = sycl::malloc_device<float>(arr1_count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(arr1_count), [=](sycl::id<1> idx) {
             data1[idx] = static_cast<float>(idx);
         });
     }).wait_and_throw();
    auto arr1 = array<float>(q, data1, arr1_count, deleter);

    auto* const data2 = sycl::malloc_device<float>(arr2_count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(arr2_count), [=](sycl::id<1> idx) {
             data2[idx] = static_cast<float>(idx);
         });
     }).wait_and_throw();
    auto arr2 = array<float>(q, data2, arr2_count, deleter);

    chunked_array<float> chunked(arr2, arr1, arr2);
    auto flattened = chunked.flatten();

    REQUIRE(chunked.get_count() == 11l);

    auto check_slice = [&](auto lhs, auto rhs) {
        auto slice = chunked.get_slice(lhs, rhs);
        REQUIRE(slice.get_count() == rhs - lhs);
        auto gtr = flattened.get_slice(lhs, rhs);
        check_if_equal(slice.flatten(), gtr);
    };

    check_slice(0l, 11l);
    check_slice(0l, 4l);
    check_slice(0l, 7l);
    check_slice(4l, 11l);
    check_slice(7l, 11l);
    check_slice(1l, 10l);
    check_slice(3l, 8l);
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::test
