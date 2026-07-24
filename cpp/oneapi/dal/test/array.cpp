/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "oneapi/dal/test/engine/common.hpp"

using namespace Catch;

namespace oneapi::dal::test {

TEST("can construct empty array") {
    array<float> arr;

    REQUIRE(arr.get_count() == 0);
    REQUIRE(!arr.has_mutable_data());
}

TEST("can construct array of zeros") {
    auto arr = array<float>::zeros(5);

    REQUIRE(arr.get_count() == 5);
    REQUIRE(arr.has_mutable_data());
    REQUIRE(arr.get_alloc_kind() == alloc_kind::non_usm);

    for (std::int32_t i = 0; i < arr.get_count(); i++) {
        REQUIRE(arr[i] == Approx(0.0f));
    }
}

TEST("can_construct_array_of_ones") {
    auto arr = array<float>::full(5, 1.0f);

    REQUIRE(arr.get_count() == 5);
    REQUIRE(arr.has_mutable_data());

    for (std::int32_t i = 0; i < arr.get_count(); i++) {
        REQUIRE(arr[i] == Approx(1.0f));
    }
}

TEST("can_construct_array_from_raw_pointer") {
    constexpr std::int64_t size = 10;
    auto ptr = new float[size];
    for (std::int64_t i = 0; i < size; i++) {
        ptr[i] = float(i);
    }

    array arr(ptr, size, [](auto ptr) {
        delete[] ptr;
    });

    REQUIRE(arr.get_count() == size);
    REQUIRE(arr.has_mutable_data());

    for (std::int32_t i = 0; i < arr.get_count(); i++) {
        REQUIRE(arr[i] == Approx(ptr[i]));
    }
}

TEST("can_construct_array_reference") {
    auto arr = array<float>::zeros(5);
    array<float> arr2 = arr;

    REQUIRE(arr.get_count() == arr2.get_count());
    REQUIRE(arr.get_data() == arr2.get_data());
    REQUIRE(arr.has_mutable_data() == arr2.has_mutable_data());
    REQUIRE(arr.get_mutable_data() == arr2.get_mutable_data());

    for (std::int32_t i = 0; i < arr.get_count(); i++) {
        REQUIRE(arr[i] == Approx(arr2[i]));
    }
}

TEST("can reset array") {
    auto arr = array<float>::zeros(5);
    arr.reset();

    REQUIRE(arr.get_count() == 0);
    REQUIRE(!arr.has_mutable_data());
}

TEST("can reset array with bigger size") {
    auto arr = array<float>::zeros(5);
    arr.reset(10);

    REQUIRE(arr.get_count() == 10);
    REQUIRE(arr.has_mutable_data());
}

TEST("can reset array with smaller size") {
    auto arr = array<float>::zeros(5);
    arr.reset(4);

    REQUIRE(arr.get_count() == 4);
    REQUIRE(arr.has_mutable_data());
}

TEST("can reset array with raw pointer") {
    auto arr = array<float>::zeros(5);

    constexpr std::int64_t size = 10;
    auto ptr = new float[size];
    arr.reset(ptr, size, [](auto ptr) {
        delete[] ptr;
    });

    REQUIRE(arr.get_count() == size);
    REQUIRE(arr.has_mutable_data());
    REQUIRE(arr.get_mutable_data() == ptr);
    REQUIRE(arr.get_data() == ptr);
}

TEST("can reset array with non owning raw pointer") {
    auto arr = array<float>::zeros(5);

    constexpr std::int64_t size = 10;
    const float* ptr = new float[size];
    arr.reset(array<float>(), ptr, size);

    REQUIRE(arr.get_count() == size);
    REQUIRE(arr.get_data() == ptr);
    REQUIRE(!arr.has_mutable_data());
    // ASSERT_THROW(arr.get_mutable_data(), std::bad_variant_access);

    delete[] ptr;
}

TEST("can make owning array from non owning readonly") {
    array<float> arr;

    float data[] = { 1.f, 2.f, 3.f };
    arr.reset(array<float>(), (const float*)(data), 3);

    REQUIRE(arr.get_count() == 3);
    REQUIRE(arr.get_data() == data);
    REQUIRE(!arr.has_mutable_data());

    arr.need_mutable_data();

    REQUIRE(arr.get_count() == 3);
    REQUIRE(arr.get_data() != data);
    REQUIRE(arr.get_mutable_data() != data);
    REQUIRE(arr.has_mutable_data());

    for (std::int64_t i = 0; i < arr.get_count(); i++) {
        REQUIRE(arr[i] == Approx(data[i]));
    }
}

TEST("can construct non owning read write array") {
    float data[] = { 1.0f, 2.0f, 3.0f };
    array<float> arr{ data, 3, [](auto) {} };

    REQUIRE(arr.get_count() == 3);
    REQUIRE(arr.get_data() == data);
    REQUIRE(arr.has_mutable_data());
}

TEST("can construct non owning read only array") {
    float data[] = { 1.0f, 2.0f, 3.0f };
    array<float> arr{ array<float>(), static_cast<const float*>(data), 3 };

    REQUIRE(arr.get_count() == 3);
    REQUIRE(arr.get_data() == data);
    REQUIRE(!arr.has_mutable_data());
}

TEST("can wrap const data") {
    const float data[] = { 1.0f, 2.0f, 3.0f };
    auto arr = array<float>::wrap(data, 3);

    REQUIRE(arr.get_count() == 3);
    REQUIRE(arr.get_data() == data);
    REQUIRE(!arr.has_mutable_data());
}

TEST("can wrap const data with offset and deleter") {
    float* data = new float[3];
    data[0] = 0.0f;
    data[1] = 1.0f;
    data[2] = 2.0f;

    const float* cdata = data;
    auto arr = array<float>(cdata, 2, [](const float* ptr) {
        delete[] ptr;
    });

    REQUIRE(arr.get_count() == 2);
    REQUIRE(arr.get_data() == cdata);
    REQUIRE(!arr.has_mutable_data());
}

TEST("can be created from shared_ptr with shared USM data") {
    constexpr std::int64_t count = 3;
    auto* data = new float[count];
    for (std::int64_t i = 0; i < count; ++i) {
        data[i] = float(i);
    }
    std::shared_ptr<float> sdata(data, [](auto data) {});

    auto darr = array<float>{ sdata, count };

    for (std::int64_t i = 0; i < count; ++i) {
        REQUIRE(darr[i] == float(i));
    }
    delete[] data;
}

TEST("can be created from shared_ptr with const data") {
    constexpr std::int64_t count = 3;
    auto* data = new float[count];
    for (std::int64_t i = 0; i < count; ++i) {
        data[i] = float(i);
    }
    const auto* cdata = data;
    std::shared_ptr<const float> sdata(cdata, [](auto data) {});

    auto darr = array<float>{ sdata, count };

    for (std::int64_t i = 0; i < count; ++i) {
        REQUIRE(darr[i] == float(i));
    }
    delete[] data;
}

TEST("can get slice of data") {
    constexpr std::int64_t size = 1024;
    constexpr std::int64_t half = size / 2l;
    constexpr std::int64_t quarter = half / 2l;

    auto* const ptr = new float[size];
    for (std::int64_t i = 0; i < size; i++) {
        ptr[i] = float(i);
    }

    array arr(ptr, size, [](auto ptr) {
        delete[] ptr;
    });

    auto slc1 = arr.get_slice(0l, half);
    REQUIRE(slc1.get_count() == half);
    REQUIRE(slc1.get_data() == ptr);

    auto slc2 = arr.get_slice(half, size);
    REQUIRE(slc2.get_count() == half);
    REQUIRE(slc2.get_data() == ptr + half);

    auto slc3 = arr.get_slice(quarter, half);
    REQUIRE(slc3.get_count() == quarter);
    REQUIRE(slc3.get_data() == ptr + quarter);
}

#ifdef ONEDAL_DATA_PARALLEL
TEST("can construct array of zeros with queue") {
    DECLARE_TEST_POLICY(policy);

    auto arr = array<float>::zeros(policy.get_queue(), 5);

    REQUIRE(arr.get_count() == 5);
    REQUIRE(arr.has_mutable_data());

    for (std::int32_t i = 0; i < arr.get_count(); i++) {
        REQUIRE(arr[i] == Approx(0.0f));
    }
}

TEST("can construct array of ones with queue") {
    DECLARE_TEST_POLICY(policy);

    auto arr = array<float>::full(policy.get_queue(), 5, 1.0f);

    REQUIRE(arr.get_count() == 5);
    REQUIRE(arr.has_mutable_data());

    for (std::int32_t i = 0; i < arr.get_count(); i++) {
        REQUIRE(arr[i] == Approx(1.0f));
    }
}

TEST("can construct device array with queue and without initialization") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    auto arr = array<float>::empty(q, 10, sycl::usm::alloc::device);

    REQUIRE(arr.get_count() == 10);
    REQUIRE(arr.has_mutable_data());

    REQUIRE(arr.get_alloc_kind() == alloc_kind::usm_device);
    REQUIRE(sycl::get_pointer_type(arr.get_data(), q.get_context()) == sycl::usm::alloc::device);
}

TEST("can construct array with queue and events") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr std::int64_t count = 10;

    auto* data = sycl::malloc_shared<float>(count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
             data[idx[0]] = idx[0];
         });
     }).wait();

    array<float> arr{ q, data, 10, detail::make_default_delete<float>(q) };

    REQUIRE(arr.get_count() == 10);
    REQUIRE(arr.has_mutable_data());
    REQUIRE(arr.get_alloc_kind() == alloc_kind::usm_shared);

    for (std::int32_t i = 0; i < arr.get_count(); i++) {
        REQUIRE(arr[i] == Approx(float(i)));
    }
}

TEST("can reset array with queue and bigger size") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    auto arr = array<float>::zeros(q, 5);
    arr.reset(q, 10);

    REQUIRE(arr.get_count() == 10);
    REQUIRE(arr.has_mutable_data());
    REQUIRE(arr.get_alloc_kind() == alloc_kind::usm_device);
}

TEST("can reset array with queue and smaller size") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    auto arr = array<float>::zeros(q, 5);
    arr.reset(q, 4);

    REQUIRE(arr.get_count() == 4);
    REQUIRE(arr.has_mutable_data());
    REQUIRE(arr.get_alloc_kind() == alloc_kind::usm_device);
}

TEST("can reset array with queue and raw pointer") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    auto arr = array<float>::zeros(q, 5);

    constexpr std::int64_t count = 10;
    auto* data = sycl::malloc_shared<float>(count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
             data[idx[0]] = idx[0];
         });
     }).wait();

    arr.reset(data, count, detail::make_default_delete<float>(q));

    REQUIRE(arr.get_alloc_kind() == alloc_kind::usm_shared);
    REQUIRE(arr.get_size() == count * sizeof(float));
    REQUIRE(arr.get_count() == count);
    REQUIRE(arr.has_mutable_data());
    REQUIRE(arr.get_mutable_data() == data);
    REQUIRE(arr.get_data() == data);
}

TEST("can wrap const data with queue, offset and deleter") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t count = 3;

    auto data = sycl::malloc_shared<float>(count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
             data[idx[0]] = idx;
         });
     }).wait();

    const float* cdata = data;
    auto arr = array<float>(q, cdata, 2, detail::make_default_delete<const float>(q));

    REQUIRE(arr.get_count() == 2);
    REQUIRE(arr.get_data() == cdata);
    REQUIRE(!arr.has_mutable_data());
    REQUIRE(arr.get_alloc_kind() == alloc_kind::usm_shared);
}
#endif

} // namespace oneapi::dal::test
