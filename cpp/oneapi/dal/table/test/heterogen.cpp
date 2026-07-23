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

#include <numeric>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/chunked_array.hpp"

#include "oneapi/dal/table/heterogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/column_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test {

template <typename Type>
inline Type* begin(const dal::array<Type>& arr) {
    ONEDAL_ASSERT(arr.has_mutable_data());
    return arr.get_mutable_data();
}

template <typename Type>
inline Type* end(const dal::array<Type>& arr) {
    return begin(arr) + arr.get_count();
}

TEST("can construct empty table") {
    heterogen_table t;

    REQUIRE(t.has_data() == false);
    REQUIRE(t.get_row_count() == 0);
    REQUIRE(t.get_column_count() == 0);
    REQUIRE(t.get_kind() == heterogen_table::kind());
}

TEST("Can create table from chunked arrays") {
    constexpr float src1[] = { 1.f, 2.f };
    constexpr float src2[] = { 3.f, 4.f, 5.f };

    auto arr1 = array<float>::wrap(src1, 2l);
    auto arr2 = array<float>::wrap(src2, 3l);

    chunked_array<float> chunked1(2);
    chunked1.set_chunk(0l, arr1);
    chunked1.set_chunk(1l, arr2);
    ONEDAL_ASSERT(chunked1.get_count() == 5l);

    chunked_array<float> chunked2(2);
    chunked2.set_chunk(0l, arr2);
    chunked2.set_chunk(1l, arr1);
    ONEDAL_ASSERT(chunked2.get_count() == 5l);

    auto table = heterogen_table::wrap( //
        chunked1,
        chunked2,
        chunked1,
        chunked2);

    REQUIRE(table.has_data() == true);
    REQUIRE(table.get_row_count() == 5l);
    REQUIRE(table.get_column_count() == 4l);
    REQUIRE(table.get_kind() == heterogen_table::kind());
}

TEST("Can create table from different chunked arrays") {
    constexpr float src1[] = { 0.f, 2.f };
    constexpr float src2[] = { 4.f, 6.f, 8.f };

    auto arr1 = array<float>::wrap(src1, 2l);
    auto arr2 = array<float>::wrap(src2, 3l);

    chunked_array<float> chunked1(2);
    chunked1.set_chunk(0l, arr1);
    chunked1.set_chunk(1l, arr2);
    ONEDAL_ASSERT(chunked1.get_count() == 5l);

    constexpr std::int8_t src3[] = { 1 };
    constexpr std::int8_t src4[] = { 3, 5 };
    constexpr std::int8_t src5[] = { 7, 9 };

    auto arr3 = array<std::int8_t>::wrap(src3, 1l);
    auto arr4 = array<std::int8_t>::wrap(src4, 2l);
    auto arr5 = array<std::int8_t>::wrap(src5, 2l);

    chunked_array<std::int8_t> chunked2(3);
    chunked2.set_chunk(0l, arr3);
    chunked2.set_chunk(1l, arr4);
    chunked2.set_chunk(2l, arr5);
    ONEDAL_ASSERT(chunked2.get_count() == 5l);

    auto table = heterogen_table::wrap( //
        chunked1,
        chunked2,
        chunked1);

    REQUIRE(table.has_data() == true);
    REQUIRE(table.get_row_count() == 5l);
    REQUIRE(table.get_column_count() == 3l);
    REQUIRE(table.get_kind() == heterogen_table::kind());
}

TEST("Can create table manually") {
    constexpr float src1[] = { 1.f, 2.f };
    constexpr float src2[] = { 3.f, 4.f, 5.f };

    auto arr1 = array<float>::wrap(src1, 2l);
    auto arr2 = array<float>::wrap(src2, 3l);

    chunked_array<float> chunked1(2);
    chunked1.set_chunk(0l, arr1);
    chunked1.set_chunk(1l, arr2);
    ONEDAL_ASSERT(chunked1.get_count() == 5l);

    constexpr std::int8_t src3[] = { 1 };
    constexpr std::int8_t src4[] = { 2, 3 };
    constexpr std::int8_t src5[] = { 4, 5 };

    auto arr3 = array<std::int8_t>::wrap(src3, 1l);
    auto arr4 = array<std::int8_t>::wrap(src4, 2l);
    auto arr5 = array<std::int8_t>::wrap(src5, 2l);

    chunked_array<std::int8_t> chunked2(3);
    chunked2.set_chunk(0l, arr3);
    chunked2.set_chunk(1l, arr4);
    chunked2.set_chunk(2l, arr5);
    ONEDAL_ASSERT(chunked2.get_count() == 5l);

    constexpr data_type dtypes[] = { data_type::float32, data_type::int8 };
    constexpr feature_type ftypes[] = { feature_type::nominal, feature_type::ratio };

    const table_metadata meta{ array<data_type>::wrap(dtypes, 2l),
                               array<feature_type>::wrap(ftypes, 2l) };

    auto table = heterogen_table::empty(meta);

    table.set_column(0l, chunked1);
    table.set_column(1l, chunked2);

    REQUIRE(table.has_data() == true);
    REQUIRE(table.get_row_count() == 5l);
    REQUIRE(table.get_column_count() == 2l);
    REQUIRE(table.get_kind() == heterogen_table::kind());
}

TEST("Can get row slice on host - 1") {
    constexpr float src1[] = { 0.f, 2.f, 4.f };
    constexpr float src2[] = { 6.f, 8.f, 10.f };

    auto arr1 = array<float>::wrap(src1, 3l);
    auto arr2 = array<float>::wrap(src2, 3l);

    chunked_array<float> chunked1(2);
    chunked1.set_chunk(0l, arr1);
    chunked1.set_chunk(1l, arr2);
    ONEDAL_ASSERT(chunked1.get_count() == 6l);

    constexpr std::int16_t src3[] = { 1 };
    constexpr std::int16_t src4[] = { 3, 5 };
    constexpr std::int16_t src5[] = { 7, 9, 11 };

    auto arr3 = array<std::int16_t>::wrap(src3, 1l);
    auto arr4 = array<std::int16_t>::wrap(src4, 2l);
    auto arr5 = array<std::int16_t>::wrap(src5, 3l);

    chunked_array<std::int16_t> chunked2(3);
    chunked2.set_chunk(0l, arr3);
    chunked2.set_chunk(1l, arr4);
    chunked2.set_chunk(2l, arr5);
    ONEDAL_ASSERT(chunked2.get_count() == 6l);

    auto table = heterogen_table::wrap( //
        chunked1,
        chunked2);

    row_accessor<const float> accessor{ table };
    auto res = accessor.pull(range{ 1l, 5l });
    REQUIRE(res.get_count() == 8l);

    for (std::int64_t i = 0l; i < 8l; ++i) {
        REQUIRE(res[i] == float(i + 2l));
    }
}

TEST("Can get row slice on host - 2") {
    const std::int64_t count = GENERATE(777, 1027, 1029, 7777);

    std::vector<std::uint64_t> column0(count);
    std::iota(column0.begin(), column0.end(), 0ul);
    auto arr0 = array<std::uint64_t>::wrap(column0.data(), count);

    std::vector<float> column1(count);
    std::iota(column1.begin(), column1.end(), 1.f);
    auto arr1 = array<float>::wrap(column1.data(), count);

    std::vector<std::int32_t> column2(count);
    std::iota(column2.begin(), column2.end(), 2);
    auto arr2 = array<std::int32_t>::wrap(column2.data(), count);

    std::vector<std::int64_t> column3(count);
    std::iota(column3.begin(), column3.end(), 3l);
    auto arr3 = array<std::int64_t>::wrap(column3.data(), count);

    std::vector<std::int16_t> column4(count);
    std::iota(column4.begin(), column4.end(), 4l);
    auto arr4 = array<std::int16_t>::wrap(column4.data(), count);

    auto table = heterogen_table::wrap(chunked_array<std::uint64_t>(arr0),
                                       chunked_array<float>(arr1),
                                       chunked_array<std::int32_t>(arr2),
                                       chunked_array<std::int64_t>(arr3),
                                       chunked_array<std::int16_t>(arr4));

    row_accessor<const float> accessor{ table };
    auto res = accessor.pull(range{ 3l, count - 3l });
    const std::int64_t slice_size = 5l * (count - 6l);
    REQUIRE(slice_size == res.get_count());

    for (std::int64_t i = 0l; i < slice_size; ++i) {
        const auto row = 3l + i / 5l;
        const auto col = i % 5l;
        const auto gtr = row + col;
        REQUIRE(res[i] == gtr);
    }
}

TEST("Can get column slice on host") {
    const std::int64_t count = GENERATE(201, 678, 999);

    std::vector<float> column0(count);
    std::iota(column0.begin(), column0.end(), 0);
    auto arr0 = array<float>::wrap(column0.data(), count);

    std::vector<double> column1(count);
    std::iota(column1.begin(), column1.end(), 0);
    auto arr1 = array<double>::wrap(column1.data(), count);

    std::vector<std::int64_t> column2(count);
    std::iota(column2.begin(), column2.end(), 0);
    auto arr2 = array<std::int64_t>::wrap(column2.data(), count);

    std::vector<std::uint16_t> column3(count);
    std::iota(column3.begin(), column3.end(), 0);
    auto arr3 = array<std::uint16_t>::wrap(column3.data(), count);

    auto table = heterogen_table::wrap(chunked_array<float>(arr0),
                                       chunked_array<double>(arr1),
                                       chunked_array<std::int64_t>(arr2),
                                       chunked_array<std::int16_t>(arr3));

    column_accessor<const float> accessor{ table };

    auto check_column = [&](auto c, auto f, auto l, const auto& arr) {
        const std::int64_t range = l - f;
        REQUIRE(range == arr.get_count());

        const auto* const arr_ptr = arr.get_data();
        for (std::int64_t i = 0l; i < range; ++i) {
            const auto gtr = float(f + i);
            const auto val = arr_ptr[i];
            CAPTURE(i, f, c, gtr, val);
            REQUIRE(gtr == val);
        }
    };

    for (std::int64_t col = 0l; col < table.get_column_count(); ++col) {
        const auto first = 3 * col, last = count - 4 * col;
        auto res = accessor.pull(col, { first, last });
        check_column(col, first, last, res);
    }
}

#ifdef ONEDAL_DATA_PARALLEL

TEST("Can get row slice from host to shared") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr auto host = sycl::usm::alloc::host;
    constexpr auto device = sycl::usm::alloc::device;

    constexpr float src1[] = { 0.f, 2.f, 4.f };
    constexpr float src2[] = { 6.f, 8.f, 10.f };

    auto arr1 = array<float>::wrap(src1, 3l);
    auto arr2 = array<float>::wrap(src2, 3l);

    chunked_array<float> chunked1(2);
    chunked1.set_chunk(0l, arr1);
    chunked1.set_chunk(1l, arr2);
    ONEDAL_ASSERT(chunked1.get_count() == 6l);

    constexpr std::int16_t src3[] = { 1 };
    constexpr std::int16_t src4[] = { 3, 5 };
    constexpr std::int16_t src5[] = { 7, 9, 11 };

    auto arr3 = array<std::int16_t>::wrap(src3, 1l);
    auto arr4 = array<std::int16_t>::wrap(src4, 2l);
    auto arr5 = array<std::int16_t>::wrap(src5, 3l);

    chunked_array<std::int16_t> chunked2(3);
    chunked2.set_chunk(0l, arr3);
    chunked2.set_chunk(1l, arr4);
    chunked2.set_chunk(2l, arr5);
    ONEDAL_ASSERT(chunked2.get_count() == 6l);

    auto table = heterogen_table::wrap( //
        chunked1,
        chunked2);

    row_accessor<const float> accessor{ table };
    auto tmp = accessor.pull(q, { 1l, 5l }, device);
    REQUIRE(tmp.get_count() == std::int64_t(8l));

    auto res = array<float>::empty(q, 8l, host);
    /* Copying to host */ detail::copy(res, tmp);

    for (std::int64_t i = 0l; i < 8l; ++i) {
        REQUIRE(res[i] == float(i + 2l));
    }
}

TEST("Can get row slice from heterogen to shared") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr auto device = sycl::usm::alloc::device;
    constexpr auto shared = sycl::usm::alloc::shared;

    auto arr1 = array<float>::empty(q, 10, shared);
    std::iota(begin(arr1), end(arr1), float(0));
    auto arr2 = array<float>::empty(q, 10, shared);
    std::iota(begin(arr2), end(arr2), float(10));
    chunked_array<float> chunked1(arr1, arr2);

    auto arr3 = array<std::int8_t>::empty(q, 20, shared);
    std::iota(begin(arr3), end(arr3), std::int8_t(0));
    chunked_array<std::int8_t> chunked2(arr3);

    // All columns of a heterogen table must share the same allocation kind.
    auto arr4 = array<std::uint16_t>::empty(q, 20, shared);
    std::iota(begin(arr4), end(arr4), std::uint16_t(0));
    chunked_array<std::uint16_t> chunked3(arr4);

    auto arr5 = array<std::int32_t>::empty(q, 20, shared);
    std::iota(begin(arr5), end(arr5), std::int32_t(0l));
    chunked_array<std::int32_t> chunked4(arr5);

    auto table = heterogen_table::wrap( //
        chunked1,
        chunked2,
        chunked3,
        chunked4);

    row_accessor<const float> accessor{ table };
    auto tmp = accessor.pull(q, { 1l, 19l }, device);
    REQUIRE(tmp.get_count() == 4l * 18l);

    auto res = array<float>::empty(4l * 18l);
    /* Copying to host */ detail::copy(res, tmp);

    for (std::int64_t i = 0l; i < res.get_count(); ++i) {
        const std::int64_t val = i / 4l + 1l;
        CAPTURE(i, val, res[i]);
        REQUIRE(res[i] == float(val));
    }
}

TEST("Can get column slice from heterogen to shared") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr auto device = sycl::usm::alloc::device;
    constexpr auto shared = sycl::usm::alloc::shared;

    const std::int64_t count = GENERATE(77, 177, 777);

    auto arr0 = array<float>::empty(q, count, shared);
    std::iota(begin(arr0), end(arr0), float(0));
    chunked_array<float> chunked0(arr0);

    auto arr1 = array<std::int16_t>::empty(q, count, shared);
    std::iota(begin(arr1), end(arr1), std::int16_t(0));
    chunked_array<std::int16_t> chunked1(arr1);

    // All columns of a heterogen table must share the same allocation kind.
    auto arr2 = array<std::uint16_t>::empty(q, count, shared);
    std::iota(begin(arr2), end(arr2), std::uint16_t(0));
    chunked_array<std::uint16_t> chunked2(arr2);

    auto arr3 = array<std::int32_t>::empty(q, count, shared);
    std::iota(begin(arr3), end(arr3), std::int32_t(0l));
    chunked_array<std::int32_t> chunked3(arr3);

    auto table = heterogen_table::wrap( //
        chunked0,
        chunked1,
        chunked2,
        chunked3);

    column_accessor<const float> accessor{ table };

    auto check_column = [&](auto c, auto f, auto l, const auto& arr) {
        const std::int64_t range = l - f;
        REQUIRE(range == arr.get_count());

        const auto* const arr_ptr = arr.get_data();
        for (std::int64_t i = 0l; i < range; ++i) {
            const auto gtr = float(f + i);
            const auto val = arr_ptr[i];
            CAPTURE(i, f, c, gtr, val);
            REQUIRE(gtr == val);
        }
    };

    for (std::int64_t col = 0l; col < table.get_column_count(); ++col) {
        const std::int64_t first = 2 * col, last = count - 2 * col;
        auto tmp = accessor.pull(q, col, { first, last }, device);
        auto res = array<float>::zeros(tmp.get_count());
        /* Copying to host */ detail::copy(res, tmp);

        check_column(col, first, last, res);
    }
}

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

TEST("Cannot build chunked array from chunks with different alloc kinds") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr auto device = sycl::usm::alloc::device;
    constexpr auto shared = sycl::usm::alloc::shared;

    auto arr0 = array<float>::empty(q, 4l, shared);
    std::iota(begin(arr0), end(arr0), float(0));
    auto arr1 = array<float>::empty(q, 4l, device);

    chunked_array<float> chunked(2);
    chunked.set_chunk(0l, arr0);
    REQUIRE_THROWS_AS(chunked.set_chunk(1l, arr1), invalid_argument);
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::test
