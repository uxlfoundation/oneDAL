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
#include "oneapi/dal/detail/debug.hpp"
#include "oneapi/dal/chunked_array.hpp"
#include "oneapi/dal/table/heterogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test {

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
        chunked1, chunked2, chunked1, chunked2);
//
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
            chunked1, chunked2, chunked1);
//
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

    const table_metadata meta{
        array<data_type>::wrap(dtypes, 2l),
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
    constexpr float src1[] = { 0.f, 2.f, 4.f};
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
            chunked1, chunked2);

    row_accessor<const float> accessor{ table };
    auto res = accessor.pull(range{1l, 5l});
    REQUIRE(res.get_count() == 8l);

    for (std::int64_t i = 0l; i < 8l; ++i) {
        REQUIRE(res[i] == float(i + 2l));
    }
}

TEST("Can get row slice on host - 2") {
    constexpr std::int64_t count = 4'097l;

    std::vector<std::uint64_t> column0(count);
    std::iota(column0.begin(), column0.end(), 0ul);
    auto arr0 = array<std::uint64_t>::wrap(column0.data(), count);

    std::vector<float> column1(count);
    std::iota(column1.begin(), column1.end(), 1.f);
    auto arr1 = array<float>::wrap(column1.data(), count);

    std::vector<double> column2(count);
    std::iota(column2.begin(), column2.end(), 2.0);
    auto arr2 = array<double>::wrap(column2.data(), count);

    std::vector<std::int64_t> column3(count);
    std::iota(column3.begin(), column3.end(), 3l);
    auto arr3 = array<std::int64_t>::wrap(column3.data(), count);

    auto table = heterogen_table::wrap(
        chunked_array<std::uint64_t>(arr0),
        chunked_array<float>(arr1),
        chunked_array<double>(arr2),
        chunked_array<std::int64_t>(arr3));

    row_accessor<const float> accessor{ table };
    auto res = accessor.pull(range{3l, count - 3l});
    const std::int64_t slice_size = 4l * (count - 6l);
    REQUIRE(slice_size == res.get_count());
    using namespace dal::detail;
    auto slc = res.get_slice(0, 12);
    std::cout << slc << std::endl;

    for (std::int64_t i = 0l; i < slice_size; ++i) {
        const auto row = 3l + i / 4l;
        const auto col = i % 4l;
        const auto gtr = row + col;
        REQUIRE(res[i] == gtr);
    }
}

} // namespace oneapi::dal::test
