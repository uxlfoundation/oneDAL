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

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/tables.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;

TEST("can construct empty table") {
    homogen_table t;

    REQUIRE(t.has_data() == false);
    REQUIRE(t.get_kind() == homogen_table::kind());
    REQUIRE(t.get_row_count() == 0);
    REQUIRE(t.get_column_count() == 0);
}

TEST("can construct rowmajor table 3x2") {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    homogen_table t{ data, 3, 2, empty_delete<const float>() };

    REQUIRE(t.has_data());
    REQUIRE(t.get_row_count() == 3);
    REQUIRE(t.get_column_count() == 2);

    REQUIRE(t.get_data_layout() == data_layout::row_major);

    auto meta = t.get_metadata();
    for (std::int64_t i = 0; i < t.get_column_count(); i++) {
        REQUIRE(meta.get_data_type(i) == data_type::float32);
        REQUIRE(meta.get_feature_type(i) == feature_type::ratio);
    }
    REQUIRE(meta.get_alloc_kind() == alloc_kind::non_usm);

    REQUIRE(t.get_data<float>() == data);
    // TODO: now have no ctor to specify feature_type of the data
}

TEST("can construct colmajor float64 table") {
    using oneapi::dal::detail::empty_delete;

    double data[] = { 1., 2., 3., 4., 5., 6. };
    homogen_table t{ data, 2, 3, empty_delete<const double>(), data_layout::column_major };

    REQUIRE(t.has_data() == true);
    REQUIRE(t.get_row_count() == 2);
    REQUIRE(t.get_column_count() == 3);

    REQUIRE(t.get_data_layout() == data_layout::column_major);

    auto meta = t.get_metadata();
    for (std::int64_t i = 0; i < t.get_column_count(); i++) {
        REQUIRE(meta.get_data_type(i) == data_type::float64);
        REQUIRE(meta.get_feature_type(i) == feature_type::ratio);
    }
    REQUIRE(meta.get_alloc_kind() == alloc_kind::non_usm);

    REQUIRE(t.get_data<double>() == data);
}

TEST("can construct table reference") {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    homogen_table t1{ data, 3, 2, empty_delete<const float>() };
    homogen_table t2 = t1;

    REQUIRE(t1.has_data() == true);
    REQUIRE(t2.has_data() == true);

    REQUIRE(t1.get_row_count() == t2.get_row_count());
    REQUIRE(t1.get_column_count() == t2.get_column_count());
    REQUIRE(t1.get_data<float>() == t2.get_data<float>());
    REQUIRE(t1.get_data<float>() == data);
    REQUIRE(t2.get_data<float>() == data);

    const auto& m1 = t1.get_metadata();
    const auto& m2 = t2.get_metadata();

    REQUIRE(t1.get_data_layout() == t2.get_data_layout());

    te::check_if_metadata_equal(m1, m2, true);
}

TEST("can construct table with move") {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    homogen_table t1{ data, 3, 2, empty_delete<const float>() };
    homogen_table t2 = std::move(t1);

    REQUIRE(t1.has_data() == false);
    REQUIRE(t2.has_data() == true);

    REQUIRE(t2.get_row_count() == 3);
    REQUIRE(t2.get_column_count() == 2);

    const auto& m2 = t2.get_metadata();

    REQUIRE(t2.get_data_layout() == data_layout::row_major);
    REQUIRE(m2.get_data_type(0) == data_type::float32);
    REQUIRE(m2.get_data_type(1) == data_type::float32);
    REQUIRE(m2.get_alloc_kind() == alloc_kind::non_usm);
    REQUIRE(t2.get_data<float>() == data);
}

TEST("can assign two table references") {
    using oneapi::dal::detail::empty_delete;

    float data_float[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    std::int32_t data_int[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

    homogen_table t1{ data_float, 3, 2, empty_delete<const float>() };
    homogen_table t2{ data_int, 4, 3, empty_delete<const std::int32_t>() };

    t1 = t2;

    REQUIRE(t1.has_data() == true);
    REQUIRE(t2.has_data() == true);

    REQUIRE(t1.get_row_count() == 4);
    REQUIRE(t1.get_column_count() == 3);
    REQUIRE(t1.get_data<std::int32_t>() == data_int);

    REQUIRE(t2.get_row_count() == 4);
    REQUIRE(t2.get_column_count() == 3);
    REQUIRE(t2.get_data<std::int32_t>() == data_int);

    const auto& m1 = t1.get_metadata();
    const auto& m2 = t2.get_metadata();
    te::check_if_metadata_equal(m1, m2, true);
}

TEST("can move assigned table reference") {
    using oneapi::dal::detail::empty_delete;

    float data_float[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    std::int32_t data_int[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

    homogen_table t1{ data_float, 3, 2, empty_delete<const float>() };
    homogen_table t2{ data_int, 4, 3, empty_delete<const std::int32_t>() };

    t1 = std::move(t2);

    REQUIRE(t1.has_data() == true);
    REQUIRE(t1.get_row_count() == 4);
    REQUIRE(t1.get_column_count() == 3);
    REQUIRE(t1.get_metadata().get_data_type(0) == data_type::int32);
    REQUIRE(t1.get_data<std::int32_t>() == data_int);
}

TEST("can upcast table") {
    float data_float[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    table t = homogen_table::wrap(data_float, 3, 2);

    REQUIRE(t.has_data() == true);
    REQUIRE(t.get_row_count() == 3);
    REQUIRE(t.get_column_count() == 2);
    REQUIRE(t.get_metadata().get_data_type(0) == data_type::float32);
    REQUIRE(homogen_table::kind() == t.get_kind());
}

TEST("create table with invalid row or column count") {
    float data_float[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    REQUIRE_NOTHROW(homogen_table::wrap(data_float, 3, 2, data_layout::row_major));
    REQUIRE_THROWS_AS(homogen_table::wrap(data_float, -1, 2, data_layout::row_major),
                      dal::domain_error);
    REQUIRE_THROWS_AS(homogen_table::wrap(data_float, 3, 0, data_layout::row_major),
                      dal::domain_error);
    REQUIRE_THROWS_AS(homogen_table::wrap(data_float, 3, 2, data_layout::unknown),
                      dal::domain_error);
}

TEST("create table from array") {
    constexpr std::int64_t row_count = 3;
    constexpr std::int64_t column_count = 2;
    float data_float[row_count * column_count] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
    const auto data_array = array<float>::wrap(data_float, row_count * column_count);

    auto t = homogen_table::wrap(data_float, row_count, column_count);

    REQUIRE(t.get_data<float>() == data_float);
    REQUIRE(t.get_row_count() == row_count);
    REQUIRE(t.get_column_count() == column_count);
}

} // namespace oneapi::dal::test
