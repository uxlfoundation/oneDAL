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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/linalg.hpp"
#include "oneapi/dal/test/engine/tables.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

TEST("can construct empty table") {
    csr_table t;

    REQUIRE(t.has_data() == false);
    REQUIRE(t.get_kind() == csr_table::kind());
    REQUIRE(t.get_row_count() == 0);
    REQUIRE(t.get_column_count() == 0);
}

TEST("can construct CSR table from raw data pointers") {
    using oneapi::dal::detail::empty_delete;

    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };

    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 7 };

    csr_table t{ data,
                 column_indices,
                 row_offsets,
                 row_count,
                 column_count,
                 empty_delete<const float>(),
                 empty_delete<const std::int64_t>(),
                 empty_delete<const std::int64_t>() };

    REQUIRE(t.has_data());
    REQUIRE(t.get_row_count() == row_count);
    REQUIRE(t.get_column_count() == column_count);
    REQUIRE(t.get_non_zero_count() == element_count);

    REQUIRE(t.get_indexing() == sparse_indexing::one_based);

    const auto& meta = t.get_metadata();
    for (std::int64_t i = 0; i < t.get_column_count(); i++) {
        REQUIRE(meta.get_data_type(i) == data_type::float32);
        REQUIRE(meta.get_feature_type(i) == feature_type::ratio);
    }
    REQUIRE(meta.get_alloc_kind() == alloc_kind::non_usm);

    REQUIRE(t.get_data<float>() == data);
    REQUIRE(t.get_column_indices() == column_indices);
    REQUIRE(t.get_row_offsets() == row_offsets);
}

TEST("can construct float64 table with zero-based indexing") {
    using oneapi::dal::detail::empty_delete;

    const double data[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 11.0, 8.0 };
    const std::int64_t column_indices[] = { 0, 1, 3, 2, 1, 3, 1 };
    const std::int64_t row_offsets[] = { 0, 3, 4, 6, 7 };

    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 7 };

    csr_table t{ data,
                 column_indices,
                 row_offsets,
                 row_count,
                 column_count,
                 empty_delete<const double>(),
                 empty_delete<const std::int64_t>(),
                 empty_delete<const std::int64_t>(),
                 sparse_indexing::zero_based };

    REQUIRE(t.has_data());
    REQUIRE(t.get_row_count() == row_count);
    REQUIRE(t.get_column_count() == column_count);
    REQUIRE(t.get_non_zero_count() == element_count);

    REQUIRE(t.get_indexing() == sparse_indexing::zero_based);

    const auto& meta = t.get_metadata();
    for (std::int64_t i = 0; i < t.get_column_count(); i++) {
        REQUIRE(meta.get_data_type(i) == data_type::float64);
        REQUIRE(meta.get_feature_type(i) == feature_type::ratio);
    }
    REQUIRE(meta.get_alloc_kind() == alloc_kind::non_usm);

    REQUIRE(t.get_data<double>() == data);
    REQUIRE(t.get_column_indices() == column_indices);
    REQUIRE(t.get_row_offsets() == row_offsets);
}

TEST("can construct table reference") {
    using oneapi::dal::detail::empty_delete;

    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };

    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };

    csr_table t1{ data,
                  column_indices,
                  row_offsets,
                  row_count,
                  column_count,
                  empty_delete<const float>(),
                  empty_delete<const std::int64_t>(),
                  empty_delete<const std::int64_t>() };
    csr_table t2 = t1;

    REQUIRE(t1.has_data());
    REQUIRE(t2.has_data());

    REQUIRE(t1.get_row_count() == t2.get_row_count());
    REQUIRE(t1.get_column_count() == t2.get_column_count());
    REQUIRE(t1.get_non_zero_count() == t2.get_non_zero_count());
    REQUIRE(t1.get_data<float>() == t2.get_data<float>());
    REQUIRE(t1.get_data<float>() == data);
    REQUIRE(t2.get_data<float>() == data);
    REQUIRE(t1.get_column_indices() == t2.get_column_indices());
    REQUIRE(t1.get_column_indices() == column_indices);
    REQUIRE(t2.get_column_indices() == column_indices);
    REQUIRE(t1.get_row_offsets() == t2.get_row_offsets());
    REQUIRE(t1.get_row_offsets() == row_offsets);
    REQUIRE(t2.get_row_offsets() == row_offsets);

    const auto& m1 = t1.get_metadata();
    const auto& m2 = t2.get_metadata();

    REQUIRE(t1.get_indexing() == t2.get_indexing());

    te::check_if_metadata_equal(m1, m2, true);
}

TEST("can construct table reference with a copy constructor") {
    using oneapi::dal::detail::empty_delete;

    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };

    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };

    csr_table t1{ data,
                  column_indices,
                  row_offsets,
                  row_count,
                  column_count,
                  empty_delete<const float>(),
                  empty_delete<const std::int64_t>(),
                  empty_delete<const std::int64_t>() };
    csr_table t2(t1);

    REQUIRE(t1.has_data());
    REQUIRE(t2.has_data());

    REQUIRE(t1.get_row_count() == t2.get_row_count());
    REQUIRE(t1.get_column_count() == t2.get_column_count());
    REQUIRE(t1.get_non_zero_count() == t2.get_non_zero_count());
    REQUIRE(t1.get_data<float>() == t2.get_data<float>());
    REQUIRE(t1.get_data<float>() == data);
    REQUIRE(t2.get_data<float>() == data);
    REQUIRE(t1.get_column_indices() == t2.get_column_indices());
    REQUIRE(t1.get_column_indices() == column_indices);
    REQUIRE(t2.get_column_indices() == column_indices);
    REQUIRE(t1.get_row_offsets() == t2.get_row_offsets());
    REQUIRE(t1.get_row_offsets() == row_offsets);
    REQUIRE(t2.get_row_offsets() == row_offsets);

    const auto& m1 = t1.get_metadata();
    const auto& m2 = t2.get_metadata();

    REQUIRE(t1.get_indexing() == t2.get_indexing());

    te::check_if_metadata_equal(m1, m2, true);
}

TEST("can construct table with move") {
    using oneapi::dal::detail::empty_delete;

    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };

    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 7 };

    csr_table t1{ data,
                  column_indices,
                  row_offsets,
                  row_count,
                  column_count,
                  empty_delete<const float>(),
                  empty_delete<const std::int64_t>(),
                  empty_delete<const std::int64_t>() };
    csr_table t2 = std::move(t1);

    REQUIRE(t1.has_data() == false);
    REQUIRE(t2.has_data() == true);

    REQUIRE(t2.get_row_count() == row_count);
    REQUIRE(t2.get_column_count() == column_count);
    REQUIRE(t2.get_non_zero_count() == element_count);

    const auto& m2 = t2.get_metadata();

    REQUIRE(t2.get_indexing() == sparse_indexing::one_based);
    for (std::int64_t i = 0; i < t2.get_column_count(); i++) {
        REQUIRE(m2.get_data_type(i) == data_type::float32);
    }
    REQUIRE(m2.get_alloc_kind() == alloc_kind::non_usm);
    REQUIRE(t2.get_data<float>() == data);
    REQUIRE(t2.get_column_indices() == column_indices);
    REQUIRE(t2.get_row_offsets() == row_offsets);
}

TEST("can assign two table references") {
    using oneapi::dal::detail::empty_delete;

    const float data1[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices1[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets1[] = { 1, 4, 5, 7, 8 };

    constexpr std::int64_t row_count1{ 4 };
    constexpr std::int64_t column_count1{ 4 };

    const std::int32_t data2[] = { 1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, -5 };
    const std::int64_t column_indices2[] = { 1, 2, 4, 1, 2, 3, 4, 5, 1, 3, 4, 2, 5 };
    const std::int64_t row_offsets2[] = { 1, 4, 6, 9, 12, 14 };

    constexpr std::int64_t row_count2{ 5 };
    constexpr std::int64_t column_count2{ 5 };
    constexpr std::int64_t element_count2{ 13 };

    csr_table t1{ data1,
                  column_indices1,
                  row_offsets1,
                  row_count1,
                  column_count1,
                  empty_delete<const float>(),
                  empty_delete<const std::int64_t>(),
                  empty_delete<const std::int64_t>() };
    csr_table t2{ data2,
                  column_indices2,
                  row_offsets2,
                  row_count2,
                  column_count2,
                  empty_delete<const std::int32_t>(),
                  empty_delete<const std::int64_t>(),
                  empty_delete<const std::int64_t>() };

    t1 = t2;

    REQUIRE(t1.has_data() == true);
    REQUIRE(t2.has_data() == true);

    REQUIRE(t1.get_row_count() == row_count2);
    REQUIRE(t1.get_column_count() == column_count2);
    REQUIRE(t1.get_non_zero_count() == element_count2);
    REQUIRE(t1.get_data<std::int32_t>() == data2);
    REQUIRE(t1.get_column_indices() == column_indices2);
    REQUIRE(t1.get_row_offsets() == row_offsets2);

    REQUIRE(t2.get_row_count() == row_count2);
    REQUIRE(t2.get_column_count() == column_count2);
    REQUIRE(t2.get_non_zero_count() == element_count2);
    REQUIRE(t2.get_data<std::int32_t>() == data2);
    REQUIRE(t2.get_column_indices() == column_indices2);
    REQUIRE(t2.get_row_offsets() == row_offsets2);

    te::check_if_metadata_equal(t1.get_metadata(), t2.get_metadata(), true);
}

TEST("can move assigned table reference") {
    using oneapi::dal::detail::empty_delete;

    const float data1[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices1[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets1[] = { 1, 4, 5, 7, 8 };

    constexpr std::int64_t row_count1{ 4 };
    constexpr std::int64_t column_count1{ 4 };

    const std::int32_t data2[] = { 1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, -5 };
    const std::int64_t column_indices2[] = { 1, 2, 4, 1, 2, 3, 4, 5, 1, 3, 4, 2, 5 };
    const std::int64_t row_offsets2[] = { 1, 4, 6, 9, 12, 14 };

    constexpr std::int64_t row_count2{ 5 };
    constexpr std::int64_t column_count2{ 5 };
    constexpr std::int64_t element_count2{ 13 };

    csr_table t1{ data1,
                  column_indices1,
                  row_offsets1,
                  row_count1,
                  column_count1,
                  empty_delete<const float>(),
                  empty_delete<const std::int64_t>(),
                  empty_delete<const std::int64_t>() };
    csr_table t2{ data2,
                  column_indices2,
                  row_offsets2,
                  row_count2,
                  column_count2,
                  empty_delete<const std::int32_t>(),
                  empty_delete<const std::int64_t>(),
                  empty_delete<const std::int64_t>() };

    t1 = std::move(t2);

    REQUIRE(t1.get_metadata().get_alloc_kind() == alloc_kind::non_usm);
    REQUIRE(t1.has_data() == true);
    REQUIRE(t1.get_row_count() == row_count2);
    REQUIRE(t1.get_column_count() == column_count2);
    REQUIRE(t1.get_non_zero_count() == element_count2);
    REQUIRE(t1.get_metadata().get_data_type(0) == data_type::int32);
    REQUIRE(t1.get_data<std::int32_t>() == data2);
    REQUIRE(t1.get_column_indices() == column_indices2);
    REQUIRE(t1.get_row_offsets() == row_offsets2);
}

TEST(
    "can construct table from data pointers and share the ownership of the data with those pointers") {
    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };

    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 7 };

    csr_table t = csr_table::wrap(data, column_indices, row_offsets, row_count, column_count);

    REQUIRE(t.has_data() == true);
    REQUIRE(t.get_row_count() == row_count);
    REQUIRE(t.get_column_count() == column_count);
    REQUIRE(t.get_non_zero_count() == element_count);
    REQUIRE(csr_table::kind() == t.get_kind());

    REQUIRE(t.get_indexing() == sparse_indexing::one_based);

    const auto& meta = t.get_metadata();
    for (std::int64_t i = 0; i < t.get_column_count(); i++) {
        REQUIRE(meta.get_data_type(i) == data_type::float32);
        REQUIRE(meta.get_feature_type(i) == feature_type::ratio);
    }
    REQUIRE(meta.get_alloc_kind() == alloc_kind::non_usm);

    REQUIRE(t.get_data<float>() == data);
    REQUIRE(t.get_column_indices() == column_indices);
    REQUIRE(t.get_row_offsets() == row_offsets);
}

#ifdef ONEDAL_DATA_PARALLEL

TEST((std::string("can construct table from data pointers allocated on the device") +
      std::string(" and share the ownership of the data with those pointers"))
         .c_str()) {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 7 };

    const float data_host[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices_host[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets_host[] = { 1, 4, 5, 7, 8 };

    auto* const data = sycl::malloc_shared<float>(element_count, q);
    auto* const column_indices = sycl::malloc_shared<std::int64_t>(element_count, q);
    auto* const row_offsets = sycl::malloc_shared<std::int64_t>(row_count + 1, q);

    auto data_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(data, data_host, element_count * sizeof(float));
    });

    auto column_indices_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(column_indices, column_indices_host, element_count * sizeof(std::int64_t));
    });

    auto row_offsets_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(row_offsets, row_offsets_host, (row_count + 1) * sizeof(std::int64_t));
    });

    auto t = csr_table::wrap(q,
                             data,
                             column_indices,
                             row_offsets,
                             row_count,
                             column_count,
                             sparse_indexing::one_based,
                             { data_event, column_indices_event, row_offsets_event });

    REQUIRE(t.has_data() == true);
    REQUIRE(t.get_row_count() == row_count);
    REQUIRE(t.get_column_count() == column_count);
    REQUIRE(t.get_non_zero_count() == element_count);
    REQUIRE(t.get_kind() == csr_table::kind());

    REQUIRE(t.get_indexing() == sparse_indexing::one_based);

    const auto& meta = t.get_metadata();
    for (std::int64_t i = 0; i < t.get_column_count(); i++) {
        REQUIRE(meta.get_data_type(i) == data_type::float32);
        REQUIRE(meta.get_feature_type(i) == feature_type::ratio);
    }
    REQUIRE(meta.get_alloc_kind() == alloc_kind::usm_shared);

    REQUIRE(t.get_data<float>() == data);
    REQUIRE(t.get_column_indices() == column_indices);
    REQUIRE(t.get_row_offsets() == row_offsets);

    sycl::free(data, q);
    sycl::free(column_indices, q);
    sycl::free(row_offsets, q);
}

#endif

TEST("can construct table from arrays and share the ownership of the data with those arrays") {
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 7 };
    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };
    const auto data_array = array<float>::wrap(data, element_count);
    const auto column_indices_array = array<std::int64_t>::wrap(column_indices, element_count);
    const auto row_offsets_array = array<std::int64_t>::wrap(row_offsets, row_count + 1);

    auto t = csr_table::wrap(data_array, column_indices_array, row_offsets_array, column_count);

    REQUIRE(t.get_data<float>() == data);
    REQUIRE(t.get_column_indices() == column_indices);
    REQUIRE(t.get_row_offsets() == row_offsets);
    REQUIRE(t.get_row_count() == row_count);
    REQUIRE(t.get_column_count() == column_count);
    REQUIRE(t.get_non_zero_count() == element_count);
}

#ifdef ONEDAL_DATA_PARALLEL

TEST((std::string("can construct table from arrays holding the data allocated on the device") +
      std::string(" and share the ownership of the data with those arrays"))
         .c_str()) {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 7 };

    const float data_host[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices_host[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets_host[] = { 1, 4, 5, 7, 8 };

    auto* const data = sycl::malloc_device<float>(element_count, q);
    auto* const column_indices = sycl::malloc_device<std::int64_t>(element_count, q);
    auto* const row_offsets = sycl::malloc_device<std::int64_t>(row_count + 1, q);

    auto data_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(data, data_host, element_count * sizeof(float));
    });

    auto column_indices_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(column_indices, column_indices_host, element_count * sizeof(std::int64_t));
    });

    auto row_offsets_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(row_offsets, row_offsets_host, (row_count + 1) * sizeof(std::int64_t));
    });

    const auto data_array = array<float>::wrap(q, data, element_count, { data_event });
    const auto column_indices_array =
        array<std::int64_t>::wrap(q, column_indices, element_count, { column_indices_event });
    const auto row_offsets_array =
        array<std::int64_t>::wrap(q, row_offsets, row_count + 1, { row_offsets_event });

    auto t = csr_table::wrap(data_array, column_indices_array, row_offsets_array, column_count);

    REQUIRE(t.get_data<float>() == data);
    REQUIRE(t.get_column_indices() == column_indices);
    REQUIRE(t.get_row_offsets() == row_offsets);
    REQUIRE(t.get_row_count() == row_count);
    REQUIRE(t.get_column_count() == column_count);
    REQUIRE(t.get_non_zero_count() == element_count);
    REQUIRE(t.get_metadata().get_alloc_kind() == alloc_kind::usm_device);

    sycl::free(data, q);
    sycl::free(column_indices, q);
    sycl::free(row_offsets, q);
}

#endif

TEST("create table with invalid row or column count") {
    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };

    REQUIRE_NOTHROW(csr_table::wrap(data, column_indices, row_offsets, row_count, column_count));
    REQUIRE_THROWS_AS(csr_table::wrap(data, column_indices, row_offsets, 0, column_count),
                      dal::domain_error);
    REQUIRE_THROWS_AS(csr_table::wrap(data, column_indices, row_offsets, row_count, 0),
                      dal::domain_error);
}

TEST("create tables that have one-based indexing and various types of incorrect indices") {
    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t column_indices_lt_min[] = { 1, 2, 4, 3, 0, 4, 2 };
    const std::int64_t column_indices_gt_max[] = { 1, 2, 5, 3, 2, 4, 2 };
    const std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };
    const std::int64_t row_offsets_lt_min[] = { 0, 4, 5, 7, 8 };
    const std::int64_t row_offsets_gt_max[] = { 1, 4, 9, 7, 8 };
    const std::int64_t row_offsets_not_ascending[] = { 1, 4, 7, 5, 8 };
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };

    REQUIRE_THROWS_AS(
        csr_table::wrap(data, column_indices_lt_min, row_offsets, row_count, column_count),
        dal::domain_error);
    REQUIRE_THROWS_AS(
        csr_table::wrap(data, column_indices_gt_max, row_offsets, row_count, column_count),
        dal::domain_error);
    REQUIRE_THROWS_AS(
        csr_table::wrap(data, column_indices, row_offsets_lt_min, row_count, column_count),
        dal::domain_error);
    REQUIRE_THROWS_AS(
        csr_table::wrap(data, column_indices, row_offsets_gt_max, row_count, column_count),
        dal::domain_error);
    REQUIRE_THROWS_AS(
        csr_table::wrap(data, column_indices, row_offsets_not_ascending, row_count, column_count),
        dal::domain_error);
}

} // namespace oneapi::dal::test
