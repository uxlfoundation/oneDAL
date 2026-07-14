/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::test::engine {

inline void check_if_metadata_equal(const table_metadata& actual, const table_metadata& reference) {
    REQUIRE(actual.get_feature_count() == reference.get_feature_count());
    for (std::int64_t i = 0; i < reference.get_feature_count(); i++) {
        REQUIRE(actual.get_feature_type(i) == reference.get_feature_type(i));
        REQUIRE(actual.get_data_type(i) == reference.get_data_type(i));
    }
}

template <typename Data>
inline void check_if_table_content_equal(const table& actual, const table& reference) {
    const auto actual_ary = row_accessor<const Data>{ actual }.pull();
    const auto reference_ary = row_accessor<const Data>{ reference }.pull();

    for (std::int64_t i = 0; i < reference_ary.get_count(); i++) {
        const Data actual = actual_ary[i];
        const Data reference = reference_ary[i];
        if (actual != reference) {
            CAPTURE(i, actual, reference);
            FAIL("Found elements mismatch in tables");
            break;
        }
    }
}

template <typename Float>
inline void check_if_table_content_equal_approx(const table& actual,
                                                const table& reference,
                                                double tolerance) {
    static_assert(std::is_floating_point_v<Float>);
    const auto actual_ary = row_accessor<const Float>{ actual }.pull();
    const auto reference_ary = row_accessor<const Float>{ reference }.pull();

    for (std::int64_t i = 0; i < reference_ary.get_count(); i++) {
        const Float actual = actual_ary[i];
        const Float reference = reference_ary[i];

        const double div = std::max(std::abs(actual), std::abs(reference));
        const double relative_error =
            (div > tolerance) ? (std::abs(double(actual) - double(reference)) / div) : 0.0;

        if (relative_error > tolerance) {
            CAPTURE(i, actual, reference, relative_error);
            FAIL("Found elements mismatch in tables");
            break;
        }
    }
}

template <typename Data>
inline void check_if_tables_equal(const table& actual, const table& reference) {
    REQUIRE(actual.get_row_count() == reference.get_row_count());
    REQUIRE(actual.get_column_count() == reference.get_column_count());
    REQUIRE(actual.get_data_layout() == reference.get_data_layout());
    REQUIRE(actual.get_kind() == reference.get_kind());

    check_if_metadata_equal(actual.get_metadata(), reference.get_metadata());
    check_if_table_content_equal<Data>(actual, reference);
}

template <typename Float>
inline void check_if_tables_equal_approx(const table& actual,
                                         const table& reference,
                                         double tolerance) {
    REQUIRE(actual.get_row_count() == reference.get_row_count());
    REQUIRE(actual.get_column_count() == reference.get_column_count());
    REQUIRE(actual.get_data_layout() == reference.get_data_layout());
    REQUIRE(actual.get_kind() == reference.get_kind());

    check_if_metadata_equal(actual.get_metadata(), reference.get_metadata());
    check_if_table_content_equal_approx<Float>(actual, reference, tolerance);
}

template <typename T>
inline array<T> get_table_block(host_test_policy&, const table& t, const range& row_range) {
    return row_accessor<const T>{ t }.pull(row_range);
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename T>
inline array<T> get_table_block(device_test_policy& p,
                                const table& t,
                                const range& row_range,
                                const sycl::usm::alloc alloc = sycl::usm::alloc::device) {
    return row_accessor<const T>{ t }.pull(p.get_queue(), row_range, alloc);
}

#endif

template <typename Float, typename TestPolicy>
inline std::vector<table> split_table_by_rows(TestPolicy& policy,
                                              const table& t,
                                              std::int64_t split_count) {
    ONEDAL_ASSERT(split_count > 0);

    const std::int64_t row_count = t.get_row_count();
    const std::int64_t column_count = t.get_column_count();
    const std::int64_t block_size_regular = row_count / split_count;
    const std::int64_t block_size_tail = row_count % split_count;

    std::vector<table> result(split_count);

    std::int64_t row_offset = 0;
    for (std::int64_t i = 0; i < split_count; i++) {
        const std::int64_t tail = std::int64_t(i + 1 == split_count) * block_size_tail;
        const std::int64_t block_size = block_size_regular + tail;

        if (block_size > 0) {
            const auto row_range = range{ row_offset, row_offset + block_size };
            const auto block = get_table_block<Float>(policy, t, row_range);
            result[i] = homogen_table::wrap(block, block_size, column_count);
        }
        else {
            result[i] = homogen_table{};
        }
        row_offset += block_size;
    }

    return result;
}

#ifdef ONEDAL_DATA_PARALLEL

inline sycl::usm::alloc get_alloc(std::int64_t i, std::int64_t split_count, bool host_first) {
    sycl::usm::alloc alloc;
    if (host_first) {
        if (i < split_count / 2) {
            alloc = sycl::usm::alloc::host;
        }
        else {
            alloc = sycl::usm::alloc::device;
        }
    }
    else {
        if (i < split_count / 2) {
            alloc = sycl::usm::alloc::device;
        }
        else {
            alloc = sycl::usm::alloc::host;
        }
    }
    return alloc;
}

inline std::string get_alloc_name(const void* ptr, const sycl::context& ctx) {
    std::string alloc_name;
    const sycl::usm::alloc alloc = sycl::get_pointer_type(ptr, ctx);
    switch (alloc) {
        case sycl::usm::alloc::host: alloc_name = "host"; break;
        case sycl::usm::alloc::device: alloc_name = "device"; break;
        case sycl::usm::alloc::shared: alloc_name = "shared"; break;
        default: alloc_name = "non-usm"; break;
    }
    return alloc_name;
}

template <typename Float, typename TestPolicy>
inline std::vector<table> split_table_by_rows_mixed(TestPolicy& policy,
                                                    const table& t,
                                                    std::int64_t split_count,
                                                    bool host_first = true) {
    ONEDAL_ASSERT(split_count > 0);
    std::cout << "in split_table_by_rows_mixed " << std::endl;

    const std::int64_t row_count = t.get_row_count();
    const std::int64_t column_count = t.get_column_count();
    const std::int64_t block_size_regular = row_count / split_count;
    const std::int64_t block_size_tail = row_count % split_count;

    std::vector<table> result(split_count);

    std::int64_t row_offset = 0;
    for (std::int64_t i = 0; i < split_count; i++) {
        const std::int64_t tail = std::int64_t(i + 1 == split_count) * block_size_tail;
        const std::int64_t block_size = block_size_regular + tail;

        if (block_size > 0) {
            const auto row_range = range{ row_offset, row_offset + block_size };

            const sycl::usm::alloc alloc = get_alloc(i, split_count, host_first);
            const array<Float> block = get_table_block<Float>(policy, t, row_range, alloc);

            std::cout << "input block " << i << " allocation: "
                      << get_alloc_name(block.get_data(), policy.get_queue().get_context())
                      << std::endl;
            result[i] = homogen_table::wrap(block, block_size, column_count);
        }
        else {
            result[i] = homogen_table{};
        }
        row_offset += block_size;
    }

    return result;
}

#endif

template <typename Float>
inline table stack_tables_by_rows(const std::vector<table>& tables) {
    if (tables.empty()) {
        return table{};
    }

    std::int64_t total_row_count = 0;
    std::int64_t total_column_count = tables[0].get_column_count();
    for (const auto& t : tables) {
        ONEDAL_ASSERT(t.has_data());
        ONEDAL_ASSERT(t.get_column_count() == total_column_count);
        total_row_count += t.get_row_count();
    }

    const auto stacked_table_memory = dal::array<Float>::empty(
        dal::detail::check_mul_overflow(total_row_count, total_column_count));

    std::int64_t offset = 0;
    for (const auto& t : tables) {
        const auto t_ary = row_accessor<const Float>{ t }.pull();
        Float* dst_ptr = stacked_table_memory.get_mutable_data() + offset;
        dal::detail::memcpy(dal::detail::default_host_policy{},
                            dst_ptr,
                            t_ary.get_data(),
                            t_ary.get_size());
        offset += t_ary.get_count();
    }

    return homogen_table::wrap(stacked_table_memory, total_row_count, total_column_count);
}

} // namespace oneapi::dal::test::engine
