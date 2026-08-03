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

#include <random>

namespace oneapi::dal::test::engine {

// NOTE: alloc_kind is compared only when check_alloc_kind is true, and is off by
// default. Allocation kind is a runtime property of where the data currently
// resides; it is not persisted in the archive and is re-established from the
// deserialization context (see the table impls' deserialize methods). After a
// context-agnostic round-trip the deserialized kind reflects the reading context,
// which may differ from the original, so callers must opt in only when the
// contexts are known to match.
inline void check_if_metadata_equal(const table_metadata& actual,
                                    const table_metadata& reference,
                                    bool check_alloc_kind = false) {
    REQUIRE(actual.get_feature_count() == reference.get_feature_count());
    for (std::int64_t i = 0; i < reference.get_feature_count(); i++) {
        REQUIRE(actual.get_feature_type(i) == reference.get_feature_type(i));
        REQUIRE(actual.get_data_type(i) == reference.get_data_type(i));
    }
    if (check_alloc_kind) {
        REQUIRE(actual.get_alloc_kind() == reference.get_alloc_kind());
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

sycl::usm::alloc get_sycl_alloc_kind(alloc_kind alloc) {
    switch (alloc) {
        case alloc_kind::usm_host: return sycl::usm::alloc::host;
        case alloc_kind::usm_device: return sycl::usm::alloc::device;
        case alloc_kind::usm_shared: return sycl::usm::alloc::shared;
        default: return sycl::usm::alloc::unknown;
    }
}

template <typename T>
inline array<T> get_table_block(device_test_policy& p,
                                const table& t,
                                const range& row_range,
                                const alloc_kind alloc = alloc_kind::usm_device) {
    if (alloc != alloc_kind::non_usm) {
        sycl::usm::alloc sycl_alloc = get_sycl_alloc_kind(alloc);
        if (sycl_alloc == sycl::usm::alloc::unknown) {
            throw invalid_argument{ "Invalid alloc_kind value" };
        }
        return row_accessor<const T>{ t }.pull(p.get_queue(), row_range, sycl_alloc);
    }
    return row_accessor<const T>{ t }.pull(row_range);
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

/// Returns a human-readable name of the allocation kind. Catch2 cannot stringify
/// `alloc_kind` on its own, so use this in `INFO`/`CAPTURE` to keep failures diagnosable.
inline std::string get_alloc_kind_name(alloc_kind alloc) {
    switch (alloc) {
        case alloc_kind::non_usm: return "non-usm";
        case alloc_kind::usm_host: return "usm-host";
        case alloc_kind::usm_device: return "usm-device";
        case alloc_kind::usm_shared: return "usm-shared";
        default: return "unknown";
    }
}

/// Returns a random allocation kind that is different from the previous one.
///
/// @param alloc_rng    Random number generator for allocation kind selection.
/// @param gen          Random number generator engine.
/// @param prev_alloc   The previous allocation kind to avoid repeating.
///
/// @return A random allocation kind different from `prev_alloc`.
alloc_kind get_random_alloc(std::uniform_int_distribution<>& alloc_rng,
                            std::mt19937& gen,
                            alloc_kind prev_alloc) {
    int alloc = alloc_rng(gen);
    while (alloc == static_cast<int>(prev_alloc)) {
        alloc = alloc_rng(gen);
    }
    return static_cast<alloc_kind>(alloc);
}

/// Split a table into multiple smaller tables by rows, with the first block having a specified allocation kind.
///
/// @tparam Float       The data type of the table elements.
/// @tparam TestPolicy  The type of the test policy (e.g., host_test_policy or device_test_policy).
///
/// @param policy            The test policy instance.
/// @param t                 The table to be split.
/// @param split_count       The number of blocks to split the table into.
/// @param first_block_alloc The allocation kind for the first block.
///
/// @pre :expr:`split_count > 0`
///
/// @return A vector of tables resulting from the split.
template <typename Float, typename TestPolicy>
inline std::vector<table> split_table_by_rows_mixed(TestPolicy& policy,
                                                    const table& t,
                                                    std::int64_t split_count,
                                                    alloc_kind first_block_alloc) {
    ONEDAL_ASSERT(split_count > 0);

    const std::int64_t row_count = t.get_row_count();
    const std::int64_t column_count = t.get_column_count();
    const std::int64_t block_size_regular = row_count / split_count;
    const std::int64_t block_size_tail = row_count % split_count;

    std::vector<table> result(split_count);

    std::int64_t row_offset = 0;
    std::mt19937 gen(7777);
    std::uniform_int_distribution<> alloc_rng(0, 3);
    alloc_kind prev_alloc = first_block_alloc;
    for (std::int64_t i = 0; i < split_count; i++) {
        const std::int64_t tail = std::int64_t(i + 1 == split_count) * block_size_tail;
        const std::int64_t block_size = block_size_regular + tail;

        if (block_size > 0) {
            const auto row_range = range{ row_offset, row_offset + block_size };

            const alloc_kind alloc =
                (i == 0) ? first_block_alloc : get_random_alloc(alloc_rng, gen, prev_alloc);
            prev_alloc = alloc;
            const array<Float> block = get_table_block<Float>(policy, t, row_range, alloc);
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
