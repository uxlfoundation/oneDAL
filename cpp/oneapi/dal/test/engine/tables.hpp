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
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

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
template <typename T>
inline array<T> get_table_block(device_test_policy& p, const table& t, const range& row_range) {
    return row_accessor<const T>{ t }.pull(p.get_queue(), row_range, sycl::usm::alloc::device);
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

/// Split a CSR table into row-blocks, host-side, using the same
/// regular-blocks-plus-tail partition as `split_table_by_rows`, so block `i` of a
/// dataset lines up row-for-row with block `i` of its weights.
///
/// A block whose rows happen to hold only implicit zeros still contributes its rows to
/// an online computation, so it must stay in the sequence rather than be dropped. It
/// cannot be encoded as a structurally empty table, because a CSR table's values and
/// column indices are `dal::array`s and `dal::array<T>::empty(0)` throws. Such a block
/// gets a single explicit zero in its first row instead: explicit zeros are a valid CSR
/// encoding and leave every statistic unchanged.
///
/// @tparam Float      Floating-point type of the table values
/// @param source      Input CSR table of size `n x p`
/// @param n_blocks    Number of row-blocks to produce, must be positive
/// @return            `n_blocks` CSR tables, each holding a contiguous row range of
///                    `source` and preserving its column count, always `zero_based`
template <typename Float>
inline std::vector<csr_table> split_csr_by_rows(const csr_table& source, std::int64_t n_blocks) {
    ONEDAL_ASSERT(n_blocks > 0);
    const auto row_count = source.get_row_count();
    const auto column_count = source.get_column_count();

    // Pull all three arrays on the host. `zero_based` is easier arithmetic; we emit
    // `zero_based` output.
    auto [data_arr, col_idx_arr, row_off_arr] =
        csr_accessor<const Float>(source).pull({ 0, -1 }, sparse_indexing::zero_based);

    std::vector<csr_table> blocks(n_blocks);
    const std::int64_t regular = row_count / n_blocks;
    const std::int64_t tail = row_count % n_blocks;
    std::int64_t row_offset = 0;
    for (std::int64_t b = 0; b < n_blocks; ++b) {
        const std::int64_t block_rows = regular + (b + 1 == n_blocks ? tail : 0);
        ONEDAL_ASSERT(block_rows > 0);
        const std::int64_t nnz_start = row_off_arr[row_offset];
        const std::int64_t nnz_end = row_off_arr[row_offset + block_rows];
        const std::int64_t nnz = nnz_end - nnz_start;

        const bool pad_empty = (nnz == 0);
        const std::int64_t sub_nnz = pad_empty ? 1 : nnz;

        // csr_accessor::pull returns column indices as std::int64_t.
        auto sub_data = dal::array<Float>::zeros(sub_nnz);
        auto sub_cols = dal::array<std::int64_t>::zeros(sub_nnz);
        auto sub_offs = dal::array<std::int64_t>::empty(block_rows + 1);
        auto sub_data_ptr = sub_data.get_mutable_data();
        auto sub_cols_ptr = sub_cols.get_mutable_data();
        auto sub_offs_ptr = sub_offs.get_mutable_data();

        if (pad_empty) {
            // `sub_data` / `sub_cols` are already zeroed: one stored element with
            // value 0 at (row 0, column 0), every other row empty.
            sub_offs_ptr[0] = 0;
            for (std::int64_t r = 1; r <= block_rows; ++r) {
                sub_offs_ptr[r] = 1;
            }
        }
        else {
            for (std::int64_t i = 0; i < nnz; ++i) {
                sub_data_ptr[i] = data_arr[nnz_start + i];
                sub_cols_ptr[i] = col_idx_arr[nnz_start + i];
            }
            for (std::int64_t r = 0; r <= block_rows; ++r) {
                sub_offs_ptr[r] = row_off_arr[row_offset + r] - nnz_start;
            }
        }

        blocks[b] = csr_table::wrap(sub_data,
                                    sub_cols,
                                    sub_offs,
                                    column_count,
                                    sparse_indexing::zero_based);
        row_offset += block_rows;
    }
    return blocks;
}

/// Build a CSR table that stores every element of a dense host buffer explicitly, zeros
/// included.
///
/// Explicit zeros are a valid CSR encoding and are the only way to express a block whose
/// values are all zero: a sparsity-driven builder cannot reliably emit one, and a
/// structurally empty (`nnz == 0`) block is not representable, because `csr_table::wrap`
/// takes its values and column indices as `dal::array`s and `dal::array<T>::empty(0)`
/// throws.
///
/// @tparam Float        Floating-point type of the table values
/// @param dense         Row-major host buffer of `row_count x column_count` values
/// @param row_count     Number of rows in `dense`
/// @param column_count  Number of columns in `dense`
/// @param indexing      Indexing of the produced table, `zero_based` or `one_based`
/// @return              A CSR table of size `row_count x column_count` with
///                      `row_count * column_count` stored values
template <typename Float>
inline csr_table dense_to_explicit_csr(const Float* dense,
                                       std::int64_t row_count,
                                       std::int64_t column_count,
                                       sparse_indexing indexing) {
    const std::int64_t nnz = row_count * column_count;
    const std::int64_t shift = (indexing == sparse_indexing::one_based) ? 1 : 0;

    auto data = dal::array<Float>::empty(nnz);
    auto cols = dal::array<std::int64_t>::empty(nnz);
    auto offs = dal::array<std::int64_t>::empty(row_count + 1);
    auto data_ptr = data.get_mutable_data();
    auto cols_ptr = cols.get_mutable_data();
    auto offs_ptr = offs.get_mutable_data();

    for (std::int64_t r = 0; r < row_count; ++r) {
        offs_ptr[r] = r * column_count + shift;
        for (std::int64_t c = 0; c < column_count; ++c) {
            const std::int64_t i = r * column_count + c;
            data_ptr[i] = dense[i];
            cols_ptr[i] = c + shift;
        }
    }
    offs_ptr[row_count] = nnz + shift;

    return csr_table::wrap(data, cols, offs, column_count, indexing);
}

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
