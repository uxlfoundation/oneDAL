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

#include "oneapi/dal/algo/basic_statistics/test/fixture.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"

namespace oneapi::dal::basic_statistics::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace bs = oneapi::dal::basic_statistics;

/// Slice a CSR table by row-blocks, host-side. Returns `n_blocks` new csr_tables
/// each holding a contiguous row range of the original. Non-zero counts per block
/// are derived from the original row_offsets; column_count is preserved.
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

        // A block whose rows happen to be all implicit zeros still contributes
        // `block_rows` rows to the online statistics, so it must be forwarded to
        // partial_compute rather than dropped. It cannot be encoded as a
        // structurally empty table, because a CSR table's values and column indices
        // are `dal::array`s and `dal::array<T>::empty(0)` throws. Store a single
        // explicit zero in the block's first row instead: explicit zeros are a valid
        // CSR encoding and leave every statistic unchanged.
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

/// Build a CSR table that stores every element of `dense` (a row-major host buffer
/// of `row_count` x `column_count`) explicitly, zeros included. Explicit zeros are
/// a valid CSR encoding and are the only way to express a batch whose values are
/// all zero: a sparsity-driven builder cannot reliably emit one, and a structurally
/// empty (nnz == 0) block is not representable, because `csr_table::wrap` takes its
/// values and column indices as `dal::array`s and `dal::array<T>::empty(0)` throws.
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

template <typename TestType>
class basic_statistics_online_csr_test
        : public basic_statistics_test<TestType, basic_statistics_online_csr_test<TestType>> {
public:
    using float_t = std::tuple_element_t<0, TestType>;
    using descriptor_t = bs::descriptor<float_t, bs::method::sparse, bs::task::compute>;

    /// Row-slice the fully-materialized dense reference to match a CSR prefix.
    /// Uses raw row_accessor pulls to keep the test transport-agnostic.
    static table dense_row_prefix(const table& dense, std::int64_t row_count) {
        if (row_count == dense.get_row_count())
            return dense;
        auto arr = row_accessor<const float_t>{ dense }.pull({ 0, row_count });
        return homogen_table::wrap(arr, row_count, dense.get_column_count());
    }

    /// Run partial_compute + finalize_compute on `n_blocks` CSR row-shards
    /// and check the result against batch::compute on the full CSR table
    /// (which is itself validated against the dense reference in csr_general_checks).
    /// Also validates every intermediate finalize(partial_k) against the dense
    /// reference restricted to the prefix that has been fed so far. This catches
    /// (a) merge bugs (e.g. min/max/sum forgetting a batch), (b) storage-aliasing
    /// bugs where a later partial_compute mutates the tables underlying an earlier
    /// finalize result, and (c) mid-stream regressions that only surface at
    /// specific block counts.
    void online_csr_general_checks(const te::csr_table_builder<>& builder,
                                   bs::result_option_id compute_mode,
                                   std::int64_t n_blocks) {
        CAPTURE(n_blocks, compute_mode);
        const auto desc = descriptor_t{}.set_result_options(compute_mode);
        const auto full_csr = builder.build_csr_table(this->get_policy());
        const auto dense_ref = builder.build_dense_table(this->get_policy());

        auto batch_result = this->compute(desc, full_csr);
        this->check_compute_result(compute_mode, dense_ref, table{}, batch_result);

        const auto blocks = split_csr_by_rows<float_t>(full_csr, n_blocks);
        dal::basic_statistics::partial_compute_result<> partial;
        std::int64_t rows_fed = 0;
        for (const auto& block : blocks) {
            partial = this->partial_compute(desc, partial, block);
            rows_fed += block.get_row_count();

            // Snapshot: verify that at this point finalize agrees with what
            // the batch algorithm would produce on the first `rows_fed` rows.
            // Skip when the prefix has fewer than two rows: the reference and
            // the kernel both divide sum_squares_centered by (n - 1), which is
            // undefined at n == 1.
            if (rows_fed < 2)
                continue;
            auto interim = this->finalize_compute(desc, partial);
            const auto ref_prefix = dense_row_prefix(dense_ref, rows_fed);
            this->check_compute_result(compute_mode, ref_prefix, table{}, interim);
        }
        // After the loop `partial` already contains all rows; the last iteration
        // finalize covered n_blocks == the full table. Still emit the final check
        // separately for symmetry with online.cpp.
        auto final_result = this->finalize_compute(desc, partial);
        this->check_compute_result(compute_mode, dense_ref, table{}, final_result);
        this->check_for_exception_for_non_requested_results(compute_mode, final_result);
    }

    /// Feed two CSR batches where one of them holds only zeros, either as the first
    /// batch or right after a non-empty one, and check both the intermediate and the
    /// final finalize_compute against the dense reference. The payload rows are
    /// strictly positive, so an all-zero batch is the only possible source of a zero
    /// minimum: a merge that silently drops such a batch is caught by `min`, and one
    /// that drops its rows from the observation count is caught by `mean`.
    void online_csr_zero_batch_checks(bs::result_option_id compute_mode, bool zeros_first) {
        CAPTURE(compute_mode, zeros_first);
        constexpr std::int64_t column_count = 4;
        constexpr std::int64_t zero_rows = 3;
        constexpr std::int64_t data_rows = 9;
        constexpr std::int64_t row_count = zero_rows + data_rows;

        auto dense_arr = dal::array<float_t>::zeros(row_count * column_count);
        auto dense_ptr = dense_arr.get_mutable_data();
        const std::int64_t data_begin = zeros_first ? zero_rows : 0;
        for (std::int64_t r = data_begin; r < data_begin + data_rows; ++r) {
            for (std::int64_t c = 0; c < column_count; ++c) {
                const std::int64_t i = r * column_count + c;
                dense_ptr[i] = float_t((i % 7) + 1);
            }
        }
        const table dense_ref = homogen_table::wrap(dense_arr, row_count, column_count);

        const std::int64_t first_rows = zeros_first ? zero_rows : data_rows;
        const std::int64_t second_rows = row_count - first_rows;
        const std::vector<csr_table> blocks{ dense_to_explicit_csr<float_t>(dense_ptr,
                                                                            first_rows,
                                                                            column_count,
                                                                            this->data_indexing_),
                                             dense_to_explicit_csr<float_t>(
                                                 dense_ptr + first_rows * column_count,
                                                 second_rows,
                                                 column_count,
                                                 this->data_indexing_) };

        const auto desc = descriptor_t{}.set_result_options(compute_mode);
        dal::basic_statistics::partial_compute_result<> partial;
        std::int64_t rows_fed = 0;
        for (const auto& block : blocks) {
            partial = this->partial_compute(desc, partial, block);
            rows_fed += block.get_row_count();
            auto interim = this->finalize_compute(desc, partial);
            this->check_compute_result(compute_mode,
                                       dense_row_prefix(dense_ref, rows_fed),
                                       table{},
                                       interim);
        }
        REQUIRE(rows_fed == row_count);
        this->check_for_exception_for_non_requested_results(compute_mode,
                                                            this->finalize_compute(desc, partial));
    }

    /// Same flow as `online_csr_general_checks`, but with per-row weights. The
    /// reference in `check_vs_reference` scales every element by its row's weight and
    /// keeps the plain row count as the observation count, which is exactly what the
    /// sparse backends do by folding the weights into the stored values, see
    /// `scale_csr_by_weights`. `te::split_table_by_rows` uses the same
    /// regular-blocks-plus-tail split as `split_csr_by_rows`, so block `i` of the
    /// weights lines up row-for-row with block `i` of the data.
    void online_csr_weighted_checks(const te::csr_table_builder<>& builder,
                                    bs::result_option_id compute_mode,
                                    std::int64_t n_blocks) {
        CAPTURE(n_blocks, compute_mode);
        const auto desc = descriptor_t{}.set_result_options(compute_mode);
        const auto full_csr = builder.build_csr_table(this->get_policy());
        const auto dense_ref = builder.build_dense_table(this->get_policy());
        const auto row_count = full_csr.get_row_count();

        // Weights on both sides of 1 so that a path ignoring them cannot pass.
        const auto weights_df =
            te::dataframe_builder{ row_count, 1 }.fill_uniform(0.2, 3.0, 4242).build();
        const table weights =
            weights_df.get_table(this->get_policy(), this->get_homogen_table_id());

        const auto blocks = split_csr_by_rows<float_t>(full_csr, n_blocks);
        const auto weight_blocks =
            te::split_table_by_rows<float_t>(this->get_policy(), weights, n_blocks);

        dal::basic_statistics::partial_compute_result<> partial;
        std::int64_t rows_fed = 0;
        for (std::int64_t b = 0; b < n_blocks; ++b) {
            partial = this->partial_compute(desc, partial, blocks[b], weight_blocks[b]);
            rows_fed += blocks[b].get_row_count();
        }
        REQUIRE(rows_fed == row_count);

        auto result = this->finalize_compute(desc, partial);
        this->check_compute_result(compute_mode, dense_ref, weights, result);
        this->check_for_exception_for_non_requested_results(compute_mode, result);
    }
};

using online_csr_types = COMBINE_TYPES((float, double), (bs::method::sparse));

TEMPLATE_LIST_TEST_M(basic_statistics_online_csr_test,
                     "basic_statistics online CSR flow",
                     "[basic_statistics][integration][online]",
                     online_csr_types) {
    SKIP_IF(this->not_float64_friendly());
    const float nnz_fraction = 0.05;
    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);
    const auto data =
        GENERATE_COPY(te::csr_table_builder(20, 10, nnz_fraction, this->data_indexing_),
                      te::csr_table_builder(100, 20, nnz_fraction, this->data_indexing_),
                      te::csr_table_builder(100, 100, nnz_fraction, this->data_indexing_));
    SKIP_IF(this->not_cpu_friendly(data));

    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));

    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_mean_varc, res_all);
    const std::int64_t n_blocks = GENERATE(1, 3, 5);

    this->online_csr_general_checks(data, compute_mode, n_blocks);
}

TEMPLATE_LIST_TEST_M(basic_statistics_online_csr_test,
                     "basic_statistics online CSR flow with an all-zero batch",
                     "[basic_statistics][integration][online]",
                     online_csr_types) {
    SKIP_IF(this->not_float64_friendly());
    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);
    // `true`: the all-zero batch is the first one, so it also has to initialize the
    // partial result. `false`: it is merged into an already populated partial.
    const bool zeros_first = GENERATE(true, false);

    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));
    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_all);

    this->online_csr_zero_batch_checks(compute_mode, zeros_first);
}

TEMPLATE_LIST_TEST_M(basic_statistics_online_csr_test,
                     "basic_statistics online CSR flow with weights",
                     "[basic_statistics][integration][online]",
                     online_csr_types) {
    SKIP_IF(this->not_float64_friendly());
    const float nnz_fraction = 0.05;
    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);
    const auto data =
        GENERATE_COPY(te::csr_table_builder(20, 10, nnz_fraction, this->data_indexing_),
                      te::csr_table_builder(100, 20, nnz_fraction, this->data_indexing_));
    SKIP_IF(this->not_cpu_friendly(data));

    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));

    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_mean_varc, res_all);
    const std::int64_t n_blocks = GENERATE(1, 3);

    this->online_csr_weighted_checks(data, compute_mode, n_blocks);
}

} // namespace oneapi::dal::basic_statistics::test
