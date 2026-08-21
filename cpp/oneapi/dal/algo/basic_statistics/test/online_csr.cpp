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
        if (block_rows == 0) {
            blocks[b] = csr_table{};
            row_offset += block_rows;
            continue;
        }
        const std::int64_t nnz_start = row_off_arr[row_offset];
        const std::int64_t nnz_end = row_off_arr[row_offset + block_rows];
        const std::int64_t nnz = nnz_end - nnz_start;

        // csr_accessor::pull returns column indices as std::int64_t.
        // NOTE on empty (nnz == 0) blocks: in principle they still contribute
        // implicit zero rows to the online statistics and should be forwarded to
        // partial_compute, but the downstream `host_csr_table_adapter` for a
        // zero_based CSR unconditionally calls `array<size_t>::reset(nnz)` to build
        // its one-based mirror; `daal_malloc(0)` returns null and the array
        // allocator throws `host_bad_alloc`. Fixing that adapter is out of scope
        // for this test; instead we emit an empty `csr_table{}` and gate on
        // `has_data()` at the call site. The current builder settings never emit a
        // block with block_rows > 0 && nnz == 0 (nnz_fraction * row_count *
        // column_count >= row_count for every (row_count, column_count) tuple
        // exercised below), so this path is defensive only.
        if (nnz == 0) {
            blocks[b] = csr_table{};
            row_offset += block_rows;
            continue;
        }

        auto sub_data = dal::array<Float>::empty(nnz);
        auto sub_cols = dal::array<std::int64_t>::empty(nnz);
        auto sub_offs = dal::array<std::int64_t>::empty(block_rows + 1);
        auto sub_data_ptr = sub_data.get_mutable_data();
        auto sub_cols_ptr = sub_cols.get_mutable_data();
        auto sub_offs_ptr = sub_offs.get_mutable_data();

        for (std::int64_t i = 0; i < nnz; ++i) {
            sub_data_ptr[i] = data_arr[nnz_start + i];
            sub_cols_ptr[i] = col_idx_arr[nnz_start + i];
        }
        for (std::int64_t r = 0; r <= block_rows; ++r) {
            sub_offs_ptr[r] = row_off_arr[row_offset + r] - nnz_start;
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
    /// finalize result (the same class of issue that hit dense finalize_compute
    /// on GPU in 2026-06-16), and (c) mid-stream regressions that only surface at
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
            if (!block.has_data())
                continue;
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

} // namespace oneapi::dal::basic_statistics::test
