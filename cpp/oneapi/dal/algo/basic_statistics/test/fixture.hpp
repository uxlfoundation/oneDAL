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

#include <limits>

#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/algo/basic_statistics/compute.hpp"
#include "oneapi/dal/algo/basic_statistics/partial_compute.hpp"
#include "oneapi/dal/algo/basic_statistics/finalize_compute.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/csr_table_builder.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/tables.hpp"

#include "oneapi/dal/table/csr_accessor.hpp"
namespace oneapi::dal::basic_statistics::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace bs = oneapi::dal::basic_statistics;
namespace dal = oneapi::dal;

constexpr inline std::uint64_t mask_full = 0xffffffffffffffff;

template <typename TestType, typename Derived>
class basic_statistics_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    sparse_indexing data_indexing_; // for sparse data testing
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using input_t = bs::compute_input<>;
    using result_t = bs::compute_result<>;
    using partial_input_t = bs::partial_compute_input<>;
    using partial_result_t = bs::partial_compute_result<>;
    using descriptor_t = bs::descriptor<float_t, method_t>;
    using csr_table = dal::csr_table;

    auto get_descriptor(bs::result_option_id compute_mode) const {
        return descriptor_t{}.set_result_options(compute_mode);
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    void general_checks(const te::dataframe& data_fr,
                        std::shared_ptr<te::dataframe> weights_fr,
                        bs::result_option_id compute_mode) {
        const auto use_weights = bool(weights_fr);
        CAPTURE(use_weights, compute_mode);

        const auto bs_desc = get_descriptor(compute_mode);
        const auto data_table_id = this->get_homogen_table_id();

        table weights, data = data_fr.get_table(this->get_policy(), data_table_id);

        bs::compute_result<> compute_result;
        if (use_weights) {
            weights = weights_fr->get_table(this->get_policy(), data_table_id);
            compute_result = this->compute(bs_desc, data, weights);
        }
        else {
            compute_result = this->compute(bs_desc, data);
        }

        check_compute_result(compute_mode, data, weights, compute_result);
        check_for_exception_for_non_requested_results(compute_mode, compute_result);
    }

    void csr_general_checks(const te::csr_table_builder<>& data,
                            bs::result_option_id compute_mode) {
        const auto desc =
            bs::descriptor<float_t, basic_statistics::method::sparse>{}.set_result_options(
                compute_mode);
        const auto csr_table = data.build_csr_table(this->get_policy());
        const auto dense_table = data.build_dense_table(this->get_policy());

        auto compute_result = this->compute(desc, csr_table);
        table weights;
        check_compute_result(compute_mode, dense_table, weights, compute_result);
    }

    // TODO: Fix DAAL code. On big datasets there is an error in computing.
    // To reproduce it remove this check from test case in batch.cpp
    bool not_cpu_friendly(const te::csr_table_builder<>& data) {
        auto policy = this->get_policy();
        return (data.row_count_ > 100 || data.column_count_ > 100) && policy.is_cpu();
    }

    void online_general_checks(const te::dataframe& data_fr,
                               std::shared_ptr<te::dataframe> weights_fr,
                               bs::result_option_id compute_mode,
                               std::int64_t nBlocks) {
        const auto use_weights = bool(weights_fr);
        CAPTURE(use_weights, compute_mode);
        const auto bs_desc = get_descriptor(compute_mode);
        const auto data_table_id = this->get_homogen_table_id();

        table weights, data = data_fr.get_table(this->get_policy(), data_table_id);
        dal::basic_statistics::partial_compute_result<> partial_result;

        auto input_table = te::split_table_by_rows<float_t>(this->get_policy(), data, nBlocks);
        if (use_weights) {
            weights = weights_fr->get_table(this->get_policy(), data_table_id);
            auto weights_table =
                te::split_table_by_rows<float_t>(this->get_policy(), weights, nBlocks);
            for (std::int64_t i = 0; i < nBlocks; ++i) {
                partial_result = this->partial_compute(bs_desc,
                                                       partial_result,
                                                       input_table[i],
                                                       weights_table[i]);
            }
            auto compute_result = this->finalize_compute(bs_desc, partial_result);
            check_compute_result(compute_mode, data, weights, compute_result);
            check_for_exception_for_non_requested_results(compute_mode, compute_result);
        }
        else {
            for (std::int64_t i = 0; i < nBlocks; ++i) {
                partial_result = this->partial_compute(bs_desc, partial_result, input_table[i]);
            }
            auto compute_result = this->finalize_compute(bs_desc, partial_result);
            check_compute_result(compute_mode, data, weights, compute_result);
            check_for_exception_for_non_requested_results(compute_mode, compute_result);
        }
    }

    void check_compute_result(bs::result_option_id compute_mode,
                              const table& data,
                              const table& weights,
                              const result_t& result) {
        SECTION("result tables' shape is expected") {
            check_result_shape(compute_mode, data, result);
        }

        SECTION("check results against reference") {
            check_vs_reference(compute_mode, data, weights, result);
        }
    }

    void check_result_shape(bs::result_option_id compute_mode,
                            const table& data,
                            const result_t& result) {
        CAPTURE(data.get_row_count());
        CAPTURE(data.get_column_count());
        if (compute_mode.test(result_options::min)) {
            REQUIRE(result.get_min().get_column_count() == data.get_column_count());
        }
        if (compute_mode.test(result_options::max)) {
            REQUIRE(result.get_max().get_column_count() == data.get_column_count());
        }
        if (compute_mode.test(result_options::sum)) {
            REQUIRE(result.get_sum().get_column_count() == data.get_column_count());
        }
        if (compute_mode.test(result_options::sum_squares)) {
            REQUIRE(result.get_sum_squares().get_column_count() == data.get_column_count());
        }
        if (compute_mode.test(result_options::sum_squares_centered)) {
            REQUIRE(result.get_sum_squares_centered().get_column_count() ==
                    data.get_column_count());
        }
        if (compute_mode.test(result_options::mean)) {
            REQUIRE(result.get_mean().get_column_count() == data.get_column_count());
        }
        if (compute_mode.test(result_options::second_order_raw_moment)) {
            REQUIRE(result.get_second_order_raw_moment().get_column_count() ==
                    data.get_column_count());
        }
        if (compute_mode.test(result_options::variance)) {
            REQUIRE(result.get_variance().get_column_count() == data.get_column_count());
        }
        if (compute_mode.test(result_options::standard_deviation)) {
            REQUIRE(result.get_standard_deviation().get_column_count() == data.get_column_count());
        }
        if (compute_mode.test(result_options::variation)) {
            REQUIRE(result.get_variation().get_column_count() == data.get_column_count());
        }
    }

    void check_if_close(const table& left,
                        const table& right,
                        std::string name = "",
                        double tol = 1e-2) {
        constexpr auto eps = std::numeric_limits<float_t>::epsilon();

        const auto c_count = left.get_column_count();
        const auto r_count = left.get_row_count();

        REQUIRE(right.get_column_count() == c_count);
        REQUIRE(right.get_row_count() == r_count);

        row_accessor<const float_t> lacc(left);
        row_accessor<const float_t> racc(right);

        const auto larr = lacc.pull({ 0, -1 });
        const auto rarr = racc.pull({ 0, -1 });

        for (std::int64_t r = 0; r < r_count; ++r) {
            for (std::int64_t c = 0; c < c_count; ++c) {
                const auto lval = larr[r * c_count + c];
                const auto rval = rarr[r * c_count + c];

                CAPTURE(name, r_count, c_count, r, c, lval, rval);

                const auto aerr = std::abs(lval - rval);
                if (aerr < tol || (!std::isfinite(lval) && !std::isfinite(rval)))
                    continue;

                const auto den = std::max({ eps, //
                                            std::abs(lval),
                                            std::abs(rval) });

                const auto rerr = aerr / den;
                CAPTURE(aerr, rerr, den, r, c, lval, rval);
                REQUIRE(rerr < tol);
            }
        }
    }

    void check_vs_reference(bs::result_option_id compute_mode,
                            const table& data,
                            const table& weights,
                            const result_t& result) {
        using limits_t = std::numeric_limits<double>;
        constexpr auto maximum = limits_t::max();
        constexpr double zero = 0.0, one = 1.0;

        CAPTURE(compute_mode);
        CAPTURE(data.get_row_count());
        CAPTURE(data.get_column_count());

        const auto data_matrix = la::matrix<double>::wrap(data);

        const auto row_count = data_matrix.get_row_count();
        const auto column_count = data_matrix.get_column_count();

        la::matrix<double> weights_matrix;
        if (weights.has_data()) {
            weights_matrix = la::matrix<double>::wrap(weights);
        }
        else {
            weights_matrix = la::matrix<double>::full({ row_count, 1 }, one);
        }

        auto ref_sum2cent = la::matrix<double>::full({ 1, column_count }, zero);
        auto ref_min = la::matrix<double>::full({ 1, column_count }, +maximum);
        auto ref_max = la::matrix<double>::full({ 1, column_count }, -maximum);
        auto ref_stdev = la::matrix<double>::full({ 1, column_count }, zero);
        auto ref_sum2 = la::matrix<double>::full({ 1, column_count }, zero);
        auto ref_mean = la::matrix<double>::full({ 1, column_count }, zero);
        auto ref_sorm = la::matrix<double>::full({ 1, column_count }, zero);
        auto ref_varc = la::matrix<double>::full({ 1, column_count }, zero);
        auto ref_vart = la::matrix<double>::full({ 1, column_count }, zero);
        auto ref_sum = la::matrix<double>::full({ 1, column_count }, zero);

        // calc mean
        for (std::int64_t row = 0; row < row_count; row++) {
            for (std::int64_t clmn = 0; clmn < column_count; clmn++) {
                ref_mean.set(0, clmn) += (data_matrix.get(row, clmn) * weights_matrix.get(row, 0));
            }
        }

        // finalize mean
        for (std::int64_t clmn = 0; clmn < column_count; clmn++) {
            ref_mean.set(0, clmn) = ref_mean.set(0, clmn) / float_t(row_count);
        }

        for (std::int64_t row = 0; row < row_count; row++) {
            for (std::int64_t clmn = 0; clmn < column_count; clmn++) {
                const auto elem = data_matrix.get(row, clmn);
                const auto weight = weights_matrix.get(row, 0);
                ref_min.set(0, clmn) = std::min(ref_min.get(0, clmn), elem * weight);
                ref_max.set(0, clmn) = std::max(ref_max.get(0, clmn), elem * weight);
                ref_sum.set(0, clmn) += elem * weight;
                ref_sum2.set(0, clmn) += (elem * weight * elem * weight);
                ref_sum2cent.set(0, clmn) += (elem * weight - ref_mean.get(0, clmn)) *
                                             (elem * weight - ref_mean.get(0, clmn));
            }
        }
        for (std::int64_t clmn = 0; clmn < column_count; clmn++) {
            ref_sorm.set(0, clmn) = ref_sum2.get(0, clmn) / float_t(row_count);
            ref_varc.set(0, clmn) = ref_sum2cent.get(0, clmn) / float_t(row_count - 1);
            ref_stdev.set(0, clmn) = std::sqrt(ref_varc.get(0, clmn));
            ref_vart.set(0, clmn) = ref_stdev.get(0, clmn) / ref_mean.get(0, clmn);
        }
        if (compute_mode.test(result_options::min)) {
            const table ref = homogen_table::wrap(ref_min.get_array(), 1l, column_count);
            check_if_close(result.get_min(), ref, "Min");
        }
        if (compute_mode.test(result_options::max)) {
            const table ref = homogen_table::wrap(ref_max.get_array(), 1l, column_count);
            check_if_close(result.get_max(), ref, "Max");
        }
        if (compute_mode.test(result_options::sum)) {
            const table ref = homogen_table::wrap(ref_sum.get_array(), 1l, column_count);
            check_if_close(result.get_sum(), ref, "Sum");
        }
        if (compute_mode.test(result_options::sum_squares)) {
            const table ref = homogen_table::wrap(ref_sum2.get_array(), 1l, column_count);
            check_if_close(result.get_sum_squares(), ref, "Sum squares");
        }
        if (compute_mode.test(result_options::sum_squares_centered)) {
            const table ref = homogen_table::wrap(ref_sum2cent.get_array(), 1l, column_count);
            check_if_close(result.get_sum_squares_centered(), ref, "Sum squares centered");
        }
        if (compute_mode.test(result_options::mean)) {
            const table ref = homogen_table::wrap(ref_mean.get_array(), 1l, column_count);
            check_if_close(result.get_mean(), ref, "Mean");
        }
        if (compute_mode.test(result_options::second_order_raw_moment)) {
            const table ref = homogen_table::wrap(ref_sorm.get_array(), 1l, column_count);
            check_if_close(result.get_second_order_raw_moment(), ref, "SORM");
        }
        if (compute_mode.test(result_options::variance)) {
            const table ref = homogen_table::wrap(ref_varc.get_array(), 1l, column_count);
            check_if_close(result.get_variance(), ref, "Variance");
        }
        if (compute_mode.test(result_options::standard_deviation)) {
            const table ref = homogen_table::wrap(ref_stdev.get_array(), 1l, column_count);
            check_if_close(result.get_standard_deviation(), ref, "Std");
        }
        if (compute_mode.test(result_options::variation)) {
            const table ref = homogen_table::wrap(ref_vart.get_array(), 1l, column_count);
            check_if_close(result.get_variation(), ref, "Variation");
        }
    }

    void check_for_exception_for_non_requested_results(bs::result_option_id compute_mode,
                                                       const result_t& result) {
        if (!compute_mode.test(result_options::min)) {
            REQUIRE_THROWS_AS(result.get_min(), domain_error);
        }
        if (!compute_mode.test(result_options::max)) {
            REQUIRE_THROWS_AS(result.get_max(), domain_error);
        }
        if (!compute_mode.test(result_options::sum)) {
            REQUIRE_THROWS_AS(result.get_sum(), domain_error);
        }
        if (!compute_mode.test(result_options::sum_squares)) {
            REQUIRE_THROWS_AS(result.get_sum_squares(), domain_error);
        }
        if (!compute_mode.test(result_options::sum_squares_centered)) {
            REQUIRE_THROWS_AS(result.get_sum_squares_centered(), domain_error);
        }
        if (!compute_mode.test(result_options::mean)) {
            REQUIRE_THROWS_AS(result.get_mean(), domain_error);
        }
        if (!compute_mode.test(result_options::second_order_raw_moment)) {
            REQUIRE_THROWS_AS(result.get_second_order_raw_moment(), domain_error);
        }
        if (!compute_mode.test(result_options::variance)) {
            REQUIRE_THROWS_AS(result.get_variance(), domain_error);
        }
        if (!compute_mode.test(result_options::standard_deviation)) {
            REQUIRE_THROWS_AS(result.get_standard_deviation(), domain_error);
        }
        if (!compute_mode.test(result_options::variation)) {
            REQUIRE_THROWS_AS(result.get_variation(), domain_error);
        }
    }

private:
    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));
};

template <typename TestType>
class basic_statistics_online_csr_test
        : public basic_statistics_test<TestType, basic_statistics_online_csr_test<TestType>> {
public:
    using float_t = std::tuple_element_t<0, TestType>;
    using descriptor_t = bs::descriptor<float_t, bs::method::sparse, bs::task::compute>;

    /// Row-slice the fully-materialized dense reference to match a CSR prefix.
    /// Uses raw row_accessor pulls to keep the test transport-agnostic.
    ///
    /// @param dense     Dense reference table of size `n x p`
    /// @param row_count Number of leading rows to keep
    /// @return          The first `row_count` rows of `dense`, or `dense` itself when the
    ///                  whole table is requested
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
    ///
    /// @param builder      Source of the CSR table and its dense reference
    /// @param compute_mode Requested result options
    /// @param n_blocks     Number of row-shards the table is fed in
    void online_csr_general_checks(const te::csr_table_builder<>& builder,
                                   bs::result_option_id compute_mode,
                                   std::int64_t n_blocks) {
        CAPTURE(n_blocks, compute_mode);
        const auto desc = descriptor_t{}.set_result_options(compute_mode);
        const auto full_csr = builder.build_csr_table(this->get_policy());
        const auto dense_ref = builder.build_dense_table(this->get_policy());

        auto batch_result = this->compute(desc, full_csr);
        this->check_compute_result(compute_mode, dense_ref, table{}, batch_result);

        const auto blocks = te::split_csr_by_rows<float_t>(full_csr, n_blocks);
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
    ///
    /// @param compute_mode Requested result options
    /// @param zeros_first  `true` feeds the all-zero batch first, so it also has to
    ///                     initialize the partial result; `false` merges it into an
    ///                     already populated partial
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
        const std::vector<csr_table> blocks{
            te::dense_to_explicit_csr<float_t>(dense_ptr,
                                               first_rows,
                                               column_count,
                                               this->data_indexing_),
            te::dense_to_explicit_csr<float_t>(dense_ptr + first_rows * column_count,
                                               second_rows,
                                               column_count,
                                               this->data_indexing_)
        };

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
    /// regular-blocks-plus-tail split as `te::split_csr_by_rows`, so block `i` of the
    /// weights lines up row-for-row with block `i` of the data.
    ///
    /// @param builder      Source of the CSR table and its dense reference
    /// @param compute_mode Requested result options
    /// @param n_blocks     Number of row-shards the table is fed in
    /// @param weight_min   Lower bound of the uniformly drawn row weights
    /// @param weight_max   Upper bound of the uniformly drawn row weights
    void online_csr_weighted_checks(const te::csr_table_builder<>& builder,
                                    bs::result_option_id compute_mode,
                                    std::int64_t n_blocks,
                                    double weight_min = 0.2,
                                    double weight_max = 3.0) {
        CAPTURE(n_blocks, compute_mode, weight_min, weight_max);
        const auto desc = descriptor_t{}.set_result_options(compute_mode);
        const auto full_csr = builder.build_csr_table(this->get_policy());
        const auto dense_ref = builder.build_dense_table(this->get_policy());
        const auto row_count = full_csr.get_row_count();

        // Weights on both sides of 1 so that a path ignoring them cannot pass.
        const auto weights_df = te::dataframe_builder{ row_count, 1 }
                                    .fill_uniform(weight_min, weight_max, 4242)
                                    .build();
        const table weights =
            weights_df.get_table(this->get_policy(), this->get_homogen_table_id());

        const auto blocks = te::split_csr_by_rows<float_t>(full_csr, n_blocks);
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

using basic_statistics_types = COMBINE_TYPES((float, double), (basic_statistics::method::dense));
using basic_statistics_sparse_types = COMBINE_TYPES((float, double),
                                                    (basic_statistics::method::sparse));
using basic_statistics_online_csr_types = COMBINE_TYPES((float, double), (bs::method::sparse));

} // namespace oneapi::dal::basic_statistics::test
