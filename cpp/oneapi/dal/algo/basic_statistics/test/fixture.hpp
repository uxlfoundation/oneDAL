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
    template <typename Float>
    std::vector<dal::table> split_table_by_rows(const dal::table& t, std::int64_t split_count) {
        ONEDAL_ASSERT(0l < split_count);
        ONEDAL_ASSERT(split_count <= t.get_row_count());

        const std::int64_t row_count = t.get_row_count();
        const std::int64_t column_count = t.get_column_count();
        const std::int64_t block_size_regular = row_count / split_count;
        const std::int64_t block_size_tail = row_count % split_count;

        std::vector<dal::table> result(split_count);

        std::int64_t row_offset = 0;
        for (std::int64_t i = 0; i < split_count; i++) {
            const std::int64_t tail = std::int64_t(i + 1 == split_count) * block_size_tail;
            const std::int64_t block_size = block_size_regular + tail;

            const auto row_range = dal::range{ row_offset, row_offset + block_size };
            const auto block = dal::row_accessor<const Float>{ t }.pull(row_range);
            result[i] = dal::homogen_table::wrap(block, block_size, column_count);
            row_offset += block_size;
        }

        return result;
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

        auto input_table = split_table_by_rows<double>(data, nBlocks);
        if (use_weights) {
            weights = weights_fr->get_table(this->get_policy(), data_table_id);
            auto weights_table = split_table_by_rows<double>(weights, nBlocks);
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

using basic_statistics_types = COMBINE_TYPES((float, double), (basic_statistics::method::dense));
using basic_statistics_sparse_types = COMBINE_TYPES((float, double),
                                                    (basic_statistics::method::sparse));

} // namespace oneapi::dal::basic_statistics::test
