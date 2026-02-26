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

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/reduction/reduction.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using reduction_types = std::tuple<std::tuple<float, sum<float>, square<float>>,
                                   std::tuple<double, sum<double>, square<double>>>;

using finiteness_types = std::tuple<std::tuple<float, sum<float>, identity<float>>,
                                    std::tuple<double, sum<double>, identity<double>>,
                                    std::tuple<float, logical_or<bool>, isinfornan<float>>,
                                    std::tuple<double, logical_or<bool>, isinfornan<double>>>;

template <typename Param>
class reduction_test_random : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate() {
        height_ = GENERATE(17, 999, 1, 5, 1001);
        width_ = GENERATE(7, 707, 1, 251, 5);
        override_init_ = GENERATE(0, 1);
        CAPTURE(override_init_, width_, height_);
        generate_input();
        generate_offset();
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    auto output(std::int64_t size) {
        check_if_initialized();
        constexpr auto alloc = sycl::usm::alloc::device;
        auto result = ndarray<float_t, 1>::empty(this->get_queue(), { size }, alloc);
        auto offsets = row_accessor<const float_t>{ offset_table_ } //
                           .pull(this->get_queue(), { 0, 1 }, alloc);
        auto input = ndview<float_t, 1>::wrap(offsets.get_data(), { size });
        return std::make_tuple(result, copy(this->get_queue(), result, input));
    }

    void generate_input() {
        const auto train_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ height_, width_ }.fill_uniform(-3.0, 4.0));
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());
    }

    void generate_offset() {
        const auto dimension = std::max<std::int64_t>(width_, height_);
        const auto offset_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ 1l, dimension }.fill_uniform(-1.0, 1.0));
        this->offset_table_ = offset_dataframe.get_table(this->get_homogen_table_id());
    }

    bool is_initialized() const {
        return width_ > 0 && height_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    array<float_t> groundtruth_rm_cw() const {
        auto res = array<float_t>::empty(width_);
        auto* const res_ptr = res.get_mutable_data();
        {
            row_accessor<const float_t> offset{ offset_table_ };
            const auto off_acc = offset.pull({ 0, 1 });
            for (std::int64_t i = 0; i < width_; ++i) {
                res_ptr[i] = override_init_ ? binary_.init_value : off_acc[i];
            }
        }
        row_accessor<const float_t> input{ input_table_ };
        for (std::int64_t j = 0; j < height_; ++j) {
            const auto row_acc = input.pull({ j, j + 1 });
            for (std::int64_t i = 0; i < width_; ++i) {
                const float_t val = row_acc[i];
                res_ptr[i] = binary_(res_ptr[i], unary_(val));
            }
        }
        return res;
    }

    array<float_t> groundtruth_rm_rw() const {
        auto res = array<float_t>::empty(height_);
        auto* const res_ptr = res.get_mutable_data();
        {
            row_accessor<const float_t> offset{ offset_table_ };
            const auto off_acc = offset.pull({ 0, 1 });
            for (std::int64_t j = 0; j < height_; ++j) {
                res_ptr[j] = override_init_ ? binary_.init_value : off_acc[j];
            }
        }
        row_accessor<const float_t> input{ input_table_ };
        for (std::int64_t j = 0; j < height_; ++j) {
            const auto row_acc = input.pull({ j, j + 1 });
            for (std::int64_t i = 0; i < width_; ++i) {
                const float_t val = row_acc[i];
                res_ptr[j] = binary_(res_ptr[j], unary_(val));
            }
        }
        return res;
    }

    void check_array(const array<float_t>& gtv,
                     const array<float_t>& arr,
                     const float_t tol = 1.e-3) {
        CAPTURE(__func__, gtv.get_count(), arr.get_count(), width_, height_, override_init_);
        REQUIRE(gtv.get_count() == arr.get_count());
        for (auto i = 0; i < arr.get_count(); ++i) {
            const auto div = std::max<float_t>({ //
                                                 std::abs(arr[i]),
                                                 std::abs(gtv[i]),
                                                 1.0 });
            const auto err = std::abs(arr[i] - gtv[i]) / div;
            if (err < -tol || tol < err) {
                CAPTURE(i, gtv[i], arr[i], div, err, tol);
                FAIL();
            }
        }
    }

    void check_output_rm_rw(ndarray<float_t, 1>& outarr, const float_t tol = 1.e-5) {
        CAPTURE(__func__, width_, height_, outarr.get_count());
        const auto gtv = groundtruth_rm_rw();
        const auto arr = outarr.flatten();
        check_array(gtv, arr, tol);
    }

    void check_output_cm_rw(ndarray<float_t, 1>& outarr, const float_t tol = 1.e-5) {
        CAPTURE(__func__, width_, height_, outarr.get_count());
        const auto gtv = groundtruth_rm_cw();
        const auto arr = outarr.flatten();
        check_array(gtv, arr, tol);
    }

    void check_output_rm_cw(ndarray<float_t, 1>& outarr, const float_t tol = 1.e-5) {
        CAPTURE(__func__, width_, height_, outarr.get_count());
        const auto gtv = groundtruth_rm_cw();
        const auto arr = outarr.flatten();
        check_array(gtv, arr, tol);
    }

    void check_output_cm_cw(ndarray<float_t, 1>& outarr, const float_t tol = 1.e-5) {
        CAPTURE(__func__, width_, height_);
        const auto gtv = groundtruth_rm_rw();
        const auto arr = outarr.flatten();
        check_array(gtv, arr, tol);
    }

    void test_rm_rw_reduce() {
        constexpr auto alloc = sycl::usm::alloc::device;
        auto input_array = row_accessor<const float_t>{ input_table_ } //
                               .pull(this->get_queue(), { 0, -1 }, alloc);
        auto [output_arr, out_event] = output(height_);

        auto input =
            ndview<float_t, 2, ndorder::c>::wrap(input_array.get_data(), { height_, width_ });

        auto reduce_event = reduce_by_rows(this->get_queue(),
                                           input,
                                           output_arr,
                                           binary_,
                                           unary_,
                                           { out_event },
                                           override_init_);

        auto host_output = output_arr.to_host(this->get_queue(), { reduce_event });

        check_output_rm_rw(host_output);
    }

    void test_rm_cw_reduce() {
        constexpr auto alloc = sycl::usm::alloc::device;
        auto input_array = row_accessor<const float_t>{ input_table_ } //
                               .pull(this->get_queue(), { 0, -1 }, alloc);
        auto [output_arr, out_event] = output(width_);
        auto input =
            ndview<float_t, 2, ndorder::c>::wrap(input_array.get_data(), { height_, width_ });

        auto reduce_event = reduce_by_columns(this->get_queue(),
                                              input,
                                              output_arr,
                                              binary_,
                                              unary_,
                                              { out_event },
                                              override_init_);

        auto host_output = output_arr.to_host(this->get_queue(), { reduce_event });

        check_output_rm_cw(host_output);
    }

    void test_cm_cw_reduce() {
        constexpr auto alloc = sycl::usm::alloc::device;
        auto input_array = row_accessor<const float_t>{ input_table_ } //
                               .pull(this->get_queue(), { 0, -1 }, alloc);
        auto [output_arr, out_event] = output(height_);
        auto input_tr =
            ndview<float_t, 2, ndorder::c>::wrap(input_array.get_data(), { height_, width_ });
        auto input = input_tr.t();

        auto reduce_event = reduce_by_columns(this->get_queue(),
                                              input,
                                              output_arr,
                                              binary_,
                                              unary_,
                                              { out_event },
                                              override_init_);

        auto host_output = output_arr.to_host(this->get_queue(), { reduce_event });

        check_output_cm_cw(host_output);
    }

    void test_cm_rw_reduce() {
        constexpr auto alloc = sycl::usm::alloc::device;
        auto input_array = row_accessor<const float_t>{ input_table_ } //
                               .pull(this->get_queue(), { 0, -1 }, alloc);
        auto [output_arr, out_event] = output(width_);
        auto input_tr =
            ndview<float_t, 2, ndorder::c>::wrap(input_array.get_data(), { height_, width_ });
        auto input = input_tr.t();

        auto reduce_event = reduce_by_rows(this->get_queue(),
                                           input,
                                           output_arr,
                                           binary_,
                                           unary_,
                                           { out_event },
                                           override_init_);

        auto host_output = output_arr.to_host(this->get_queue(), { reduce_event });

        check_output_cm_rw(host_output);
    }

protected:
    const binary_t binary_{};
    const unary_t unary_{};
    std::int64_t height_;
    std::int64_t width_;
    table offset_table_;
    table input_table_;
    bool override_init_;
};

TEMPLATE_LIST_TEST_M(reduction_test_random,
                     "Randomly filled reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_rm_rw_reduce();
    this->test_rm_cw_reduce();
    this->test_cm_cw_reduce();
    this->test_cm_rw_reduce();
}

template <typename Param>
class infinite_sum_test_random : public reduction_test_random<Param> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate(bool maxval) {
        this->height_ = GENERATE(17, 999, 1, 5, 1001);
        this->width_ = GENERATE(7, 707, 1, 251, 5);
        this->override_init_ = true; // poorly named variable
        CAPTURE(this->override_init_, this->width_, this->height_, maxval);
        generate_input(maxval);
        this->generate_offset();
    }

    void generate_input(bool maxval) {
        float mininp = 0.9 * (float)maxval * std::numeric_limits<float>::max() - 1.0f;
        float maxinp = (float)maxval * std::numeric_limits<float>::max();
        const auto train_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ this->height_, this->width_ }.fill_uniform(mininp, maxinp));
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());
    }
};

TEMPLATE_LIST_TEST_M(infinite_sum_test_random,
                     "Randomly filled reduction with infinte sum",
                     "[reduction][rm][small]",
                     finiteness_types) {
    // Temporary workaround: skip tests on architectures that do not support native float64
    SKIP_IF(!this->get_policy().has_native_float64());
    const bool use_infnan = GENERATE(0, 1);
    this->generate(use_infnan);
    SECTION("Reduce Row-Major by Rows") {
        this->test_rm_rw_reduce();
    }
    SECTION("Reduce Row-Major by Cols") {
        this->test_rm_cw_reduce();
    }
    SECTION("Reduce Col-Major by Rows") {
        this->test_cm_cw_reduce();
    }
    SECTION("Reduce Row-Major by Cols") {
        this->test_cm_rw_reduce();
    }
}

template <typename Param>
class single_infinite_test_random : public reduction_test_random<Param> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate(bool infval) {
        this->height_ = GENERATE(17, 999, 1, 5, 1001);
        this->width_ = GENERATE(7, 707, 1, 251, 5);
        this->override_init_ = true; // poorly named variable
        CAPTURE(this->override_init_, this->width_, this->height_, infval);
        generate_input(infval);
        this->generate_offset();
    }

    void generate_input(bool infval) {
        const auto train_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ this->height_, this->width_ }.fill_uniform(-3.0, 4.0));
        auto train_data = train_dataframe.get_array().get_mutable_data();

        // train_data is a float array
        train_data[5] = infval ? std::numeric_limits<float>::infinity()
                               : std::numeric_limits<float>::quiet_NaN();
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());
    }
};

TEMPLATE_LIST_TEST_M(single_infinite_test_random,
                     "Randomly filled reduction with single inf or nan",
                     "[reduction][rm][small]",
                     finiteness_types) {
    // Temporary workaround: skip tests on architectures that do not support native float64
    SKIP_IF(!this->get_policy().has_native_float64());
    const bool use_infnan = GENERATE(0, 1);
    this->generate(use_infnan);
    SECTION("Reduce Row-Major by Rows") {
        this->test_rm_rw_reduce();
    }
    SECTION("Reduce Row-Major by Cols") {
        this->test_rm_cw_reduce();
    }
    SECTION("Reduce Col-Major by Rows") {
        this->test_cm_cw_reduce();
    }
    SECTION("Reduce Col-Major by Cols") {
        this->test_cm_rw_reduce();
    }
}

} // namespace oneapi::dal::backend::primitives::test
