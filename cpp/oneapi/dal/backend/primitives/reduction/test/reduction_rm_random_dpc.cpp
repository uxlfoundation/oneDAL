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

#include "oneapi/dal/backend/primitives/reduction/reduction_rm_rw_dpc.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

constexpr auto rm_order = ndorder::c;

using reduction_types = std::tuple<std::tuple<float, sum<float>, identity<float>>,
                                   std::tuple<float, sum<float>, square<float>>,
                                   std::tuple<float, sum<float>, abs<float>>,
                                   std::tuple<float, max<float>, identity<float>>,
                                   std::tuple<float, min<float>, identity<float>>,
                                   std::tuple<double, sum<double>, identity<double>>,
                                   std::tuple<double, sum<double>, square<double>>,
                                   std::tuple<double, sum<double>, abs<double>>,
                                   std::tuple<double, max<double>, identity<double>>,
                                   std::tuple<double, min<double>, identity<double>>>;

using finiteness_types = std::tuple<std::tuple<float, sum<float>, identity<float>>,
                                    std::tuple<double, sum<double>, identity<double>>,
                                    std::tuple<float, logical_or<bool>, isinfornan<float>>,
                                    std::tuple<float, logical_or<bool>, isinf<float>>,
                                    std::tuple<double, logical_or<bool>, isinfornan<double>>,
                                    std::tuple<double, logical_or<bool>, isinf<double>>>;

template <typename Param>
class reduction_rm_test_random : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate() {
        width_ = GENERATE(7, 707, 5);
        stride_ = GENERATE(707, 812, 1024);
        height_ = GENERATE(17, 999, 1, 1001);
        CAPTURE(width_, stride_, height_);
        generate_input();
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    auto output(std::int64_t size) {
        check_if_initialized();
        return ndarray<float_t, 1, rm_order>::full(this->get_queue(),
                                                   { size },
                                                   binary_t{}.init_value);
    }

    void generate_input() {
        const auto train_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ height_, stride_ }.fill_uniform(-0.2, 0.5));
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());
    }

    bool is_initialized() const {
        return width_ > 0 && stride_ > 0 && height_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    bool should_be_skipped() {
        if (width_ > stride_) {
            return true;
        }
        return false;
    }

    array<float_t> groundtruth_cw() const {
        auto res = array<float_t>::full(width_, static_cast<float_t>(binary_.init_value));
        auto* res_ptr = res.get_mutable_data();
        for (std::int64_t j = 0; j < height_; ++j) {
            //input_table_ is a float ndarray
            const auto row_acc = row_accessor<const float>{ input_table_ }.pull({ j, j + 1 });
            for (std::int64_t i = 0; i < width_; ++i) {
                const auto val = row_acc[i];
                res_ptr[i] = binary_(res_ptr[i], unary_(val));
            }
        }
        return res;
    }

    array<float_t> groundtruth_rw() const {
        auto res = array<float_t>::full(height_, static_cast<float_t>(binary_.init_value));
        auto* res_ptr = res.get_mutable_data();
        for (std::int64_t j = 0; j < height_; ++j) {
            const auto row_acc = row_accessor<const float>{ input_table_ }.pull({ j, j + 1 });
            for (std::int64_t i = 0; i < width_; ++i) {
                const auto val = row_acc[i];
                res_ptr[j] = binary_(res_ptr[j], unary_(val));
            }
        }
        return res;
    }

    void check_output_rw(ndarray<float_t, 1, rm_order>& outarr, const float_t tol = 1.e-3) {
        CAPTURE(__func__, width_, height_, stride_);
        const auto gtv = groundtruth_rw();
        const auto arr = outarr.flatten();
        for (auto i = 0; i < height_; ++i) {
            const auto diff = arr[i] - gtv[i];
            if (diff < -tol || tol < diff) {
                CAPTURE(i, gtv[i], arr[i], diff, tol);
                FAIL();
            }
        }
    }

    void check_output_cw(ndarray<float_t, 1, rm_order>& outarr, const float_t tol = 1.e-3) {
        CAPTURE(__func__, width_, height_, stride_);
        const auto gtv = groundtruth_cw();
        const auto arr = outarr.flatten();
        for (auto i = 0; i < width_; ++i) {
            const auto diff = arr[i] - gtv[i];
            if (diff < -tol || tol < diff) {
                CAPTURE(i, gtv[i], arr[i], diff, tol);
                FAIL();
            }
        }
    }

    void test_raw_rw_reduce_narrow() {
        using acc_t = bin_op_t<binary_t>;
        using reduction_t = reduction_rm_rw_narrow<float_t, acc_t, binary_t, unary_t>;
        const auto input_array =
            row_accessor<const float_t>{ input_table_ }.pull(this->get_queue());
        auto [out_array, out_event] = output(height_);

        const float_t* inp_ptr = input_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(this->get_queue());
        reducer(inp_ptr, out_ptr, width_, height_, stride_, binary_, unary_, { out_event })
            .wait_and_throw();

        check_output_rw(out_array);
    }

    void test_raw_rw_reduce_wide() {
        using acc_t = bin_op_t<binary_t>;
        using reduction_t = reduction_rm_rw_wide<float_t, acc_t, binary_t, unary_t>;
        const auto input_array =
            row_accessor<const float_t>{ input_table_ }.pull(this->get_queue());
        auto [out_array, out_event] = output(height_);

        const float_t* inp_ptr = input_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(this->get_queue());
        reducer(inp_ptr, out_ptr, width_, height_, stride_, binary_, unary_, { out_event })
            .wait_and_throw();

        check_output_rw(out_array);
    }

    void test_raw_rw_reduce_wrapper() {
        using acc_t = bin_op_t<binary_t>;
        using reduction_t = reduction_rm_rw<float_t, acc_t, binary_t, unary_t>;
        const auto input_array =
            row_accessor<const float_t>{ input_table_ }.pull(this->get_queue());
        auto [out_array, out_event] = output(height_);

        const float_t* inp_ptr = input_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(this->get_queue());
        reducer(inp_ptr, out_ptr, width_, height_, stride_, binary_, unary_, { out_event })
            .wait_and_throw();

        check_output_rw(out_array);
    }

    void test_raw_cw_reduce_naive() {
        using acc_t = bin_op_t<binary_t>;
        using reduction_t = reduction_rm_cw_naive<float_t, acc_t, binary_t, unary_t>;
        const auto input_array =
            row_accessor<const float_t>{ input_table_ }.pull(this->get_queue());
        auto [out_array, out_event] = output(width_);

        const float_t* inp_ptr = input_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(this->get_queue());
        reducer(inp_ptr, out_ptr, width_, height_, stride_, binary_, unary_, { out_event })
            .wait_and_throw();

        check_output_cw(out_array);
    }

    void test_raw_cw_reduce_atomic() {
        using acc_t = bin_op_t<binary_t>;
        using reduction_t = reduction_rm_cw_atomic<float_t, acc_t, binary_t, unary_t>;
        const auto input_array =
            row_accessor<const float_t>{ input_table_ }.pull(this->get_queue());
        auto [out_array, out_event] = output(width_);

        const float_t* inp_ptr = input_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(this->get_queue());
        reducer(inp_ptr, out_ptr, width_, height_, stride_, binary_, unary_, { out_event })
            .wait_and_throw();

        check_output_cw(out_array);
    }

    void test_raw_cw_reduce_wrapper() {
        using acc_t = bin_op_t<binary_t>;
        using reduction_t = reduction_rm_cw<float_t, acc_t, binary_t, unary_t>;
        const auto input_array =
            row_accessor<const float_t>{ input_table_ }.pull(this->get_queue());
        auto [out_array, out_event] = output(width_);

        const float_t* inp_ptr = input_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(this->get_queue());
        reducer(inp_ptr, out_ptr, width_, height_, stride_, binary_, unary_, { out_event })
            .wait_and_throw();

        check_output_cw(out_array);
    }

protected:
    const binary_t binary_{};
    const unary_t unary_{};
    std::int64_t width_;
    std::int64_t stride_;
    std::int64_t height_;
    table input_table_;
};

TEMPLATE_LIST_TEST_M(reduction_rm_test_random,
                     "Randomly filled Row-Major Row-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    SKIP_IF(this->should_be_skipped());
    this->test_raw_rw_reduce_wide();
    this->test_raw_rw_reduce_narrow();
    this->test_raw_rw_reduce_wrapper();
}

TEMPLATE_LIST_TEST_M(reduction_rm_test_random,
                     "Randomly filled Row-Major Col-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    SKIP_IF(this->should_be_skipped());
    this->test_raw_cw_reduce_naive();
    // TODO: Investigation into atomic reduction discrepancies ongoing
    // this->test_raw_cw_reduce_atomic();
    this->test_raw_cw_reduce_wrapper();
}

template <typename Param>
class infinite_sum_rm_test_random : public reduction_rm_test_random<Param> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate(bool maxval) {
        this->width_ = GENERATE(7, 707, 5);
        this->stride_ = GENERATE(707, 812, 1024);
        this->height_ = GENERATE(17, 999, 1, 1001);
        CAPTURE(this->width_, this->stride_, this->height_, maxval);
        generate_input(maxval);
    }

    void generate_input(bool maxval) {
        float mininp = 0.9 * (float)maxval * std::numeric_limits<float>::max() - 1.0f;
        float maxinp = (float)maxval * std::numeric_limits<float>::max();
        const auto train_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ this->height_, this->stride_ }.fill_uniform(mininp, maxinp));
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());
    }
};

TEMPLATE_LIST_TEST_M(infinite_sum_rm_test_random,
                     "Randomly filled Row-Major Row-Wise reduction with infinte sum",
                     "[reduction][rm][small]",
                     finiteness_types) {
    // Temporary workaround: skip tests on architectures that do not support native float64
    SKIP_IF(!this->get_policy().has_native_float64());
    const bool use_infnan = GENERATE(0, 1);
    this->generate(use_infnan);
    SKIP_IF(this->should_be_skipped());
    this->test_raw_rw_reduce_wide();
    this->test_raw_rw_reduce_narrow();
    this->test_raw_rw_reduce_wrapper();
}

TEMPLATE_LIST_TEST_M(infinite_sum_rm_test_random,
                     "Randomly filled Row-Major Col-Wise reduction with infinte sum",
                     "[reduction][rm][small]",
                     finiteness_types) {
    // Temporary workaround: skip tests on architectures that do not support native float64
    SKIP_IF(!this->get_policy().has_native_float64());
    const bool use_infnan = GENERATE(0, 1);
    this->generate(use_infnan);
    SKIP_IF(this->should_be_skipped());
    this->test_raw_cw_reduce_naive();
    // TODO: Investigation into atomic reduction discrepancies ongoing
    // this->test_raw_cw_reduce_atomic();
    this->test_raw_cw_reduce_wrapper();
}

template <typename Param>
class single_infinite_rm_test_random : public reduction_rm_test_random<Param> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;
    void generate(bool infval) {
        this->width_ = GENERATE(7, 707, 5);
        this->stride_ = GENERATE(707, 812, 1024);
        this->height_ = GENERATE(17, 999, 1, 1001);
        CAPTURE(this->width_, this->stride_, this->height_, infval);
        generate_input(infval);
    }

    void generate_input(bool infval) {
        float infinp = infval ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
        const auto train_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ this->height_, this->stride_ }.fill_uniform(-0.2, 0.5));

        // train_data is a float ndarray
        auto train_data = train_dataframe.get_array().get_mutable_data();
        train_data[0] = infinp;
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());
        // no inf added to see what will happen in testing
    }
};

TEMPLATE_LIST_TEST_M(single_infinite_rm_test_random,
                     "Randomly filled Row-Major Row-Wise reduction with single inf or nan",
                     "[reduction][rm][small]",
                     finiteness_types) {
    // Temporary workaround: skip tests on architectures that do not support native float64
    SKIP_IF(!this->get_policy().has_native_float64());
    const bool use_infnan = GENERATE(0, 1);
    this->generate(use_infnan);
    SKIP_IF(this->should_be_skipped());
    this->test_raw_rw_reduce_wide();
    this->test_raw_rw_reduce_narrow();
    this->test_raw_rw_reduce_wrapper();
}

TEMPLATE_LIST_TEST_M(single_infinite_rm_test_random,
                     "Randomly filled Row-Major Col-Wise reduction with single inf or nan",
                     "[reduction][rm][small]",
                     finiteness_types) {
    // Temporary workaround: skip tests on architectures that do not support native float64
    SKIP_IF(!this->get_policy().has_native_float64());
    const bool use_infnan = GENERATE(0, 1);
    this->generate(use_infnan);
    SKIP_IF(this->should_be_skipped());
    SECTION("Reduce Naive") {
        this->test_raw_cw_reduce_naive();
    }
    // TODO: Investigation into atomic reduction discrepancies ongoing
    //SECTION("Reduce Atomic") {
    // this->test_raw_cw_reduce_atomic();
    //}
    SECTION("Reduce Wrapper") {
        this->test_raw_cw_reduce_wrapper();
    }
}

} // namespace oneapi::dal::backend::primitives::test
