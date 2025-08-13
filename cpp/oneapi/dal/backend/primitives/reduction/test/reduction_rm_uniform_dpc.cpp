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
                                   std::tuple<double, sum<double>, identity<double>>,
                                   std::tuple<double, sum<double>, square<double>>,
                                   std::tuple<double, sum<double>, abs<double>>,
                                   std::tuple<float, logical_or<float>, isinfornan<float>>,
                                   std::tuple<float, logical_or<float>, isinf<float>>,
                                   std::tuple<double, logical_or<double>, isinfornan<double>>,
                                   std::tuple<double, logical_or<double>, isinf<double>>>;

template <typename Param>
class reduction_rm_test_uniform : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate() {
        arg_ = GENERATE(-7., 0, 3.);
        width_ = GENERATE(7, 707, 5);
        stride_ = GENERATE(707, 812, 1024);
        height_ = GENERATE(171, 999, 1001);
        CAPTURE(arg_, width_, stride_, height_);
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

    auto input() {
        check_if_initialized();
        return ndarray<float_t, 2, rm_order>::full(this->get_queue(), { stride_, height_ }, arg_);
    }

    auto output(std::int64_t size) {
        check_if_initialized();
        return ndarray<float_t, 1, rm_order>::full(this->get_queue(),
                                                   { size },
                                                   binary_t{}.init_value);
    }

    float_t val_rw() const {
        if (std::is_same_v<sum<float_t>, binary_t>) {
            if (std::is_same_v<identity<float_t>, unary_t>) {
                return width_ * arg_;
            }
            if (std::is_same_v<abs<float_t>, unary_t>) {
                return width_ * std::abs(arg_);
            }
            if (std::is_same_v<square<float_t>, unary_t>) {
                return width_ * (arg_ * arg_);
            }
        }
        if (std::is_same_v<min<float_t>, binary_t>) {
            if (std::is_same_v<identity<float_t>, unary_t>) {
                return arg_;
            }
            if (std::is_same_v<abs<float_t>, unary_t>) {
                return std::abs(arg_);
            }
            if (std::is_same_v<square<float_t>, unary_t>) {
                return (arg_ * arg_);
            }
        }
        if (std::is_same_v<max<float_t>, binary_t>) {
            if (std::is_same_v<identity<float_t>, unary_t>) {
                return arg_;
            }
            if (std::is_same_v<abs<float_t>, unary_t>) {
                return std::abs(arg_);
            }
            if (std::is_same_v<square<float_t>, unary_t>) {
                return (arg_ * arg_);
            }
        }
        if (std::is_same_v<logical_or<float_t>, binary_t>) {
            if (std::is_same_v<isinf<float_t>, unary_t>) {
                return static_cast<float>(std::isinf(arg_));
            }
            if (std::is_same_v<isinfornan<float_t>, unary_t>) {
                return static_cast<float>(std::isinf(arg_) || std::isnan(arg_));
            }
        }
        ONEDAL_ASSERT(false);
        return 0;
    }

    float_t val_cw() const {
        if (std::is_same_v<sum<float_t>, binary_t>) {
            if (std::is_same_v<identity<float_t>, unary_t>) {
                return height_ * arg_;
            }
            if (std::is_same_v<abs<float_t>, unary_t>) {
                return height_ * std::abs(arg_);
            }
            if (std::is_same_v<square<float_t>, unary_t>) {
                return height_ * (arg_ * arg_);
            }
        }
        if (std::is_same_v<min<float_t>, binary_t>) {
            if (std::is_same_v<identity<float_t>, unary_t>) {
                return arg_;
            }
            if (std::is_same_v<abs<float_t>, unary_t>) {
                return std::abs(arg_);
            }
            if (std::is_same_v<square<float_t>, unary_t>) {
                return (arg_ * arg_);
            }
        }
        if (std::is_same_v<max<float_t>, binary_t>) {
            if (std::is_same_v<identity<float_t>, unary_t>) {
                return arg_;
            }
            if (std::is_same_v<abs<float_t>, unary_t>) {
                return std::abs(arg_);
            }
            if (std::is_same_v<square<float_t>, unary_t>) {
                return (arg_ * arg_);
            }
        }
        if (std::is_same_v<logical_or<float_t>, binary_t>) {
            if (std::is_same_v<isinf<float_t>, unary_t>) {
                return static_cast<float_t>(std::isinf(arg_));
            }
            if (std::is_same_v<isinfornan<float_t>, unary_t>) {
                return static_cast<float_t>(std::isinf(arg_) || std::isnan(arg_));
            }
        }
        ONEDAL_ASSERT(false);
        return 0;
    }

    void check_output_rw(ndarray<float_t, 1, rm_order>& outarr, const float_t tol = 1.e-5) {
        CAPTURE(__func__, arg_, width_, height_, stride_);
        const auto gtv = val_rw();
        const auto arr = outarr.flatten();
        for (auto i = 0; i < height_; ++i) {
            const auto diff = arr[i] - gtv;
            if (diff < -tol || tol < diff) {
                CAPTURE(gtv, arr[i], diff, tol);
                FAIL();
            }
        }
    }

    void check_output_cw(ndarray<float_t, 1, rm_order>& outarr, const float_t tol = 1.e-5) {
        CAPTURE(__func__, arg_, width_, height_, stride_);
        const auto gtv = val_cw();
        const auto arr = outarr.flatten();
        for (auto i = 0; i < width_; ++i) {
            const auto diff = arr[i] - gtv;
            if (diff < -tol || tol < diff) {
                CAPTURE(gtv, arr[i], diff, tol);
                FAIL();
            }
        }
    }

    void test_raw_rw_reduce_narrow() {
        using reduction_t = reduction_rm_rw_narrow<float_t, binary_t, unary_t>;
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output(height_);

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(this->get_queue());
        reducer(inp_ptr,
                out_ptr,
                width_,
                height_,
                stride_,
                binary_t{},
                unary_t{},
                { inp_event, out_event })
            .wait_and_throw();

        check_output_rw(out_array);
    }

    void test_raw_rw_reduce_wide() {
        using reduction_t = reduction_rm_rw_wide<float_t, binary_t, unary_t>;
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output(height_);

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(this->get_queue());
        reducer(inp_ptr,
                out_ptr,
                width_,
                height_,
                stride_,
                binary_t{},
                unary_t{},
                { inp_event, out_event })
            .wait_and_throw();

        check_output_rw(out_array);
    }

    void test_raw_rw_reduce_wrapper() {
        using reduction_t = reduction_rm_rw<float_t, binary_t, unary_t>;
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output(height_);

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(this->get_queue());
        reducer(inp_ptr,
                out_ptr,
                width_,
                height_,
                stride_,
                binary_t{},
                unary_t{},
                { inp_event, out_event })
            .wait_and_throw();

        check_output_rw(out_array);
    }

    void test_raw_cw_reduce_naive() {
        using reduction_t = reduction_rm_cw_naive<float_t, binary_t, unary_t>;
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output(width_);

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(this->get_queue());
        reducer(inp_ptr,
                out_ptr,
                width_,
                height_,
                stride_,
                binary_t{},
                unary_t{},
                { inp_event, out_event })
            .wait_and_throw();

        check_output_cw(out_array);
    }

    void test_raw_cw_reduce_atomic() {
        using reduction_t = reduction_rm_cw_atomic<float_t, binary_t, unary_t>;
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output(width_);

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(this->get_queue());
        reducer(inp_ptr,
                out_ptr,
                width_,
                height_,
                stride_,
                binary_t{},
                unary_t{},
                { inp_event, out_event })
            .wait_and_throw();

        check_output_cw(out_array);
    }

    void test_raw_cw_reduce_wrapper() {
        using reduction_t = reduction_rm_cw<float_t, binary_t, unary_t>;
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output(width_);

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(this->get_queue());
        reducer(inp_ptr,
                out_ptr,
                width_,
                height_,
                stride_,
                binary_t{},
                unary_t{},
                { inp_event, out_event })
            .wait_and_throw();

        check_output_cw(out_array);
    }

private:
    float_t arg_;
    std::int64_t width_;
    std::int64_t stride_;
    std::int64_t height_;
};

TEMPLATE_LIST_TEST_M(reduction_rm_test_uniform,
                     "Uniformly filled Row-Major Row-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    SKIP_IF(this->should_be_skipped());
    this->test_raw_rw_reduce_wide();
    this->test_raw_rw_reduce_narrow();
    this->test_raw_rw_reduce_wrapper();
}

TEMPLATE_LIST_TEST_M(reduction_rm_test_uniform,
                     "Uniformly filled Row-Major Col-Wise reduction",
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

} // namespace oneapi::dal::backend::primitives::test
