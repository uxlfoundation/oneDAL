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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/test/engine/io.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/backend/primitives/stat/cov.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace pr = dal::backend::primitives;

template <typename Float>
class cov_test : public te::float_algo_fixture<Float> {
public:
    ndarray<Float, 2> generate_diagonal_data(std::int64_t row_count,
                                             std::int64_t column_count,
                                             Float diag_element) {
        ONEDAL_ASSERT(row_count >= column_count);

        auto data = ndarray<Float, 2>::zeros({ row_count, column_count });
        Float* data_ptr = data.get_mutable_data();
        for (std::int64_t i = 0; i < column_count; i++) {
            data_ptr[i * column_count + i] = diag_element;
        }

        return data.to_device(this->get_queue());
    }

    auto allocate_arrays(std::int64_t column_count) {
        auto& q = this->get_queue();
        auto sums = ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
        auto corr =
            ndarray<Float, 2>::empty(q, { column_count, column_count }, sycl::usm::alloc::device);
        auto cov =
            ndarray<Float, 2>::empty(q, { column_count, column_count }, sycl::usm::alloc::device);
        auto means = ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
        auto vars = ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
        return std::make_tuple(sums, corr, cov, means, vars);
    }

    void check_nans(const ndarray<Float, 2>& corr) {
        const auto corr_mat = la::matrix<Float>::wrap_nd(corr.to_host(this->get_queue()));
        la::enumerate_linear(corr_mat, [](std::int64_t i, Float x) {
            CAPTURE(i, x);
            REQUIRE(x == x);
        });
    }

    void check_diagonal_is_ones(const ndarray<Float, 2>& corr) {
        const auto corr_mat = la::matrix<Float>::wrap_nd(corr.to_host(this->get_queue()));
        const double eps = te::get_tolerance<Float>(1e-4, 1e-6);

        la::enumerate(corr_mat, [&](std::int64_t i, std::int64_t j, double x) {
            if (i == j) {
                if (std::abs(x - 1.0) > eps) {
                    CAPTURE(i, j, x, eps);
                    FAIL("Unexpected diagonal element of correlation matrix");
                }
            }
        });
    }

    void check_correlation_for_diagonal_matrix(const ndarray<Float, 2>& corr,
                                               double expected_off_diagonal_element) {
        const auto corr_mat = la::matrix<Float>::wrap_nd(corr.to_host(this->get_queue()));
        const double eps = te::get_tolerance<Float>(1e-4, 1e-6);

        la::enumerate(corr_mat, [&](std::int64_t i, std::int64_t j, double x) {
            if (i == j) {
                if (std::abs(x - 1.0) > eps) {
                    CAPTURE(i, j, x, eps);
                    FAIL("Unexpected diagonal element of correlation matrix");
                }
            }
            else {
                if (std::abs(x - expected_off_diagonal_element) > eps) {
                    CAPTURE(i, j, x, eps);
                    FAIL("Unexpected non-diagonal element of correlation matrix");
                }
            }
        });
    }

    void check_constant_variance(const ndarray<Float, 1>& vars,
                                 std::int64_t row_count,
                                 double expected_var) {
        const auto vars_mat = la::matrix<Float>::wrap_nd(vars.to_host(this->get_queue()));

        const double corrected_var = expected_var > 0 ? expected_var : 1.0;
        const double eps = std::abs(corrected_var) * te::get_tolerance_for_sum<Float>(row_count);

        la::enumerate_linear(vars_mat, [&](std::int64_t i, double var) {
            if (std::abs(var - expected_var) > eps) {
                CAPTURE(i, var, expected_var);
                FAIL("Unexpected variance");
            }
        });
    }

    void check_constant_mean(const ndarray<Float, 1>& means,
                             std::int64_t row_count,
                             double expected_mean) {
        const auto means_mat = la::matrix<Float>::wrap_nd(means.to_host(this->get_queue()));

        const double corrected_mean = expected_mean > 0 ? expected_mean : 1.0;
        const double eps = std::abs(corrected_mean) * te::get_tolerance_for_sum<Float>(row_count);

        la::enumerate_linear(means_mat, [&](std::int64_t i, double mean) {
            if (std::abs(mean - expected_mean) > eps) {
                CAPTURE(i, mean, expected_mean);
                FAIL("Unexpected mean");
            }
        });
    }

    auto get_gold_input() {
        constexpr std::int64_t row_count = 10;
        constexpr std::int64_t column_count = 5;
        const Float data_host[row_count * column_count] = {
            4.59,  0.81,  -1.37, -0.04, -0.75, //
            4.87,  0.34,  -0.98, 4.1,   -0.12, //
            4.44,  0.11,  -0.4,  3.27,  4.82, //
            0.59,  0.98,  -1.88, -0.64, 2.54, //
            -1.98, 2.57,  4.11,  -1.3,  -0.66, //
            3.26,  2.8,   2.65,  0.83,  2.12, //
            0.21,  4.23,  2.71,  2.2,   3.85, //
            1.27,  -1.15, 2.84,  1.11,  -1.12, //
            0.25,  1.61,  1.69,  4.51,  0.09, //
            -0.01, 0.58,  0.83,  2.73,  -1.33, //
        };
        const Float sums_host[column_count] = {
            17.49, 12.88, 10.2, 16.77, 9.44,
        };

        auto data_host_arr = ndarray<Float, 2>::wrap(data_host, { row_count, column_count });
        auto sums_host_arr = ndarray<Float, 1>::wrap(sums_host, column_count);
        auto data = data_host_arr.to_device(this->get_queue());
        auto sums = sums_host_arr.to_device(this->get_queue());

        return std::make_tuple(data, sums);
    }

    auto get_gold_result() {
        constexpr std::int64_t column_count = 5;
        const Float corr_host[column_count * column_count] = {
            1.,          -0.36877335, -0.60139209, 0.30105244,  0.21180973, //
            -0.36877335, 1.,          0.44353402,  -0.16607388, 0.36798015, //
            -0.60139209, 0.44353402,  1.,          -0.14177578, -0.14416016, //
            0.30105244,  -0.16607388, -0.14177578, 1.,          0.11189629, //
            0.21180973,  0.36798015,  -0.14416016, 0.11189629,  1.
        };
        const Float means_host[column_count] = {
            1.749, 1.288, 1.02, 1.677, 0.944,
        };
        const Float vars_host[column_count] = {
            5.61381, 2.41595111, 4.333, 4.00386778, 4.90371556,
        };

        auto corr = ndarray<Float, 2>::copy(corr_host, { column_count, column_count });
        auto means = ndarray<Float, 1>::copy(means_host, { column_count });
        auto vars = ndarray<Float, 1>::copy(vars_host, { column_count });

        return std::make_tuple(corr, means, vars);
    }

    void check_gold_results(const ndarray<Float, 2>& corr,
                            const ndarray<Float, 1>& means,
                            const ndarray<Float, 1>& vars) {
        const auto [gold_corr, gold_means, gold_vars] = get_gold_result();
        const double eps = te::get_tolerance<Float>(1e-4, 1e-7);

        SECTION("compare correlation matrix with gold") {
            const auto corr_mat = la::matrix<Float>::wrap_nd(corr.to_host(this->get_queue()));
            const auto gold_corr_mat = la::matrix<Float>::wrap_nd(gold_corr);

            REQUIRE(la::equal_approx(corr_mat, gold_corr_mat, eps).all());
        }

        SECTION("compare means with gold") {
            const auto means_mat = la::matrix<Float>::wrap_nd(means.to_host(this->get_queue()));
            const auto gold_means_mat = la::matrix<Float>::wrap_nd(gold_means);

            REQUIRE(la::equal_approx(means_mat, gold_means_mat, eps).all());
        }

        SECTION("compare vars with gold") {
            const auto vars_mat = la::matrix<Float>::wrap_nd(vars.to_host(this->get_queue()));
            const auto gold_vars_mat = la::matrix<Float>::wrap_nd(gold_vars);

            REQUIRE(la::equal_approx(vars_mat, gold_vars_mat, eps).all());
        }
    }
};

using cov_types = COMBINE_TYPES((float, double));

TEMPLATE_LIST_TEST_M(cov_test, "correlation on diagonal data", "[cor]", cov_types) {
    using float_t = TestType;
    using dim2_t = std::tuple<std::int64_t, std::int64_t>;

    // DPC++ GEMM used underneath correlation is not supported on GPU
    SKIP_IF(this->get_policy().is_cpu());

    // Test takes too long time if HW emulates float64
    SKIP_IF(this->not_float64_friendly());

    const float_t diag_element = 10.5;
    const auto [row_count, column_count] =
        GENERATE(dim2_t(10, 10), dim2_t(100, 10), dim2_t(1000, 100));

    // Generate dataset, where the upper square part of the matrix is diagonal
    // and the rest are zeros, for example:
    // [ x 0 0 ]
    // [ 0 x 0 ]
    // [ 0 0 x ]
    // [ 0 0 0 ]
    const auto data = this->generate_diagonal_data(row_count, column_count, diag_element);
    const bool bias = false;
    const bool assume_centered = false;

    auto [sums, corr, cov, means, vars] = this->allocate_arrays(column_count);
    auto sums_event = sums.fill(this->get_queue(), diag_element);
    INFO("run correlation");
    auto gemm_event_cov =
        pr::gemm(this->get_queue(), data.t(), data, cov, float_t(1), float_t(0), { sums_event });
    auto gemm_event_corr = pr::gemm(this->get_queue(),
                                    data.t(),
                                    data,
                                    corr,
                                    float_t(1),
                                    float_t(0),
                                    { gemm_event_cov });
    pr::means(this->get_queue(), data.get_dimension(0), sums, means, { gemm_event_corr });
    auto cov_event = pr::covariance(this->get_queue(),
                                    data.get_dimension(0),
                                    sums,
                                    cov,
                                    bias,
                                    assume_centered,
                                    { gemm_event_corr });
    pr::variances(this->get_queue(), cov, vars, { cov_event }).wait_and_throw();
    auto corr_event =
        correlation(this->get_queue(), data.get_dimension(0), sums, corr, { gemm_event_corr });

    // `correlation` holds its own device scratch for the intermediate diagonal, so it must not
    // hand back an event that is still in flight: `sycl::free` does not synchronize, and the
    // scratch would be released from under the finalize kernel.
    const auto corr_status =
        corr_event.template get_info<sycl::info::event::command_execution_status>();
    REQUIRE(corr_status == sycl::info::event_command_status::complete);

    corr_event.wait_and_throw();

    // The upper part of data matrix is diagonal. In diagonal matrix each column
    // contains only one non-zero element (`diag_element`), so mean and
    // variances for each feature can be computed trivially using `diag_element`
    // value.

    SECTION("check if correlation matrix for diagonal matrix") {
        const double n = row_count;
        const double off_diag_element = -1.0 / (n - 1.0);
        this->check_correlation_for_diagonal_matrix(corr, off_diag_element);
    }

    SECTION("check if mean is expected") {
        const double n = row_count;
        const double expected_mean = double(diag_element) / n;
        this->check_constant_mean(means, n, expected_mean);
    }

    SECTION("check if variance is expected") {
        const double n = row_count;
        const double d = double(diag_element) * double(diag_element);
        ONEDAL_ASSERT(n > 1);
        const double expected_var = (d - d / n) / (n - 1.0);
        this->check_constant_variance(vars, n, expected_var);
    }
}

using cov_types_one_row = COMBINE_TYPES((float));

TEMPLATE_LIST_TEST_M(cov_test, "correlation on one-row table", "[cor]", cov_types_one_row) {
    using float_t = TestType;
    // DPC++ GEMM used underneath correlation is not supported on GPU
    SKIP_IF(this->get_policy().is_cpu());

    constexpr std::int64_t column_count = 3;
    const float data_ptr[column_count] = { 0.1f, 0.2f, 0.3f };
    const auto data_host = ndarray<float_t, 2>::wrap(data_ptr, { 1, column_count });
    const auto data = data_host.to_device(this->get_queue());
    const bool bias = false;
    const bool assume_centered = false;

    auto [sums, corr, cov, means, vars] = this->allocate_arrays(column_count);

    auto sums_event = sums.assign(this->get_queue(), data.get_data(), column_count);
    auto gemm_event_cov =
        pr::gemm(this->get_queue(), data.t(), data, cov, float_t(1), float_t(0), {});
    auto gemm_event_corr =
        pr::gemm(this->get_queue(), data.t(), data, corr, float_t(1), float_t(0), {});

    auto cov_event = pr::covariance(this->get_queue(),
                                    data.get_dimension(0),
                                    sums,
                                    cov,
                                    bias,
                                    assume_centered,
                                    { gemm_event_cov });
    auto var_event = pr::variances(this->get_queue(), cov, vars, { cov_event });
    auto corr_event =
        correlation(this->get_queue(), data.get_dimension(0), sums, corr, { gemm_event_corr });

    INFO("check if there is no NaNs in correlation matrix");
    corr_event.wait_and_throw();
    this->check_nans(corr);

    INFO("check if diagonal elements are ones");
    this->check_diagonal_is_ones(corr);

    INFO("check if variance is zero");
    var_event.wait_and_throw();
    this->check_constant_variance(vars, 1, 0.0);
}

TEMPLATE_LIST_TEST_M(cov_test, "correlation on gold data", "[cor]", cov_types) {
    using float_t = TestType;
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    auto [data, sums] = this->get_gold_input();
    auto [_, corr, cov, means, vars] = this->allocate_arrays(data.get_dimension(1));
    const bool bias = false;
    const bool assume_centered = false;

    INFO("run correlation");
    auto gemm_event_cov = pr::gemm(this->get_queue(), data.t(), data, cov, float_t(1), float_t(0));
    auto gemm_event_corr = pr::gemm(this->get_queue(),
                                    data.t(),
                                    data,
                                    corr,
                                    float_t(1),
                                    float_t(0),
                                    { gemm_event_cov });
    pr::means(this->get_queue(), data.get_dimension(0), sums, means, { gemm_event_corr });
    pr::covariance(this->get_queue(),
                   data.get_dimension(0),
                   sums,
                   cov,
                   bias,
                   assume_centered,
                   { gemm_event_corr });
    pr::variances(this->get_queue(), cov, vars, { gemm_event_corr });
    correlation(this->get_queue(), data.get_dimension(0), sums, corr, { gemm_event_corr })
        .wait_and_throw();

    this->check_gold_results(corr, means, vars);
}

} // namespace oneapi::dal::backend::primitives::test
