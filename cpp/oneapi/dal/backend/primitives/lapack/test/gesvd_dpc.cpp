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

#include <cmath>
#include <vector>

#include "oneapi/dal/backend/primitives/lapack/gesvd.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename Float>
class gesvd_test : public te::float_algo_fixture<Float> {
public:
    using float_t = Float;

    std::int64_t generate_dim() const {
        return GENERATE(3, 28, 125);
    }

    /// Runs `gesvd` over a square matrix and returns the singular values on the host.
    ///
    /// The decomposition is requested exactly the way the PCA SVD backend requests it -- the
    /// left singular vectors only, with oneMKL's Fortran-style argument order -- so that this
    /// test covers the same instantiation the library actually ships.
    ///
    /// @param[in] input The square matrix to decompose. It is copied, `gesvd` overwrites its
    ///                  input.
    ///
    /// @return The `dim` singular values, in the order `gesvd` produced them.
    auto call_gesvd(const la::matrix<Float>& input) {
        ONEDAL_ASSERT(input.get_row_count() == input.get_column_count());

        auto& q = this->get_queue();
        const std::int64_t dim = input.get_row_count();

        const auto input_copy = input.copy().get_array();
        auto data_host = ndarray<Float, 2>::wrap_mutable(input_copy, { dim, dim });
        auto data = data_host.to_device(q);
        auto s = ndarray<Float, 1>::empty(q, { dim }, sycl::usm::alloc::device);
        auto u = ndarray<Float, 2>::empty(q, { dim, dim }, sycl::usm::alloc::device);
        auto vt = ndarray<Float, 2>::empty(q, { 1, 1 }, sycl::usm::alloc::device);

        auto gesvd_event = gesvd<mkl::jobsvd::somevec,
                                 mkl::jobsvd::novec>(q, dim, dim, data, dim, s, u, dim, vt, 1, {});

        // `gesvd` allocates the LAPACK scratchpad itself, so it must not hand back an event that
        // is still in flight: `sycl::free` does not synchronize, and the scratchpad would be
        // released from under the decomposition.
        const auto status =
            gesvd_event.template get_info<sycl::info::event::command_execution_status>();
        REQUIRE(status == sycl::info::event_command_status::complete);

        gesvd_event.wait_and_throw();

        return la::matrix<Float>::wrap_nd(s.to_host(q));
    }

    void check_singular_values_are_descending(const la::matrix<Float>& s) const {
        INFO("check singular values are non-negative and ordered descending");
        la::enumerate_linear(s, [&](std::int64_t i, Float x) {
            CAPTURE(i, x);
            REQUIRE(x >= Float(0));
            if (i > 0) {
                REQUIRE(s.get(i - 1) >= x);
            }
        });
    }

    /// The squared singular values of any matrix sum to its squared Frobenius norm. This holds
    /// for an arbitrary input and needs no reference decomposition to check against.
    void check_frobenius_norm(const la::matrix<Float>& input, const la::matrix<Float>& s) const {
        double sum_of_squares = 0.0;
        la::enumerate_linear(input, [&](std::int64_t, Float x) {
            sum_of_squares += double(x) * double(x);
        });

        double sum_of_squared_singular_values = 0.0;
        la::enumerate_linear(s, [&](std::int64_t, Float x) {
            sum_of_squared_singular_values += double(x) * double(x);
        });

        const double tol = te::get_tolerance<Float>(1e-4, 1e-10) * sum_of_squares;
        CAPTURE(sum_of_squares, sum_of_squared_singular_values, tol);
        REQUIRE(std::abs(sum_of_squares - sum_of_squared_singular_values) < tol);
    }

    la::matrix<Float> generate_uniform() {
        const std::int64_t dim = this->generate_dim();
        return la::generate_uniform_matrix<Float>({ dim, dim }, -1, 1, seed_);
    }

    /// A diagonal matrix has its absolute diagonal entries as singular values, which makes the
    /// expected result exactly computable without a reference solver.
    ///
    /// @return The matrix together with its singular values in descending order.
    auto generate_diagonal() {
        const std::int64_t dim = this->generate_dim();
        auto result = la::matrix<Float>::zeros({ dim, dim });
        std::vector<Float> expected;
        for (std::int64_t i = 0; i < dim; ++i) {
            const Float value = Float(dim - i);
            result.set(i, i) = value;
            expected.push_back(value);
        }
        return std::make_tuple(result, expected);
    }

    void check_singular_values_against(const std::vector<Float>& expected,
                                       const la::matrix<Float>& s) const {
        INFO("check singular values against the diagonal");
        REQUIRE(s.get_count() == std::int64_t(expected.size()));
        for (std::int64_t i = 0; i < s.get_count(); ++i) {
            const double tol = te::get_tolerance<Float>(1e-4, 1e-10) * expected[i];
            CAPTURE(i, expected[i], s.get(i), tol);
            REQUIRE(std::abs(double(expected[i]) - double(s.get(i))) < tol);
        }
    }

private:
    static constexpr int seed_ = 7777;
};

using gesvd_types = COMBINE_TYPES((float, double));

TEMPLATE_LIST_TEST_M(gesvd_test, "gesvd on a diagonal matrix", "[gesvd]", gesvd_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto [input, expected] = this->generate_diagonal();
    const auto s = this->call_gesvd(input);

    this->check_singular_values_are_descending(s);
    this->check_singular_values_against(expected, s);
}

TEMPLATE_LIST_TEST_M(gesvd_test, "gesvd on a uniform random matrix", "[gesvd]", gesvd_types) {
    SKIP_IF(this->not_float64_friendly());

    const auto input = this->generate_uniform();
    const auto s = this->call_gesvd(input);

    this->check_singular_values_are_descending(s);
    this->check_frobenius_norm(input, s);
}

} // namespace oneapi::dal::backend::primitives::test
