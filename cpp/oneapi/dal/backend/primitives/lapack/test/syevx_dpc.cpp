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

#include "oneapi/dal/backend/primitives/lapack/syevd.hpp"
#include "oneapi/dal/backend/primitives/lapack/syevx.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename Float>
class syevx_test : public te::float_algo_fixture<Float> {
public:
    la::matrix<Float> generate_symmetric_positive(std::int64_t dim) {
        return la::generate_symmetric_positive_matrix<Float>(dim, -1, 1, seed_);
    }

    /// Returns the eigenvectors (one per row) and the eigenvalues of the `count` largest
    /// eigenpairs of `s`, in ascending order.
    auto call_syevx(const la::matrix<Float>& s, std::int64_t count) {
        const std::int64_t dim = s.get_row_count();
        auto& q = this->get_queue();
        auto a = ndarray<Float, 2>::wrap_mutable(s.copy().get_array(), { dim, dim }).to_device(q);
        auto w = ndarray<Float, 1>::empty(q, { dim }, sycl::usm::alloc::device);
        auto z = ndarray<Float, 2>::empty(q, { count, dim }, sycl::usm::alloc::device);
        const auto event = syevx<mkl::uplo::upper>(q, a, dim - count + 1, dim, w, z, {});
        // syevx owns its scratchpad, so the event it returns must already be complete
        const auto status = event.template get_info<sycl::info::event::command_execution_status>();
        REQUIRE(status == sycl::info::event_command_status::complete);
        const auto w_host = w.to_host(q);
        auto eigenvalues = la::matrix<Float>::full({ 1, count }, Float(0));
        for (std::int64_t i = 0; i < count; ++i) {
            eigenvalues.set(0, i) = w_host.get_data()[i];
        }
        return std::make_tuple(la::matrix<Float>::wrap_nd(z.to_host(q)), eigenvalues);
    }

    /// Returns all eigenvalues of `s` in ascending order.
    la::matrix<Float> call_syevd_eigenvalues(const la::matrix<Float>& s) {
        const std::int64_t dim = s.get_row_count();
        auto& q = this->get_queue();
        auto a = ndarray<Float, 2>::wrap_mutable(s.copy().get_array(), { dim, dim }).to_device(q);
        auto w = ndarray<Float, 1>::empty(q, { dim }, sycl::usm::alloc::device);
        syevd<mkl::job::vec, mkl::uplo::upper>(q, dim, a, dim, w, {}).wait_and_throw();
        return la::matrix<Float>::wrap_nd(w.to_host(q));
    }

    void check_eigenpairs(const la::matrix<Float>& s,
                          const la::matrix<Float>& eigenvectors,
                          const la::matrix<Float>& eigenvalues) const {
        const auto s_f64 = la::astype<double>(s);
        const auto vectors_f64 = la::astype<double>(eigenvectors);
        for (std::int64_t i = 0; i < eigenvectors.get_row_count(); ++i) {
            const auto v = la::transpose(vectors_f64.get_row(i));
            const double w = eigenvalues.get(0, i);
            CAPTURE(i, w);
            REQUIRE(w > 0);
            const double tol = te::get_tolerance<Float>(1e-4, 1e-10) * w;
            REQUIRE(la::rel_error(la::dot(s_f64, v), la::multiply(w, v), tol) < tol);
        }
    }

private:
    static constexpr int seed_ = 7777;
};

using eigen_types = COMBINE_TYPES((float, double));

TEMPLATE_LIST_TEST_M(syevx_test,
                     "syevx returns the largest eigenpairs of syevd",
                     "[sym_eigvals]",
                     eigen_types) {
    SKIP_IF(this->not_float64_friendly());
    const std::int64_t dim = GENERATE(28, 125, 256);
    const std::int64_t count = GENERATE(1, 5, 20);
    CAPTURE(dim, count);

    const auto s = this->generate_symmetric_positive(dim);
    const auto [eigenvectors, eigenvalues] = this->call_syevx(s, count);
    const auto all_eigenvalues = this->call_syevd_eigenvalues(s);

    this->check_eigenpairs(s, eigenvectors, eigenvalues);
    for (std::int64_t i = 0; i < count; ++i) {
        const double expected = all_eigenvalues.get(dim - count + i);
        CAPTURE(i, expected);
        REQUIRE(std::abs(eigenvalues.get(0, i) - expected) <=
                te::get_tolerance<TestType>(1e-4, 1e-10) * std::abs(expected));
    }
}

} // namespace oneapi::dal::backend::primitives::test
