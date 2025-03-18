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

#include "oneapi/dal/backend/primitives/lapack/syevd.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename Float>
class syevd_test : public te::float_algo_fixture<Float> {
public:
    using float_t = Float;
    std::int64_t generate_dim() const {
        return GENERATE(3, 28, 125, 256);
    }

    la::matrix<Float> generate_symmetric_positive() {
        const std::int64_t dim = this->generate_dim();
        return la::generate_symmetric_positive_matrix<Float>(dim, -1, 1, seed_);
    }

    auto call_sym_eigvals_inplace(const la::matrix<Float>& symmetric_matrix) {
        constexpr bool is_ascending = true;
        return call_sym_eigvals_inplace_generic(symmetric_matrix, is_ascending);
    }

    auto call_sym_eigvals_inplace_descending(const la::matrix<Float>& symmetric_matrix) {
        constexpr bool is_ascending = false;
        return call_sym_eigvals_inplace_generic(symmetric_matrix, is_ascending);
    }

    auto call_sym_eigvals_inplace_generic(const la::matrix<Float>& symmetric_matrix,
                                          bool is_ascending) {
        ONEDAL_ASSERT(symmetric_matrix.get_row_count() == symmetric_matrix.get_column_count());

        const std::int64_t dim = symmetric_matrix.get_row_count();
        const auto s_copy_flat = symmetric_matrix.copy().get_array();

        auto data_or_eigenvectors_nd = ndarray<Float, 2>::wrap_mutable(s_copy_flat, { dim, dim });
        data_or_eigenvectors_nd.to_device(this->get_queue());
        auto eigenvalues_nd =
            ndarray<Float, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::device);
        if (is_ascending) {
            auto syevd_event = syevd<mkl::job::vec, mkl::uplo::upper>(this->get_queue(),
                                                                      dim,
                                                                      data_or_eigenvectors_nd,
                                                                      dim,
                                                                      eigenvalues_nd,
                                                                      {});
            syevd_event.wait_and_throw();
            const auto eigenvectors =
                la::matrix<Float>::wrap_nd(data_or_eigenvectors_nd.to_host(this->get_queue()));
            const auto eigenvalues =
                la::matrix<Float>::wrap_nd(eigenvalues_nd.to_host(this->get_queue()));
            return std::make_tuple(eigenvectors, eigenvalues);
        }
        else {
            auto syevd_event = syevd<mkl::job::vec, mkl::uplo::upper>(this->get_queue(),
                                                                      dim,
                                                                      data_or_eigenvectors_nd,
                                                                      dim,
                                                                      eigenvalues_nd,
                                                                      {});
            syevd_event.wait_and_throw();

            auto data_ptr = eigenvalues_nd.get_data();
            auto flipped_eigenvalues =
                ndarray<Float, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::device);
            auto flipped_eigenvalues_ptr = flipped_eigenvalues.get_mutable_data();
            auto queue = this->get_queue();
            auto flip_event = queue.submit([&](sycl::handler& h) {
                const auto range = make_range_1d(dim);
                h.depends_on({ syevd_event });
                h.parallel_for(range, [=](sycl::id<1> id) {
                    const std::int64_t col = id[0];
                    flipped_eigenvalues_ptr[col] = data_ptr[(dim - 1) - col];
                });
            });
            const auto eigenvectors =
                la::matrix<Float>::wrap_nd(data_or_eigenvectors_nd.to_host(this->get_queue()));
            const auto eigenvalues =
                la::matrix<Float>::wrap_nd(flipped_eigenvalues.to_host(this->get_queue()));
            return std::make_tuple(eigenvectors, eigenvalues);
        }
    }

    void check_eigvals_definition(const la::matrix<Float>& s,
                                  const la::matrix<Float>& eigvecs,
                                  const la::matrix<Float>& eigvals) const {
        INFO("convert results to float64");
        const auto s_f64 = la::astype<double>(s);
        const auto eigvals_f64 = la::astype<double>(eigvals);
        const auto eigvecs_f64 = la::astype<double>(eigvecs);

        INFO("check eigenvectors and eigenvalues definition");
        for (std::int64_t i = 0; i < eigvecs.get_row_count(); i++) {
            const auto v = la::transpose(eigvecs_f64.get_row(i));
            const double w = eigvals_f64.get(i);
            CAPTURE(i, w);

            // Input matrix is positive-definite, so all eigenvalues must be positive
            REQUIRE(w > 0);

            const double tol = te::get_tolerance<float>(1e-4, 1e-10) * w;

            // Check condition: $S \times v_i = w_i \dot v_i$
            const double err = la::rel_error(la::dot(s_f64, v), la::multiply(w, v), tol);
            REQUIRE(err < tol);
        }
    }

    void check_eigvals_with_eigen(const la::matrix<Float>& s,
                                  const la::matrix<Float>& eigvecs,
                                  const la::matrix<Float>& eigvals) const {
        INFO("convert results to float64");
        const auto s_f64 = la::astype<double>(s);
        const auto eigvals_f64 = la::astype<double>(eigvals);
        const auto eigvecs_f64 = la::astype<double>(eigvecs);
        std::int64_t row_count = s.get_row_count();
        std::int64_t column_count = s.get_column_count();
        const Float* data = s.get_data();

        Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix(row_count, column_count);
        for (int i = 0; i < eigen_matrix.rows(); ++i) {
            for (int j = 0; j < eigen_matrix.cols(); ++j) {
                eigen_matrix(i, j) = data[i * column_count + j];
            }
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>> es(
            eigen_matrix);

        auto eigenvalues = es.eigenvalues().real();
        INFO("oneDAL eigvals vs Eigen eigvals");
        la::enumerate_linear(eigvals_f64, [&](std::int64_t i, Float x) {
            REQUIRE(abs(eigvals_f64.get(i) - eigenvalues(i)) < 0.1);
        });

        INFO("oneDAL eigvectors vs Eigen eigvectors");
        auto eigenvectors = es.eigenvectors().real();

        const double* eigenvec_ptr = eigvecs_f64.get_data();
        //TODO: investigate Eigen classes and align checking between oneDAL and Eigen classes.
        for (int j = 0; j < eigvecs.get_column_count(); ++j) {
            auto column_eigen = eigenvectors.col(j);
            for (int i = 0; i < eigvecs.get_row_count(); ++i) {
                REQUIRE((abs(eigenvec_ptr[j * row_count + i]) - abs(column_eigen(i))) < 0.1);
            }
        }
    }

    void check_eigvals_are_ascending(const la::matrix<Float>& eigvals) const {
        INFO("check eigenvalues order is ascending");
        la::enumerate_linear(eigvals, [&](std::int64_t i, Float x) {
            if (i > 0) {
                REQUIRE(eigvals.get(i - 1) <= x);
            }
        });
    }

    void check_eigvals_are_descending(const la::matrix<Float>& eigvals) const {
        INFO("check eigenvalues order is descending");
        la::enumerate_linear(eigvals, [&](std::int64_t i, Float x) {
            if (i > 0) {
                REQUIRE(eigvals.get(i - 1) >= x);
            }
        });
    }

private:
    static constexpr int seed_ = 7777;
};

using eigen_types = COMBINE_TYPES((float));

TEMPLATE_LIST_TEST_M(syevd_test, "test syevd with pos def matrix", "[sym_eigvals]", eigen_types) {
    const auto s = this->generate_symmetric_positive();
    const auto [eigenvectors, eigenvalues] = this->call_sym_eigvals_inplace(s);

    this->check_eigvals_definition(s, eigenvectors, eigenvalues);
    this->check_eigvals_are_ascending(eigenvalues);
    this->check_eigvals_with_eigen(s, eigenvectors, eigenvalues);
}

TEMPLATE_LIST_TEST_M(syevd_test, "test syevd with pos def matrix 2", "[sym_eigvals]", eigen_types) {
    const auto s = this->generate_symmetric_positive();

    const auto [eigenvectors, eigenvalues] = this->call_sym_eigvals_inplace_descending(s);

    this->check_eigvals_are_descending(eigenvalues);
}

} // namespace oneapi::dal::backend::primitives::test
