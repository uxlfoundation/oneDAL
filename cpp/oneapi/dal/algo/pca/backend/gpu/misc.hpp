/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"

namespace oneapi::dal::pca::backend {

#ifdef ONEDAL_DATA_PARALLEL

using alloc = sycl::usm::alloc;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

// Common

///  A wrapper that computes 1d array of sums of the columns from 2d data array
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  data  The input data of size `row_count` x `column_count`
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 1d array of sums
/// of size `column_count` and the second element is a SYCL event indicating the availability
/// of the sums array for reading and writing
template <typename Float>
auto compute_sums(sycl::queue& queue,
                  const pr::ndview<Float, 2>& data,
                  const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_sums, queue);
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(data.get_dimension(1) > 0);

    const std::int64_t column_count = data.get_dimension(1);
    auto sums = pr::ndarray<Float, 1>::empty(queue, { column_count }, alloc::device);
    constexpr pr::sum<Float> binary{};
    constexpr pr::identity<Float> unary{};
    auto sums_event = pr::reduce_by_columns(queue, data, sums, binary, unary, deps);
    return std::make_tuple(sums, sums_event);
}

///  A wrapper that computes 1d array of eigenvalues and 2d array of eigenvectors from the covariance matrix
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  corr  The input covariance/correlation matrix of size `column_count` x `column_count`
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 2d array of eigenvectors
/// of size `component_count` x `column_count` and the second element is the resulting 1d array of eigenvalues
template <typename Float>
auto syevd_computation(sycl::queue& queue,
                       pr::ndview<Float, 2>& corr,
                       const bk::event_vector& deps = {}) {
    sycl::event::wait_and_throw(deps);

    const std::int64_t column_count = corr.get_dimension(1);
    auto eigenvalues = pr::ndarray<Float, 1>::empty(queue, { column_count }, alloc::device);

    std::int64_t lda = column_count;

    sycl::event syevd_event;
    {
        syevd_event = pr::syevd<mkl::job::vec, mkl::uplo::upper>(queue,
                                                                 column_count,
                                                                 corr,
                                                                 lda,
                                                                 eigenvalues,
                                                                 { deps });
    }
    syevd_event.wait_and_throw();
    return std::make_tuple(eigenvalues, syevd_event);
}

///  A wrapper that flips 2d array of eigenvectors from the syevd result in necessary order
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  data  The input eigenvectors in ascending order of size `column_count` x `column_count`
/// @param[in]  component_count  The number of `component_count` of the descriptor
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return The resulting 2d array of eigenvectors
/// Flips both eigenvalues and eigenvectors in descending order in a single parallel operation
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in] queue The SYCL queue
/// @param[in] eigenvalues The input eigenvalues in ascending order (1D)
/// @param[in] eigenvectors The input eigenvectors (2D)
/// @param[in] component_count The number of components to process
/// @param[in] deps Events indicating availability of the data for reading or writing
///
/// @return Tuple containing flipped eigenvalues (1D) and eigenvectors (2D)
template <typename Float>
auto flip_eigen_data(sycl::queue& queue,
                     pr::ndview<Float, 1>& eigenvalues,
                     pr::ndview<Float, 2>& eigenvectors,
                     std::int64_t component_count,
                     const bk::event_vector& deps = {}) {
    // Get dimensions
    const std::int64_t eigval_count = eigenvalues.get_dimension(0);
    const std::int64_t row_count = eigenvectors.get_dimension(0);
    const std::int64_t column_count = eigenvectors.get_dimension(1);

    // Input pointers
    auto eigval_ptr = eigenvalues.get_data();
    auto eigvec_ptr = eigenvectors.get_data();

    // Create output arrays
    auto flipped_eigenvalues =
        pr::ndarray<Float, 1>::empty(queue, { component_count }, alloc::device);
    auto flipped_eigenvectors =
        pr::ndarray<Float, 2>::empty(queue, { component_count, column_count }, alloc::device);

    // Output pointers
    auto flipped_eigval_ptr = flipped_eigenvalues.get_mutable_data();
    auto flipped_eigvec_ptr = flipped_eigenvectors.get_mutable_data();

    auto flip_event = queue.submit([&](sycl::handler& h) {
        // Use 2D range to handle both arrays
        const auto range = bk::make_range_2d(component_count, column_count + 1);
        h.depends_on(deps);

        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::int64_t row = id[0];
            const std::int64_t col = id[1];

            // Process eigenvalues in column 0
            if (col == 0 && row < component_count) {
                flipped_eigval_ptr[row] = eigval_ptr[(eigval_count - 1) - row];
            }
            // Process eigenvectors in columns 1 to column_count
            if (col > 0 && col <= column_count) {
                flipped_eigvec_ptr[row * column_count + (col - 1)] =
                    eigvec_ptr[(row_count - 1 - row) * column_count + (col - 1)];
            }
        });
    });

    flip_event.wait_and_throw();

    // Transfer to host
    auto flipped_eigval_host = flipped_eigenvalues.to_host(queue);
    auto flipped_eigvec_host = flipped_eigenvectors.to_host(queue);

    return std::make_tuple(flipped_eigval_host, flipped_eigvec_host);
}

template <typename Float>
auto flip_eigen_data_gpu(sycl::queue& queue,
                         pr::ndview<Float, 1>& eigenvalues,
                         pr::ndview<Float, 2>& eigenvectors,
                         std::int64_t component_count,
                         const bk::event_vector& deps = {}) {
    // Get dimensions
    const std::int64_t eigval_count = eigenvalues.get_dimension(0);
    const std::int64_t row_count = eigenvectors.get_dimension(0);
    const std::int64_t column_count = eigenvectors.get_dimension(1);

    // Input pointers
    auto eigval_ptr = eigenvalues.get_data();
    auto eigvec_ptr = eigenvectors.get_data();

    // Create output arrays
    auto flipped_eigenvalues =
        pr::ndarray<Float, 1>::empty(queue, { component_count }, alloc::device);
    auto flipped_eigenvectors =
        pr::ndarray<Float, 2>::empty(queue, { component_count, column_count }, alloc::device);

    // Output pointers
    auto flipped_eigval_ptr = flipped_eigenvalues.get_mutable_data();
    auto flipped_eigvec_ptr = flipped_eigenvectors.get_mutable_data();

    auto flip_event = queue.submit([&](sycl::handler& h) {
        // Use 2D range to handle both arrays
        const auto range = bk::make_range_2d(component_count, column_count + 1);
        h.depends_on(deps);

        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::int64_t row = id[0];
            const std::int64_t col = id[1];

            // Process eigenvalues in column 0
            if (col == 0 && row < component_count) {
                flipped_eigval_ptr[row] = eigval_ptr[(eigval_count - 1) - row];
            }
            // Process eigenvectors in columns 1 to column_count
            if (col > 0 && col <= column_count) {
                flipped_eigvec_ptr[row * column_count + (col - 1)] =
                    eigvec_ptr[(row_count - 1 - row) * column_count + (col - 1)];
            }
        });
    });

    flip_event.wait_and_throw();

    return std::make_tuple(flipped_eigenvalues, flipped_eigenvectors);
}

template <typename Float>
void sign_flip_gpu(sycl::queue& queue,
                   pr::ndview<Float, 2>& eigvecs,
                   const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(eigvecs.get_dimension(0) > 0);
    ONEDAL_ASSERT(eigvecs.get_dimension(1) > 0);

    const std::int64_t row_count = eigvecs.get_dimension(0);
    const std::int64_t column_count = eigvecs.get_dimension(1);
    Float* eigvecs_ptr = eigvecs.get_mutable_data();

    auto flip_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const auto range = sycl::range<1>(row_count);

        h.parallel_for(range, [=](sycl::id<1> id) {
            const std::int64_t row = id[0];
            Float* row_ptr = eigvecs_ptr + row * column_count;

            Float max_val = row_ptr[0];
            Float max_abs = sycl::fabs(row_ptr[0]);

            for (std::int64_t j = 1; j < column_count; j++) {
                Float abs_val = sycl::fabs(row_ptr[j]);
                if (abs_val > max_abs) {
                    max_abs = abs_val;
                    max_val = row_ptr[j];
                }
            }

            if (max_val < Float(0)) {
                for (std::int64_t j = 0; j < column_count; j++) {
                    row_ptr[j] = -row_ptr[j];
                }
            }
        });
    });

    flip_event.wait_and_throw();
}

///  A wrapper that computes 1d array of means of the columns from precomputed sums
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  sums  The input sums of size `column_count`
/// @param[in]  row_count  The number of `row_count` of the input data
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 1d array of means
/// of size `column_count` and the second element is a SYCL event indicating the availability
/// of the means array for reading and writing
template <typename Float>
auto compute_means(sycl::queue& queue,
                   const pr::ndview<Float, 1>& sums,
                   std::int64_t row_count,
                   const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_means, queue);
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(sums.get_dimension(0) > 0);

    const std::int64_t column_count = sums.get_dimension(0);
    auto means = pr::ndarray<Float, 1>::empty(queue, { column_count }, alloc::device);
    auto means_event = pr::means(queue, row_count, sums, means, deps);
    return std::make_tuple(means, means_event);
}

///  A wrapper that computes 1d array of explained variances ratio from the eigenvalues
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  eigenvalues  The input eigenvalues of size `component_count`
/// @param[in]  vars  The input variances of size `component_count`
/// @param[in]  deps  Events indicating availability of the `eigenvalues` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 1d array of explained variances ratio
/// of size `column_count` and the second element is a SYCL event indicating the availability
/// of the explained variances ratio array for reading and writing
template <typename Float>
double compute_noise_variance_on_host(sycl::queue& queue,
                                      pr::ndarray<Float, 1> eigenvalues,
                                      std::int64_t range,
                                      const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_noise_variance_on_host);
    ONEDAL_ASSERT(eigenvalues.has_mutable_data());

    auto eigvals_ptr = eigenvalues.get_data();

    Float sum = 0;
    for (std::int64_t i = 0; i < range; ++i) {
        sum += eigvals_ptr[i];
    }

    return sum / range;
}

///  A wrapper that computes 1d array of explained variances ratio from the eigenvalues
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  eigenvalues  The input eigenvalues of size `component_count`
/// @param[in]  vars  The input variances of size `component_count`
/// @param[in]  deps  Events indicating availability of the `eigenvalues` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 1d array of explained variances ratio
/// of size `column_count` and the second element is a SYCL event indicating the availability
/// of the explained variances ratio array for reading and writing
template <typename Float>
auto compute_explained_variances_on_host(sycl::queue& queue,
                                         pr::ndarray<Float, 1> eigenvalues,
                                         pr::ndarray<Float, 1> vars,
                                         const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_explained_variances_on_host);
    ONEDAL_ASSERT(eigenvalues.has_mutable_data());

    const std::int64_t component_count = eigenvalues.get_dimension(0);
    const std::int64_t column_count = vars.get_dimension(0);
    auto explained_variances_ratio = pr::ndarray<Float, 1>::empty(component_count);

    auto eigvals_ptr = eigenvalues.get_data();
    auto vars_ptr = vars.get_data();
    auto explained_variances_ratio_ptr = explained_variances_ratio.get_mutable_data();

    Float sum = 0;
    for (std::int64_t i = 0; i < column_count; ++i) {
        sum += vars_ptr[i];
    }
    ONEDAL_ASSERT(sum > 0);
    const Float inverse_sum = 1.0 / sum;
    for (std::int64_t i = 0; i < component_count; ++i) {
        explained_variances_ratio_ptr[i] = eigvals_ptr[i] * inverse_sum;
    }
    return explained_variances_ratio;
}

template <typename Float>
auto compute_explained_variances_on_gpu(sycl::queue& queue,
                                        pr::ndarray<Float, 1> eigenvalues,
                                        pr::ndarray<Float, 1> vars,
                                        const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_explained_variances_on_gpu);
    ONEDAL_ASSERT(eigenvalues.has_mutable_data());

    const std::int64_t component_count = eigenvalues.get_dimension(0);
    const std::int64_t column_count = vars.get_dimension(0);
    auto explained_variances_ratio =
        pr::ndarray<Float, 1>::empty(queue, { component_count }, alloc::device);

    auto eigvals_ptr = eigenvalues.get_data();
    auto vars_host = vars.to_host(queue);
    auto vars_ptr = vars_host.get_data();
    auto explained_variances_ratio_ptr = explained_variances_ratio.get_mutable_data();

    Float sum = 0;
    for (std::int64_t i = 0; i < column_count; ++i) {
        sum += vars_ptr[i];
    }

    ONEDAL_ASSERT(sum > 0);
    const Float inverse_sum = 1.0 / sum;

    auto compute_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(component_count), [=](sycl::id<1> i) {
            explained_variances_ratio_ptr[i] = eigvals_ptr[i] * inverse_sum;
        });
    });

    compute_event.wait_and_throw();
    return explained_variances_ratio;
}

// Cov and Precomputed methods

///  A wrapper that computes 2d array of covariance matrix from 2d xtx array
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  row_count  The number of `row_count` of the input data
/// @param[in]  xtx  The input xtx matrix of size `column_count` x `column_count`
/// @param[in]  sums  The input sums of size `column_count`
/// @param[in]  bias  The input bias value
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 2d array of covariance matrix
/// of size `column_count` x `column_count` and the second element is a SYCL event indicating the availability
/// of the covariance matrix array for reading and writing
template <typename Float>
auto compute_covariance(sycl::queue& queue,
                        std::int64_t row_count,
                        const pr::ndview<Float, 2>& xtx,
                        const pr::ndarray<Float, 1>& sums,
                        bool bias,
                        const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_covariance, queue);
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(xtx.has_data());
    ONEDAL_ASSERT(xtx.get_dimension(1) > 0);

    const std::int64_t column_count = xtx.get_dimension(1);

    auto cov = pr::ndarray<Float, 2>::empty(queue, { column_count, column_count }, alloc::device);

    auto copy_event = copy(queue, cov, xtx, { deps });
    bool assume_centering = false;
    auto cov_event =
        pr::covariance(queue, row_count, sums, cov, bias, assume_centering, { copy_event });
    return std::make_tuple(cov, cov_event);
}

///  A wrapper that computes 2d array of correlation matrix from 2d xtx array
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  row_count  The number of `row_count` of the input data
/// @param[in]  xtx  The input xtx matrix of size  `column_count` x `column_count`
/// @param[in]  sums  The input sums of size `column_count`
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 2d array of correlation matrix
/// of size `column_count` x `column_count` and the second element is a SYCL event indicating the availability
/// of the correlation matrix array for reading and writing
template <typename Float>
auto compute_correlation(sycl::queue& queue,
                         std::int64_t row_count,
                         const pr::ndview<Float, 2>& xtx,
                         const pr::ndarray<Float, 1>& sums,
                         const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_correlation, queue);
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(xtx.has_data());
    ONEDAL_ASSERT(xtx.get_dimension(1) > 0);

    const std::int64_t column_count = xtx.get_dimension(1);

    auto corr = pr::ndarray<Float, 2>::empty(queue, { column_count, column_count }, alloc::device);

    auto copy_event = copy(queue, corr, xtx, { deps });

    auto corr_event = pr::correlation(queue, row_count, sums, corr, { copy_event });

    return std::make_tuple(corr, corr_event);
}

///  A wrapper that computes 2d array of crossproduct matrix for the online algorthm
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  data  The input block of the data of size `row_count` x `column_count`
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 2d array of crossproduct matrix
/// of size `column_count` x `column_count` and the second element is a SYCL event indicating the availability
/// of the crossproduct matrix array for reading and writing
template <typename Float>
auto compute_crossproduct(sycl::queue& queue,
                          const pr::ndview<Float, 2>& data,
                          const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_crossproduct, queue);
    ONEDAL_ASSERT(data.has_data());

    const std::int64_t column_count = data.get_dimension(1);
    auto xtx = pr::ndarray<Float, 2>::empty(queue, { column_count, column_count }, alloc::device);

    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, queue);
        gemm_event = gemm(queue, data.t(), data, xtx, Float(1.0), Float(0.0));
    }

    return std::make_tuple(xtx, gemm_event);
}

///  A wrapper that computes 1d array of variances of the columns from the covariance matrix
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  cov  The input covariance matrix of size `column_count` x `column_count`
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 1d array of variances
/// of size `column_count` and the second element is a SYCL event indicating the availability
/// of the variances array for reading and writing
template <typename Float>
auto compute_variances(sycl::queue& queue,
                       const pr::ndview<Float, 2>& cov,
                       const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_vars, queue);
    ONEDAL_ASSERT(cov.has_data());
    ONEDAL_ASSERT(cov.get_dimension(0) > 0);
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");

    auto column_count = cov.get_dimension(0);
    auto vars = pr::ndarray<Float, 1>::empty(queue, { column_count }, alloc::device);
    auto vars_event = pr::variances(queue, cov, vars, deps);
    return std::make_tuple(vars, vars_event);
}

///  A wrapper that computes 2d array of correlation matrix from the covariance matrix
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  row_count  The number of `row_count` of the input data
/// @param[in]  cov  The input covariance matrix of size `column_count` x `column_count`
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 2d array of correlation matrix
/// of size `column_count` x `column_count` and the second element is a SYCL event indicating the availability
/// of the correlation matrix array for reading and writing
template <typename Float>
auto compute_correlation_from_covariance(sycl::queue& queue,
                                         std::int64_t row_count,
                                         const pr::ndview<Float, 2>& cov,
                                         const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_correlation, queue);
    ONEDAL_ASSERT(cov.has_data());
    ONEDAL_ASSERT(cov.get_dimension(0) > 0);
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");

    const std::int64_t column_count = cov.get_dimension(1);

    auto tmp = pr::ndarray<Float, 1>::empty(queue, { column_count }, alloc::device);

    auto corr = pr::ndarray<Float, 2>::empty(queue, { column_count, column_count }, alloc::device);

    const bool bias = false; // Currently we use only unbiased covariance for PCA computation.
    auto corr_event = pr::correlation_from_covariance(queue, row_count, cov, corr, tmp, bias, deps);

    return std::make_tuple(corr, corr_event);
}

// SVD method

///  A wrapper that computes 1d array of eigenvalues from the 1d array of the singular values
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  singular_values  The input singular values matrix of size `column_count` x `column_count`
/// @param[in]  component_count  The number of `component_count` of the descriptor
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return The resulting 2d array of eigenvalues
template <typename Float>
auto compute_eigenvalues_on_host(sycl::queue& queue,
                                 pr::ndarray<Float, 1> singular_values,
                                 std::int64_t row_count,
                                 const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_eigenvalues_on_host);

    const std::int64_t component_count = singular_values.get_dimension(0);

    auto eigenvalues = pr::ndarray<Float, 1>::empty(component_count);

    auto singular_values_ptr = singular_values.get_data();
    auto eigvals_ptr = eigenvalues.get_mutable_data();

    const Float factor = row_count - 1;
    for (std::int64_t i = 0; i < component_count; ++i) {
        eigvals_ptr[i] = singular_values_ptr[i] * singular_values_ptr[i] / factor;
    }
    return eigenvalues;
}

///  A wrapper that computes 1d array of singular values from the eigenvalues
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  eigenvalues  The input eigenvalues array of size `component_count`
/// @param[in]  row_count  The number of `row_count` of the input data
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return The resulting 2d array of singular values
template <typename Float>
auto compute_singular_values_on_host(sycl::queue& queue,
                                     pr::ndarray<Float, 1> eigenvalues,
                                     std::int64_t row_count,
                                     const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_singular_values_on_host);
    ONEDAL_ASSERT(eigenvalues.has_mutable_data());

    const std::int64_t component_count = eigenvalues.get_dimension(0);

    auto singular_values = pr::ndarray<Float, 1>::empty(component_count);

    auto eigvals_ptr = eigenvalues.get_data();
    auto singular_values_ptr = singular_values.get_mutable_data();

    const Float factor = row_count - 1;
    for (std::int64_t i = 0; i < component_count; ++i) {
        singular_values_ptr[i] = std::sqrt(factor * eigvals_ptr[i]);
    }
    return singular_values;
}

template <typename Float>
auto compute_singular_values_on_gpu(sycl::queue& queue,
                                    pr::ndarray<Float, 1> eigenvalues,
                                    std::int64_t row_count,
                                    const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_singular_values_on_gpu);
    ONEDAL_ASSERT(eigenvalues.has_mutable_data());

    const std::int64_t component_count = eigenvalues.get_dimension(0);
    auto singular_values = pr::ndarray<Float, 1>::empty(queue, { component_count }, alloc::device);

    auto eigvals_ptr = eigenvalues.get_data();
    auto singular_values_ptr = singular_values.get_mutable_data();

    const Float factor = row_count - 1;

    auto compute_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(component_count), [=](sycl::id<1> i) {
            singular_values_ptr[i] = sycl::sqrt(factor * eigvals_ptr[i]);
        });
    });

    compute_event.wait_and_throw();
    return singular_values;
}
///  A wrapper that sliced 2d array of eigenvectors with necessary dimensions
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  data  The input eigenvectors matrix with size `column_count` x `column_count`
/// @param[in]  component_count  The number of `component_count` of the descriptor
/// @param[in]  column_count  The number of `column_count` of the input data
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 2d array of eigenvectors
/// of size `component_count` x `column_count` and the second element is a SYCL event indicating the availability
/// of the eigenvectors array for reading and writing
template <typename Float>
auto slice_data(sycl::queue& queue,
                const pr::ndview<Float, 2>& data,
                std::int64_t component_count,
                std::int64_t column_count,
                const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_means, queue);
    const std::int64_t column_count_local = data.get_dimension(1);
    auto sliced_data =
        pr::ndarray<Float, 2>::empty(queue, { component_count, column_count }, alloc::device);
    auto sliced_data_ptr = sliced_data.get_mutable_data();
    auto data_ptr = data.get_data();
    auto slice_event = queue.submit([&](sycl::handler& h) {
        const auto range = bk::make_range_2d(component_count, column_count);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::int64_t row = id[0];
            const std::int64_t column = id[1];
            sliced_data_ptr[row * column_count + column] =
                data_ptr[row * column_count_local + column];
        });
    });
    return std::make_tuple(sliced_data, slice_event);
}

///  A wrapper that computes the mean centered data from the input data
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  data  The input block of the data of size `row_count` x `column_count`
/// @param[in]  means  The input means of size `column_count`
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the mean centered data array for reading and writing
template <typename Float>
auto get_centered(sycl::queue& queue,
                  pr::ndview<Float, 2>& data,
                  const pr::ndview<Float, 1>& means,
                  const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_centered_data, queue);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto centered_data_ptr = data.get_mutable_data();
    auto means_ptr = means.get_data();

    auto centered_event = queue.submit([&](sycl::handler& h) {
        const auto range = bk::make_range_2d(row_count, column_count);
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::size_t i = id[0];
            const std::size_t j = id[1];
            centered_data_ptr[i * column_count + j] =
                centered_data_ptr[i * column_count + j] - means_ptr[j];
        });
    });
    return centered_event;
}

///  A wrapper that computes the scaled data from the mean centered data
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  data  The mean centered data of size `row_count` x `column_count`
/// @param[in]  variances  The input variances of size `column_count`
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the scaled data array for reading and writing
template <typename Float>
auto get_scaled(sycl::queue& queue,
                pr::ndview<Float, 2>& data,
                const pr::ndview<Float, 1>& variances,
                const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_scaled_data, queue);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto scaled_data_ptr = data.get_mutable_data();
    auto variances_ptr = variances.get_data();

    auto scaled_event = queue.submit([&](sycl::handler& h) {
        const auto range = bk::make_range_2d(row_count, column_count);
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::size_t i = id[0];
            const std::size_t j = id[1];
            const Float sqrt_var = sycl::sqrt(variances_ptr[j]);
            const Float inv_var =
                sqrt_var < std::numeric_limits<Float>::epsilon() ? 0 : 1 / sqrt_var;
            scaled_data_ptr[i * column_count + j] = scaled_data_ptr[i * column_count + j] * inv_var;
        });
    });
    return scaled_event;
}

///  A wrapper that computes 1d array of variances of the columns from 2d mean centered data array
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  data  The mean centered data of size `row_count` x `column_count`
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 1d array of variances
/// of size `column_count` and the second element is a SYCL event indicating the availability
/// of the variances array for reading and writing
template <typename Float>
auto compute_variances_device(sycl::queue& queue,
                              const pr::ndview<Float, 2>& data,
                              const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_means, queue);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto vars = pr::ndarray<Float, 1>::empty(queue, { column_count }, alloc::device);

    constexpr pr::sum<Float> binary;
    constexpr pr::square<Float> unary;

    auto vars_event_ = pr::reduce_by_columns(queue, data, vars, binary, unary, deps);
    auto vars_ptr = vars.get_mutable_data();
    auto vars_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(vars_event_);
        const auto range = bk::make_range_1d(column_count);
        h.parallel_for(range, [=](sycl::id<1> id) {
            const std::int64_t i = id[0];

            vars_ptr[i] = vars_ptr[i] / (row_count - 1);
        });
    });
    return std::make_tuple(vars, vars_event);
}

// Online Cov method

///  A wrapper that initiates the first iteration partial rows of online algorithm
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  row_count  The number of `row_count` of the input data
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting number of rows
/// and the second element is a SYCL event indicating the availability
/// of the resulting number of rows for reading and writing
template <typename Float>
auto init(sycl::queue& queue,
          const std::int64_t row_count,
          const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(init_partial_results, queue);

    auto result_nobs = pr::ndarray<Float, 1>::empty(queue, 1);

    auto result_nobs_ptr = result_nobs.get_mutable_data();

    auto init_event = queue.submit([&](sycl::handler& cgh) {
        const auto range = sycl::range<1>(1);
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<1> id) {
            result_nobs_ptr[0] = row_count;
        });
    });

    return std::make_tuple(result_nobs, init_event);
}

///  A wrapper that updates partial results of online algorithm
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  crossproducts  The new crossproducts of size `column_count` x `column_count`
/// @param[in]  sums  The new sums of size `column_count`
/// @param[in]  current_crossproducts  The current crossproducts of size `column_count` x `column_count`
/// @param[in]  current_sums  The current sums of size `column_count`
/// @param[in]  current_nobs  The current nobs of size `1`
/// @param[in]  row_count  The number of `row_count` of the input data
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of four elements, where the first element is the resulting sums,
/// the second is the resulting crossproducts, the third is the resulting number of partial rows
/// and the fourth element is a SYCL event indicating the availability
/// of the all arrays for reading and writing
template <typename Float>
auto update_partial_results(sycl::queue& queue,
                            const pr::ndview<Float, 2>& crossproducts,
                            const pr::ndview<Float, 1>& sums,
                            const pr::ndview<Float, 2>& current_crossproducts,
                            const pr::ndview<Float, 1>& current_sums,
                            const pr::ndview<Float, 1>& current_nobs,
                            const std::int64_t row_count,
                            const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(update_partial_results, queue);

    auto column_count = crossproducts.get_dimension(1);
    auto result_sums = pr::ndarray<Float, 1>::empty(queue, column_count, alloc::device);
    auto result_crossproducts =
        pr::ndarray<Float, 2>::empty(queue, { column_count, column_count }, alloc::device);
    auto result_nobs = pr::ndarray<Float, 1>::empty(queue, 1);

    auto result_sums_ptr = result_sums.get_mutable_data();
    auto result_crossproducts_ptr = result_crossproducts.get_mutable_data();
    auto result_nobs_ptr = result_nobs.get_mutable_data();

    auto current_crossproducts_data = current_crossproducts.get_data();
    auto current_sums_data = current_sums.get_data();
    auto current_nobs_data = current_nobs.get_data();

    auto crossproducts_data = crossproducts.get_data();
    auto sums_data = sums.get_data();

    auto update_event = queue.submit([&](sycl::handler& cgh) {
        const auto range = sycl::range<2>(column_count, column_count);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<2> id) {
            const std::int64_t row = id[0], col = id[1];

            result_crossproducts_ptr[row * column_count + col] =
                crossproducts_data[row * column_count + col] +
                current_crossproducts_data[row * column_count + col];

            if (static_cast<std::int64_t>(col) < column_count) {
                result_sums_ptr[col] = sums_data[col] + current_sums_data[col];
            }

            if (row == 0 && col == 0) {
                result_nobs_ptr[0] = row_count + current_nobs_data[0];
            }
        });
    });

    return std::make_tuple(result_sums, result_crossproducts, result_nobs, update_event);
}

} // namespace oneapi::dal::pca::backend

#endif // ONEDAL_DATA_PARALLEL
