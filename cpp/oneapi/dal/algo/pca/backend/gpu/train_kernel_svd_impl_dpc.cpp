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

#include "oneapi/dal/algo/pca/backend/gpu/train_kernel_svd_impl.hpp"
#include "oneapi/dal/algo/pca/backend/gpu/misc.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"

namespace oneapi::dal::pca::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;
namespace mkl = oneapi::mkl;
using alloc = sycl::usm::alloc;

using bk::context_gpu;

using task_t = task::dim_reduction;
using input_t = train_input<task_t>;
using result_t = train_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

///  A wrapper that computes svd decomposition of the input data
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  data  The input data of size `row_count` x `column_count`
/// @param[in]  component_count  The number of `component_count` of the algorithm descriptor
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of four elements, where the first element is the U matrix,
/// the second is the resulting S array, the third is the resulting V_T matrix
/// and the fourth element is a SYCL event indicating the availability
/// of the all arrays are ready for reading and writing
template <typename Float, pr::ndorder order>
auto svd_decomposition(sycl::queue& queue,
                       pr::ndview<Float, 2, order>& data,
                       std::int64_t component_count,
                       const bk::event_vector& deps = {}) {
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto U = pr::ndarray<Float, 2>::empty(queue, { column_count, column_count }, alloc::device);

    auto S = pr::ndarray<Float, 1>::empty(queue, { component_count }, alloc::device);

    auto V_T = pr::ndarray<Float, 2>::empty(queue, { 1, 1 }, alloc::device);

    std::int64_t lda = column_count;
    std::int64_t ldu = column_count;
    std::int64_t ldvt = 1;
    sycl::event gesvd_event;
    {
        ONEDAL_PROFILER_TASK(gesvd, queue);
        // Due to MKL uses Fortran API column count and row count are provided in inverted order
        gesvd_event = pr::gesvd<mkl::jobsvd::somevec, mkl::jobsvd::novec>(queue,
                                                                          column_count,
                                                                          row_count,
                                                                          data,
                                                                          lda,
                                                                          S,
                                                                          U,
                                                                          ldu,
                                                                          V_T,
                                                                          ldvt,
                                                                          { deps });
    }
    return std::make_tuple(U, S, V_T, gesvd_event);
}

template <typename Float>
result_t train_kernel_svd_impl<Float>::operator()(const descriptor_t& desc, const input_t& input) {
    ONEDAL_ASSERT(input.get_data().has_data());
    const auto data = input.get_data();
    const std::int64_t row_count = data.get_row_count();
    ONEDAL_ASSERT(data.get_column_count() > 0);

    const std::int64_t column_count = data.get_column_count();
    ONEDAL_ASSERT(column_count > 0);
    const std::int64_t component_count = get_component_count(desc, data);
    ONEDAL_ASSERT(component_count > 0);
    dal::detail::check_mul_overflow(column_count, component_count);

    bool zscore = false;
    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());
    auto data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);

    auto [sums, sums_event] = compute_sums(q_, data_nd);
    auto [means, means_event] = compute_means(q_, sums, row_count, { sums_event });

    sycl::event mean_centered_event;
    if (desc.get_normalization_mode() != normalization::none) {
        mean_centered_event = get_centered(q_, data_nd, means, { means_event });
    }

    auto [vars, vars_event] = compute_variances_device(q_, data_nd, { mean_centered_event });

    sycl::event scaled_event;
    if (desc.get_normalization_mode() == normalization::zscore) {
        zscore = true;
        scaled_event = get_scaled(q_, data_nd, vars, { vars_event, mean_centered_event });
    }

    auto [U, S, V_T, gesvd_event] =
        svd_decomposition(q_, data_nd, component_count, { scaled_event, mean_centered_event });

    auto S_host = S.to_host(q_);
    auto eigenvalues = compute_eigenvalues_on_host(q_, S_host, row_count, { gesvd_event });
    if (desc.get_result_options().test(result_options::singular_values)) {
        if (zscore) {
            result.set_singular_values(
                homogen_table::wrap(eigenvalues.flatten(), 1, component_count));
        }
        else {
            result.set_singular_values(homogen_table::wrap(S_host.flatten(), 1, component_count));
        }
    }

    result.set_means(homogen_table::wrap(means.flatten(q_, { gesvd_event }), 1, column_count));

    if (desc.get_result_options().test(result_options::eigenvalues)) {
        result.set_eigenvalues(homogen_table::wrap(eigenvalues.flatten(), 1, component_count));
    }

    if (desc.get_result_options().test(result_options::vars)) {
        result.set_variances(
            homogen_table::wrap(vars.flatten(q_, { gesvd_event }), 1, column_count));
    }

    if (desc.get_result_options().test(result_options::explained_variances_ratio)) {
        auto vars_host = vars.to_host(q_);
        auto explained_variances_ratio =
            compute_explained_variances_on_host(q_, eigenvalues, vars_host, { gesvd_event });
        result.set_explained_variances_ratio(
            homogen_table::wrap(explained_variances_ratio.flatten(), 1, component_count));
    }

    auto eigenvectors_final =
        prepare_eigenvectors_svd(q_, U, component_count, desc.get_deterministic(), { gesvd_event });

    if (desc.get_result_options().test(result_options::eigenvectors)) {
        result.set_eigenvectors(homogen_table::wrap(eigenvectors_final.flatten(q_, {}),
                                                    eigenvectors_final.get_dimension(0),
                                                    eigenvectors_final.get_dimension(1)));
    }

    return result;
}

template class train_kernel_svd_impl<float>;
template class train_kernel_svd_impl<double>;

} // namespace oneapi::dal::pca::backend
