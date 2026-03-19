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

#include "oneapi/dal/algo/pca/backend/gpu/finalize_train_kernel.hpp"
#include "oneapi/dal/algo/pca/backend/gpu/finalize_train_kernel_cov_impl.hpp"
#include "oneapi/dal/algo/pca/backend/gpu/misc.hpp"
#include "oneapi/dal/algo/pca/backend/common.hpp"

#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::pca::backend {

namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;
using alloc = sycl::usm::alloc;

using bk::context_gpu;

using task_t = task::dim_reduction;
using input_t = partial_train_result<task_t>;
using result_t = train_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
result_t finalize_train_kernel_cov_impl<Float>::operator()(const descriptor_t& desc,
                                                           const input_t& input) {
    constexpr bool bias = false; // Currently we use only unbiased covariance for PCA computation.

    const std::int64_t column_count = input.get_partial_crossproduct().get_column_count();
    ONEDAL_ASSERT(column_count > 0);
    const std::int64_t component_count =
        get_component_count(desc, input.get_partial_crossproduct());
    ONEDAL_ASSERT(component_count > 0);

    dal::detail::check_mul_overflow(column_count, column_count);

    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());

    const auto nobs_host = pr::table2ndarray<Float>(input.get_partial_n_rows());
    auto rows_count_global = nobs_host.get_data()[0];
    auto sums = pr::table2ndarray_1d<Float>(q, input.get_partial_sum(), sycl::usm::alloc::device);
    auto xtx =
        pr::table2ndarray<Float>(q, input.get_partial_crossproduct(), sycl::usm::alloc::device);
    if (comm_.get_rank_count() > 1) {
        {
            ONEDAL_PROFILER_TASK(allreduce_rows_count_global);
            comm_.allreduce(rows_count_global, spmd::reduce_op::sum).wait();
        }
        auto sums_copy =
            pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
        auto copy_event = copy(q, sums_copy, sums, {});
        copy_event.wait_and_throw();
        sums = sums_copy;

        auto xtx_copy = pr::ndarray<Float, 2>::empty(q,
                                                     { column_count, column_count },
                                                     sycl::usm::alloc::device);
        copy_event = copy(q, xtx_copy, xtx, {});
        copy_event.wait_and_throw();
        xtx = xtx_copy;

        {
            ONEDAL_PROFILER_TASK(allreduce_sums, q);
            comm_.allreduce(sums.flatten(q, {}), spmd::reduce_op::sum).wait();
        }

        {
            ONEDAL_PROFILER_TASK(allreduce_xtx, q);
            comm_.allreduce(xtx.flatten(q, {}), spmd::reduce_op::sum).wait();
        }
    }

    if (desc.get_result_options().test(result_options::means)) {
        auto [means, means_event] = compute_means(q, sums, rows_count_global, {});
        result.set_means(homogen_table::wrap(means.flatten(q, { means_event }), 1, column_count));
    }

    auto [cov, cov_event] = compute_covariance(q, rows_count_global, xtx, sums, {});

    auto [vars, vars_event] = compute_variances(q, cov, { cov_event });

    if (desc.get_result_options().test(result_options::vars)) {
        result.set_variances(homogen_table::wrap(vars.flatten(q, { vars_event }), 1, column_count));
    }

    auto data_to_compute = cov;

    sycl::event corr_event;
    if (desc.get_normalization_mode() == normalization::zscore) {
        auto corr = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);
        corr_event =
            pr::correlation_from_covariance(q, rows_count_global, cov, corr, bias, { cov_event });
        data_to_compute = corr;
    }

    auto [eigvals, syevd_event] =
        syevd_computation(q, data_to_compute, { cov_event, corr_event, vars_event });

    // Compute noise variance on the host (CPU) since it's a simple operation and
    // doesn't benefit much from parallelization on the device (GPU/accelerator).
    // This avoids unnecessary overhead from device-side execution.
    if (desc.get_result_options().test(result_options::noise_variance)) {
        std::int64_t rows_count_ = rows_count_global;
        auto range = std::min(rows_count_, column_count) - component_count;
        auto noise_variance = compute_noise_variance(q, eigvals, range, { syevd_event });
        result.set_noise_variance(noise_variance);
    }

    auto [reordered_eigvals, reordered_eigenvectors] =
        reorder_eigenpairs_descending(q,
                                      eigvals,
                                      data_to_compute,
                                      component_count,
                                      { syevd_event });

    if (desc.get_result_options().test(result_options::eigenvalues)) {
        result.set_eigenvalues(
            homogen_table::wrap(reordered_eigvals.flatten(q, {}), 1, component_count));
    }

    if (desc.get_result_options().test(result_options::singular_values)) {
        auto singular_values =
            compute_singular_values(q, reordered_eigvals, rows_count_global, { syevd_event });
        result.set_singular_values(
            homogen_table::wrap(singular_values.flatten(q), 1, component_count));
    }

    if (desc.get_result_options().test(result_options::explained_variances_ratio)) {
        auto explained_variances_ratio =
            compute_explained_variances(q, reordered_eigvals, vars, { syevd_event, vars_event });
        result.set_explained_variances_ratio(
            homogen_table::wrap(explained_variances_ratio.flatten(q), 1, component_count));
    }

    if (desc.get_deterministic()) {
        sign_flip(q, reordered_eigenvectors, {}).wait_and_throw();
    }

    if (desc.get_result_options().test(result_options::eigenvectors)) {
        result.set_eigenvectors(homogen_table::wrap(reordered_eigenvectors.flatten(q),
                                                    reordered_eigenvectors.get_dimension(0),
                                                    reordered_eigenvectors.get_dimension(1)));
    }
    return result;
}

template class finalize_train_kernel_cov_impl<float>;
template class finalize_train_kernel_cov_impl<double>;

} // namespace oneapi::dal::pca::backend

#endif // ONEDAL_DATA_PARALLEL
