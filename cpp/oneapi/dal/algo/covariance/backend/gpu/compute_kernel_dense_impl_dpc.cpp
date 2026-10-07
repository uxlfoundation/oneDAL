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

#include "oneapi/dal/algo/covariance/backend/gpu/compute_kernel_dense_impl.hpp"
#include "oneapi/dal/algo/covariance/backend/gpu/misc.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::covariance::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;
using parameters_t = detail::compute_parameters<task_t>;

template <typename Float>
result_t compute_kernel_dense_impl<Float>::operator()(const descriptor_t& desc,
                                                      const parameters_t& params,
                                                      const input_t& input) {
    ONEDAL_ASSERT(input.get_data().has_data());
    ONEDAL_PROFILER_TASK_WITH_ARGS_QUEUE(covariance_algo_compute,
                                         q_,
                                         input.get_data().get_row_count(),
                                         input.get_data().get_column_count(),
                                         desc.get_bias(),
                                         desc.get_assume_centered());

    const auto data = input.get_data();

    const std::int64_t row_count = data.get_row_count();
    ONEDAL_ASSERT(row_count > 0);
    auto rows_count_global = row_count;
    const std::int64_t column_count = data.get_column_count();
    ONEDAL_ASSERT(column_count > 0);

    auto bias = desc.get_bias();
    auto assume_centered = desc.get_assume_centered();

    auto result = compute_result<task_t>{}.set_result_options(desc.get_result_options());

    const auto data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);

    auto [sums, sums_event] = pr::compute_sums(q_, data_nd, assume_centered, {});

    {
        ONEDAL_PROFILER_TASK(allreduce_sums, q_);
        comm_.allreduce(sums.flatten(q_, { sums_event }), spmd::reduce_op::sum).wait();
    }

    {
        ONEDAL_PROFILER_TASK_WITH_ARGS(allreduce_rows_count_global, rows_count_global);
        comm_.allreduce(rows_count_global, spmd::reduce_op::sum).wait();
    }

    auto [means, means_event] =
        (assume_centered
             ? pr::ndarray<Float, 1>::zeros(q_, { column_count }, sycl::usm::alloc::device)
             : pr::compute_means(q_, sums, rows_count_global, { sums_event }));

    // The local rows are centered with the global means, so that the local
    // cross-products stay additive across the ranks.
    auto x_centered = data_nd;
    sycl::event center_event = means_event;
    if (!assume_centered) {
        x_centered = pr::ndarray<Float, 2>::empty(q_, { row_count, column_count }, alloc::device);
        auto copy_event = pr::copy(q_, x_centered, data_nd);
        center_event = pr::get_centered(q_, x_centered, means, { copy_event, means_event });
    }

    auto cross_product =
        pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, alloc::device);

    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, q_);
        gemm_event = gemm(q_,
                          x_centered.t(),
                          x_centered,
                          cross_product,
                          Float(1.0),
                          Float(0.0),
                          { center_event });
    }

    {
        ONEDAL_PROFILER_TASK(cross_product, q_);
        comm_.allreduce(cross_product.flatten(q_, { gemm_event }), spmd::reduce_op::sum).wait();
    }

    const bool has_cov = desc.get_result_options().test(result_options::cov_matrix);
    const bool has_cor = desc.get_result_options().test(result_options::cor_matrix);

    if (has_cov || has_cor) {
        // Compute covariance matrix from the cross-product matrix.
        auto cov = pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, alloc::device);
        auto copy_event = pr::copy(q_, cov, cross_product);
        auto cov_event =
            pr::compute_covariance_centered(q_, rows_count_global, cov, bias, { copy_event });

        if (has_cov) {
            result.set_cov_matrix(
                (homogen_table::wrap(cov.flatten(q_, { cov_event }), column_count, column_count)));
        }
        if (has_cor) {
            auto corr =
                pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, alloc::device);
            auto corr_event = pr::correlation_from_covariance(q_, cov, corr, bias, { cov_event });
            result.set_cor_matrix((
                homogen_table::wrap(corr.flatten(q_, { corr_event }), column_count, column_count)));
        }
    }
    if (desc.get_result_options().test(result_options::means)) {
        result.set_means(homogen_table::wrap(means.flatten(q_, { means_event }), 1, column_count));
    }
    return result;
}

template class compute_kernel_dense_impl<float>;
template class compute_kernel_dense_impl<double>;

} // namespace oneapi::dal::covariance::backend

#endif // ONEDAL_DATA_PARALLEL
