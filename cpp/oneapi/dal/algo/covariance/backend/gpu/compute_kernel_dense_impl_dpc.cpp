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

    auto [sums, sums_event] = compute_sums(q_, data_nd, assume_centered, {});

    {
        ONEDAL_PROFILER_TASK(allreduce_sums, q_);
        comm_.allreduce(sums.flatten(q_, { sums_event }), spmd::reduce_op::sum).wait();
    }

    auto xtx = pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, alloc::device);

    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, q_);
        gemm_event = gemm(q_, data_nd.t(), data_nd, xtx, Float(1.0), Float(0.0));
    }

    {
        ONEDAL_PROFILER_TASK(allreduce_xtx, q_);
        comm_.allreduce(xtx.flatten(q_, { gemm_event }), spmd::reduce_op::sum).wait();
    }

    {
        ONEDAL_PROFILER_TASK(allreduce_rows_count_global);
        comm_.allreduce(rows_count_global, spmd::reduce_op::sum).wait();
    }

    if (desc.get_result_options().test(result_options::cov_matrix)) {
        auto [cov, cov_event] = compute_covariance(q_,
                                                   rows_count_global,
                                                   xtx,
                                                   sums,
                                                   bias,
                                                   assume_centered,
                                                   { gemm_event });
        result.set_cov_matrix(
            (homogen_table::wrap(cov.flatten(q_, { cov_event }), column_count, column_count)));
    }
    if (desc.get_result_options().test(result_options::cor_matrix)) {
        auto [corr, corr_event] =
            compute_correlation(q_, rows_count_global, xtx, sums, { gemm_event });
        result.set_cor_matrix(
            (homogen_table::wrap(corr.flatten(q_, { corr_event }), column_count, column_count)));
    }
    if (desc.get_result_options().test(result_options::means)) {
        if (!assume_centered) {
            auto [means, means_event] = compute_means(q_, sums, rows_count_global, { gemm_event });
            result.set_means(
                homogen_table::wrap(means.flatten(q_, { means_event }), 1, column_count));
        }
        else {
            auto [zero_means, zeros_event] =
                pr::ndarray<Float, 1>::zeros(q_, { column_count }, sycl::usm::alloc::device);
            result.set_means(
                homogen_table::wrap(zero_means.flatten(q_, { zeros_event }), 1, column_count));
        }
    }
    return result;
}

template class compute_kernel_dense_impl<float>;
template class compute_kernel_dense_impl<double>;

} // namespace oneapi::dal::covariance::backend

#endif // ONEDAL_DATA_PARALLEL
