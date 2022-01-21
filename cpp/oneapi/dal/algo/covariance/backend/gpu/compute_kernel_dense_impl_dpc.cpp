/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"
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

template <typename Float>
auto compute_sums(sycl::queue& q,
                  const pr::ndview<Float, 2>& data,
                  const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_sums, q);
    ONEDAL_ASSERT(data.has_data());

    const std::int64_t column_count = data.get_dimension(1);
    auto sums = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);
    auto reduce_event =
        pr::reduce_by_columns(q, data, sums, pr::sum<Float>{}, pr::identity<Float>{}, deps);
    return std::make_tuple(sums, reduce_event);
}

template <typename Float>
auto compute_means(sycl::queue& q,
                   const pr::ndview<Float, 1>& sums,
                   std::int64_t row_count,
                   const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_means, q);
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(sums.get_dimension(0) > 0);

    const std::int64_t column_count = sums.get_dimension(0);
    auto means = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);
    auto means_event = pr::means(q, row_count, sums, means, deps);
    return std::make_tuple(means, means_event);
}

template <typename Float>
auto compute_covariance(sycl::queue& q,
                        std::int64_t row_count,
                        const pr::ndview<Float, 2>& xtx,
                        const pr::ndarray<Float, 1>& sums,
                        const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_covariance, q);
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(xtx.has_data());

    const std::int64_t column_count = xtx.get_dimension(1);

    auto cov = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);

    auto copy_event = copy(q, cov, xtx, { deps });

    auto cov_event = pr::covariance_with_distributed(q, row_count, sums, cov, { copy_event });
    return std::make_tuple(cov, cov_event);
}

template <typename Float>
auto compute_correlation(sycl::queue& q,
                         std::int64_t row_count,
                         const pr::ndview<Float, 2>& xtx,
                         const pr::ndarray<Float, 1>& sums,
                         const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_correlation, q);
    ONEDAL_ASSERT(sums.has_data());

    const std::int64_t column_count = xtx.get_dimension(1);

    auto tmp = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);

    auto corr = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);

    auto copy_event = copy(q, corr, xtx, { deps });

    auto corr_event =
        pr::correlation_with_distributed(q, row_count, sums, corr, tmp, { copy_event });

    auto smart_event = bk::smart_event{ corr_event }.attach(tmp);
    return std::make_tuple(corr, smart_event);
}

template <typename Float>
result_t compute_kernel_dense_impl<Float>::operator()(const descriptor_t& desc,
                                                      const input_t& input) {
    ONEDAL_ASSERT(data.has_data());

    const auto data = input.get_data();
    std::int64_t row_count = data.get_row_count();
    auto rows_count_global = row_count;
    std::int64_t column_count = data.get_column_count();
    auto result = compute_result<task_t>{}.set_result_options(desc.get_result_options());

    const auto data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);

    auto [sums, sums_event] = compute_sums(q_, data_nd);

    comm_.allreduce(sums.flatten(q_, { sums_event }), spmd::reduce_op::sum).wait();

    auto xtx = pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, alloc::device);

    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, q_);
        gemm_event = gemm(q_, data_nd.t(), data_nd, xtx, Float(1.0), Float(0.0));
        gemm_event.wait_and_throw();
    }

    comm_.allreduce(xtx.flatten(q_, { gemm_event }), spmd::reduce_op::sum).wait();

    comm_.allreduce(rows_count_global, spmd::reduce_op::sum).wait();

    if (desc.get_result_options().test(result_options::cov_matrix)) {
        auto [cov, cov_event] =
            compute_covariance(q_, rows_count_global, xtx, sums, { gemm_event });
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
        auto [means, means_event] = compute_means(q_, sums, rows_count_global, { gemm_event });
        result.set_means(homogen_table::wrap(means.flatten(q_, { gemm_event }), 1, column_count));
    }
    return result;
}

template class compute_kernel_dense_impl<float>;
template class compute_kernel_dense_impl<double>;

} // namespace oneapi::dal::covariance::backend

#endif // ONEDAL_DATA_PARALLEL
