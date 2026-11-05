/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/backend/primitives/optimizers/common.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
linear_matrix_operator<Float>::linear_matrix_operator(sycl::queue& q, const ndview<Float, 2>& A)
        : base_matrix_operator<Float>(),
          q_(q),
          A_(A) {}

template <typename Float>
sycl::event linear_matrix_operator<Float>::operator()(const ndview<Float, 1>& vec,
                                                      ndview<Float, 1>& out,
                                                      const event_vector& deps) {
    ONEDAL_ASSERT(A_.get_dimension(1) == vec.get_dimension(0));
    ONEDAL_ASSERT(out.get_dimension(0) == vec.get_dimension(0));
    sycl::event fill_out_event = fill<Float>(q_, out, Float(0), deps);
    return gemv(q_, A_, vec, out, Float(1), Float(0), { fill_out_event });
}

template <typename Float>
sycl::event dot_product(sycl::queue& queue,
                        const ndview<Float, 1>& x,
                        const ndview<Float, 1>& y,
                        Float* res_gpu,
                        Float* res_host,
                        const event_vector& deps) {
    const std::int64_t n = x.get_dimension(0);
    auto* const x_ptr = x.get_mutable_data();
    auto* const y_ptr = y.get_mutable_data();
    sycl::event fill_res_event = queue.fill(res_gpu, Float(0), 1, deps);

    auto reduction_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(fill_res_event);
        const auto range = make_range_1d(n);
        auto sum_reduction = sycl::reduction(res_gpu, sycl::plus<>());
        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum) {
            sum += x_ptr[idx] * y_ptr[idx];
        });
    });

    return dal::backend::memcpy(queue, res_host, res_gpu, sizeof(Float), { reduction_event });
}

template <typename Float>
sycl::event l1_norm(sycl::queue& queue,
                    const ndview<Float, 1>& x,
                    Float* res_gpu,
                    Float* res_host,
                    const event_vector& deps) {
    const std::int64_t n = x.get_dimension(0);
    auto* const x_ptr = x.get_mutable_data();

    sycl::event fill_res_event = queue.fill(res_gpu, Float(0), 1, deps);

    auto reduction_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(fill_res_event);
        const auto range = make_range_1d(n);
        auto sum_reduction = sycl::reduction(res_gpu, sycl::plus<>());
        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum) {
            const Float val = x_ptr[idx];
            sum += sycl::fabs(val);
        });
    });

    return dal::backend::memcpy(queue, res_host, res_gpu, sizeof(Float), { reduction_event });
}

template <typename Float>
sycl::event max_abs(sycl::queue& queue,
                    const ndview<Float, 1>& x,
                    Float* res_gpu,
                    Float* res_host,
                    const event_vector& deps) {
    const std::int64_t n = x.get_dimension(0);
    auto* const x_ptr = x.get_mutable_data();

    sycl::event fill_res_event = queue.fill(res_gpu, Float(0), 1, deps);

    auto reduction_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(fill_res_event);
        const auto range = make_range_1d(n);
        auto max_reduction = sycl::reduction(res_gpu, sycl::maximum<>());
        cgh.parallel_for(range, max_reduction, [=](sycl::id<1> idx, auto& mx) {
            const Float val = x_ptr[idx];
            mx.combine(sycl::fabs(val));
        });
    });

    return dal::backend::memcpy(queue, res_host, res_gpu, sizeof(Float), { reduction_event });
}

#define INSTANTIATE(F)                                        \
    template sycl::event dot_product<F>(sycl::queue&,         \
                                        const ndview<F, 1>&,  \
                                        const ndview<F, 1>&,  \
                                        F*,                   \
                                        F*,                   \
                                        const event_vector&); \
    template sycl::event l1_norm<F>(sycl::queue&,             \
                                    const ndview<F, 1>&,      \
                                    F*,                       \
                                    F*,                       \
                                    const event_vector&);     \
    template sycl::event max_abs<F>(sycl::queue&,             \
                                    const ndview<F, 1>&,      \
                                    F*,                       \
                                    F*,                       \
                                    const event_vector&);     \
    template class base_matrix_operator<F>;                   \
    template class linear_matrix_operator<F>;                 \
    template class base_function<F>;

INSTANTIATE(float);
INSTANTIATE(double);

} // namespace oneapi::dal::backend::primitives
