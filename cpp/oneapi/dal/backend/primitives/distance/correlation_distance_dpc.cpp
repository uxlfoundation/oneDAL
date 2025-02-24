/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/distance/correlation_distance_misc.hpp"

#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
template <ndorder order>
auto distance<Float, correlation_metric<Float>>::get_inversed_norms(const ndview<Float, 2, order>& inp,
                                                               const event_vector& deps) const
    -> inv_norms_res_t {
    return compute_inversed_l2_norms(q_, inp, deps);
}

template <typename Float>
template <ndorder order1, ndorder order2>
sycl::event distance<Float, correlation_metric<Float>>::operator()(const ndview<Float, 2, order1>& inp1,
                                                              const ndview<Float, 2, order2>& inp2,
                                                              ndview<Float, 2>& out,
                                                              const ndview<Float, 1>& inp1_norms,
                                                              const ndview<Float, 1>& inp2_norms,
                                                              const event_vector& deps) const {
    auto ip_event = compute_correlation_inner_product(q_, inp1, inp2, out, deps);
    return finalize_correlation(q_, inp1_norms, inp2_norms, out, { ip_event });
}

template <typename Float>
template <ndorder order1, ndorder order2>
sycl::event distance<Float, correlation_metric<Float>>::operator()(const ndview<Float, 2, order1>& inp1,
                                                              const ndview<Float, 2, order2>& inp2,
                                                              ndview<Float, 2>& out,
                                                              const event_vector& deps) const {
    const std::int64_t n = inp1.get_dimension(0);
    const std::int64_t p = inp1.get_dimension(1);
    auto inp1_sum = ndarray<Float, 1>::empty(q, { n });
    auto inp2_sum = ndarray<Float, 1>::empty(q, { n });
    auto inp1_mean = ndarray<Float, 1>::empty(q, { n });
    auto inp2_mean = ndarray<Float, 1>::empty(q, { n });
    sycl::event evt1 = reduce_by_rows(q,
                                      inp1,
                                      inp1_sum,
                                      reduction::plus<Float>{},
                                      reduction::identity<Float>{},
                                      deps);
    sycl::event evt2 = reduce_by_rows(q,
                                      inp2,
                                      inp2_sum,
                                      reduction::plus<Float>{},
                                      reduction::identity<Float>{},
                                      { evt1 });
    sycl::event evt3 = means(q, p, inp1_sum, inp1_mean, { evt2 });
    sycl::event evt4 = means(q, p, inp2_sum, inp2_mean, { evt3 });
    auto centered_inp1 = ndarray<Float, 2>::empty(q, { n, p });
    auto centered_inp2 = ndarray<Float, 2>::empty(q, { n, p });
    sycl::event evt5 = q.submit([&](sycl::handler& h) {
        h.depends_on({ evt4 });
        auto inp1_acc = inp1.get_access<sycl::access::mode::read>(h);
        auto inp2_acc = inp2.get_access<sycl::access::mode::read>(h);
        auto inp1_mean_acc = inp1_mean.get_access<sycl::access::mode::read>(h);
        auto inp2_mean_acc = inp2_mean.get_access<sycl::access::mode::read>(h);
        auto centered1_acc = centered_inp1.get_access<sycl::access::mode::write>(h);
        auto centered2_acc = centered_inp2.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<2>(n, p), [=](sycl::id<2> idx) {
            const std::int64_t row = idx[0];
            const std::int64_t col = idx[1];
            centered1_acc(row, col) = inp1_acc(row, col) - inp1_mean_acc[row];
            centered2_acc(row, col) = inp2_acc(row, col) - inp2_mean_acc[row];
        });
    });
    evt5.wait();

    auto [inv_norms1_array, inv_norms1_event] = get_inversed_norms(centered_inp1, { evt5 });
    auto [inv_norms2_array, inv_norms2_event] = get_inversed_norms(centered_inp2, { evt5 });
    return this->operator()(inp1,
                            inp2,
                            out,
                            inv_norms1_array,
                            inv_norms2_array,
                            { inv_norms1_event, inv_norms2_event });
}

#define INSTANTIATE(F, A, B)                                                                   \
    template sycl::event distance<F, correlation_metric<F>>::operator()(const ndview<F, 2, A>&,     \
                                                                   const ndview<F, 2, B>&,     \
                                                                   ndview<F, 2>&,              \
                                                                   const ndview<F, 1>&,        \
                                                                   const ndview<F, 1>&,        \
                                                                   const event_vector&) const; \
    template sycl::event distance<F, correlation_metric<F>>::operator()(const ndview<F, 2, A>&,     \
                                                                   const ndview<F, 2, B>&,     \
                                                                   ndview<F, 2>&,              \
                                                                   const event_vector&) const;

#define INSTANTIATE_B(F, A)                                                       \
    INSTANTIATE(F, A, ndorder::c)                                                 \
    INSTANTIATE(F, A, ndorder::f)                                                 \
    template std::tuple<ndarray<F, 1>, sycl::event>                               \
    distance<F, correlation_metric<F>>::get_inversed_norms(const ndview<F, 2, A>& inp, \
                                                      const event_vector& deps) const;

#define INSTANTIATE_F(F)         \
    INSTANTIATE_B(F, ndorder::c) \
    INSTANTIATE_B(F, ndorder::f) \
    template class distance<F, squared_l2_metric<F>>;

INSTANTIATE_F(float);
INSTANTIATE_F(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
