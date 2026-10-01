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
#include "oneapi/dal/backend/primitives/distance/cosine_distance_misc.hpp"

#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
template <ndorder order>
auto distance<Float, cosine_metric<Float>>::get_inversed_norms(const ndview<Float, 2, order>& inp,
                                                               const event_vector& deps) const
    -> inv_norms_res_t {
    return compute_inversed_l2_norms(q_, inp, deps);
}

template <typename Float>
template <ndorder order1, ndorder order2>
sycl::event distance<Float, cosine_metric<Float>>::operator()(const ndview<Float, 2, order1>& inp1,
                                                              const ndview<Float, 2, order2>& inp2,
                                                              ndview<Float, 2>& out,
                                                              const ndview<Float, 1>& inp1_norms,
                                                              const ndview<Float, 1>& inp2_norms,
                                                              const event_vector& deps) const {
    auto ip_event = compute_cosine_inner_product(q_, inp1, inp2, out, deps);
    return finalize_cosine(q_, inp1_norms, inp2_norms, out, { ip_event });
}

template <typename Float>
template <ndorder order1, ndorder order2>
sycl::event distance<Float, cosine_metric<Float>>::operator()(const ndview<Float, 2, order1>& inp1,
                                                              const ndview<Float, 2, order2>& inp2,
                                                              ndview<Float, 2>& out,
                                                              const event_vector& deps) const {
    auto [inv_norms1_array, inv_norms1_event] = get_inversed_norms(inp1, deps);
    auto [inv_norms2_array, inv_norms2_event] = get_inversed_norms(inp2, deps);
    auto last_event = this->operator()(inp1,
                                       inp2,
                                       out,
                                       inv_norms1_array,
                                       inv_norms2_array,
                                       { inv_norms1_event, inv_norms2_event });

    // The inversed-norm arrays are owned by this overload, and freeing device USM is not a
    // synchronizing operation: returning while `finalize_cosine` still reads them would let the
    // array destructors free memory out from under the running kernel.
    last_event.wait_and_throw();

    return last_event;
}

#define INSTANTIATE(F, A, B)                                                                   \
    template sycl::event distance<F, cosine_metric<F>>::operator()(const ndview<F, 2, A>&,     \
                                                                   const ndview<F, 2, B>&,     \
                                                                   ndview<F, 2>&,              \
                                                                   const ndview<F, 1>&,        \
                                                                   const ndview<F, 1>&,        \
                                                                   const event_vector&) const; \
    template sycl::event distance<F, cosine_metric<F>>::operator()(const ndview<F, 2, A>&,     \
                                                                   const ndview<F, 2, B>&,     \
                                                                   ndview<F, 2>&,              \
                                                                   const event_vector&) const;

#define INSTANTIATE_B(F, A)                                                       \
    INSTANTIATE(F, A, ndorder::c)                                                 \
    INSTANTIATE(F, A, ndorder::f)                                                 \
    template std::tuple<ndarray<F, 1>, sycl::event>                               \
    distance<F, cosine_metric<F>>::get_inversed_norms(const ndview<F, 2, A>& inp, \
                                                      const event_vector& deps) const;

#define INSTANTIATE_F(F)         \
    INSTANTIATE_B(F, ndorder::c) \
    INSTANTIATE_B(F, ndorder::f) \
    template class distance<F, squared_l2_metric<F>>;

INSTANTIATE_F(float);
INSTANTIATE_F(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
