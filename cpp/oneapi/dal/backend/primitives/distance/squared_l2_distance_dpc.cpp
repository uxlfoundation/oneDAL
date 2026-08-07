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

#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_misc.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
template <ndorder order>
auto distance<Float, squared_l2_metric<Float>>::get_norms(const ndview<Float, 2, order>& inp,
                                                          const event_vector& deps) const
    -> norms_res_t {
    return compute_squared_l2_norms(q_, inp, deps);
}

template <typename Float>
template <ndorder order1, ndorder order2>
sycl::event distance<Float, squared_l2_metric<Float>>::operator()(
    const ndview<Float, 2, order1>& inp1,
    const ndview<Float, 2, order2>& inp2,
    ndview<Float, 2>& out,
    const ndview<Float, 1>& inp1_norms,
    const ndview<Float, 1>& inp2_norms,
    const event_vector& deps) const {
    check_inputs(inp1, inp2, out);
    auto scatter_event = scatter_2d(q_, inp1_norms, inp2_norms, out, { deps });
    return compute_inner_product(q_, inp1, inp2, out, { scatter_event });
}

template <typename Float>
template <ndorder order1, ndorder order2>
sycl::event distance<Float, squared_l2_metric<Float>>::operator()(
    const ndview<Float, 2, order1>& inp1,
    const ndview<Float, 2, order2>& inp2,
    ndview<Float, 2>& out,
    const event_vector& deps) const {
    auto [norms1_array, norms1_event] = get_norms(inp1, deps);
    auto [norms2_array, norms2_event] = get_norms(inp2, deps);
    return this
        ->operator()(inp1, inp2, out, norms1_array, norms2_array, { norms1_event, norms2_event });
}

#define INSTANTIATE(F, A, B)                                                                       \
    template sycl::event distance<F, squared_l2_metric<F>>::operator()(const ndview<F, 2, A>&,     \
                                                                       const ndview<F, 2, B>&,     \
                                                                       ndview<F, 2>&,              \
                                                                       const ndview<F, 1>&,        \
                                                                       const ndview<F, 1>&,        \
                                                                       const event_vector&) const; \
    template sycl::event distance<F, squared_l2_metric<F>>::operator()(const ndview<F, 2, A>&,     \
                                                                       const ndview<F, 2, B>&,     \
                                                                       ndview<F, 2>&,              \
                                                                       const event_vector&) const;

#define INSTANTIATE_B(F, A)                                                                       \
    INSTANTIATE(F, A, ndorder::c)                                                                 \
    INSTANTIATE(F, A, ndorder::f)                                                                 \
    template std::tuple<ndarray<F, 1>, sycl::event> distance<F, squared_l2_metric<F>>::get_norms( \
        const ndview<F, 2, A>& inp,                                                               \
        const event_vector& deps) const;

#define INSTANTIATE_F(F)         \
    INSTANTIATE_B(F, ndorder::c) \
    INSTANTIATE_B(F, ndorder::f) \
    template class distance<F, squared_l2_metric<F>>;

INSTANTIATE_F(float);
INSTANTIATE_F(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
