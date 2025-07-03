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

#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/distance/cosine_distance_misc.hpp"
#include "oneapi/dal/backend/primitives/distance/correlation_distance_misc.hpp"

#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
template <ndorder order>
auto distance<Float, correlation_metric<Float>>::get_inversed_norms(
    const ndview<Float, 2, order>& inp,
    const event_vector& deps) const -> inv_norms_res_t {
    return compute_inversed_l2_norms(q_, inp, deps);
}

template <typename Float>
template <ndorder order>
auto distance<Float, correlation_metric<Float>>::get_deviation(const ndview<Float, 2, order>& inp,
                                                               const event_vector& deps) const
    -> comp_dev_res_t {
    return compute_deviation(q_, inp, deps);
}

template <typename Float>
template <ndorder order1, ndorder order2>
sycl::event distance<Float, correlation_metric<Float>>::operator()(
    const ndview<Float, 2, order1>& inp1,
    const ndview<Float, 2, order2>& inp2,
    ndview<Float, 2>& out,
    const ndview<Float, 1>& inp1_norms,
    const ndview<Float, 1>& inp2_norms,
    const event_vector& deps) const {
    auto ip_event = compute_cosine_inner_product(q_, inp1, inp2, out, deps);
    auto correlation_event = finalize_cosine(q_, inp1_norms, inp2_norms, out, { ip_event });
    const auto n = out.get_dimension(0);
    const auto p = out.get_dimension(1);
    auto* out_ptr = out.get_mutable_data();
    auto out_stride = out.get_leading_stride();
    auto out_range = make_range_2d(n, p);
    return q_.submit([&](sycl::handler& h) {
        h.depends_on({ correlation_event });
        h.parallel_for(out_range, [=](sycl::id<2> idx) {
            const auto offset = idx[0] * out_stride + idx[1];
            if (std::isnan(out_ptr[offset]) ||
                (std::isnan(out_ptr[offset]) && std::signbit(out_ptr[offset])))
                out_ptr[offset] = 1.0;
        });
    });
}

template <typename Float>
template <ndorder order1, ndorder order2>
sycl::event distance<Float, correlation_metric<Float>>::operator()(
    const ndview<Float, 2, order1>& inp1,
    const ndview<Float, 2, order2>& inp2,
    ndview<Float, 2>& out,
    const event_vector& deps) const {
    auto [centered_inp1, comp_dev1_event] = get_deviation(inp1, deps);
    auto [centered_inp2, comp_dev2_event] = get_deviation(inp2, deps);

    auto [inv_norms1_array, inv_norms1_event] =
        get_inversed_norms(centered_inp1, { comp_dev1_event });
    auto [inv_norms2_array, inv_norms2_event] =
        get_inversed_norms(centered_inp2, { comp_dev2_event });
    return this->operator()(centered_inp1,
                            centered_inp2,
                            out,
                            inv_norms1_array,
                            inv_norms2_array,
                            { inv_norms1_event, inv_norms2_event });
}

#define INSTANTIATE(F, A, B)                                                                    \
    template sycl::event distance<F, correlation_metric<F>>::operator()(const ndview<F, 2, A>&, \
                                                                        const ndview<F, 2, B>&, \
                                                                        ndview<F, 2>&,          \
                                                                        const ndview<F, 1>&,    \
                                                                        const ndview<F, 1>&,    \
                                                                        const event_vector&)    \
        const;                                                                                  \
    template sycl::event distance<F, correlation_metric<F>>::operator()(const ndview<F, 2, A>&, \
                                                                        const ndview<F, 2, B>&, \
                                                                        ndview<F, 2>&,          \
                                                                        const event_vector&)    \
        const;

#define INSTANTIATE_B(F, A)                                                                 \
    INSTANTIATE(F, A, ndorder::c)                                                           \
    INSTANTIATE(F, A, ndorder::f)                                                           \
    template std::tuple<ndarray<F, 1>, sycl::event>                                         \
    distance<F, correlation_metric<F>>::get_inversed_norms(const ndview<F, 2, A>& inp,      \
                                                           const event_vector& deps) const; \
    template std::tuple<ndarray<F, 2>, sycl::event>                                         \
    distance<F, correlation_metric<F>>::get_deviation(const ndview<F, 2, A>& inp,           \
                                                      const event_vector& deps) const;

#define INSTANTIATE_F(F)         \
    INSTANTIATE_B(F, ndorder::c) \
    INSTANTIATE_B(F, ndorder::f) \
    template class distance<F, cosine_metric<F>>;

INSTANTIATE_F(float);
INSTANTIATE_F(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
