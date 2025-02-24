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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/distance/cosine_distance_misc.hpp"
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_misc.hpp"

#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
inline sycl::event inverse_l2_norms(sycl::queue& q,
                                    ndview<Float, 1>& out,
                                    const event_vector& deps) {
    ONEDAL_PROFILER_TASK(distance.inverse_l2_norms, q);

    ONEDAL_ASSERT(out.has_mutable_data());
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const auto count = out.get_count();
        const auto range = make_range_1d(count);
        auto* const ptr = out.get_mutable_data();
        h.parallel_for(range, [=](sycl::id<1> idx) {
            auto& ref = ptr[idx];
            ref = sycl::rsqrt(ref);
        });
    });
}

template <typename Float, ndorder order>
sycl::event compute_inversed_l2_norms(sycl::queue& q,
                                      const ndview<Float, 2, order>& inp,
                                      ndview<Float, 1>& out,
                                      const event_vector& deps) {
    ONEDAL_ASSERT(inp.has_data());
    ONEDAL_ASSERT(out.has_mutable_data());
    auto sq_event = compute_squared_l2_norms(q, inp, out, deps);
    return inverse_l2_norms(q, out, { sq_event });
}

template <typename Float, ndorder order>
std::tuple<ndarray<Float, 1>, sycl::event> compute_inversed_l2_norms(
    sycl::queue& q,
    const ndview<Float, 2, order>& inp,
    const event_vector& deps,
    const sycl::usm::alloc& alloc) {
    const auto n_samples = inp.get_dimension(0);
    auto res_array = ndarray<Float, 1>::empty(q, { n_samples }, alloc);
    return { res_array, compute_inversed_l2_norms(q, inp, res_array, deps) };
}

template <typename Float>
sycl::event finalize_cosine(sycl::queue& q,
                            const ndview<Float, 1>& inp1,
                            const ndview<Float, 1>& inp2,
                            ndview<Float, 2>& out,
                            const event_vector& deps) {
    ONEDAL_PROFILER_TASK(distance.finalize_cosine, q);

    ONEDAL_ASSERT(inp1.has_data());
    ONEDAL_ASSERT(inp2.has_data());
    ONEDAL_ASSERT(out.has_mutable_data());
    const auto out_stride = out.get_leading_stride();
    const auto n_samples1 = inp1.get_dimension(0);
    const auto n_samples2 = inp2.get_dimension(0);
    ONEDAL_ASSERT(n_samples1 <= out.get_dimension(0));
    ONEDAL_ASSERT(n_samples2 <= out.get_dimension(1));
    const auto* const inp1_ptr = inp1.get_data();
    const auto* const inp2_ptr = inp2.get_data();
    auto* const out_ptr = out.get_mutable_data();
    const auto out_range = make_range_2d(n_samples1, n_samples2);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(out_range, [=](sycl::id<2> idx) {
            constexpr Float one = 1;
            auto& out = *(out_ptr + out_stride * idx[0] + idx[1]);
            out = one - out * inp1_ptr[idx[0]] * inp2_ptr[idx[1]];
        });
    });
}

template <typename Float, ndorder order1, ndorder order2>
sycl::event compute_cosine_inner_product(sycl::queue& q,
                                         const ndview<Float, 2, order1>& inp1,
                                         const ndview<Float, 2, order2>& inp2,
                                         ndview<Float, 2>& out,
                                         const event_vector& deps) {
    check_inputs(inp1, inp2, out);
    auto event = gemm(q, inp1, inp2.t(), out, Float(+1.0), Float(0.0), deps);
    // Workaround for abort in async mode. Should be removed later.
    event.wait_and_throw();
    return event;
}

#define INSTANTIATE(F, A, B)                                                           \
    template sycl::event compute_cosine_inner_product<F, A, B>(sycl::queue&,           \
                                                               const ndview<F, 2, A>&, \
                                                               const ndview<F, 2, B>&, \
                                                               ndview<F, 2>&,          \
                                                               const event_vector&);

#define INSTANTIATE_A(F, B)                                                          \
    INSTANTIATE(F, ndorder::c, B)                                                    \
    INSTANTIATE(F, ndorder::f, B)                                                    \
    template sycl::event compute_inversed_l2_norms<F, B>(sycl::queue&,               \
                                                         const ndview<F, 2, B>&,     \
                                                         ndview<F, 1>&,              \
                                                         const event_vector&);       \
    template std::tuple<ndarray<F, 1>, sycl::event> compute_inversed_l2_norms<F, B>( \
        sycl::queue&,                                                                \
        const ndview<F, 2, B>&,                                                      \
        const event_vector&,                                                         \
        const sycl::usm::alloc&);

#define INSTANTIATE_F(F)                                         \
    INSTANTIATE_A(F, ndorder::c)                                 \
    INSTANTIATE_A(F, ndorder::f)                                 \
    template sycl::event finalize_cosine<F>(sycl::queue & q,     \
                                            const ndview<F, 1>&, \
                                            const ndview<F, 1>&, \
                                            ndview<F, 2>&,       \
                                            const event_vector&);

INSTANTIATE_F(float);
INSTANTIATE_F(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
