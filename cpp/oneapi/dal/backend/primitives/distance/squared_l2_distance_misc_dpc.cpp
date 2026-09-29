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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_misc.hpp"

#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, ndorder order>
sycl::event compute_squared_l2_norms(sycl::queue& q,
                                     const ndview<Float, 2, order>& inp,
                                     ndview<Float, 1>& out,
                                     const event_vector& deps) {
    return reduce_by_rows(q, inp, out, sum<Float>{}, square<Float>{}, deps);
}

template <typename Float, ndorder order>
std::tuple<ndarray<Float, 1>, sycl::event> compute_squared_l2_norms(
    sycl::queue& q,
    const ndview<Float, 2, order>& inp,
    const event_vector& deps,
    const sycl::usm::alloc& alloc) {
    const auto n_samples = inp.get_dimension(0);
    auto res_array = ndarray<Float, 1>::empty(q, { n_samples }, alloc);
    return { res_array, compute_squared_l2_norms(q, inp, res_array, deps) };
}

template <typename Float>
sycl::event scatter_2d(sycl::queue& q,
                       const ndview<Float, 1>& inp1,
                       const ndview<Float, 1>& inp2,
                       ndview<Float, 2>& out,
                       const event_vector& deps) {
    ONEDAL_PROFILER_TASK(distance.scatter_2d, q);

    const auto n_samples1 = inp1.get_count();
    const auto n_samples2 = inp2.get_count();
    const auto out_stride = out.get_leading_stride();

    ONEDAL_ASSERT(n_samples1 <= out.get_dimension(0));
    ONEDAL_ASSERT(n_samples2 <= out.get_dimension(1));
    const auto* const inp1_ptr = inp1.get_data();
    const auto* const inp2_ptr = inp2.get_data();
    auto* const out_ptr = out.get_mutable_data();

    // `n_samples1 * n_samples2` can leave int32 for a large square output, so
    // scatter in row blocks instead of one grid. Each block writes its own rows
    // and only depends on `deps`, so the launches are free to overlap.
    const auto block_rows = max_range_2d_rows(n_samples2);

    event_vector block_events;
    for (std::int64_t row_start = 0; row_start < n_samples1; row_start += block_rows) {
        const auto rows = std::min(block_rows, n_samples1 - row_start);
        block_events.push_back(q.submit([&](sycl::handler& h) {
            h.depends_on(deps);

            h.parallel_for(make_range_2d(rows, n_samples2), [=](sycl::id<2> idx) {
                const auto row = row_start + static_cast<std::int64_t>(idx[0]);
                auto* out_place = out_ptr + out_stride * row + idx[1];
                *out_place = inp1_ptr[row] + inp2_ptr[idx[1]];
            });
        }));
    }

    if (block_events.size() == 1u) {
        return block_events.front();
    }
    return q.submit([&](sycl::handler& h) {
        h.depends_on(block_events);
        h.single_task([]() {});
    });
}

template <typename Float, ndorder order1, ndorder order2>
sycl::event compute_inner_product(sycl::queue& q,
                                  const ndview<Float, 2, order1>& inp1,
                                  const ndview<Float, 2, order2>& inp2,
                                  ndview<Float, 2>& out,
                                  const event_vector& deps) {
    check_inputs(inp1, inp2, out);
    auto event = gemm(q, inp1, inp2.t(), out, Float(-2.0), Float(+1.0), deps);
    // Workaround for abort in async mode. Should be removed later.
    event.wait_and_throw();
    return event;
}

#define INSTANTIATE(F, A, B)                                                    \
    template sycl::event compute_inner_product<F, A, B>(sycl::queue & q,        \
                                                        const ndview<F, 2, A>&, \
                                                        const ndview<F, 2, B>&, \
                                                        ndview<F, 2>&,          \
                                                        const event_vector&);

#define INSTANTIATE_A(F, B)                                                         \
    INSTANTIATE(F, ndorder::c, B)                                                   \
    INSTANTIATE(F, ndorder::f, B)                                                   \
    template sycl::event compute_squared_l2_norms<F, B>(sycl::queue&,               \
                                                        const ndview<F, 2, B>&,     \
                                                        ndview<F, 1>&,              \
                                                        const event_vector&);       \
    template std::tuple<ndarray<F, 1>, sycl::event> compute_squared_l2_norms<F, B>( \
        sycl::queue&,                                                               \
        const ndview<F, 2, B>&,                                                     \
        const event_vector&,                                                        \
        const sycl::usm::alloc&);

#define INSTANTIATE_F(F)                                    \
    INSTANTIATE_A(F, ndorder::c)                            \
    INSTANTIATE_A(F, ndorder::f)                            \
    template sycl::event scatter_2d<F>(sycl::queue & q,     \
                                       const ndview<F, 1>&, \
                                       const ndview<F, 1>&, \
                                       ndview<F, 2>&,       \
                                       const event_vector&);

INSTANTIATE_F(float);
INSTANTIATE_F(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
