
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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/distance/correlation_distance_misc.hpp"

#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat/cov.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, ndorder order>
sycl::event compute_deviation(sycl::queue& q,
                              const ndview<Float, 2, order>& inp,
                              ndview<Float, 2>& out,
                              const event_vector& deps) {
    ONEDAL_ASSERT(inp.has_data());
    ONEDAL_ASSERT(out.has_mutable_data());
    const auto n = out.get_dimension(0);
    const auto p = out.get_dimension(1);
    ONEDAL_ASSERT(n == inp.get_dimension(0));
    ONEDAL_ASSERT(p == inp.get_dimension(1));
    auto* inp_ptr = inp.get_data();
    auto* out_ptr = out.get_mutable_data();
    auto out_stride = out.get_leading_stride();
    auto out_range = make_range_2d(n, p);
    auto inp_sum = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);
    auto inp_mean = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);

    // Collect sums of each row of input matrix
    auto sums_event = reduce_by_rows(q, inp, inp_sum, sum<Float>{}, identity<Float>{}, deps);

    // Compute mean of each row of input matrix using sum event
    auto means_event = means(q, p, inp_sum, inp_mean, { sums_event });
    auto inp_mean_ptr = inp_mean.get_data();

    // Return event that updates output matrix with centered values (input(x) - input_mean(x))
    return q.submit([&](sycl::handler& h) {
        h.depends_on({ means_event });
        h.parallel_for(out_range, [=](sycl::id<2> idx) {
            const auto offset = idx[0] * out_stride + idx[1];
            out_ptr[offset] = inp_ptr[offset] - inp_mean_ptr[idx[0]];
        });
    });
}

template <typename Float, ndorder order>
std::tuple<ndarray<Float, 2>, sycl::event> compute_deviation(sycl::queue& q,
                                                             const ndview<Float, 2, order>& inp,
                                                             const event_vector& deps,
                                                             const sycl::usm::alloc& alloc) {
    const auto n = inp.get_dimension(0);
    const auto p = inp.get_dimension(1);
    auto res_array = ndarray<Float, 2>::empty(q, { n, p }, alloc);

    // Return output array and event for computing deviations of input matrix
    return { res_array, compute_deviation(q, inp, res_array, deps) };
}

#define INSTANTIATE(F, B)                                                    \
    template sycl::event compute_deviation<F, B>(sycl::queue&,               \
                                                 const ndview<F, 2, B>&,     \
                                                 ndview<F, 2>&,              \
                                                 const event_vector&);       \
    template std::tuple<ndarray<F, 2>, sycl::event> compute_deviation<F, B>( \
        sycl::queue&,                                                        \
        const ndview<F, 2, B>&,                                              \
        const event_vector&,                                                 \
        const sycl::usm::alloc&);
#define INSTANTIATE_F(F)       \
    INSTANTIATE(F, ndorder::c) \
    INSTANTIATE(F, ndorder::f)

INSTANTIATE_F(float);
INSTANTIATE_F(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
