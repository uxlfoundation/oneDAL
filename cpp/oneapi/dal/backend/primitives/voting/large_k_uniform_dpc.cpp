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

#include <algorithm>

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/backend/primitives/voting/uniform.hpp"

namespace oneapi::dal::backend::primitives {

template <typename ClassType>
large_k_uniform_voting<ClassType>::large_k_uniform_voting(sycl::queue& q,
                                                          std::int64_t max_block,
                                                          std::int64_t k_response)
        : base_t{ q },
          swp_(
              ndarray<ClassType, 2>::empty(q, { max_block, k_response }, sycl::usm::alloc::device)),
          out_(
              ndarray<ClassType, 2>::empty(q, { max_block, k_response }, sycl::usm::alloc::device)),
          sorting_{ q } {}

template <typename ClassType>
sycl::event large_k_uniform_voting<ClassType>::select_winner(ndview<ClassType, 1>& results,
                                                             const event_vector& deps) const {
    const auto inp_str = out_.get_leading_stride();
    const auto inp_wdt = out_.get_dimension(1);
    const auto* const inp_ptr = out_.get_data();
    const auto out_len = results.get_dimension(0);
    auto* const out_ptr = results.get_mutable_data();
    const auto range = make_range_1d(out_len);
    return this->get_queue().submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<1> idx) {
            const auto* const row = inp_ptr + idx * inp_str;
            ClassType last = -1, winner = -1;
            std::int32_t last_span = -1, winner_span = -1;
            for (std::int32_t i = 0; i < inp_wdt; ++i) {
                const ClassType& cur = *(row + i);
                if (cur == last) {
                    ++last_span;
                }
                else {
                    last = cur;
                    last_span = 1;
                }
                if (last_span > winner_span) {
                    winner = last;
                    winner_span = last_span;
                }
            }
            *(out_ptr + idx) = winner;
        });
    });
}

template <typename ClassType>
sycl::event large_k_uniform_voting<ClassType>::operator()(const ndview<ClassType, 2>& responses,
                                                          ndview<ClassType, 1>& results,
                                                          const event_vector& deps) {
    ONEDAL_PROFILER_TASK(voting.uniform, this->get_queue());

    const auto n = responses.get_dimension(0);
    ONEDAL_ASSERT(n <= swp_.get_dimension(0));
    ONEDAL_ASSERT(n <= out_.get_dimension(0));
    [[maybe_unused]] const auto r = responses.get_dimension(1);
    ONEDAL_ASSERT(r == swp_.get_dimension(1));
    ONEDAL_ASSERT(r == out_.get_dimension(1));
    auto swp_slice = swp_.get_row_slice(0, n);
    auto out_slice = out_.get_row_slice(0, n);
    auto cpy_event = copy(this->get_queue(), swp_slice, responses, deps);
    auto srt_event = sorting_(swp_slice, out_slice, { cpy_event });
    return select_winner(results, { srt_event });
}

#define INSTANTIATE(CLASS) template class large_k_uniform_voting<CLASS>;

INSTANTIATE(std::int32_t);
INSTANTIATE(std::int64_t);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
