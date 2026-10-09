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
#include <utility>

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/distance/distance.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, ndorder ord, typename Idx>
struct dkeeper {};

template <typename Float, typename Idx>
struct dkeeper<Float, ndorder::c, Idx> {
    using row_iterator = const Float*;

    auto get_row_bound_iterators(Idx idx) const {
        return std::make_pair(get_first_in_row_iterator(idx), get_last_in_row_iterator(idx));
    }

    row_iterator get_first_in_row_iterator(Idx idx) const {
        return (ptr + idx * str);
    }

    row_iterator get_last_in_row_iterator(Idx idx) const {
        return (ptr + idx * str + width);
    }

    const Float* const ptr;
    const Idx width;
    const Idx str;
};

template <typename Float, typename Idx>
struct dkeeper<Float, ndorder::f, Idx> {
    struct row_iterator {
        explicit row_iterator(Idx col, Idx str, const Float* row)
                : col_(std::move(col)),
                  str_(std::move(str)),
                  row_(std::move(row)) {}

        const Float& operator*() const {
            return *(row_ + col_ * str_);
        }

        row_iterator& operator++() {
            ++col_;
            return *this;
        }

        bool operator!=(const row_iterator& rhs) const {
            const auto& lhs = *this;
            return lhs.col_ != rhs.col_;
        }

    private:
        Idx col_ = 0;
        const Idx str_;
        const Float* const row_;
    };

    auto get_row_bound_iterators(Idx idx) const {
        return std::make_pair(get_first_in_row_iterator(idx), get_last_in_row_iterator(idx));
    }

    row_iterator get_first_in_row_iterator(Idx idx) const {
        const Float* const row = ptr + idx;
        return row_iterator{ Idx(0), str, row };
    }

    row_iterator get_last_in_row_iterator(Idx idx) const {
        const Float* const row = ptr + idx;
        return row_iterator{ width, str, row };
    }

    const Float* const ptr;
    const Idx width;
    const Idx str;
};

template <typename Float, ndorder order, typename Idx = std::int64_t>
auto make_dkeeper(const ndview<Float, 2, order>& data) {
    return dkeeper<Float, order, Idx>{ data.get_data(),
                                       data.get_dimension(1),
                                       data.get_leading_stride() };
}

template <typename Float, typename Metric>
template <ndorder order1, ndorder order2>
sycl::event distance<Float, Metric>::operator()(const ndview<Float, 2, order1>& inp1,
                                                const ndview<Float, 2, order2>& inp2,
                                                ndview<Float, 2>& out,
                                                const event_vector& deps) const {
    ONEDAL_PROFILER_TASK(distance.default, q_);

    check_inputs(inp1, inp2, out);
    // Getting raw USM pointers
    auto* out_ptr = out.get_mutable_data();
    // Getting info about dimensions
    const auto dkeeper1 = make_dkeeper(inp1);
    const auto dkeeper2 = make_dkeeper(inp2);
    const auto n_samples1 = inp1.get_dimension(0);
    const auto n_samples2 = inp2.get_dimension(0);
    // Getting info about strides
    const auto out_stride = out.get_leading_stride();
    // Metric instance
    const auto& metric = this->m_;

    // `n_samples1 * n_samples2` can leave int32 for a large square output.
    return parallel_for_2d_by_row_blocks(
        q_,
        n_samples1,
        n_samples2,
        [=](std::int64_t row, std::int64_t col) {
            auto [f1, l1] = dkeeper1.get_row_bound_iterators(row);
            auto [f2, l2] = dkeeper2.get_row_bound_iterators(col);
            out_ptr[out_stride * row + col] = metric(f1, l1, f2);
        },
        deps);
}

#define INSTANTIATE(F, A, B)                                                                  \
    template sycl::event distance<F, lp_metric<F>>::operator()(const ndview<F, 2, A>&,        \
                                                               const ndview<F, 2, B>&,        \
                                                               ndview<F, 2>&,                 \
                                                               const event_vector&) const;    \
    template sycl::event distance<F, chebyshev_metric<F>>::operator()(const ndview<F, 2, A>&, \
                                                                      const ndview<F, 2, B>&, \
                                                                      ndview<F, 2>&,          \
                                                                      const event_vector&) const;

#define INSTANTIATE_B(F, A)       \
    INSTANTIATE(F, A, ndorder::c) \
    INSTANTIATE(F, A, ndorder::f)

#define INSTANTIATE_F(F)                      \
    INSTANTIATE_B(F, ndorder::c)              \
    INSTANTIATE_B(F, ndorder::f)              \
    template class distance<F, lp_metric<F>>; \
    template class distance<F, chebyshev_metric<F>>;

INSTANTIATE_F(float);
INSTANTIATE_F(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
