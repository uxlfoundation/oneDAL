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

#include "oneapi/dal/algo/svm/backend/gpu/working_set_selector.hpp"
#include "oneapi/dal/backend/primitives/selection/select_flagged.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::svm::backend {

namespace pr = dal::backend::primitives;
namespace de = dal::detail;

template <typename Float>
working_set_selector<Float>::working_set_selector(const sycl::queue& q,
                                                  const pr::ndarray<Float, 1>& labels,
                                                  const Float C,
                                                  const std::int32_t row_count)
        : q_(q),
          row_count_(row_count),
          C_(C),
          labels_(labels) {
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(row_count <= de::limits<std::int32_t>::max());
    ONEDAL_ASSERT(labels.get_count() == row_count);

    auto [indicator, indicator_event] =
        pr::ndarray<std::uint8_t, 1>::zeros(q_, { row_count_ }, sycl::usm::alloc::device);
    sorted_f_indices_ =
        pr::ndarray<std::int32_t, 1>::empty(q_, { row_count_ }, sycl::usm::alloc::device);

    ws_count_ = propose_working_set_size(q_, row_count);

    tmp_sort_values_ = pr::ndarray<Float, 1>::empty(q_, { row_count_ }, sycl::usm::alloc::device);
    buff_indices_ =
        pr::ndarray<std::int32_t, 1>::empty(q_, { row_count_ }, sycl::usm::alloc::device);

    indicator_event.wait_and_throw();
    indicator_ = indicator;
}

template <typename Float>
std::tuple<const std::int32_t, sycl::event> working_set_selector<Float>::select_ws_edge(
    const pr::ndview<Float, 1>& alpha,
    pr::ndview<std::int32_t, 1>& ws_indices,
    const std::int32_t need_select_count,
    const std::int32_t already_selected,
    violating_edge edge,
    const dal::backend::event_vector& deps) {
    ONEDAL_ASSERT(ws_indices.get_dimension(0) >= need_select_count);
    ONEDAL_ASSERT(ws_indices.get_dimension(0) == ws_count_);
    ONEDAL_ASSERT(alpha.get_dimension(0) == row_count_);
    ONEDAL_PROFILER_TASK(select_ws.select_ws_edge, q_);
    auto select_ws_edge_event =
        check_violating_edge(q_, labels_, alpha, indicator_, C_, edge, deps);

    /* Reset indicator for busy Indices */
    if (already_selected > 0) {
        select_ws_edge_event =
            reset_indicator(ws_indices, indicator_, already_selected, { select_ws_edge_event });
    }
    std::int64_t select_flagged_count = 0;
    auto select_flagged = pr::select_flagged_index<std::int32_t, std::uint8_t>{ q_ };
    select_ws_edge_event = select_flagged(indicator_,
                                          sorted_f_indices_,
                                          buff_indices_,
                                          select_flagged_count,
                                          { select_ws_edge_event });

    const std::int32_t select_count =
        std::min(de::integral_cast<std::int32_t>(select_flagged_count), need_select_count);

    std::int32_t* ws_indices_ptr = ws_indices.get_mutable_data();
    const std::int32_t* buff_indices_ptr = buff_indices_.get_data();

    if (select_count > 0) {
        std::int32_t offset = 0;
        if (edge == violating_edge::low) {
            offset = select_flagged_count - select_count;
        }
        select_ws_edge_event = dal::backend::copy(q_,
                                                  ws_indices_ptr + already_selected,
                                                  buff_indices_ptr + offset,
                                                  select_count,
                                                  { select_ws_edge_event });
    }

    return { select_count, select_ws_edge_event };
}

template <typename Float>
sycl::event working_set_selector<Float>::select(const pr::ndview<Float, 1>& alpha,
                                                const pr::ndview<Float, 1>& f,
                                                pr::ndview<std::int32_t, 1>& ws_indices,
                                                std::int32_t selected_count,
                                                const dal::backend::event_vector& deps) {
    ONEDAL_PROFILER_TASK(select_ws, q_);
    ONEDAL_ASSERT(labels_.get_dimension(0) == alpha.get_dimension(0));
    ONEDAL_ASSERT(labels_.get_dimension(0) == f.get_dimension(0));
    ONEDAL_ASSERT(alpha.get_dimension(0) == f.get_dimension(0));
    ONEDAL_ASSERT(ws_indices.get_dimension(0) == ws_count_);
    ONEDAL_ASSERT(ws_indices.has_mutable_data());

    std::int32_t left_to_select = ws_count_ - selected_count;
    std::int32_t already_selected = selected_count;

    auto event = sort_f_indices(q_, f, deps);

    const std::int32_t need_select_up = left_to_select / 2;
    std::tie(selected_count, event) = select_ws_edge(alpha,
                                                     ws_indices,
                                                     need_select_up,
                                                     already_selected,
                                                     violating_edge::up,
                                                     { event });
    left_to_select -= selected_count;
    already_selected += selected_count;

    const std::int32_t need_select_count_low = left_to_select;
    std::tie(selected_count, event) = select_ws_edge(alpha,
                                                     ws_indices,
                                                     need_select_count_low,
                                                     already_selected,
                                                     violating_edge::low,
                                                     { event });
    left_to_select -= selected_count;
    already_selected += selected_count;

    if (left_to_select > 0) {
        std::tie(selected_count, event) = select_ws_edge(alpha,
                                                         ws_indices,
                                                         left_to_select,
                                                         already_selected,
                                                         violating_edge::up,
                                                         { event });
        left_to_select -= selected_count;
        already_selected += selected_count;
    }
    ONEDAL_ASSERT(left_to_select == 0);

    return event;
}

template <typename Float>
sycl::event working_set_selector<Float>::reset_indicator(const pr::ndview<std::int32_t, 1>& idx,
                                                         pr::ndview<std::uint8_t, 1>& indicator,
                                                         const std::int32_t need_to_reset,
                                                         const dal::backend::event_vector& deps) {
    ONEDAL_PROFILER_TASK(select_ws.select_ws_edge.reset_indicator, q_);
    ONEDAL_ASSERT(idx.get_dimension(0) == ws_count_);
    ONEDAL_ASSERT(need_to_reset <= ws_count_);
    ONEDAL_ASSERT(indicator.get_dimension(0) == row_count_);
    ONEDAL_ASSERT(indicator.has_mutable_data());

    const std::int32_t* idx_ptr = idx.get_data();
    std::uint8_t* indicator_ptr = indicator.get_mutable_data();

    const auto wg_size =
        std::min(de::integral_cast<std::int32_t>(dal::backend::propose_wg_size(q_)), need_to_reset);
    const auto range = dal::backend::make_multiple_nd_range_1d(need_to_reset, wg_size);

    auto reset_indicator_event = q_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);

        cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
            const std::int32_t i = item.get_global_id(0);
            indicator_ptr[idx_ptr[i]] = 0;
        });
    });

    return reset_indicator_event;
}

template <typename Float>
sycl::event working_set_selector<Float>::sort_f_indices(sycl::queue& q,
                                                        const pr::ndview<Float, 1>& f,
                                                        const dal::backend::event_vector& deps) {
    ONEDAL_PROFILER_TASK(select_ws.sort_f_indices, q);
    ONEDAL_ASSERT(f.get_dimension(0) == row_count_);

    const Float* f_ptr = f.get_data();
    Float* tmp_sort_ptr = tmp_sort_values_.get_mutable_data();

    auto copy_event = dal::backend::copy(q, tmp_sort_ptr, f_ptr, row_count_, deps);
    auto arange_event = sorted_f_indices_.arange(q);
    auto radix_sort = pr::radix_sort_indices_inplace<Float, std::int32_t>{ q };
    auto radix_sort_event =
        radix_sort(tmp_sort_values_, sorted_f_indices_, { copy_event, arange_event });

    return radix_sort_event;
}

template class working_set_selector<float>;
template class working_set_selector<double>;

} // namespace oneapi::dal::svm::backend

#endif
