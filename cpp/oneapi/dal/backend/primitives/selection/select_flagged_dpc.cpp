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

#include "oneapi/dal/backend/primitives/selection/select_flagged.hpp"

#include <sycl/ext/oneapi/experimental/builtins.hpp>

namespace oneapi::dal::backend::primitives {

namespace de = dal::detail;

using sycl::plus;

template <typename Data, typename Flag>
sycl::event select_flagged_base<Data, Flag>::scan(sycl::queue& queue,
                                                  const mask_t& mask_accessor,
                                                  ndarray<integer_t, 1>& part_sum,
                                                  integer_t elem_count,
                                                  integer_t local_size,
                                                  integer_t local_sum_count,
                                                  sycl::event& deps) {
    ONEDAL_ASSERT(part_sum.get_count() == sum_buff_size_);

    const sycl::nd_range<1> nd_range =
        make_multiple_nd_range_1d(de::check_mul_overflow(local_size, local_sum_count), local_size);

    integer_t* part_sum_ptr = part_sum.get_mutable_data();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();

            const integer_t n_groups = item.get_group_range(0);
            const integer_t n_sub_groups = sbg.get_group_range()[0];
            const integer_t n_total_sub_groups = n_sub_groups * n_groups;
            const integer_t elems_for_sbg =
                elem_count / n_total_sub_groups + bool(elem_count % n_total_sub_groups);
            const integer_t local_size = sbg.get_local_range()[0];

            const integer_t local_id = sbg.get_local_id();
            const integer_t sub_group_id = sbg.get_group_id();
            const integer_t group_id = item.get_group(0) * n_sub_groups + sub_group_id;

            integer_t ind_start = group_id * elems_for_sbg;
            integer_t ind_end =
                sycl::min(static_cast<integer_t>((group_id + 1) * elems_for_sbg), elem_count);

            integer_t sum = 0;

            for (integer_t i = ind_start + local_id; i < ind_end; i += local_size) {
                const integer_t value = static_cast<integer_t>(mask_accessor[i]);
                sum += sycl::reduce_over_group(sbg, value, plus<integer_t>());
            }

            if (local_id == 0) {
                part_sum_ptr[group_id] = sum;
            }
        });
    });

    return event;
}

template <typename Data, typename Flag>
sycl::event select_flagged_base<Data, Flag>::sum_scan(sycl::queue& queue,
                                                      const ndarray<integer_t, 1>& part_sum,
                                                      ndarray<integer_t, 1>& part_prefix_sum,
                                                      integer_t local_size,
                                                      integer_t local_sum_count,
                                                      ndarray<integer_t, 1>& total_sum,
                                                      sycl::event& deps) {
    ONEDAL_ASSERT(part_sum.get_count() == sum_buff_size_);
    ONEDAL_ASSERT(part_prefix_sum.get_count() == sum_buff_size_);

    const integer_t* part_sum_ptr = part_sum.get_data();
    integer_t* part_prefix_sum_ptr = part_prefix_sum.get_mutable_data();
    integer_t* total_sum_ptr = total_sum.get_mutable_data();

    const sycl::nd_range<1> nd_range = make_multiple_nd_range_1d(local_size, local_size);

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }

            const integer_t local_size = sbg.get_local_range()[0];
            const integer_t local_id = sbg.get_local_id();

            integer_t sum = 0;
            for (integer_t i = local_id; i < local_sum_count; i += local_size) {
                integer_t value = part_sum_ptr[i];
                integer_t boundary = sycl::exclusive_scan_over_group(sbg, value, plus<integer_t>());
                part_prefix_sum_ptr[i] = sum + boundary;
                sum += sycl::reduce_over_group(sbg, value, plus<integer_t>());
            }

            if (local_id == 0) {
                total_sum_ptr[0] = sum;
                part_prefix_sum_ptr[local_sum_count] = sum;
            }
        });
    });

    return event;
}

template <typename Data, typename Flag>
sycl::event select_flagged_base<Data, Flag>::reorder(sycl::queue& queue,
                                                     const mask_t& mask_accessor,
                                                     const ndview<Data, 1>& in,
                                                     ndview<Data, 1>& out,
                                                     ndarray<integer_t, 1>& part_prefix_sum,
                                                     integer_t elem_count,
                                                     integer_t local_size,
                                                     integer_t local_sum_count,
                                                     sycl::event& deps) {
    ONEDAL_ASSERT(part_prefix_sum.get_count() == sum_buff_size_);

    const sycl::nd_range<1> nd_range =
        make_multiple_nd_range_1d(de::check_mul_overflow(local_size, local_sum_count), local_size);

    integer_t* part_prefix_sum_ptr = part_prefix_sum.get_mutable_data();
    const Data* in_ptr = in.get_data();
    Data* out_ptr = out.get_mutable_data();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();

            const integer_t n_groups = item.get_group_range(0);
            const integer_t n_sub_groups = sbg.get_group_range()[0];
            const integer_t n_total_sub_groups = n_sub_groups * n_groups;
            const integer_t elems_for_sbg =
                elem_count / n_total_sub_groups + bool(elem_count % n_total_sub_groups);
            const integer_t local_size = sbg.get_local_range()[0];

            const integer_t local_id = sbg.get_local_id();
            const integer_t sub_group_id = sbg.get_group_id();
            const integer_t group_id = item.get_group(0) * n_sub_groups + sub_group_id;

            integer_t ind_start = group_id * elems_for_sbg;
            integer_t ind_end =
                sycl::min(static_cast<integer_t>((group_id + 1) * elems_for_sbg), elem_count);

            integer_t group_offset = part_prefix_sum_ptr[group_id];

            integer_t sum = 0;

            for (integer_t i = ind_start + local_id; i < ind_end; i += local_size) {
                const integer_t part = static_cast<integer_t>(mask_accessor[i]);
                const integer_t boundary =
                    group_offset + sum +
                    sycl::exclusive_scan_over_group(sbg, part, plus<integer_t>());
                if (part)
                    out_ptr[boundary] = in_ptr[i];
                sum += sycl::reduce_over_group(sbg, part, plus<integer_t>());
            }
        });
    });

    return event;
}

template <typename Data, typename Flag>
select_flagged_base<Data, Flag>::select_flagged_base(const sycl::queue& queue)
        : queue_(queue),
          elem_count_(0) {}

template <typename Data, typename Flag>
select_flagged_base<Data, Flag>::~select_flagged_base() {
    select_flagged_base_event_.wait_and_throw();
}

template <typename Data, typename Flag>
void select_flagged_base<Data, Flag>::init(sycl::queue& queue, std::int64_t elem_count) {
    ONEDAL_ASSERT(elem_count > 0);
    ONEDAL_ASSERT(elem_count <= de::limits<integer_t>::max());

    const integer_t uint_elem_count = de::integral_cast<integer_t>(elem_count);
    if (elem_count_ != uint_elem_count) {
        elem_count_ = uint_elem_count;
        local_size_ = preferable_sbg_size_;
        local_sum_count_ = de::check_mul_overflow(max_local_sum_count_, local_size_) < elem_count_
                               ? max_local_sum_count_
                               : (elem_count_ / local_size_) + bool(elem_count_ % local_size_);

        sum_buff_size_ = local_sum_count_ + 1;

        part_sum_ =
            ndarray<integer_t, 1>::empty(queue, { sum_buff_size_ }, sycl::usm::alloc::device);
        part_prefix_sum_ =
            ndarray<integer_t, 1>::empty(queue, { sum_buff_size_ }, sycl::usm::alloc::device);
        total_sum_ = ndarray<integer_t, 1>::empty(queue, { 1 }, sycl::usm::alloc::device);
    }
}

template <typename Data, typename Flag>
sycl::event select_flagged_base<Data, Flag>::select_flagged_base_impl(
    const mask_t& mask_accessor,
    const ndview<Data, 1>& in,
    ndview<Data, 1>& out,
    std::int64_t& selected_elem_count,
    const event_vector& deps) {
    ONEDAL_PROFILER_TASK(selection.select_flagged, queue_);
    ONEDAL_ASSERT(out.has_mutable_data());
    ONEDAL_ASSERT(in.get_count() == out.get_count());

    if (in.get_count() > de::limits<integer_t>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_number_of_elements_to_process());
    }

    sycl::event::wait_and_throw(deps);

    init(queue_, in.get_count());

    auto scan_deps = scan(queue_,
                          mask_accessor,
                          part_sum_,
                          elem_count_,
                          local_size_,
                          local_sum_count_,
                          select_flagged_base_event_);
    auto sum_scan_deps = sum_scan(queue_,
                                  part_sum_,
                                  part_prefix_sum_,
                                  local_size_,
                                  local_sum_count_,
                                  total_sum_,
                                  scan_deps);
    select_flagged_base_event_ = reorder(queue_,
                                         mask_accessor,
                                         in,
                                         out,
                                         part_prefix_sum_,
                                         elem_count_,
                                         local_size_,
                                         local_sum_count_,
                                         sum_scan_deps);

    auto total_sum_host = total_sum_.to_host(queue_, { select_flagged_base_event_ });
    selected_elem_count = static_cast<std::int64_t>(total_sum_host.get_data()[0]);
    return select_flagged_base_event_;
}

#define INSTANTIATE_SELECT_FLAGGED_BASE(D, FLG) \
    template class ONEDAL_EXPORT select_flagged_base<D, FLG>;
#define INSTANTIATE_SELECT_FLAGGED(D, FLG) template class ONEDAL_EXPORT select_flagged<D, FLG>;
#define INSTANTIATE_SELECT_FLAGGED_INDEX(D, FLG) \
    template class ONEDAL_EXPORT select_flagged_index<D, FLG>;

INSTANTIATE_SELECT_FLAGGED_BASE(float, std::uint8_t)
INSTANTIATE_SELECT_FLAGGED_BASE(double, std::uint8_t)
INSTANTIATE_SELECT_FLAGGED_BASE(float, std::uint32_t)
INSTANTIATE_SELECT_FLAGGED_BASE(double, std::uint32_t)
INSTANTIATE_SELECT_FLAGGED_BASE(std::int32_t, std::uint8_t)
INSTANTIATE_SELECT_FLAGGED_BASE(std::uint32_t, std::uint8_t)
INSTANTIATE_SELECT_FLAGGED_BASE(std::int32_t, std::uint32_t)
INSTANTIATE_SELECT_FLAGGED_BASE(std::uint32_t, std::uint32_t)

INSTANTIATE_SELECT_FLAGGED(float, std::uint8_t)
INSTANTIATE_SELECT_FLAGGED(double, std::uint8_t)
INSTANTIATE_SELECT_FLAGGED(float, std::uint32_t)
INSTANTIATE_SELECT_FLAGGED(double, std::uint32_t)
INSTANTIATE_SELECT_FLAGGED_INDEX(std::int32_t, std::uint8_t)
INSTANTIATE_SELECT_FLAGGED_INDEX(std::uint32_t, std::uint8_t)
INSTANTIATE_SELECT_FLAGGED_INDEX(std::int32_t, std::uint32_t)
INSTANTIATE_SELECT_FLAGGED_INDEX(std::uint32_t, std::uint32_t)
} // namespace oneapi::dal::backend::primitives
