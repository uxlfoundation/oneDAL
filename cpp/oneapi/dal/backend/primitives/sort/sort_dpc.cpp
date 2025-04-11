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

#include "oneapi/dal/backend/primitives/sort/sort.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include <sycl/ext/oneapi/experimental/builtins.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-local-typedef"

#include <oneapi/dpl/experimental/kernel_templates>

#pragma clang diagnostic pop

namespace oneapi::dal::backend::primitives {

namespace de = dal::detail;

inline std::uint32_t inv_bits(std::uint32_t x) {
    return x ^ (-(x >> 31) | 0x80000000u);
}

inline std::uint64_t inv_bits(std::uint64_t x) {
    return x ^ (-(x >> 63) | 0x8000000000000000ul);
}

using sycl::ext::oneapi::plus;

template <typename Float, typename Index>
sycl::event radix_sort_indices_inplace<Float, Index>::radix_scan(sycl::queue& queue,
                                                                 const ndview<Float, 1>& val,
                                                                 ndarray<Index, 1>& part_hist,
                                                                 Index elem_count,
                                                                 std::uint32_t bit_offset,
                                                                 std::int64_t local_size,
                                                                 std::int64_t local_hist_count,
                                                                 sycl::event& deps) {
    ONEDAL_ASSERT(part_hist.get_count() == hist_buff_size_);

    const sycl::nd_range<1> nd_range =
        make_multiple_nd_range_1d(de::check_mul_overflow(local_size, local_hist_count), local_size);

    const radix_integer_t* val_ptr = reinterpret_cast<const radix_integer_t*>(val.get_data());
    Index* part_hist_ptr = part_hist.get_mutable_data();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }
            const std::uint32_t n_groups = item.get_group_range(0);
            const std::uint32_t n_sub_groups = sbg.get_group_range()[0];
            const std::uint32_t n_total_sub_groups = n_sub_groups * n_groups;
            const Index elems_for_sbg =
                elem_count / n_total_sub_groups + bool(elem_count % n_total_sub_groups);
            const std::uint32_t local_size = sbg.get_local_range()[0];

            const std::uint32_t local_id = sbg.get_local_id();
            const std::uint32_t sub_group_id = sbg.get_group_id();
            const std::uint32_t group_id = item.get_group(0) * n_sub_groups + sub_group_id;

            Index ind_start = group_id * elems_for_sbg;
            Index ind_end =
                sycl::min(static_cast<Index>((group_id + 1) * elems_for_sbg), elem_count);

            Index offset[radix_range_];
            for (std::uint32_t i = 0; i < radix_range_; i++) {
                offset[i] = 0;
            }

            for (Index i = ind_start + local_id; i < ind_end; i += local_size) {
                radix_integer_t data_bits = ((inv_bits(val_ptr[i]) >> bit_offset) & radix_range_1_);
                for (std::uint32_t j = 0; j < radix_range_; j++) {
                    Index value = static_cast<Index>(data_bits == j);
                    Index partial_offset = sycl::reduce_over_group(sbg, value, plus<Index>());
                    offset[j] += partial_offset;
                }
            }

            if (local_id == 0) {
                for (std::uint32_t j = 0; j < radix_range_; j++) {
                    part_hist_ptr[group_id * radix_range_ + j] = offset[j];
                }
            }
        });
    });

    return event;
}

template <typename Float, typename Index>
sycl::event radix_sort_indices_inplace<Float, Index>::radix_hist_scan(
    sycl::queue& queue,
    const ndarray<Index, 1>& part_hist,
    ndarray<Index, 1>& part_prefix_hist,
    std::int64_t local_size,
    std::int64_t local_hist_count,
    sycl::event& deps) {
    ONEDAL_ASSERT(part_hist.get_count() == hist_buff_size_);
    ONEDAL_ASSERT(part_prefix_hist.get_count() == hist_buff_size_);

    const Index* part_hist_ptr = part_hist.get_data();
    Index* part_prefix_hist_ptr = part_prefix_hist.get_mutable_data();

    const sycl::nd_range<1> nd_range = make_multiple_nd_range_1d(local_size, local_size);

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }

            const std::uint32_t local_size = sbg.get_local_range()[0];
            const std::uint32_t local_id = sbg.get_local_id();

            Index offset[radix_range_];
            for (std::uint32_t i = 0; i < radix_range_; i++) {
                offset[i] = 0;
            }

            for (std::uint32_t i = local_id; i < local_hist_count; i += local_size) {
                for (std::uint32_t j = 0; j < radix_range_; j++) {
                    Index value = part_hist_ptr[i * radix_range_ + j];
                    Index boundary = sycl::exclusive_scan_over_group(sbg, value, plus<Index>());
                    part_prefix_hist_ptr[i * radix_range_ + j] = offset[j] + boundary;
                    Index partial_offset = sycl::reduce_over_group(sbg, value, plus<Index>());
                    offset[j] += partial_offset;
                }
            }

            if (local_id == 0) {
                Index total_sum = 0;
                for (std::uint32_t j = 0; j < radix_range_; j++) {
                    part_prefix_hist_ptr[local_hist_count * radix_range_ + j] = total_sum;
                    total_sum += offset[j];
                }
            }
        });
    });

    return event;
}

template <typename Float, typename Index>
sycl::event radix_sort_indices_inplace<Float, Index>::radix_reorder(
    sycl::queue& queue,
    const ndview<Float, 1>& val_in,
    const ndview<Index, 1>& ind_in,
    const ndview<Index, 1>& part_prefix_hist,
    ndview<Float, 1>& val_out,
    ndview<Index, 1>& ind_out,
    Index elem_count,
    std::uint32_t bit_offset,
    std::int64_t local_size,
    std::int64_t local_hist_count,
    sycl::event& deps) {
    ONEDAL_ASSERT(part_prefix_hist.get_count() == ((local_hist_count + 1) << radix_bits_));
    ONEDAL_ASSERT(val_in.get_count() == ind_in.get_count());
    ONEDAL_ASSERT(val_in.get_count() == val_out.get_count());
    ONEDAL_ASSERT(val_in.get_count() == ind_out.get_count());

    const radix_integer_t* val_in_ptr = reinterpret_cast<const radix_integer_t*>(val_in.get_data());
    const Index* ind_in_ptr = ind_in.get_data();
    const Index* part_prefix_hist_ptr = part_prefix_hist.get_data();
    radix_integer_t* val_out_ptr = reinterpret_cast<radix_integer_t*>(val_out.get_mutable_data());
    Index* ind_out_ptr = ind_out.get_mutable_data();

    const sycl::nd_range<1> nd_range =
        make_multiple_nd_range_1d(de::check_mul_overflow(local_size, local_hist_count), local_size);

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }

            const std::uint32_t n_groups = item.get_group_range(0);
            const std::uint32_t n_sub_groups = sbg.get_group_range()[0];
            const std::uint32_t n_total_sub_groups = n_sub_groups * n_groups;
            const Index elems_for_sbg =
                elem_count / n_total_sub_groups + bool(elem_count % n_total_sub_groups);
            const std::uint32_t local_size = sbg.get_local_range()[0];

            const std::uint32_t local_id = sbg.get_local_id();
            const std::uint32_t sub_group_id = sbg.get_group_id();
            const std::uint32_t group_id = item.get_group(0) * n_sub_groups + sub_group_id;

            Index ind_start = group_id * elems_for_sbg;
            Index ind_end =
                sycl::min(static_cast<Index>((group_id + 1) * elems_for_sbg), elem_count);

            Index offset[radix_range_];

            for (std::uint32_t i = 0; i < radix_range_; i++) {
                offset[i] = part_prefix_hist_ptr[group_id * radix_range_ + i] +
                            part_prefix_hist_ptr[n_total_sub_groups * radix_range_ + i];
            }

            for (Index i = ind_start + local_id; i < ind_end; i += local_size) {
                radix_integer_t data_value = val_in_ptr[i];
                radix_integer_t data_bits = ((inv_bits(data_value) >> bit_offset) & radix_range_1_);
                Index pos_new = 0;
                for (std::uint32_t j = 0; j < radix_range_; j++) {
                    Index value = static_cast<Index>(data_bits == j);
                    Index boundary = sycl::exclusive_scan_over_group(sbg, value, plus<Index>());
                    pos_new |= value * (offset[j] + boundary);
                    Index partial_offset = sycl::reduce_over_group(sbg, value, plus<Index>());
                    offset[j] = offset[j] + partial_offset;
                }
                val_out_ptr[pos_new] = data_value;
                ind_out_ptr[pos_new] = ind_in_ptr[i];
            }
        });
    });

    return event;
}

template <typename Float, typename Index>
radix_sort_indices_inplace<Float, Index>::radix_sort_indices_inplace(const sycl::queue& queue)
        : queue_(queue),
          elem_count_(0) {}

template <typename Float, typename Index>
radix_sort_indices_inplace<Float, Index>::~radix_sort_indices_inplace() {
    sort_event_.wait_and_throw();
}

template <typename Float, typename Index>
void radix_sort_indices_inplace<Float, Index>::init(sycl::queue& queue, std::int64_t elem_count) {
    ONEDAL_ASSERT(elem_count > 0);
    ONEDAL_ASSERT(elem_count <= de::limits<std::uint32_t>::max());

    const std::uint32_t uint_elem_count = de::integral_cast<std::uint32_t>(elem_count);
    if (elem_count_ != uint_elem_count) {
        elem_count_ = uint_elem_count;
        local_size_ = preferable_sbg_size_;
        local_hist_count_ = de::check_mul_overflow(max_local_hist_count_, local_size_) < elem_count_
                                ? max_local_hist_count_
                                : (elem_count_ / local_size_) + bool(elem_count_ % local_size_);

        hist_buff_size_ = (local_hist_count_ + 1) << radix_bits_;

        part_hist_ = ndarray<Index, 1>::empty(queue, { hist_buff_size_ }, sycl::usm::alloc::device);
        part_prefix_hist_ =
            ndarray<Index, 1>::empty(queue, { hist_buff_size_ }, sycl::usm::alloc::device);
        val_buff_ = ndarray<Float, 1>::empty(queue_, { elem_count_ }, sycl::usm::alloc::device);

        ind_buff_ = ndarray<Index, 1>::empty(queue_, { elem_count_ }, sycl::usm::alloc::device);
    }
}

template <typename Float, typename Index>
sycl::event radix_sort_indices_inplace<Float, Index>::operator()(ndview<Float, 1>& val_in,
                                                                 ndview<Index, 1>& ind_in,
                                                                 const event_vector& deps) {
    ONEDAL_PROFILER_TASK(sort.radix_sort_indices_inplace, queue_);
    ONEDAL_ASSERT(val_in.has_mutable_data());
    ONEDAL_ASSERT(ind_in.has_mutable_data());
    ONEDAL_ASSERT(val_in.get_count() == ind_in.get_count());

    if (val_in.get_count() > de::limits<std::uint32_t>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_number_of_elements_to_sort());
    }

    sycl::event::wait_and_throw(deps);
    sort_event_.wait_and_throw();

    init(queue_, val_in.get_count());

    std::uint32_t rev = 1;

    sycl::event res_deps = {};
    for (std::uint32_t bit_offset = 0; bit_offset < byte_range_ * sizeof(Float);
         bit_offset += radix_bits_, rev ^= 1) {
        if (rev) {
            auto scan_deps = radix_scan(queue_,
                                        val_in,
                                        part_hist_,
                                        elem_count_,
                                        bit_offset,
                                        local_size_,
                                        local_hist_count_,
                                        res_deps);
            auto hist_scan_deps = radix_hist_scan(queue_,
                                                  part_hist_,
                                                  part_prefix_hist_,
                                                  local_size_,
                                                  local_hist_count_,
                                                  scan_deps);
            res_deps = radix_reorder(queue_,
                                     val_in,
                                     ind_in,
                                     part_prefix_hist_,
                                     val_buff_,
                                     ind_buff_,
                                     elem_count_,
                                     bit_offset,
                                     local_size_,
                                     local_hist_count_,
                                     hist_scan_deps);
        }
        else {
            auto scan_deps = radix_scan(queue_,
                                        val_buff_,
                                        part_hist_,
                                        elem_count_,
                                        bit_offset,
                                        local_size_,
                                        local_hist_count_,
                                        res_deps);
            auto hist_scan_deps = radix_hist_scan(queue_,
                                                  part_hist_,
                                                  part_prefix_hist_,
                                                  local_size_,
                                                  local_hist_count_,
                                                  scan_deps);
            res_deps = radix_reorder(queue_,
                                     val_buff_,
                                     ind_buff_,
                                     part_prefix_hist_,
                                     val_in,
                                     ind_in,
                                     elem_count_,
                                     bit_offset,
                                     local_size_,
                                     local_hist_count_,
                                     hist_scan_deps);
        }
    }

    sort_event_ = res_deps;
    return res_deps;
}

template <typename Integer>
radix_sort<Integer>::radix_sort(const sycl::queue& queue) : queue_(queue),
                                                            vector_count_(0) {}

template <typename Integer>
radix_sort<Integer>::~radix_sort() {
    sort_event_.wait_and_throw();
}

template <typename Integer>
void radix_sort<Integer>::init(sycl::queue& queue, std::int64_t vector_count) {
    ONEDAL_ASSERT(vector_count > 0);
    ONEDAL_ASSERT(vector_count <= de::limits<std::uint32_t>::max());

    const std::uint32_t uint_vector_count = de::integral_cast<std::uint32_t>(vector_count);
    if (vector_count_ != uint_vector_count) {
        vector_count_ = uint_vector_count;

        buffer_ = ndarray<Integer, 2>::empty(queue_,
                                             { vector_count_, radix_range_ },
                                             sycl::usm::alloc::device);
    }
}

template <typename Integer>
sycl::event radix_sort<Integer>::operator()(ndview<Integer, 2>& val_in,
                                            ndview<Integer, 2>& val_out,
                                            std::int64_t sorted_elem_count,
                                            const event_vector& deps) {
    ONEDAL_PROFILER_TASK(sort.radix_sort, queue_);
    // radixBuf should be big enough to accumulate radix_range elements
    ONEDAL_ASSERT(val_in.get_dimension(1) > 0);
    ONEDAL_ASSERT(sorted_elem_count > 0);
    ONEDAL_ASSERT(val_in.get_dimension(0) == val_out.get_dimension(0));
    ONEDAL_ASSERT(val_in.get_dimension(1) == val_out.get_dimension(1));
    ONEDAL_ASSERT(val_out.has_mutable_data());

    if (val_in.get_dimension(0) > de::limits<std::uint32_t>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_rows());
    }
    if (val_in.get_dimension(1) > de::limits<std::uint32_t>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_columns());
    }
    if (sorted_elem_count > de::limits<std::uint32_t>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_number_of_elements_to_sort());
    }

    sort_event_.wait_and_throw();

    init(queue_, val_in.get_dimension(0));

    const std::uint32_t vector_count = de::integral_cast<std::uint32_t>(val_in.get_dimension(0));
    const std::uint32_t vector_offset = de::integral_cast<std::uint32_t>(val_in.get_dimension(1));

    const std::uint32_t _sorted_elem_count = de::integral_cast<std::uint32_t>(sorted_elem_count);

    Integer* labels = val_in.get_mutable_data();
    Integer* sorted = val_out.get_mutable_data();
    Integer* radixbuf = buffer_.get_mutable_data();

    const sycl::nd_range<2> nd_range =
        make_multiple_nd_range_2d({ vector_count, preferable_wg_size_ },
                                  { 1, preferable_wg_size_ });

    sort_event_ = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();
            // Code is written for a single subgroup. It's necessary to adjust the local range if idle subgoups are presented
            if (sbg.get_group_id()[0] > 0) {
                return;
            }

            const std::uint32_t global_id = item.get_global_id()[0];
            const std::uint32_t local_id = item.get_local_id()[1];

            const std::uint32_t local_size = sbg.get_local_range()[0];
            const std::uint32_t group_aligned_size =
                _sorted_elem_count - _sorted_elem_count % local_size;
            const std::uint32_t rem = _sorted_elem_count - group_aligned_size;

            Integer* input = &labels[global_id * vector_offset];
            Integer* output = &sorted[global_id * vector_offset];
            Integer* counters = &radixbuf[global_id * radix_range_];
            //  Radix sort
            for (std::uint32_t i = 0; i < radix_count_; i++) {
                std::uint8_t* cinput = reinterpret_cast<std::uint8_t*>(input);
                for (std::uint32_t j = local_id; j < radix_range_; j += local_size)
                    counters[j] = 0;
                //  Count elements in sub group to write once per value
                for (std::uint32_t j = local_id; j < group_aligned_size + local_size;
                     j += local_size) {
                    bool exists = j < group_aligned_size || local_id < rem;
                    std::uint8_t c = exists ? cinput[j * radix_count_ + i] : 0;
                    std::uint32_t entry = 0;
                    bool entry_found = false;
                    for (std::uint32_t k = 0; k < local_size; k++) {
                        bool correct = j < group_aligned_size || k < rem;
                        std::uint32_t done = sycl::group_broadcast(sbg, correct ? 0 : 1, k);
                        if (done)
                            break;
                        std::uint8_t value = sycl::group_broadcast(sbg, c, k);
                        if (!entry_found && value == c) {
                            entry = k;
                            entry_found = true;
                        }
                        Integer count = sycl::reduce_over_group(
                            sbg,
                            static_cast<Integer>(exists && value == c ? 1 : 0),
                            plus<Integer>());
                        if (entry_found && entry == local_id && entry == k) {
                            counters[value] += count;
                        }
                    }
                    sycl::group_barrier(sbg);
                }
                //  Parallel scan on counters to generate offsets in place
                Integer offset = 0;
                for (std::uint32_t j = local_id; j < radix_range_; j += local_size) {
                    Integer value = counters[j];
                    Integer boundary = sycl::exclusive_scan_over_group(sbg, value, plus<Integer>());
                    counters[j] = offset + boundary;
                    Integer partial_offset = sycl::reduce_over_group(sbg, value, plus<Integer>());
                    offset += partial_offset;
                }

                sycl::group_barrier(sbg);
                for (std::uint32_t j = local_id; j < group_aligned_size + local_size;
                     j += local_size) {
                    bool exists = j < group_aligned_size || local_id < rem;
                    std::uint8_t c = exists ? cinput[j * radix_count_ + i] : 0;
                    Integer local_offset = 0;
                    std::uint32_t entry = 0;
                    bool entry_found = false;

                    for (std::uint32_t k = 0; k < local_size; k++) {
                        bool correct = j < group_aligned_size || k < rem;
                        std::uint32_t done = sycl::group_broadcast(sbg, correct ? 0 : 1, k);
                        if (done)
                            break;
                        std::uint32_t skip = sycl::group_broadcast(sbg, entry_found ? 1 : 0, k);
                        if (skip)
                            continue;
                        std::uint8_t value = sycl::group_broadcast(sbg, c, k);
                        if (!entry_found && value == c) {
                            entry = k;
                            entry_found = true;
                        }
                        Integer offset = sycl::exclusive_scan_over_group(
                            sbg,
                            static_cast<Integer>(exists && value == c ? 1 : 0),
                            plus<Integer>());
                        if (value == c) {
                            local_offset = offset + counters[value];
                        }
                        Integer count = sycl::reduce_over_group(
                            sbg,
                            static_cast<Integer>(exists && value == c ? 1 : 0),
                            plus<Integer>());
                        if (entry_found && entry == local_id && entry == k) {
                            counters[value] += count;
                        }
                    }
                    sycl::group_barrier(sbg);
                    if (exists)
                        output[local_offset] = input[j];
                }
                std::swap(input, output);
            }
            for (std::uint32_t i = local_id; i < _sorted_elem_count; i += local_size)
                output[i] = input[i];
        });
    });

    return sort_event_;
}

template <typename Float, typename Index>
sycl::event radix_sort_indices_inplace_dpl(sycl::queue& queue,
                                           ndview<Float, 1>& val_in,
                                           ndview<Index, 1>& ind_in,
                                           const event_vector& deps) {
    ONEDAL_PROFILER_TASK(sort.radix_sort_indices_inplace, queue);
    ONEDAL_ASSERT(val_in.has_mutable_data());
    ONEDAL_ASSERT(ind_in.has_mutable_data());
    ONEDAL_ASSERT(val_in.get_count() == ind_in.get_count());

    if (val_in.get_count() > de::limits<std::uint32_t>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_number_of_elements_to_sort());
    }

    auto event = oneapi::dpl::experimental::kt::gpu::esimd::radix_sort_by_key<true, 8>(
        queue,
        val_in.get_mutable_data(),
        val_in.get_mutable_data() + val_in.get_count(),
        ind_in.get_mutable_data(),
        // Parameters have been chosen based on the oneDPL example for radix sort by key.
        // Reference: https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-7/radix-sort-by-key.html
        // These parameters ensure optimal performance for the given data type and distribution.
        dpl::experimental::kt::kernel_param<96, 64>{});
    return event;
}

template <typename Integer>
sycl::event radix_sort_dpl(sycl::queue& queue,
                           ndview<Integer, 2>& val_in,
                           ndview<Integer, 2>& val_out,
                           std::int64_t sorted_elem_count,
                           const event_vector& deps) {
    ONEDAL_PROFILER_TASK(sort.radix_sort, queue);

    const auto row_count = val_in.get_dimension(0);
    const auto col_count = val_in.get_dimension(1);
    sycl::event radix_sort_event;

    for (std::int64_t row = 0; row < row_count; ++row) {
        Integer* row_start_in = val_in.get_mutable_data() + row * col_count;
        Integer* row_start_out = val_out.get_mutable_data() + row * col_count;

        const auto row_sorted_elem_count = std::min(sorted_elem_count, col_count);

        radix_sort_event = oneapi::dpl::experimental::kt::gpu::esimd::radix_sort<true, 8>(
            queue,
            row_start_in,
            row_start_in + row_sorted_elem_count,
            row_start_out,
            // Parameters have been chosen based on the oneDPL example for radix sort by key.
            // Reference: https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-7/radix-sort-by-key.html
            // These parameters ensure optimal performance for the given data type and distribution.
            dpl::experimental::kt::kernel_param<96, 64>{});
    }

    return radix_sort_event;
}

template <typename Integer>
sycl::event radix_sort_dpl(sycl::queue& queue,
                           ndview<Integer, 2>& val_in,
                           ndview<Integer, 2>& val_out,
                           const event_vector& deps) {
    ONEDAL_PROFILER_TASK(sort.radix_sort, queue);
    return radix_sort_dpl(queue, val_in, val_out, val_in.get_dimension(1), deps);
}

template <typename Integer>
sycl::event radix_sort<Integer>::operator()(ndview<Integer, 2>& val_in,
                                            ndview<Integer, 2>& val_out,
                                            const event_vector& deps) {
    ONEDAL_PROFILER_TASK(sort.radix_sort, queue_);
    return this->operator()(val_in, val_out, val_in.get_dimension(1), deps);
}

#define INSTANTIATE_SORT_INDICES(F, I) \
    template class ONEDAL_EXPORT radix_sort_indices_inplace<F, I>;

#define INSTANTIATE_SORT(I) template class ONEDAL_EXPORT radix_sort<I>;

INSTANTIATE_SORT_INDICES(float, std::uint32_t)
INSTANTIATE_SORT_INDICES(double, std::uint32_t)
INSTANTIATE_SORT_INDICES(float, std::int32_t)
INSTANTIATE_SORT_INDICES(double, std::int32_t)

INSTANTIATE_SORT(std::int32_t)
INSTANTIATE_SORT(std::uint32_t)
INSTANTIATE_SORT(std::int64_t)
INSTANTIATE_SORT(std::uint64_t)

#define INSTANTIATE_SORT_INDICES_DPL(Float, Index)                                   \
    template ONEDAL_EXPORT sycl::event radix_sort_indices_inplace_dpl<Float, Index>( \
        sycl::queue&,                                                                \
        ndview<Float, 1>&,                                                           \
        ndview<Index, 1>&,                                                           \
        const event_vector&);

#define INSTANTIATE_FLOAT(Index)               \
    INSTANTIATE_SORT_INDICES_DPL(float, Index) \
    INSTANTIATE_SORT_INDICES_DPL(double, Index)

INSTANTIATE_FLOAT(std::uint32_t)
INSTANTIATE_FLOAT(std::int32_t)

#define INSTANTIATE_RADIX_SORT(Index)                                           \
    template ONEDAL_EXPORT sycl::event radix_sort_dpl<Index>(sycl::queue&,      \
                                                             ndview<Index, 2>&, \
                                                             ndview<Index, 2>&, \
                                                             const event_vector&);

INSTANTIATE_RADIX_SORT(std::int32_t)
INSTANTIATE_RADIX_SORT(std::uint32_t)
INSTANTIATE_RADIX_SORT(std::int64_t)
INSTANTIATE_RADIX_SORT(std::uint64_t)

#define INSTANTIATE_RADIX_SORT_WITH_COUNT(Index)                                \
    template ONEDAL_EXPORT sycl::event radix_sort_dpl<Index>(sycl::queue&,      \
                                                             ndview<Index, 2>&, \
                                                             ndview<Index, 2>&, \
                                                             std::int64_t,      \
                                                             const event_vector&);

INSTANTIATE_RADIX_SORT_WITH_COUNT(std::int32_t)
INSTANTIATE_RADIX_SORT_WITH_COUNT(std::uint32_t)
INSTANTIATE_RADIX_SORT_WITH_COUNT(std::int64_t)
INSTANTIATE_RADIX_SORT_WITH_COUNT(std::uint64_t)

} // namespace oneapi::dal::backend::primitives
