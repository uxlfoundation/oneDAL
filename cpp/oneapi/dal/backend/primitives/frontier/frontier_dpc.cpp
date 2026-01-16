/*******************************************************************************
* Copyright contributors to the oneDAL project
* Copyright 2025 University of Salerno
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

#include "oneapi/dal/backend/primitives/frontier/frontier.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename ElementType>
frontier<ElementType>::frontier(sycl::queue& queue, std::uint64_t num_items, sycl::usm::alloc alloc)
        : _queue(queue),
          _num_items(num_items) {
    std::int64_t array_size = (num_items + bitset<ElementType>::element_bitsize - 1) /
                              bitset<ElementType>::element_bitsize;
    std::int64_t mlb_size = (array_size + bitset<ElementType>::element_bitsize - 1) /
                            bitset<ElementType>::element_bitsize;
    _data_layer = ndarray<ElementType, 1>::empty(_queue, { array_size }, alloc);
    _mlb_layer = ndarray<ElementType, 1>::empty(_queue, { mlb_size }, alloc);
    _offsets = ndarray<std::uint32_t, 1>::empty(
        _queue,
        { array_size + 1 },
        alloc); /// First offset is to keep the size of the frontier
    _buffer = ndarray<std::uint32_t, 1>::empty(_queue, { 10 }, alloc);

    sycl::event e1, e2, e3;
    e1 = _data_layer.fill(_queue, ElementType(0));
    e2 = _mlb_layer.fill(_queue, ElementType(0));
    e3 = _offsets.fill(_queue, std::uint32_t(0));

    e1.wait_and_throw();
    e2.wait_and_throw();
    e3.wait_and_throw();
}

template <typename ElementType>
bool frontier<ElementType>::empty() {
    ndview<std::uint32_t, 1> empty_buff = _buffer.slice(0, 1);
    auto copy_e = fill(_queue, empty_buff, buffer_t(0));
    auto* const empty_buff_ptr = empty_buff.get_mutable_data();

    auto e = _queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(copy_e);
        const auto range = make_range_1d(_mlb_layer.get_count());
        auto sum_reduction = sycl::reduction(empty_buff_ptr, sycl::plus<>());
        auto* const f_ptr = _mlb_layer.get_mutable_data();

        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum_v) {
            sum_v += f_ptr[idx];
        });
    });
    auto empty_sum = empty_buff.at_device(_queue, 0, { e });
    return empty_sum == 0;
}

template <typename ElementType>
inline void frontier<ElementType>::insert(ElementType idx) {
    auto view = this->get_device_view();
    _queue
        .submit([&](sycl::handler& cgh) {
            cgh.single_task([=]() {
                view.insert(idx);
            });
        })
        .wait_and_throw();
}

template <typename ElementType>
inline bool frontier<ElementType>::check(ElementType idx) {
    ElementType tmp_data =
        _data_layer.at_device(_queue, idx / bitset<ElementType>::element_bitsize);
    return (tmp_data &
            (static_cast<ElementType>(1) << (idx % bitset<ElementType>::element_bitsize))) != 0;
}

template <typename ElementType>
inline void frontier<ElementType>::clear() {
    auto e = _data_layer.fill(_queue, ElementType(0));
    auto e1 = _mlb_layer.fill(_queue, ElementType(0));
    auto e2 = _offsets.fill(_queue, ElementType(0));
    e.wait_and_throw();
    e1.wait_and_throw();
    e2.wait_and_throw();
}

template <typename ElementType>
inline sycl::event frontier<ElementType>::compute_active_frontier() {
    auto bitmap = this->get_device_view();

    auto offsets_size_pointer = _offsets.get_mutable_data();
    auto offsets_pointer = _offsets.get_mutable_data() + 1;

    const uint32_t element_bitsize = bitmap.get_element_bitsize();
    const std::uint64_t local_range = 256; // propose_wg_size(this->_queue);
    const std::uint64_t mlb_count = _mlb_layer.get_count();
    const std::uint64_t global_range = (mlb_count % local_range == 0)
                                           ? mlb_count
                                           : (mlb_count + local_range - (mlb_count % local_range));

    // check if local memory is enough
    bool use_local_mem =
        device_local_mem_size(this->_queue) >=
        static_cast<std::int64_t>(local_range * element_bitsize * sizeof(uint32_t));

    auto e0 = _queue.submit([&](sycl::handler& cgh) {
        cgh.single_task([=,
                         offsets_size = this->get_offsets_size().get_mutable_data(),
                         buffer_ptr = this->_buffer.get_mutable_data(),
                         CAF_FLAG = this->_CAF_FLAG]() {
            buffer_ptr[CAF_FLAG] = offsets_size[0] == 0 ? 1 : 0;
        });
    });

    auto e1 = this->_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(e0);

        if (!use_local_mem) {
            cgh.parallel_for(make_range_1d(_mlb_layer.get_count()),
                             [=,
                              offsets = offsets_pointer,
                              offsets_size = offsets_size_pointer,
                              data_layer = this->_mlb_layer.get_mutable_data(),
                              buffer = this->_buffer.get_mutable_data(),
                              CAF_FLAG = this->_CAF_FLAG](sycl::id<1> idx) {
                                 if (!buffer[CAF_FLAG])
                                     return;

                                 sycl::atomic_ref<uint32_t,
                                                  sycl::memory_order::relaxed,
                                                  sycl::memory_scope::device>
                                     offsets_size_ref{ offsets_size[0] };

                                 ElementType data = data_layer[idx];
                                 for (std::uint64_t i = 0; i < element_bitsize; i++) {
                                     if (data & (static_cast<ElementType>(1) << i)) {
                                         offsets[offsets_size_ref++] = i + idx * element_bitsize;
                                     }
                                 }
                             });
        }
        else {
            sycl::local_accessor<uint32_t, 1> local_offsets(local_range * element_bitsize, cgh);
            sycl::local_accessor<uint32_t, 1> local_size(1, cgh);

            cgh.parallel_for(
                make_multiple_nd_range_1d(global_range, local_range),
                [=,
                 offsets = offsets_pointer,
                 offsets_size = offsets_size_pointer,
                 data_layer = this->_mlb_layer.get_mutable_data(),
                 size = mlb_count,
                 buffer = this->_buffer.get_mutable_data(),
                 CAF_FLAG = this->_CAF_FLAG](sycl::nd_item<1> item) {
                    if (!buffer[CAF_FLAG])
                        return;

                    auto group = item.get_group();
                    sycl::atomic_ref<uint32_t,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group>
                        local_size_ref{ local_size[0] };
                    sycl::atomic_ref<uint32_t,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        offsets_size_ref{ offsets_size[0] };

                    if (group.leader()) {
                        local_size_ref.store(0);
                    }
                    sycl::group_barrier(group);
                    for (uint32_t gid = item.get_global_linear_id(); gid < size;
                         gid += item.get_global_range(0)) {
                        ElementType data = data_layer[gid];
                        for (std::uint64_t i = 0; i < element_bitsize; i++) {
                            if (data & (static_cast<ElementType>(1) << i)) {
                                local_offsets[local_size_ref++] = i + gid * element_bitsize;
                            }
                        }
                    }

                    sycl::group_barrier(group);

                    std::uint64_t data_offset = 0;
                    if (group.leader()) {
                        data_offset = offsets_size_ref.fetch_add(local_size_ref.load());
                    }
                    data_offset = sycl::group_broadcast(group, data_offset, 0);
                    for (std::uint64_t i = item.get_local_linear_id(); i < local_size_ref.load();
                         i += item.get_local_range(0)) {
                        offsets[data_offset + i] = local_offsets[i];
                    }
                });
        }
    });
    return e1;
}

#define INSTANTIATE(F) template class frontier<F>;

INSTANTIATE(std::uint64_t)
INSTANTIATE(std::uint32_t)

#endif

} // namespace oneapi::dal::backend::primitives
