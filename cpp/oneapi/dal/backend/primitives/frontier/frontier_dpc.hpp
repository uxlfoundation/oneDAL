#pragma once

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/graph/detail/common.hpp"
#include "oneapi/dal/graph/detail/container.hpp"
#include "oneapi/dal/backend/primitives/bitset.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

template <typename ElementType = std::uint32_t>
struct frontier_view {
    using bitmap_t = ElementType;
    static constexpr std::size_t divide_factor = bitset<bitmap_t>::element_bitsize;

    frontier_view() = default;
    frontier_view(bitmap_t* data_layer,
                  bitmap_t* mlb_layer,
                  std::uint32_t* offsets,
                  std::uint32_t* offsets_size,
                  size_t num_items)
            : _num_items(num_items),
              _data_layer(data_layer),
              _mlb_layer(mlb_layer),
              _offsets(offsets),
              _offsets_size(offsets_size) {}

    inline void insert(std::uint32_t idx) const {
        if (!_data_layer.test(idx)) {
            _data_layer.atomic_set(idx);
        }
        if (!_mlb_layer.test(idx / divide_factor)) {
            _mlb_layer.atomic_set(idx / divide_factor);
        };
    }

    inline void remove(std::uint32_t idx) const {
        throw dal::domain_error("Frontier does not support remove operation");
    }

    inline bool check(std::uint32_t idx) const {
        return _data_layer.atomic_test(idx);
    }

    size_t _num_items;

    bitset<bitmap_t> _data_layer;
    bitset<bitmap_t> _mlb_layer;

    std::uint32_t* _offsets;
    std::uint32_t* _offsets_size;
};

template <typename ElementType = std::uint32_t, sycl::usm::alloc Alloc = sycl::usm::alloc::shared>
class frontier {
    using bitmap_t = ElementType;
    using buffer_t = std::uint32_t;

public:
    frontier(sycl::queue& queue, std::size_t num_items) : _queue(queue), _num_items(num_items) {
        auto array_size =
            (num_items + bitset<bitmap_t>::element_bitsize - 1) / bitset<bitmap_t>::element_bitsize;
        auto mlb_size = (array_size + bitset<bitmap_t>::element_bitsize - 1) /
                        bitset<bitmap_t>::element_bitsize;
        auto [data_layer, e1] =
            ndarray<bitmap_t, 1>::zeros(queue, { static_cast<std::int64_t>(array_size) }, Alloc);
        auto [mlb_layer, e2] =
            ndarray<bitmap_t, 1>::zeros(queue, { static_cast<std::int64_t>(mlb_size) }, Alloc);
        auto [offsets, e3] = ndarray<std::uint32_t, 1>::zeros(
            queue,
            { static_cast<std::int64_t>(array_size + 1) },
            sycl::usm::alloc::shared); /// First offset is to keep the size of the frontier
        _buffer = ndarray<std::uint32_t, 1>::empty(queue, { static_cast<std::int64_t>(10) }, Alloc);

        e1.wait_and_throw();
        e2.wait_and_throw();
        e3.wait_and_throw();

        _data_layer = std::move(data_layer);
        _mlb_layer = std::move(mlb_layer);
        _offsets = std::move(offsets);
    }

    frontier_view<bitmap_t> get_device_view() const {
        auto offsets_size_pointer = _offsets.get_mutable_data();
        auto offsets_pointer = _offsets.get_mutable_data() + 1;

        return frontier_view<bitmap_t>(this->get_data_ptr(),
                                       this->get_mlb_ptr(),
                                       offsets_pointer,
                                       offsets_size_pointer,
                                       _data_layer.get_count());
    }

    inline bitmap_t* get_data_ptr() const {
        return _data_layer.get_mutable_data();
    }

    inline bitmap_t* get_mlb_ptr() const {
        return _mlb_layer.get_mutable_data();
    }

    inline std::uint32_t* get_offsets_ptr() const {
        return &_offsets.get_mutable_data()[1];
    }

    inline std::uint32_t* get_offsets_size_ptr() const {
        return _offsets.get_mutable_data();
    }

    inline size_t get_data_size() const {
        return _data_layer.get_count();
    }

    inline size_t get_mlb_size() const {
        return _mlb_layer.get_count();
    }

    bool empty() const {
        auto empty_buff = _buffer.slice(0, 1);
        fill(_queue, empty_buff, buffer_t(0)).wait();
        auto empty_buff_ptr = empty_buff.get_mutable_data();

        auto e = _queue.submit([&](sycl::handler& cgh) {
            const auto range = make_range_1d(_mlb_layer.get_count());
            auto sum_reduction = sycl::reduction(empty_buff_ptr, sycl::plus<>());
            auto* f_ptr = this->get_mlb_ptr();

            cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum_v) {
                sum_v += f_ptr[idx];
            });
        });
        e.wait_and_throw();
        auto empty_sum = empty_buff.at_device(_queue, 0, { e });
        return empty_sum == 0;
    }

    void insert(bitmap_t idx) {
        if (idx >= _data_layer.get_count() * bitset<bitmap_t>::element_bitsize) {
            throw dal::domain_error("Index is out of range");
        }
        auto view = this->get_device_view();
        _queue
            .single_task([=]() {
                view.insert(idx);
            })
            .wait_and_throw();
    }

    bool check(bitmap_t idx) const {
        auto check_buff = _buffer.slice(0, 1);
        fill(_queue, check_buff, buffer_t(0)).wait();
        auto check_buff_ptr = check_buff.get_mutable_data();

        auto e = _queue.submit([&](sycl::handler& cgh) {
            auto view = this->get_device_view();

            cgh.single_task([=]() {
                check_buff_ptr[0] = view.check(idx) ? buffer_t(1) : buffer_t(0);
            });
        });
        auto check_res = check_buff.at_device(_queue, 0, { e });
        return check_res == buffer_t(1);
    }

    void clear() {
        auto e = fill(_queue, _data_layer, bitmap_t(0));
        auto e1 = fill(_queue, _mlb_layer, bitmap_t(0));
        auto e2 = fill(_queue, _offsets, buffer_t(0));
        e.wait_and_throw();
        e1.wait_and_throw();
        e2.wait_and_throw();
    }

    sycl::event compute_active_frontier() const {
        auto bitmap = this->get_device_view();

        uint32_t range = decltype(bitmap._mlb_layer)::element_bitsize;
        size_t local_range = 1024; // Adjust as needed
        size_t global_range =
            _mlb_layer.get_count() + local_range - (_mlb_layer.get_count() % local_range);

        bool use_local_mem =
            device_local_mem_size(this->_queue) >= (local_range * range * sizeof(uint32_t));

        auto e = this->_queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint32_t, 1> local_offsets(local_range * range, cgh);
            sycl::local_accessor<uint32_t, 1> local_size(1, cgh);

            if (use_local_mem) {
                cgh.parallel_for(
                    make_multiple_nd_range_1d(global_range, local_range),
                    [=,
                     offsets = this->get_offsets_ptr(),
                     offsets_size = this->get_offsets_size_ptr(),
                     data_layer = this->get_mlb_ptr(),
                     size = this->_num_items](sycl::nd_item<1> item) {
                        if (offsets_size[0] > 0) {
                            return;
                        }

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
                             gid += local_range) {
                            bitmap_t data = data_layer[gid];
                            for (size_t i = 0; i < range; i++) {
                                if (data & (static_cast<bitmap_t>(1) << i)) {
                                    local_offsets[local_size_ref++] = i + gid * range;
                                }
                            }
                        }

                        sycl::group_barrier(group);

                        size_t data_offset = 0;
                        if (group.leader()) {
                            data_offset = offsets_size_ref.fetch_add(local_size_ref.load());
                        }
                        data_offset = sycl::group_broadcast(group, data_offset, 0);
                        for (size_t i = item.get_local_linear_id(); i < local_size_ref.load();
                             i += item.get_local_range(0)) {
                            offsets[data_offset + i] = local_offsets[i];
                        }
                    });
            }
            else {
                cgh.parallel_for(make_multiple_nd_range_1d(global_range, 256),
                                 [=,
                                  offsets = this->get_offsets_ptr(),
                                  offsets_size = this->get_offsets_size_ptr(),
                                  data_layer = this->get_mlb_ptr(),
                                  size = this->_num_items](sycl::nd_item<1> item) {
                                     if (offsets_size[0] > 0) {
                                         return;
                                     }

                                     sycl::atomic_ref<uint32_t,
                                                      sycl::memory_order::relaxed,
                                                      sycl::memory_scope::device>
                                         offsets_size_ref{ offsets_size[0] };
                                     auto gid = item.get_global_linear_id();
                                     if (gid >= size)
                                         return;

                                     bitmap_t data = data_layer[gid];
                                     for (size_t i = 0; i < range; i++) {
                                         if (data & (static_cast<bitmap_t>(1) << i)) {
                                             offsets[offsets_size_ref++] = i + gid * range;
                                         }
                                     }
                                 });
            }
        });
        return e;
    }

private:
    sycl::queue& _queue;
    size_t _num_items;

    ndarray<bitmap_t, 1> _data_layer;
    ndarray<bitmap_t, 1> _mlb_layer;
    ndarray<std::uint32_t, 1> _offsets;
    ndarray<buffer_t, 1> _buffer;
};

} // namespace oneapi::dal::backend::primitives
