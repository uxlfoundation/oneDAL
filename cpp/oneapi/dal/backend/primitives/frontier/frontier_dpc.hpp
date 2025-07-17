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
              _data_layer(bitset<bitmap_t>{data_layer}),
              _mlb_layer(bitset<bitmap_t>{mlb_layer}),
              _offsets(offsets),
              _offsets_size(offsets_size) {}

    inline void insert(std::uint32_t idx) const {
        if (!_data_layer.test(idx)) {
            _data_layer.atomic_set(idx);
        }
        if (!_mlb_layer.test(idx / divide_factor)) {
            _mlb_layer.atomic_set(idx / divide_factor);
        }
    }

    inline void remove(std::uint32_t idx) const {
        throw dal::domain_error("Frontier does not support remove operation");
    }

    inline bool check(std::uint32_t idx) const {
        return _data_layer.atomic_test(idx);
    }

    inline size_t get_element_bitsize() const {
        return bitset<bitmap_t>::element_bitsize;
    }

    inline std::uint32_t* get_offsets() const {
        return _offsets;
    }

    inline std::uint32_t* get_offsets_size() const {
        return _offsets_size;
    }

    size_t _num_items;

    bitset<bitmap_t> _data_layer;
    bitset<bitmap_t> _mlb_layer;

    std::uint32_t* _offsets;
    std::uint32_t* _offsets_size;
};

#ifdef ONEDAL_DATA_PARALLEL

template <typename ElementType = std::uint32_t>
class frontier {
    using bitmap_t = ElementType;
    using buffer_t = std::uint32_t;

public:
    frontier(sycl::queue& queue, std::size_t num_items, sycl::usm::alloc alloc = sycl::usm::alloc::shared) : _queue(queue), _num_items(num_items) {
        std::int64_t array_size =
            (num_items + bitset<bitmap_t>::element_bitsize - 1) / bitset<bitmap_t>::element_bitsize;
        std::int64_t mlb_size = (array_size + bitset<bitmap_t>::element_bitsize - 1) /
                        bitset<bitmap_t>::element_bitsize;
        _data_layer =
            ndarray<bitmap_t, 1>::empty(_queue, { array_size }, alloc);
        _mlb_layer =
            ndarray<bitmap_t, 1>::empty(_queue, { mlb_size }, alloc);
        _offsets =
            ndarray<std::uint32_t, 1>::empty(_queue, { array_size + 1 }, alloc); /// First offset is to keep the size of the frontier
        _buffer = ndarray<std::uint32_t, 1>::empty(_queue, { 10 }, alloc);

        sycl::event e1, e2, e3;
        e1 = _data_layer.fill(_queue, bitmap_t(0));
        e2 = _mlb_layer.fill(_queue, bitmap_t(0));
        e3 = _offsets.fill(_queue, buffer_t(0));

        e1.wait_and_throw();
        e2.wait_and_throw();
        e3.wait_and_throw();
    }

    const frontier_view<bitmap_t> get_device_view() const {
        auto offsets_size_pointer = _offsets.get_mutable_data();
        auto offsets_pointer = _offsets.get_mutable_data() + 1;

        return frontier_view<bitmap_t>(_data_layer.get_mutable_data(),
                                       _mlb_layer.get_mutable_data(),
                                       offsets_pointer,
                                       offsets_size_pointer,
                                       _data_layer.get_count());
    }

    inline ndview<bitmap_t, 1> get_data() const {
        return _data_layer;
    }

    inline ndview<bitmap_t, 1> get_mlb() const {
        return _mlb_layer;
    }

    inline ndview<std::uint32_t, 1> get_offsets() const {
        return _offsets.slice(1, _offsets.get_count());
    }

    inline ndview<std::uint32_t, 1> get_offsets_size() const {
        return _offsets.slice(0, 1);
    }

    bool empty() const {
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

    void insert(bitmap_t idx) {
        auto view = this->get_device_view();
        _queue.submit(
            [&](sycl::handler& cgh) {
                cgh.single_task([=]() {
                    view.insert(idx);
                });
            }
        ).wait_and_throw();
    }

    bool check(bitmap_t idx) const {
        bitmap_t tmp_data = _data_layer.at_device(_queue, idx / bitset<bitmap_t>::element_bitsize);
        return (tmp_data & (static_cast<bitmap_t>(1) << (idx % bitset<bitmap_t>::element_bitsize))) != 0;
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

        auto offsets_size_pointer = _offsets.get_mutable_data();
        auto offsets_pointer = _offsets.get_mutable_data() + 1;

        uint32_t element_bitsize = decltype(bitmap._mlb_layer)::element_bitsize;
        size_t local_range = 1024; // Adjust as needed
        size_t global_range =
            _mlb_layer.get_count() + local_range - (_mlb_layer.get_count() % local_range);

        bool use_local_mem =
            device_local_mem_size(this->_queue) >= static_cast<std::int64_t>(local_range * element_bitsize * sizeof(uint32_t));

        auto e0 = _queue.submit([&](sycl::handler& cgh) {
            cgh.single_task([=, offsets_size = this->get_offsets_size().get_mutable_data(), buffer_ptr = this->_buffer.get_mutable_data(), CAF_FLAG = this->_CAF_FLAG]() {
                buffer_ptr[CAF_FLAG] = offsets_size[0] == 0 ? 1 : 0;
            });
        });

        auto e1 = this->_queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(e0);

            if (!use_local_mem || true) { // Force to use global memory for simplicity
                cgh.parallel_for(make_range_1d(_mlb_layer.get_count()),
                    [=,
                    offsets = offsets_pointer,
                    offsets_size = offsets_size_pointer,
                    data_layer = this->_mlb_layer.get_mutable_data(),
                    buffer = this->_buffer.get_mutable_data(),
                    CAF_FLAG = this->_CAF_FLAG](sycl::id<1> idx) {
    
                        if (!buffer[CAF_FLAG]) return;
    
                        sycl::atomic_ref<uint32_t,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device>
                            offsets_size_ref{ offsets_size[0] };
    
                        bitmap_t data = data_layer[idx];
                        for (size_t i = 0; i < element_bitsize; i++) {
                            if (data & (static_cast<bitmap_t>(1) << i)) {
                                offsets[offsets_size_ref++] = i + idx * element_bitsize;
                            }
                        }
                });
            } else {
                sycl::local_accessor<uint32_t, 1> local_offsets(local_range * element_bitsize, cgh);
                sycl::local_accessor<uint32_t, 1> local_size(1, cgh);

                cgh.parallel_for(
                    make_multiple_nd_range_1d(global_range, local_range),
                    [=,
                    offsets = offsets_pointer,
                    offsets_size = offsets_size_pointer,
                    data_layer = this->_mlb_layer.get_mutable_data(),
                    size = this->_num_items,
                    buffer = this->_buffer.get_mutable_data(),
                    CAF_FLAG = this->_CAF_FLAG](sycl::nd_item<1> item) {
                        if (!buffer[CAF_FLAG]) return;
    
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
                            for (size_t i = 0; i < element_bitsize; i++) {
                                if (data & (static_cast<bitmap_t>(1) << i)) {
                                    local_offsets[local_size_ref++] = i + gid * element_bitsize;
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

        });
        return e1;
    }

private:
    sycl::queue& _queue;
    size_t _num_items;

    ndarray<bitmap_t, 1> _data_layer;
    ndarray<bitmap_t, 1> _mlb_layer;
    ndarray<std::uint32_t, 1> _offsets;
    ndarray<buffer_t, 1> _buffer;
    const size_t _TMP_VAR = 0;
    const size_t _CAF_FLAG = 1; // Compute Active Frontier Flag (1 if already computed, 0 otherwise)
};

#endif

} // namespace oneapi::dal::backend::primitives
