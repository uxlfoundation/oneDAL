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

public:
    frontier(sycl::queue& queue, std::size_t num_items) : _queue(queue) {
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
            { static_cast<std::int64_t>(mlb_size + 1) },
            sycl::usm::alloc::shared); /// First offset is to keep the size of the frontier
        _buffer = ndarray<std::uint32_t, 1>::empty(queue, { static_cast<std::int64_t>(10) }, Alloc);

        e1.wait_and_throw();
        e2.wait_and_throw();
        e3.wait_and_throw();

        _data_layer = std::move(data_layer);
        _mlb_layer = std::move(mlb_layer);
        _offsets = std::move(offsets);
    }

    frontier_view<bitmap_t> get_device_view() {
        return frontier_view<bitmap_t>(this->get_data_layer(),
                                       this->get_mlb_layer(),
                                       this->get_offsets(),
                                       this->get_offsets_size(),
                                       _data_layer.get_count());
    }

    bitmap_t* get_data_layer() const {
        return _data_layer.get_mutable_data();
    }

    size_t get_data_layer_size() const {
        return _data_layer.get_count();
    }

    size_t get_mlb_layer_size() const {
        return _mlb_layer.get_count();
    }

    bitmap_t* get_mlb_layer() const {
        return _mlb_layer.get_mutable_data();
    }

    std::uint32_t* get_offsets() const {
        return _offsets.get_mutable_data() + 1;
    }

    std::uint32_t* get_offsets_size() const {
        return _offsets.get_mutable_data();
    }

    bool empty() const {
        auto empty_buff = _buffer.slice(0, 1);
        fill(_queue, empty_buff, bitmap_t(0)).wait();
        auto empty_buff_ptr = empty_buff.get_mutable_data();

        auto e = _queue.submit([&](sycl::handler& cgh) {
            const auto range = make_range_1d(_mlb_layer.get_count());
            auto sum_reduction = sycl::reduction(empty_buff_ptr, sycl::plus<>());
            auto* f_ptr = this->get_mlb_layer();

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
            .wait();
    }

private:
    sycl::queue& _queue;

    ndarray<bitmap_t, 1> _data_layer;
    ndarray<bitmap_t, 1> _mlb_layer;
    ndarray<std::uint32_t, 1> _offsets;
    ndarray<std::uint32_t, 1> _buffer;
};

} // namespace oneapi::dal::backend::primitives
