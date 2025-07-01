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
        if (_mlb_layer.atomic_gas(static_cast<std::uint32_t>(idx / divide_factor))) {
            _data_layer.atomic_set(idx);
        }
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

template <typename ElementType = std::uint32_t>
class frontier {
    using bitmap_t = ElementType;

public:
    frontier(sycl::queue& queue, std::size_t num_items) : _queue(queue) {
        auto array_size =
            (num_items + bitset<bitmap_t>::element_bitsize - 1) / bitset<bitmap_t>::element_bitsize;
        auto mlb_size = (array_size + bitset<bitmap_t>::element_bitsize - 1) /
                        bitset<bitmap_t>::element_bitsize;
        _data_layer = ndarray<bitmap_t, 1>::empty(queue,
                                                  { static_cast<std::int64_t>(array_size) },
                                                  sycl::usm::alloc::shared);
        _mlb_layer = ndarray<bitmap_t, 1>::empty(queue,
                                                 { static_cast<std::int64_t>(mlb_size) },
                                                 sycl::usm::alloc::shared);
        _offsets = ndarray<std::uint32_t, 1>::empty(
            queue,
            { static_cast<std::int64_t>(mlb_size + 1) },
            sycl::usm::alloc::shared); /// First offset is to keep the size of the frontier
    }

    frontier_view<bitmap_t> get_device_view() {
        return frontier_view<bitmap_t>(this->get_data_layer(),
                                       this->get_mlb_layer(),
                                       this->get_offsets(),
                                       this->get_offsets_size(),
                                       _data_layer.get_count());
    }

    bitmap_t* get_data_layer() {
        return _data_layer.get_mutable_data();
    }

    bitmap_t* get_mlb_layer() {
        return _mlb_layer.get_mutable_data();
    }

    std::uint32_t* get_offsets() {
        return _offsets.get_mutable_data() + 1;
    }

    std::uint32_t* get_offsets_size() {
        return _offsets.get_mutable_data();
    }

private:
    sycl::queue& _queue;

    ndarray<bitmap_t, 1> _data_layer;
    ndarray<bitmap_t, 1> _mlb_layer;
    ndarray<std::uint32_t, 1> _offsets;
};

} // namespace oneapi::dal::backend::primitives
