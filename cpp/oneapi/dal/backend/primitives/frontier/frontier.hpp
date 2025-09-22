/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#pragma once

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/frontier/bitset.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

/// @brief A view of a frontier that provides an interface for the kernel to interact with the frontier data structure.
template <typename ElementType>
class frontier_view {
public:
    using bitmap_t = ElementType;
    static constexpr std::size_t divide_factor = bitset<bitmap_t>::element_bitsize;

    frontier_view() = default;
    frontier_view(bitmap_t* data_layer,
                  bitmap_t* mlb_layer,
                  std::uint32_t* offsets,
                  std::uint32_t* offsets_size,
                  size_t num_items)
            : _num_items(num_items),
              _data_layer(bitset<bitmap_t>{ data_layer }),
              _mlb_layer(bitset<bitmap_t>{ mlb_layer }),
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

private:
    size_t _num_items;

    bitset<bitmap_t> _data_layer;
    bitset<bitmap_t> _mlb_layer;

    std::uint32_t* _offsets;
    std::uint32_t* _offsets_size;
};

/// @brief The Two-Layer bitmap frontier class
template <typename ElementType>
class frontier {
    using buffer_t = std::uint32_t;

public:
    frontier(sycl::queue& queue,
             std::size_t num_items,
             sycl::usm::alloc alloc = sycl::usm::alloc::shared);

    const frontier_view<ElementType> get_device_view() const {
        auto offsets_size_pointer = _offsets.get_mutable_data();
        auto offsets_pointer = _offsets.get_mutable_data() + 1;

        return { _data_layer.get_mutable_data(),
                 _mlb_layer.get_mutable_data(),
                 offsets_pointer,
                 offsets_size_pointer,
                 static_cast<std::size_t>(_data_layer.get_count()) };
    }

    inline ndview<ElementType, 1> get_data() const {
        return _data_layer;
    }

    inline ndview<ElementType, 1> get_mlb() const {
        return _mlb_layer;
    }

    inline ndview<std::uint32_t, 1> get_offsets() const {
        return _offsets.slice(1, _offsets.get_count());
    }

    inline ndview<std::uint32_t, 1> get_offsets_size() const {
        return _offsets.slice(0, 1);
    }

    bool empty();

    void insert(ElementType idx);

    bool check(ElementType idx);

    void clear();

    /// @brief Computes the active frontier by filling the offsets array with the indices of the active vertices.
    /// It returns a sycl::event that can be used to synchronize the execution.
    /// The offsets array is filled with the indices of the active vertices in the first layer of the frontier.
    /// This is used by the primitives to understand which vertices are active in the frontier.
    sycl::event compute_active_frontier();

    static void swap(frontier<ElementType>& f1, frontier<ElementType>& f2) {
        using std::swap;
        swap(f1._data_layer, f2._data_layer);
        swap(f1._mlb_layer, f2._mlb_layer);
        swap(f1._offsets, f2._offsets);
        swap(f1._queue, f2._queue);
        swap(f1._num_items, f2._num_items);
    }

private:
    sycl::queue& _queue;
    size_t _num_items;

    ndarray<ElementType, 1> _data_layer;
    ndarray<ElementType, 1> _mlb_layer;
    ndarray<std::uint32_t, 1> _offsets;
    ndarray<buffer_t, 1> _buffer;
    const size_t _TMP_VAR = 0;
    const size_t _CAF_FLAG = 1; // Compute Active Frontier Flag (1 if already computed, 0 otherwise)
};

template <typename ElementType>
void swap_frontiers(frontier<ElementType>& f1, frontier<ElementType>& f2) {
    frontier<ElementType>::swap(f1, f2);
}

} // namespace oneapi::dal::backend::primitives
