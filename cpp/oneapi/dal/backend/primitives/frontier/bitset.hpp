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

#pragma once

#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend::primitives {

/**
 * @brief A bitset class that provides a set of operations on a bitset.
 */
template <typename ElementType = std::uint32_t>
class bitset {
public:
    using element_t = ElementType;
    static constexpr size_t element_bitsize = sizeof(element_t) * 8; // Number of bits in an element

    bitset(element_t* data) : _data(data) {}

    /// Sets the bit at the specified index to 1.
    inline void set(std::uint32_t index) const {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        _data[element_index] |= (element_t(1) << bit_index);
    }

    /// Clears the bit at the specified index to 0.
    inline void unset(std::uint32_t index) const {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        _data[element_index] &= ~(element_t(1) << bit_index);
    }

    /// Checks if the bit at the specified index is set.
    inline bool test(std::uint32_t index) const {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        return (_data[element_index] & (element_t(1) << bit_index)) != 0;
    }

    /// Sets the bit at the specified index using atomic operations.
    template <sycl::memory_order mem_order = sycl::memory_order::relaxed,
              sycl::memory_scope mem_scope = sycl::memory_scope::device>
    inline void atomic_set(std::uint32_t index) const {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        sycl::atomic_ref<element_t, mem_order, mem_scope> atomic_element(_data[element_index]);
        atomic_element |= (element_t(1) << bit_index);
    }

    /// Clears the bit at the specified index using atomic operations.
    template <sycl::memory_order mem_order = sycl::memory_order::relaxed,
              sycl::memory_scope mem_scope = sycl::memory_scope::device>
    inline void atomic_unset(std::uint32_t index) {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        sycl::atomic_ref<element_t, mem_order, mem_scope> atomic_element(_data[element_index]);
        atomic_element &= ~(element_t(1) << bit_index);
    }

    /// Checks if the bit at the specified index is set using atomic operations.
    template <sycl::memory_order mem_order = sycl::memory_order::relaxed,
              sycl::memory_scope mem_scope = sycl::memory_scope::device>
    inline bool atomic_test(std::uint32_t index) const {
        element_t element_index = index / element_bitsize;
        element_t bit_index = index % element_bitsize;
        sycl::atomic_ref<element_t, mem_order, mem_scope> atomic_element(_data[element_index]);
        return (atomic_element.load() & (element_t(1) << bit_index)) != 0;
    }

    /// Returns the pointer to the underlying data.
    inline element_t* get_data() {
        return _data;
    }

    inline void set_data(element_t* data) {
        _data = data;
    }

    /// Returns the pointer to the underlying data (const version).
    inline const element_t* get_data() const {
        return _data;
    }

    /// override operator []
    inline element_t& operator[](std::size_t index) {
        return _data[index];
    }

private:
    element_t* _data;
};

} // namespace oneapi::dal::backend::primitives
