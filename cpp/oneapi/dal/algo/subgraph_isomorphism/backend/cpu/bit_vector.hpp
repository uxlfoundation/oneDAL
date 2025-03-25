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

#pragma once

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/compiler_adapt.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/inner_alloc.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/bit_vector_popcount.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template <typename Cpu>
class bit_vector {
public:
    static constexpr std::int64_t byte(std::int64_t x) {
        ONEDAL_ASSERT(x >= 0);
        return x >> 3;
    };

    static constexpr std::uint8_t bit(std::int64_t x) {
        ONEDAL_ASSERT(x >= 0);
        return 1 << static_cast<std::uint8_t>(x & 7);
    };

    static constexpr std::int64_t bit_vector_size(std::int64_t vertex_count) {
        return -~(vertex_count >> 3); /* (vertex_count / 8) + 1 */
    };

    static constexpr std::uint8_t power_of_two(const std::uint8_t bit_val) {
        return 31 - ONEDAL_lzcnt_u32<Cpu>(bit_val);
    };

    bit_vector(const inner_alloc alloc);
    bit_vector(bit_vector<Cpu>& bvec);
    bit_vector(const std::int64_t vector_size, inner_alloc alloc);
    ~bit_vector();
    void set_bit(std::int64_t vertex);
    std::uint8_t* get_vector_pointer() const;
    std::int64_t size() const;
    std::int64_t popcount() const;

    bit_vector<Cpu>& operator&=(bit_vector<Cpu>& a);
    bit_vector<Cpu>& operator|=(bit_vector<Cpu>& a);
    bit_vector<Cpu>& operator^=(bit_vector<Cpu>& a);
    bit_vector<Cpu>& operator=(const bit_vector<Cpu>& a);
    bit_vector<Cpu>& operator~();
    bit_vector<Cpu>& operator&=(const std::uint8_t* pa);
    bit_vector<Cpu>& operator|=(const std::uint8_t* pa);
    bit_vector<Cpu>& operator^=(const std::uint8_t* pa);
    bit_vector<Cpu>& operator=(const std::uint8_t* pa);

    bit_vector(bit_vector<Cpu>&& a);

    bit_vector<Cpu>& andn(bit_vector<Cpu>& a);

    void set(const std::uint8_t byte_val = 0x0);
    static void set(const std::int64_t vector_size,
                    std::uint8_t* result_vector,
                    const std::uint8_t byte_val = 0);
    static void set(const std::int64_t vector_size,
                    std::uint8_t* result_vector,
                    const std::uint8_t* vector);
    static void set_bit(std::uint8_t* result_vector,
                        const std::int64_t vertex,
                        const std::int64_t vector_size_in_bits);
    static std::int64_t popcount(const std::int64_t vector_size, const std::uint8_t* vector);
    static bool test_bit(const std::int64_t vector_size,
                         const std::uint8_t* vector,
                         const std::int64_t vertex);

private:
    inner_alloc allocator;
    std::uint8_t* vector;
    std::int64_t n; // size in bytes
};

template <typename Cpu>
void or_equal(std::uint8_t* vec, const std::uint8_t* pa, std::int64_t size) {
    PRAGMA_IVDEP
    for (std::int64_t i = 0; i < size; i++) {
        vec[i] |= pa[i];
    }
}

template <typename Cpu>
void and_equal(std::uint8_t* vec, const std::uint8_t* pa, std::int64_t size) {
    PRAGMA_IVDEP
    for (std::int64_t i = 0; i < size; i++) {
        vec[i] &= pa[i];
    }
}

template <typename Cpu>
void or_equal(std::uint8_t* vec, const std::int64_t* bit_index, const std::int64_t list_size) {
    PRAGMA_IVDEP
    for (std::int64_t i = 0; i < list_size; i++) {
        vec[bit_vector<Cpu>::byte(bit_index[i])] |= bit_vector<Cpu>::bit(bit_index[i]);
    }
}

template <typename Cpu>
void set(std::uint8_t* vec, std::int64_t size, const std::uint8_t byte_val = 0x0) {
    PRAGMA_VECTOR_ALWAYS
    for (std::int64_t i = 0; i < size; i++) {
        vec[i] = byte_val;
    }
}

template <typename Cpu>
void and_equal(std::uint8_t* vec,
               const std::int64_t* bit_index,
               const std::int64_t bit_size,
               const std::int64_t list_size,
               std::int64_t* tmp_array = nullptr,
               const std::int64_t tmp_size = 0) {
    std::int64_t counter = 0;
    PRAGMA_IVDEP
    for (std::int64_t i = 0; i < list_size; i++) {
        tmp_array[counter] = bit_index[i];
        counter += precomputed_popcount(vec[bit_vector<Cpu>::byte(bit_index[i])] &
                                        bit_vector<Cpu>::bit(bit_index[i]));
    }

    set<Cpu>(vec, bit_size, 0x0);

    PRAGMA_IVDEP
    for (std::int64_t i = 0; i < counter; i++) {
        vec[bit_vector<Cpu>::byte(tmp_array[i])] |= bit_vector<Cpu>::bit(tmp_array[i]);
    }
}

template <typename Cpu>
void bit_vector<Cpu>::set(const std::int64_t vector_size,
                          std::uint8_t* result_vector,
                          const std::uint8_t* vector) {
    ONEDAL_ASSERT(result_vector != nullptr);
    ONEDAL_ASSERT(vector != nullptr);
    ONEDAL_ASSERT(vector_size >= 0);

    for (std::int64_t i = 0; i < vector_size; i++) {
        result_vector[i] = vector[i];
    }
}

template <typename Cpu>
void bit_vector<Cpu>::set(const std::int64_t vector_size,
                          std::uint8_t* result_vector,
                          const std::uint8_t byte_val) {
    ONEDAL_ASSERT(result_vector != nullptr);
    for (std::int64_t i = 0; i < vector_size; i++) {
        result_vector[i] = byte_val;
    }
}

template <typename Cpu>
void bit_vector<Cpu>::set(const std::uint8_t byte_val) {
    PRAGMA_VECTOR_ALWAYS
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] = byte_val;
    }
}

template <typename Cpu>
void bit_vector<Cpu>::set_bit(std::uint8_t* result_vector,
                              const std::int64_t vertex,
                              const std::int64_t vector_size_in_bits) {
    ONEDAL_ASSERT(result_vector != nullptr);
    ONEDAL_ASSERT(vertex < vector_size_in_bits);

    result_vector[byte(vertex)] |= bit(vertex);
}

template <typename Cpu>
bit_vector<Cpu>::bit_vector(const inner_alloc alloc) : allocator(alloc) {
    vector = nullptr;
    n = 0;
}

template <typename Cpu>
bit_vector<Cpu>::bit_vector(bit_vector<Cpu>& bvec)
        : allocator(bvec.allocator.get_byte_allocator()) {
    n = bvec.n;
    vector = allocator.allocate<std::uint8_t>(n);
    this->set(n, vector, bvec.vector);
}

template <typename Cpu>
bit_vector<Cpu>::bit_vector(const std::int64_t vector_size_in_bytes, inner_alloc alloc)
        : allocator(alloc),
          n(vector_size_in_bytes) {
    vector = allocator.allocate<std::uint8_t>(n);
    this->set(n, vector);
}

template <typename Cpu>
bit_vector<Cpu>::bit_vector(bit_vector<Cpu>&& a)
        : allocator(a.allocator.get_byte_allocator()),
          vector(a.vector),
          n(a.n) {
    a.n = 0;
    a.vector = nullptr;
}

template <typename Cpu>
bit_vector<Cpu>::~bit_vector() {
    if (vector != nullptr) {
        allocator.deallocate(vector, n);
    }
}

template <typename Cpu>
void bit_vector<Cpu>::set_bit(std::int64_t vertex) {
    return set_bit(vector, vertex, n * 8);
}

template <typename Cpu>
std::uint8_t* bit_vector<Cpu>::get_vector_pointer() const {
    return vector;
}

template <typename Cpu>
std::int64_t bit_vector<Cpu>::size() const {
    return n;
}

template <typename Cpu>
bit_vector<Cpu>& bit_vector<Cpu>::operator&=(bit_vector<Cpu>& a) {
    ONEDAL_ASSERT(n == a.size());
    const std::uint8_t* pa = a.get_vector_pointer();
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] &= pa[i];
    }
    return *this;
}

template <typename Cpu>
bit_vector<Cpu>& bit_vector<Cpu>::operator|=(bit_vector<Cpu>& a) {
    ONEDAL_ASSERT(n == a.size());
    const std::uint8_t* pa = a.get_vector_pointer();
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] |= pa[i];
    }
    return *this;
}

template <typename Cpu>
bit_vector<Cpu>& bit_vector<Cpu>::operator^=(bit_vector<Cpu>& a) {
    ONEDAL_ASSERT(n == a.size());
    std::uint8_t* pa = a.get_vector_pointer();
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] ^= pa[i];
    }
    return *this;
}

template <typename Cpu>
bit_vector<Cpu>& bit_vector<Cpu>::operator=(const bit_vector<Cpu>& a) {
    ONEDAL_ASSERT(n == a.size());
    const std::uint8_t* pa = a.get_vector_pointer();
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] = pa[i];
    }
    return *this;
}

template <typename Cpu>
bit_vector<Cpu>& bit_vector<Cpu>::operator~() {
    PRAGMA_IVDEP
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] = ~vector[i];
    }
    return *this;
}

template <typename Cpu>
bit_vector<Cpu>& bit_vector<Cpu>::operator&=(const std::uint8_t* pa) {
    PRAGMA_IVDEP
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] &= pa[i];
    }
    return *this;
}

template <typename Cpu>
bit_vector<Cpu>& bit_vector<Cpu>::operator|=(const std::uint8_t* pa) {
    PRAGMA_IVDEP
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] |= pa[i];
    }
    return *this;
}

template <typename Cpu>
bit_vector<Cpu>& bit_vector<Cpu>::operator^=(const std::uint8_t* pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] ^= pa[i];
    }
    return *this;
}

template <typename Cpu>
bit_vector<Cpu>& bit_vector<Cpu>::operator=(const std::uint8_t* pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] = pa[i];
    }
    return *this;
}

template <typename Cpu>
bit_vector<Cpu>& bit_vector<Cpu>::andn(bit_vector<Cpu>& a) {
    ONEDAL_ASSERT(n == a.size());
    std::uint8_t* pa = a.get_vector_pointer();
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] &= ~pa[i];
    }
    return *this;
}

template <typename Cpu>
bool bit_vector<Cpu>::test_bit(const std::int64_t vector_size,
                               const std::uint8_t* vector,
                               const std::int64_t vertex) {
    if (vertex / 8 > vector_size) {
        return false;
    }
    return vector[byte(vertex)] & bit(vertex);
}

template <typename Cpu>
std::int64_t bit_vector<Cpu>::popcount(const std::int64_t vector_size, const std::uint8_t* vector) {
    ONEDAL_ASSERT(vector != nullptr);

    std::int64_t result = 0;
    for (std::int64_t i = 0; i < vector_size; i++) {
        result += precomputed_popcount(vector[i]);
    }

    return result;
}

template <typename Cpu>
std::int64_t bit_vector<Cpu>::popcount() const {
    return popcount(n, vector);
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
