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

#include "oneapi/dal/graph/detail/container.hpp"
#include "oneapi/dal/test/engine/common.hpp"

#include <vector>

namespace oneapi::dal::graph::test {
using namespace oneapi::dal::preview::detail;

std::int64_t allocated_bytes_count = 0;

template <class T>
struct CountingAllocator {
    typedef T value_type;
    typedef T* pointer;

    CountingAllocator() {}

    template <class U>
    CountingAllocator(const CountingAllocator<U>& other) {}

    template <class U>
    auto operator=(const CountingAllocator<U>& other) {
        return *this;
    }

    template <class U>
    bool operator!=(const CountingAllocator<U>& other) {
        return true;
    }

    bool operator!=(const CountingAllocator<T>& other) {
        return false;
    }

    T* allocate(const std::size_t n) {
        allocated_bytes_count += n * sizeof(T);
        if (n > static_cast<std::size_t>(-1) / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        void* const pv = malloc(n * sizeof(T));
        if (!pv) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(pv);
    }

    void deallocate(T* const p, std::size_t n) noexcept {
        allocated_bytes_count -= n * sizeof(T);
        free(p);
    }
};

std::int64_t lower_bound_pow(std::int64_t value) {
    int n = 1;
    while (n <= value) {
        n <<= 1;
    }
    return n;
}

template <typename T>
bool compare_content(const T& lhs, const T& rhs) {
    return lhs == rhs;
}

template <typename T, typename Allocator>
bool compare_content(const vector_container<T, Allocator>& lhs,
                     const vector_container<T, Allocator>& rhs) {
    bool equal = lhs.size() == rhs.size();
    for (std::int64_t i = 0; equal && i < lhs.size(); i++) {
        equal &= compare_content(lhs[i], rhs[i]);
    }
    return equal;
}

template <typename First, typename Second, typename Third>
bool compare_content(const std::tuple<First, Second, Third>& lhs,
                     const std::tuple<First, Second, Third>& rhs) {
    return compare_content(std::get<0>(lhs), std::get<0>(rhs)) &&
           compare_content(std::get<1>(lhs), std::get<1>(rhs)) &&
           compare_content(std::get<2>(lhs), std::get<2>(rhs));
}

template <typename T>
void fill_content(T& value, std::int64_t n = 0) {
    value = static_cast<T>(n);
}

template <typename First, typename Second, typename Third>
void fill_content(std::tuple<First, Second, Third>& value, std::int64_t n = 0) {
    fill_content(std::get<0>(value), n);
    fill_content(std::get<1>(value), n + 1);
    fill_content(std::get<2>(value), n + 2);
}

template <typename T, typename Allocator, template <typename, typename> class Container>
void fill_content(Container<T, Allocator>& value, std::int64_t n = 0) {
    if (value.empty()) {
        value.resize(n);
    }
    for (std::int64_t i = 0; i < dal::detail::integral_cast<std::int64_t>(value.size()); i++) {
        fill_content(value[i], i);
    }
}

template <typename T>
void check_construct_empty() {
    vector_container<T> vec;
    REQUIRE(vec.empty());
    REQUIRE(vec.get_count() == 0);
    REQUIRE(vec.size() == 0);
    REQUIRE(vec.capacity() == 1);
}

template <typename T>
void check_construct_empty_with_custom_allocator() {
    allocated_bytes_count = 0;
    {
        CountingAllocator<T> alloc;
        vector_container<T, CountingAllocator<T>> vec(alloc);
        REQUIRE(vec.empty());
        REQUIRE(vec.get_count() == 0);
        REQUIRE(vec.size() == 0);
        REQUIRE(vec.capacity() == 1);
    }
    REQUIRE(allocated_bytes_count == 0);
}

template <typename T>
void check_construct_given_size(std::int64_t n) {
    vector_container<T> vec(n);
    REQUIRE(!vec.empty());
    REQUIRE(vec.get_count() == n);
    REQUIRE(vec.size() == n);
    REQUIRE(vec.capacity() == lower_bound_pow(n));
}

template <typename T>
void check_construct_given_size_with_custom_allocator(std::int64_t n) {
    allocated_bytes_count = 0;
    {
        CountingAllocator<T> alloc;
        vector_container<T, CountingAllocator<T>> vec(n, alloc);
        REQUIRE(!vec.empty());
        REQUIRE(vec.get_count() == n);
        REQUIRE(vec.size() == n);
        REQUIRE(vec.capacity() == lower_bound_pow(n));
    }
    REQUIRE(allocated_bytes_count == 0);
}

template <typename T>
void check_construct_given_size_with_value_with_custom_allocator(std::int64_t n, const T& value) {
    CountingAllocator<T> alloc;
    vector_container<T, CountingAllocator<T>> vec(n, value, alloc);
    REQUIRE(!vec.empty());
    REQUIRE(vec.get_count() == n);
    REQUIRE(vec.size() == n);
    REQUIRE(vec.capacity() == lower_bound_pow(n));
    bool equal_content = true;
    for (std::int64_t i = 0; i < n; i++) {
        equal_content &= compare_content(vec[i], value);
    }
    REQUIRE(equal_content);
}

template <typename T>
void check_copy_constructor() {
    allocated_bytes_count = 0;
    {
        CountingAllocator<T> alloc;
        vector_container<T, CountingAllocator<T>> src(10, alloc);
        fill_content(src);
        vector_container<T, CountingAllocator<T>> dst(src);
        REQUIRE(src.size() == dst.size());
        REQUIRE(src.get_count() == dst.get_count());
        REQUIRE(src.capacity() == dst.capacity());
        bool equal_content = true;
        for (std::int64_t i = 0; i < src.size(); i++) {
            equal_content &= compare_content(src[i], dst[i]);
        }
        REQUIRE(equal_content);
    }
    REQUIRE(allocated_bytes_count == 0);
}

template <typename T>
void check_copy_assignment() {
    allocated_bytes_count = 0;
    {
        CountingAllocator<T> alloc;
        vector_container<T, CountingAllocator<T>> src(10, alloc);
        fill_content(src);
        vector_container<T, CountingAllocator<T>> dst(11, alloc);
        dst = src;
        REQUIRE(src.size() == dst.size());
        REQUIRE(src.get_count() == dst.get_count());
        REQUIRE(src.capacity() == dst.capacity());
        bool equal_content = true;
        for (std::int64_t i = 0; i < src.size(); i++) {
            equal_content &= compare_content(src[i], dst[i]);
        }
        REQUIRE(equal_content);
    }
    REQUIRE(allocated_bytes_count == 0);
}

template <typename T>
void check_begin_end_copy() {
    allocated_bytes_count = 0;
    {
        CountingAllocator<T> alloc;
        vector_container<T, CountingAllocator<T>> src(10, alloc);
        fill_content(src);
        vector_container<T, CountingAllocator<T>> dst(10, alloc);
        copy(src.begin(), src.end(), dst.begin());
        bool equal_content = true;
        for (std::int64_t i = 0; i < src.size(); i++) {
            equal_content &= compare_content(src[i], dst[i]);
        }
        REQUIRE(equal_content);
    }
    REQUIRE(allocated_bytes_count == 0);
}

template <typename T>
void check_copy() {
    allocated_bytes_count = 0;
    {
        CountingAllocator<T> alloc;
        vector_container<T, CountingAllocator<T>> src(10, alloc);
        fill_content(src);
        vector_container<T, CountingAllocator<T>> dst(11, alloc);
        copy(src, dst);
        REQUIRE(src.size() == dst.size());
        REQUIRE(src.get_count() == dst.get_count());
        REQUIRE(src.capacity() == dst.capacity());
        bool equal_content = true;
        for (std::int64_t i = 0; i < src.size(); i++) {
            equal_content &= compare_content(src[i], dst[i]);
        }
        REQUIRE(equal_content);
    }
    REQUIRE(allocated_bytes_count == 0);
}

template <typename T>
void check_fill(const T& value) {
    CountingAllocator<T> alloc;
    vector_container<T, CountingAllocator<T>> vec(10, alloc);
    fill(vec.begin(), vec.end(), value);
    bool equal_content = true;
    for (std::int64_t i = 0; i < vec.size(); i++) {
        equal_content &= compare_content(vec[i], value);
    }
    REQUIRE(equal_content);
}

TEST("can construct empty vector_container of primitives") {
    check_construct_empty<std::int32_t>();
}

TEST("can construct empty vector_container of vector_containers") {
    check_construct_empty<vector_container<std::int32_t>>();
}

TEST("can construct empty vector_container of vector_containers of vector_containers") {
    check_construct_empty<vector_container<vector_container<std::int32_t>>>();
}

TEST("can construct empty vector_container of std::containers") {
    check_construct_empty<std::vector<std::int32_t>>();
}

TEST("can construct empty vector_container of primitive type with custom allocator") {
    check_construct_empty_with_custom_allocator<std::int32_t>();
}

TEST("can construct empty vector_container of vector_container with custom allocator") {
    check_construct_empty_with_custom_allocator<
        vector_container<std::int32_t, CountingAllocator<std::int32_t>>>();
}

TEST("can construct empty vector_container of std::container with custom allocator") {
    check_construct_empty_with_custom_allocator<
        std::vector<std::int32_t, CountingAllocator<std::int32_t>>>();
}

TEST("can construct vector_container of given size of primitives") {
    check_construct_given_size<std::int32_t>(10);
}

TEST("can construct vector_container of given size of vector_containers") {
    check_construct_given_size<vector_container<std::int32_t>>(10);
}

TEST("can construct vector_container of given size of std::containers") {
    check_construct_given_size<std::vector<std::int32_t>>(10);
}

TEST("can construct vector_container of given size of primitive type with custom allocator") {
    check_construct_given_size_with_custom_allocator<std::int32_t>(10);
}

TEST(
    "can construct vector_container of given size of vector_containers type with custom allocator") {
    check_construct_given_size_with_custom_allocator<
        vector_container<std::int32_t, CountingAllocator<std::int32_t>>>(10);
}

TEST("can construct vector_container of given size of std::containers type with custom allocator") {
    check_construct_given_size_with_custom_allocator<
        std::vector<std::int32_t, CountingAllocator<std::int32_t>>>(10);
}

TEST(
    "can construct vector_container of given size with value of primitive type with custom allocator") {
    allocated_bytes_count = 0;
    {
        check_construct_given_size_with_value_with_custom_allocator<std::int32_t>(10, 1);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST(
    "can construct vector_container of given size with value of vector_containers type with custom allocator") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(10, alloc);
        fill_content(vec);
        check_construct_given_size_with_value_with_custom_allocator<
            vector_container<std::int32_t, CountingAllocator<std::int32_t>>>(10, vec);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST(
    "can construct vector_container of given size with value of std::containers type with custom allocator") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        std::vector<std::int32_t, CountingAllocator<std::int32_t>> vec(10, alloc);
        fill_content(vec);
        check_construct_given_size_with_value_with_custom_allocator<
            std::vector<std::int32_t, CountingAllocator<std::int32_t>>>(10, vec);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("can create vector_container of primitives from another vector_container") {
    check_copy_constructor<std::int32_t>();
}

TEST(
    "can create vector_container of vector_containers from another vector_container using copy constructor") {
    check_copy_constructor<vector_container<std::int32_t, CountingAllocator<std::int32_t>>>();
}

TEST(
    "can create vector_container of std::containers from another vector_container using copy constructor") {
    check_copy_constructor<std::vector<std::int32_t, CountingAllocator<std::int32_t>>>();
}

TEST(
    "can create vector_container of primitives from another vector_container using copy assignment") {
    check_copy_assignment<std::int32_t>();
}

TEST(
    "can create vector_container of vector_containers from another vector_container using copy assignment") {
    check_copy_assignment<vector_container<std::int32_t, CountingAllocator<std::int32_t>>>();
}

TEST(
    "can create vector_container of std::containers from another vector_container using copy assignment") {
    check_copy_assignment<std::vector<std::int32_t, CountingAllocator<std::int32_t>>>();
}

TEST("can copy vector_container of primitives with begin(), end() iterators") {
    check_begin_end_copy<std::int32_t>();
}

TEST("can copy vector_container of vector_containers with begin(), end() iterators") {
    check_begin_end_copy<vector_container<std::int32_t, CountingAllocator<std::int32_t>>>();
}

TEST("can copy vector_container of std::containers with begin(), end() iterators") {
    check_begin_end_copy<std::vector<std::int32_t, CountingAllocator<std::int32_t>>>();
}

TEST("can copy vector_container of tuples primitives with begin(), end() iterators") {
    check_begin_end_copy<std::tuple<float, std::int32_t, std::int64_t>>();
}

TEST("can copy vector_container of tuples vector_containers with begin(), end() iterators") {
    check_begin_end_copy<
        std::tuple<vector_container<float, CountingAllocator<float>>,
                   vector_container<std::int32_t, CountingAllocator<std::int32_t>>,
                   vector_container<std::int64_t, CountingAllocator<std::int64_t>>>>();
}

TEST("can copy vector_container of tuples std::containers with begin(), end() iterators") {
    check_begin_end_copy<std::tuple<std::vector<float, CountingAllocator<float>>,
                                    std::vector<std::int32_t, CountingAllocator<std::int32_t>>,
                                    std::vector<std::int64_t, CountingAllocator<std::int64_t>>>>();
}

TEST("can copy vector_container of primitives") {
    check_copy<std::int32_t>();
}

TEST("can copy vector_container of vector_containers") {
    check_copy<vector_container<std::int32_t, CountingAllocator<std::int32_t>>>();
}

TEST("can copy vector_container of std::contaier") {
    check_copy<std::vector<std::int32_t, CountingAllocator<std::int32_t>>>();
}

TEST("can copy vector_container of tuples of primitives") {
    check_copy<std::tuple<float, std::int32_t, std::int64_t>>();
}

TEST("can copy vector_container of tuples of vector_containers") {
    check_copy<std::tuple<vector_container<float, CountingAllocator<float>>,
                          vector_container<std::int32_t, CountingAllocator<std::int32_t>>,
                          vector_container<std::int64_t, CountingAllocator<std::int64_t>>>>();
}

TEST("can copy vector_container of tuples of std::contaier") {
    check_copy<std::tuple<std::vector<float, CountingAllocator<float>>,
                          std::vector<std::int32_t, CountingAllocator<std::int32_t>>,
                          std::vector<std::int64_t, CountingAllocator<std::int64_t>>>>();
}

TEST("can fill vector_container of primitives with value") {
    allocated_bytes_count = 0;
    {
        check_fill<std::int32_t>(1);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("can fill vector_container of vector_container with value") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(10, alloc);
        fill_content(vec);
        check_fill<vector_container<std::int32_t, CountingAllocator<std::int32_t>>>(vec);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("can fill vector_container of std::contaiers with value") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        std::vector<std::int32_t, CountingAllocator<std::int32_t>> vec(10, alloc);
        fill_content(vec);
        check_fill<std::vector<std::int32_t, CountingAllocator<std::int32_t>>>(vec);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("check operator[]") {
    vector_container<std::int32_t> vec(10);
    REQUIRE_NOTHROW(vec[0]);
    // REQUIRE_THROWS_AS(vec[-1]);
    // REQUIRE_THROWS_AS(vec[11]);
}

TEST("check vector_container capacity after push_back() with size < capacity") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(5, alloc);
        vec.push_back(1);
        REQUIRE(vec.size() == 6);
        REQUIRE(vec.capacity() == 8);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("check vector_container capacity after push_back() with size = capacity") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(4, alloc);
        vec.push_back(1);
        REQUIRE(vec.size() == 5);
        REQUIRE(vec.capacity() == 8);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("can reserve new_capacity < old_capacity") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(4, alloc);
        REQUIRE(vec.capacity() == 8);
        REQUIRE(vec.size() == 4);
        vec.reserve(2);
        REQUIRE(vec.capacity() == 8);
        REQUIRE(vec.size() == 4);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("can reserve new_capacity > old_capacity") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(4, alloc);
        REQUIRE(vec.capacity() == 8);
        REQUIRE(vec.size() == 4);
        vec.reserve(9);
        REQUIRE(vec.capacity() == 9);
        REQUIRE(vec.size() == 4);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("can reserve new_capacity > old_capacity with vector_container of vector_containers") {
    allocated_bytes_count = 0;
    {
        using vc_t = vector_container<std::int32_t, CountingAllocator<std::int32_t>>;
        CountingAllocator<vc_t> alloc;
        vector_container<vc_t, CountingAllocator<vc_t>> vec(4, alloc);
        REQUIRE(vec.capacity() == 8);
        REQUIRE(vec.size() == 4);
        vec.reserve(9);
        REQUIRE(vec.capacity() == 9);
        REQUIRE(vec.size() == 4);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("can resize new_size < old_size") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(4, alloc);
        REQUIRE(vec.capacity() == 8);
        REQUIRE(vec.size() == 4);
        vec.resize(1);
        REQUIRE(vec.capacity() == 8);
        REQUIRE(vec.size() == 1);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("can resize new_size > old_size") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(4, alloc);
        REQUIRE(vec.capacity() == 8);
        REQUIRE(vec.size() == 4);
        vec.resize(9);
        REQUIRE(vec.capacity() == 16);
        REQUIRE(vec.size() == 9);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("check mutability of get_mutable_data()") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        std::int64_t n = 5;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(n, alloc);
        std::int32_t* data = vec.get_mutable_data();
        for (int i = 0; i < n; i++) {
            data[i] = i;
        }
        bool equal = true;
        for (int i = 0; i < n; i++) {
            equal &= (i == vec[i]);
        }
        REQUIRE(equal);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("compare get_data() with operator[] result") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        std::int64_t n = 5;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(n, alloc);
        fill_content(vec);
        const std::int32_t* data = vec.get_data();
        bool equal = true;
        for (int i = 0; i < n; i++) {
            equal &= (vec[i] == data[i]);
        }
        REQUIRE(equal);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("check begin() + end() iterators in range-based for") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        std::int64_t n = 5;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(n, alloc);
        int index = 0;
        for (std::int32_t& value : vec) {
            value = index++;
        }
        bool equal = true;
        for (int i = 0; i < n; i++) {
            equal &= (vec[i] == i);
        }
        REQUIRE(equal);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("check begin() + end() iterators for random access") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        std::int64_t n = 5;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(n, alloc);
        REQUIRE(std::distance(vec.begin(), vec.end()) == n);
        REQUIRE(vec[3] == 0);
        *(vec.begin() + 3) = 3;
        REQUIRE(vec[3] == 3);
        REQUIRE(vec[2] == 0);
        *(vec.end() - 3) = 2;
        REQUIRE(vec[2] == 2);
    }
    REQUIRE(allocated_bytes_count == 0);
}

TEST("can return allocator") {
    allocated_bytes_count = 0;
    {
        CountingAllocator<std::int32_t> alloc;
        std::int64_t n = 5;
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec(n, alloc);
        REQUIRE(typeid(alloc) == typeid(vec.get_allocator()));
        vector_container<std::int32_t, CountingAllocator<std::int32_t>> vec_alloc(
            n,
            vec.get_allocator());
        REQUIRE(typeid(vec.get_allocator()) == typeid(vec_alloc.get_allocator()));
    }
    REQUIRE(allocated_bytes_count == 0);
}

} // namespace oneapi::dal::graph::test
