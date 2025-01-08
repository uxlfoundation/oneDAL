/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/chunked_array_impl.hpp"

namespace oneapi::dal {
namespace v2 {

template <typename T>
class ONEDAL_EXPORT chunked_array : public detail::chunked_array_base {
    static_assert(!std::is_const_v<T>);

    using base_t = detail::chunked_array_base;

    template <typename U>
    friend class array;

    template <typename U>
    friend class detail::array_impl;

    friend detail::pimpl_accessor;

    friend detail::serialization_accessor;

public:
    using data_t = T;

    /// @brief       Move constructor
    /// @param array Source array
    chunked_array(chunked_array&& array) : chunked_array_base{ std::forward<base_t>(array) } {}

    /// @brief       Zero copy construct `chunked_array`
    /// @param array Source array
    chunked_array(const chunked_array& array) : chunked_array_base{ array } {}

    /// @brief       Move assignment
    /// @param array Source array
    chunked_array& operator=(chunked_array&& array) = default;

    /// @brief       Copy assignment
    /// @param array Source array
    chunked_array& operator=(const chunked_array& array) = default;

    /// @brief        Destructor
    ~chunked_array() = default;

    /// @brief             Constructs an empty `chunked_array`
    ///                    with unpopulated chunks
    /// @param chunk_count Number of empty chunks in the
    ///                    constructed `chunked_array`
    chunked_array(std::int64_t chunk_count) : chunked_array_base{ chunk_count } {}

    /// @brief       Consumes one contiguous array and converts
    ///              it to a chunked_array
    /// @param array Particular array source
    chunked_array(dal::array<T>&& array)
            : chunked_array_base{ std::forward<dal::array<T>>(array) } {}

    /// @brief            Construct `chunked_array` from list of arrays
    ///                   of the same type or rather `chunked_array`s
    /// @tparam ...Arrays Types of arrays
    /// @param ...arrays  Constant references to arrays
    template <typename... Arrays>
    chunked_array(Arrays&&... arrays) : chunked_array_base{ std::forward<Arrays>(arrays)... } {}

    /// @brief         Constructs dense array from current `chunked_array`
    /// @tparam Policy Should be either `default_host or `data_parallel` policy
    /// @tparam Alloc  Either `host_allocator` or `data_parallel_allocator`
    ///                to allocate `byte_t`
    /// @param policy  Particular instance of policy
    /// @param alloc   Particular instance of allocator
    /// @return        Contiguous array of type `T`
    /// @todo          Optimize to support zero copy in case of
    ///                contiguous array disguised as a `chunked_array`
    template <typename Policy, typename Alloc>
    array<T> flatten(const Policy& policy, const Alloc& alloc) const {
        ONEDAL_ASSERT(base_t::validate());

        const auto raw = base_t::flatten_impl(policy, alloc);
        const auto impl = detail::array_impl<T>::reinterpret(raw);
        return array<T>{ new detail::array_impl<T>{ std::move(impl) } };
    }

    /// @brief         Constructs dense array from current `chunked_array`
    /// @tparam Policy Should be either `default_host or `data_parallel` policy
    /// @param policy  Particular instance of policy
    /// @return        Contiguous array of type `T`
    template <typename Policy>
    array<T> flatten(const Policy& policy) const {
        using alloc_t = detail::policy_allocator_t<Policy>;
        return this->flatten(policy, alloc_t{ policy });
    }

    /// @brief  Constructs dense array from current `chunked_array`
    /// @return Contiguous array allocated on host with default allocator
    array<T> flatten() const {
        const detail::default_host_policy policy{};
        return this->flatten(policy);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// @brief        Constructs dense array from current `chunked_array`
    /// @param policy Particular instance of `data_parallel_policy`
    /// @param deps   Data dependency list
    /// @return       Contiguous array allocated on `shared` memory
    array<T> flatten(const detail::data_parallel_policy& policy,
                     const std::vector<sycl::event>& deps) const {
        sycl::event::wait_and_throw(deps);
        return this->flatten(policy);
    }

    /// @brief       Constructs dense array from current `chunked_array`
    /// @param queue Queue that represent device backend
    /// @param alloc Type of USM allocation
    /// @param deps  Data dependency list
    /// @return      Contiguous array allocated according to `alloc` value
    array<T> flatten(sycl::queue& queue,
                     sycl::usm::alloc alloc = sycl::usm::alloc::shared,
                     const std::vector<sycl::event>& deps = {}) const {
        sycl::event::wait_and_throw(deps);

        using alloc_t = detail::data_parallel_allocator<byte_t>;
        const auto policy = detail::data_parallel_policy(queue);
        return this->flatten(policy, alloc_t{ policy, alloc });
    }
#endif // ONEDAL_DATA_PARALLEL

    /// @brief         Returns array of pointers to data chunks
    /// @tparam Policy Defines how an output array will be allocated
    /// @tparam Alloc  Defines where an output array will be allocated
    /// @param policy  Specific instance of policy
    /// @param alloc   Specific instance of allocator
    /// @return        Array of pointers to memory spans
    template <typename Policy, typename Alloc>
    array<const T*> get_data(const Policy& policy, const Alloc& alloc) const {
        ONEDAL_ASSERT(base_t::validate());

        const auto raw = base_t::get_data_impl(policy, alloc);
        const auto impl = detail::array_impl<const T*>::reinterpret(raw);
        return array<const T*>{ new detail::array_impl<const T*>{ std::move(impl) } };
    }

    template <typename Policy>
    array<const T*> get_data(const Policy& policy) const {
        using alloc_t = detail::policy_allocator_t<Policy, const byte_t*>;
        return this->get_data(policy, alloc_t{ policy });
    }

    /// @brief  Constructs array of pointers to data chunks
    /// @return Array of pointers allocated on host device
    array<const T*> get_data() const {
        const detail::default_host_policy policy{};
        return this->get_data(policy);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// @brief        Returns pointer to immutable data chunks
    /// @param policy Data parallel policy to describe device
    /// @param deps   List of data dependencies
    /// @return       Array of pointers to the immutable data
    array<const T*> get_data(const detail::data_parallel_policy& policy,
                             const std::vector<sycl::event>& deps) const {
        sycl::event::wait_and_throw(deps);
        return this->get_data(policy);
    }

    /// @brief       Constructs array of pointers to the data chunks
    /// @param queue `sycl::queue` storing information abot the device
    /// @param alloc Type of memory allocation for the result
    /// @param deps  List of data dependencies
    /// @return      Array  of pointers located on the device
    array<const T*> get_data(sycl::queue& queue,
                             sycl::usm::alloc alloc = sycl::usm::alloc::shared,
                             const std::vector<sycl::event>& deps = {}) const {
        sycl::event::wait_and_throw(deps);

        using alloc_t = detail::data_parallel_allocator<const byte_t*>;
        const auto policy = detail::data_parallel_policy(queue);
        return this->get_data(policy, alloc_t{ policy, alloc });
    }
#endif // ONEDAL_DATA_PARALLEL

    /// @brief       Provides access to the chunk as
    ///              dense array of the same type
    /// @param chunk Index of chunk in question
    /// @return      Dense array representing chunk
    array<T> get_chunk(std::int64_t chunk) const {
        const auto& raw = base_t::get_chunk_impl(chunk);
        const auto impl = detail::array_impl<T>::reinterpret(raw);
        return array<T>{ new detail::array_impl<T>{ std::move(impl) } };
    }

    /// @brief     Sets particular chunk
    /// @param i   Index of chunk to be set
    /// @param arr Array that will be casted to the chunk
    /// @return    Reference to the same `chunked_array`
    /// @note      Can be relatively slow since `offsets`
    ///            of chunks will be reconstructed
    chunked_array<T>& set_chunk(std::int64_t i, const array<T>& arr) {
        base_t::set_chunk_impl(i, arr);
        return *this;
    }

    /// @brief   Computes number of elements of type `T`
    ///          in the particular instance of `chunked_array`
    /// @pre         The `chunked_array` should be valid
    ///              namely `carray.valid() == true`
    /// @return  Number of elements in current `chunked_array`
    std::int64_t get_count() const {
        constexpr auto dtype = detail::make_data_type<T>();
        return detail::get_element_count(dtype, *this);
    }

    /// @brief       Constructs slice of data from
    ///              current `chunked_array`
    /// @pre         The `chunked_array` should be valid
    ///              namely `carray.valid() == true`
    /// @param first Index of the first element of slice
    /// @param last  Index after the last element in slice
    /// @return      Newly constructed `chunked array`
    ///              pointing to the same memory
    /// @note        Zero copy, however - not that performant
    ///              since require iterating throw all chunks
    chunked_array<T> get_slice(std::int64_t first, std::int64_t last) const {
        constexpr std::int64_t size = sizeof(T);
        const auto first_in_bytes = first * size;
        const auto last_in_bytes = last * size;

        const auto res = base_t::get_slice_impl( //
            first_in_bytes,
            last_in_bytes);
        return chunked_array<T>{ std::move(res) };
    }

    /// @brief  Appends multiple arrays to the current
    ///         one each can be an instance of either
    ///         `array<T>` or rather `chunked_array<T>`
    template <typename... Arrays>
    chunked_array& append(const Arrays&... arrays) {
        // Check for having the same type
        detail::apply(
            [](const auto& array) -> void {
                using arr_t = std::decay_t<decltype(array)>;
                using arr_data_t = typename arr_t::data_t;
                static_assert(std::is_same_v<data_t, arr_data_t>);
            },
            arrays...);

        base_t::append(arrays...);

        return *this;
    }

    bool is_contiguous() const {
        return base_t::is_contiguous();
    }

    bool have_same_policies() const {
        return base_t::have_same_policies();
    }

    bool validate() const noexcept {
        return base_t::validate();
    }

    std::int64_t get_size_in_bytes() const {
        return base_t::get_size_in_bytes();
    }

    std::int64_t get_chunk_count() const noexcept {
        return base_t::get_chunk_count();
    }

    template <typename... Args>
    static inline chunked_array<T> make(Args&&... args) {
        return chunked_array<T>(std::forward<Args>(args)...);
    }

protected:
    chunked_array(const chunked_array_base& other) : chunked_array_base{ other } {}

    chunked_array(chunked_array_base&& other)
            : chunked_array_base{ std::forward<chunked_array>(other) } {}

    /// @brief Private methode to store content into
    ///        memory single span
    /// @note  Will call `flatten` function on host
    void serialize(detail::output_archive& ar) const {
        ar(get_dtype());

        base_t::serialize_impl(ar);
    }

    /// @brief Private method to restore content inplace
    /// @note  Will construct a single chunk backend
    void deserialize(detail::input_archive& ar) {
        constexpr auto dtype = get_dtype();
        const auto ar_dtype = ar.pop<data_type>();
        ONEDAL_ASSERT(dtype == ar_dtype);

        base_t::deserialize_impl(dtype, ar);
    }

    constexpr static data_type get_dtype() {
        return detail::make_data_type<T>();
    }
}; // class chunked_array

} // namespace v2

using v2::chunked_array;

} // namespace oneapi::dal
