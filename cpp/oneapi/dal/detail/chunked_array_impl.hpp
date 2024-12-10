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

#include <variant>
#include <optional>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/array_impl.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::detail {

namespace v2 {

template <typename T>
class chunked_array;

class chunked_array_impl;

class ONEDAL_EXPORT chunked_array_base : public base {
    using this_t = chunked_array_base;
    using impl_t = chunked_array_impl;
    using impl_ptr_t = detail::shared<impl_t>;
    using array_impl_t = detail::array_impl<byte_t>;

    template <typename T>
    friend class chunked_array;

public:
    bool is_contiguous() const;
    bool have_same_policies() const;
    bool validate() const noexcept;
    std::int64_t get_size_in_bytes() const;
    std::int64_t get_chunk_count() const noexcept;

    chunked_array_base(impl_ptr_t&& impl) : impl_{ std::forward<impl_ptr_t>(impl) } {}

    chunked_array_base(const impl_ptr_t& impl) : impl_{ impl } {}

    chunked_array_base(chunked_array_base&& other) {
        reset(std::forward<chunked_array_base>(other));
    }

    chunked_array_base(const chunked_array_base& other) {
        reset(other);
    }

    chunked_array_base& operator=(chunked_array_base&& other) {
        reset(std::forward<chunked_array_base>(other));
        return *this;
    }

    chunked_array_base& operator=(const chunked_array_base& other) {
        reset(other);
        return *this;
    }

    ~chunked_array_base() = default;

    chunked_array_base() {
        reset();
    }

    template <typename... Arrays>
    void append(const Arrays&... arrays) const {
        detail::apply(
            [&, this](const auto& array) {
                this->append_impl(array);
            },
            arrays...);
    }

    chunked_array_base(std::int64_t chunk_count) {
        impl_ = make_array_impl(chunk_count);
    }

    template <typename Type>
    chunked_array_base(const array<Type>& array) : chunked_array_base(1l) {
        this->set_chunk_impl(0l, array);
        ONEDAL_ASSERT(this->validate());
    }

    template <typename... Types>
    chunked_array_base(const array<Types>&... arrays) : chunked_array_base(sizeof...(Types)) {
        // Ensures that it can be converted directly
        const auto chunk_count = this->get_chunk_count();

        std::int64_t chunk = 0l;

        // Template trick to init implementation using fold expression
        detail::apply(
            [&, this](const auto& array) -> void {
                this->set_chunk_impl(chunk++, array);
            },
            arrays...);

        // Checks that we used all input arrays
        ONEDAL_ASSERT(chunk == this->get_chunk_count());
        ONEDAL_ASSERT(chunk == chunk_count);

        // Checks that result is full with data
        ONEDAL_ASSERT(this->validate());
    }

    void deserialize_impl(data_type dtype, detail::input_archive& ar);
    void serialize_impl(detail::output_archive& ar) const;

    chunked_array_base get_slice_impl(std::int64_t first, std::int64_t last) const;

    template <typename Policy, typename Alloc = policy_allocator_t<Policy, const byte_t*>>
    array_impl<const byte_t*> get_data_impl(const Policy& policy, const Alloc& allocator) const;

    template <typename Policy, typename Alloc = policy_allocator_t<Policy, byte_t*>>
    array_impl<byte_t*> get_mutable_data_impl(const Policy& policy, const Alloc& allocator) const;

    template <typename Policy, typename Alloc = policy_allocator_t<Policy>>
    array_impl_t flatten_impl(const Policy& policy, const Alloc& allocator) const;

#ifdef ONEDAL_DATA_PARALLEL // Minor
    array_impl_t flatten_impl(const data_parallel_policy& policy,
                              const data_parallel_allocator<byte_t>& alloc,
                              const std::vector<sycl::event>& deps) const;
#endif // ONEDAL_DATA_PARALLEL

    void set_chunk_impl(std::int64_t i, array_impl_t array);
    const array_impl_t& get_chunk_impl(std::int64_t i) const;
    array_impl_t& get_mut_chunk_impl(std::int64_t) const;

    template <typename Type>
    void set_chunk_impl(std::int64_t chunk, const array<Type>& array) {
        auto byte_array = as_byte_array(array);

        this->set_chunk_impl(chunk, std::move(byte_array));
    }

    template <typename Type>
    void append_impl(const array<Type>& arr) const {
        [[maybe_unused]] const auto init_count = this->get_chunk_count();

        const auto byte_array = as_byte_array(arr);
        this->append_impl(std::move(byte_array));

        ONEDAL_ASSERT(this->get_chunk_count() == init_count + 1l);
    }

    template <typename Type>
    void append_impl(const chunked_array<Type>& arr) const {
        [[maybe_unused]] const auto init_count = this->get_chunk_count();
        [[maybe_unused]] const auto chunk_count = arr.get_chunk_count();

        const auto& ref = static_cast<const chunked_array_base&>(arr);
        ONEDAL_ASSERT(ref.validate());
        this->append_impl(ref);

        ONEDAL_ASSERT(this->get_chunk_count() == init_count + chunk_count);
    }

    void append_impl(array_impl_t arr) const;
    void append_impl(chunked_array_base arr) const;

private:
    void reset() {
        this->impl_ = make_array_impl(0l);
    }

    void reset(chunked_array_base&& other) {
        this->impl_ = other.impl_;
        auto empty = chunked_array_base{};
        std::swap(other.impl_, empty.impl_);
    }

    void reset(const chunked_array_base& other) {
        this->impl_ = other.impl_;
    }

    template <typename Type>
    static array_impl_t as_byte_array(const array<Type>& arr) {
        constexpr detail::pimpl_accessor acc{};
        const auto& internals = acc.get_pimpl(arr);
        return array_impl_t::reinterpret(*internals);
    }

    static impl_ptr_t make_array_impl(std::int64_t chunk_count);

    impl_ptr_t impl_;
};

inline std::int64_t get_element_count(data_type dt, const chunked_array_base& arr) {
    const auto array_size = arr.get_size_in_bytes();
    const auto elem_size = get_data_type_size(dt);

    const auto result = array_size / elem_size;
    ONEDAL_ASSERT(result * elem_size == array_size);

    return result;
}

} // namespace v2

using v2::chunked_array_base;
using v2::get_element_count;

} // namespace oneapi::dal::detail
