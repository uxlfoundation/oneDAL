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

#include <vector>
#include <numeric>
#include <variant>
#include <optional>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/array_impl.hpp"
#include "oneapi/dal/detail/array_utils.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/chunked_array_impl.hpp"
#include "oneapi/dal/detail/chunked_array_utils.hpp"

namespace oneapi::dal::detail::v2 {

class chunked_array_impl {
public:
    using array_impl_t = detail::array_impl<byte_t>;
    using array_impl_ptr_t = detail::unique<array_impl_t>;

    // Implements RAII principle to guarantee
    // that offsets will be recalculated
    // after each individual access
    class mutable_accessor {
    public:
        mutable_accessor(chunked_array_impl& ref) : parent{ ref } {}

        auto& get_chunks() {
            return parent.chunks;
        }

        auto& get_offsets() {
            return parent.offsets;
        }

        mutable_accessor(const mutable_accessor&) = delete;

        mutable_accessor& operator=(const mutable_accessor&) = delete;

        ~mutable_accessor() {
            parent.update_offsets();
        }

    private:
        chunked_array_impl& parent;
    };

    class immutable_accessor {
    public:
        immutable_accessor(const chunked_array_impl& ref) : parent{ ref } {}

        const auto& get_chunks() const {
            return parent.chunks;
        }

        const auto& get_offsets() const {
            return parent.offsets;
        }

    private:
        const chunked_array_impl& parent;
    };

    explicit chunked_array_impl() {
        update_offsets();
    }

    explicit chunked_array_impl(std::int64_t chunk_count) {
        auto raw = detail::integral_cast<std::size_t>(chunk_count);
        const auto empty_chunk = detail::array_impl<byte_t>{};
        chunks = std::vector<array_impl_t>(raw, empty_chunk);
        [[maybe_unused]] auto res = update_offsets();
        ONEDAL_ASSERT(res == std::int64_t{ 0l });
    }

    std::int64_t update_offsets() {
        const auto new_size = chunks.size();
        offsets.resize(new_size);

        if (new_size == std::size_t{ 0ul }) {
            return std::int64_t{ 0l };
        }

        std::int64_t acc = 0l, iter = 0l;

        for (const auto& chunk : chunks) {
            auto size = chunk.get_size_in_bytes();
            offsets.at(iter++) = std::int64_t(acc += size);
        }

        ONEDAL_ASSERT(offsets.back() == std::int64_t{ acc });
        ONEDAL_ASSERT(std::is_sorted(offsets.cbegin(), offsets.cend()));
        ONEDAL_ASSERT(iter == integral_cast_debug<std::int64_t>(new_size));

        return acc;
    }

    mutable_accessor mutable_access() {
        return mutable_accessor{ *this };
    }

    immutable_accessor immutable_access() const {
        return immutable_accessor{ *this };
    }

private:
    std::vector<std::int64_t> offsets;
    std::vector<array_impl_t> chunks;
};

template <typename Policy, typename Alloc>
auto chunked_array_base::flatten_impl(const Policy& dst_policy, const Alloc& alloc) const
    -> array_impl_t {
    const auto full_size = this->get_size_in_bytes();

    if (full_size == std::int64_t{ 0l }) {
        return array_impl_t{};
    }
    else {
        auto result = array_impl_t::empty_unique( //
            dst_policy,
            full_size,
            alloc);
        copy(*result, *this);
        return *result;
    }
}

#ifdef ONEDAL_DATA_PARALLEL
auto chunked_array_base::flatten_impl(const data_parallel_policy& policy,
                                      const data_parallel_allocator<byte_t>& alloc,
                                      const std::vector<sycl::event>& deps) const -> array_impl_t {
    sycl::event::wait_and_throw(deps);
    return flatten_impl(policy, alloc);
}
#endif // ONEDAL_DATA_PARALLEL

std::int64_t chunked_array_base::get_size_in_bytes() const {
    constexpr std::int64_t zero{ 0l };
    const auto chunk_count = this->get_chunk_count();

    if (chunk_count == zero) {
        return zero;
    }

    const auto accessor = impl_->immutable_access();
    const auto raw_size = accessor.get_offsets().back();
    const auto casted_size = detail::integral_cast_debug<std::int64_t>(raw_size);

#ifdef ONEDAL_ENABLE_ASSERT
    std::int64_t size_acc = zero;
    for (std::int64_t c = zero; c < chunk_count; ++c) {
        const auto& chunk = this->get_chunk_impl(c);
        size_acc += chunk.get_size_in_bytes();
    }
    ONEDAL_ASSERT(casted_size == size_acc);
#endif // ONEDAL_ENABLE_ASSERT

    return casted_size;
}

bool chunked_array_base::validate() const noexcept {
    const auto chunk_count = this->get_chunk_count();
    for (std::int64_t c = 0l; c < chunk_count; ++c) {
        const auto& chunk = this->get_chunk_impl(c);
        if (chunk.get_data() == nullptr) {
            return false;
        }
    }

    return true;
}

template <typename Policy1, typename Policy2>
inline bool same_policy_impl(const Policy1& l, const Policy2& r) {
    using host_policy = detail::default_host_policy;
    constexpr bool l_is_host_policy = std::is_same_v<Policy1, host_policy>;
    constexpr bool r_is_host_policy = std::is_same_v<Policy2, host_policy>;

    if constexpr (l_is_host_policy && r_is_host_policy) {
        return true;
    }
    else {
#ifdef ONEDAL_DATA_PARALLEL
        using device_policy = detail::data_parallel_policy;
        constexpr bool l_is_device_policy = std::is_same_v<Policy1, device_policy>;
        constexpr bool r_is_device_policy = std::is_same_v<Policy2, device_policy>;

        if constexpr (l_is_device_policy && r_is_device_policy) {
            const sycl::queue& queue1 = l.get_queue();
            const sycl::queue& queue2 = r.get_queue();
            using namespace sycl;
            return queue1 == queue2;
        }

#endif // ONEDAL_DATA_PARALLEL
        return false;
    }
}

template <typename PolicyVar>
inline bool same_policy(const PolicyVar& zero, const PolicyVar& curr) {
    const auto impl = [](const auto& l, const auto& r) {
        return same_policy_impl(l, r);
    };

    return std::visit(impl, zero, curr);
}

bool chunked_array_base::have_same_policies() const {
#ifdef ONEDAL_DATA_PARALLEL
    const auto chunk_count = this->get_chunk_count();

    if (std::int64_t{ 1l } < chunk_count) {
        const auto& zero_chunk = this->get_chunk_impl(0);
        const auto zero_policy = zero_chunk.get_policy();

        for (std::int64_t i = 1l; i < chunk_count; ++i) {
            const auto& curr = this->get_chunk_impl(i);
            const auto curr_policy = curr.get_policy();

            if (!same_policy(zero_policy, curr_policy)) {
                return false;
            }
        }
    }
#endif // ONEDAL_DATA_PARALLEL
    return true;
}

bool chunked_array_base::is_contiguous() const {
    ONEDAL_ASSERT(this->validate());

    const auto chunk_count = this->get_chunk_count();
    const auto same_policy = this->have_same_policies();

    if (same_policy && std::int64_t{ 1l } < chunk_count) {
        for (std::int64_t c = 1l; c < chunk_count; ++c) {
            const auto& prev = this->get_chunk_impl(c - 1);
            const auto& curr = this->get_chunk_impl(c);

            const auto* const prev_ptr = prev.get_data();
            const auto* const curr_ptr = curr.get_data();

            const auto prev_size = prev.get_size_in_bytes();

            if (prev_ptr + prev_size != curr_ptr) {
                return false;
            }
        }
    }

    return true;
}

std::int64_t chunked_array_base::get_chunk_count() const noexcept {
    const std::size_t res = impl_->immutable_access().get_chunks().size();
    return detail::integral_cast_debug<std::int64_t>(res);
}

const array_impl<byte_t>& chunked_array_base::get_chunk_impl(std::int64_t i) const {
    const auto cbegin = impl_->immutable_access().get_chunks().cbegin();
    using diff_t = typename decltype(cbegin)::difference_type;
    const auto element = detail::integral_cast_debug<diff_t>(i);

    return *std::next(cbegin, element);
}

array_impl<byte_t>& chunked_array_base::get_mut_chunk_impl(std::int64_t i) const {
    auto accessor = impl_->mutable_access();
    const auto begin = accessor.get_chunks().begin();

    using diff_t = typename decltype(begin)::difference_type;
    const auto element = detail::integral_cast_debug<diff_t>(i);

    return *std::next(begin, element);
}

void chunked_array_base::set_chunk_impl(std::int64_t i, array_impl_t array) {
    auto accessor = impl_->mutable_access();
    const auto begin = accessor.get_chunks().begin();

    using diff_t = typename decltype(begin)::difference_type;
    const auto position = detail::integral_cast_debug<diff_t>(i);

#ifdef ONEDAL_ENABLE_ASSERT
    const auto* const ptr_check = array.get_data();
    const auto size_check = array.get_size_in_bytes();
#endif // ONEDAL_ENABLE_ASSERT

    auto iterator = std::next(begin, position);
    *iterator = std::move(array);

#ifdef ONEDAL_ENABLE_ASSERT
    ONEDAL_ASSERT((*iterator).get_data() == ptr_check);
    ONEDAL_ASSERT((*iterator).get_size_in_bytes() == size_check);
#endif // ONEDAL_ENABLE_ASSERT
}

void chunked_array_base::append_impl(array_impl_t arr) const {
    auto accessor = impl_->mutable_access();
    accessor.get_chunks().emplace_back(std::move(arr));
}

void chunked_array_base::append_impl(chunked_array_base arr) const {
    const auto chunk_count = arr.get_chunk_count();
    [[maybe_unused]] const auto init_count = this->get_chunk_count();

    for (std::int64_t c = 0l; c < chunk_count; ++c) {
        const auto& chunk = arr.get_chunk_impl(c);
        this->append_impl(chunk);
    }

    ONEDAL_ASSERT(init_count + chunk_count == this->get_chunk_count());
}

auto chunked_array_base::make_array_impl(std::int64_t chunk_count) -> impl_ptr_t {
    return std::make_shared<impl_t>(chunk_count);
}

template <typename Policy, typename Alloc>
array_impl<const byte_t*> chunked_array_base::get_data_impl(const Policy& dst_policy,
                                                            const Alloc& alloc) const {
    using def_policy_t = detail::default_host_policy;
    using def_alloc_t = policy_allocator_t<def_policy_t, const byte_t*>;

    const def_alloc_t tmp_alloc{};
    const def_policy_t tmp_policy{};

    const auto chunk_count = this->get_chunk_count();
    if constexpr (std::is_same_v<Policy, def_policy_t>) {
        auto result = array_impl<const byte_t*>::empty_unique( //
            tmp_policy,
            chunk_count,
            tmp_alloc);
        auto* const res_ptr = result->get_mutable_data();

        for (std::int64_t c = 0l; c < chunk_count; ++c) {
            const auto& chunk = this->get_chunk_impl(c);
            res_ptr[c] = chunk.get_data();
        }

        return *result;
    }
    else {
        auto tmp = get_data_impl(tmp_policy, tmp_alloc);
        return copy_impl(dst_policy, tmp, alloc);
    }
}

template <typename Policy, typename Alloc>
array_impl<byte_t*> chunked_array_base::get_mutable_data_impl(const Policy& dst_policy,
                                                              const Alloc& alloc) const {
    using def_policy_t = detail::default_host_policy;
    using def_alloc_t = policy_allocator_t<def_policy_t, byte_t*>;

    const def_alloc_t tmp_alloc{};
    const def_policy_t tmp_policy{};

    const auto chunk_count = this->get_chunk_count();
    if constexpr (std::is_same_v<Policy, def_policy_t>) {
        auto result = array_impl<byte_t*>::empty_unique( //
            tmp_policy,
            chunk_count,
            tmp_alloc);
        auto* const res_ptr = result->get_mutable_data();

        for (std::int64_t c = 0l; c < chunk_count; ++c) {
            const auto& chunk = this->get_chunk_impl(c);
            res_ptr[c] = chunk.get_mutable_data();
        }

        return *result;
    }
    else {
        auto tmp = get_mutable_data_impl(tmp_policy, tmp_alloc);
        return copy_impl(dst_policy, tmp, alloc);
    }
}

chunked_array_base chunked_array_base::get_slice_impl(std::int64_t first, std::int64_t last) const {
    constexpr std::int64_t zero = 0l;

    if (first == last) {
        return chunked_array_base{};
    }

    const auto accessor = impl_->immutable_access();

    const auto& offsets = accessor.get_offsets();
    const auto fiter = offsets.cbegin();
    const auto liter = offsets.cend();

    const auto first_array = std::upper_bound(fiter, liter, first);
    const auto maybe_last = std::lower_bound(first_array, liter, last);
    const auto last_array = std::next(maybe_last);

    const auto first_idx = std::distance(fiter, first_array);
    const auto last_idx = std::distance(fiter, last_array);
    ONEDAL_ASSERT(first_idx <= last_idx);

    std::int64_t offset = zero;
    chunked_array_base result(last_idx - first_idx);
    for (std::int64_t i = first_idx; i < last_idx; ++i) {
        const auto& chunk = this->get_chunk_impl(i);

        const auto first_in_chunk = (i != zero) ? offsets.at(i - 1l) : zero;
        const auto last_in_chunk = offsets.at(i);
        const auto chunk_size = last_in_chunk - first_in_chunk;
        ONEDAL_ASSERT(chunk_size == chunk.get_size_in_bytes());

        const auto rel_idx = i - first_idx;
        // The most straightforward path
        if ((first <= first_in_chunk) && (last_in_chunk <= last)) {
            array_impl_t temp = chunk;
            offset += chunk.get_size_in_bytes();
            result.set_chunk_impl(rel_idx, std::move(temp));
        }
        else {
            const auto prefix_offset = first - first_in_chunk;
            const auto suffix_offset = chunk_size - (last_in_chunk - last);

            const auto prefix = std::max(prefix_offset, zero);
            const auto suffix = std::min(suffix_offset, chunk_size);

            auto slice = chunk.get_slice(prefix, suffix);
            offset += slice.get_size_in_bytes();
            result.set_chunk_impl(rel_idx, std::move(slice));
        }
    }
    ONEDAL_ASSERT(offset == last - first);

    return result;
}

void chunked_array_base::deserialize_impl(data_type dtype, detail::input_archive& archive) {
    constexpr std::int64_t zero{ 0l };
    const std::int64_t size_in_bytes = archive.pop<std::int64_t>();
    if (size_in_bytes < zero || zero < size_in_bytes % get_data_type_size(dtype)) {
        throw invalid_argument{ error_messages::archive_content_does_not_match_type() };
    }

    if (zero < size_in_bytes) {
        array_impl_t result;

        const detail::default_host_policy policy{};
        const auto deleter = make_default_delete<byte_t>(policy);
        byte_t* const data = malloc<byte_t>(policy, size_in_bytes);

        archive.range(data, data + size_in_bytes);

        auto accessor = impl_->mutable_access();

        accessor.get_chunks().emplace_back(policy, //
                                           data,
                                           size_in_bytes,
                                           std::move(deleter));
    }
}

void chunked_array_base::serialize_impl(detail::output_archive& archive) const {
    const detail::default_host_policy policy{};
    const detail::host_allocator<byte_t> alloc{};

    auto flat_array = this->flatten_impl(policy, alloc);
    const auto size = flat_array.get_size_in_bytes();
    archive(size);

    if (std::int64_t{ 0l } < size) {
        const auto* const ptr = flat_array.get_data();
        archive.range(ptr, ptr + size);
    }
}

template array_impl<byte_t> chunked_array_base::flatten_impl(const default_host_policy&,
                                                             const host_allocator<byte_t>&) const;
template array_impl<const byte_t*> chunked_array_base::get_data_impl(
    const default_host_policy&,
    const host_allocator<const byte_t*>&) const;
template array_impl<byte_t*> chunked_array_base::get_mutable_data_impl(
    const default_host_policy&,
    const host_allocator<byte_t*>&) const;

#ifdef ONEDAL_DATA_PARALLEL
template array_impl<byte_t> chunked_array_base::flatten_impl(
    const data_parallel_policy&,
    const data_parallel_allocator<byte_t>&) const;
template array_impl<const byte_t*> chunked_array_base::get_data_impl(
    const data_parallel_policy&,
    const data_parallel_allocator<const byte_t*>&) const;
template array_impl<byte_t*> chunked_array_base::get_mutable_data_impl(
    const data_parallel_policy&,
    const data_parallel_allocator<byte_t*>&) const;
#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::detail::v2
