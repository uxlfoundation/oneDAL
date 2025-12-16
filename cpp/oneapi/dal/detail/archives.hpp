/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/detail/paged_vector.hpp"

namespace oneapi::dal::detail {
namespace v1 {

constexpr std::uint32_t binary_archive_magic = 0x4441414F;

class binary_output_archive : public base {
public:
    binary_output_archive() = default;

    binary_output_archive(const binary_output_archive&) = delete;
    binary_output_archive& operator=(const binary_output_archive&) = delete;

    ~binary_output_archive() = default;

    void prologue() {
        is_valid_ = false;
        const std::uint32_t magic = binary_archive_magic;
        operator()(&magic, make_data_type<std::uint32_t>());
    }

    void epilogue() {
        is_valid_ = true;
    }

    void operator()(const void* data, data_type dtype, std::int64_t count = 1) {
        ONEDAL_ASSERT(data);
        ONEDAL_ASSERT(count > 0);

        const std::int64_t type_size = get_data_type_size(dtype);
        const std::int64_t byte_count = check_mul_overflow(type_size, count);

        content_.push_back(reinterpret_cast<const byte_t*>(data), byte_count);
    }

    void reset() {
        is_valid_ = true;
        content_.reset();
    }

    bool is_valid() const {
        return is_valid_;
    }

    std::int64_t get_size() const {
        return integral_cast<std::int64_t>(content_.get_count());
    }

    array<byte_t> to_array() const {
        if (!is_valid_) {
            throw internal_error{ error_messages::archive_is_in_invalid_state() };
        }

        return content_.to_array();
    }

private:
    static constexpr std::int64_t min_page_size = 32;
    paged_vector<byte_t> content_{ min_page_size };
    bool is_valid_ = true;
};

class binary_input_archive : public base {
public:
    explicit binary_input_archive(const array<byte_t>& data) : input_data_(data) {}

    binary_input_archive(const byte_t* data, std::int64_t size_in_bytes)
            : input_data_(array<byte_t>::wrap(data, size_in_bytes)) {}

    void prologue() {
        is_valid_ = false;

        std::uint32_t magic;
        operator()(&magic, make_data_type<std::uint32_t>());
        if (magic != binary_archive_magic) {
            throw invalid_argument{ error_messages::archive_content_does_not_match_type() };
        }
    }

    void epilogue() {
        is_valid_ = true;
    }

    void operator()(void* data, data_type dtype, std::int64_t count = 1) {
        ONEDAL_ASSERT(data);
        ONEDAL_ASSERT(count > 0);

        const std::int64_t type_size = get_data_type_size(dtype);
        const std::int64_t byte_count = check_mul_overflow(type_size, count);

        if (position_ + byte_count > input_data_.get_count()) {
            throw invalid_argument{ error_messages::archive_content_does_not_match_type() };
        }

        for (std::int64_t i = 0; i < byte_count; i++) {
            reinterpret_cast<byte_t*>(data)[i] = input_data_[position_ + i];
        }
        position_ += byte_count;
    }

    bool is_valid() const {
        return is_valid_;
    }

private:
    array<byte_t> input_data_;
    std::int64_t position_ = 0;
    bool is_valid_ = true;
};

} // namespace v1

using v1::binary_output_archive;
using v1::binary_input_archive;

} // namespace oneapi::dal::detail
