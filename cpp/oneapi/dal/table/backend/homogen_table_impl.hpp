/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::backend {

enum class alloc_kind {
    host, /// Non-USM pointer allocated on host
    usm_host, /// USM pointer allocated by sycl::alloc_host
    usm_device, /// USM pointer allocated by sycl::alloc_device
    usm_shared /// USM pointer allocated by sycl::alloc_shared
};

#ifdef ONEDAL_DATA_PARALLEL
inline alloc_kind alloc_kind_from_sycl(sycl::usm::alloc alloc) {
    using error_msg = dal::detail::error_messages;
    switch (alloc) {
        case sycl::usm::alloc::host: return alloc_kind::usm_host;
        case sycl::usm::alloc::device: return alloc_kind::usm_device;
        case sycl::usm::alloc::shared: return alloc_kind::usm_shared;
        default: throw invalid_argument{ error_msg::unsupported_usm_alloc() };
    }
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
inline sycl::usm::alloc alloc_kind_to_sycl(alloc_kind kind) {
    using error_msg = dal::detail::error_messages;
    switch (kind) {
        case alloc_kind::usm_host: return sycl::usm::alloc::host;
        case alloc_kind::usm_device: return sycl::usm::alloc::device;
        case alloc_kind::usm_shared: return sycl::usm::alloc::shared;
        default: throw invalid_argument{ error_msg::unsupported_usm_alloc() };
    }
}
#endif

inline bool alloc_kind_requires_copy(alloc_kind src_alloc_kind, alloc_kind dst_alloc_kind) {
#ifdef ONEDAL_DATA_PARALLEL
    switch (dst_alloc_kind) {
        case alloc_kind::host: //
            return (src_alloc_kind == alloc_kind::usm_device);
        case alloc_kind::usm_host: //
            return (src_alloc_kind == alloc_kind::host) || //
                   (src_alloc_kind == alloc_kind::usm_device);
        case alloc_kind::usm_device: //
            return (src_alloc_kind == alloc_kind::host) || //
                   (src_alloc_kind == alloc_kind::usm_host);
        case alloc_kind::usm_shared: //
            return (src_alloc_kind != alloc_kind::usm_shared);
        default: //
            ONEDAL_ASSERT(!"Unsupported alloc_kind");
    }
#else
    ONEDAL_ASSERT(src_alloc_kind == alloc_kind::host);
    ONEDAL_ASSERT(dst_alloc_kind == alloc_kind::host);
    return false;
#endif
}

template <typename T>
inline alloc_kind get_alloc_kind(const array<T>& array) {
#ifdef ONEDAL_DATA_PARALLEL
    const auto opt_queue = array.get_queue();
    if (opt_queue.has_value()) {
        const auto queue = opt_queue.value();
        const auto array_alloc = sycl::get_pointer_type(array.get_data(), queue.get_context());
        return alloc_kind_from_sycl(array_alloc);
    }
    else {
        return alloc_kind::host;
    }
#else
    return alloc_kind::host;
#endif
}

table_metadata create_homogen_metadata(std::int64_t feature_count, data_type dtype);

class homogen_table_impl {
public:
    struct host_alloc_t {};

    homogen_table_impl() : row_count_(0), col_count_(0), layout_(data_layout::unknown) {}

    homogen_table_impl(std::int64_t row_count,
                       std::int64_t column_count,
                       const array<byte_t>& data,
                       data_type dtype,
                       data_layout layout)
            : meta_(create_homogen_metadata(column_count, dtype)),
              data_(data),
              row_count_(row_count),
              col_count_(column_count),
              layout_(layout) {
        using error_msg = dal::detail::error_messages;

        if (row_count <= 0) {
            throw dal::domain_error(error_msg::rc_leq_zero());
        }

        if (column_count <= 0) {
            throw dal::domain_error(error_msg::cc_leq_zero());
        }

        detail::check_mul_overflow(row_count, column_count);
        const int64_t element_count = row_count * column_count;
        const int64_t dtype_size = detail::get_data_type_size(dtype);

        detail::check_mul_overflow(element_count, dtype_size);
        if (data.get_count() != element_count * dtype_size) {
            throw dal::domain_error(error_msg::invalid_data_block_size());
        }
        if (layout != data_layout::row_major && layout != data_layout::column_major) {
            throw dal::domain_error(error_msg::unsupported_data_layout());
        }
    }

    std::int64_t get_column_count() const {
        return col_count_;
    }

    std::int64_t get_row_count() const {
        return row_count_;
    }

    const table_metadata& get_metadata() const {
        return meta_;
    }

    const void* get_data() const {
        return data_.get_data();
    }

    array<byte_t> get_data_array() const {
        return data_;
    }

    data_layout get_data_layout() const {
        return layout_;
    }

    template <typename Data>
    void pull_rows(array<Data>& block, const range& rows) const {
        pull_rows_impl(detail::default_host_policy{}, block, rows, alloc_kind::host);
    }

    template <typename Data>
    void push_rows(const array<Data>& block, const range& rows) {
        push_rows_impl(detail::default_host_policy{}, block, rows);
    }

    template <typename Data>
    void pull_column(array<Data>& block, std::int64_t column_index, const range& rows) const {
        pull_column_impl(detail::default_host_policy{},
                         block,
                         column_index,
                         rows,
                         alloc_kind::host);
    }

    template <typename Data>
    void push_column(const array<Data>& block, std::int64_t column_index, const range& rows) {
        push_column_impl(detail::default_host_policy{}, block, column_index, rows);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Data>
    void pull_rows(sycl::queue& queue,
                   array<Data>& block,
                   const range& rows,
                   const sycl::usm::alloc& alloc) const {
        pull_rows_impl(detail::data_parallel_policy{ queue },
                       block,
                       rows,
                       alloc_kind_from_sycl(alloc));
    }

    template <typename Data>
    void push_rows(sycl::queue& queue, const array<Data>& block, const range& rows) {
        push_rows_impl(detail::data_parallel_policy{ queue }, block, rows);
    }

    template <typename Data>
    void pull_column(sycl::queue& queue,
                     array<Data>& block,
                     std::int64_t column_index,
                     const range& rows,
                     const sycl::usm::alloc& alloc) const {
        pull_column_impl(detail::data_parallel_policy{ queue },
                         block,
                         column_index,
                         rows,
                         alloc_kind_from_sycl(alloc));
    }

    template <typename Data>
    void push_column(sycl::queue& queue,
                     const array<Data>& block,
                     std::int64_t column_index,
                     const range& rows) {
        push_column_impl(detail::data_parallel_policy{ queue }, block, column_index, rows);
    }
#endif

private:
    template <typename Policy, typename Data>
    void pull_rows_impl(const Policy& policy,
                        array<Data>& block,
                        const range& rows,
                        const alloc_kind& kind) const;

    template <typename Policy, typename Data>
    void pull_column_impl(const Policy& policy,
                          array<Data>& block,
                          std::int64_t column_index,
                          const range& rows,
                          const alloc_kind& kind) const;

    template <typename Policy, typename Data>
    void push_rows_impl(const Policy& policy, const array<Data>& block, const range& rows);

    template <typename Policy, typename Data>
    void push_column_impl(const Policy& policy,
                          const array<Data>& block,
                          std::int64_t column_index,
                          const range& rows);

    template <typename Policy, typename Data, typename Body>
    void override_policy(const Policy& policy, const array<Data>& block, Body&& body) const;

    table_metadata meta_;
    array<byte_t> data_;
    int64_t row_count_;
    int64_t col_count_;
    data_layout layout_;
};

} // namespace oneapi::dal::backend
