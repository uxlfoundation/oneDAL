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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/backend/common_kernels.hpp"
#include "oneapi/dal/table/backend/homogen_kernels.hpp"
#include "oneapi/dal/backend/serialization.hpp"
#include <iostream>
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::backend {

class homogen_table_impl : public detail::homogen_table_template<homogen_table_impl>,
                           public ONEDAL_SERIALIZABLE(homogen_table_id) {
public:
    homogen_table_impl() : row_count_(0), col_count_(0), layout_(data_layout::unknown) {}

    homogen_table_impl(std::int64_t row_count,
                       std::int64_t column_count,
                       const array<byte_t>& data,
                       data_type dtype,
                       data_layout layout)
            : meta_(create_metadata(column_count, dtype)),
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
        const std::int64_t element_count = row_count * column_count;
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);

        detail::check_mul_overflow(element_count, dtype_size);
        if (data.get_count() != element_count * dtype_size) {
            throw dal::domain_error(error_msg::invalid_data_block_size());
        }
        if (layout != data_layout::row_major && layout != data_layout::column_major) {
            throw dal::domain_error(error_msg::unsupported_data_layout());
        }
    }

    std::int64_t get_column_count() const override {
        return col_count_;
    }

    std::int64_t get_row_count() const override {
        return row_count_;
    }

    const table_metadata& get_metadata() const override {
        return meta_;
    }

    array<byte_t> get_data() const override {
        return data_;
    }

    data_layout get_data_layout() const override {
        return layout_;
    }

    std::int64_t get_kind() const override {
        return 1;
    }

    template <typename T>
    void pull_rows_template(const detail::default_host_policy& policy,
                            array<T>& block,
                            const range& rows) const {
        homogen_pull_rows(policy, get_info(), data_, block, rows, alloc_kind::host);
    }

    template <typename T>
    void pull_column_template(const detail::default_host_policy& policy,
                              array<T>& block,
                              std::int64_t column_index,
                              const range& rows) const {
        homogen_pull_column(policy, get_info(), data_, block, column_index, rows, alloc_kind::host);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_rows_template(const detail::data_parallel_policy& policy,
                            array<T>& block,
                            const range& rows,
                            sycl::usm::alloc alloc) const {
        ONEDAL_PROFILER_TASK(pull_rows_template, policy.get_queue());
        homogen_pull_rows(policy, get_info(), data_, block, rows, alloc_kind_from_sycl(alloc));
    }
#endif

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_column_template(const detail::data_parallel_policy& policy,
                              array<T>& block,
                              std::int64_t column_index,
                              const range& rows,
                              sycl::usm::alloc alloc) const {
        homogen_pull_column(policy,
                            get_info(),
                            data_,
                            block,
                            column_index,
                            rows,
                            alloc_kind_from_sycl(alloc));
    }
#endif

    void serialize(detail::output_archive& ar) const override {
        ar(meta_, data_, row_count_, col_count_, layout_);
    }

    void deserialize(detail::input_archive& ar) override {
        ar(meta_, data_, row_count_, col_count_, layout_);
    }

private:
    homogen_info get_info() const {
        return { row_count_, col_count_, meta_.get_data_type(0), layout_ };
    }

    table_metadata meta_;
    array<byte_t> data_;
    std::int64_t row_count_;
    std::int64_t col_count_;
    data_layout layout_;
};

} // namespace oneapi::dal::backend
