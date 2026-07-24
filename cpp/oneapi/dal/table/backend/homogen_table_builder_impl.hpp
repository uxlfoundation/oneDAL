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

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/backend/homogen_kernels.hpp"
#include "oneapi/dal/table/backend/homogen_table_impl.hpp"
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend {

class homogen_table_builder_impl
        : public detail::homogen_table_builder_template<homogen_table_builder_impl> {
public:
    homogen_table_builder_impl() {
        reset();
    }

    void reset() {
        data_.reset();
        row_count_ = 0;
        column_count_ = 0;
        layout_ = data_layout::row_major;
        dtype_ = data_type::float32;
    }

    void reset(const array<byte_t>& data,
               std::int64_t row_count,
               std::int64_t column_count) override {
        if (get_data_size(row_count, column_count, dtype_) != data.get_count()) {
            throw dal::range_error(dal::detail::error_messages::invalid_data_block_size());
        }

        data_ = data;
        row_count_ = row_count;
        column_count_ = column_count;
    }

    void set_data_type(data_type dt) override {
        dtype_ = dt;
        data_.reset();
        row_count_ = 0;
        column_count_ = 0;
    }

    void set_feature_type(feature_type ft) override {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    void allocate(std::int64_t row_count, std::int64_t column_count) override {
        if (row_count <= 0) {
            throw dal::domain_error(dal::detail::error_messages::rc_leq_zero());
        }

        if (column_count <= 0) {
            throw dal::domain_error(dal::detail::error_messages::cc_leq_zero());
        }

        const std::int64_t data_size = get_data_size(row_count, column_count, dtype_);
        data_.reset(data_size);
        row_count_ = row_count;
        column_count_ = column_count;
    }

    void set_layout(data_layout layout) override {
        layout_ = layout;
    }

    void copy_data(const void* data, std::int64_t row_count, std::int64_t column_count) override {
        check_copy_data_preconditions(row_count, column_count);

        __ONEDAL_IF_QUEUE__(data_.get_queue(), {
            auto this_q = data_.get_queue().value();
            ONEDAL_ASSERT(is_known_usm(data_));
            detail::memcpy_host2usm(this_q, data_.get_mutable_data(), data, data_.get_size());
        });

        __ONEDAL_IF_NO_QUEUE__(data_.get_queue(), {
            detail::memcpy(detail::default_host_policy{},
                           data_.get_mutable_data(),
                           data,
                           data_.get_size());
        });
    }

    void copy_data(const dal::array<byte_t>& data) override {
        const std::int64_t dtype_size = detail::get_data_type_size(dtype_);
        const std::int64_t input_element_count = data.get_size() / dtype_size;
        ONEDAL_ASSERT(input_element_count * dtype_size == data.get_size());

        __ONEDAL_IF_QUEUE__(data.get_queue(), {
            auto input_q = data.get_queue().value();
            copy_data(input_q, data.get_data(), input_element_count, 1);
        });

        __ONEDAL_IF_NO_QUEUE__(data.get_queue(), { //
            copy_data(data.get_data(), input_element_count, 1);
        });
    }

    detail::homogen_table_iface* build_homogen() override {
        auto new_table =
            new homogen_table_impl{ row_count_, column_count_, data_, dtype_, layout_ };
        reset();
        return new_table;
    }

    detail::homogen_table_iface* build() override {
        return build_homogen();
    }

#ifdef ONEDAL_DATA_PARALLEL
    void allocate(const detail::data_parallel_policy& policy,
                  std::int64_t row_count,
                  std::int64_t column_count,
                  sycl::usm::alloc kind) override {
        if (row_count <= 0) {
            throw dal::domain_error(dal::detail::error_messages::rc_leq_zero());
        }

        if (column_count <= 0) {
            throw dal::domain_error(dal::detail::error_messages::cc_leq_zero());
        }

        const std::int64_t data_size = get_data_size(row_count, column_count, dtype_);
        data_.reset(policy.get_queue(), data_size, kind);
        row_count_ = row_count;
        column_count_ = column_count;
    }

    void copy_data(const detail::data_parallel_policy& policy,
                   const void* data,
                   std::int64_t row_count,
                   std::int64_t column_count) override {
        check_copy_data_preconditions(row_count, column_count);

        auto& input_q = policy.get_queue();
        ONEDAL_ASSERT(is_known_usm(input_q, data));

        __ONEDAL_IF_QUEUE__(data_.get_queue(), {
            auto this_q = data_.get_queue().value();

            ONEDAL_ASSERT(is_known_usm(data_));
            ONEDAL_ASSERT(is_known_usm(this_q, data));
            detail::memcpy(this_q, data_.get_mutable_data(), data, data_.get_size());
        });

        __ONEDAL_IF_NO_QUEUE__(data_.get_queue(), {
            detail::memcpy_usm2host(input_q, data_.get_mutable_data(), data, data_.get_size());
        });
    }
#endif

    template <typename T>
    void pull_rows_template(const detail::default_host_policy& policy,
                            array<T>& block,
                            const range& rows) const {
        constexpr bool preserve_mutability = true;
        homogen_pull_rows(policy,
                          get_info(),
                          data_,
                          block,
                          rows,
                          alloc_kind::non_usm,
                          preserve_mutability);
    }

    template <typename T>
    void pull_column_template(const detail::default_host_policy& policy,
                              array<T>& block,
                              std::int64_t column_index,
                              const range& rows) const {
        constexpr bool preserve_mutability = true;
        homogen_pull_column(policy,
                            get_info(),
                            data_,
                            block,
                            column_index,
                            rows,
                            alloc_kind::non_usm,
                            preserve_mutability);
    }

    template <typename T>
    void push_rows_template(const detail::default_host_policy& policy,
                            const array<T>& block,
                            const range& rows) {
        homogen_push_rows(policy, get_info(), data_, block, rows);
    }

    template <typename T>
    void push_column_template(const detail::default_host_policy& policy,
                              const array<T>& block,
                              std::int64_t column_index,
                              const range& rows) {
        homogen_push_column(policy, get_info(), data_, block, column_index, rows);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_rows_template(const detail::data_parallel_policy& policy,
                            array<T>& block,
                            const range& rows,
                            sycl::usm::alloc alloc) const {
        constexpr bool preserve_mutability = true;
        homogen_pull_rows(policy,
                          get_info(),
                          data_,
                          block,
                          rows,
                          alloc_kind_from_sycl(alloc),
                          preserve_mutability);
    }

    template <typename T>
    void pull_column_template(const detail::data_parallel_policy& policy,
                              array<T>& block,
                              std::int64_t column_index,
                              const range& rows,
                              sycl::usm::alloc alloc) const {
        constexpr bool preserve_mutability = true;
        homogen_pull_column(policy,
                            get_info(),
                            data_,
                            block,
                            column_index,
                            rows,
                            alloc_kind_from_sycl(alloc),
                            preserve_mutability);
    }

    template <typename T>
    void push_rows_template(const detail::data_parallel_policy& policy,
                            const array<T>& block,
                            const range& rows) {
        homogen_push_rows(policy, get_info(), data_, block, rows);
    }

    template <typename T>
    void push_column_template(const detail::data_parallel_policy& policy,
                              const array<T>& block,
                              std::int64_t column_index,
                              const range& rows) {
        homogen_push_column(policy, get_info(), data_, block, column_index, rows);
    }
#endif

private:
    static std::int64_t get_data_size(std::int64_t row_count,
                                      std::int64_t column_count,
                                      data_type dtype) {
        detail::check_mul_overflow(row_count, column_count);
        const std::int64_t element_count = row_count * column_count;
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);

        detail::check_mul_overflow(element_count, dtype_size);
        return element_count * dtype_size;
    }

    homogen_info get_info() const {
        return { row_count_, column_count_, dtype_, layout_ };
    }

    void check_copy_data_preconditions(std::int64_t row_count, std::int64_t column_count) {
        const std::int64_t reqired_size = get_data_size(row_count, column_count, dtype_);
        const std::int64_t allocated_size = get_data_size(row_count_, column_count_, dtype_);
        if (allocated_size < reqired_size) {
            using msg = dal::detail::error_messages;
            throw dal::invalid_argument{ msg::allocated_memory_size_is_not_enough_to_copy_data() };
        }
    }

    array<byte_t> data_;
    std::int64_t row_count_;
    std::int64_t column_count_;
    data_layout layout_;
    data_type dtype_;
};

} // namespace oneapi::dal::backend
