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

#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/backend/csr_kernels.hpp"
#include "oneapi/dal/table/backend/csr_table_impl.hpp"
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend {

class csr_table_builder_impl : public detail::csr_table_builder_template<csr_table_builder_impl> {
public:
#ifdef ONEDAL_DATA_PARALLEL
    csr_table_builder_impl() : dependencies_(nullptr) {
        reset();
    }
#else
    csr_table_builder_impl() {
        reset();
    }
#endif

    void reset() {
        data_.reset();
        column_indices_.reset();
        row_offsets_.reset();
        row_count_ = 0;
        column_count_ = 0;
        element_count_ = 0;
        dtype_ = data_type::float32;
        indexing_ = sparse_indexing::one_based;
#ifdef ONEDAL_DATA_PARALLEL
        dependencies_ = nullptr;
#endif
    }

    void reset(const dal::array<byte_t>& data,
               const dal::array<std::int64_t>& column_indices,
               const dal::array<std::int64_t>& row_offsets,
               std::int64_t row_count,
               std::int64_t column_count,
               sparse_indexing indexing) override {
        get_data_size(column_indices.get_count(), dtype_);
        data_ = data;
        column_indices_ = column_indices;
        row_offsets_ = row_offsets;
        row_count_ = row_count;
        column_count_ = column_count;
        dtype_ = data_type::float32;
        indexing_ = indexing;
    }

    void set_data_type(data_type dt) override {
        dtype_ = dt;
        data_.reset();
        column_indices_.reset();
        row_offsets_.reset();
        row_count_ = 0;
        column_count_ = 0;
        element_count_ = 0;
        indexing_ = sparse_indexing::one_based;
    }

    detail::csr_table_iface* build_csr() override {
        csr_table_impl* new_table = nullptr;
#ifdef ONEDAL_DATA_PARALLEL
        if (dependencies_ && data_.get_queue().has_value())
            new_table = new csr_table_impl{ detail::data_parallel_policy(data_.get_queue().value()),
                                            data_,
                                            column_indices_,
                                            row_offsets_,
                                            column_count_,
                                            dtype_,
                                            indexing_,
                                            *dependencies_ };
        else
#endif
            new_table = new csr_table_impl{ data_,         column_indices_, row_offsets_,
                                            column_count_, dtype_,          indexing_ };
        reset();
        return new_table;
    }

    detail::csr_table_iface* build() override {
        return build_csr();
    }

    template <typename T>
    void pull_csr_block_template(const detail::default_host_policy& policy,
                                 array<T>& data,
                                 array<std::int64_t>& column_indices,
                                 array<std::int64_t>& row_offsets,
                                 const sparse_indexing& indexing,
                                 const range& row_range) const {
        constexpr bool preserve_mutability = true;

        block_info block_info{ row_range.start_idx,
                               row_range.get_element_count(row_count_),
                               indexing };

        csr_pull_block(policy,
                       get_info(),
                       block_info,
                       data_,
                       column_indices_,
                       row_offsets_,
                       data,
                       column_indices,
                       row_offsets,
                       alloc_kind::non_usm,
                       preserve_mutability);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void reset(const dal::array<byte_t>& data,
               const dal::array<std::int64_t>& column_indices,
               const dal::array<std::int64_t>& row_offsets,
               std::int64_t row_count,
               std::int64_t column_count,
               sparse_indexing indexing,
               const std::vector<sycl::event>& dependencies) override {
        reset(data, column_indices, row_offsets, row_count, column_count, indexing);
        dependencies_ = &dependencies;
    }

    template <typename T>
    void pull_csr_block_template(const detail::data_parallel_policy& policy,
                                 dal::array<T>& data,
                                 dal::array<std::int64_t>& column_indices,
                                 dal::array<std::int64_t>& row_offsets,
                                 const sparse_indexing& indexing,
                                 const range& row_range,
                                 sycl::usm::alloc alloc) const {
        constexpr bool preserve_mutability = true;

        block_info block_info{ row_range.start_idx,
                               row_range.get_element_count(row_count_),
                               indexing };

        csr_pull_block(policy,
                       get_info(),
                       block_info,
                       data_,
                       column_indices_,
                       row_offsets_,
                       data,
                       column_indices,
                       row_offsets,
                       alloc_kind_from_sycl(alloc),
                       preserve_mutability);
    }
#endif

private:
    static std::int64_t get_data_size(std::int64_t element_count, data_type dtype) {
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);

        detail::check_mul_overflow(element_count, dtype_size);
        return element_count * dtype_size;
    }

    csr_info get_info() const {
        return { dtype_, row_count_, column_count_, element_count_, indexing_ };
    }

    array<byte_t> data_;
    array<std::int64_t> column_indices_;
    array<std::int64_t> row_offsets_;
    std::int64_t row_count_;
    std::int64_t column_count_;
    std::int64_t element_count_;
    data_type dtype_;
    sparse_indexing indexing_;
#ifdef ONEDAL_DATA_PARALLEL
    const std::vector<sycl::event>* dependencies_;
#endif
};

} // namespace oneapi::dal::backend
