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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/backend/common_kernels.hpp"
#include "oneapi/dal/table/backend/csr_kernels.hpp"
#include "oneapi/dal/table/detail/csr_access_iface.hpp"
#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::backend {

class csr_table_impl : public detail::csr_table_template<csr_table_impl>,
                       public ONEDAL_SERIALIZABLE(csr_table_id) {
public:
    csr_table_impl()
            : col_count_(0),
              row_count_(0),
              layout_(data_layout::row_major),
              indexing_(sparse_indexing::one_based) {}

    csr_table_impl(const array<byte_t>& data,
                   const array<std::int64_t>& column_indices,
                   const array<std::int64_t>& row_offsets,
                   std::int64_t column_count,
                   data_type dtype,
                   sparse_indexing indexing)
            : meta_(create_metadata(column_count, dtype, data.get_alloc_kind())),
              data_(data),
              column_indices_(column_indices),
              row_offsets_(row_offsets),
              col_count_(column_count),
              row_count_(row_offsets.get_count() - 1),
              layout_(data_layout::row_major),
              indexing_(indexing) {
        using error_msg = dal::detail::error_messages;

        const std::int64_t element_count = column_indices.get_count();
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);

        detail::check_mul_overflow(element_count, dtype_size);
        if (data.get_count() != element_count * dtype_size) {
            throw dal::domain_error(error_msg::invalid_data_block_size());
        }

        const std::int64_t* const row_offsets_ptr = row_offsets.get_data();
        if (!std::is_sorted(row_offsets_ptr, row_offsets_ptr + row_offsets.get_count())) {
            throw dal::domain_error(error_msg::row_offsets_not_ascending());
        }

        const std::int64_t min_index = (indexing == sparse_indexing::zero_based) ? 0 : 1;
        const std::int64_t max_index =
            (indexing == sparse_indexing::zero_based) ? column_count - 1 : column_count;

        const auto status = backend::check_bounds(column_indices, min_index, max_index);
        if (status == backend::out_of_bound_type::less_than_min) {
            throw dal::domain_error(error_msg::column_indices_lt_min_value());
        }
        if (status == backend::out_of_bound_type::greater_than_max) {
            throw dal::domain_error(error_msg::column_indices_gt_max_value());
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    csr_table_impl(const detail::data_parallel_policy& policy,
                   const array<byte_t>& data,
                   const array<std::int64_t>& column_indices,
                   const array<std::int64_t>& row_offsets,
                   std::int64_t column_count,
                   data_type dtype,
                   sparse_indexing indexing,
                   const std::vector<sycl::event>& dependencies)
            : meta_(create_metadata(column_count, dtype, data.get_alloc_kind())),
              data_(data),
              column_indices_(column_indices),
              row_offsets_(row_offsets),
              col_count_(column_count),
              row_count_(row_offsets.get_count() - 1),
              layout_(data_layout::row_major),
              indexing_(indexing) {
        using error_msg = dal::detail::error_messages;

        const std::int64_t element_count = column_indices.get_count();
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);

        detail::check_mul_overflow(element_count, dtype_size);
        if (data.get_count() != element_count * dtype_size) {
            throw dal::domain_error(error_msg::invalid_data_block_size());
        }

        if (!backend::is_sorted(row_offsets, dependencies)) {
            throw dal::domain_error(error_msg::row_offsets_not_ascending());
        }

        const std::int64_t min_index = (indexing == sparse_indexing::zero_based) ? 0 : 1;
        const std::int64_t max_index =
            (indexing == sparse_indexing::zero_based) ? column_count - 1 : column_count;

        const auto status =
            backend::check_bounds(column_indices, min_index, max_index, dependencies);
        if (status == backend::out_of_bound_type::less_than_min) {
            throw dal::domain_error(error_msg::column_indices_lt_min_value());
        }
        if (status == backend::out_of_bound_type::greater_than_max) {
            throw dal::domain_error(error_msg::column_indices_gt_max_value());
        }
    }
#endif

    std::int64_t get_column_count() const override {
        return col_count_;
    }

    std::int64_t get_row_count() const override {
        return row_count_;
    }

    /// The number of non-zero elements in the table.
    std::int64_t get_non_zero_count() const override {
        return csr_get_non_zero_count(row_offsets_);
    }

#ifdef ONEDAL_DATA_PARALLEL
    static std::int64_t get_non_zero_count(sycl::queue& queue,
                                           const std::int64_t row_count,
                                           const std::int64_t* row_offsets,
                                           const std::vector<sycl::event>& dependencies) {
        return csr_get_non_zero_count(queue, row_count, row_offsets, dependencies);
    }
#endif

    sparse_indexing get_indexing() const override {
        return indexing_;
    }

    const table_metadata& get_metadata() const override {
        return meta_;
    }

    std::int64_t get_kind() const override {
        return 10;
    }

    array<byte_t> get_data() const override {
        return data_;
    }

    array<std::int64_t> get_column_indices() const override {
        return column_indices_;
    }

    array<std::int64_t> get_row_offsets() const override {
        return row_offsets_;
    }

    data_layout get_data_layout() const override {
        return layout_;
    }

    template <typename T>
    void pull_csr_block_template(const detail::default_host_policy& policy,
                                 dal::array<T>& data,
                                 dal::array<std::int64_t>& column_indices,
                                 dal::array<std::int64_t>& row_offsets,
                                 const sparse_indexing& indexing,
                                 const range& rows) const {
        csr_info origin_info{ meta_.get_data_type(0),
                              row_count_,
                              col_count_,
                              get_non_zero_count(),
                              indexing_ };

        // Overflow is checked here
        check_block_row_range(rows);

        block_info block_info{ rows.start_idx, rows.get_element_count(row_count_), indexing };

        csr_pull_block(policy,
                       origin_info,
                       block_info,
                       data_,
                       column_indices_,
                       row_offsets_,
                       data,
                       column_indices,
                       row_offsets,
                       alloc_kind::non_usm);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_csr_block_template(const detail::data_parallel_policy& policy,
                                 dal::array<T>& data,
                                 dal::array<std::int64_t>& column_indices,
                                 dal::array<std::int64_t>& row_offsets,
                                 const sparse_indexing& indexing,
                                 const range& rows,
                                 sycl::usm::alloc alloc) const {
        csr_info origin_info{ meta_.get_data_type(0),
                              row_count_,
                              col_count_,
                              get_non_zero_count(),
                              indexing_ };

        // Overflow is checked here
        check_block_row_range(rows);

        block_info block_info{ rows.start_idx, rows.get_element_count(row_count_), indexing };

        csr_pull_block(policy,
                       origin_info,
                       block_info,
                       data_,
                       column_indices_,
                       row_offsets_,
                       data,
                       column_indices,
                       row_offsets,
                       alloc_kind_from_sycl(alloc));
    }
#endif

    void serialize(detail::output_archive& ar) const override {
        ar(meta_, data_, column_indices_, row_offsets_, col_count_, row_count_, layout_, indexing_);
    }

    void deserialize(detail::input_archive& ar) override {
        ar(meta_, data_, column_indices_, row_offsets_, col_count_, row_count_, layout_, indexing_);
        // The allocation kind is not persisted in the archive; re-establish it
        // from the deserialized data, which was materialized according to the
        // deserialization context (queue/policy). The deserialized feature and
        // data types are preserved as-is.
        if (data_.get_count() > 0) {
            meta_ = table_metadata{ meta_.get_data_types(),
                                    meta_.get_feature_types(),
                                    data_.get_alloc_kind() };
        }
    }

private:
    void check_block_row_range(const range& rows) const {
        const std::int64_t range_row_count = rows.get_element_count(row_count_);
        detail::check_sum_overflow(rows.start_idx, range_row_count);
        if (rows.start_idx + range_row_count > row_count_) {
            throw range_error{ detail::error_messages::invalid_range_of_rows() };
        }
    }

    table_metadata meta_;
    array<byte_t> data_;
    array<std::int64_t> column_indices_;
    array<std::int64_t> row_offsets_;
    std::int64_t col_count_;
    std::int64_t row_count_;
    data_layout layout_;
    sparse_indexing indexing_;
};

} // namespace oneapi::dal::backend
