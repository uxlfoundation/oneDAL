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
#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::backend {

class empty_table_impl : public detail::generic_table_template<empty_table_impl>,
                         public ONEDAL_SERIALIZABLE(empty_table_id)
#ifdef ONEDAL_DATA_PARALLEL
        ,
                         public detail::queue_provider_iface
#endif
{
public:
    static constexpr std::int64_t pure_empty_table_kind = 0;

    std::int64_t get_column_count() const override {
        return 0;
    }

    std::int64_t get_row_count() const override {
        return 0;
    }

    std::int64_t get_kind() const override {
        return pure_empty_table_kind;
    }

    data_layout get_data_layout() const override {
        return data_layout::unknown;
    }

    const table_metadata& get_metadata() const override {
        static table_metadata metadata;
        return metadata;
    }

    template <typename T>
    void pull_rows_template(const detail::default_host_policy& policy,
                            array<T>& block,
                            const range& rows) const {}

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_rows_template(const detail::data_parallel_policy& policy,
                            array<T>& block,
                            const range& rows,
                            sycl::usm::alloc alloc) const {}
#endif

    template <typename T>
    void pull_column_template(const detail::default_host_policy& policy,
                              array<T>& block,
                              std::int64_t column_index,
                              const range& rows) const {}

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_column_template(const detail::data_parallel_policy& policy,
                              array<T>& block,
                              std::int64_t column_index,
                              const range& rows,
                              sycl::usm::alloc alloc) const {}
#endif

    template <typename T>
    void pull_csr_block_template(const detail::default_host_policy& policy,
                                 dal::array<T>& data,
                                 dal::array<std::int64_t>& column_indices,
                                 dal::array<std::int64_t>& row_indices,
                                 const sparse_indexing& indexing,
                                 const range& row_range) const {}

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    void pull_csr_block_template(const detail::data_parallel_policy& policy,
                                 dal::array<T>& data,
                                 dal::array<std::int64_t>& column_indices,
                                 dal::array<std::int64_t>& row_indices,
                                 const sparse_indexing& indexing,
                                 const range& row_range,
                                 sycl::usm::alloc alloc) const {}
#endif

#ifdef ONEDAL_DATA_PARALLEL
    std::optional<sycl::queue> get_queue() const override {
        return {};
    }
#endif

    void serialize(detail::output_archive& ar) const override {
        // Nothing to serialize
    }

    void deserialize(detail::input_archive& ar) override {
        // Nothing to deserialize
    }
};

} // namespace oneapi::dal::backend
