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

#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/detail/table_kinds.hpp"
#include "oneapi/dal/table/detail/table_utils.hpp"
#include "oneapi/dal/table/backend/csr_table_impl.hpp"

namespace oneapi::dal {
namespace v1 {

static std::shared_ptr<detail::csr_table_iface> get_csr_iface(const table& other) {
    if (const auto csr_iface = detail::get_csr_table_iface(other)) {
        return csr_iface;
    }
    return std::make_shared<backend::csr_table_impl>();
}

std::int64_t csr_table::kind() {
    return detail::get_csr_table_kind();
}

std::int64_t csr_table::get_non_zero_count() const {
    const auto& impl = detail::cast_impl<const detail::csr_table_iface>(*this);
    return impl.get_non_zero_count();
}

sparse_indexing csr_table::get_indexing() const {
    const auto& impl = detail::cast_impl<const detail::csr_table_iface>(*this);
    return impl.get_indexing();
}

csr_table::csr_table() : csr_table(new backend::csr_table_impl{}) {}

csr_table::csr_table(const table& other) : csr_table(get_csr_iface(other)) {}

const void* csr_table::get_data() const {
    const auto& impl = detail::cast_impl<const detail::csr_table_iface>(*this);
    return impl.get_data().get_data();
}

const std::int64_t* csr_table::get_column_indices() const {
    const auto& impl = detail::cast_impl<const detail::csr_table_iface>(*this);
    return impl.get_column_indices().get_data();
}

const std::int64_t* csr_table::get_row_offsets() const {
    const auto& impl = detail::cast_impl<const detail::csr_table_iface>(*this);
    return impl.get_row_offsets().get_data();
}

#ifdef ONEDAL_DATA_PARALLEL
std::int64_t csr_table::get_non_zero_count(sycl::queue& queue,
                                           const std::int64_t row_count,
                                           const std::int64_t* row_offsets,
                                           const std::vector<sycl::event>& dependencies) {
    return backend::csr_table_impl::get_non_zero_count(queue, row_count, row_offsets, dependencies);
}
#endif

void csr_table::init_impl(const dal::array<byte_t>& data,
                          const dal::array<std::int64_t>& column_indices,
                          const dal::array<std::int64_t>& row_offsets,
                          std::int64_t column_count,
                          const data_type& dtype,
                          sparse_indexing indexing) {
#ifdef ONEDAL_DATA_PARALLEL
    if (data.get_queue().has_value()) {
        if (column_indices.get_queue() != data.get_queue() ||
            row_offsets.get_queue() != data.get_queue()) {
            throw invalid_argument{ dal::detail::error_messages::queues_of_arrays_do_not_match() };
        }
        table::init_impl(
            new backend::csr_table_impl{ detail::data_parallel_policy{ data.get_queue().value() },
                                         data,
                                         column_indices,
                                         row_offsets,
                                         column_count,
                                         dtype,
                                         indexing,
                                         {} });
        return;
    }
#endif
    table::init_impl(new backend::csr_table_impl{ data,
                                                  column_indices,
                                                  row_offsets,
                                                  column_count,
                                                  dtype,
                                                  indexing });
}

#ifdef ONEDAL_DATA_PARALLEL
void csr_table::init_impl(const detail::data_parallel_policy& policy,
                          const dal::array<byte_t>& data,
                          const dal::array<std::int64_t>& column_indices,
                          const dal::array<std::int64_t>& row_offsets,
                          std::int64_t column_count,
                          const data_type& dtype,
                          sparse_indexing indexing,
                          const std::vector<sycl::event>& dependencies) {
    table::init_impl(new backend::csr_table_impl{ policy,
                                                  data,
                                                  column_indices,
                                                  row_offsets,
                                                  column_count,
                                                  dtype,
                                                  indexing,
                                                  dependencies });
}
#endif

} // namespace v1
} // namespace oneapi::dal
