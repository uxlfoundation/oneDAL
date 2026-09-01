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

#include "oneapi/dal/table/detail/table_utils.hpp"

namespace oneapi::dal::detail {
namespace v1 {

ONEDAL_EXPORT homogen_table_iface* get_homogen_table_iface_impl(table_iface* table) {
    return dynamic_cast<homogen_table_iface*>(table);
}

ONEDAL_EXPORT heterogen_table_iface* get_heterogen_table_iface_impl(table_iface* table) {
    return dynamic_cast<heterogen_table_iface*>(table);
}

ONEDAL_EXPORT csr_table_iface* get_csr_table_iface_impl(table_iface* table) {
    return dynamic_cast<csr_table_iface*>(table);
}

#ifdef ONEDAL_DATA_PARALLEL
ONEDAL_EXPORT queue_provider_iface* get_queue_provider_iface_impl(table_iface* table) {
    return dynamic_cast<queue_provider_iface*>(table);
}
#endif // ONEDAL_DATA_PARALLEL

ONEDAL_EXPORT pull_rows_iface* get_pull_rows_iface_impl(table_iface* table) {
    ONEDAL_ASSERT(table);
    return table->get_pull_rows_iface();
}

ONEDAL_EXPORT pull_column_iface* get_pull_column_iface_impl(table_iface* table) {
    ONEDAL_ASSERT(table);
    return table->get_pull_column_iface();
}

ONEDAL_EXPORT pull_csr_block_iface* get_pull_csr_block_iface_impl(table_iface* table) {
    ONEDAL_ASSERT(table);
    return table->get_pull_csr_block_iface();
}

ONEDAL_EXPORT pull_rows_iface* get_pull_rows_iface_impl(table_builder_iface* table_builder) {
    ONEDAL_ASSERT(table_builder);
    return table_builder->get_pull_rows_iface();
}

ONEDAL_EXPORT push_rows_iface* get_push_rows_iface_impl(table_builder_iface* table_builder) {
    ONEDAL_ASSERT(table_builder);
    return table_builder->get_push_rows_iface();
}

ONEDAL_EXPORT pull_column_iface* get_pull_column_iface_impl(table_builder_iface* table_builder) {
    ONEDAL_ASSERT(table_builder);
    return table_builder->get_pull_column_iface();
}

ONEDAL_EXPORT push_column_iface* get_push_column_iface_impl(table_builder_iface* table_builder) {
    ONEDAL_ASSERT(table_builder);
    return table_builder->get_push_column_iface();
}

} // namespace v1
} // namespace oneapi::dal::detail
