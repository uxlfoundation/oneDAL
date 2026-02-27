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

#include "oneapi/dal/table/backend/interop/host_soa_table_adapter.hpp"
#include "oneapi/dal/table/backend/interop/common.hpp"

namespace oneapi::dal::backend::interop {

namespace daal_dm = daal::data_management;

template <typename Data>
auto host_soa_table_adapter::create(const homogen_table& table) -> ptr_t {
    status_t internal_stat;
    auto result = ptr_t{ new host_soa_table_adapter(table, internal_stat, Data{}) };
    status_to_exception(internal_stat);
    return result;
}

// TODO: change 'equal' flags across this constructor after implemeting the method
// of features equality defining for table_metadata class.
template <typename Data>
host_soa_table_adapter::host_soa_table_adapter(const homogen_table& table,
                                               status_t& stat,
                                               Data dummy)
        : base(dal::detail::integral_cast<std::size_t>(table.get_column_count()),
               dal::detail::integral_cast<std::size_t>(table.get_row_count()),
               daal_dm::DictionaryIface::equal),
          original_table_(table) {
    if (!stat.ok()) {
        return;
    }
    else if (!table.has_data()) {
        stat.add(daal::services::ErrorIncorrectParameter);
        return;
    }

    if (table.get_data_layout() != data_layout::column_major) {
        stat.add(daal::services::ErrorMethodNotImplemented);
        return;
    }

    // The following const_cast is safe only when this class is used for read-only
    // operations. Use on write leads to undefined behaviour.
    const auto original_data = const_cast<Data*>(table.get_data<Data>());

    const std::size_t column_count =
        dal::detail::integral_cast<std::size_t>(table.get_column_count());
    const std::size_t row_count = dal::detail::integral_cast<std::size_t>(table.get_row_count());

    for (std::size_t i = 0; i < column_count; i++) {
        stat |= base::setArray<Data>(&original_data[i * row_count], i);

        if (!stat.ok()) {
            return;
        }
    }

    this->_memStatus = daal_dm::NumericTableIface::userAllocated;
    this->_layout = daal_dm::NumericTableIface::soa;

    convert_feature_information_to_daal(original_table_.get_metadata(),
                                        *this->getDictionarySharedPtr());
}

auto host_soa_table_adapter::getBlockOfRows(std::size_t vector_idx,
                                            std::size_t vector_num,
                                            rw_mode_t rwflag,
                                            block_desc_t<double>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

auto host_soa_table_adapter::getBlockOfRows(std::size_t vector_idx,
                                            std::size_t vector_num,
                                            rw_mode_t rwflag,
                                            block_desc_t<float>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

auto host_soa_table_adapter::getBlockOfRows(std::size_t vector_idx,
                                            std::size_t vector_num,
                                            rw_mode_t rwflag,
                                            block_desc_t<int>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

auto host_soa_table_adapter::getBlockOfColumnValues(std::size_t feature_idx,
                                                    std::size_t vector_idx,
                                                    std::size_t value_num,
                                                    rw_mode_t rwflag,
                                                    block_desc_t<double>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

auto host_soa_table_adapter::getBlockOfColumnValues(std::size_t feature_idx,
                                                    std::size_t vector_idx,
                                                    std::size_t value_num,
                                                    rw_mode_t rwflag,
                                                    block_desc_t<float>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

auto host_soa_table_adapter::getBlockOfColumnValues(std::size_t feature_idx,
                                                    std::size_t vector_idx,
                                                    std::size_t value_num,
                                                    rw_mode_t rwflag,
                                                    block_desc_t<int>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

auto host_soa_table_adapter::releaseBlockOfRows(block_desc_t<double>& block) -> status_t {
    block.reset();
    return status_t();
}

auto host_soa_table_adapter::releaseBlockOfRows(block_desc_t<float>& block) -> status_t {
    block.reset();
    return status_t();
}

auto host_soa_table_adapter::releaseBlockOfRows(block_desc_t<int>& block) -> status_t {
    block.reset();
    return status_t();
}

auto host_soa_table_adapter::releaseBlockOfColumnValues(block_desc_t<double>& block) -> status_t {
    block.reset();
    return status_t();
}

auto host_soa_table_adapter::releaseBlockOfColumnValues(block_desc_t<float>& block) -> status_t {
    block.reset();
    return status_t();
}

auto host_soa_table_adapter::releaseBlockOfColumnValues(block_desc_t<int>& block) -> status_t {
    block.reset();
    return status_t();
}

auto host_soa_table_adapter::assign(float) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto host_soa_table_adapter::assign(double) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto host_soa_table_adapter::assign(int) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto host_soa_table_adapter::allocateDataMemoryImpl(daal::MemType) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto host_soa_table_adapter::setNumberOfColumnsImpl(std::size_t) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

int host_soa_table_adapter::getSerializationTag() const {
    ONEDAL_ASSERT(!"host_soa_table_adapter: getSerializationTag() is not implemented");
    return -1;
}

auto host_soa_table_adapter::serializeImpl(daal_dm::InputDataArchive* arch) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto host_soa_table_adapter::deserializeImpl(const daal_dm::OutputDataArchive* arch) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

void host_soa_table_adapter::freeDataMemoryImpl() {
    base::freeDataMemoryImpl();
    original_table_ = homogen_table{};
}

template <typename BlockData>
auto host_soa_table_adapter::read_rows_impl(std::size_t vector_idx,
                                            std::size_t vector_num,
                                            rw_mode_t rwflag,
                                            block_desc_t<BlockData>& block) -> status_t {
    if (rwflag != daal_dm::readOnly) {
        ONEDAL_ASSERT(!"Data is accessible in read-only mode by design");
        return daal::services::ErrorMethodNotImplemented;
    }

    return base::getBlockOfRows(vector_idx, vector_num, rwflag, block);
}

template <typename BlockData>
auto host_soa_table_adapter::read_column_values_impl(std::size_t feature_idx,
                                                     std::size_t vector_idx,
                                                     std::size_t value_num,
                                                     rw_mode_t rwflag,
                                                     block_desc_t<BlockData>& block) -> status_t {
    if (rwflag != daal_dm::readOnly) {
        ONEDAL_ASSERT(!"Data is accessible in read-only mode by design");
        return daal::services::ErrorMethodNotImplemented;
    }

    return base::getBlockOfColumnValues(feature_idx, vector_idx, value_num, rwflag, block);
}

bool host_soa_table_adapter::check_row_indexes_in_range(const block_info& info) const {
    const std::int64_t row_count = original_table_.get_row_count();
    return info.row_begin_index < row_count && info.row_end_index <= row_count;
}

bool host_soa_table_adapter::check_column_index_in_range(const block_info& info) const {
    const std::int64_t column_count = original_table_.get_column_count();
    return info.single_column_requested && info.column_index < column_count;
}

template auto host_soa_table_adapter::create<std::int32_t>(const homogen_table&) -> ptr_t;
template auto host_soa_table_adapter::create<float>(const homogen_table&) -> ptr_t;
template auto host_soa_table_adapter::create<double>(const homogen_table&) -> ptr_t;

} // namespace oneapi::dal::backend::interop
