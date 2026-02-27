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

#include <daal/include/data_management/data/soa_numeric_table.h>

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/daal_object_owner.hpp"
#include "oneapi/dal/table/backend/interop/block_info.hpp"

namespace oneapi::dal::backend::interop {

// This class shall be used only to represent immutable data on DAAL side. Any
// attempts to change the data inside objects of that class lead to undefined
// behavior.
class host_soa_table_adapter : public daal::data_management::SOANumericTable {
    using base = daal::data_management::SOANumericTable;
    using status_t = daal::services::Status;
    using rw_mode_t = daal::data_management::ReadWriteMode;
    using ptr_t = daal::services::SharedPtr<host_soa_table_adapter>;

    template <typename T>
    using block_desc_t = daal::data_management::BlockDescriptor<T>;

public:
    template <typename Data>
    static ptr_t create(const homogen_table& table);

private:
    template <typename Data>
    explicit host_soa_table_adapter(const homogen_table& table, status_t& stat, Data dummy);

    status_t getBlockOfRows(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<double>& block) override;

    status_t getBlockOfRows(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<float>& block) override;

    status_t getBlockOfRows(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<int>& block) override;

    status_t getBlockOfColumnValues(std::size_t feature_idx,
                                    std::size_t vector_idx,
                                    std::size_t value_num,
                                    rw_mode_t rwflag,
                                    block_desc_t<double>& block) override;

    status_t getBlockOfColumnValues(std::size_t feature_idx,
                                    std::size_t vector_idx,
                                    std::size_t value_num,
                                    rw_mode_t rwflag,
                                    block_desc_t<float>& block) override;

    status_t getBlockOfColumnValues(std::size_t feature_idx,
                                    std::size_t vector_idx,
                                    std::size_t value_num,
                                    rw_mode_t rwflag,
                                    block_desc_t<int>& block) override;

    status_t releaseBlockOfRows(block_desc_t<double>& block) override;
    status_t releaseBlockOfRows(block_desc_t<float>& block) override;
    status_t releaseBlockOfRows(block_desc_t<int>& block) override;

    status_t releaseBlockOfColumnValues(block_desc_t<double>& block) override;
    status_t releaseBlockOfColumnValues(block_desc_t<float>& block) override;
    status_t releaseBlockOfColumnValues(block_desc_t<int>& block) override;

    status_t assign(float value) override;
    status_t assign(double value) override;
    status_t assign(int value) override;

    int getSerializationTag() const override;
    status_t serializeImpl(daal::data_management::InputDataArchive* arch) override;
    status_t deserializeImpl(const daal::data_management::OutputDataArchive* arch) override;

    status_t allocateDataMemoryImpl(daal::MemType) override;
    status_t setNumberOfColumnsImpl(std::size_t) override;
    void freeDataMemoryImpl() override;

    template <typename BlockData>
    status_t read_rows_impl(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<BlockData>& block);

    template <typename BlockData>
    status_t read_column_values_impl(std::size_t feature_idx,
                                     std::size_t vector_idx,
                                     std::size_t value_num,
                                     rw_mode_t rwflag,
                                     block_desc_t<BlockData>& block);

    bool check_row_indexes_in_range(const block_info& info) const;
    bool check_column_index_in_range(const block_info& info) const;

    homogen_table original_table_;
};

} // namespace oneapi::dal::backend::interop
