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

#include <daal/include/data_management/data/csr_numeric_table.h>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/daal_object_owner.hpp"
#include "oneapi/dal/table/backend/interop/block_info.hpp"

namespace oneapi::dal::backend::interop {

// This class shall be used only to represent immutable data on DAAL side.
// Any attempts to change the data inside objects of that class lead to undefined behavior.
template <typename Data>
class host_csr_table_adapter : public daal::data_management::CSRNumericTable {
    using base = daal::data_management::CSRNumericTable;
    using ptr_t = daal::services::SharedPtr<host_csr_table_adapter>;
    using ptr_data_t = daal::services::SharedPtr<Data>;
    using ptr_index_t = daal::services::SharedPtr<std::size_t>;
    using status_t = daal::services::Status;
    using rw_mode_t = daal::data_management::ReadWriteMode;

    template <typename T>
    using block_desc_t = daal::data_management::CSRBlockDescriptor<T>;

public:
    static ptr_t create(const csr_table& table);

private:
    status_t getSparseBlock(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<double>& block) override;
    status_t getSparseBlock(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<float>& block) override;
    status_t getSparseBlock(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<int>& block) override;

    status_t releaseSparseBlock(block_desc_t<double>& block) override;
    status_t releaseSparseBlock(block_desc_t<float>& block) override;
    status_t releaseSparseBlock(block_desc_t<int>& block) override;

    std::size_t getDataSize() override;

    void freeDataMemoryImpl() override;

    template <typename BlockData>
    status_t read_sparse_values_impl(std::size_t vector_idx,
                                     std::size_t vector_num,
                                     rw_mode_t rwflag,
                                     block_desc_t<BlockData>& block);

    host_csr_table_adapter(const csr_table& table, status_t& stat);

private:
    csr_table original_table_;
    array<size_t> one_based_column_indices_;
    array<size_t> one_based_row_offsets_;
};

} // namespace oneapi::dal::backend::interop
