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

#include "oneapi/dal/table/backend/interop/host_csr_table_adapter.hpp"

namespace oneapi::dal::backend::interop {

static daal::data_management::features::FeatureType get_daal_feature_type(feature_type t) {
    namespace daal_dm = daal::data_management;

    switch (t) {
        case feature_type::nominal: return daal_dm::features::DAAL_CATEGORICAL;
        case feature_type::ordinal: return daal_dm::features::DAAL_ORDINAL;
        case feature_type::interval: return daal_dm::features::DAAL_CONTINUOUS;
        case feature_type::ratio: return daal_dm::features::DAAL_CONTINUOUS;
        default: throw dal::internal_error(detail::error_messages::unsupported_feature_type());
    }
}

static void convert_feature_information_to_daal(
    const table_metadata& src,
    daal::data_management::NumericTableDictionary& dst) {
    ONEDAL_ASSERT(std::size_t(src.get_feature_count()) == dst.getNumberOfFeatures());
    for (std::int64_t i = 0; i < src.get_feature_count(); i++) {
        auto& daal_feature = dst[i];
        daal_feature.featureType = get_daal_feature_type(src.get_feature_type(i));
    }
}

template <typename Data>
auto host_csr_table_adapter<Data>::create(const csr_table& table) -> host_csr_table_adapter<Data>* {
    status_t internal_stat;
    auto result = new host_csr_table_adapter(table, internal_stat);
    if (!internal_stat.ok()) {
        delete result;
        result = nullptr;
    }
    status_to_exception(internal_stat);
    return result;
}

template <typename Data>
auto host_csr_table_adapter<Data>::getSparseBlock(std::size_t vector_idx,
                                                  std::size_t vector_num,
                                                  rw_mode_t rwflag,
                                                  block_desc_t<double>& block) -> status_t {
    return read_sparse_values_impl(vector_idx, vector_num, rwflag, block);
}

template <typename Data>
auto host_csr_table_adapter<Data>::getSparseBlock(std::size_t vector_idx,
                                                  std::size_t vector_num,
                                                  rw_mode_t rwflag,
                                                  block_desc_t<float>& block) -> status_t {
    return read_sparse_values_impl(vector_idx, vector_num, rwflag, block);
}

template <typename Data>
auto host_csr_table_adapter<Data>::getSparseBlock(std::size_t vector_idx,
                                                  std::size_t vector_num,
                                                  rw_mode_t rwflag,
                                                  block_desc_t<int>& block) -> status_t {
    return read_sparse_values_impl(vector_idx, vector_num, rwflag, block);
}

template <typename Data>
auto host_csr_table_adapter<Data>::releaseSparseBlock(block_desc_t<double>& block) -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto host_csr_table_adapter<Data>::releaseSparseBlock(block_desc_t<float>& block) -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto host_csr_table_adapter<Data>::releaseSparseBlock(block_desc_t<int>& block) -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
std::size_t host_csr_table_adapter<Data>::getDataSize() {
    return base::getDataSize();
}

template <typename Data>
void host_csr_table_adapter<Data>::freeDataMemoryImpl() {
    base::freeDataMemoryImpl();
    original_table_ = csr_table{};
}

template <typename Data>
template <typename BlockData>
auto host_csr_table_adapter<Data>::read_sparse_values_impl(std::size_t vector_idx,
                                                           std::size_t vector_num,
                                                           rw_mode_t rwflag,
                                                           block_desc_t<BlockData>& block)
    -> status_t {
    if (rwflag != daal::data_management::readOnly) {
        return daal::services::ErrorMethodNotImplemented;
    }

    return base::getSparseBlock(vector_idx, vector_num, rwflag, block);
}

template <typename Data>
host_csr_table_adapter<Data>::host_csr_table_adapter(const csr_table& table, status_t& stat)
        // The following const_cast is safe only when this class is used for read-only
        // operations. Use on write leads to undefined behaviour.
        : base(ptr_data_t{ const_cast<Data*>(table.get_data<Data>()), daal_object_owner(table) },
               ptr_index_t(),
               ptr_index_t(),
               table.get_column_count(),
               table.get_row_count(),
               daal::data_management::CSRNumericTableIface::CSRIndexing::oneBased,
               stat) {
    if (!stat.ok()) {
        return;
    }
    if (!table.has_data()) {
        stat.add(daal::services::ErrorIncorrectParameter);
        return;
    }

    const std::int64_t column_indices_count = table.get_non_zero_count();
    const std::int64_t row_count = table.get_row_count();
    size_t* column_indices =
        const_cast<std::size_t*>(reinterpret_cast<const std::size_t*>(table.get_column_indices()));
    size_t* row_offsets =
        const_cast<std::size_t*>(reinterpret_cast<const std::size_t*>(table.get_row_offsets()));

    // Convert zero-based indices to one-based if needed.
    // Because DAAL tables support only one-based indexing
    if (table.get_indexing() == sparse_indexing::zero_based) {
        one_based_column_indices_.reset(column_indices_count);
        one_based_row_offsets_.reset(row_count + 1);
        size_t* one_based_column_indices_ptr = one_based_column_indices_.get_mutable_data();
        size_t* one_based_row_offsets_ptr = one_based_row_offsets_.get_mutable_data();
        for (std::int64_t i = 0; i < column_indices_count; i++) {
            one_based_column_indices_ptr[i] = column_indices[i] + 1;
        }
        column_indices = one_based_column_indices_ptr;
        for (std::int64_t i = 0; i <= row_count; i++) {
            one_based_row_offsets_ptr[i] = row_offsets[i] + 1;
        }
        row_offsets = one_based_row_offsets_ptr;
    }

    this->_status |= setArrays<Data>(
        ptr_data_t{ const_cast<Data*>(table.get_data<Data>()), daal_object_owner(table) },
        ptr_index_t{ column_indices, daal_object_owner(table) },
        ptr_index_t{ row_offsets, daal_object_owner(table) });
    if (!this->_status.ok())
        return;

    _defaultFeature.setType<Data>();
    this->_status |= this->_ddict->setAllFeatures(_defaultFeature);
    if (!this->_status.ok())
        return;

    original_table_ = table;

    this->_memStatus = daal::data_management::NumericTableIface::userAllocated;
    this->_layout = daal::data_management::NumericTableIface::csrArray;

    auto& daal_dictionary = *this->getDictionarySharedPtr();
    convert_feature_information_to_daal(original_table_.get_metadata(), daal_dictionary);
}

template class host_csr_table_adapter<std::int32_t>;
template class host_csr_table_adapter<float>;
template class host_csr_table_adapter<double>;

} // namespace oneapi::dal::backend::interop
