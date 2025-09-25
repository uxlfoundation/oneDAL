/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/misc.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/handle.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

/// Fill the internal CSR data structure of the matrix handle with the data
/// provided as 3 arrays in CSR layout.
///
/// @tparam Float   The type of elements in the CSR matrix.
///                 The `Float` type should be at least `float` or `double`.
///
/// @param queue            The SYCL* queue object.
/// @param handle           CSR matrix handle to fill with the data.
/// @param row_count        Number of rows in the matrix.
/// @param column_count     Number of columns in the matrix.
/// @param indexing         The indexing scheme used to access the data in the input arrays
///                         in CSR layout. Should be :literal:`sparse_indexing::zero_based` or
///                         :literal:`sparse_indexing::one_based`.
/// @param data             The array of values in the CSR layout.
/// @param column_indices   The array of column indices in the CSR layout.
/// @param row_offsets      The array of row offsets in the CSR layout.
/// @param dependencies     Events indicating availability of the arrays `data`, `column_indices`
///                         and `row_offsets` for reading or writing.
///
/// @return A SYCL* event that can be used to track the completion of asynchronous events
///         that were enqueued during the API call.
template <typename Float>
sycl::event set_csr_data(sycl::queue &queue,
                         sparse_matrix_handle &handle,
                         const std::int64_t row_count,
                         const std::int64_t column_count,
                         dal::sparse_indexing indexing,
                         const dal::array<const Float> &data,
                         const dal::array<const std::int64_t> &column_indices,
                         const dal::array<const std::int64_t> &row_offsets,
                         const event_vector &dependencies = {});

/// Fill the internal CSR data structure of the matrix handle with the data
/// provided as 3 memory blocks in CSR layout.
///
/// @tparam Float   The type of elements in the CSR matrix.
///                 The `Float` type should be at least `float` or `double`.
///
/// @param queue            The SYCL* queue object.
/// @param handle           CSR matrix handle to fill with the data.
/// @param row_count        Number of rows in the matrix.
/// @param column_count     Number of columns in the matrix.
/// @param indexing         The indexing scheme used to access the data in the input arrays
///                         in CSR layout. Should be :literal:`sparse_indexing::zero_based` or
///                         :literal:`sparse_indexing::one_based`.
/// @param data             The pointer to values block in the CSR layout.
/// @param column_indices   The pointer to column indices block in the CSR layout.
/// @param row_offsets      The pointer to row offsets block in the CSR layout.
/// @param dependencies     Events indicating availability of the `data`, `column_indices`
///                         and `row_offsets` for reading or writing.
///
/// @return A SYCL* event that can be used to track the completion of asynchronous events
///         that were enqueued during the API call.
template <typename Float>
sycl::event set_csr_data(sycl::queue &queue,
                         sparse_matrix_handle &handle,
                         const std::int64_t row_count,
                         const std::int64_t column_count,
                         dal::sparse_indexing indexing,
                         const Float *data,
                         const std::int64_t *column_indices,
                         const std::int64_t *row_offsets,
                         const event_vector &dependencies = {});

/// Fill the internal CSR data structure of the matrix handle with the data
/// provided as a reference to `csr_table` object.
///
/// @param queue        The SYCL* queue object.
/// @param handle       CSR matrix handle to fill with the data.
/// @param table        Input CSR data table.
/// @param dependencies Events indicating availability of the `data`, `column_indices`
///                     and `row_offsets` for reading or writing.
///
/// @return A SYCL* event that can be used to track the completion of asynchronous events
///         that were enqueued during the API call.
sycl::event set_csr_data(sycl::queue &queue,
                         sparse_matrix_handle &handle,
                         const dal::csr_table &table,
                         const event_vector &dependencies = {});

#endif // ifdef ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
