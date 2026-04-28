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

#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/set_csr_data.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/handle.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event set_csr_data(sycl::queue &queue,
                         sparse_matrix_handle &handle,
                         const std::int64_t row_count,
                         const std::int64_t column_count,
                         const std::int64_t nnz,
                         dal::sparse_indexing indexing,
                         const dal::array<Float> &data,
                         const dal::array<std::int64_t> &column_indices,
                         const dal::array<std::int64_t> &row_offsets,
                         const event_vector &deps) {
    ONEDAL_ASSERT(data.get_count());
    ONEDAL_ASSERT(column_indices.get_count());
    ONEDAL_ASSERT(row_offsets.get_count() == row_count + 1);
#if INTEL_MKL_VERSION >= 20250300
    return oneapi::mkl::sparse::set_csr_data(queue,
                                             dal::detail::get_impl(handle).get(),
                                             row_count,
                                             column_count,
                                             nnz,
                                             sparse_indexing_to_mkl(indexing),
                                             const_cast<std::int64_t *>(row_offsets.get_data()),
                                             const_cast<std::int64_t *>(column_indices.get_data()),
                                             const_cast<Float *>(data.get_data()),
                                             deps);
#else
    return oneapi::mkl::sparse::set_csr_data(queue,
                                             dal::detail::get_impl(handle).get(),
                                             row_count,
                                             column_count,
                                             sparse_indexing_to_mkl(indexing),
                                             const_cast<std::int64_t *>(row_offsets.get_data()),
                                             const_cast<std::int64_t *>(column_indices.get_data()),
                                             const_cast<Float *>(data.get_data()),
                                             deps);
#endif
}

template <typename Float>
sycl::event set_csr_data(sycl::queue &queue,
                         sparse_matrix_handle &handle,
                         const std::int64_t row_count,
                         const std::int64_t column_count,
                         const std::int64_t nnz,
                         dal::sparse_indexing indexing,
                         const Float *data,
                         const std::int64_t *column_indices,
                         const std::int64_t *row_offsets,
                         const event_vector &deps) {
    ONEDAL_ASSERT(data);
    ONEDAL_ASSERT(column_indices);
    ONEDAL_ASSERT(row_offsets);
#if INTEL_MKL_VERSION >= 20250300
    return oneapi::mkl::sparse::set_csr_data(queue,
                                             dal::detail::get_impl(handle).get(),
                                             row_count,
                                             column_count,
                                             nnz,
                                             sparse_indexing_to_mkl(indexing),
                                             const_cast<std::int64_t *>(row_offsets),
                                             const_cast<std::int64_t *>(column_indices),
                                             const_cast<Float *>(data),
                                             deps);
#else
    return oneapi::mkl::sparse::set_csr_data(queue,
                                             dal::detail::get_impl(handle).get(),
                                             row_count,
                                             column_count,
                                             sparse_indexing_to_mkl(indexing),
                                             const_cast<std::int64_t *>(row_offsets),
                                             const_cast<std::int64_t *>(column_indices),
                                             const_cast<Float *>(data),
                                             deps);
#endif
}

sycl::event set_csr_data(sycl::queue &queue,
                         sparse_matrix_handle &handle,
                         const dal::csr_table &table,
                         const event_vector &deps) {
    ONEDAL_ASSERT(table.has_data());
    data_type dtype = table.get_metadata().get_data_type(0);
    ONEDAL_ASSERT(dtype == data_type::float32 || dtype == data_type::float64);

    if (dtype == data_type::float32) {
        return set_csr_data(queue,
                            handle,
                            table.get_row_count(),
                            table.get_column_count(),
                            table.get_non_zero_count(),
                            table.get_indexing(),
                            table.get_data<float>(),
                            table.get_column_indices(),
                            table.get_row_offsets(),
                            deps);
    }
    else if (dtype == data_type::float64) {
        return set_csr_data(queue,
                            handle,
                            table.get_row_count(),
                            table.get_column_count(),
                            table.get_non_zero_count(),
                            table.get_indexing(),
                            table.get_data<double>(),
                            table.get_column_indices(),
                            table.get_row_offsets(),
                            deps);
    }
    return sycl::event();
}

#define INSTANTIATE(F)                                                                     \
    template ONEDAL_EXPORT sycl::event set_csr_data<F>(                                    \
        sycl::queue & queue,                                                               \
        sparse_matrix_handle & handle,                                                     \
        const std::int64_t row_count,                                                      \
        const std::int64_t column_count,                                                   \
        const std::int64_t nnz,                                                            \
        dal::sparse_indexing indexing,                                                     \
        const dal::array<F> &data,                                                         \
        const dal::array<std::int64_t> &column_indices,                                    \
        const dal::array<std::int64_t> &row_offsets,                                       \
        const event_vector &deps);                                                         \
                                                                                           \
    template ONEDAL_EXPORT sycl::event set_csr_data<F>(sycl::queue & queue,                \
                                                       sparse_matrix_handle & handle,      \
                                                       const std::int64_t row_count,       \
                                                       const std::int64_t column_count,    \
                                                       const std::int64_t nnz,             \
                                                       dal::sparse_indexing indexing,      \
                                                       const F *data,                      \
                                                       const std::int64_t *column_indices, \
                                                       const std::int64_t *row_offsets,    \
                                                       const event_vector &deps);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::backend::primitives
