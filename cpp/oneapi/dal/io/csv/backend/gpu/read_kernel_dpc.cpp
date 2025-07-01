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

#ifndef ONEDAL_DATA_CONVERSION

#define ONEDAL_DATA_CONVERSION
#include "daal/include/data_management/data_source/csv_feature_manager.h"
#include "daal/include/data_management/data_source/file_data_source.h"
#undef ONEDAL_DATA_CONVERSION

#endif

#include <fstream>
#include <sstream>
#include <vector>

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/io/csv/backend/gpu/read_kernel.hpp"
#include "oneapi/dal/io/csv/backend/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::csv::backend {

namespace interop = dal::backend::interop;
namespace daal_dm = daal::data_management;

template <typename Float>
struct read_kernel_gpu<table, Float> {
    table operator()(const dal::backend::context_gpu& ctx,
                     const detail::data_source_base& ds,
                     const read_args<table>& args) const {
        auto& queue = ctx.get_queue();

        daal_dm::CsvDataSourceOptions csv_options(daal_dm::operator|(
            daal_dm::operator|(daal_dm::CsvDataSourceOptions::allocateNumericTable,
                               daal_dm::CsvDataSourceOptions::createDictionaryFromContext),
            (ds.get_parse_header() ? daal_dm::CsvDataSourceOptions::parseHeader
                                   : daal_dm::CsvDataSourceOptions::byDefault)));

        daal_dm::FileDataSource<daal_dm::CSVFeatureManager> daal_data_source(
            ds.get_file_name().c_str(),
            csv_options);
        interop::status_to_exception(daal_data_source.status());

        daal_data_source.getFeatureManager().setDelimiter(ds.get_delimiter());
        daal_data_source.loadDataBlock();
        interop::status_to_exception(daal_data_source.status());

        auto nt = daal_data_source.getNumericTable();

        daal_dm::BlockDescriptor<Float> block;
        const std::int64_t row_count = nt->getNumberOfRows();
        const std::int64_t column_count = nt->getNumberOfColumns();

        interop::status_to_exception(nt->getBlockOfRows(0, row_count, daal_dm::readOnly, block));
        Float* data = block.getBlockPtr();

        auto arr = array<Float>::empty(queue, row_count * column_count, sycl::usm::alloc::device);
        dal::detail::memcpy_host2usm(queue,
                                     arr.get_mutable_data(),
                                     data,
                                     sizeof(Float) * row_count * column_count);

        interop::status_to_exception(nt->releaseBlockOfRows(block));

        return dal::detail::homogen_table_builder{}.reset(arr, row_count, column_count).build();
    }
};

template <typename Float>
struct read_kernel_gpu<csr_table, Float> {
    csr_table operator()(const dal::backend::context_gpu& ctx,
                         const detail::data_source_base& ds,
                         const read_args<csr_table>& args) const {
        // Read CSR data from the CSV file
        auto [row_offsets, col_indices, values] = read_csr_data<Float>(ds, args);
        std::int64_t feature_count = args.get_feature_count();
        sparse_indexing indexing = args.get_sparse_indexing();

        // moving to device memory
        auto& queue = ctx.get_queue();
        auto data_arr = array<Float>::empty(queue, values.size(), sycl::usm::alloc::device);
        auto col_indices_arr =
            array<std::int64_t>::empty(queue, col_indices.size(), sycl::usm::alloc::device);
        auto row_offsets_arr =
            array<std::int64_t>::empty(queue, row_offsets.size(), sycl::usm::alloc::device);

        dal::detail::memcpy_host2usm(queue,
                                     data_arr.get_mutable_data(),
                                     values.data(),
                                     sizeof(Float) * values.size());
        dal::detail::memcpy_host2usm(queue,
                                     col_indices_arr.get_mutable_data(),
                                     col_indices.data(),
                                     sizeof(std::int64_t) * col_indices.size());
        dal::detail::memcpy_host2usm(queue,
                                     row_offsets_arr.get_mutable_data(),
                                     row_offsets.data(),
                                     sizeof(std::int64_t) * row_offsets.size());

        return csr_table::wrap(data_arr,
                               col_indices_arr,
                               row_offsets_arr,
                               feature_count,
                               indexing // sparse indexing
        );
    }
};

template struct read_kernel_gpu<table, float>;
template struct read_kernel_gpu<table, double>;

template struct read_kernel_gpu<csr_table, float>;
template struct read_kernel_gpu<csr_table, double>;

} // namespace oneapi::dal::csv::backend
