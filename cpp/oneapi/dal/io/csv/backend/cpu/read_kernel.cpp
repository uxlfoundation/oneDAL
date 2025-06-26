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
#include "oneapi/dal/io/csv/backend/cpu/read_kernel.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::csv::backend {

namespace interop = dal::backend::interop;
namespace daal_dm = daal::data_management;

template <typename Float>
struct read_kernel_cpu<table, Float> {
    table operator()(const dal::backend::context_cpu& ctx,
                     const detail::data_source_base& ds,
                     const read_args<table>& args) const {
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

        return oneapi::dal::backend::interop::convert_from_daal_homogen_table<Float>(
            daal_data_source.getNumericTable());
    }
};

template <typename Float>
struct read_kernel_cpu<csr_table, Float> {
    csr_table operator()(const dal::backend::context_cpu& ctx,
                         const detail::data_source_base& ds,
                         const read_args<csr_table>& args) const {
        std::ifstream file(ds.get_file_name());
        if (!file.is_open()) {
            throw dal::range_error(dal::detail::error_messages::file_not_found());
        }

        // csr table format:
        // first line: row offsets, i.e., the indices (within non-zero elems array) of the first non-zero element in each row
        // second line: column indices of the non-zero elements
        // third line: non-zero elements of the original matrix

        std::string line;
        std::vector<std::int64_t> row_offsets;
        std::vector<std::int64_t> col_indices;
        std::vector<Float> values;

        char delimiter = ds.get_delimiter();
        std::int64_t feature_count = args.get_feature_count();
        sparse_indexing indexing = args.get_sparse_indexing();

        // Read the first line (row offsets)
        if (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string value;
            while (std::getline(ss, value, delimiter)) {
                row_offsets.push_back(static_cast<std::int64_t>(std::stoi(value)));
            }
        }

        // Read the second line (column indices)
        if (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string value;
            while (std::getline(ss, value, delimiter)) {
                col_indices.push_back(static_cast<std::int64_t>(std::stoi(value)));
            }
        }

        // Read the third line (non-zero elements)
        if (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string value;
            while (std::getline(ss, value, delimiter)) {
                values.push_back(static_cast<Float>(std::stod(value)));
            }
        }

        file.close();

        std::int64_t row_offsets_checker = static_cast<std::int64_t>(values.size()) +
                                           (indexing == sparse_indexing::one_based ? 1 : 0);
        if (values.size() != col_indices.size() || row_offsets.size() == 0 ||
            row_offsets[row_offsets.size() - 1] != row_offsets_checker) {
            throw dal::invalid_argument(dal::detail::error_messages::invalid_csr_format());
        }

        if (!col_indices.empty() && feature_count == 0) {
            // If feature count is not specified, we estimate it from the column indices
            // This is a best-effort estimation
            feature_count = *std::max_element(col_indices.begin(), col_indices.end());
            if (indexing == sparse_indexing::zero_based) {
                feature_count += 1;
            }
        }

        // Transfer vector data to arrays
        auto values_array = dal::array<Float>::empty(values.size());
        auto col_indices_array = dal::array<std::int64_t>::empty(col_indices.size());
        auto row_offsets_array = dal::array<std::int64_t>::empty(row_offsets.size());

        std::copy(values.begin(), values.end(), values_array.get_mutable_data());
        std::copy(col_indices.begin(), col_indices.end(), col_indices_array.get_mutable_data());
        std::copy(row_offsets.begin(), row_offsets.end(), row_offsets_array.get_mutable_data());

        return csr_table::wrap(values_array,
                               col_indices_array,
                               row_offsets_array,
                               feature_count,
                               indexing // sparse indexing
        );
    }
};

template struct read_kernel_cpu<table, float>;
template struct read_kernel_cpu<table, double>;

template struct read_kernel_cpu<csr_table, float>;
template struct read_kernel_cpu<csr_table, double>;

} // namespace oneapi::dal::csv::backend
