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

#ifndef ONEAPI_DAL_DATA_CONVERSION

    #define ONEAPI_DAL_DATA_CONVERSION
    #include "daal/include/data_management/data_source/csv_feature_manager.h"
    #include "daal/include/data_management/data_source/file_data_source.h"
    #undef ONEAPI_DAL_DATA_CONVERSION

#endif

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/io/csv/backend/cpu/read_kernel.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::csv::backend {

namespace interop = dal::backend::interop;

table read_kernel_cpu<table>::operator()(const dal::backend::context_cpu& ctx,
                                         const data_source& ds,
                                         const read_args<table>& args) const {
    using namespace daal::data_management;

    CsvDataSourceOptions csv_options = CsvDataSourceOptions::allocateNumericTable |
                                       CsvDataSourceOptions::createDictionaryFromContext |
                                       (ds.get_parse_header() ? CsvDataSourceOptions::parseHeader
                                                              : CsvDataSourceOptions::byDefault);

    FileDataSource<CSVFeatureManager> daal_data_source(ds.get_file_name().c_str(), csv_options);
    daal_data_source.getFeatureManager().setDelimiter(ds.get_delimiter());
    daal_data_source.loadDataBlock();

    interop::status_to_exception(daal_data_source.status());

    return oneapi::dal::backend::interop::convert_from_daal_homogen_table<DAAL_DATA_TYPE>(
        daal_data_source.getNumericTable());
}

} // namespace oneapi::dal::csv::backend
