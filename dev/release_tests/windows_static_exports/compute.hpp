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

#include <cmath>

#include "daal.h"
#include "oneapi/dal/algo/basic_statistics.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

/// Computes the column means of a 3 x 2 table with both interfaces.
///
/// @return 0 when both match the expected means, 1 otherwise
inline int compute_means() {
    const double data[] = { 1.0, 10.0, 2.0, 20.0, 3.0, 30.0 };
    const double expected[] = { 2.0, 20.0 };

    using namespace daal;
    auto table = data_management::HomogenNumericTable<double>::create(const_cast<double*>(data), 2, 3);
    algorithms::low_order_moments::Batch<double> moments;
    moments.input.set(algorithms::low_order_moments::data, table);
    moments.compute();
    auto daal_mean = moments.getResult()->get(algorithms::low_order_moments::mean);
    data_management::BlockDescriptor<double> block;
    daal_mean->getBlockOfRows(0, 1, data_management::readOnly, block);
    const double* daal_values = block.getBlockPtr();

    namespace dal = oneapi::dal;
    const auto dal_table = dal::homogen_table::wrap(data, 3, 2);
    const auto result = dal::compute(dal::basic_statistics::descriptor{}, dal_table);
    const auto dal_values = dal::row_accessor<const double>(result.get_mean()).pull();

    int status = 0;
    for (int j = 0; j < 2; ++j) {
        if (std::abs(daal_values[j] - expected[j]) > 1e-12 ||
            std::abs(dal_values[j] - expected[j]) > 1e-12) {
            status = 1;
        }
    }
    daal_mean->releaseBlockOfRows(block);
    return status;
}
