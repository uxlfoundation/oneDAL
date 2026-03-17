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

#include <sycl/sycl.hpp>
#include <iomanip>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
template <typename TableReadType>
void run(sycl::queue &q) {
    const auto data_file_name = get_data_path("data/covcormoments_dense.csv");

    const auto table =
        dal::read<dal::table>(q, dal::csv::data_source<TableReadType>{ data_file_name });

    const auto type = table.get_metadata().get_data_type(0);
    switch (type) {
        case dal::data_type::float64: std::cout << "table data type: double " << std::endl; break;
        case dal::data_type::float32: std::cout << "table data type: float " << std::endl; break;
        default: break;
    }

    std::cout << table << std::endl;
}

int main(int argc, char const *argv[]) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run<float>(q);
        run<double>(q);
    }

    return 0;
}
