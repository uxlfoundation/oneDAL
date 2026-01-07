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

#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
template <typename TableReadType>
void run() {
    const auto data_file_name = get_data_path("dev/data/covcormoments_dense.csv");

    const auto table =
        dal::read<dal::table>(dal::csv::data_source<TableReadType>{ data_file_name });

    const auto type = table.get_metadata().get_data_type(0);
    switch (type) {
        case dal::data_type::float64: std::cout << "table data type: double " << std::endl; break;
        case dal::data_type::float32: std::cout << "table data type: float " << std::endl; break;
        default: break;
    }

    std::cout << table << std::endl;
}

int main(int argc, char const *argv[]) {
    run<float>();
    run<double>();

    return 0;
}
