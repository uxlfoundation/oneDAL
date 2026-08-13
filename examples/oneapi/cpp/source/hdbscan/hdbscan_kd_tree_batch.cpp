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

#include "oneapi/dal/algo/hdbscan.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

int main(int argc, char const* argv[]) {
    const auto data_file_name = get_data_path("data/hdbscan_dense.csv");

    const auto x_data = dal::read<dal::table>(dal::csv::data_source{ data_file_name });

    std::cout << "Data dimensions: " << x_data.get_row_count() << " x " << x_data.get_column_count()
              << std::endl;

    const std::int64_t min_cluster_size = 15;
    const std::int64_t min_samples = 10;

    auto hdbscan_desc =
        dal::hdbscan::descriptor<float, dal::hdbscan::method::kd_tree>(min_cluster_size,
                                                                       min_samples);
    hdbscan_desc.set_result_options(dal::hdbscan::result_options::responses);

    const auto result = dal::compute(hdbscan_desc, x_data);

    std::cout << "Cluster count: " << result.get_cluster_count() << std::endl;
    std::cout << "Responses:\n" << result.get_responses() << std::endl;
    return 0;
}
