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
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/hdbscan.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue& q) {
    const char* data_env = std::getenv("HDBSCAN_DATA_PATH");
    const std::string data_file_name =
        data_env ? std::string(data_env) : get_data_path("data/hdbscan_dense.csv");

    const char* mcs_env = std::getenv("HDBSCAN_MIN_CLUSTER_SIZE");
    const std::int64_t min_cluster_size = mcs_env ? std::atol(mcs_env) : 15;

    const char* ms_env = std::getenv("HDBSCAN_MIN_SAMPLES");
    const std::int64_t min_samples = ms_env ? std::atol(ms_env) : 10;

    const auto x_data = dal::read<dal::table>(q, dal::csv::data_source{ data_file_name });

    std::cout << "Data dimensions: " << x_data.get_row_count() << " x " << x_data.get_column_count()
              << std::endl;
    std::cout << "Parameters: min_cluster_size=" << min_cluster_size
              << ", min_samples=" << min_samples << std::endl;

    // Use kd_tree method — memory-efficient, avoids O(N²) distance matrix
    auto hdbscan_desc =
        dal::hdbscan::descriptor<float, dal::hdbscan::method::kd_tree>(min_cluster_size,
                                                                       min_samples);
    hdbscan_desc.set_result_options(dal::hdbscan::result_options::responses);

    // Warmup run
    dal::compute(q, hdbscan_desc, x_data);

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    const auto result = dal::compute(q, hdbscan_desc, x_data);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Cluster count: " << result.get_cluster_count() << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(1) << ms << " ms" << std::endl;

    const auto responses = result.get_responses();
    const auto acc = dal::row_accessor<const std::int32_t>(responses);
    const auto labels = acc.pull({ 0, -1 });
    const std::int32_t* lbl_ptr = labels.get_data();
    const std::int64_t n = responses.get_row_count();

    std::cout << "Labels:";
    for (std::int64_t i = 0; i < n; i++) {
        std::cout << " " << lbl_ptr[i];
    }
    std::cout << std::endl;
}

int main(int argc, char const* argv[]) {
    for (auto d : list_devices()) {
        const auto platform_name = d.get_platform().get_info<sycl::info::platform::name>();
        std::cout << "Running on " << platform_name << ", "
                  << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
