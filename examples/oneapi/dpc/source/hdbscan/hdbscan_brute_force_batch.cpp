/*******************************************************************************
* Copyright 2024 Intel Corporation
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

/// HDBSCAN benchmark: reads hdbscan_bench.csv, runs HDBSCAN,
/// saves labels to hdbscan_bench_onedal_labels.csv for comparison with sklearn.

#include <sycl/sycl.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/hdbscan.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue& q) {
    const auto data_file_name = get_data_path("data/hdbscan_bench.csv");

    const auto x_data = dal::read<dal::table>(q, dal::csv::data_source{ data_file_name });

    std::cout << "Data: " << x_data.get_row_count() << " rows x " << x_data.get_column_count()
              << " cols" << std::endl;

    constexpr std::int64_t min_cluster_size = 15;
    constexpr std::int64_t min_samples = 10;

    auto hdbscan_desc = dal::hdbscan::descriptor<double>(min_cluster_size, min_samples);
    hdbscan_desc.set_result_options(dal::hdbscan::result_options::responses);

    // Warmup
    dal::compute(q, hdbscan_desc, x_data);
    q.wait_and_throw();

    auto t0 = std::chrono::high_resolution_clock::now();
    const auto result = dal::compute(q, hdbscan_desc, x_data);
    q.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

    std::cout << "Time: " << std::fixed << std::setprecision(1) << elapsed_ms << " ms" << std::endl;
    std::cout << "Cluster count: " << result.get_cluster_count() << std::endl;

    // Count noise and save labels
    const auto responses = result.get_responses();
    const auto row_count = responses.get_row_count();
    dal::row_accessor<const double> acc(responses);
    const auto arr = acc.pull({ 0, -1 });

    std::int64_t noise_count = 0;
    for (std::int64_t i = 0; i < row_count; i++) {
        if (static_cast<std::int32_t>(arr[i]) == -1)
            noise_count++;
    }
    std::cout << "Noise points: " << noise_count << " / " << row_count << std::endl;

    // Save labels for comparison
    const std::string output_file =
        "/export/users/asolovev/oneDAL/data/hdbscan_bench_onedal_labels.csv";
    std::ofstream ofs(output_file);
    if (ofs.is_open()) {
        for (std::int64_t i = 0; i < row_count; i++) {
            ofs << static_cast<std::int32_t>(arr[i]) << "\n";
        }
        ofs.close();
        std::cout << "Labels saved to: " << output_file << std::endl;
    }
    else {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
    }
}

int main(int argc, char const* argv[]) {
    for (auto d : list_devices_gpu_only()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
