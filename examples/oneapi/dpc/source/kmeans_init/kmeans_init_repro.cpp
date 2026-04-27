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
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/algo/kmeans_init.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue& q,
         const std::vector<float>& data,
         std::int64_t n_samples,
         std::int64_t n_features,
         std::int64_t n_clusters) {
    constexpr std::int64_t max_iter = 5;

    auto x_table = dal::homogen_table::wrap(data.data(), n_samples, n_features);
    std::cout << "Table: rows=" << x_table.get_row_count() << " cols=" << x_table.get_column_count()
              << std::endl;

    // Step 1: kmeans++ init
    const auto init_desc =
        dal::kmeans_init::descriptor<float, dal::kmeans_init::method::plus_plus_dense>()
            .set_cluster_count(n_clusters);

    std::cout << "Running kmeans_init plus_plus_dense..." << std::endl;
    std::cout << std::flush;
    std::cerr << std::flush;

    const auto init_result = dal::compute(q, init_desc, x_table);

    std::cout << "Init done! Centroids shape: " << init_result.get_centroids().get_row_count()
              << " x " << init_result.get_centroids().get_column_count() << std::endl;

    // Step 2: kmeans Lloyd training
    const auto kmeans_desc = dal::kmeans::descriptor<float>()
                                 .set_cluster_count(n_clusters)
                                 .set_max_iteration_count(max_iter);

    std::cout << "Running kmeans train (Lloyd, max_iter=" << max_iter << ")..." << std::endl;
    std::cout << std::flush;

    const auto train_result = dal::train(q, kmeans_desc, x_table, init_result.get_centroids());

    std::cout << "Train done! Iterations: " << train_result.get_iteration_count()
              << ", Objective: " << train_result.get_objective_function_value() << std::endl;
}

int main(int argc, char const* argv[]) {
    constexpr std::int64_t n_samples = 17408;
    constexpr std::int64_t n_features = 17;
    constexpr std::int64_t n_clusters = 32;

    std::vector<float> data;

    // Try to load exact Python test data from binary file
    const char* data_file = "/tmp/kmeans_test_data.bin";
    std::ifstream fin(data_file, std::ios::binary);
    if (fin.good()) {
        data.resize(n_samples * n_features);
        fin.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
        std::cout << "Loaded test data from " << data_file << " (" << data.size() << " floats)"
                  << std::endl;
    }
    else {
        std::cout << "No data file found, generating random data" << std::endl;
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        data.resize(n_samples * n_features);
        for (auto& v : data) {
            v = dist(gen);
        }
    }

    std::cout << "Data: " << n_samples << " x " << n_features << ", clusters=" << n_clusters
              << ", float32" << std::endl;

    // Try GPU first
    for (auto d : sycl::platform::get_platforms()) {
        for (auto dev : d.get_devices()) {
            if (dev.is_gpu()) {
                std::cout << "Running on GPU: " << dev.get_info<sycl::info::device::name>()
                          << std::endl;
                auto q = sycl::queue{ dev };
                run(q, data, n_samples, n_features, n_clusters);
                std::cout << "SUCCESS" << std::endl;
                return 0;
            }
        }
    }
    std::cout << "No GPU found, trying CPU..." << std::endl;
    auto q = sycl::queue{ sycl::cpu_selector_v };
    run(q, data, n_samples, n_features, n_clusters);
    std::cout << "SUCCESS" << std::endl;
    return 0;
}
