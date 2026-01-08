/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/spmd/mpi/communicator.hpp"

#include "utils.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue& queue) {
    const auto train_data_file_name = get_data_path("data/kmeans_dense_train_data.csv");
    const auto initial_centroids_file_name = get_data_path("data/kmeans_dense_train_centroids.csv");

    const auto x_train =
        dal::read<dal::table>(queue, dal::csv::data_source{ train_data_file_name });
    const auto initial_centroids =
        dal::read<dal::table>(queue, dal::csv::data_source{ initial_centroids_file_name });

    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(20)
                                 .set_max_iteration_count(5)
                                 .set_accuracy_threshold(0.001);
    auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::mpi>(queue);
    auto rank_id = comm.get_rank();
    auto rank_count = comm.get_rank_count();

    auto input_vec = split_table_by_rows<float>(queue, x_train, rank_count);
    dal::kmeans::train_input local_input{ input_vec[rank_id], initial_centroids };

    const auto result_train = dal::preview::train(comm, kmeans_desc, local_input);
    if (comm.get_rank() == 0) {
        std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
        std::cout << "Objective function value: " << result_train.get_objective_function_value()
                  << std::endl;
        std::cout << "Centroids:\n" << result_train.get_model().get_centroids() << std::endl;
    }
}

int main(int argc, char const* argv[]) {
    int status = MPI_Init(nullptr, nullptr);
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI init" };
    }

    auto device = sycl::device(sycl::gpu_selector_v);
    std::cout << "Running on " << device.get_platform().get_info<sycl::info::platform::name>()
              << ", " << device.get_info<sycl::info::device::name>() << std::endl;

    sycl::queue q{ device };
    run(q);

    status = MPI_Finalize();
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI finalize" };
    }
    return 0;
}
