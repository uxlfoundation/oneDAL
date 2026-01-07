/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/knn.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/spmd/ccl/communicator.hpp"

#include "utils.hpp"

void run(sycl::queue& queue) {
    const auto train_data_file_name = get_data_path("dev/data/k_nearest_neighbors_train_data.csv");
    const auto train_response_file_name =
        get_data_path("dev/data/k_nearest_neighbors_train_label.csv");
    const auto test_data_file_name = get_data_path("dev/data/k_nearest_neighbors_test_data.csv");
    const auto test_response_file_name =
        get_data_path("dev/data/k_nearest_neighbors_test_label.csv");

    const auto x_train =
        dal::read<dal::table>(queue, dal::csv::data_source{ train_data_file_name });
    const auto y_train =
        dal::read<dal::table>(queue, dal::csv::data_source{ train_response_file_name });
    const auto x_test = dal::read<dal::table>(queue, dal::csv::data_source{ test_data_file_name });
    const auto y_test =
        dal::read<dal::table>(queue, dal::csv::data_source{ test_response_file_name });

    auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::ccl>(queue);
    auto rank_id = comm.get_rank();
    auto rank_count = comm.get_rank_count();

    auto x_train_vec = split_table_by_rows<float>(queue, x_train, rank_count);
    auto y_train_vec = split_table_by_rows<float>(queue, y_train, rank_count);
    auto x_test_vec = split_table_by_rows<float>(queue, x_test, rank_count);
    auto y_test_vec = split_table_by_rows<float>(queue, y_test, rank_count);

    const auto knn_desc = dal::knn::descriptor(5, 1).set_result_options(
        dal::knn::result_options::responses | dal::knn::result_options::indices |
        dal::knn::result_options::distances);

    dal::knn::train_input local_input_train{ x_train_vec[rank_id], y_train_vec[rank_id] };

    const auto result_train = dal::preview::train(comm, knn_desc, local_input_train);
    const auto result_infer =
        dal::preview::infer(comm, knn_desc, x_test_vec[rank_id], result_train.get_model());
    if (comm.get_rank() == 0) {
        std::cout << "Prediction results:\n" << result_infer.get_responses() << std::endl;

        std::cout << "Ground truth:\n" << y_test_vec.at(rank_id) << std::endl;
    }
}

int main(int argc, char const* argv[]) {
    ccl::init();
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
