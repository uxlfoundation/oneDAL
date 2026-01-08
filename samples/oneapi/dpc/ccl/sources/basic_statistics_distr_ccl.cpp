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

#include "oneapi/dal/algo/basic_statistics.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/spmd/ccl/communicator.hpp"

#include "utils.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue &queue) {
    const auto data_file_name = get_data_path("data/covcormoments_dense.csv");

    const auto data = dal::read<dal::table>(queue, dal::csv::data_source{ data_file_name });

    const auto bs_desc = dal::basic_statistics::descriptor{};

    auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::ccl>(queue);
    auto rank_id = comm.get_rank();
    auto rank_count = comm.get_rank_count();

    auto input_vec = split_table_by_rows<float>(queue, data, rank_count);

    const auto result = dal::preview::compute(comm, bs_desc, input_vec[rank_id]);
    if (comm.get_rank() == 0) {
        std::cout << "Minimum:\n" << result.get_min() << std::endl;
        std::cout << "Maximum:\n" << result.get_max() << std::endl;
        std::cout << "Sum:\n" << result.get_sum() << std::endl;
        std::cout << "Sum of squares:\n" << result.get_sum_squares() << std::endl;
        std::cout << "Sum of squared difference from the means:\n"
                  << result.get_sum_squares_centered() << std::endl;
        std::cout << "Mean:\n" << result.get_mean() << std::endl;
        std::cout << "Second order raw moment:\n"
                  << result.get_second_order_raw_moment() << std::endl;
        std::cout << "Variance:\n" << result.get_variance() << std::endl;
        std::cout << "Standard deviation:\n" << result.get_standard_deviation() << std::endl;
        std::cout << "Variation:\n" << result.get_variation() << std::endl;
    }
}

int main(int argc, char const *argv[]) {
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
