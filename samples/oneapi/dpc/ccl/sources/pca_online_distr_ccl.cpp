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
#include <mpi.h>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/spmd/ccl/communicator.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "utils.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue& queue) {
    const auto data_file_name = get_data_path("data/pca_normalized.csv");
    const std::int64_t nBlocks = 10;
    const auto data = dal::read<dal::table>(queue, dal::csv::data_source{ data_file_name });

    const auto pca_desc = dal::pca::descriptor{};

    auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::ccl>(queue);
    auto rank_id = comm.get_rank();
    auto rank_count = comm.get_rank_count();

    auto input_vec = split_table_by_rows<float>(queue, data, rank_count);

    auto input_blocks = split_table_by_rows<float>(queue, input_vec[rank_id], nBlocks);
    dal::pca::partial_train_result<> partial_result;

    for (std::int64_t i = 0; i < nBlocks; i++) {
        partial_result = dal::partial_train(queue, pca_desc, partial_result, input_blocks[i]);
    }
    const auto result = dal::preview::finalize_train(comm, pca_desc, partial_result);

    if (comm.get_rank() == 0) {
        std::cout << "Eigenvectors:\n" << result.get_eigenvectors() << std::endl;
        std::cout << "Eigenvalues:\n" << result.get_eigenvalues() << std::endl;
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
