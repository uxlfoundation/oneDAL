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

#include "oneapi/dal/algo/basic_statistics.hpp"
#include "oneapi/dal/spmd/mpi/communicator.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "utils.hpp"

namespace dal = oneapi::dal;

float run(sycl::queue& queue, MPI_Comm split_comm, int color) {
    const std::int64_t row_count = 1000;
    const std::int64_t column_count = 10;
    const std::int64_t count = row_count * column_count;

    // Each rank produces its own local block of data (zeros for group 0,
    // ones for group 1) using dal::array's built-in factories -- no manual
    // sycl::queue::fill needed. Aggregation across the ranks of each
    // sub-communicator is performed by the compute() collective below. The two
    // groups run this section concurrently -- collectives on the
    // sub-communicator only synchronize the ranks within that group.
    auto arr = (color == 0)
                   ? dal::array<float>::zeros(queue, count, sycl::usm::alloc::device)
                   : dal::array<float>::full(queue, count, 1.0f, sycl::usm::alloc::device);
    const auto data = dal::homogen_table::wrap(arr, row_count, column_count);

    const auto bs_desc = dal::basic_statistics::descriptor{};

    // Convert the MPI_Comm handle into a portable int64 representation using
    // MPI_Comm_c2f so it can be passed to make_communicator. This is what
    // enables a oneDAL sub-communicator to be built from a user-provided
    // MPI communicator rather than defaulting to MPI_COMM_WORLD.
    const std::int64_t comm_handle = static_cast<std::int64_t>(MPI_Comm_c2f(split_comm));

    auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::mpi>(
        queue,
        comm_handle);

    const auto result = dal::preview::compute(comm, bs_desc, data);

    // Pull the mean row onto the host so we can return a scalar. basic_statistics
    // returns a 1 x column_count table for each statistic; every column of our
    // input is filled with the same value, so every entry of the mean row must
    // equal that value (0.0 for color 0, 1.0 for color 1).
    const auto mean_arr = dal::row_accessor<const float>{ result.get_mean() }.pull(queue);
    return mean_arr[0];
}

int main(int argc, char const* argv[]) {
    int status = MPI_Init(nullptr, nullptr);
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI init" };
    }

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (world_rank == 0) {
            std::cerr << "This sample requires at least 2 MPI ranks so that MPI_COMM_WORLD "
                         "can be split into two sub-communicators."
                      << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    // Split MPI_COMM_WORLD into two groups: the lower half (color = 0) computes
    // basic statistics over a dataset of zeros, the upper half (color = 1) over
    // a dataset of ones.
    const int color = (world_rank < world_size / 2) ? 0 : 1;
    MPI_Comm split_comm = MPI_COMM_NULL;
    status = MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &split_comm);
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI_Comm_split" };
    }

    auto device = sycl::device(sycl::gpu_selector_v);
    sycl::queue q{ device };

    const float mean_value = run(q, split_comm, color);

    int sub_rank = 0;
    int sub_size = 0;
    MPI_Comm_rank(split_comm, &sub_rank);
    MPI_Comm_size(split_comm, &sub_size);

    // Serialize the printing across the two groups so their output does not
    // interleave -- the compute above already ran concurrently. Every rank in
    // MPI_COMM_WORLD must participate in each MPI_Barrier, so the color guards
    // wrap only the std::cout call; both groups reach both barriers.
    if (color == 0 && sub_rank == 0) {
        std::cout << "Group 0 (sub-communicator size = " << sub_size
                  << ") mean: " << mean_value << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (color == 1 && sub_rank == 0) {
        std::cout << "Group 1 (sub-communicator size = " << sub_size
                  << ") mean: " << mean_value << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Guarantee that the mean equals the color: 0.0 for group 0, 1.0 for group 1.
    const float expected = static_cast<float>(color);
    if (mean_value != expected) {
        std::cerr << "Rank " << world_rank << " (color " << color << ") got mean " << mean_value
                  << ", expected " << expected << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_free(&split_comm);

    status = MPI_Finalize();
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI finalize" };
    }
    return 0;
}
