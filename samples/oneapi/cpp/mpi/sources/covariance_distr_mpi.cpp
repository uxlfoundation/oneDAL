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

#include <iomanip>
#include <iostream>

#include "oneapi/dal/algo/covariance.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/spmd/mpi/communicator.hpp"

#include "utils.hpp"

namespace dal = oneapi::dal;

void run() {
    const auto data_file_name = get_data_path("covcormoments_dense.csv");

    const auto data = dal::read<dal::table>(dal::csv::data_source{ data_file_name });

    const auto cov_desc = dal::covariance::descriptor<float>{}.set_result_options(
        dal::covariance::result_options::cov_matrix | dal::covariance::result_options::means);

    auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::mpi>();
    auto rank_id = comm.get_rank();
    auto rank_count = comm.get_rank_count();

    auto input_vec = split_table_by_rows<float>(data, rank_count);

    const auto result = dal::preview::compute(comm, cov_desc, input_vec[rank_id]);

    if (comm.get_rank() == 0) {
        std::cout << "Sample covariance:\n" << result.get_cov_matrix() << std::endl;

        std::cout << "Means:\n" << result.get_means() << std::endl;
    }
}

int main(int argc, char const *argv[]) {
    int status = MPI_Init(nullptr, nullptr);
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI init" };
    }

    run();

    status = MPI_Finalize();
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI finalize" };
    }
    return 0;
}
