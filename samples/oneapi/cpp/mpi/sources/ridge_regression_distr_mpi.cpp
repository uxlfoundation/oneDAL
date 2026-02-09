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

#include "oneapi/dal/algo/linear_regression.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/spmd/mpi/communicator.hpp"

#include "utils.hpp"

namespace dal = oneapi::dal;
namespace lr = dal::linear_regression;

void run() {
    const auto train_data_file_name = get_data_path("linear_regression_train_data.csv");
    const auto train_response_file_name = get_data_path("linear_regression_train_responses.csv");
    const auto test_data_file_name = get_data_path("linear_regression_test_data.csv");
    const auto test_response_file_name = get_data_path("linear_regression_test_responses.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });
    const auto y_train = dal::read<dal::table>(dal::csv::data_source{ train_response_file_name });
    const auto x_test = dal::read<dal::table>(dal::csv::data_source{ test_data_file_name });
    const auto y_test = dal::read<dal::table>(dal::csv::data_source{ test_response_file_name });

    auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::mpi>();
    auto rank_id = comm.get_rank();
    auto rank_count = comm.get_rank_count();

    auto x_train_vec = split_table_by_rows<float>(x_train, rank_count);
    auto y_train_vec = split_table_by_rows<float>(y_train, rank_count);
    auto x_test_vec = split_table_by_rows<float>(x_test, rank_count);
    auto y_test_vec = split_table_by_rows<float>(y_test, rank_count);

    const double alpha = 1.0;
    const auto rr_desc = lr::descriptor<float>{ true, alpha };

    const auto result_train =
        dal::preview::train(comm, rr_desc, x_train_vec.at(rank_id), y_train_vec.at(rank_id));

    const auto result_infer =
        dal::preview::infer(comm, rr_desc, x_test_vec.at(rank_id), result_train.get_model());

    if (comm.get_rank() == 0) {
        std::cout << "Ridge regression alpha: " << alpha << std::endl;

        std::cout << "Prediction results:\n" << result_infer.get_responses() << std::endl;

        std::cout << "Ground truth:\n" << y_test_vec.at(rank_id) << std::endl;
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
