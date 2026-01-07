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

#include "oneapi/dal/algo/covariance.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
int main(int argc, char const *argv[]) {
    const auto input_file_name = get_data_path("data/covcormoments_dense.csv");
    const std::int64_t nBlocks = 10;

    const auto input = dal::read<dal::table>(dal::csv::data_source{ input_file_name });
    const auto cov_desc = dal::covariance::descriptor{}
                              .set_result_options(dal::covariance::result_options::cov_matrix |
                                                  dal::covariance::result_options::means)
                              .set_bias(true);

    dal::covariance::partial_compute_result<> partial_result;

    auto input_table = split_table_by_rows<double>(input, nBlocks);
    for (std::int64_t i = 0; i < nBlocks; i++) {
        partial_result = dal::partial_compute(cov_desc, partial_result, input_table[i]);
    }
    auto result = dal::finalize_compute(cov_desc, partial_result);

    std::cout << "Maximum likelihood covariance estimation:\n"
              << result.get_cov_matrix() << std::endl;
}
