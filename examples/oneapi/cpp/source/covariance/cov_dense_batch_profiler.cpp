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

// Covariance under the kernel profiler. Run with ONEDAL_VERBOSE set, for example
//
//     ONEDAL_VERBOSE=3 ./cov_dense_batch_profiler --result all --repeat 10
//
// and the library prints the per-stage time and memory report when the process exits. The
// profiler reads ONEDAL_VERBOSE before the first oneDAL call, so it must be set in the environment.
//
// Options:
//   --data FILE               CSV input (default: data/covcormoments_dense.csv)
//   --result cov|cor|means|all  what to compute (default: all)
//   --repeat N                number of compute calls; the report merges them into one row (default: 1)

#include <chrono>

#include "oneapi/dal/algo/covariance.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/args_helpers.hpp"
#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

int main(int argc, char const *argv[]) {
    if (has_flag(argc, argv, "--help")) {
        std::cout << "Usage: cov_dense_batch_profiler [--data FILE] [--result cov|cor|means|all] "
                     "[--repeat N]\n"
                     "Set ONEDAL_VERBOSE=3 to get the per-stage report; see docs/profiling.\n";
        return 0;
    }
    const auto input_file_name =
        get_data_path(get_option(argc, argv, "--data", "data/covcormoments_dense.csv"));
    const std::string result = get_option(argc, argv, "--result", "all");
    const std::int64_t repeat = get_option(argc, argv, "--repeat", std::int64_t(1));

    auto result_options = dal::covariance::result_options::cov_matrix |
                          dal::covariance::result_options::cor_matrix |
                          dal::covariance::result_options::means;
    if (result == "cov") {
        result_options = dal::covariance::result_options::cov_matrix;
    }
    else if (result == "cor") {
        result_options = dal::covariance::result_options::cor_matrix;
    }
    else if (result == "means") {
        result_options = dal::covariance::result_options::means;
    }
    else if (result != "all") {
        std::cerr << "Unknown --result value: " << result << std::endl;
        return 1;
    }

    const auto input = dal::read<dal::table>(dal::csv::data_source{ input_file_name });
    const auto cov_desc = dal::covariance::descriptor{}.set_result_options(result_options);

    std::cout << "Input: " << input.get_row_count() << " rows x " << input.get_column_count()
              << " columns, result: " << result << ", repeat: " << repeat << std::endl;

    const auto start = std::chrono::steady_clock::now();
    dal::covariance::compute_result<> compute_result;
    for (std::int64_t i = 0; i < repeat; ++i) {
        compute_result = dal::compute(cov_desc, input);
    }
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start;
    std::cout << "Wall time per compute call: " << elapsed.count() / double(repeat) << " ms"
              << std::endl;

    if (result_options.test(dal::covariance::result_options::means)) {
        std::cout << "Means:\n" << compute_result.get_means() << std::endl;
    }
    if (result_options.test(dal::covariance::result_options::cov_matrix)) {
        std::cout << "Sample covariance:\n" << compute_result.get_cov_matrix() << std::endl;
    }
    if (result_options.test(dal::covariance::result_options::cor_matrix)) {
        std::cout << "Sample correlation:\n" << compute_result.get_cor_matrix() << std::endl;
    }
    return 0;
}
