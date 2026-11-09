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

// PCA under the kernel profiler: with ONEDAL_VERBOSE=3 set in the environment, the per-stage time
// and memory report is printed at exit. `--help` lists the options.

#include <chrono>

#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/args_helpers.hpp"
#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
namespace pca = dal::pca;

template <typename Method>
void run(const dal::table &x_train, std::int64_t component_count, std::int64_t repeat) {
    const auto pca_desc = pca::descriptor<float, Method>()
                              .set_component_count(component_count)
                              .set_deterministic(true);

    const auto start = std::chrono::steady_clock::now();
    pca::train_result<> result_train;
    pca::infer_result<> result_infer;
    for (std::int64_t i = 0; i < repeat; ++i) {
        result_train = dal::train(pca_desc, x_train);
        result_infer = dal::infer(pca_desc, result_train.get_model(), x_train);
    }
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start;
    std::cout << "Wall time per train and infer call: " << elapsed.count() / double(repeat) << " ms"
              << std::endl;

    std::cout << "Eigenvectors:\n" << result_train.get_eigenvectors() << std::endl;
    std::cout << "Eigenvalues:\n" << result_train.get_eigenvalues() << std::endl;
    std::cout << "Transformed data:\n" << result_infer.get_transformed_data() << std::endl;
}

int main(int argc, char const *argv[]) {
    if (has_flag(argc, argv, "--help")) {
        std::cout
            << "Usage: ONEDAL_VERBOSE=3 pca_dense_batch_profiler [options]\n"
               "  --data FILE       CSV input (default: data/pca_normalized.csv)\n"
               "  --method cov|svd  training method (default: cov)\n"
               "  --components N    number of components (default: 5)\n"
               "  --repeat N        train and infer calls, merged in the report (default: 1)\n"
               "ONEDAL_VERBOSE is read before the first oneDAL call; see docs/profiling.\n";
        return 0;
    }
    const auto train_data_file_name =
        get_data_path(get_option(argc, argv, "--data", "data/pca_normalized.csv"));
    const std::string method = get_option(argc, argv, "--method", "cov");
    const std::int64_t component_count = get_option(argc, argv, "--components", std::int64_t(5));
    const std::int64_t repeat = get_option(argc, argv, "--repeat", std::int64_t(1));

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });
    std::cout << "Input: " << x_train.get_row_count() << " rows x " << x_train.get_column_count()
              << " columns, method: " << method << ", components: " << component_count
              << ", repeat: " << repeat << std::endl;

    if (method == "cov") {
        run<pca::method::cov>(x_train, component_count, repeat);
    }
    else if (method == "svd") {
        run<pca::method::svd>(x_train, component_count, repeat);
    }
    else {
        std::cerr << "Unknown --method value: " << method << std::endl;
        return 1;
    }
    return 0;
}
