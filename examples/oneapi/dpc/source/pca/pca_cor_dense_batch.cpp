/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include <sycl/sycl.hpp>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
namespace pca = dal::pca;
template <typename Method>
void run(sycl::queue& q, const dal::table& x_train, const std::string& method_name, bool whiten) {
    const auto pca_desc =
        pca::descriptor<>().set_component_count(5).set_deterministic(true).set_whiten(whiten);

    const auto result_train = dal::train(q, pca_desc, x_train);

    std::cout << method_name << "\n" << std::endl;

    std::cout << "Eigenvectors:\n" << result_train.get_eigenvectors() << std::endl;

    std::cout << "Eigenvalues:\n" << result_train.get_eigenvalues() << std::endl;

    std::cout << "Singular Values:\n" << result_train.get_singular_values() << std::endl;

    std::cout << "Variances:\n" << result_train.get_variances() << std::endl;

    std::cout << "Means:\n" << result_train.get_means() << std::endl;

    std::cout << "Explained variances ratio:\n"
              << result_train.get_explained_variances_ratio() << std::endl;

    const auto result_infer = dal::infer(q, pca_desc, result_train.get_model(), x_train);

    std::cout << "Transformed data:\n" << result_infer.get_transformed_data() << std::endl;
}

int main(int argc, char const* argv[]) {
    const auto train_data_file_name = get_data_path("dev/data/pca_normalized.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });

    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run<pca::method::cov>(q, x_train, "Training method: Correlation Whiten:false", false);
        run<pca::method::cov>(q, x_train, "Training method: Correlation Whiten:false", true);
    }
    return 0;
}
