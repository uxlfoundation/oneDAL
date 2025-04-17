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

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/covariance.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue &q) {
    const auto input_file_name = get_data_path("covcormoments_dense.csv");

    const auto input = dal::read<dal::table>(q, dal::csv::data_source{ input_file_name });
    auto cov_desc = dal::covariance::descriptor{}.set_result_options(
        dal::covariance::result_options::cov_matrix);

    auto result = dal::compute(q, cov_desc, input);

    std::cout << "Sample covariance:\n" << result.get_cov_matrix() << std::endl;
}

int main(int argc, char const *argv[]) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
