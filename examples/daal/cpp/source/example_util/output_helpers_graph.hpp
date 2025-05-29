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

#pragma once

#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/table/homogen.hpp"

template <typename Task>
void print_vertex_similarity_result(
    const oneapi::dal::preview::jaccard::vertex_similarity_result<Task> &result) {
    using namespace oneapi::dal;
    const std::int64_t nonzero_coeff_count = result.get_nonzero_coeff_count();
    const std::int64_t lines_count = 20;
    const std::int64_t print_count =
        (nonzero_coeff_count < lines_count) ? nonzero_coeff_count : lines_count;

    std::vector<std::int64_t> print_range(print_count);
    auto middle_iterator = print_range.begin() + print_count / 2;
    std::iota(print_range.begin(), middle_iterator, 0);
    std::iota(middle_iterator,
              print_range.end(),
              nonzero_coeff_count - print_count / 2 - print_count % 2);

    const auto rows_count = result.get_vertex_pairs().get_row_count();

    if (result.get_vertex_pairs().get_kind() == homogen_table::kind() &&
        result.get_coeffs().get_kind() == homogen_table::kind()) {
        auto vertex_pairs_table = result.get_vertex_pairs();
        homogen_table &vertex_pairs = static_cast<homogen_table &>(vertex_pairs_table);
        const auto vertex_pairs_data = vertex_pairs.get_data<int>();

        auto coeffs_table = result.get_coeffs();
        homogen_table &coeffs = static_cast<homogen_table &>(coeffs_table);
        const auto jaccard_coeffs_data = coeffs.get_data<float>();

        for (auto i : print_range) {
            std::cout << std::setw(5) << i << ": ";
            for (auto j : { 0, 1 }) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(0) << vertex_pairs_data[j * rows_count + i];
            }
            std::cout << std::setw(15) << std::setiosflags(std::ios::fixed) << std::setprecision(3)
                      << jaccard_coeffs_data[i];
            std::cout << std::endl;

            if (i == print_count / 2 - 1 && print_count < nonzero_coeff_count)
                std::cout << "..." << (nonzero_coeff_count - print_count) << " lines skipped..."
                          << std::endl;
        }
    }
}
