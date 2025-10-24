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

#pragma once

#include "oneapi/dal/backend/dispatcher.hpp"

#include "oneapi/dal/algo/decision_forest/infer_types.hpp"

namespace oneapi::dal::decision_forest::parameters {

using dal::backend::context_cpu;

std::int64_t ONEDAL_EXPORT propose_block_size(const context_cpu& ctx);

constexpr std::int64_t propose_min_trees_for_threading() {
    return 100l;
}

constexpr std::int64_t propose_min_number_of_rows_for_vect_seq_compute() {
    return 32l;
}

constexpr double propose_scale_factor_for_vect_seq_compute() {
    return 0.3f;
}

template <typename Float, typename Method, typename Task>
struct infer_parameters_cpu {
    using params_t = detail::infer_parameters<Task>;
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const infer_input<Task>& input) const {
        const auto block = propose_block_size(ctx);
        const auto trees = propose_min_trees_for_threading();
        const auto seq_nrows = propose_min_number_of_rows_for_vect_seq_compute();
        const auto seq_trees = propose_scale_factor_for_vect_seq_compute();

        return params_t{}
            .set_block_size(block)
            .set_min_trees_for_threading(trees)
            .set_min_number_of_rows_for_vect_seq_compute(seq_nrows)
            .set_scale_factor_for_vect_parallel_compute(seq_trees);
    }
};
} // namespace oneapi::dal::decision_forest::parameters
