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

#include "oneapi/dal/algo/decision_forest/train_types.hpp"

namespace oneapi::dal::decision_forest::parameters {

using dal::backend::context_cpu;

/// Logic defining the values of the hyperparameters for the decision forest training
/// algorithm to go here.
/// The logic should be based on the context, descriptor, and input data.
/// For now the dummy implementation is provided.

template <typename Task>
std::int64_t propose_small_classes_threshold() {
    return detail::train_parameters<Task>{}.get_small_classes_threshold();
}

template <typename Task>
std::int64_t propose_min_part_coefficient() {
    return detail::train_parameters<Task>{}.get_min_part_coefficient();
}

template <typename Task>
std::int64_t propose_min_size_coefficient() {
    return detail::train_parameters<Task>{}.get_min_size_coefficient();
}

template <typename Float, typename Method, typename Task>
struct train_parameters_cpu {
    using params_t = detail::train_parameters<Task>;
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const train_input<Task>& input) const {
        return params_t{};
    }
};

template <typename Float, typename Method>
struct train_parameters_cpu<Float, Method, task::regression> {
    using task_t = task::regression;
    using params_t = detail::train_parameters<task_t>;
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<task_t>& desc,
                        const train_input<task_t>& input) const {
        const auto min_part_coeff = propose_min_part_coefficient<task_t>();
        const auto min_size_coeff = propose_min_size_coefficient<task_t>();
        return params_t{}
            .set_min_part_coefficient(min_part_coeff)
            .set_min_size_coefficient(min_size_coeff);
    }
};

template <typename Float, typename Method>
struct train_parameters_cpu<Float, Method, task::classification> {
    using task_t = task::classification;
    using params_t = detail::train_parameters<task_t>;
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<task_t>& desc,
                        const train_input<task_t>& input) const {
        const auto small_cls_thr = propose_small_classes_threshold<task_t>();
        const auto min_part_coeff = propose_min_part_coefficient<task_t>();
        const auto min_size_coeff = propose_min_size_coefficient<task_t>();
        return params_t{}
            .set_small_classes_threshold(small_cls_thr)
            .set_min_part_coefficient(min_part_coeff)
            .set_min_size_coefficient(min_size_coeff);
    }
};

} // namespace oneapi::dal::decision_forest::parameters
