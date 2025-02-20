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

#include "oneapi/dal/algo/decision_forest/parameters/gpu/train_parameters.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::decision_forest::parameters {

using dal::backend::context_gpu;
using task::classification;
using task::regression;

template <typename Float, typename Task>
struct train_parameters_gpu<Float, method::dense, Task> {
    using params_t = detail::train_parameters<Task>;
    params_t operator()(const context_gpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const train_input<Task>& input) const {
        return params_t{};
    }
};

template <typename Float, typename Task>
struct train_parameters_gpu<Float, method::hist, Task> {
    using params_t = detail::train_parameters<Task>;
    params_t operator()(const context_gpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const train_input<Task>& input) const {
        return params_t{};
    }
};

template struct ONEDAL_EXPORT train_parameters_gpu<float, method::dense, classification>;
template struct ONEDAL_EXPORT train_parameters_gpu<float, method::dense, regression>;
template struct ONEDAL_EXPORT train_parameters_gpu<float, method::hist, classification>;
template struct ONEDAL_EXPORT train_parameters_gpu<float, method::hist, regression>;
template struct ONEDAL_EXPORT train_parameters_gpu<double, method::dense, classification>;
template struct ONEDAL_EXPORT train_parameters_gpu<double, method::dense, regression>;
template struct ONEDAL_EXPORT train_parameters_gpu<double, method::hist, classification>;
template struct ONEDAL_EXPORT train_parameters_gpu<double, method::hist, regression>;

} // namespace oneapi::dal::decision_forest::parameters
