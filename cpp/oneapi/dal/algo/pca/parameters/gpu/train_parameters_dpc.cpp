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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

#include "oneapi/dal/algo/pca/common.hpp"
#include "oneapi/dal/algo/pca/train_types.hpp"

#include "oneapi/dal/algo/pca/parameters/gpu/train_parameters.hpp"

namespace oneapi::dal::pca::parameters {

using dal::backend::context_gpu;

template <typename Float, typename Method>
struct train_parameters_gpu<Float, Method, task::dim_reduction> {
    using params_t = detail::train_parameters<task::dim_reduction>;
    params_t operator()(const context_gpu& ctx,
                        const detail::descriptor_base<task::dim_reduction>& desc,
                        const train_input<task::dim_reduction>& input) const {
        return params_t{};
    }
    params_t operator()(const context_gpu& ctx,
                        const detail::descriptor_base<task::dim_reduction>& desc,
                        const partial_train_input<task::dim_reduction>& input) const {
        return params_t{};
    }
    params_t operator()(const context_gpu& ctx,
                        const detail::descriptor_base<task::dim_reduction>& desc,
                        const partial_train_result<task::dim_reduction>& input) const {
        return params_t{};
    }
};

template struct ONEDAL_EXPORT train_parameters_gpu<float, method::cov, task::dim_reduction>;
template struct ONEDAL_EXPORT train_parameters_gpu<double, method::cov, task::dim_reduction>;
template struct ONEDAL_EXPORT train_parameters_gpu<float, method::precomputed, task::dim_reduction>;
template struct ONEDAL_EXPORT
    train_parameters_gpu<double, method::precomputed, task::dim_reduction>;
template struct ONEDAL_EXPORT train_parameters_gpu<float, method::svd, task::dim_reduction>;
template struct ONEDAL_EXPORT train_parameters_gpu<double, method::svd, task::dim_reduction>;

} // namespace oneapi::dal::pca::parameters
