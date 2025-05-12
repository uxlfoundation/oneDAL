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

#include "oneapi/dal/algo/covariance/backend/cpu/compute_kernel_common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"

namespace oneapi::dal::covariance::backend {

template <typename Float, typename Task>
daal_hyperparameters_t convert_parameters(const detail::compute_parameters<Task>& params) {
    using daal::algorithms::covariance::internal::HyperparameterId;

    const std::int64_t block = params.get_cpu_macro_block();
    const std::int64_t grain_size = params.get_cpu_grain_size();

    daal_hyperparameters_t daal_hyperparameter;
    auto status = daal_hyperparameter.set(HyperparameterId::denseUpdateStepBlockSize, block);
    status |= daal_hyperparameter.set(HyperparameterId::denseUpdateStepGrainSize, grain_size);
    dal::backend::interop::status_to_exception(status);

    return daal_hyperparameter;
}

template daal_hyperparameters_t convert_parameters<double, task::compute>(
    const detail::compute_parameters<task::compute>&);
template daal_hyperparameters_t convert_parameters<float, task::compute>(
    const detail::compute_parameters<task::compute>&);
} // namespace oneapi::dal::covariance::backend
