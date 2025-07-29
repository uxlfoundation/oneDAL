/*
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
 */

#include "oneapi/dal/backend/interop/common.hpp"

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train_types.hpp"

namespace oneapi::dal::linear_regression::backend {

namespace daal_lr = daal::algorithms::linear_regression;
using daal_lr_hyperparameters_t = daal_lr::internal::Hyperparameter;
namespace interop = dal::backend::interop;

template <typename Float, typename Task>
static daal_lr_hyperparameters_t convert_parameters(const detail::train_parameters<Task>& params) {
    using daal_lr::internal::HyperparameterId;

    const std::int64_t block = params.get_cpu_macro_block();
    const std::int64_t max_cols_batched = params.get_cpu_max_cols_batched();
    const std::int64_t small_rows_threshold = params.get_cpu_small_rows_threshold();
    const std::int64_t small_rows_max_cols_batched = params.get_cpu_small_rows_max_cols_batched();
    const std::int64_t grain_size = params.get_cpu_grain_size();

    daal_lr_hyperparameters_t daal_hyperparameter;
    auto status = daal_hyperparameter.set(HyperparameterId::denseUpdateStepBlockSize, block);
    interop::status_to_exception(status);
    status = daal_hyperparameter.set(HyperparameterId::denseUpdateMaxColsBatched, max_cols_batched);
    interop::status_to_exception(status);
    status =
        daal_hyperparameter.set(HyperparameterId::denseSmallRowsThreshold, small_rows_threshold);
    interop::status_to_exception(status);
    status = daal_hyperparameter.set(HyperparameterId::denseSmallRowsMaxColsBatched,
                                     small_rows_max_cols_batched);
    interop::status_to_exception(status);

    status = daal_hyperparameter.set(HyperparameterId::denseGrainSize, grain_size);
    interop::status_to_exception(status);

    return daal_hyperparameter;
}

} // namespace oneapi::dal::linear_regression::backend
