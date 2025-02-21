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

#include "oneapi/dal/algo/decision_forest/parameters/cpu/train_parameters.hpp"
#include "oneapi/dal/algo/decision_forest/parameters/gpu/train_parameters.hpp"
#include "oneapi/dal/algo/decision_forest/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/decision_forest/detail/train_ops.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::decision_forest::detail {
namespace v1 {

template <typename Policy, typename Float, typename Task, typename Method>
struct train_ops_dispatcher<Policy, Float, Task, Method> {
    using input_t = oneapi::dal::decision_forest::v2::train_input<Task>;
    train_result<Task> operator()(const Policy& policy,
                                  const descriptor_base<Task>& desc,
                                  const train_parameters<Task>& params,
                                  const input_t& input) const {
        return implementation(policy, desc, params, input);
    }
    train_parameters<Task> select_parameters(const Policy& policy,
                                             const descriptor_base<Task>& desc,
                                             const input_t& input) const {
        using kernel_dispatcher_t = dal::backend::kernel_dispatcher<
            KERNEL_SINGLE_NODE_CPU(parameters::train_parameters_cpu<Float, Method, Task>),
            KERNEL_UNIVERSAL_SPMD_GPU(parameters::train_parameters_gpu<Float, Method, Task>)>;
        return kernel_dispatcher_t{}(policy, desc, input);
    }
    train_result<Task> operator()(const Policy& policy,
                                  const descriptor_base<Task>& desc,
                                  const input_t& input) const {
        const auto params = select_parameters(policy, desc, input);
        return implementation(policy, desc, params, input);
    }

private:
    inline auto implementation(const Policy& policy,
                               const descriptor_base<Task>& desc,
                               const train_parameters<Task>& params,
                               const input_t& input) const {
        using kernel_dispatcher_t = dal::backend::kernel_dispatcher<
            KERNEL_SINGLE_NODE_CPU(backend::train_kernel_cpu<Float, Method, Task>),
            KERNEL_UNIVERSAL_SPMD_GPU(backend::train_kernel_gpu<Float, Method, Task>)>;
        return kernel_dispatcher_t{}(policy, desc, params, input);
    }
};

#define INSTANTIATE(F, T, M)                                              \
    template struct ONEDAL_EXPORT                                         \
        train_ops_dispatcher<dal::detail::data_parallel_policy, F, T, M>; \
    template struct ONEDAL_EXPORT                                         \
        train_ops_dispatcher<dal::detail::spmd_data_parallel_policy, F, T, M>;

INSTANTIATE(float, task::classification, method::dense)
INSTANTIATE(float, task::classification, method::hist)
INSTANTIATE(double, task::classification, method::dense)
INSTANTIATE(double, task::classification, method::hist)

INSTANTIATE(float, task::regression, method::dense)
INSTANTIATE(float, task::regression, method::hist)
INSTANTIATE(double, task::regression, method::dense)
INSTANTIATE(double, task::regression, method::hist)

} // namespace v1
} // namespace oneapi::dal::decision_forest::detail
