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

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::decision_forest::backend {

template <typename Float, typename Task>
struct train_kernel_gpu<Float, method::dense, Task> {
    train_result<Task> operator()(const dal::backend::context_gpu& ctx,
                                  const detail::descriptor_base<Task>& descr,
                                  const detail::train_parameters<Task>& params,
                                  const train_input<Task>& input) const {
        throw unimplemented(dal::detail::error_messages::
                                decision_forest_train_dense_method_is_not_implemented_for_gpu());
    }
};

template struct train_kernel_gpu<float, method::dense, task::classification>;
template struct train_kernel_gpu<double, method::dense, task::classification>;
template struct train_kernel_gpu<float, method::dense, task::regression>;
template struct train_kernel_gpu<double, method::dense, task::regression>;

} // namespace oneapi::dal::decision_forest::backend
