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

#include "oneapi/dal/algo/spectral_embedding/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::spectral_embedding::backend {

template <typename Float, typename Task>
struct compute_kernel_gpu<Float, method::dense_batch, Task> {
    compute_result<Task> operator()(const dal::backend::context_gpu& ctx,
                                    const detail::descriptor_base<Task>& desc,
                                    const compute_input<Task>& input) const {
        // CHANGE ERROR MESSAGE
        throw unimplemented(
            dal::detail::error_messages::sp_emb_dense_batch_method_is_not_implemented_for_gpu());
    }
};

template struct compute_kernel_gpu<float, method::dense_batch, task::compute>;
template struct compute_kernel_gpu<double, method::dense_batch, task::compute>;

} // namespace oneapi::dal::spectral_embedding::backend
