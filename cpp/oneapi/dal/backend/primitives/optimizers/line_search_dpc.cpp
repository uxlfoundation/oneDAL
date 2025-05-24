/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/backend/primitives/optimizers/line_search.hpp"
#include "oneapi/dal/backend/primitives/optimizers/common.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
Float backtracking(sycl::queue queue,
                   base_function<Float>& f,
                   const ndview<Float, 1>& x,
                   const ndview<Float, 1>& direction,
                   ndview<Float, 1>& result,
                   Float alpha,
                   Float c1,
                   bool x0_initialized,
                   const event_vector& deps) {
    ONEDAL_PROFILER_TASK(armijo_backtracking, queue);
    using dal::backend::operator+;
    event_vector precompute = {};
    if (!x0_initialized) {
        precompute = f.update_x(x, true, false, deps);
    }
    Float f0 = f.get_value();
    auto grad_f0 = f.get_gradient();
    Float df0 = 0;
    dot_product(queue, grad_f0, direction, result.get_mutable_data(), &df0, deps + precompute)
        .wait_and_throw();
    Float dir_inf_norm = 0;
    max_abs(queue, direction, result.get_mutable_data(), &dir_inf_norm, deps).wait_and_throw();
    Float cur_val = 0;
    constexpr Float eps = std::numeric_limits<Float>::epsilon();
    bool is_first_iter = true;
    while ((is_first_iter || cur_val - f0 > c1 * alpha * df0) && alpha * dir_inf_norm > eps) {
        // TODO check that conditions are the same across diferent devices
        if (!is_first_iter) {
            alpha /= 2;
        }
        else {
            is_first_iter = false;
        }
        const auto update_x_kernel = [=](const Float x_val, const Float dir_val) -> Float {
            return x_val + alpha * dir_val;
        };

        auto update_x_event = element_wise(queue, update_x_kernel, x, direction, result, {});
        auto func_event_vec = f.update_x(result, false, false, { update_x_event });
        wait_or_pass(func_event_vec).wait_and_throw();
        cur_val = f.get_value();
    }
    if (alpha * dir_inf_norm <= eps) {
        // if we add alpha * dir the vector of weights will not change
        alpha = 0;
    }
    return alpha;
}

#define INSTANTIATE(F)                           \
    template F backtracking(sycl::queue,         \
                            base_function<F>&,   \
                            const ndview<F, 1>&, \
                            const ndview<F, 1>&, \
                            ndview<F, 1>&,       \
                            F,                   \
                            F,                   \
                            bool,                \
                            const event_vector&);

INSTANTIATE(float);
INSTANTIATE(double);

} // namespace oneapi::dal::backend::primitives
