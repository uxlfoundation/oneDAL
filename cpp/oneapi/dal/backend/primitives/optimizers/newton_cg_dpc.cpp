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

#include "oneapi/dal/backend/primitives/optimizers/common.hpp"
#include "oneapi/dal/backend/primitives/optimizers/line_search.hpp"
#include "oneapi/dal/backend/primitives/optimizers/cg_solver.hpp"
#include "oneapi/dal/backend/primitives/optimizers/newton_cg.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include <cmath>

namespace oneapi::dal::backend::primitives {

template <typename Float>
std::tuple<sycl::event, std::int64_t, std::int64_t> newton_cg(sycl::queue& queue,
                                                              base_function<Float>& f,
                                                              ndview<Float, 1>& x,
                                                              Float tol,
                                                              std::int64_t maxiter,
                                                              std::int64_t maxinner,
                                                              const event_vector& deps) {
    ONEDAL_PROFILER_TASK(newton_cg, queue);
    std::int64_t n = x.get_dimension(0);

    const auto kernel_minus = [=](const Float val, Float) -> Float {
        return -val;
    };
    auto buffer = ndarray<Float, 1>::empty(queue, { 4 * n + 1 }, sycl::usm::alloc::device);

    auto buffer1 = buffer.get_slice(0, n);
    auto buffer2 = buffer.get_slice(n, 2 * n);
    auto buffer3 = buffer.get_slice(2 * n, 3 * n);
    auto direction = buffer.get_slice(3 * n, 4 * n);
    Float* tmp_gpu = buffer.get_mutable_data();
    tmp_gpu += n * 4;

    event_vector last_iter_deps = deps;
    sycl::event last = {};

    Float update_norm = tol + 1;

    std::int64_t cur_iter_id = 0;
    std::int64_t inner_iter_sum = 0;
    while (cur_iter_id < maxiter) {
        cur_iter_id++;
        auto update_event_vec = f.update_x(x, true, true, last_iter_deps);
        auto gradient = f.get_gradient();

        Float grad_norm = 0, grad_max_abs = 0;
        l1_norm(queue, gradient, tmp_gpu, &grad_norm, update_event_vec).wait_and_throw();
        max_abs(queue, gradient, tmp_gpu, &grad_max_abs, update_event_vec).wait_and_throw();

        if (grad_max_abs < tol) {
            // TODO check that conditions are the same across diferent devices
            break;
        }

        Float tol_k = std::min<Float>(sqrt(grad_norm), 0.5);

        // Note that this changes the sign of gradient stored in function (f) buffer
        // To reuse this gradient without recomputation, sign should be changed again
        auto prepare_grad_event =
            element_wise(queue, kernel_minus, gradient, Float(0), gradient, update_event_vec);

        // Initialize direction with 0
        auto init_dir_event = fill(queue, direction, Float(0), { prepare_grad_event });

        Float desc = -1;
        std::int32_t iter_num = 0;
        auto last_event = init_dir_event;
        while (desc < 0 && iter_num < 10) {
            // TODO check that conditions are the same across diferent devices
            if (iter_num > 0) {
                tol_k /= 10;
            }
            iter_num++;

            auto [solve_event, inner_iter] = cg_solve(queue,
                                                      f.get_hessian_product(),
                                                      gradient,
                                                      direction,
                                                      buffer1,
                                                      buffer2,
                                                      buffer3,
                                                      tol_k,
                                                      Float(0),
                                                      maxinner,
                                                      { last_event });
            inner_iter_sum += inner_iter;
            // <-grad, direction> should be > 0 if direction is descent direction
            last_event = dot_product(queue, gradient, direction, tmp_gpu, &desc, { solve_event });
            last_event.wait_and_throw();
        }
        if (desc < 0) {
            // failed to find descent direction
            return make_tuple(last_event, cur_iter_id, inner_iter_sum);
        }

        // Change the sign of gradient back to reuse it in backtracking function
        prepare_grad_event =
            element_wise(queue, kernel_minus, gradient, Float(0), gradient, { last_event });
        Float alpha_opt = backtracking(queue,
                                       f,
                                       x,
                                       direction,
                                       buffer2,
                                       Float(1),
                                       Float(1e-4),
                                       true,
                                       { last_event, prepare_grad_event });
        // Backtracking failed, early-stop
        if (alpha_opt == 0) {
            return make_tuple(last_event, cur_iter_id, inner_iter_sum);
        }

        update_norm = 0;
        dot_product(queue, direction, direction, tmp_gpu, &update_norm, { last_event })
            .wait_and_throw();

        update_norm = sqrt(update_norm) * alpha_opt;

        // updated x is in buffer2
        last = copy(queue, x, buffer2, {});
        last_iter_deps = { last };
    }
    return make_tuple(last, cur_iter_id, inner_iter_sum);
}

#define INSTANTIATE(F)                                                         \
    template std::tuple<sycl::event, std::int64_t, std::int64_t> newton_cg<F>( \
        sycl::queue&,                                                          \
        base_function<F>&,                                                     \
        ndview<F, 1>&,                                                         \
        F,                                                                     \
        std::int64_t,                                                          \
        std::int64_t,                                                          \
        const event_vector&);

INSTANTIATE(float);
INSTANTIATE(double);

} // namespace oneapi::dal::backend::primitives
