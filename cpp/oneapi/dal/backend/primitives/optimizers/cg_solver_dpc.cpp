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

#include "oneapi/dal/backend/primitives/optimizers/cg_solver.hpp"
#include "oneapi/dal/backend/primitives/optimizers/common.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
std::pair<sycl::event, std::int64_t> cg_solve(sycl::queue& queue,
                                              base_matrix_operator<Float>& mul_operator,
                                              const ndview<Float, 1>& b,
                                              ndview<Float, 1>& x,
                                              ndview<Float, 1>& residual,
                                              ndview<Float, 1>& conj_vector,
                                              ndview<Float, 1>& buffer,
                                              Float tol,
                                              Float atol,
                                              std::int64_t maxiter,
                                              const event_vector& deps) {
    ONEDAL_PROFILER_TASK(cg_solve, queue);
    // Solving the equation mul_operator(x) = b
    const std::int64_t p = b.get_dimension(0);
    ONEDAL_ASSERT(b.has_data());
    ONEDAL_ASSERT(x.has_mutable_data());
    ONEDAL_ASSERT(residual.has_mutable_data());
    ONEDAL_ASSERT(conj_vector.has_mutable_data());
    ONEDAL_ASSERT(buffer.has_mutable_data());
    ONEDAL_ASSERT(x.get_dimension(0) == p);
    ONEDAL_ASSERT(residual.get_dimension(0) == p);
    ONEDAL_ASSERT(conj_vector.get_dimension(0) == p);
    ONEDAL_ASSERT(buffer.get_dimension(0) == p);

    using msg = dal::detail::error_messages;

    Float alpha = 0, beta = 0;
    Float r_norm = 0, b_l1_norm = 0, r_l1_norm = 0;

    const auto kernel_minus = [=](const Float a, const Float b) -> Float {
        return a - b;
    };

    auto compute_ax0_event = mul_operator(x, residual, deps); // r = Ax0

    auto compute_r0_event = element_wise(queue,
                                         kernel_minus,
                                         residual,
                                         b,
                                         residual,
                                         { compute_ax0_event }); // r0 = Ax0 - b

    auto tmp_gpu = ndarray<Float, 1>::empty(queue, { 1 }, sycl::usm::alloc::device);
    auto* const tmp_ptr = tmp_gpu.get_mutable_data();

    l1_norm<Float>(queue, b, tmp_ptr, &b_l1_norm, deps).wait_and_throw(); // compute l1_norm(b)

    // Tolerances for convergence norm(residual) <= max(tol*l1_norm(b), atol)
    Float threshold = std::max(tol * b_l1_norm, atol);

    const auto init_conj_kernel = [=](const Float residual_val, Float) -> Float {
        return -residual_val;
    };

    auto compute_conj_event = element_wise(queue,
                                           init_conj_kernel,
                                           residual,
                                           Float(0),
                                           conj_vector,
                                           { compute_r0_event }); // p0 = -r0 + 0 * p
    dot_product<Float>(queue, residual, residual, tmp_ptr, &r_norm, { compute_r0_event })
        .wait_and_throw(); // compute r^T r

    l1_norm<Float>(queue, residual, tmp_ptr, &r_l1_norm, { compute_r0_event })
        .wait_and_throw(); // compute l1_norm for stopping condition
    std::int64_t iter_num;
    for (iter_num = 0; iter_num < maxiter; ++iter_num) {
        if (r_l1_norm < threshold) {
            // TODO check that r_norms are the same across diferent devices
            break;
        }

        auto compute_matmul_event =
            mul_operator(conj_vector, buffer, { compute_conj_event }); // compute A p_i
        dot_product<Float>(queue, conj_vector, buffer, tmp_ptr, &alpha, { compute_matmul_event })
            .wait_and_throw(); // compute p_i^T A p_i

        if (alpha <= 0) {
            // if p^t A p is less or equal to zero then matrix A is not positively definite
            // if residual norm is higher than thr then p != 0
            throw domain_error(msg::matrix_is_not_positively_definite());
        }
        alpha = r_norm / alpha;
        const auto update_x_kernel = [=](const Float x_val, const Float conj_val) -> Float {
            return x_val + alpha * conj_val;
        };
        auto update_x_event = element_wise(queue,
                                           update_x_kernel,
                                           x,
                                           conj_vector,
                                           x,
                                           { compute_conj_event }); // x_i+1 = x_i + alpha * p_i
        auto update_residual_event =
            element_wise(queue,
                         update_x_kernel,
                         residual,
                         buffer,
                         residual,
                         { compute_matmul_event }); // r_i+1 = r_i + alpha * A p_i
        beta = r_norm;
        dot_product<Float>(queue,
                           residual,
                           residual,
                           tmp_ptr,
                           &r_norm,
                           { update_x_event, update_residual_event })
            .wait_and_throw(); // compute r^T r
        l1_norm<Float>(queue,
                       residual,
                       tmp_ptr,
                       &r_l1_norm,
                       { update_x_event, update_residual_event })
            .wait_and_throw(); // compute l1_norm for stopping condition

        beta = r_norm / beta; // beta = r_i+1^T r_i+1 / (r_i^T r_i)

        const auto update_conj_kernel = [=](const Float residual_val,
                                            const Float conj_val) -> Float {
            return -residual_val + beta * conj_val;
        };
        compute_conj_event =
            element_wise(queue,
                         update_conj_kernel,
                         residual,
                         conj_vector,
                         conj_vector,
                         { update_x_event, update_residual_event }); // p_i+1 = -r_i+1 + beta * p_i
    }

    return { compute_conj_event, iter_num };
}

#define INSTANTIATE(F)                                                                  \
    template std::pair<sycl::event, std::int64_t> cg_solve<F>(sycl::queue&,             \
                                                              base_matrix_operator<F>&, \
                                                              const ndview<F, 1>&,      \
                                                              ndview<F, 1>&,            \
                                                              ndview<F, 1>&,            \
                                                              ndview<F, 1>&,            \
                                                              ndview<F, 1>&,            \
                                                              F,                        \
                                                              F,                        \
                                                              std::int64_t,             \
                                                              const event_vector&);

INSTANTIATE(float);
INSTANTIATE(double);

} // namespace oneapi::dal::backend::primitives
