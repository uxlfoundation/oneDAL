/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/algo/svm/backend/gpu/misc.hpp"

namespace oneapi::dal::svm::backend {

namespace pr = dal::backend::primitives;

using sycl::ONEAPI::reduce;
using sycl::ONEAPI::maximum;
using sycl::ONEAPI::minimum;

template <typename Data>
using local_accessor_rw_t =
    sycl::accessor<Data, 1, sycl::access::mode::read_write, sycl::access::target::local>;

template <typename Float>
struct key_value {
    std::uint32_t index;
    Float value;
};

template <typename Float>
inline void reduce_arg_max(sycl::nd_item<1> item,
                           Float* objective_func,
                           key_value<Float>* local_cache,
                           key_value<Float>* result) {
    auto sg = item.get_sub_group();
    const std::uint32_t local_id = item.get_local_id(0);
    const std::uint32_t sg_count = item.get_group_range(0);
    const std::uint32_t sg_size = sg.get_local_range()[0];
    const std::uint32_t sg_id = sg.get_group_id(); // [0] or (0)
    const std::uint32_t sg_local_id = sg.get_local_id(); // [0] or (0)

    const std::uint32_t int_max = dal::detail::limits<std::uint32_t>::max();

    Float x = objective_func[local_id];
    std::uint32_t x_index = local_id;

    Float res_max = reduce(sg, x, maximum<Float>());
    std::uint32_t res_index =
        reduce(sg,
               res_max == x ? x_index : int_max,
               minimum<Float>());

    if (sg_local_id == 0) {
        local_cache[sg_id].value = res_max;
        local_cache[sg_id].index = res_index;
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (sg_id == 0 && sg_local_id < sg_count) {
        x = local_cache[sg_local_id].value;
        x_index = local_cache[sg_local_id].index;
        res_max = reduce(sg, x, maximum<Float>());
        res_index = reduce(sg,
                           res_max == x ? x_index : int_max,
                           minimum<Float>());

        for (std::uint32_t group_index = sg_size; group_index < sg_count; group_index += sg_size) {
            x = local_cache[group_index + sg_local_id].value;
            x_index = local_cache[group_index + sg_local_id].index;

            const Float inner_max = reduce(sg, x, maximum<Float>());
            if (inner_max > res_max) {
                res_max = inner_max;
                res_index =
                    reduce(sg,
                           res_max == x ? x_index : int_max,
                           minimum<Float>());
            }
        }

        if (sg_local_id == 0) {
            result[0].value = res_max;
            result[0].index = res_index;
        }
    }

    item.barrier(sycl::access::fence_space::local_space);
}

template <typename Float>
sycl::event solve_smo(sycl::queue& queue,
                      const pr::ndview<Float, 1>& kernel_values,
                      const pr::ndview<std::uint32_t, 1>& ws_indices,
                      const pr::ndarray<Float, 1>& labels,
                      const std::int64_t row_count,
                      const std::int64_t ws_count,
                      const std::int64_t max_inner_iter,
                      const Float C,
                      const Float eps,
                      const Float tau,
                      pr::ndview<Float, 1>& alpha,
                      pr::ndview<Float, 1>& delta_alpha,
                      pr::ndview<Float, 1>& f,
                      pr::ndview<Float, 1>& f_diff,
                      pr::ndview<std::uint32_t, 1>& inner_iter_count,
                      const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(row_count <= dal::detail::limits<std::uint32_t>::max());
    ONEDAL_ASSERT(row_count == labels.get_dimension(0));
    ONEDAL_ASSERT(max_inner_iter > 0);
    ONEDAL_ASSERT(max_inner_iter <= dal::detail::limits<std::uint32_t>::max());
    ONEDAL_ASSERT(ws_count > 0);
    ONEDAL_ASSERT(ws_count <= dal::detail::limits<std::uint32_t>::max());
    ONEDAL_ASSERT(ws_indices.get_dimension(0) == ws_count);
    ONEDAL_ASSERT(delta_alpha.get_dimension(0) == ws_count);
    ONEDAL_ASSERT(labels.get_dimension(0) == f.get_dimension(0));
    ONEDAL_ASSERT(alpha.get_dimension(0) == f.get_dimension(0));

    const Float fp_min = dal::detail::limits<Float>::min();

    const Float* labels_ptr = labels.get_data();
    const Float* kernel_values_ptr = kernel_values.get_data();
    const std::uint32_t* ws_indices_ptr = ws_indices.get_data();
    Float* alpha_ptr = alpha.get_mutable_data();
    Float* delta_alpha_ptr = delta_alpha.get_mutable_data();
    Float* f_ptr = f.get_mutable_data();
    Float* f_diff_ptr = f_diff.get_mutable_data();
    std::uint32_t* inner_iter_count_ptr = inner_iter_count.get_mutable_data();

    const sycl::nd_range<1> nd_range = dal::backend::make_multiple_nd_range_1d(ws_count, ws_count);
    auto solve_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<Float> kd(ws_count, cgh);
        local_accessor_rw_t<Float> objective_func(ws_count, cgh);
        local_accessor_rw_t<key_value<Float>> local_cache(64, cgh);  // SIMD_WIDTH = 64
        local_accessor_rw_t<Float> delta_Bi(1, cgh);
        local_accessor_rw_t<Float> delta_Bj(1, cgh);
        local_accessor_rw_t<key_value<Float>> max_val_ind(1, cgh);
        local_accessor_rw_t<Float> local_diff(1, cgh);
        local_accessor_rw_t<Float> local_eps(1, cgh);

        sycl::stream out(1024, 256, cgh);
        
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            const std::uint32_t i = item.get_local_id(0);

            const std::uint32_t ws_index = ws_indices_ptr[i];

            Float f_i = f_ptr[ws_index];
            Float aplha_i = alpha_ptr[ws_index];
            const Float old_alpha_i = aplha_i;
            const Float labels_i = labels_ptr[ws_index];

            

            std::uint32_t Bi = 0;
            std::uint32_t Bj = 0;

            Float* kd_ptr = kd.get_pointer().get();
            Float* objective_func_ptr = objective_func.get_pointer().get();
            key_value<Float>* local_cache_ptr = local_cache.get_pointer().get();
            Float* delta_Bi_ptr = delta_Bi.get_pointer().get();
            Float* delta_Bj_ptr = delta_Bj.get_pointer().get();
            key_value<Float>* max_val_ind_ptr = max_val_ind.get_pointer().get();
            Float* local_diff_ptr = local_diff.get_pointer().get();
            Float* local_eps_ptr = local_eps.get_pointer().get();

            kd_ptr[i] = kernel_values_ptr[i * row_count + ws_index];
            item.barrier(sycl::access::fence_space::local_space);

            std::uint32_t inner_iter = 0;
            for (; inner_iter < max_inner_iter; ++inner_iter) {
                /* m(alpha) = min(grad[i]): i belongs to I_UP (alpha) */
                objective_func_ptr[i] = is_upper_edge<Float>(labels_i, aplha_i, C) ? -f_i : fp_min;

                /* Find i index of the working set (Bi) */
                reduce_arg_max(item, objective_func_ptr, local_cache_ptr, max_val_ind_ptr);
                Bi = max_val_ind_ptr[0].index;
                const Float ma = -max_val_ind_ptr[0].value;

                /* maxgrad(alpha) = max(grad[i]): i belongs to I_low (alpha) */
                objective_func_ptr[i] = is_lower_edge<Float>(labels_i, aplha_i, C) ? f_i : fp_min;

                /* Find max gradient */
                reduce_arg_max(item, objective_func_ptr, local_cache_ptr, max_val_ind_ptr);

                if (i == 0) {
                    const Float max_f = max_val_ind_ptr[0].value;

                    /* for condition check: m(alpha) >= maxgrad */
                    local_diff_ptr[0] = max_f - ma;
                    if (inner_iter == 0) {
                        local_eps_ptr[0] = sycl::fmax(eps, local_diff_ptr[0] * Float(1e-1)); // 1e-1
                        f_diff_ptr[0] = local_diff_ptr[0];
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);
                if (local_diff_ptr[0] < local_eps_ptr[0]) {
                    break;
                }

                const Float Kii = kd_ptr[i];
                const Float KBiBi = kd_ptr[Bi];
                const Float KiBi = kernel_values_ptr[Bi * row_count + ws_index];

                if (is_lower_edge<Float>(labels_i, aplha_i, C) && ma < f_i) {
                    /* M(alpha) = max((b^2/a) : i belongs to I_low(alpha) and ma < f(alpha) */
                    const Float b = ma - f_i;
                    const Float a = sycl::fmax(Kii + KBiBi - Float(2.0) * KiBi, tau);
                    const Float dt = b / a;

                    objective_func_ptr[i] = b * dt;
                }
                else {
                    objective_func_ptr[i] = fp_min;
                }

                /* Find j index of the working set (Bj) */
                reduce_arg_max(item, objective_func_ptr, local_cache_ptr, max_val_ind_ptr);
                Bj = max_val_ind_ptr[0].index;

                const Float KiBj = kernel_values_ptr[Bj * row_count + ws_index];

                /* Update alpha */
                if (i == Bi) {
                    delta_Bi_ptr[0] = labels_i > 0 ? C - aplha_i : aplha_i;
                }
                if (i == Bj) {
                    delta_Bj_ptr[0] = labels_i > 0 ? aplha_i : C - aplha_i;
                    const Float b = ma - f_i;
                    const Float a = sycl::fmax(Kii + KBiBi - Float(2.0) * KiBi, tau);

                    const Float dt = -b / a;
                    delta_Bj_ptr[0] = sycl::fmin(delta_Bj_ptr[0], dt);
                }

                item.barrier(sycl::access::fence_space::local_space);

                const Float delta = sycl::fmin(delta_Bi_ptr[0], delta_Bj_ptr[0]);
                if (i == Bi) {
                    aplha_i = aplha_i + labels_i * delta;
                }
                if (i == Bj) {
                    aplha_i = aplha_i - labels_i * delta;
                }

                /* Update f */
                f_i = f_i + delta * (KiBi - KiBj);
            }
            alpha_ptr[ws_index] = aplha_i;
            delta_alpha_ptr[i] = (aplha_i - old_alpha_i) * labels_i;
            f_ptr[ws_index] = f_i;
            if (i == 0) {
                inner_iter_count_ptr[0] = inner_iter;
            }
        });
    });
    return solve_event;
}

} // namespace oneapi::dal::svm::backend