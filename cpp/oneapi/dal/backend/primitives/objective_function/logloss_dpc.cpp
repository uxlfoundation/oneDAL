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

#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas.hpp"

namespace oneapi::dal::backend::primitives {

namespace pr = dal::backend::primitives;

template <typename Float>
sycl::event compute_probabilities(sycl::queue& q,
                                  const ndview<Float, 1>& parameters,
                                  const ndview<Float, 2>& data,
                                  ndview<Float, 1>& probabilities,
                                  bool fit_intercept,
                                  const event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_probabilities, q);
    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(parameters.has_data());
    ONEDAL_ASSERT(probabilities.has_mutable_data());
    ONEDAL_ASSERT(parameters.get_dimension(0) == fit_intercept ? p + 1 : p);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);

    auto fill_event = fill<Float>(q, probabilities, Float(1), deps);
    using oneapi::dal::backend::operator+;

    Float w0 =
        fit_intercept ? parameters.get_slice(0, 1).at_device(q, 0l, deps) : 0; // Poor perfomance
    ndview<Float, 1> param_suf = fit_intercept ? parameters.get_slice(1, p + 1) : parameters;

    sycl::event gemv_event;
    {
        gemv_event = gemv(q, data, param_suf, probabilities, Float(1), w0, { fill_event });
        gemv_event.wait_and_throw();
    }
    auto* const prob_ptr = probabilities.get_mutable_data();

    // Note: these same thresholds are also used in the non-DPC version.
    // If modified, should be modified simulatenously in both files.
    const Float bottom = sizeof(Float) == 4 ? 1e-7 : 1e-15;
    const Float top = Float(1.0) - bottom;
    // Log Loss is undefined for p = 0 and p = 1 so probabilities are clipped into [eps, 1 - eps]

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(gemv_event);
        const auto range = make_range_1d(n);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            prob_ptr[idx] = 1 / (1 + sycl::exp(-prob_ptr[idx]));
            if (prob_ptr[idx] < bottom) {
                prob_ptr[idx] = bottom;
            }
            if (prob_ptr[idx] > top) {
                prob_ptr[idx] = top;
            }
        });
    });
}

template <typename Float>
sycl::event compute_probabilities_sparse(sycl::queue& q,
                                         const ndview<Float, 1>& parameters,
                                         sparse_matrix_handle& sp_handler,
                                         ndview<Float, 1>& probabilities,
                                         bool fit_intercept,
                                         const event_vector& deps) {
    ONEDAL_ASSERT(probabilities.has_mutable_data());
    ONEDAL_PROFILER_TASK(compute_probabilities_sparse, q);

    const std::int64_t n = probabilities.get_dimension(0);
    const std::int64_t p = parameters.get_dimension(0) - (fit_intercept ? 1 : 0);

    auto fill_event = fill<Float>(q, probabilities, Float(1), deps);
    Float w0 =
        fit_intercept ? parameters.get_slice(0, 1).at_device(q, 0l, deps) : 0; // Poor perfomance
    ndview<Float, 1> param_suf = fit_intercept ? parameters.get_slice(1, p + 1) : parameters;

    sycl::event gemv_event;
    {
        gemv_event = gemv(q,
                          transpose::nontrans,
                          sp_handler,
                          param_suf,
                          probabilities,
                          Float(1),
                          w0,
                          { fill_event });
        // To ensure sparse blas kernel stability
        gemv_event.wait_and_throw();
    }

    auto* const prob_ptr = probabilities.get_mutable_data();

    const Float bottom = sizeof(Float) == 4 ? 1e-7 : 1e-15;
    const Float top = Float(1.0) - bottom;
    // Log Loss is undefined for p = 0 and p = 1 so probabilities are clipped into [eps, 1 - eps]

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(gemv_event);
        const auto range = make_range_1d(n);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            prob_ptr[idx] = 1 / (1 + sycl::exp(-prob_ptr[idx]));
            if (prob_ptr[idx] < bottom) {
                prob_ptr[idx] = bottom;
            }
            if (prob_ptr[idx] > top) {
                prob_ptr[idx] = top;
            }
        });
    });
}

template <typename Float>
sycl::event compute_logloss(sycl::queue& q,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 1>& out,
                            bool fit_intercept,
                            const event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_logloss, q);
    const std::int64_t n = labels.get_dimension(0);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);
    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(probabilities.has_data());

    const auto* const labels_ptr = labels.get_data();
    const auto* const prob_ptr = probabilities.get_data();

    auto* const out_ptr = out.get_mutable_data();

    auto loss_event = q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_1d(n);
        using oneapi::dal::backend::operator+;
        using sycl::reduction;

        cgh.depends_on(deps);

        auto sum_reduction = reduction(out_ptr, sycl::plus<>());

        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum) {
            const Float prob = prob_ptr[idx];
            const std::int32_t label = labels_ptr[idx];
            sum += -label * sycl::log(prob) - (1 - label) * sycl::log(1 - prob);
        });
    });
    return loss_event;
}

template <typename Float>
sycl::event compute_logloss_with_der_sparse(sycl::queue& q,
                                            sparse_matrix_handle& sp_handler,
                                            const ndview<std::int32_t, 1>& labels,
                                            const ndview<Float, 1>& probabilities,
                                            ndview<Float, 1>& out,
                                            ndview<Float, 1>& out_derivative,
                                            bool fit_intercept,
                                            const event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_logloss_with_grad_sparse, q);

    const std::int64_t n = labels.get_dimension(0);
    const std::int64_t p = out_derivative.get_dimension(0) - (fit_intercept ? 1 : 0);

    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(probabilities.has_data());
    ONEDAL_ASSERT(out.has_mutable_data());
    ONEDAL_ASSERT(out_derivative.has_mutable_data());
    ONEDAL_ASSERT(out.get_dimension(0) == 1);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);

    auto derivative_object = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);

    auto* const der_obj_ptr = derivative_object.get_mutable_data();
    const auto* const proba_ptr = probabilities.get_data();
    const auto* const labels_ptr = labels.get_data();
    auto* const out_ptr = out.get_mutable_data();
    auto* const out_derivative_ptr = out_derivative.get_mutable_data();

    auto loss_event = q.submit([&](sycl::handler& cgh) {
        using oneapi::dal::backend::operator+;
        using sycl::reduction;

        cgh.depends_on(deps);
        auto sum_reduction_logloss = reduction(out_ptr, sycl::plus<>());
        const auto wg_size = propose_wg_size(q);
        const auto range = make_multiple_nd_range_1d(n, wg_size);

        cgh.parallel_for(range, sum_reduction_logloss, [=](sycl::nd_item<1> id, auto& sum_logloss) {
            auto idx = id.get_group_linear_id() * wg_size + id.get_local_linear_id();
            if (idx >= std::size_t(n))
                return;
            const Float prob = proba_ptr[idx];
            const float label = labels_ptr[idx];
            sum_logloss += -label * sycl::log(prob) - (1 - label) * sycl::log(1 - prob);
            der_obj_ptr[idx] = prob - label;
        });
    });
    sycl::event derw0_event = sycl::event{};
    if (fit_intercept) {
        derw0_event = q.submit([&](sycl::handler& cgh) {
            using oneapi::dal::backend::operator+;
            using sycl::reduction;

            cgh.depends_on(deps + loss_event);
            auto sum_reduction_derivative_w0 = reduction(out_derivative_ptr, sycl::plus<>());
            const auto wg_size = propose_wg_size(q);
            const auto range = make_multiple_nd_range_1d(n, wg_size);

            cgh.parallel_for(range,
                             sum_reduction_derivative_w0,
                             [=](sycl::nd_item<1> id, auto& sum_dw0) {
                                 auto idx =
                                     id.get_group_linear_id() * wg_size + id.get_local_linear_id();
                                 if (idx >= std::size_t(n))
                                     return;
                                 sum_dw0 += der_obj_ptr[idx];
                             });
        });
    }

    auto out_der_suffix = fit_intercept ? out_derivative.get_slice(1, p + 1) : out_derivative;
    sycl::event gemv_event;
    {
        gemv_event = gemv(q,
                          transpose::trans,
                          sp_handler,
                          derivative_object,
                          out_der_suffix,
                          Float(1),
                          Float(0),
                          { loss_event, derw0_event });
    }
    return gemv_event;
}

template <typename Float>
sycl::event compute_logloss_with_der(sycl::queue& q,
                                     const ndview<Float, 2>& data,
                                     const ndview<std::int32_t, 1>& labels,
                                     const ndview<Float, 1>& probabilities,
                                     ndview<Float, 1>& out,
                                     ndview<Float, 1>& out_derivative,
                                     bool fit_intercept,
                                     const event_vector& deps) {
    // out, out_derivative should be filled with zeros
    ONEDAL_PROFILER_TASK(compute_logloss_with_grad, q);
    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(labels.get_dimension(0) == n);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);
    ONEDAL_ASSERT(out.get_dimension(0) == 1);
    ONEDAL_ASSERT(out_derivative.get_dimension(0) == fit_intercept ? p + 1 : p);

    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(probabilities.has_data());
    ONEDAL_ASSERT(out.has_mutable_data());
    ONEDAL_ASSERT(out_derivative.has_mutable_data());

    // d loss_i / d pred_i
    auto derivative_object = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);

    auto* const der_obj_ptr = derivative_object.get_mutable_data();
    const auto* const proba_ptr = probabilities.get_data();
    const auto* const labels_ptr = labels.get_data();
    auto* const out_ptr = out.get_mutable_data();
    auto* const out_derivative_ptr = out_derivative.get_mutable_data();

    auto loss_event = q.submit([&](sycl::handler& cgh) {
        using oneapi::dal::backend::operator+;
        using sycl::reduction;

        cgh.depends_on(deps);
        auto sum_reduction_logloss = reduction(out_ptr, sycl::plus<>());
        const auto wg_size = propose_wg_size(q);
        const auto range = make_multiple_nd_range_1d(n, wg_size);

        cgh.parallel_for(range, sum_reduction_logloss, [=](sycl::nd_item<1> id, auto& sum_logloss) {
            auto idx = id.get_group_linear_id() * wg_size + id.get_local_linear_id();
            if (idx >= std::size_t(n))
                return;
            const Float prob = proba_ptr[idx];
            const float label = labels_ptr[idx];
            sum_logloss += -label * sycl::log(prob) - (1 - label) * sycl::log(1 - prob);
            der_obj_ptr[idx] = prob - label;
        });
    });
    sycl::event derw0_event = sycl::event{};
    if (fit_intercept) {
        derw0_event = q.submit([&](sycl::handler& cgh) {
            using oneapi::dal::backend::operator+;
            using sycl::reduction;

            cgh.depends_on(deps + loss_event);
            auto sum_reduction_derivative_w0 = reduction(out_derivative_ptr, sycl::plus<>());
            const auto wg_size = propose_wg_size(q);
            const auto range = make_multiple_nd_range_1d(n, wg_size);

            cgh.parallel_for(range,
                             sum_reduction_derivative_w0,
                             [=](sycl::nd_item<1> id, auto& sum_dw0) {
                                 auto idx =
                                     id.get_group_linear_id() * wg_size + id.get_local_linear_id();
                                 if (idx >= std::size_t(n))
                                     return;
                                 sum_dw0 += der_obj_ptr[idx];
                             });
        });
    }

    auto out_der_suffix = fit_intercept ? out_derivative.get_slice(1, p + 1) : out_derivative;
    sycl::event gemv_event;
    {
        gemv_event =
            gemv(q, data.t(), derivative_object, out_der_suffix, { loss_event, derw0_event });
        gemv_event.wait_and_throw();
    }
    return gemv_event;
}

template <typename Float>
sycl::event compute_derivative(sycl::queue& q,
                               const ndview<Float, 2>& data,
                               const ndview<std::int32_t, 1>& labels,
                               const ndview<Float, 1>& probabilities,
                               ndview<Float, 1>& out_derivative,
                               bool fit_intercept,
                               const event_vector& deps) {
    // out_derivative should be filled with zeros
    ONEDAL_PROFILER_TASK(compute_logloss_grad, q);
    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(labels.get_dimension(0) == n);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);
    ONEDAL_ASSERT(out_derivative.get_dimension(0) == fit_intercept ? p + 1 : p);

    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(probabilities.has_data());
    ONEDAL_ASSERT(out_derivative.has_mutable_data());

    // d loss_i / d pred_i
    auto derivative_object = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);

    auto* const der_obj_ptr = derivative_object.get_mutable_data();
    const auto* const proba_ptr = probabilities.get_data();
    const auto* const labels_ptr = labels.get_data();
    auto* const out_derivative_ptr = out_derivative.get_mutable_data();

    auto loss_event = q.submit([&](sycl::handler& cgh) {
        using sycl::reduction;

        cgh.depends_on(deps);
        const auto wg_size = propose_wg_size(q);
        const auto range = make_multiple_nd_range_1d(n, wg_size);
        if (fit_intercept) {
            auto sum_reduction_derivative_w0 = reduction(out_derivative_ptr, sycl::plus<>());
            cgh.parallel_for(range,
                             sum_reduction_derivative_w0,
                             [=](sycl::nd_item<1> id, auto& sum_dw0) {
                                 auto idx =
                                     id.get_group_linear_id() * wg_size + id.get_local_linear_id();
                                 if (idx >= std::size_t(n))
                                     return;
                                 const Float prob = proba_ptr[idx];
                                 const Float label = labels_ptr[idx];
                                 der_obj_ptr[idx] = prob - label;
                                 sum_dw0 += der_obj_ptr[idx];
                             });
        }
        else {
            cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
                auto idx = id.get_group_linear_id() * wg_size + id.get_local_linear_id();
                if (idx >= std::size_t(n))
                    return;
                const Float prob = proba_ptr[idx];
                const Float label = labels_ptr[idx];
                der_obj_ptr[idx] = prob - label;
            });
        }
    });

    auto out_der_suffix = fit_intercept ? out_derivative.get_slice(1, p + 1) : out_derivative;

    sycl::event der_event;
    {
        der_event = gemv(q, data.t(), derivative_object, out_der_suffix, { loss_event });
        der_event.wait_and_throw();
    }
    return der_event;
}

template <typename Float>
sycl::event add_regularization_loss(sycl::queue& q,
                                    const ndview<Float, 1>& parameters,
                                    ndview<Float, 1>& out,
                                    Float L1,
                                    Float L2,
                                    bool fit_intercept,
                                    const event_vector& deps) {
    ONEDAL_PROFILER_TASK(add_regularization_loss, q);
    using dal::backend::operator+;
    auto [out_reg, out_reg_e] = ndarray<Float, 1>::zeros(q, { 1 }, sycl::usm::alloc::device);
    auto* const reg_ptr = out_reg.get_mutable_data();
    auto* const out_ptr = out.get_mutable_data();
    const auto* const param_ptr = parameters.get_data();
    auto new_deps = deps + out_reg_e;
    const std::int64_t p =
        fit_intercept ? parameters.get_dimension(0) - 1 : parameters.get_dimension(0);
    auto reg_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(new_deps);
        const auto range = make_range_1d(p);
        auto sum_reduction = sycl::reduction(reg_ptr, sycl::plus<>());
        const std::int64_t st_id = fit_intercept;
        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum) {
            const Float param = param_ptr[idx + st_id];
            sum += L1 * sycl::fabs(param) + L2 * param * param;
        });
    });
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ reg_event });
        cgh.single_task([=] {
            *out_ptr += *reg_ptr;
        });
    });
}

template <typename Float>
sycl::event add_regularization_gradient_loss(sycl::queue& q,
                                             const ndview<Float, 1>& parameters,
                                             ndview<Float, 1>& out,
                                             ndview<Float, 1>& out_derivative,
                                             Float L1,
                                             Float L2,
                                             bool fit_intercept,
                                             const event_vector& deps) {
    ONEDAL_PROFILER_TASK(add_regularization_grad_loss, q);
    using dal::backend::operator+;
    auto [reg_val, reg_val_e] = ndarray<Float, 1>::zeros(q, { 1 }, sycl::usm::alloc::device);

    const std::int64_t p =
        fit_intercept ? parameters.get_dimension(0) - 1 : parameters.get_dimension(0);

    const auto* const param_ptr = parameters.get_data();
    auto* const reg_ptr = reg_val.get_mutable_data();
    auto* const out_ptr = out.get_mutable_data();
    auto* const grad_ptr = out_derivative.get_mutable_data();
    auto new_deps = deps + reg_val_e;
    auto reg_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(new_deps);
        const auto range = make_range_1d(p);
        auto sum_reduction = sycl::reduction(reg_ptr, sycl::plus<>());
        std::int64_t st_id = fit_intercept;
        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum) {
            const Float param = param_ptr[idx + st_id];
            sum += L1 * sycl::fabs(param) + L2 * param * param;
            grad_ptr[idx + st_id] += L2 * 2 * param;
        });
    });

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ reg_event });
        cgh.single_task([=] {
            *out_ptr += *reg_ptr;
        });
    });
}

template <typename Float>
sycl::event add_regularization_gradient(sycl::queue& q,
                                        const ndview<Float, 1>& parameters,
                                        ndview<Float, 1>& out_derivative,
                                        Float L1,
                                        Float L2,
                                        bool fit_intercept,
                                        const event_vector& deps) {
    ONEDAL_PROFILER_TASK(add_regularization_grad, q);
    auto* const grad_ptr = out_derivative.get_mutable_data();
    const auto* const param_ptr = parameters.get_data();
    const std::int64_t p =
        fit_intercept ? parameters.get_dimension(0) - 1 : parameters.get_dimension(0);
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        const auto range = make_range_1d(p);
        std::int64_t st_id = fit_intercept;
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            const Float param = param_ptr[idx + st_id];
            grad_ptr[idx + st_id] += L2 * 2 * param;
        });
    });
}

template <typename Float>
sycl::event compute_hessian(sycl::queue& q,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 2>& out_hessian,
                            const Float L1,
                            const Float L2,
                            bool fit_intercept,
                            const event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_logloss_hessian, q);
    const int64_t n = data.get_dimension(0);
    const int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(labels.get_dimension(0) == n);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);
    ONEDAL_ASSERT(out_hessian.get_dimension(0) == (p + 1));
    ONEDAL_ASSERT(out_hessian.get_dimension(1) == (p + 1));

    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(probabilities.has_data());
    ONEDAL_ASSERT(out_hessian.has_mutable_data());

    const auto* const data_ptr = data.get_data();
    auto* const hes_ptr = out_hessian.get_mutable_data();
    const auto* const proba_ptr = probabilities.get_data();

    const auto max_wg = device_max_wg_size(q);
    const auto wg = std::min(p + 1, max_wg);
    const auto inp_str = data.get_leading_stride();
    const auto out_str = out_hessian.get_leading_stride();

    constexpr std::int64_t block_size = 32;
    const std::int64_t num_blocks = (n + block_size - 1) / block_size;
    const auto range = make_multiple_nd_range_3d({ num_blocks, p + 1, wg }, { 1, 1, wg });

    auto hes_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::nd_item<3> item) {
            const std::int64_t obj_ind = item.get_global_id(0);
            const auto j = item.get_global_id(1);
            const auto param_ind_2 = item.get_global_id(2);
            Float val = 0;
            for (auto k = param_ind_2; k < j + 1; k += wg) {
                if (!fit_intercept && (j == 0 || k == 0)) {
                    continue;
                }
                val = 0;
                const std::int64_t last_ind = std::min((obj_ind + 1) * block_size, n);
                for (auto i = obj_ind * block_size; i < last_ind; ++i) {
                    const Float x1 = j > 0 ? data_ptr[i * inp_str + (j - 1)] : 1;
                    const Float x2 = k > 0 ? data_ptr[i * inp_str + (k - 1)] : 1;
                    const Float prob = proba_ptr[i] * (1 - proba_ptr[i]);
                    val += x1 * x2 * prob;
                }
                Float& out = hes_ptr[j * out_str + k];
                sycl::atomic_ref<Float,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::ext_intel_global_device_space>(out)
                    .fetch_add(val);
            }
        });
    });

    auto make_symmetric = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ hes_event });
        const auto range = make_range_2d(p + 1, p + 1);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            auto j = idx[0];
            auto k = idx[1];
            if (j > k) {
                hes_ptr[k * out_str + j] = hes_ptr[j * out_str + k];
            }
            else if (j == k && j > 0) {
                hes_ptr[j * out_str + j] += 2 * L2;
            }
        });
    });

    return make_symmetric;
}

template <typename Float>
sycl::event compute_raw_hessian(sycl::queue& q,
                                const ndview<Float, 1>& probabilities,
                                ndview<Float, 1>& out_hessian,
                                const event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_raw_hessian, q);
    const std::int64_t n = probabilities.get_dimension(0);

    ONEDAL_ASSERT(out_hessian.get_dimension(0) == n);
    ONEDAL_ASSERT(probabilities.has_data());
    ONEDAL_ASSERT(out_hessian.has_mutable_data());
    const auto kernel = [=](const Float val, Float) -> Float {
        constexpr Float one(1);
        return val * (one - val);
    };
    return element_wise(q, kernel, probabilities, Float(0), out_hessian, deps);
}

#define INSTANTIATE(F)                                                                      \
    template sycl::event compute_probabilities<F>(sycl::queue&,                             \
                                                  const ndview<F, 1>&,                      \
                                                  const ndview<F, 2>&,                      \
                                                  ndview<F, 1>&,                            \
                                                  bool,                                     \
                                                  const event_vector&);                     \
    template sycl::event compute_probabilities_sparse<F>(sycl::queue&,                      \
                                                         const ndview<F, 1>&,               \
                                                         sparse_matrix_handle&,             \
                                                         ndview<F, 1>&,                     \
                                                         bool,                              \
                                                         const event_vector&);              \
    template sycl::event compute_logloss<F>(sycl::queue&,                                   \
                                            const ndview<std::int32_t, 1>&,                 \
                                            const ndview<F, 1>&,                            \
                                            ndview<F, 1>&,                                  \
                                            bool,                                           \
                                            const event_vector&);                           \
    template sycl::event compute_logloss_with_der<F>(sycl::queue&,                          \
                                                     const ndview<F, 2>&,                   \
                                                     const ndview<std::int32_t, 1>&,        \
                                                     const ndview<F, 1>&,                   \
                                                     ndview<F, 1>&,                         \
                                                     ndview<F, 1>&,                         \
                                                     bool,                                  \
                                                     const event_vector&);                  \
    template sycl::event compute_logloss_with_der_sparse<F>(sycl::queue&,                   \
                                                            sparse_matrix_handle&,          \
                                                            const ndview<std::int32_t, 1>&, \
                                                            const ndview<F, 1>&,            \
                                                            ndview<F, 1>&,                  \
                                                            ndview<F, 1>&,                  \
                                                            bool,                           \
                                                            const event_vector&);           \
    template sycl::event compute_derivative<F>(sycl::queue&,                                \
                                               const ndview<F, 2>&,                         \
                                               const ndview<std::int32_t, 1>&,              \
                                               const ndview<F, 1>&,                         \
                                               ndview<F, 1>&,                               \
                                               bool,                                        \
                                               const event_vector&);                        \
    template sycl::event add_regularization_loss<F>(sycl::queue&,                           \
                                                    const ndview<F, 1>&,                    \
                                                    ndview<F, 1>&,                          \
                                                    F,                                      \
                                                    F,                                      \
                                                    bool,                                   \
                                                    const event_vector&);                   \
    template sycl::event add_regularization_gradient_loss<F>(sycl::queue&,                  \
                                                             const ndview<F, 1>&,           \
                                                             ndview<F, 1>&,                 \
                                                             ndview<F, 1>&,                 \
                                                             F,                             \
                                                             F,                             \
                                                             bool,                          \
                                                             const event_vector&);          \
    template sycl::event add_regularization_gradient<F>(sycl::queue&,                       \
                                                        const ndview<F, 1>&,                \
                                                        ndview<F, 1>&,                      \
                                                        F,                                  \
                                                        F,                                  \
                                                        bool,                               \
                                                        const event_vector&);               \
    template sycl::event compute_hessian<F>(sycl::queue&,                                   \
                                            const ndview<F, 2>&,                            \
                                            const ndview<std::int32_t, 1>&,                 \
                                            const ndview<F, 1>&,                            \
                                            ndview<F, 2>&,                                  \
                                            const F,                                        \
                                            const F,                                        \
                                            bool,                                           \
                                            const event_vector&);                           \
    template sycl::event compute_raw_hessian<F>(sycl::queue&,                               \
                                                const ndview<F, 1>&,                        \
                                                ndview<F, 1>&,                              \
                                                const event_vector&);

INSTANTIATE(float);
INSTANTIATE(double);

} // namespace oneapi::dal::backend::primitives
