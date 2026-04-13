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

#include "oneapi/dal/algo/svm/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/model_interop.hpp"
#include "oneapi/dal/algo/svm/backend/model_conversion.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "oneapi/dal/algo/svm/backend/utils.hpp"

#include "oneapi/dal/algo/svm/backend/gpu/misc.hpp"
#include "oneapi/dal/algo/svm/backend/gpu/svm_cache.hpp"
#include "oneapi/dal/algo/svm/backend/gpu/train_results.hpp"
#include "oneapi/dal/algo/svm/backend/gpu/working_set_selector.hpp"
#include "oneapi/dal/algo/svm/backend/gpu/smo_solver.hpp"

#include "oneapi/dal/backend/primitives/blas.hpp"

namespace oneapi::dal::svm::backend {

using dal::backend::context_gpu;
using model_t = model<task::classification>;
using input_t = train_input<task::classification>;
using result_t = train_result<task::classification>;
using descriptor_t = detail::descriptor_base<task::classification>;

namespace pr = dal::backend::primitives;
namespace de = dal::detail;

template <typename Float>
inline auto update_grad(sycl::queue& q,
                        const pr::ndview<Float, 2>& kernel_values_nd,
                        const pr::ndview<Float, 1>& delta_alpha_nd,
                        pr::ndview<Float, 1>& grad_nd) {
    ONEDAL_ASSERT(kernel_values_nd.get_dimension(0) == delta_alpha_nd.get_dimension(0));
    ONEDAL_ASSERT(kernel_values_nd.get_dimension(1) == grad_nd.get_dimension(0));
    ONEDAL_PROFILER_TASK(update_grad, q);
    auto reshape_delta =
        delta_alpha_nd.reshape(pr::ndshape<2>{ delta_alpha_nd.get_dimension(0), 1 });
    auto reshape_grad = grad_nd.reshape(pr::ndshape<2>{ grad_nd.get_dimension(0), 1 });
    auto gemm_event =
        pr::gemm(q, kernel_values_nd.t(), reshape_delta, reshape_grad, Float(1), Float(1));
    return gemm_event;
}

template <typename Float>
inline bool check_stop_condition(const Float diff,
                                 const Float prev_diff,
                                 const Float eps,
                                 std::int32_t& same_local_diff_count) {
    constexpr std::int32_t max_unchanged_repetitions = 5;
    same_local_diff_count = std::abs(diff - prev_diff) < eps * 1e-2 ? same_local_diff_count + 1 : 0;

    if (same_local_diff_count > max_unchanged_repetitions || diff < eps) {
        return true;
    }
    return false;
}

template <typename Float>
static result_t train(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto& q = ctx.get_queue();

    const std::uint64_t class_count = desc.get_class_count();
    if (class_count > 2) {
        throw unimplemented(de::error_messages::svm_multiclass_not_implemented_for_gpu());
    }

    const auto data = input.get_data();
    const auto responses = input.get_responses();

    if (data.get_row_count() > de::limits<std::int32_t>::max()) {
        throw domain_error(de::error_messages::invalid_range_of_rows());
    }
    if (data.get_column_count() > de::limits<std::int32_t>::max()) {
        throw domain_error(de::error_messages::invalid_range_of_columns());
    }
    if (responses.get_row_count() > de::limits<std::int32_t>::max()) {
        throw domain_error(de::error_messages::invalid_range_of_rows());
    }
    if (responses.get_column_count() > de::limits<std::int32_t>::max()) {
        throw domain_error(de::error_messages::invalid_range_of_columns());
    }

    const std::int32_t row_count = de::integral_cast<std::int32_t>(data.get_row_count());

    const binary_response_t<Float> old_unique_responses = get_unique_responses<Float>(q, responses);
    const auto new_responses =
        convert_binary_responses(q, responses, { Float(-1.0), Float(1.0) }, old_unique_responses);

    ONEDAL_ASSERT(data.get_row_count() == new_responses.get_row_count());

    const auto data_nd = pr::table2ndarray<Float>(q, data, sycl::usm::alloc::device);
    const auto responses_nd =
        pr::table2ndarray_1d<Float>(q, new_responses, sycl::usm::alloc::device);

    const double C(desc.get_c());
    const double accuracy_threshold(desc.get_accuracy_threshold());
    const double tau(desc.get_tau());
    const double cache_size(desc.get_cache_size());
    const auto kernel_ptr = detail::get_kernel_ptr(desc);
    if (!kernel_ptr) {
        throw internal_error{ de::error_messages::unknown_kernel_function_type() };
    }
    const std::int32_t max_iteration_count(
        de::integral_cast<std::int32_t>(desc.get_max_iteration_count()));

    auto [alpha_nd, alpha_zeros_event] =
        pr::ndarray<Float, 1>::zeros(q, row_count, sycl::usm::alloc::device);

    auto grad_nd = pr::ndarray<Float, 1>::empty(q, { row_count }, sycl::usm::alloc::device);
    auto invert_responses_event = invert_values<Float>(q, responses_nd, grad_nd);

    const std::int32_t ws_count = propose_working_set_size(q, row_count);
    auto ws_indices_nd =
        pr::ndarray<std::int32_t, 1>::empty(q, { ws_count }, sycl::usm::alloc::device);
    auto working_set = working_set_selector<Float>(q, responses_nd, C, row_count);

    // The maximum numbers of iteration of the subtask is number of observation in WS x inner_iterations.
    // It's enough to find minimum for subtask.
    constexpr std::int32_t inner_iterations = 1000;
    const std::int32_t max_inner_iterations_count(ws_count * inner_iterations);

    auto delta_alpha_nd = pr::ndarray<Float, 1>::empty(q, { ws_count }, sycl::usm::alloc::device);
    auto f_diff_nd = pr::ndarray<Float, 1>::empty(q, { 1 }, sycl::usm::alloc::device);
    auto inner_iter_count_nd =
        pr::ndarray<std::int32_t, 1>::empty(q, { 1 }, sycl::usm::alloc::device);

    Float diff = Float(0);
    Float prev_diff = Float(0);

    std::int32_t same_local_diff_count = 0;

    std::shared_ptr<svm_cache_iface<Float>> svm_cache_ptr =
        std::make_shared<svm_cache<no_cache, Float>>(q, data_nd, cache_size, ws_count, row_count);

    sycl::event copy_ws_indices_event;
    sycl::event copy_cache_event;
    std::int32_t ws_indices_copy_count = 0;

    std::int32_t iter = 0;
    for (; iter < max_iteration_count; iter++) {
        if (iter != 0) {
            std::tie(ws_indices_copy_count, copy_ws_indices_event) =
                copy_last_to_first(q, ws_indices_nd);
        }

        working_set
            .select(alpha_nd,
                    grad_nd,
                    ws_indices_nd,
                    ws_indices_copy_count,
                    { alpha_zeros_event, invert_responses_event, copy_ws_indices_event })
            .wait_and_throw();

        const auto kernel_values_nd =
            svm_cache_ptr->compute(kernel_ptr, data, data_nd, ws_indices_nd);

        auto solve_smo_event = solve_smo<Float>(q,
                                                kernel_values_nd,
                                                ws_indices_nd,
                                                responses_nd,
                                                grad_nd,
                                                row_count,
                                                ws_count,
                                                max_inner_iterations_count,
                                                C,
                                                accuracy_threshold,
                                                tau,
                                                alpha_nd,
                                                delta_alpha_nd,
                                                f_diff_nd,
                                                inner_iter_count_nd);
        auto f_diff_host = f_diff_nd.to_host(q, { solve_smo_event }).flatten();
        diff = *f_diff_host.get_data();

        update_grad(q, kernel_values_nd, delta_alpha_nd, grad_nd).wait_and_throw();
        if (check_stop_condition<Float>(diff,
                                        prev_diff,
                                        accuracy_threshold,
                                        same_local_diff_count)) {
            break;
        }

        prev_diff = diff;
    }

    auto [bias, sv_count, sv_coeffs, support_indices, support_vectors] =
        compute_train_results<Float>(q, data_nd, responses_nd, grad_nd, alpha_nd, C);

    if (sv_count == 0) {
        return result_t{};
    }

    auto [biases_nd, full_event] =
        pr::ndarray<Float, 1>::full(q, { 1 }, static_cast<Float>(bias), sycl::usm::alloc::device);
    full_event.wait_and_throw();

    auto [iter_counts_nd, iter_counts_event] =
        pr::ndarray<int, 1>::full(q, { 1 }, static_cast<int>(iter), sycl::usm::alloc::device);
    iter_counts_event.wait_and_throw();
    auto model =
        model_t()
            .set_support_vectors(homogen_table::wrap(support_vectors.flatten(q),
                                                     support_vectors.get_dimension(0),
                                                     support_vectors.get_dimension(1)))
            .set_coeffs(homogen_table::wrap(sv_coeffs.flatten(q), sv_coeffs.get_dimension(0), 1))
            .set_iteration_counts(homogen_table::wrap(iter_counts_nd.flatten(q), 1, 1))
            .set_biases(homogen_table::wrap(biases_nd.flatten(q), 1, 1))
            .set_first_class_response(old_unique_responses.first)
            .set_second_class_response(old_unique_responses.second);

    de::get_impl(model).bias = bias;

    return result_t().set_model(model).set_support_indices(
        homogen_table::wrap(support_indices.flatten(q), support_indices.get_dimension(0), 1));
}

template <typename Float>
struct train_kernel_gpu<Float, method::thunder, task::classification> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template <typename Float>
struct train_kernel_gpu<Float, method::thunder, task::nu_classification> {
    train_result<task::nu_classification> operator()(
        const dal::backend::context_gpu& ctx,
        const detail::descriptor_base<task::nu_classification>& params,
        const train_input<task::nu_classification>& input) const {
        throw unimplemented(de::error_messages::nu_svm_thunder_method_is_not_implemented_for_gpu());
    }
};

template struct train_kernel_gpu<float, method::thunder, task::classification>;
template struct train_kernel_gpu<double, method::thunder, task::classification>;
template struct train_kernel_gpu<float, method::thunder, task::nu_classification>;
template struct train_kernel_gpu<double, method::thunder, task::nu_classification>;

} // namespace oneapi::dal::svm::backend
