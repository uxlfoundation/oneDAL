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

#include "oneapi/dal/algo/svm/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/gpu/svm_predict.hpp"

#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::svm::backend {

using dal::backend::context_gpu;
using model_t = model<task::classification>;
using input_t = infer_input<task::classification>;
using result_t = infer_result<task::classification>;
using descriptor_t = detail::descriptor_base<task::classification>;

namespace pr = dal::backend::primitives;
namespace bk = dal::backend;

template <typename Float>
auto make_responses(sycl::queue& q,
                    const pr::ndarray<Float, 1>& distances,
                    const std::int64_t first_class_response,
                    const std::int64_t second_class_response,
                    const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(distances.has_data());
    ONEDAL_ASSERT(distances.get_dimension(0) > 0);

    const auto size = distances.get_count();
    auto response = pr::ndarray<Float, 1>::empty(q, size, sycl::usm::alloc::device);

    auto response_data = response.get_mutable_data();
    const auto distance_data = distances.get_data();

    auto res_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(bk::make_range_1d(size), [=](sycl::id<1> idx) {
            response_data[idx] =
                distance_data[idx] >= 0 ? second_class_response : first_class_response;
        });
    });
    return std::make_tuple(response, res_event);
}

template <typename Float>
static result_t infer(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto& q = ctx.get_queue();

    const std::uint64_t class_count = desc.get_class_count();
    if (class_count > 2) {
        throw unimplemented(dal::detail::error_messages::svm_multiclass_not_implemented_for_gpu());
    }

    const auto data = input.get_data();
    const auto data_nd = pr::table2ndarray<Float>(q, data, sycl::usm::alloc::device);
    const auto trained_model = input.get_model();

    // A binary GPU descriptor combined with a model whose public arrays carry
    // multi-class structure (class_count > 2 or n_support_per_class populated
    // with more than two entries) would silently run the binary code path on
    // aggregated multi-class arrays. Refuse instead of producing meaningless
    // decision values.
    if (trained_model.get_class_count() > 2 ||
        (trained_model.get_n_support_per_class().has_data() &&
         trained_model.get_n_support_per_class().get_column_count() > 2)) {
        throw unimplemented(dal::detail::error_messages::svm_multiclass_not_implemented_for_gpu());
    }

    const auto kernel_ptr = detail::get_kernel_ptr(desc);
    if (!kernel_ptr) {
        throw internal_error{ dal::detail::error_messages::unknown_kernel_function_type() };
    }

    const std::int64_t row_count = data.get_row_count();

    auto distance_nd = pr::ndarray<Float, 1>::empty(q, { row_count }, sycl::usm::alloc::device);

    auto sv_coeffs = trained_model.get_coeffs();
    const std::int64_t sv_count = trained_model.get_support_vector_count();

    if (sv_count == 0) {
        distance_nd.fill(q, Float(0)).wait_and_throw();
    }
    else {
        const auto sv_coeff_nd =
            pr::table2ndarray_1d<Float>(q, sv_coeffs, sycl::usm::alloc::device);

        const auto biases = pr::table2ndarray_1d<Float>(trained_model.get_biases());
        const auto bias = *(biases.get_data());
        auto fill_event = distance_nd.fill(q, bias);

        auto support_vectors_nd = pr::table2ndarray<Float>(q,
                                                           trained_model.get_support_vectors(),
                                                           sycl::usm::alloc::device);
        auto support_vectors = homogen_table::wrap(q,
                                                   support_vectors_nd.get_data(),
                                                   support_vectors_nd.get_dimension(0),
                                                   support_vectors_nd.get_dimension(1));

        const std::int64_t max_rows_per_block = 1024;
        const std::int64_t blocks_count =
            row_count / max_rows_per_block + !!(row_count % max_rows_per_block);

        std::shared_ptr<predict_task<Float>> predict_task =
            std::make_shared<predict_task_dense<Float>>(q,
                                                        max_rows_per_block,
                                                        data_nd,
                                                        support_vectors,
                                                        kernel_ptr);

        for (std::int64_t block_i = 0; block_i < blocks_count; ++block_i) {
            const std::int64_t start_row = block_i * max_rows_per_block;
            const std::int64_t rows_per_block_count =
                (block_i != blocks_count - 1) ? max_rows_per_block
                                              : row_count - block_i * max_rows_per_block;

            auto distance_block_nd =
                pr::ndarray<Float, 2>::wrap(distance_nd.get_mutable_data() + start_row,
                                            { rows_per_block_count, 1 });
            auto kernel_values_nd =
                predict_task->kernel_compute(start_row, rows_per_block_count, sv_count);
            auto reshape_sv_coeff = sv_coeff_nd.reshape(pr::ndshape<2>{ sv_count, 1 });
            {
                ONEDAL_PROFILER_TASK(gemm, q);
                pr::gemm(q,
                         kernel_values_nd,
                         reshape_sv_coeff,
                         distance_block_nd,
                         Float(1),
                         Float(1),
                         { fill_event })
                    .wait_and_throw();
            }
        }
    }

    auto [response_nd, responses_event] = make_responses(q,
                                                         distance_nd,
                                                         trained_model.get_first_class_response(),
                                                         trained_model.get_second_class_response(),
                                                         {});
    responses_event.wait_and_throw();

    return result_t()
        .set_decision_function(homogen_table::wrap(distance_nd.flatten(q), row_count, 1))
        .set_responses(homogen_table::wrap(response_nd.flatten(q), row_count, 1));
}

template <typename Float>
struct infer_kernel_gpu<Float, method::by_default, task::classification> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template <typename Float>
struct infer_kernel_gpu<Float, method::by_default, task::nu_classification> {
    infer_result<task::nu_classification> operator()(
        const context_gpu& ctx,
        const detail::descriptor_base<task::nu_classification>& desc,
        const infer_input<task::nu_classification>& input) const {
        throw unimplemented(
            dal::detail::error_messages::svm_nu_classification_task_is_not_implemented_for_gpu());
    }
};

template struct infer_kernel_gpu<float, method::by_default, task::classification>;
template struct infer_kernel_gpu<double, method::by_default, task::classification>;
template struct infer_kernel_gpu<float, method::by_default, task::nu_classification>;
template struct infer_kernel_gpu<double, method::by_default, task::nu_classification>;

} // namespace oneapi::dal::svm::backend
