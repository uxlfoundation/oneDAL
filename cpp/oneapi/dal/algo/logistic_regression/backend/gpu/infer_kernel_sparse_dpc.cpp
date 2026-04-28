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

#include "oneapi/dal/algo/logistic_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/logistic_regression/common.hpp"
#include "oneapi/dal/algo/logistic_regression/train_types.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/ndindexer.hpp"
#include "oneapi/dal/backend/primitives/objective_function.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas.hpp"

#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"

namespace oneapi::dal::logistic_regression::backend {

using dal::backend::context_gpu;

namespace be = dal::backend;
namespace pr = be::primitives;
namespace interop = dal::backend::interop;

template <typename Float, typename Task>
static infer_result<Task> call_dal_kernel(const context_gpu& ctx,
                                          const detail::descriptor_base<Task>& desc,
                                          const table& infer,
                                          const model<Task>& m) {
    auto queue = ctx.get_queue();
    ONEDAL_PROFILER_TASK(logreg_infer_kernel, queue);

    constexpr auto alloc = sycl::usm::alloc::device;

    const auto& betas = m.get_packed_coefficients();

    const auto sample_count = infer.get_row_count();
    const auto feature_count = infer.get_column_count();
    const bool fit_intercept = desc.get_compute_intercept();
    ONEDAL_ASSERT((feature_count + 1) == betas.get_column_count());
    ONEDAL_ASSERT(1 == betas.get_row_count());
    ONEDAL_ASSERT(infer.get_kind() == dal::csr_table::kind());

    pr::ndarray<Float, 1> params = pr::table2ndarray_1d<Float>(queue, betas, alloc);
    pr::ndview<Float, 1> params_suf = fit_intercept ? params : params.slice(1, feature_count);

    pr::ndarray<Float, 1> probs = pr::ndarray<Float, 1>::empty(queue, { sample_count }, alloc);
    pr::ndarray<std::int32_t, 1> responses =
        pr::ndarray<std::int32_t, 1>::empty(queue, { sample_count }, alloc);

    auto [csr_data, column_indices, row_offsets] =
        csr_accessor<const Float>(static_cast<const csr_table&>(infer))
            .pull(queue, { 0, -1 }, sparse_indexing::zero_based);

    auto csr_data_gpu =
        pr::ndarray<Float, 1>::wrap(csr_data.get_data(), csr_data.get_count()).to_device(queue);
    auto column_indices_gpu =
        pr::ndarray<std::int64_t, 1>::wrap(column_indices.get_data(), column_indices.get_count())
            .to_device(queue);
    auto row_offsets_gpu =
        pr::ndarray<std::int64_t, 1>::wrap(row_offsets.get_data(), row_offsets.get_count())
            .to_device(queue);

    table data_gpu = csr_table::wrap(queue,
                                     csr_data_gpu.get_data(),
                                     column_indices_gpu.get_data(),
                                     row_offsets_gpu.get_data(),
                                     sample_count,
                                     feature_count,
                                     sparse_indexing::zero_based);

    dal::backend::primitives::sparse_matrix_handle sp_handle(queue);
    auto set_csr_data_event =
        set_csr_data(queue, sp_handle, static_cast<const csr_table&>(data_gpu));

    sycl::event probabilities_event = compute_probabilities_sparse(queue,
                                                                   params_suf,
                                                                   sp_handle,
                                                                   probs,
                                                                   fit_intercept,
                                                                   { set_csr_data_event });

    const auto* const prob_ptr = probs.get_data();
    auto* const resp_ptr = responses.get_mutable_data();

    auto fill_resp_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(probabilities_event);
        const auto range = be::make_range_1d(sample_count);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            constexpr Float half = 0.5f;
            resp_ptr[idx] = prob_ptr[idx] < half ? 0 : 1;
        });
    });

    auto resp_table =
        homogen_table::wrap(responses.flatten(queue, { fill_resp_event }), sample_count, 1);
    auto prob_table =
        homogen_table::wrap(probs.flatten(queue, { probabilities_event }), sample_count, 1);

    auto result = infer_result<Task>().set_responses(resp_table).set_probabilities(prob_table);

    return result;
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_gpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const infer_input<Task>& input) {
    return call_dal_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float, typename Task>
struct infer_kernel_gpu<Float, method::sparse, Task> {
    infer_result<Task> operator()(const context_gpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const infer_input<Task>& input) const {
        return infer<Float, Task>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, method::sparse, task::classification>;
template struct infer_kernel_gpu<double, method::sparse, task::classification>;

} // namespace oneapi::dal::logistic_regression::backend
