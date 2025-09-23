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

#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/knn/backend/model_conversion.hpp"
#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel_impl.hpp"
#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel_impl_dpc.hpp"
#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel_impl_dpc_distr.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/algo/knn/backend/distance_impl.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::knn::backend {

using dal::backend::context_gpu;

inline bool is_col_major(const table& t) {
    const auto t_layout = t.get_data_layout();
    return t_layout == decltype(t_layout)::column_major;
}

template <typename Float, bool is_cm>
struct ndarray_t_map;

template <typename Float>
struct ndarray_t_map<Float, true> {
    using type = pr::ndarray<Float, 2, pr::ndorder::f>;
};

template <typename Float>
struct ndarray_t_map<Float, false> {
    using type = pr::ndarray<Float, 2, pr::ndorder::c>;
};

template <typename Float, bool is_cm>
using ndarray_t = typename ndarray_t_map<Float, is_cm>::type;

template <typename Type, pr::ndorder order>
constexpr pr::ndorder get_ndorder(const pr::ndarray<Type, 2, order>&) {
    return order;
}

template <typename Task>
using descriptor_t = detail::descriptor_base<Task>;
template <typename Task>
using model_t = model<Task>;

template <typename Float, typename Task, bool cm_train, bool cm_query>
static infer_result<Task> kernel(const descriptor_t<Task>& desc,
                                 const table& infer,
                                 const model<Task>& m,
                                 sycl::queue& queue,
                                 bk::communicator<spmd::device_memory_access::usm> comm) {
    using res_t = response_t<Task, Float>;

    const bool distr_mode = comm.get_rank_count() > 1;

    auto distance_impl = detail::get_distance_impl(desc);
    if (!distance_impl) {
        throw internal_error{ de::error_messages::unknown_distance_type() };
    }

    const auto trained_model = dynamic_cast_to_knn_model<Task, brute_force_model_impl<Task>>(m);
    const auto train = trained_model->get_data();
    const auto resps = trained_model->get_responses();

    const std::int64_t infer_row_count = infer.get_row_count();
    const std::int64_t neighbor_count = desc.get_neighbor_count();

    ONEDAL_ASSERT(train.get_column_count() == infer.get_column_count());

    auto arr_responses = array<res_t>{};
    auto wrapped_responses = pr::ndview<res_t, 1>{};
    if (desc.get_result_options().test(result_options::responses)) {
        arr_responses = array<res_t>::empty(queue, infer_row_count, sycl::usm::alloc::device);
        wrapped_responses = pr::ndview<res_t, 1>::wrap_mutable(arr_responses, { infer_row_count });
    }
    auto arr_distances = array<Float>{};
    auto wrapped_distances = pr::ndview<Float, 2>{};
    if (desc.get_result_options().test(result_options::distances) ||
        (desc.get_voting_mode() == voting_t::distance)) {
        const auto length = de::check_mul_overflow(infer_row_count, neighbor_count);
        arr_distances = array<Float>::empty(queue, length, sycl::usm::alloc::device);
        wrapped_distances =
            pr::ndview<Float, 2>::wrap_mutable(arr_distances, { infer_row_count, neighbor_count });
    }
    auto arr_indices = array<idx_t>{};
    auto wrapped_indices = pr::ndview<idx_t, 2>{};
    if (desc.get_result_options().test(result_options::indices)) {
        const auto length = de::check_mul_overflow(infer_row_count, neighbor_count);
        arr_indices = array<idx_t>::empty(queue, length, sycl::usm::alloc::device);
        wrapped_indices =
            pr::ndview<idx_t, 2>::wrap_mutable(arr_indices, { infer_row_count, neighbor_count });
    }

    // only doing this in single gpu (not in distributed)
    using train_t = ndarray_t<Float, cm_train>;
    auto train_var = pr::table2ndarray_variant<Float>(queue, train, sycl::usm::alloc::device);
    const train_t& train_data = std::get<train_t>(train_var);

    using query_t = ndarray_t<Float, cm_query>;
    auto query_var = pr::table2ndarray_variant<Float>(queue, infer, sycl::usm::alloc::device);
    const query_t& query_data = std::get<query_t>(query_var);

    auto responses_data = pr::ndarray<res_t, 1>{};
    if (desc.get_result_options().test(result_options::responses)) {
        responses_data = pr::table2ndarray_1d<res_t>(queue, resps, sycl::usm::alloc::device);
    }

    if (distr_mode) {
        auto part_distances = array<Float>{};
        auto wrapped_part_distances = pr::ndview<Float, 2>{};
        if (desc.get_result_options().test(result_options::distances) ||
            (desc.get_voting_mode() == voting_t::distance)) {
            const auto part_length = de::check_mul_overflow(2 * infer_row_count, neighbor_count);
            part_distances = array<Float>::empty(queue, part_length, sycl::usm::alloc::device);
            wrapped_part_distances =
                pr::ndview<Float, 2>::wrap_mutable(part_distances,
                                                   { infer_row_count, 2 * neighbor_count });
        }
        auto part_indices = array<idx_t>{};
        auto wrapped_part_indices = pr::ndview<idx_t, 2>{};
        if (desc.get_result_options().test(result_options::indices)) {
            const auto part_length = de::check_mul_overflow(2 * infer_row_count, neighbor_count);
            part_indices = array<idx_t>::empty(queue, part_length, sycl::usm::alloc::device);
            wrapped_part_indices =
                pr::ndview<idx_t, 2>::wrap_mutable(part_indices,
                                                   { infer_row_count, 2 * neighbor_count });
        }
        auto part_responses = array<res_t>{};
        auto wrapped_part_responses = pr::ndview<res_t, 2>{};
        auto intermediate_responses = array<res_t>{};
        auto wrapped_intermediate_responses = pr::ndview<res_t, 2>{};
        if (desc.get_result_options().test(result_options::responses)) {
            const auto part_length = de::check_mul_overflow(2 * infer_row_count, neighbor_count);
            part_responses = array<res_t>::empty(queue, part_length, sycl::usm::alloc::device);
            wrapped_part_responses =
                pr::ndview<res_t, 2>::wrap_mutable(part_responses,
                                                   { infer_row_count, 2 * neighbor_count });
            intermediate_responses = array<res_t>::empty(queue,
                                                         infer_row_count * neighbor_count,
                                                         sycl::usm::alloc::device);
            wrapped_intermediate_responses =
                pr::ndview<res_t, 2>::wrap_mutable(intermediate_responses,
                                                   { infer_row_count, neighbor_count });
        }
        bf_kernel_distr(queue,
                        comm,
                        desc,
                        train,
                        query_data,
                        resps,
                        wrapped_distances,
                        wrapped_part_distances,
                        wrapped_indices,
                        wrapped_part_indices,
                        wrapped_responses,
                        wrapped_part_responses,
                        wrapped_intermediate_responses)
            .wait_and_throw();
    }
    else {
        bf_kernel(queue,
                  comm,
                  desc,
                  train_data,
                  query_data,
                  responses_data,
                  wrapped_distances,
                  wrapped_indices,
                  wrapped_responses)
            .wait_and_throw();
    }

    auto result = infer_result<Task>{}.set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        if constexpr (detail::is_not_search_v<Task>) {
            result = result.set_responses(homogen_table::wrap(arr_responses, infer_row_count, 1));
        }
    }

    if (desc.get_result_options().test(result_options::indices)) {
        result =
            result.set_indices(homogen_table::wrap(arr_indices, infer_row_count, neighbor_count));
    }

    if (desc.get_result_options().test(result_options::distances)) {
        result = result.set_distances(
            homogen_table::wrap(arr_distances, infer_row_count, neighbor_count));
    }

    return result;
}

template <typename Float, typename Task>
static infer_result<Task> call_kernel(const context_gpu& ctx,
                                      const descriptor_t<Task>& desc,
                                      const table& infer,
                                      const model<Task>& m) {
    auto& c = ctx.get_communicator();
    auto& q = ctx.get_queue();
    const auto trained_model = dynamic_cast_to_knn_model<Task, brute_force_model_impl<Task>>(m);
    const auto train = trained_model->get_data();
    const bool cm_train = is_col_major(train);
    const bool cm_query = is_col_major(infer);
    if (cm_train) {
        if (cm_query)
            return kernel<Float, Task, true, true>(desc, infer, m, q, c);
        else
            return kernel<Float, Task, true, false>(desc, infer, m, q, c);
    }
    else {
        if (cm_query)
            return kernel<Float, Task, false, true>(desc, infer, m, q, c);
        else
            return kernel<Float, Task, false, false>(desc, infer, m, q, c);
    }
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_gpu& ctx,
                                const descriptor_t<Task>& desc,
                                const infer_input<Task>& input) {
    return call_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float, typename Task>
struct infer_kernel_gpu<Float, method::brute_force, Task> {
    infer_result<Task> operator()(const context_gpu& ctx,
                                  const descriptor_t<Task>& desc,
                                  const infer_input<Task>& input) const {
        return infer<Float, Task>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, method::brute_force, task::classification>;
template struct infer_kernel_gpu<double, method::brute_force, task::classification>;
template struct infer_kernel_gpu<float, method::brute_force, task::regression>;
template struct infer_kernel_gpu<double, method::brute_force, task::regression>;
template struct infer_kernel_gpu<float, method::brute_force, task::search>;
template struct infer_kernel_gpu<double, method::brute_force, task::search>;

} // namespace oneapi::dal::knn::backend
