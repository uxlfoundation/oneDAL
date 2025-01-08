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

#include "oneapi/dal/algo/kmeans/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_integral.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_csr_impl.hpp"
#include "oneapi/dal/algo/kmeans/detail/train_init_centroids.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/empty_cluster_handling.hpp"

#include "oneapi/dal/detail/profiler.hpp"

#include <tuple>

namespace oneapi::dal::kmeans::backend {

using dal::backend::context_gpu;
using descriptor_t = detail::descriptor_base<task::clustering>;
using event_vector = std::vector<sycl::event>;

namespace interop = dal::backend::interop;
namespace pr = dal::backend::primitives;
namespace de = dal::detail;
namespace bk = dal::backend;

// Initializes centroids randomly on CPU if it was not set by user.
template <typename Float, typename Method>
static pr::ndarray<Float, 2> get_initial_centroids(const dal::backend::context_gpu& ctx,
                                                   const descriptor_t& params,
                                                   const train_input<task::clustering>& input) {
    auto& queue = ctx.get_queue();

    const auto data = static_cast<const csr_table&>(input.get_data());

    const std::int64_t column_count = data.get_column_count();
    const std::int64_t cluster_count = params.get_cluster_count();

    if (!input.get_initial_centroids().has_data()) {
        auto daal_initial_centroids =
            oneapi::dal::kmeans::detail::daal_generate_centroids<Float, Method>(params, data);
        daal::data_management::BlockDescriptor<Float> block;
        daal_initial_centroids->getBlockOfRows(0,
                                               cluster_count,
                                               daal::data_management::readOnly,
                                               block);
        Float* initial_centroids_ptr = block.getBlockPtr();
        auto arr_host_initial =
            pr::ndarray<Float, 2>::wrap(initial_centroids_ptr, { cluster_count, column_count });
        return arr_host_initial.to_device(queue);
    }
    auto initial_centroids_ptr = row_accessor<const Float>(input.get_initial_centroids())
                                     .pull(queue, { 0, -1 }, sycl::usm::alloc::device);
    return pr::ndarray<Float, 2>::wrap(initial_centroids_ptr, { cluster_count, column_count });
}

/// Main entrypoint for GPU CSR Kmeans algorithm
/// @param[in] ctx          GPU context structure
/// @param[in] params       A descriptor containing parameters for algorithm
/// @param[in] input        A train input
template <typename Float>
struct train_kernel_gpu<Float, method::lloyd_csr, task::clustering> {
    train_result<task::clustering> operator()(const dal::backend::context_gpu& ctx,
                                              const descriptor_t& params,
                                              const train_input<task::clustering>& input) const {
        auto& queue = ctx.get_queue();
        ONEDAL_ASSERT(input.get_data().get_kind() == dal::csr_table::kind());
        const csr_table& data = static_cast<const csr_table&>(input.get_data());
        const std::int64_t row_count = data.get_row_count();
        const std::int64_t column_count = data.get_column_count();
        const std::int64_t cluster_count = params.get_cluster_count();
        const std::int64_t max_iteration_count = params.get_max_iteration_count();
        const double accuracy_threshold = params.get_accuracy_threshold();
        dal::detail::check_mul_overflow(cluster_count, column_count);

        auto [arr_val, arr_col, arr_row] =
            csr_accessor<const Float>(data).pull(queue,
                                                 { 0, -1 },
                                                 sparse_indexing::zero_based,
                                                 sycl::usm::alloc::device);
        auto values = pr::ndarray<Float, 1>::wrap(arr_val.get_data(), arr_val.get_count());
        auto column_indices =
            pr::ndarray<std::int64_t, 1>::wrap(arr_col.get_data(), arr_col.get_count());
        auto row_offsets =
            pr::ndarray<std::int64_t, 1>::wrap(arr_row.get_data(), arr_row.get_count());

        pr::sparse_matrix_handle data_handle(queue);
        auto set_csr_data_event = pr::set_csr_data(queue,
                                                   data_handle,
                                                   row_count,
                                                   column_count,
                                                   sparse_indexing::zero_based,
                                                   arr_val.get_data(),
                                                   arr_col.get_data(),
                                                   arr_row.get_data());

        auto arr_initial = get_initial_centroids<Float, method::lloyd_csr>(ctx, params, input);
        auto arr_centroid_squares =
            pr::ndarray<Float, 1>::empty(queue, cluster_count, sycl::usm::alloc::device);
        auto arr_data_squares =
            pr::ndarray<Float, 1>::empty(queue, row_count, sycl::usm::alloc::device);
        auto data_squares_event = compute_data_squares(queue,
                                                       values,
                                                       column_indices,
                                                       row_offsets,
                                                       arr_data_squares,
                                                       { set_csr_data_event });

        auto distances = pr::ndarray<Float, 2>::empty(queue,
                                                      { row_count, cluster_count },
                                                      sycl::usm::alloc::device);

        auto arr_closest_distances =
            pr::ndarray<Float, 2>::empty(queue, { row_count, 1 }, sycl::usm::alloc::device);
        auto arr_centroids = pr::ndarray<Float, 2>::empty(queue,
                                                          { cluster_count, column_count },
                                                          sycl::usm::alloc::device);
        auto arr_responses =
            pr::ndarray<std::int32_t, 2>::empty(queue, { row_count, 1 }, sycl::usm::alloc::device);
        auto cluster_counts =
            pr::ndarray<std::int32_t, 1>::empty(queue, cluster_count, sycl::usm::alloc::device);

        Float prev_objective_function = de::limits<Float>::max();
        std::int64_t iter;
        sycl::event last_event = data_squares_event;

        for (iter = 0; iter < max_iteration_count; iter++) {
            auto centroid_squares_event =
                kernels_fp<Float>::compute_squares(queue,
                                                   iter == 0 ? arr_initial : arr_centroids,
                                                   arr_centroid_squares,
                                                   { last_event });

            auto assign_event = assign_clusters(queue,
                                                row_count,
                                                data_handle,
                                                arr_data_squares,
                                                iter == 0 ? arr_initial : arr_centroids,
                                                arr_centroid_squares,
                                                distances,
                                                arr_responses,
                                                arr_closest_distances,
                                                { centroid_squares_event });

            auto count_event = count_clusters(queue,
                                              arr_responses,
                                              cluster_count,
                                              cluster_counts,
                                              { assign_event });

            auto objective_function =
                calc_objective_function(queue, arr_closest_distances, { count_event });

            auto update_event = update_centroids(queue,
                                                 values,
                                                 column_indices,
                                                 row_offsets,
                                                 column_count,
                                                 arr_responses,
                                                 arr_centroids,
                                                 cluster_counts);

            const std::int64_t empty_cluster_count =
                count_empty_clusters(queue, cluster_count, cluster_counts, { count_event });

            Float correction(0);
            sycl::event empty_cluster_event;
            if (empty_cluster_count > 0) {
                std::tie(correction, empty_cluster_event) =
                    handle_empty_clusters(queue,
                                          values,
                                          column_indices,
                                          row_offsets,
                                          row_count,
                                          arr_centroids,
                                          empty_cluster_count,
                                          cluster_counts,
                                          arr_closest_distances,
                                          { update_event });
            }

            objective_function += correction;

            last_event = empty_cluster_event;

            if (accuracy_threshold > 0 &&
                objective_function + accuracy_threshold > prev_objective_function) {
                iter++;
                break;
            }
            prev_objective_function = objective_function;
        }

        auto centroid_squares_event =
            kernels_fp<Float>::compute_squares(queue,
                                               iter == 0 ? arr_initial : arr_centroids,
                                               arr_centroid_squares,
                                               { last_event });
        auto assign_event = assign_clusters(queue,
                                            row_count,
                                            data_handle,
                                            arr_data_squares,
                                            iter == 0 ? arr_initial : arr_centroids,
                                            arr_centroid_squares,
                                            distances,
                                            arr_responses,
                                            arr_closest_distances,
                                            { last_event, centroid_squares_event });
        auto objective_function =
            calc_objective_function(queue,
                                    arr_closest_distances,
                                    { last_event, centroid_squares_event, assign_event });

        model<task::clustering> model;
        model.set_centroids(
            dal::homogen_table::wrap(arr_centroids.flatten(queue), cluster_count, column_count));
        return train_result<task::clustering>()
            .set_responses(dal::homogen_table::wrap(arr_responses.flatten(queue), row_count, 1))
            .set_iteration_count(iter)
            .set_objective_function_value(objective_function)
            .set_model(model);
    }
};

template struct train_kernel_gpu<float, method::lloyd_csr, task::clustering>;
template struct train_kernel_gpu<double, method::lloyd_csr, task::clustering>;

} // namespace oneapi::dal::kmeans::backend
