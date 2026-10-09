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

/// GPU HDBSCAN kd_tree and ball_tree: k-NN core distances + GPU Boruvka MST. Neither builds a
/// tree on the GPU, so both methods share this pipeline and all computation stays on device.

#include "oneapi/dal/algo/hdbscan/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/hdbscan/backend/gpu/kernel_impl.hpp"
#include "oneapi/dal/algo/hdbscan/backend/gpu/results.hpp"

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

namespace oneapi::dal::hdbscan::backend {

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using dal::backend::context_gpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

/// Pick the query block size of the core-distance search: `distance_block_size` if set, else
/// about 256 MB of `B x N` distances, within `[256, row_count]`.
///
/// @param[in] row_count  Number of points `N`
/// @param[in] float_size `sizeof(Float)`
/// @param[in] user_hint  Descriptor-provided `distance_block_size` (0 = auto)
///
/// @return Chosen block size `B`
static std::int64_t choose_block_size(std::int64_t row_count,
                                      std::int64_t float_size,
                                      std::int64_t user_hint) {
    if (user_hint > 0) {
        return (user_hint > row_count) ? row_count : user_hint;
    }
    const std::int64_t target_bytes = 256 * 1024 * 1024;
    std::int64_t bs = target_bytes / (row_count * float_size);
    if (bs < 256)
        bs = 256;
    if (bs > row_count)
        bs = row_count;
    return bs;
}

/// Run the kd_tree / ball_tree HDBSCAN GPU pipeline without the `n x n` MRD matrix: k-NN core
/// distances, Boruvka MST with on-the-fly distances, sort and cluster extraction.
///
/// @tparam Float Floating-point type
///
/// @param[in] ctx        GPU dispatch context
/// @param[in] desc       Algorithm descriptor
/// @param[in] local_data Input data table of size `n x d`
///
/// @return oneAPI `compute_result` with responses and cluster count
template <typename Float>
static result_t compute_kernel_tree_impl(const context_gpu& ctx,
                                         const descriptor_t& desc,
                                         const table& local_data) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_tree, ctx.get_queue());

    auto& queue = ctx.get_queue();

    const std::int64_t row_count = local_data.get_row_count();
    const std::int64_t col_count = local_data.get_column_count();
    const std::int64_t min_samples = desc.get_min_samples();
    const std::int64_t edge_count = row_count - 1;
    const auto metric = desc.get_metric();
    const double degree = desc.get_degree();
    const double alpha = desc.get_alpha();

    const auto data_nd = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);
    queue.wait_and_throw();

    // Step 1: core distances.
    const std::int64_t block_size =
        choose_block_size(row_count, sizeof(Float), desc.get_distance_block_size());

    auto core_distances = pr::ndarray<Float, 1>::empty(queue, row_count, sycl::usm::alloc::device);
    const auto prev_block_event = compute_core_distances_blocked<Float>(queue,
                                                                        data_nd,
                                                                        core_distances,
                                                                        min_samples,
                                                                        block_size,
                                                                        metric,
                                                                        degree);

    // Step 2: Boruvka MST.
    auto [mst_from, mst_from_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [mst_to, mst_to_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [mst_weights, mst_weights_event] =
        pr::ndarray<Float, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    mst_from_event.wait_and_throw();
    mst_to_event.wait_and_throw();
    mst_weights_event.wait_and_throw();

    // Alpha scales only `dist(i, j)` inside MRD; core distances stay unscaled.
    auto mst_event =
        build_mst_otf<Float>(queue,
                             data_nd,
                             core_distances,
                             mst_from,
                             mst_to,
                             mst_weights,
                             row_count,
                             col_count,
                             metric,
                             degree,
                             alpha,
                             { prev_block_event, mst_from_event, mst_to_event, mst_weights_event });
    mst_event.wait_and_throw();

    // Step 3: sort the MST edges by weight.
    auto sort_event =
        sort_mst_by_weight<Float>(queue, mst_from, mst_to, mst_weights, edge_count, { mst_event });
    sort_event.wait_and_throw();

    // Step 4: extract the flat clusters.
    return make_results_from_sorted_mst<Float>(queue,
                                               desc,
                                               local_data,
                                               mst_from,
                                               mst_to,
                                               mst_weights,
                                               { sort_event });
}

template <typename Float>
struct compute_kernel_gpu<Float, method::kd_tree, task::clustering> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute_kernel_tree_impl<Float>(ctx, desc, input.get_data());
    }
};

template <typename Float>
struct compute_kernel_gpu<Float, method::ball_tree, task::clustering> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute_kernel_tree_impl<Float>(ctx, desc, input.get_data());
    }
};

template struct compute_kernel_gpu<float, method::kd_tree, task::clustering>;
template struct compute_kernel_gpu<double, method::kd_tree, task::clustering>;
template struct compute_kernel_gpu<float, method::ball_tree, task::clustering>;
template struct compute_kernel_gpu<double, method::ball_tree, task::clustering>;

} // namespace oneapi::dal::hdbscan::backend
