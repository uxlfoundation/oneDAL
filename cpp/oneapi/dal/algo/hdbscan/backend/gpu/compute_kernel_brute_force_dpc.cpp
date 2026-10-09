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

/// Run the brute-force HDBSCAN GPU pipeline: distance matrix, core distances, MRD matrix, Boruvka
/// MST, sort and cluster extraction.
///
/// @tparam Float Floating-point type
///
/// @param[in] ctx        GPU dispatch context (carries the SYCL queue)
/// @param[in] desc       Algorithm descriptor
/// @param[in] local_data Input data table of size `n x d`
///
/// @return oneAPI `compute_result` with responses and cluster count
template <typename Float>
static result_t compute_kernel_dense_impl(const context_gpu& ctx,
                                          const descriptor_t& desc,
                                          const table& local_data) {
    ONEDAL_PROFILER_TASK(hdbscan.compute, ctx.get_queue());

    auto& queue = ctx.get_queue();

    const std::int64_t row_count = local_data.get_row_count();
    const std::int64_t min_cluster_size = desc.get_min_cluster_size();
    const std::int64_t min_samples = desc.get_min_samples();
    const std::int64_t edge_count = row_count - 1;
    const auto metric = desc.get_metric();
    const double degree = desc.get_degree();
    const std::int32_t cluster_selection =
        (desc.get_cluster_selection() == cluster_selection_method::leaf) ? 1 : 0;
    const bool allow_single_cluster = desc.get_allow_single_cluster();
    const Float cluster_selection_epsilon =
        static_cast<Float>(desc.get_cluster_selection_epsilon());
    const std::int64_t max_cluster_size = desc.get_max_cluster_size();
    const double alpha = desc.get_alpha();

    check_mrd_matrix_fits_on_device<Float>(queue,
                                           row_count,
                                           local_data.get_column_count(),
                                           min_samples);

    const auto data_nd = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);
    queue.wait_and_throw();

    // Step 1: pairwise distance matrix. Not zero-filled: every entry is written, and a fill of
    // more than 2^31 elements is rejected.
    auto dist_matrix =
        pr::ndarray<Float, 2>::empty(queue, { row_count, row_count }, sycl::usm::alloc::device);

    auto dist_event =
        compute_distance_matrix<Float>(queue, data_nd, dist_matrix, metric, degree, {});
    dist_event.wait_and_throw();

    // Step 2: core distances, unscaled; step 3 applies alpha.
    auto [core_distances, core_dist_event] =
        pr::ndarray<Float, 1>::zeros(queue, row_count, sycl::usm::alloc::device);
    core_dist_event.wait_and_throw();

    auto core_event = compute_core_distances<Float>(queue,
                                                    dist_matrix,
                                                    core_distances,
                                                    min_samples,
                                                    row_count,
                                                    metric,
                                                    { dist_event, core_dist_event });
    core_event.wait_and_throw();

    // Step 3: MRD matrix in place.
    auto& mrd_matrix = dist_matrix;

    auto mrd_compute_event =
        compute_mrd_matrix<Float>(queue, core_distances, mrd_matrix, metric, alpha, { core_event });
    mrd_compute_event.wait_and_throw();

    // Step 4: Boruvka MST.
    auto [mst_from, mst_from_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [mst_to, mst_to_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [mst_weights, mst_weights_event] =
        pr::ndarray<Float, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    mst_from_event.wait_and_throw();
    mst_to_event.wait_and_throw();
    mst_weights_event.wait_and_throw();

    auto mst_event =
        build_mst<Float>(queue,
                         mrd_matrix,
                         mst_from,
                         mst_to,
                         mst_weights,
                         row_count,
                         { mrd_compute_event, mst_from_event, mst_to_event, mst_weights_event });
    mst_event.wait_and_throw();

    // Free the `n x n` matrix before the remaining steps.
    dist_matrix = pr::ndarray<Float, 2>{};
    queue.wait_and_throw();

    // Step 5: sort the MST edges by weight.
    auto sort_event =
        sort_mst_by_weight<Float>(queue, mst_from, mst_to, mst_weights, edge_count, { mst_event });
    sort_event.wait_and_throw();

    // Step 6: extract the flat clusters.
    auto [arr_responses, responses_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, -1, sycl::usm::alloc::device);
    responses_event.wait_and_throw();

    // Empty unless requested, which skips the probability kernels; the centers need them too.
    const bool need_probabilities = desc.get_result_options().test(result_options::probabilities) ||
                                    (desc.get_store_centers() != store_centers_method::none &&
                                     desc.get_result_options().test(result_options::responses));
    pr::ndarray<Float, 1> arr_probabilities;
    if (need_probabilities) {
        arr_probabilities =
            std::get<0>(pr::ndarray<Float, 1>::zeros(queue, row_count, sycl::usm::alloc::device));
        queue.wait_and_throw();
    }

    // Empty unless requested; there is no merge below two rows.
    const bool need_single_linkage_tree =
        desc.get_result_options().test(result_options::single_linkage_tree) && edge_count > 0;
    pr::ndarray<Float, 1> arr_single_linkage_tree;
    if (need_single_linkage_tree) {
        arr_single_linkage_tree = std::get<0>(
            pr::ndarray<Float, 1>::zeros(queue, 4 * edge_count, sycl::usm::alloc::device));
        queue.wait_and_throw();
    }

    auto cluster_event = extract_clusters<Float>(
        queue,
        mst_from,
        mst_to,
        mst_weights,
        arr_responses,
        row_count,
        min_cluster_size,
        { sort_event, responses_event },
        cluster_selection,
        allow_single_cluster,
        cluster_selection_epsilon,
        max_cluster_size,
        need_probabilities ? arr_probabilities.get_mutable_data() : nullptr,
        need_single_linkage_tree ? arr_single_linkage_tree.get_mutable_data() : nullptr);
    cluster_event.wait_and_throw();

    // The cluster count is the largest label plus one.
    auto [max_label_arr, ml_ev] =
        pr::ndarray<std::int32_t, 1>::full(queue, 1, -1, sycl::usm::alloc::device);
    sycl::event ml_alloc_ev = ml_ev;
    const std::int32_t* r_ptr = arr_responses.get_data();
    std::int32_t* ml_ptr = max_label_arr.get_mutable_data();
    auto reduce_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ ml_alloc_ev });
        h.single_task([=]() {
            std::int32_t mx = -1;
            for (std::int64_t i = 0; i < row_count; i++) {
                if (r_ptr[i] > mx)
                    mx = r_ptr[i];
            }
            ml_ptr[0] = mx;
        });
    });
    reduce_event.wait_and_throw();
    auto max_label_host = max_label_arr.to_host(queue, { reduce_event });
    const std::int32_t max_label = max_label_host.get_data()[0];
    const std::int64_t cluster_count = (max_label >= 0) ? (max_label + 1) : 0;

    return make_results<Float>(queue,
                               desc,
                               arr_responses,
                               cluster_count,
                               local_data,
                               arr_probabilities,
                               arr_single_linkage_tree);
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return compute_kernel_dense_impl<Float>(ctx, desc, input.get_data());
}

template <typename Float>
struct compute_kernel_gpu<Float, method::brute_force, task::clustering> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::brute_force, task::clustering>;
template struct compute_kernel_gpu<double, method::brute_force, task::clustering>;

} // namespace oneapi::dal::hdbscan::backend
