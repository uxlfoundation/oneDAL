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

#include <iostream>

namespace oneapi::dal::hdbscan::backend {

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using dal::backend::context_gpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

/// Run the brute-force HDBSCAN GPU pipeline for a single floating-point type.
///
/// Pipeline: pairwise distance matrix -> core distances via k-select ->
/// in-place MRD matrix (alpha is applied only to dist inside MRD) -> GPU
/// Boruvka MST (`build_mst`) -> radix sort by weight -> `extract_clusters`.
/// Outputs per-point responses and the cluster count.
///
/// @tparam Float Floating-point type
///
/// @param[in] ctx        GPU dispatch context (carries the SYCL queue)
/// @param[in] desc       Algorithm descriptor
/// @param[in] local_data Input data table of size `n × d`
///
/// @return oneAPI `compute_result` with responses and cluster count
template <typename Float>
static result_t compute_kernel_dense_impl(const context_gpu& ctx,
                                          const descriptor_t& desc,
                                          const table& local_data) {
    ONEDAL_PROFILER_TASK(hdbscan.compute, ctx.get_queue());

    auto& queue = ctx.get_queue();
    std::cout << "[hdbscan-gpu][brute_force] enter compute_kernel_dense_impl" << std::endl;

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
    std::cout << "[hdbscan-gpu][brute_force] descriptor read: row_count=" << row_count
              << " min_cluster_size=" << min_cluster_size << " min_samples=" << min_samples
              << " edge_count=" << edge_count << " metric=" << static_cast<int>(metric)
              << " degree=" << degree << " cluster_selection=" << cluster_selection
              << " allow_single_cluster=" << allow_single_cluster
              << " cluster_selection_epsilon=" << static_cast<double>(cluster_selection_epsilon)
              << " max_cluster_size=" << max_cluster_size << " alpha=" << alpha << std::endl;

    std::cout << "[hdbscan-gpu][brute_force] step0 BEGIN table2ndarray" << std::endl;
    const auto data_nd = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);
    queue.wait_and_throw();
    std::cout << "[hdbscan-gpu][brute_force] step0 END   table2ndarray ok" << std::endl;

    // Step 1: Compute pairwise distance matrix
    std::cout << "[hdbscan-gpu][brute_force] step1 BEGIN alloc dist_matrix (" << row_count << " x "
              << row_count << ")" << std::endl;
    auto [dist_matrix, dist_alloc_event] =
        pr::ndarray<Float, 2>::zeros(queue, { row_count, row_count }, sycl::usm::alloc::device);
    dist_alloc_event.wait_and_throw();
    std::cout << "[hdbscan-gpu][brute_force] step1 END   alloc dist_matrix ok" << std::endl;

    std::cout << "[hdbscan-gpu][brute_force] step2 BEGIN compute_distance_matrix" << std::endl;
    auto dist_event = compute_distance_matrix<Float>(queue,
                                                     data_nd,
                                                     dist_matrix,
                                                     metric,
                                                     degree,
                                                     { dist_alloc_event });
    dist_event.wait_and_throw();
    std::cout << "[hdbscan-gpu][brute_force] step2 END   compute_distance_matrix ok" << std::endl;

    // Step 2: Compute core distances from the unscaled distance matrix.
    // Per the canonical HDBSCAN definition, alpha must NOT touch the k-NN
    // core distance — it scales only the dist term inside MRD (Step 3).
    std::cout << "[hdbscan-gpu][brute_force] step3 BEGIN alloc core_distances" << std::endl;
    auto [core_distances, core_dist_event] =
        pr::ndarray<Float, 1>::zeros(queue, row_count, sycl::usm::alloc::device);
    core_dist_event.wait_and_throw();
    std::cout << "[hdbscan-gpu][brute_force] step3 END   alloc core_distances ok" << std::endl;

    std::cout << "[hdbscan-gpu][brute_force] step4 BEGIN compute_core_distances" << std::endl;
    auto core_event = compute_core_distances<Float>(queue,
                                                    dist_matrix,
                                                    core_distances,
                                                    min_samples,
                                                    row_count,
                                                    metric,
                                                    { dist_event, core_dist_event });
    core_event.wait_and_throw();
    std::cout << "[hdbscan-gpu][brute_force] step4 END   compute_core_distances ok" << std::endl;

    // Step 3: Transform distances into MRD matrix in-place, applying 1/alpha
    // only to the dist(i,j) term inside the max with core_i, core_j.
    auto& mrd_matrix = dist_matrix;

    std::cout << "[hdbscan-gpu][brute_force] step5 BEGIN compute_mrd_matrix" << std::endl;
    auto mrd_compute_event =
        compute_mrd_matrix<Float>(queue, core_distances, mrd_matrix, metric, alpha, { core_event });
    mrd_compute_event.wait_and_throw();
    std::cout << "[hdbscan-gpu][brute_force] step5 END   compute_mrd_matrix ok" << std::endl;

    // Step 4: Build MST using GPU Boruvka's algorithm with precomputed MRD matrix
    std::cout << "[hdbscan-gpu][brute_force] step6 BEGIN alloc mst_from/to/weights (" << edge_count
              << " edges)" << std::endl;
    auto [mst_from, mst_from_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [mst_to, mst_to_event] =
        pr::ndarray<std::int32_t, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    auto [mst_weights, mst_weights_event] =
        pr::ndarray<Float, 1>::zeros(queue, edge_count, sycl::usm::alloc::device);
    mst_from_event.wait_and_throw();
    mst_to_event.wait_and_throw();
    mst_weights_event.wait_and_throw();
    std::cout << "[hdbscan-gpu][brute_force] step6 END   alloc mst arrays ok" << std::endl;

    std::cout << "[hdbscan-gpu][brute_force] step7 BEGIN build_mst (Boruvka)" << std::endl;
    auto mst_event =
        build_mst<Float>(queue,
                         mrd_matrix,
                         mst_from,
                         mst_to,
                         mst_weights,
                         row_count,
                         { mrd_compute_event, mst_from_event, mst_to_event, mst_weights_event });
    mst_event.wait_and_throw();
    std::cout << "[hdbscan-gpu][brute_force] step7 END   build_mst ok" << std::endl;

    // Step 5: Sort MST edges by weight using radix sort primitive
    std::cout << "[hdbscan-gpu][brute_force] step8 BEGIN sort_mst_by_weight" << std::endl;
    auto sort_event =
        sort_mst_by_weight<Float>(queue, mst_from, mst_to, mst_weights, edge_count, { mst_event });
    sort_event.wait_and_throw();
    std::cout << "[hdbscan-gpu][brute_force] step8 END   sort_mst_by_weight ok" << std::endl;

    // Step 6: Extract flat clusters using EOM on device
    std::cout << "[hdbscan-gpu][brute_force] step9 BEGIN alloc arr_responses" << std::endl;
    auto [arr_responses, responses_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, -1, sycl::usm::alloc::device);
    responses_event.wait_and_throw();
    std::cout << "[hdbscan-gpu][brute_force] step9 END   alloc arr_responses ok" << std::endl;

    std::cout << "[hdbscan-gpu][brute_force] step10 BEGIN extract_clusters" << std::endl;
    auto cluster_event = extract_clusters<Float>(queue,
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
                                                 max_cluster_size);
    cluster_event.wait_and_throw();
    std::cout << "[hdbscan-gpu][brute_force] step10 END   extract_clusters ok" << std::endl;

    // Count clusters via GPU max reduction
    std::cout << "[hdbscan-gpu][brute_force] step11 BEGIN max-label reduction" << std::endl;
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
    std::cout << "[hdbscan-gpu][brute_force] step11 END   max_label=" << max_label
              << " cluster_count=" << cluster_count << std::endl;

    std::cout << "[hdbscan-gpu][brute_force] step12 BEGIN make_results" << std::endl;
    auto out = make_results<Float>(queue, desc, arr_responses, cluster_count, local_data);
    std::cout << "[hdbscan-gpu][brute_force] step12 END   make_results ok" << std::endl;
    return out;
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
