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

#pragma once

#include "oneapi/dal/algo/hdbscan/backend/gpu/block_tree.hpp"
#include "oneapi/dal/algo/hdbscan/backend/gpu/kernel_impl.hpp"
#include "oneapi/dal/algo/hdbscan/backend/gpu/results.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

namespace oneapi::dal::hdbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

/// Run HDBSCAN on the GPU over the block tree, for every metric but cosine: tree k-NN core
/// distances, Boruvka MST over the same tree, cluster extraction on host.
///
/// @tparam Float Floating-point type
///
/// @param[in] queue             The SYCL queue
/// @param[in] desc              Algorithm descriptor
/// @param[in] local_data        Input data table of size `n x d`
/// @param[in] alpha_scales_core Divide the core distances by alpha too
///
/// @return oneAPI `compute_result`
template <typename Float>
inline compute_result<task::clustering> compute_with_block_tree(
    sycl::queue& queue,
    const detail::descriptor_base<task::clustering>& desc,
    const table& local_data,
    bool alpha_scales_core) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_block_tree, queue);

    const std::int64_t row_count = local_data.get_row_count();
    const auto metric = desc.get_metric();
    const double degree = desc.get_degree();
    const Float inv_alpha = static_cast<Float>(1.0 / desc.get_alpha());
    ONEDAL_ASSERT(metric != distance_metric::cosine);

    array<std::int32_t> mst_from;
    array<std::int32_t> mst_to;
    array<Float> mst_weights;

    if (row_count > 1) {
        const auto data_nd = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);
        const auto tree = build_block_tree<Float>(queue, data_nd);

        const std::int64_t min_samples = std::min(desc.get_min_samples(), row_count);
        auto core = compute_block_tree_core_distances<Float>(queue,
                                                             data_nd,
                                                             tree,
                                                             min_samples,
                                                             metric,
                                                             degree,
                                                             desc.get_distance_block_size());
        if (alpha_scales_core) {
            Float* core_ptr = core.get_mutable_data();
            queue
                .submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::range<1>(row_count), [=](sycl::id<1> idx) {
                        core_ptr[idx[0]] *= inv_alpha;
                    });
                })
                .wait_and_throw();
        }

        std::tie(mst_from, mst_to, mst_weights) =
            build_mst_block_tree<Float>(queue, tree, core, metric, degree, inv_alpha);
    }

    return make_results_from_mst<Float>(queue, desc, local_data, mst_from, mst_to, mst_weights);
}

/// Run HDBSCAN on the GPU over the `n x n` MRD matrix: distance matrix, core distances by
/// k-selection, Boruvka MST over the matrix, cluster extraction on host.
///
/// @tparam Float Floating-point type
///
/// @param[in] queue             The SYCL queue
/// @param[in] desc              Algorithm descriptor
/// @param[in] local_data        Input data table of size `n x d`
/// @param[in] alpha_scales_core Divide the core distances by alpha too
///
/// @return oneAPI `compute_result`
template <typename Float>
inline compute_result<task::clustering> compute_with_matrix(
    sycl::queue& queue,
    const detail::descriptor_base<task::clustering>& desc,
    const table& local_data,
    bool alpha_scales_core) {
    ONEDAL_PROFILER_TASK(hdbscan.compute_matrix, queue);

    const std::int64_t row_count = local_data.get_row_count();
    const std::int64_t min_samples = std::min(desc.get_min_samples(), row_count);
    const auto metric = desc.get_metric();

    array<std::int32_t> mst_from;
    array<std::int32_t> mst_to;
    array<Float> mst_weights;

    if (row_count > 1) {
        const auto data_nd = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);

        // Not zero-filled: every entry is written, and a fill of more than 2^31 elements is
        // rejected.
        auto matrix =
            pr::ndarray<Float, 2>::empty(queue, { row_count, row_count }, sycl::usm::alloc::device);
        auto dist_event =
            compute_distance_matrix<Float>(queue, data_nd, matrix, metric, desc.get_degree());

        auto core = pr::ndarray<Float, 1>::empty(queue, row_count, sycl::usm::alloc::device);
        auto core_event = compute_core_distances<Float>(queue,
                                                        matrix,
                                                        core,
                                                        min_samples,
                                                        row_count,
                                                        metric,
                                                        { dist_event });
        auto mrd_event = compute_mrd_matrix<Float>(queue,
                                                   core,
                                                   matrix,
                                                   metric,
                                                   desc.get_alpha(),
                                                   alpha_scales_core,
                                                   { core_event });
        std::tie(mst_from, mst_to, mst_weights) = build_mst<Float>(queue, matrix, { mrd_event });
    }

    return make_results_from_mst<Float>(queue, desc, local_data, mst_from, mst_to, mst_weights);
}

/// Smallest column count at which the matrix path is used when the matrix fits: the box bounds
/// prune little there, while the matrix comes from GEMM.
constexpr std::int64_t matrix_min_column_count = 16;

/// Run HDBSCAN on the GPU through the matrix or the block tree, whichever is faster and fits.
///
/// Both give the exact MST, so the choice only changes the speed and the memory.
///
/// @tparam Float Floating-point type
///
/// @param[in] queue             The SYCL queue
/// @param[in] desc              Algorithm descriptor
/// @param[in] local_data        Input data table of size `n x d`
/// @param[in] alpha_scales_core Divide the core distances by alpha too, as scikit-learn's
///                              brute-force path does; the tree methods leave them unscaled
///
/// @return oneAPI `compute_result`
///
/// @throws domain_error for cosine when the matrix does not fit on the device
template <typename Float>
inline compute_result<task::clustering> compute_gpu(
    sycl::queue& queue,
    const detail::descriptor_base<task::clustering>& desc,
    const table& local_data,
    bool alpha_scales_core) {
    const std::int64_t row_count = local_data.get_row_count();
    const std::int64_t column_count = local_data.get_column_count();
    const std::int64_t min_samples = std::min(desc.get_min_samples(), row_count);
    const bool cosine = desc.get_metric() == distance_metric::cosine;
    const bool fits = mrd_matrix_fits_device<Float>(queue, row_count, column_count, min_samples);
    if (cosine && !fits) {
        throw domain_error(
            dal::detail::error_messages::hdbscan_brute_force_matrix_does_not_fit_on_device());
    }
    if (cosine || (fits && column_count >= matrix_min_column_count)) {
        return compute_with_matrix<Float>(queue, desc, local_data, alpha_scales_core);
    }
    return compute_with_block_tree<Float>(queue, desc, local_data, alpha_scales_core);
}

#endif

} // namespace oneapi::dal::hdbscan::backend
