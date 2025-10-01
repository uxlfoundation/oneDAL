/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/communicator.hpp"
#include <sycl/sycl.hpp>
#include <limits>
#include <cstdint>
#include <algorithm>
namespace oneapi::dal::hdbscan::backend {

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;
namespace spmd = oneapi::dal::preview::spmd;
namespace de = oneapi::dal::detail;

using dal::backend::context_gpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

constexpr std::int64_t BEST_CAP = 512;

template <typename Float, bool use_weights>
struct get_core_narrow_kernel {
    // signature: data, (optional) weights, cores, core_distances, min_observations, deps
    static auto run(sycl::queue& queue,
                    const pr::ndview<Float, 2>& data,
                    const pr::ndview<Float, 2>& /*weights*/,
                    pr::ndview<std::int32_t, 1>& cores,
                    pr::ndview<Float, 1>& core_distances,
                    std::int64_t min_observations,
                    const bk::event_vector& deps) {
        const auto local_row_count = data.get_dimension(0);
        ONEDAL_ASSERT(local_row_count > 0);
        ONEDAL_ASSERT(cores.get_dimension(0) >= local_row_count);
        ONEDAL_ASSERT(core_distances.get_dimension(0) >= local_row_count);

        ONEDAL_ASSERT(min_observations > 0);
        ONEDAL_ASSERT(
            min_observations <= BEST_CAP &&
            "min_observations exceeds BEST_CAP; increase BEST_CAP or use another approach");

        const std::int64_t column_count = data.get_dimension(1);

        const Float* data_ptr = data.get_data();
        std::int32_t* cores_ptr = cores.get_mutable_data();
        Float* core_dist_ptr = core_distances.get_mutable_data();

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(sycl::range<1>{ std::size_t(local_row_count) }, [=](sycl::id<1> idx) {
                const std::int64_t row = idx[0];

                Float best[BEST_CAP];
                for (std::int64_t t = 0; t < min_observations; ++t) {
                    best[t] = std::numeric_limits<Float>::max();
                }
                for (std::int64_t j = 0; j < local_row_count; ++j) {
                    if (j == row)
                        continue;

                    Float sum = Float(0);
                    for (std::int64_t c = 0; c < column_count; ++c) {
                        Float diff =
                            data_ptr[row * column_count + c] - data_ptr[j * column_count + c];
                        sum += diff * diff;
                    }
                    Float dist = sycl::sqrt(sum);

                    std::int64_t worst_idx = 0;
                    Float worst_val = best[0];
                    for (std::int64_t t = 1; t < min_observations; ++t) {
                        if (best[t] > worst_val) {
                            worst_val = best[t];
                            worst_idx = t;
                        }
                    }

                    if (dist < worst_val) {
                        best[worst_idx] = dist;
                    }
                }

                Float kth = best[0];
                for (std::int64_t t = 1; t < min_observations; ++t) {
                    if (best[t] > kth)
                        kth = best[t];
                }

                core_dist_ptr[row] = kth;

                cores_ptr[row] =
                    (kth < std::numeric_limits<Float>::max()) ? std::int32_t(1) : std::int32_t(0);
            });
        });
        return event;
    }
};

// ---------------- compute_mutual_reachability_kernel ----------------
template <typename Float>
struct compute_mutual_reachability_kernel {
    static auto run(sycl::queue& queue,
                    const pr::ndview<Float, 2>& data,
                    const pr::ndview<Float, 1>& core_distances,
                    pr::ndview<Float, 2>& mutual_reachability,
                    const bk::event_vector& deps) {
        const std::int64_t row_count = data.get_dimension(0);
        const std::int64_t column_count = data.get_dimension(1);

        const Float* data_ptr = data.get_data();
        const Float* core_dist_ptr = core_distances.get_data();
        Float* mr_ptr = mutual_reachability.get_mutable_data();

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(
                sycl::range<2>{ std::size_t(row_count), std::size_t(row_count) },
                [=](sycl::id<2> idx) {
                    const std::int64_t i = idx[0];
                    const std::int64_t j = idx[1];

                    if (i <= j) {
                        Float sum = Float(0);
                        for (std::int64_t c = 0; c < column_count; ++c) {
                            Float diff =
                                data_ptr[i * column_count + c] - data_ptr[j * column_count + c];
                            sum += diff * diff;
                        }
                        Float distance = sycl::sqrt(sum);
                        Float mr =
                            sycl::fmax(sycl::fmax(core_dist_ptr[i], core_dist_ptr[j]), distance);
                        mr_ptr[i * row_count + j] = mr;
                        mr_ptr[j * row_count + i] = mr;
                    }
                });
        });
        return event;
    }
};

struct DSU {
    std::vector<int> p, r;
    DSU(int n) : p(n), r(n, 0) {
        for (int i = 0; i < n; i++)
            p[i] = i;
    }
    int find(int x) {
        if (p[x] != x)
            p[x] = find(p[x]);
        return p[x];
    }
    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b)
            return false;
        if (r[a] < r[b])
            std::swap(a, b);
        p[b] = a;
        if (r[a] == r[b])
            r[a]++;
        return true;
    }
};

std::vector<std::tuple<int, int, float>> build_mst(const float* data, int n) {
    std::vector<std::tuple<int, int, float>> edges;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            edges.emplace_back(i, j, data[i * n + j]);
    std::sort(edges.begin(), edges.end(), [](auto& a, auto& b) {
        return std::get<2>(a) < std::get<2>(b);
    });
    DSU dsu(n);
    std::vector<std::tuple<int, int, float>> mst;
    for (auto& [u, v, w] : edges)
        if (dsu.unite(u, v))
            mst.emplace_back(u, v, w);
    return mst;
}

std::vector<std::tuple<int, int, double>> build_mst(const double* data, int n) {
    std::vector<std::tuple<int, int, double>> edges;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            edges.emplace_back(i, j, data[i * n + j]);
    std::sort(edges.begin(), edges.end(), [](auto& a, auto& b) {
        return std::get<2>(a) < std::get<2>(b);
    });
    DSU dsu(n);
    std::vector<std::tuple<int, int, double>> mst;
    for (auto& [u, v, w] : edges)
        if (dsu.unite(u, v))
            mst.emplace_back(u, v, w);
    return mst;
}

std::vector<int> extract_clusters(std::vector<std::tuple<int, int, float>>& mst,
                                  int n,
                                  int num_clusters) {
    std::sort(mst.begin(), mst.end(), [](auto& a, auto& b) {
        return std::get<2>(a) > std::get<2>(b);
    });
    DSU dsu(n);
    for (int i = num_clusters - 1; i < (int)mst.size(); i++) {
        auto [u, v, w] = mst[i];
        dsu.unite(u, v);
    }
    std::vector<int> labels(n);
    for (int i = 0; i < n; i++)
        labels[i] = dsu.find(i);
    return labels;
}

std::vector<int> extract_clusters(std::vector<std::tuple<int, int, double>>& mst,
                                  int n,
                                  int num_clusters) {
    std::sort(mst.begin(), mst.end(), [](auto& a, auto& b) {
        return std::get<2>(a) > std::get<2>(b);
    });
    DSU dsu(n);
    for (int i = num_clusters - 1; i < (int)mst.size(); i++) {
        auto [u, v, w] = mst[i];
        dsu.unite(u, v);
    }
    std::vector<int> labels(n);
    for (int i = 0; i < n; i++)
        labels[i] = dsu.find(i);
    return labels;
}

template <typename Float>
static result_t compute_kernel_dense_impl(const context_gpu& ctx,
                                          const descriptor_t& desc,
                                          const table& local_data,
                                          const table& local_weights) {
    auto& queue = ctx.get_queue();

    const std::int64_t row_count = local_data.get_row_count();
    //const std::int64_t column_count = local_data.get_column_count();

    auto data_nd = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);

    const std::int64_t min_observations = desc.get_min_cluster_size();

    auto [arr_cores, cores_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, 0, sycl::usm::alloc::device);

    auto [arr_core_distances, core_dist_event] =
        pr::ndarray<Float, 1>::full(queue,
                                    row_count,
                                    std::numeric_limits<Float>::infinity(),
                                    sycl::usm::alloc::device);

    auto [arr_mutual_reachability, mr_event] =
        pr::ndarray<Float, 2>::full(queue,
                                    { row_count, row_count },
                                    Float(0),
                                    sycl::usm::alloc::device);

    sycl::event::wait({ cores_event, core_dist_event, mr_event });

    pr::ndview<Float, 2> weights_nd;
    bool use_weights = false;

    auto compute_core_distances_event =
        use_weights ? get_core_narrow_kernel<Float, true>::run(queue,
                                                               data_nd,
                                                               weights_nd,
                                                               arr_cores,
                                                               arr_core_distances,
                                                               min_observations,
                                                               {})
                    : get_core_narrow_kernel<Float, false>::run(queue,
                                                                data_nd,
                                                                weights_nd,
                                                                arr_cores,
                                                                arr_core_distances,
                                                                min_observations,
                                                                {});

    auto compute_mr_distances_event =
        compute_mutual_reachability_kernel<Float>::run(queue,
                                                       data_nd,
                                                       arr_core_distances,
                                                       arr_mutual_reachability,
                                                       { compute_core_distances_event });

    compute_mr_distances_event.wait();
    auto arr_host = arr_mutual_reachability.to_host(queue);
    const auto* data = arr_host.get_data();
    std::int64_t nrows = arr_host.get_dimension(0);

    auto mst = build_mst(data, nrows);

    auto labels = extract_clusters(mst, nrows, 3);

    std::cout << "MST edges:\n";
    for (auto& [u, v, w] : mst)
        std::cout << u << " - " << v << " : " << w << "\n";

    std::cout << "\nCluster labels:\n";
    for (int i = 0; i < nrows; i++)
        std::cout << "Point " << i << " -> Cluster " << labels[i] << "\n";
    result_t result;

    return result;
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return compute_kernel_dense_impl<Float>(ctx, desc, input.get_data(), input.get_weights());
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
