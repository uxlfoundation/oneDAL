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

#include <algorithm>
#include <type_traits>
#include <vector>

#include "oneapi/dal/algo/hdbscan/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/atomic.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"

namespace oneapi::dal::hdbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

/// Points per leaf of `block_tree`.
constexpr std::int64_t block_tree_leaf_size = 32;

/// Children per internal node of `block_tree`.
constexpr std::int64_t block_tree_fanout = 8;

/// Traversal stack depth: `fanout` entries per level, for up to 2^31 points.
constexpr std::int32_t block_tree_stack_size = 96;

/// Work-group size of the traversal kernels.
constexpr std::int64_t block_tree_wg_size = 32;

/// Largest `min_samples` the tree k-NN keeps in registers.
constexpr std::int64_t block_tree_max_k = 256;

/// The Lp distances in accumulated form: squared L2, L1, sum of `|x|^p`, or L-infinity.
///
/// Every term is monotone in `|x_c - y_c|`, so a bounding-box gap never accumulates above the
/// distance to a point inside the box, rounding included.
enum class lp_kind { squared_l2, l1, lp, linf };

/// Accumulate and finalize one of the `lp_kind` distances.
///
/// @tparam Float Floating-point type
/// @tparam Kind  Distance kind
template <typename Float, lp_kind Kind>
struct lp_ops {
    Float degree;

    Float add(Float acc, Float diff) const {
        const Float a = sycl::fabs(diff);
        if constexpr (Kind == lp_kind::squared_l2) {
            return acc + a * a;
        }
        else if constexpr (Kind == lp_kind::l1) {
            return acc + a;
        }
        else if constexpr (Kind == lp_kind::lp) {
            return acc + sycl::pow(a, degree);
        }
        else {
            return sycl::fmax(acc, a);
        }
    }

    Float finalize(Float acc) const {
        if constexpr (Kind == lp_kind::squared_l2) {
            return sycl::sqrt(sycl::fmax(acc, Float(0)));
        }
        else if constexpr (Kind == lp_kind::lp) {
            return sycl::pow(acc, Float(1) / degree);
        }
        else {
            return acc;
        }
    }

    /// Accumulated distance between two rows of length `d`.
    Float distance(const Float* x, const Float* y, std::int64_t d) const {
        Float acc = Float(0);
        for (std::int64_t c = 0; c < d; c++) {
            acc = add(acc, x[c] - y[c]);
        }
        return acc;
    }

    /// Accumulated distance from `x` to the box `[lo, hi]`; 0 inside it.
    Float box_distance(const Float* x, const Float* lo, const Float* hi, std::int64_t d) const {
        Float acc = Float(0);
        for (std::int64_t c = 0; c < d; c++) {
            const Float gap = sycl::fmax(sycl::fmax(lo[c] - x[c], x[c] - hi[c]), Float(0));
            acc = add(acc, gap);
        }
        return acc;
    }
};

/// Call `body(lp_ops<Float, Kind>{...})` with the kind matching `metric`.
///
/// @param[in] metric Any metric but cosine
/// @param[in] degree Minkowski degree
/// @param[in] body   Generic callable taking the `lp_ops`
template <typename Float, typename Body>
inline auto dispatch_lp_ops(distance_metric metric, double degree, Body&& body) {
    ONEDAL_ASSERT(metric != distance_metric::cosine);
    switch (metric) {
        case distance_metric::manhattan: return body(lp_ops<Float, lp_kind::l1>{ Float(1) });
        case distance_metric::minkowski:
            return body(lp_ops<Float, lp_kind::lp>{ static_cast<Float>(degree) });
        case distance_metric::chebyshev: return body(lp_ops<Float, lp_kind::linf>{ Float(1) });
        default: return body(lp_ops<Float, lp_kind::squared_l2>{ Float(2) });
    }
}

/// Write the `lp_ops` distance between every pair of rows into `dist`, from coordinate differences
/// staged through local memory. Euclidean entries stay squared, the others are final.
///
/// The terms are summed in column order, as in the tree search, so both paths see the same
/// distances.
///
/// @tparam Float Floating-point type
/// @tparam Ops   `lp_ops` instance
///
/// @param[in]  queue The SYCL queue
/// @param[in]  data  Input `n x d`, device USM
/// @param[in]  ops   Distance
/// @param[out] dist  `n x n` output, row-major
/// @param[in]  deps  Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float, typename Ops>
inline sycl::event compute_lp_distance_matrix(sycl::queue& queue,
                                              const pr::ndview<Float, 2>& data,
                                              const Ops& ops,
                                              pr::ndview<Float, 2>& dist,
                                              const bk::event_vector& deps = {}) {
    // A 16 x 16 work-group computes a 64 x 64 tile, 4 x 4 entries per lane, over 16-column
    // chunks of both row blocks staged in local memory.
    constexpr std::int64_t lanes = 16;
    constexpr std::int64_t per_lane = 4;
    constexpr std::int64_t tile = lanes * per_lane;
    constexpr std::int64_t chunk = 16;
    const std::int64_t n = data.get_dimension(0);
    const std::int64_t d = data.get_dimension(1);
    const std::int64_t tiles = (n + tile - 1) / tile;
    const std::int64_t stride = dist.get_leading_stride();
    const Float* x = data.get_data();
    Float* out = dist.get_mutable_data();
    constexpr bool keep_squared = std::is_same_v<Ops, lp_ops<Float, lp_kind::squared_l2>>;

    // Blocks of tile rows keep each launch inside the int32 range.
    const std::int64_t block_tiles =
        std::max<std::int64_t>(1, bk::max_range_2d_rows(tiles * lanes) / lanes);
    bk::event_vector events;
    for (std::int64_t first_tile = 0; first_tile < tiles; first_tile += block_tiles) {
        const std::int64_t row_tiles = std::min(block_tiles, tiles - first_tile);
        events.push_back(queue.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            sycl::local_accessor<Float, 2> a({ tile, chunk + 1 }, h);
            sycl::local_accessor<Float, 2> b({ tile, chunk + 1 }, h);
            const sycl::nd_range<2> range({ static_cast<std::size_t>(row_tiles * lanes),
                                            static_cast<std::size_t>(tiles * lanes) },
                                          { lanes, lanes });
            h.parallel_for(range, [=](sycl::nd_item<2> it) {
                const std::int64_t li = it.get_local_id(0);
                const std::int64_t lj = it.get_local_id(1);
                const std::int64_t lane = li * lanes + lj;
                const std::int64_t i0 = (first_tile + it.get_group(0)) * tile;
                const std::int64_t j0 = it.get_group(1) * tile;
                Float acc[per_lane][per_lane];
                for (std::int64_t r = 0; r < per_lane; r++) {
                    for (std::int64_t q = 0; q < per_lane; q++) {
                        acc[r][q] = Float(0);
                    }
                }
                for (std::int64_t c0 = 0; c0 < d; c0 += chunk) {
                    // Each lane stages 4 of the 64 x 16 entries of either block.
                    for (std::int64_t e = lane; e < tile * chunk; e += lanes * lanes) {
                        const std::int64_t row = e / chunk;
                        const std::int64_t c = c0 + e % chunk;
                        a[row][e % chunk] =
                            (c < d && i0 + row < n) ? x[(i0 + row) * d + c] : Float(0);
                        b[row][e % chunk] =
                            (c < d && j0 + row < n) ? x[(j0 + row) * d + c] : Float(0);
                    }
                    sycl::group_barrier(it.get_group());
                    const std::int64_t width = std::min(chunk, d - c0);
                    for (std::int64_t cc = 0; cc < width; cc++) {
                        Float av[per_lane];
                        Float bv[per_lane];
                        for (std::int64_t r = 0; r < per_lane; r++) {
                            av[r] = a[li + r * lanes][cc];
                            bv[r] = b[lj + r * lanes][cc];
                        }
                        for (std::int64_t r = 0; r < per_lane; r++) {
                            for (std::int64_t q = 0; q < per_lane; q++) {
                                acc[r][q] = ops.add(acc[r][q], av[r] - bv[q]);
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
                for (std::int64_t r = 0; r < per_lane; r++) {
                    const std::int64_t i = i0 + li + r * lanes;
                    for (std::int64_t q = 0; q < per_lane; q++) {
                        const std::int64_t j = j0 + lj + q * lanes;
                        if (i < n && j < n) {
                            out[i * stride + j] =
                                keep_squared ? acc[r][q] : ops.finalize(acc[r][q]);
                        }
                    }
                }
            });
        }));
    }
    return bk::wait_or_pass(events);
}

/// The input rows in `kd_order`, grouped into a bounding-box tree.
///
/// Leaves hold `block_tree_leaf_size` consecutive sorted points, internal nodes
/// `block_tree_fanout` consecutive nodes of the level below. Node ids are level-major, leaves
/// first, so a node is a leaf iff its id is below `leaf_count`.
///
/// @tparam Float Floating-point type
template <typename Float>
struct block_tree {
    pr::ndarray<Float, 2> points; ///< `n x d`, in tree order
    pr::ndarray<std::int32_t, 1> original; ///< Input row of each tree position
    pr::ndarray<Float, 2> lower; ///< `node_count x d` box minima
    pr::ndarray<Float, 2> upper; ///< `node_count x d` box maxima
    pr::ndarray<std::int32_t, 1> first; ///< First child, or first position for a leaf
    pr::ndarray<std::int32_t, 1> last; ///< One past the last child or position
    pr::ndarray<std::int32_t, 1> min_original; ///< Smallest input row under each node
    pr::ndarray<std::int32_t, 1> leaf_begin; ///< First leaf under each node
    pr::ndarray<std::int32_t, 1> leaf_end; ///< One past the last leaf under each node
    std::vector<std::int64_t> level_begin; ///< First node id of each level, plus `node_count`
    std::int64_t row_count = 0;
    std::int64_t column_count = 0;

    std::int64_t leaf_count() const {
        return level_begin[1];
    }
    std::int64_t node_count() const {
        return level_begin.back();
    }
    std::int64_t level_count() const {
        return static_cast<std::int64_t>(level_begin.size()) - 1;
    }
};

/// Reduce a per-position value to every tree node, bottom up.
///
/// @param[in]  queue       The SYCL queue
/// @param[in]  tree        The tree
/// @param[in]  point_value Value of each tree position, length `n`
/// @param[out] node_value  Value of each node, length `node_count`
/// @param[in]  combine     Associative device callable `(T, T) -> T`
/// @param[in]  deps        Events that must complete before submission
///
/// @return Event signaling completion of the root level
template <typename Float, typename T, typename Combine>
inline sycl::event reduce_to_nodes(sycl::queue& queue,
                                   const block_tree<Float>& tree,
                                   const T* point_value,
                                   T* node_value,
                                   Combine combine,
                                   const bk::event_vector& deps) {
    const std::int32_t* first = tree.first.get_data();
    const std::int32_t* last = tree.last.get_data();
    sycl::event event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(tree.leaf_count()), [=](sycl::id<1> idx) {
            const std::int64_t node = idx[0];
            T acc = point_value[first[node]];
            for (std::int32_t p = first[node] + 1; p < last[node]; p++) {
                acc = combine(acc, point_value[p]);
            }
            node_value[node] = acc;
        });
    });
    for (std::int64_t level = 1; level < tree.level_count(); level++) {
        const std::int64_t begin = tree.level_begin[level];
        const std::int64_t count = tree.level_begin[level + 1] - begin;
        event = queue.submit([&](sycl::handler& h) {
            h.depends_on(event);
            h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
                const std::int64_t node = begin + idx[0];
                T acc = node_value[first[node]];
                for (std::int32_t c = first[node] + 1; c < last[node]; c++) {
                    acc = combine(acc, node_value[c]);
                }
                node_value[node] = acc;
            });
        });
    }
    return event;
}

/// Order the rows of `data` like a balanced kd-tree: each level splits every segment at its
/// median along the segment's widest column, down to segments of `block_tree_leaf_size` rows.
///
/// @param[in] queue The SYCL queue
/// @param[in] data  Input `n x d`, device USM
///
/// @return Input row of each tree position; complete on return
template <typename Float>
inline pr::ndarray<std::uint32_t, 1> kd_order(sycl::queue& queue,
                                              const pr::ndview<Float, 2>& data) {
    const std::int64_t n = data.get_dimension(0);
    const std::int64_t d = data.get_dimension(1);
    const auto alloc = sycl::usm::alloc::device;

    std::int64_t level_count = 0;
    while ((block_tree_leaf_size << level_count) < n) {
        level_count++;
    }
    // Segments are position ranges of a padded length, so a level needs no segment table.
    const std::int64_t padded = block_tree_leaf_size << level_count;
    const std::int64_t max_segments = std::int64_t(1) << std::max<std::int64_t>(level_count - 1, 0);

    auto order = pr::ndarray<std::uint32_t, 1>::empty(queue, n, alloc);
    auto next = pr::ndarray<std::uint32_t, 1>::empty(queue, n, alloc);
    auto moved = pr::ndarray<std::uint32_t, 1>::empty(queue, n, alloc);
    auto keys = pr::ndarray<Float, 1>::empty(queue, n, alloc);
    auto seg_lo = pr::ndarray<Float, 1>::empty(queue, max_segments * d, alloc);
    auto seg_hi = pr::ndarray<Float, 1>::empty(queue, max_segments * d, alloc);
    sycl::event last_event = order.arange(queue);

    const Float* data_ptr = data.get_data();
    Float* key_ptr = keys.get_mutable_data();
    Float* lo_ptr = seg_lo.get_mutable_data();
    Float* hi_ptr = seg_hi.get_mutable_data();
    for (std::int64_t level = 0; level < level_count; level++) {
        const std::int64_t seg_count = std::int64_t(1) << level;
        const std::int64_t seg_size = padded >> level;
        const std::uint32_t* order_ptr = order.get_data();
        std::uint32_t* moved_ptr = moved.get_mutable_data();
        std::uint32_t* next_ptr = next.get_mutable_data();
        // One work-group per (segment, column) box side, in launches that stay within int32.
        const std::int64_t wg = std::min<std::int64_t>(seg_size, 256);
        const std::int64_t group_count = seg_count * d;
        const std::int64_t block_groups = bk::max_range_2d_rows(wg);
        bk::event_vector box_events;
        for (std::int64_t first_group = 0; first_group < group_count; first_group += block_groups) {
            const std::int64_t groups = std::min(block_groups, group_count - first_group);
            box_events.push_back(queue.submit([&](sycl::handler& h) {
                h.depends_on(last_event);
                h.parallel_for(
                    bk::make_multiple_nd_range_1d(groups * wg, wg),
                    [=](sycl::nd_item<1> it) {
                        const std::int64_t g = first_group + it.get_group(0);
                        const std::int64_t s = g / d;
                        const std::int64_t c = g % d;
                        const std::int64_t end = std::min(n, (s + 1) * seg_size);
                        Float lo = std::numeric_limits<Float>::max();
                        Float hi = std::numeric_limits<Float>::lowest();
                        for (std::int64_t p = s * seg_size + it.get_local_id(0); p < end; p += wg) {
                            const Float v =
                                data_ptr[static_cast<std::int64_t>(order_ptr[p]) * d + c];
                            lo = sycl::fmin(lo, v);
                            hi = sycl::fmax(hi, v);
                        }
                        lo = sycl::reduce_over_group(it.get_group(), lo, sycl::minimum<Float>());
                        hi = sycl::reduce_over_group(it.get_group(), hi, sycl::maximum<Float>());
                        if (it.get_local_id(0) == 0) {
                            lo_ptr[s * d + c] = lo;
                            hi_ptr[s * d + c] = hi;
                        }
                    });
            }));
        }

        // One sort on `segment * 2^bits + coordinate cell` while that stays below 2^24, which
        // every float key holds exactly; past that, by coordinate and then stably by segment.
        const std::int64_t cell_bits = 24 - level;
        const bool one_sort = cell_bits >= 8;
        const Float cells = one_sort ? Float((std::int64_t(1) << cell_bits) - 1) : Float(0);
        auto key_event = queue.submit([&](sycl::handler& h) {
            h.depends_on(box_events);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                const std::int64_t p = idx[0];
                const std::int64_t s = p / seg_size;
                std::int64_t widest = 0;
                Float width = hi_ptr[s * d] - lo_ptr[s * d];
                for (std::int64_t c = 1; c < d; c++) {
                    const Float w = hi_ptr[s * d + c] - lo_ptr[s * d + c];
                    if (w > width) {
                        width = w;
                        widest = c;
                    }
                }
                const Float v = data_ptr[static_cast<std::int64_t>(order_ptr[p]) * d + widest];
                if (one_sort) {
                    const Float t =
                        (width > Float(0)) ? (v - lo_ptr[s * d + widest]) / width : Float(0);
                    // `!(t >= 0)` also catches NaN, which the MST rejects later.
                    const Float cell = !(t >= Float(0))  ? Float(0)
                                       : (t >= Float(1)) ? cells
                                                         : sycl::floor(t * cells);
                    key_ptr[p] = static_cast<Float>(s << cell_bits) + cell;
                }
                else {
                    key_ptr[p] = v;
                }
                moved_ptr[p] = static_cast<std::uint32_t>(p);
            });
        });
        auto sort_event = pr::radix_sort_indices_inplace_dpl<Float, std::uint32_t>(queue,
                                                                                   keys,
                                                                                   moved,
                                                                                   { key_event });
        if (!one_sort) {
            auto seg_key_event = queue.submit([&](sycl::handler& h) {
                h.depends_on(sort_event);
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    key_ptr[idx[0]] = static_cast<Float>(moved_ptr[idx[0]] / seg_size);
                });
            });
            sort_event =
                pr::radix_sort_indices_inplace_dpl<Float, std::uint32_t>(queue,
                                                                         keys,
                                                                         moved,
                                                                         { seg_key_event });
        }
        last_event = queue.submit([&](sycl::handler& h) {
            h.depends_on(sort_event);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                next_ptr[idx[0]] = order_ptr[moved_ptr[idx[0]]];
            });
        });
        std::swap(order, next);
    }
    last_event.wait_and_throw();
    return order;
}

/// Order the rows of `data` with `kd_order` and build the `block_tree` over them.
///
/// @param[in] queue The SYCL queue
/// @param[in] data  Input `n x d`, device USM
///
/// @return The tree; complete on return
template <typename Float>
inline block_tree<Float> build_block_tree(sycl::queue& queue, const pr::ndview<Float, 2>& data) {
    ONEDAL_PROFILER_TASK(hdbscan.build_block_tree, queue);

    const std::int64_t n = data.get_dimension(0);
    const std::int64_t d = data.get_dimension(1);
    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(n <= dal::detail::limits<std::int32_t>::max());

    block_tree<Float> tree;
    tree.row_count = n;
    tree.column_count = d;

    const auto order = kd_order<Float>(queue, data);
    const Float* data_ptr = data.get_data();

    tree.original = pr::ndarray<std::int32_t, 1>::empty(queue, n, sycl::usm::alloc::device);
    tree.points = pr::ndarray<Float, 2>::empty(queue, { n, d }, sycl::usm::alloc::device);
    const std::uint32_t* order_ptr = order.get_data();
    std::int32_t* original_ptr = tree.original.get_mutable_data();
    Float* points_ptr = tree.points.get_mutable_data();
    auto original_event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            original_ptr[idx[0]] = static_cast<std::int32_t>(order_ptr[idx[0]]);
        });
    });
    auto gather_event =
        bk::parallel_for_2d_by_row_blocks(queue, n, d, [=](std::int64_t pos, std::int64_t c) {
            points_ptr[pos * d + c] = data_ptr[static_cast<std::int64_t>(order_ptr[pos]) * d + c];
        });

    // Levels, leaves first, until the top level fits one node's children.
    std::vector<std::int64_t> level_size{ (n + block_tree_leaf_size - 1) / block_tree_leaf_size };
    while (level_size.back() > block_tree_fanout) {
        level_size.push_back((level_size.back() + block_tree_fanout - 1) / block_tree_fanout);
    }
    tree.level_begin.assign(1, 0);
    for (const auto size : level_size) {
        tree.level_begin.push_back(tree.level_begin.back() + size);
    }
    ONEDAL_ASSERT(tree.level_count() * block_tree_fanout <= block_tree_stack_size);

    const std::int64_t node_count = tree.node_count();
    auto first_host = array<std::int32_t>::empty(node_count);
    auto last_host = array<std::int32_t>::empty(node_count);
    std::int32_t* first_h = first_host.get_mutable_data();
    std::int32_t* last_h = last_host.get_mutable_data();
    for (std::int64_t leaf = 0; leaf < tree.leaf_count(); leaf++) {
        first_h[leaf] = static_cast<std::int32_t>(leaf * block_tree_leaf_size);
        last_h[leaf] = static_cast<std::int32_t>(std::min(n, (leaf + 1) * block_tree_leaf_size));
    }
    for (std::int64_t level = 1; level < tree.level_count(); level++) {
        const std::int64_t child_begin = tree.level_begin[level - 1];
        const std::int64_t child_end = tree.level_begin[level];
        for (std::int64_t t = 0; t < level_size[level]; t++) {
            const std::int64_t node = tree.level_begin[level] + t;
            first_h[node] = static_cast<std::int32_t>(child_begin + t * block_tree_fanout);
            last_h[node] = static_cast<std::int32_t>(
                std::min(child_end, child_begin + (t + 1) * block_tree_fanout));
        }
    }
    auto leaf_begin_host = array<std::int32_t>::empty(node_count);
    auto leaf_end_host = array<std::int32_t>::empty(node_count);
    std::int32_t* leaf_begin_h = leaf_begin_host.get_mutable_data();
    std::int32_t* leaf_end_h = leaf_end_host.get_mutable_data();
    for (std::int64_t node = 0; node < node_count; node++) {
        const bool is_leaf = node < tree.leaf_count();
        leaf_begin_h[node] =
            is_leaf ? static_cast<std::int32_t>(node) : leaf_begin_h[first_h[node]];
        leaf_end_h[node] =
            is_leaf ? static_cast<std::int32_t>(node + 1) : leaf_end_h[last_h[node] - 1];
    }
    tree.first = pr::ndarray<std::int32_t, 1>::wrap(first_host).to_device(queue);
    tree.last = pr::ndarray<std::int32_t, 1>::wrap(last_host).to_device(queue);
    tree.leaf_begin = pr::ndarray<std::int32_t, 1>::wrap(leaf_begin_host).to_device(queue);
    tree.leaf_end = pr::ndarray<std::int32_t, 1>::wrap(leaf_end_host).to_device(queue);

    // Boxes: leaves from their points, then each level from its children.
    tree.lower = pr::ndarray<Float, 2>::empty(queue, { node_count, d }, sycl::usm::alloc::device);
    tree.upper = pr::ndarray<Float, 2>::empty(queue, { node_count, d }, sycl::usm::alloc::device);
    Float* lower_ptr = tree.lower.get_mutable_data();
    Float* upper_ptr = tree.upper.get_mutable_data();
    const std::int32_t* first_ptr = tree.first.get_data();
    const std::int32_t* last_ptr = tree.last.get_data();
    sycl::event box_event = bk::parallel_for_2d_by_row_blocks(
        queue,
        tree.leaf_count(),
        d,
        [=](std::int64_t node, std::int64_t c) {
            Float lo = points_ptr[first_ptr[node] * d + c];
            Float hi = lo;
            for (std::int32_t p = first_ptr[node] + 1; p < last_ptr[node]; p++) {
                lo = sycl::fmin(lo, points_ptr[p * d + c]);
                hi = sycl::fmax(hi, points_ptr[p * d + c]);
            }
            lower_ptr[node * d + c] = lo;
            upper_ptr[node * d + c] = hi;
        },
        { gather_event });
    for (std::int64_t level = 1; level < tree.level_count(); level++) {
        const std::int64_t begin = tree.level_begin[level];
        box_event = bk::parallel_for_2d_by_row_blocks(
            queue,
            level_size[level],
            d,
            [=](std::int64_t t, std::int64_t c) {
                const std::int64_t node = begin + t;
                Float lo = lower_ptr[first_ptr[node] * d + c];
                Float hi = upper_ptr[first_ptr[node] * d + c];
                for (std::int32_t k = first_ptr[node] + 1; k < last_ptr[node]; k++) {
                    lo = sycl::fmin(lo, lower_ptr[k * d + c]);
                    hi = sycl::fmax(hi, upper_ptr[k * d + c]);
                }
                lower_ptr[node * d + c] = lo;
                upper_ptr[node * d + c] = hi;
            },
            { box_event });
    }

    tree.min_original =
        pr::ndarray<std::int32_t, 1>::empty(queue, node_count, sycl::usm::alloc::device);
    auto min_original_event = reduce_to_nodes(queue,
                                              tree,
                                              tree.original.get_data(),
                                              tree.min_original.get_mutable_data(),
                                              [](std::int32_t a, std::int32_t b) {
                                                  return sycl::min(a, b);
                                              },
                                              { original_event });

    // `order` and the host staging arrays are freed on return.
    sycl::event::wait_and_throw({ box_event, min_original_event });
    return tree;
}

/// Push the nodes `[begin, end)` onto a traversal stack so that the one holding `own_leaf`, if
/// any, is popped first and the rest in id order.
///
/// @param[in,out] stack      Traversal stack
/// @param[in,out] sp         Stack size
/// @param[in]     begin      First node to push
/// @param[in]     end        One past the last node to push
/// @param[in]     own_leaf   Leaf of the query
/// @param[in]     leaf_begin First leaf under each node
/// @param[in]     leaf_end   One past the last leaf under each node
inline void push_nodes_own_first(std::int32_t* stack,
                                 std::int32_t& sp,
                                 std::int32_t begin,
                                 std::int32_t end,
                                 std::int64_t own_leaf,
                                 const std::int32_t* leaf_begin,
                                 const std::int32_t* leaf_end) {
    std::int32_t own = -1;
    for (std::int32_t c = end - 1; c >= begin; c--) {
        if (leaf_begin[c] <= own_leaf && own_leaf < leaf_end[c]) {
            own = c;
            continue;
        }
        stack[sp++] = c;
    }
    if (own >= 0) {
        stack[sp++] = own;
    }
}

/// Exact `k`-th nearest neighbor distance of every point, by a pruned tree traversal.
///
/// Each sub-group walks the tree together and descends into a node when any lane's
/// `k`-th best can still improve there. The point itself counts as its first neighbor.
///
/// @tparam Float Floating-point type
/// @tparam K     Capacity of the per-lane neighbor list, at least `k`
/// @tparam Ops   `lp_ops` instance
///
/// @param[in]  queue          The SYCL queue
/// @param[in]  tree           The tree
/// @param[in]  ops            Distance
/// @param[in]  k              `min_samples`, in `[1, min(K, n)]`
/// @param[out] core_distances Core distance of each tree position, length `n`
/// @param[in]  deps           Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float, std::int64_t K, typename Ops>
inline sycl::event block_tree_core_distances(sycl::queue& queue,
                                             const block_tree<Float>& tree,
                                             const Ops& ops,
                                             std::int64_t k,
                                             Float* core_distances,
                                             const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(k >= 1 && k <= K);
    ONEDAL_ASSERT(k <= tree.row_count);

    const std::int64_t n = tree.row_count;
    const std::int64_t d = tree.column_count;
    const std::int64_t leaf_count = tree.leaf_count();
    const std::int32_t root_begin = static_cast<std::int32_t>(tree.level_begin.end()[-2]);
    const std::int32_t root_end = static_cast<std::int32_t>(tree.node_count());
    const Float* points = tree.points.get_data();
    const Float* lower = tree.lower.get_data();
    const Float* upper = tree.upper.get_data();
    const std::int32_t* first = tree.first.get_data();
    const std::int32_t* last = tree.last.get_data();
    const std::int32_t* leaf_begin = tree.leaf_begin.get_data();
    const std::int32_t* leaf_end = tree.leaf_end.get_data();
    const auto range = bk::make_multiple_nd_range_1d(n, block_tree_wg_size);

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::nd_item<1> it) {
            const auto sg = it.get_sub_group();
            const std::int64_t pos = it.get_global_id(0);
            const bool active = pos < n;
            // Padding lanes take the last row, so a sub-group shares one leaf and one stack.
            const std::int64_t safe_pos = active ? pos : n - 1;
            const Float* x = points + safe_pos * d;
            const std::int64_t own_leaf = safe_pos / block_tree_leaf_size;

            Float best[K];
            for (std::int64_t t = 0; t < K; t++) {
                best[t] = std::numeric_limits<Float>::max();
            }
            auto push = [&](Float v) {
                if (v < best[k - 1]) {
                    std::int64_t t = k - 1;
                    for (; t > 0 && best[t - 1] > v; t--) {
                        best[t] = best[t - 1];
                    }
                    best[t] = v;
                }
            };

            if (active) {
                for (std::int32_t p = first[own_leaf]; p < last[own_leaf]; p++) {
                    push(ops.distance(x, points + p * d, d));
                }
            }

            std::int32_t stack[block_tree_stack_size];
            std::int32_t sp = 0;
            push_nodes_own_first(stack, sp, root_begin, root_end, own_leaf, leaf_begin, leaf_end);
            while (sp > 0) {
                const std::int32_t node = stack[--sp];
                const bool want =
                    active &&
                    ops.box_distance(x, lower + node * d, upper + node * d, d) < best[k - 1];
                if (!sycl::any_of_group(sg, want)) {
                    continue;
                }
                if (node < leaf_count) {
                    if (want && node != own_leaf) {
                        for (std::int32_t p = first[node]; p < last[node]; p++) {
                            push(ops.distance(x, points + p * d, d));
                        }
                    }
                }
                else {
                    push_nodes_own_first(stack,
                                         sp,
                                         first[node],
                                         last[node],
                                         own_leaf,
                                         leaf_begin,
                                         leaf_end);
                }
            }

            if (active) {
                core_distances[pos] = ops.finalize(best[k - 1]);
            }
        });
    });
}

/// One Boruvka round over the tree: each point's nearest point in another component under MRD,
/// and each component's smallest MRD.
///
/// Matches a full scan exactly: a point keeps its smallest MRD, ties going to the lowest input
/// row. A point is pruned only where it cannot reach its component's minimum, which
/// `comp_best_mrd` bounds from above while the round runs; such points report no candidate or a
/// larger MRD, which the component reduction ignores.
///
/// @tparam Float Floating-point type
/// @tparam Ops   `lp_ops` instance
///
/// @param[in]     queue          The SYCL queue
/// @param[in]     tree           The tree
/// @param[in]     ops            Distance
/// @param[in]     core           Core distance of each tree position, as it enters MRD
/// @param[in]     comp           Component of each tree position, an input row id
/// @param[in]     node_min_core  Smallest `core` under each node
/// @param[in]     node_comp      Component shared by every point under a node, or -1
/// @param[in]     inv_alpha      Scale of `dist(i, j)` inside MRD
/// @param[in,out] comp_best_mrd  Per-component smallest MRD, by component id; reset to the max
///                               value before the round
/// @param[out]    pt_best_mrd    Best MRD of each input row
/// @param[out]    pt_best_idx    Best input row of each input row, -1 if none
/// @param[in]     seed_only      Only search the query's leaf and its sibling leaves and lower
///                               `comp_best_mrd`, without writing the per-point bests; run
///                               before the full search so its pruning starts from a finite bound
/// @param[in]     deps           Events that must complete before submission
///
/// @return Event signaling completion
template <typename Float, typename Ops>
inline sycl::event block_tree_boruvka_find(sycl::queue& queue,
                                           const block_tree<Float>& tree,
                                           const Ops& ops,
                                           const Float* core,
                                           const std::int32_t* comp,
                                           const Float* node_min_core,
                                           const std::int32_t* node_comp,
                                           Float inv_alpha,
                                           Float* comp_best_mrd,
                                           Float* pt_best_mrd,
                                           std::int32_t* pt_best_idx,
                                           bool seed_only,
                                           const bk::event_vector& deps = {}) {
    const std::int64_t n = tree.row_count;
    const std::int64_t d = tree.column_count;
    const std::int64_t leaf_count = tree.leaf_count();
    const std::int32_t root_begin = static_cast<std::int32_t>(tree.level_begin.end()[-2]);
    const std::int32_t root_end = static_cast<std::int32_t>(tree.node_count());
    const Float* points = tree.points.get_data();
    const std::int32_t* original = tree.original.get_data();
    const Float* lower = tree.lower.get_data();
    const Float* upper = tree.upper.get_data();
    const std::int32_t* first = tree.first.get_data();
    const std::int32_t* last = tree.last.get_data();
    const std::int32_t* min_original = tree.min_original.get_data();
    const std::int32_t* leaf_begin = tree.leaf_begin.get_data();
    const std::int32_t* leaf_end = tree.leaf_end.get_data();
    const std::int64_t leaf_parent_begin = (tree.level_count() > 1) ? tree.level_begin[1] : -1;
    const auto range = bk::make_multiple_nd_range_1d(n, block_tree_wg_size);

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::nd_item<1> it) {
            using bound_ref_t = sycl::atomic_ref<Float,
                                                 sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>;
            constexpr std::int32_t none = std::numeric_limits<std::int32_t>::max();
            const auto sg = it.get_sub_group();
            const std::int64_t pos = it.get_global_id(0);
            // Padding lanes take the last row, so a sub-group shares one leaf and one stack.
            const std::int64_t safe_pos = (pos < n) ? pos : n - 1;
            const Float* x = points + safe_pos * d;
            const std::int64_t own_leaf = safe_pos / block_tree_leaf_size;
            const std::int32_t my_comp = comp[safe_pos];
            const Float my_core = core[safe_pos];
            bound_ref_t bound_ref(comp_best_mrd[my_comp]);

            // Not the component minimum if its own lower bound already exceeds it.
            bool active = (pos < n) && !(my_core > bound_ref.load());
            Float best = std::numeric_limits<Float>::max();
            std::int32_t best_row = none;

            auto visit_leaf = [&](std::int64_t leaf) {
                const Float bound = bound_ref.load();
                for (std::int32_t p = first[leaf]; p < last[leaf]; p++) {
                    if (comp[p] == my_comp) {
                        continue;
                    }
                    const Float lb = sycl::fmax(my_core, core[p]);
                    const std::int32_t row = original[p];
                    if (lb > best || (lb == best && row > best_row) || lb > bound) {
                        continue;
                    }
                    const Float dist = ops.finalize(ops.distance(x, points + p * d, d));
                    const Float mrd = sycl::fmax(lb, dist * inv_alpha);
                    if (mrd < best || (best_row != none && mrd == best && row < best_row)) {
                        best = mrd;
                        best_row = row;
                    }
                }
            };

            if (seed_only) {
                if (!active) {
                    return;
                }
                if (leaf_parent_begin < 0) {
                    visit_leaf(own_leaf);
                }
                else {
                    const std::int64_t parent = leaf_parent_begin + own_leaf / block_tree_fanout;
                    for (std::int32_t leaf = first[parent]; leaf < last[parent]; leaf++) {
                        visit_leaf(leaf);
                    }
                }
                if (best_row != none) {
                    bk::atomic_global_min(comp_best_mrd + my_comp, best);
                }
                return;
            }

            if (active) {
                visit_leaf(own_leaf);
                if (best_row != none) {
                    bk::atomic_global_min(comp_best_mrd + my_comp, best);
                }
            }

            std::int32_t stack[block_tree_stack_size];
            std::int32_t sp = 0;
            push_nodes_own_first(stack, sp, root_begin, root_end, own_leaf, leaf_begin, leaf_end);
            while (sp > 0) {
                const std::int32_t node = stack[--sp];
                bool want = active && node_comp[node] != my_comp;
                if (want) {
                    // Device sqrt and pow are not correctly rounded, so the box bound is shrunk
                    // a few ulp to stay below every finalized point distance.
                    const Float box =
                        ops.finalize(ops.box_distance(x, lower + node * d, upper + node * d, d)) *
                        (Float(1) - 16 * std::numeric_limits<Float>::epsilon());
                    const Float lb =
                        sycl::fmax(sycl::fmax(my_core, node_min_core[node]), box * inv_alpha);
                    want = !(lb > best || (lb == best && min_original[node] > best_row) ||
                             lb > bound_ref.load());
                }
                if (!sycl::any_of_group(sg, want)) {
                    continue;
                }
                if (node < leaf_count) {
                    if (want && node != own_leaf) {
                        visit_leaf(node);
                    }
                }
                else {
                    push_nodes_own_first(stack,
                                         sp,
                                         first[node],
                                         last[node],
                                         own_leaf,
                                         leaf_begin,
                                         leaf_end);
                }
            }

            if (pos < n) {
                const std::int32_t row = original[pos];
                if (active && best_row != none) {
                    bk::atomic_global_min(comp_best_mrd + my_comp, best);
                    pt_best_mrd[row] = best;
                    pt_best_idx[row] = best_row;
                }
                else {
                    pt_best_mrd[row] = std::numeric_limits<Float>::max();
                    pt_best_idx[row] = -1;
                }
            }
        });
    });
}

#endif

} // namespace oneapi::dal::hdbscan::backend
