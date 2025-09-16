/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/knn/backend/distance_impl.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"

#include "oneapi/dal/backend/primitives/distributed.hpp"
#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/regression.hpp"
#include "oneapi/dal/backend/primitives/search.hpp"
#include "oneapi/dal/backend/primitives/selection.hpp"
#include "oneapi/dal/backend/primitives/voting.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/split_table.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::knn::backend {

template <typename Float>
inline std::int64_t propose_distributed_block_size(const sycl::queue& queue, std::int64_t fcount) {
    return 16 * pr::propose_train_block<Float>(queue, fcount);
}

template <typename Float, typename Task>
class knn_callback_distr {
    using dst_t = Float;
    using idx_t = std::int32_t;
    using res_t = response_t<Task, Float>;
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;
    using selc_t = pr::kselect_by_rows<Float>;

    using uniform_voting_t = std::unique_ptr<pr::uniform_voting<res_t>>;
    using distance_voting_t = std::unique_ptr<pr::distance_voting<dst_t>>;
    using uniform_regression_t = std::unique_ptr<pr::uniform_regression<res_t>>;
    using distance_regression_t = std::unique_ptr<pr::distance_regression<dst_t>>;

public:
    knn_callback_distr(sycl::queue& q,
                       comm_t c,
                       result_option_id results,
                       std::int64_t query_block,
                       std::int64_t query_length,
                       std::int64_t k_neighbors)
            : queue_(q),
              comm_(c),
              result_options_(results),
              query_block_(query_block),
              query_length_(query_length),
              k_neighbors_(k_neighbors) {}

    auto& set_euclidean_distance(bool is_euclidean_distance) {
        this->compute_sqrt_ = is_euclidean_distance;
        return *this;
    }

    auto& set_train_responses(const pr::ndview<res_t, 1>& train_responses) {
        if (result_options_.test(result_options::responses)) {
            this->train_responses_ = train_responses;
        }
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_uniform_voting(uniform_voting_t voting) {
        this->uniform_voting_ = std::move(voting);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_distance_voting(distance_voting_t voting) {
        this->distance_voting_ = std::move(voting);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_regression_t<T>>
    auto& set_uniform_regression(uniform_regression_t regression) {
        this->uniform_regression_ = std::move(regression);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_regression_t<T>>
    auto& set_distance_regression(distance_regression_t regression) {
        this->distance_regression_ = std::move(regression);
        return *this;
    }

    auto& set_responses(const pr::ndview<res_t, 1>& responses) {
        if (result_options_.test(result_options::responses)) {
            ONEDAL_ASSERT(responses.get_count() == query_length_);
            this->responses_ = responses;
        }
        return *this;
    }

    auto& set_part_responses(const pr::ndview<res_t, 2>& part_responses) {
        if (result_options_.test(result_options::responses)) {
            ONEDAL_ASSERT(part_responses.get_dimension(0) == query_length_);
            ONEDAL_ASSERT(part_responses.get_dimension(1) == 2 * k_neighbors_);
            this->part_responses_ = part_responses;
        }
        return *this;
    }

    auto& set_intermediate_responses(const pr::ndview<res_t, 2>& intermediate_responses) {
        if (result_options_.test(result_options::responses)) {
            ONEDAL_ASSERT(intermediate_responses.get_dimension(0) == query_length_);
            ONEDAL_ASSERT(intermediate_responses.get_dimension(1) == k_neighbors_);
            this->intermediate_responses_ = intermediate_responses;
        }
        return *this;
    }

    auto& set_indices(const pr::ndview<idx_t, 2>& indices) {
        if (result_options_.test(result_options::indices)) {
            ONEDAL_ASSERT(indices.get_dimension(0) == query_length_);
            ONEDAL_ASSERT(indices.get_dimension(1) == k_neighbors_);
            this->indices_ = indices;
        }
        return *this;
    }

    auto& set_part_indices(const pr::ndview<idx_t, 2>& part_indices) {
        if (result_options_.test(result_options::indices)) {
            ONEDAL_ASSERT(part_indices.get_dimension(0) == query_length_);
            ONEDAL_ASSERT(part_indices.get_dimension(1) == 2 * k_neighbors_);
            this->part_indices_ = part_indices;
        }
        return *this;
    }

    auto& set_distances(const pr::ndview<dst_t, 2>& distances) {
        if (result_options_.test(result_options::distances)) {
            ONEDAL_ASSERT(distances.get_dimension(0) == query_length_);
            ONEDAL_ASSERT(distances.get_dimension(1) == k_neighbors_);
            this->distances_ = distances;
        }
        return *this;
    }

    auto& set_part_distances(const pr::ndview<dst_t, 2>& part_distances) {
        if (result_options_.test(result_options::distances)) {
            ONEDAL_ASSERT(part_distances.get_dimension(0) == query_length_);
            ONEDAL_ASSERT(part_distances.get_dimension(1) == 2 * k_neighbors_);
            this->part_distances_ = part_distances;
        }
        return *this;
    }

    sycl::event reset_dists_inds(const bk::event_vector& deps) {
        constexpr Float default_dst_value = de::limits<Float>::max();
        constexpr idx_t default_idx_value = -1;
        auto out_dsts = pr::fill(this->queue_, this->distances_, default_dst_value, deps);
        auto out_idcs = pr::fill(this->queue_, this->indices_, default_idx_value, { out_dsts });
        return out_idcs;
    }

    auto& set_global_index_offset(std::int64_t offset) {
        global_index_offset_ = offset;
        return *this;
    }

    auto& set_last_iteration(bool last_iteration) {
        this->last_iteration_ = last_iteration;
        return *this;
    }

    sycl::event operator()(std::int64_t qb_id,
                           pr::ndview<idx_t, 2>& inp_indices,
                           pr::ndview<Float, 2>& inp_distances,
                           const bk::event_vector& deps = {}) {
        ONEDAL_PROFILER_TASK(query_loop.callback, queue_);
        sycl::event copy_actual_dist_event, copy_current_dist_event, copy_actual_indc_event,
            copy_current_indc_event, copy_actual_resp_event, copy_current_resp_event;
        const auto& bounds = this->block_bounds(qb_id);
        const auto& [first, last] = bounds;
        const auto len = last - first;
        ONEDAL_ASSERT(last > first);
        ONEDAL_ASSERT(inp_indices.get_dimension(0) == len);
        ONEDAL_ASSERT(inp_indices.get_dimension(1) == k_neighbors_);
        ONEDAL_ASSERT(inp_distances.get_dimension(0) == len);
        ONEDAL_ASSERT(inp_distances.get_dimension(1) == k_neighbors_);

        pr::ndview<res_t, 2> current_min_resp_dest;
        pr::ndview<res_t, 2> min_resp_dest;
        if (result_options_.test(result_options::responses)) {
            current_min_resp_dest = part_responses_.get_col_slice(k_neighbors_, 2 * k_neighbors_)
                                        .get_row_slice(first, last);
            copy_current_resp_event = pr::select_indexed(queue_,
                                                         inp_indices,
                                                         train_responses_,
                                                         current_min_resp_dest,
                                                         deps);
            min_resp_dest = intermediate_responses_.get_row_slice(first, last);
        }

        const pr::ndshape<2> typical_blocking(last - first, 2 * k_neighbors_);
        auto select = selc_t(queue_, typical_blocking, k_neighbors_);

        auto min_dist_dest = distances_.get_row_slice(first, last);
        auto min_indc_dest = indices_.get_row_slice(first, last);

        // add global offset value to input indices
        ONEDAL_ASSERT(global_index_offset_ != -1);
        deps.push_back(copy_current_resp_event);
        auto treat_event = pr::treat_indices(queue_, inp_indices, global_index_offset_, deps);

        auto actual_min_dist_copy_dest =
            part_distances_.get_col_slice(0, k_neighbors_).get_row_slice(first, last);
        auto current_min_dist_dest = part_distances_.get_col_slice(k_neighbors_, 2 * k_neighbors_)
                                         .get_row_slice(first, last);
        copy_actual_dist_event =
            pr::copy(queue_, actual_min_dist_copy_dest, min_dist_dest, { treat_event });
        copy_current_dist_event =
            pr::copy(queue_, current_min_dist_dest, inp_distances, { treat_event });

        auto actual_min_indc_copy_dest =
            part_indices_.get_col_slice(0, k_neighbors_).get_row_slice(first, last);
        auto current_min_indc_dest =
            part_indices_.get_col_slice(k_neighbors_, 2 * k_neighbors_).get_row_slice(first, last);
        copy_actual_indc_event =
            pr::copy(queue_, actual_min_indc_copy_dest, min_indc_dest, { treat_event });
        copy_current_indc_event =
            pr::copy(queue_, current_min_indc_dest, inp_indices, { treat_event });
        if (result_options_.test(result_options::responses)) {
            auto actual_min_resp_copy_dest =
                part_responses_.get_col_slice(0, k_neighbors_).get_row_slice(first, last);
            copy_actual_resp_event =
                pr::copy(queue_, actual_min_resp_copy_dest, min_resp_dest, { treat_event });
        }

        sycl::event select_event, select_resp_event;
        {
            ONEDAL_PROFILER_TASK(query_loop.selection, queue_);
            auto kselect_block = part_distances_.get_row_slice(first, last);
            select_event = select(queue_,
                                  kselect_block,
                                  k_neighbors_,
                                  min_dist_dest,
                                  min_indc_dest,
                                  { copy_actual_dist_event,
                                    copy_current_dist_event,
                                    copy_actual_indc_event,
                                    copy_current_indc_event,
                                    copy_actual_resp_event,
                                    copy_current_resp_event });
        }
        if (result_options_.test(result_options::responses)) {
            select_resp_event = select_indexed(queue_,
                                               min_indc_dest,
                                               part_responses_.get_row_slice(first, last),
                                               min_resp_dest,
                                               { select_event });
        }
        sycl::event select_indc_event = select_indexed(queue_,
                                                       min_indc_dest,
                                                       part_indices_.get_row_slice(first, last),
                                                       min_indc_dest,
                                                       { select_event, select_resp_event });
        if (last_iteration_) {
            sycl::event copy_sqrt_event;
            if (this->compute_sqrt_) {
                copy_sqrt_event =
                    copy_with_sqrt(queue_, min_dist_dest, min_dist_dest, { select_indc_event });
            }
            if (result_options_.test(result_options::responses)) {
                return this->output_responses(bounds,
                                              indices_,
                                              distances_,
                                              { select_indc_event, copy_sqrt_event });
            }
            if (this->compute_sqrt_) {
                return copy_sqrt_event;
            }
        }
        return select_indc_event;
    }

protected:
    auto get_blocking() const {
        return bk::uniform_blocking(query_length_, query_block_);
    }

    auto block_bounds(std::int64_t qb_id) const {
        const auto blocking = this->get_blocking();
        const auto first = blocking.get_block_start_index(qb_id);
        const auto last = blocking.get_block_end_index(qb_id);
        return std::make_pair(first, last);
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    sycl::event do_ucls(const std::pair<idx_t, idx_t>& bnds,
                        const pr::ndview<res_t, 2>& tmp_rps,
                        const bk::event_vector& deps = {}) {
        ONEDAL_ASSERT(tmp_rps.has_data());
        ONEDAL_ASSERT(bool(this->uniform_voting_));
        ONEDAL_ASSERT(this->result_options_.test(result_options::responses));

        const auto& [first, last] = bnds;
        ONEDAL_ASSERT(last > first);

        auto out_rps = this->responses_.get_slice(first, last);
        ONEDAL_ASSERT((last - first) == out_rps.get_count());
        return (*(this->uniform_voting_))(tmp_rps, out_rps, deps);
    }

    template <typename T = Task, typename = detail::enable_if_regression_t<T>>
    sycl::event do_ureg(const std::pair<idx_t, idx_t>& bnds,
                        const pr::ndview<res_t, 2>& tmp_rps,
                        const bk::event_vector& deps = {}) {
        ONEDAL_ASSERT(tmp_rps.has_data());
        ONEDAL_ASSERT(bool(this->uniform_regression_));
        ONEDAL_ASSERT(this->result_options_.test(result_options::responses));

        const auto& [first, last] = bnds;
        ONEDAL_ASSERT(last > first);

        auto out_rps = this->responses_.get_slice(first, last);
        ONEDAL_ASSERT((last - first) == out_rps.get_count());
        return (*(this->uniform_regression_))(tmp_rps, out_rps, deps);
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    sycl::event do_dcls(const std::pair<idx_t, idx_t>& bnds,
                        const pr::ndview<res_t, 2>& tmp_rps,
                        pr::ndview<dst_t, 2>& inp_dts,
                        const bk::event_vector& deps = {}) {
        ONEDAL_ASSERT(inp_dts.has_data());
        ONEDAL_ASSERT(tmp_rps.has_mutable_data());
        ONEDAL_ASSERT(bool(this->distance_voting_));
        ONEDAL_ASSERT(this->result_options_.test(result_options::responses));

        const auto& [first, last] = bnds;
        ONEDAL_ASSERT(last > first);

        auto out_rps = this->responses_.get_slice(first, last);
        ONEDAL_ASSERT((last - first) == out_rps.get_count());
        return (*(this->distance_voting_))(tmp_rps, inp_dts, out_rps, deps);
    }

    template <typename T = Task, typename = detail::enable_if_regression_t<T>>
    sycl::event do_dreg(const std::pair<idx_t, idx_t>& bnds,
                        const pr::ndview<res_t, 2>& tmp_rps,
                        pr::ndview<dst_t, 2>& inp_dts,
                        const bk::event_vector& deps = {}) {
        ONEDAL_ASSERT(inp_dts.has_data());
        ONEDAL_ASSERT(tmp_rps.has_mutable_data());
        ONEDAL_ASSERT(bool(this->distance_regression_));
        ONEDAL_ASSERT(this->result_options_.test(result_options::responses));

        const auto& [first, last] = bnds;
        ONEDAL_ASSERT(last > first);

        auto out_rps = this->responses_.get_slice(first, last);
        ONEDAL_ASSERT((last - first) == out_rps.get_count());
        return (*(this->distance_regression_))(tmp_rps, inp_dts, out_rps, deps);
    }

    sycl::event output_responses(const std::pair<idx_t, idx_t>& bnds,
                                 const pr::ndview<idx_t, 2>& inp_ids,
                                 pr::ndview<dst_t, 2>& inp_dts,
                                 const bk::event_vector& deps = {}) {
        ONEDAL_ASSERT(inp_ids.has_data());
        ONEDAL_ASSERT(this->result_options_.test(result_options::responses));

        const auto& [first, last] = bnds;
        ONEDAL_ASSERT(last > first);

        auto tmp_rps = this->intermediate_responses_.get_row_slice(first, last);

        if constexpr (std::is_same_v<Task, task::classification>) {
            const auto ucls = bool(this->uniform_voting_);
            if (ucls)
                return this->do_ucls(bnds, tmp_rps, deps);

            const auto dcls = bool(this->distance_voting_);
            if (dcls)
                return this->do_dcls(bnds, tmp_rps, inp_dts, deps);
        }

        if constexpr (std::is_same_v<Task, task::regression>) {
            const auto ureg = bool(this->uniform_regression_);
            if (ureg)
                return this->do_ureg(bnds, tmp_rps, deps);

            const auto dreg = bool(this->distance_regression_);
            if (dreg)
                return this->do_dreg(bnds, tmp_rps, inp_dts, deps);
        }

        ONEDAL_ASSERT(false);
        return sycl::event();
    }

private:
    sycl::queue& queue_;
    comm_t comm_;
    const result_option_id result_options_;
    const std::int64_t query_block_, query_length_, k_neighbors_;
    pr::ndview<res_t, 1> train_responses_;
    pr::ndview<res_t, 1> responses_;
    pr::ndview<res_t, 2> part_responses_;
    pr::ndview<res_t, 2> intermediate_responses_;
    pr::ndview<Float, 2> distances_;
    pr::ndview<Float, 2> part_distances_;
    pr::ndview<idx_t, 2> indices_;
    pr::ndview<idx_t, 2> part_indices_;
    int64_t global_index_offset_ = -1;
    uniform_voting_t uniform_voting_;
    distance_voting_t distance_voting_;
    uniform_regression_t uniform_regression_;
    distance_regression_t distance_regression_;
    bool compute_sqrt_ = false;
    bool last_iteration_ = false;
};

template <typename Task, typename Float, pr::ndorder qorder, typename RespT>
sycl::event bf_kernel_distr(sycl::queue& queue,
                            bk::communicator<spmd::device_memory_access::usm> comm,
                            const descriptor_t<Task>& desc,
                            const table& train,
                            const pr::ndview<Float, 2, qorder>& query,
                            const table& tresps,
                            pr::ndview<Float, 2>& distances,
                            pr::ndview<Float, 2>& part_distances,
                            pr::ndview<idx_t, 2>& indices,
                            pr::ndview<idx_t, 2>& part_indices,
                            pr::ndview<RespT, 1>& qresps,
                            pr::ndview<RespT, 2>& part_responses,
                            pr::ndview<RespT, 2>& intermediate_responses,
                            const bk::event_vector& deps = {}) {
    using res_t = response_t<Task, Float>;
    constexpr auto torder = pr::ndorder::c;

    // Input arrays test section
    ONEDAL_ASSERT(train.has_data());
    ONEDAL_ASSERT(query.has_data());
    const auto tcount = train.get_row_count();
    const auto qcount = query.get_dimension(0);
    const auto fcount = train.get_column_count();
    const auto kcount = desc.get_neighbor_count();
    ONEDAL_ASSERT(fcount == query.get_dimension(1));
    // Output arrays test section
    const auto& ropts = desc.get_result_options();
    if (ropts.test(result_options::responses)) {
        ONEDAL_ASSERT(tresps.has_data());
        ONEDAL_ASSERT(qresps.has_mutable_data());
        ONEDAL_ASSERT(tcount == tresps.get_row_count());
        ONEDAL_ASSERT(qcount == qresps.get_count());
        ONEDAL_ASSERT(qcount == part_responses.get_dimension(0));
        ONEDAL_ASSERT(2 * kcount == part_responses.get_dimension(1));
        ONEDAL_ASSERT(qcount == intermediate_responses.get_dimension(0));
        ONEDAL_ASSERT(kcount == intermediate_responses.get_dimension(1));
    }
    if (ropts.test(result_options::indices)) {
        ONEDAL_ASSERT(indices.has_mutable_data());
        ONEDAL_ASSERT(qcount == indices.get_dimension(0));
        ONEDAL_ASSERT(kcount == indices.get_dimension(1));
        ONEDAL_ASSERT(qcount == part_indices.get_dimension(0));
        ONEDAL_ASSERT(2 * kcount == part_indices.get_dimension(1));
    }
    if (ropts.test(result_options::distances)) {
        ONEDAL_ASSERT(distances.has_mutable_data());
        ONEDAL_ASSERT(qcount == distances.get_dimension(0));
        ONEDAL_ASSERT(kcount == distances.get_dimension(1));
        ONEDAL_ASSERT(qcount == part_distances.get_dimension(0));
        ONEDAL_ASSERT(2 * kcount == part_distances.get_dimension(1));
    }
    const auto ccount = desc.get_class_count();

    auto rank_count = comm.get_rank_count();
    auto block_size = propose_distributed_block_size<Float>(queue, fcount);
    auto node_sample_counts = pr::ndarray<std::int64_t, 1>::empty({ rank_count });

    comm.allgather(tcount, node_sample_counts.flatten()).wait();

    // TODO: implement max/min for ndarray
    std::int64_t max_tcount = 0;
    for (std::int64_t index = 0; index < node_sample_counts.get_count(); ++index) {
        max_tcount = std::max(node_sample_counts.at(index), max_tcount);
    }
    block_size = std::min(max_tcount, block_size);

    auto current_rank = comm.get_rank();
    auto prev_node = (current_rank - 1 + rank_count) % rank_count;
    auto next_node = (current_rank + 1) % rank_count;
    ONEDAL_ASSERT(prev_node >= 0);

    auto [nodes, boundaries] = pr::get_boundary_indices(node_sample_counts, block_size);
    std::int64_t block_count = nodes.size();
    std::int64_t bounds_size = boundaries.size();
    ONEDAL_ASSERT(block_count + 1 == bounds_size);

    auto train_block_queue = pr::split_table<Float>(queue, train, block_size);
    auto tresps_queue = pr::split_table<res_t>(queue, tresps, block_size);
    std::int64_t tbq_size = train_block_queue.size();
    std::int64_t trq_size = tresps_queue.size();
    ONEDAL_ASSERT(tbq_size <= block_count);
    ONEDAL_ASSERT(trq_size == tbq_size);

    const auto qbcount = pr::propose_query_block<Float>(queue, fcount);
    const auto tbcount = pr::propose_train_block<Float>(queue, fcount);

    knn_callback_distr<Float, Task> callback(queue, comm, ropts, qbcount, qcount, kcount);

    callback.set_distances(distances);
    callback.set_part_distances(part_distances);
    callback.set_responses(qresps);
    callback.set_part_responses(part_responses);
    callback.set_intermediate_responses(intermediate_responses);
    callback.set_indices(indices);
    callback.set_part_indices(part_indices);

    auto next_event = callback.reset_dists_inds(deps);

    if constexpr (std::is_same_v<Task, task::classification>) {
        if (desc.get_result_options().test(result_options::responses) &&
            (desc.get_voting_mode() == voting_mode::uniform)) {
            callback.set_uniform_voting(std::move(pr::make_uniform_voting(queue, qbcount, kcount)));
        }

        if (desc.get_result_options().test(result_options::responses) &&
            (desc.get_voting_mode() == voting_mode::distance)) {
            callback.set_distance_voting(
                std::move(pr::make_distance_voting<Float>(queue, qbcount, ccount)));
        }
    }

    if constexpr (std::is_same_v<Task, task::regression>) {
        if (desc.get_result_options().test(result_options::responses) &&
            (desc.get_voting_mode() == voting_mode::uniform)) {
            callback.set_uniform_regression(
                std::move(pr::make_uniform_regression<res_t>(queue, qbcount, kcount)));
        }

        if (desc.get_result_options().test(result_options::responses) &&
            (desc.get_voting_mode() == voting_mode::distance)) {
            callback.set_distance_regression(
                std::move(pr::make_distance_regression<Float>(queue, qbcount, kcount)));
        }
    }

    auto distance_impl = detail::get_distance_impl(desc);
    if (!distance_impl) {
        throw internal_error{ de::error_messages::unknown_distance_type() };
    }

    using daal_distance_t = decltype(distance_impl->get_daal_distance_type());
    const bool is_minkowski_distance =
        distance_impl->get_daal_distance_type() == daal_distance_t::minkowski;
    const bool is_chebyshev_distance =
        distance_impl->get_daal_distance_type() == daal_distance_t::chebyshev;
    const bool is_cosine_distance =
        distance_impl->get_daal_distance_type() == daal_distance_t::cosine;
    const bool is_euclidean_distance =
        is_minkowski_distance && (distance_impl->get_degree() == 2.0);
    ONEDAL_ASSERT(is_minkowski_distance ^ is_chebyshev_distance ^ is_cosine_distance);

    const auto it = std::find(nodes.begin(), nodes.end(), current_rank);
    auto relative_block_offset = std::distance(nodes.begin(), it);
    ONEDAL_ASSERT(it != nodes.end());

    pr::ndarray<res_t, 2> current_tresps;

    for (std::int64_t relative_block_idx = 0; relative_block_idx < block_count;
         ++relative_block_idx) {
        auto current_block = train_block_queue.front();
        train_block_queue.pop_front();
        ONEDAL_ASSERT(current_block.has_data());

        auto absolute_block_idx = (relative_block_idx + relative_block_offset) % block_count;
        ONEDAL_ASSERT(absolute_block_idx + 1 < bounds_size);
        auto actual_rows_in_block =
            boundaries.at(absolute_block_idx + 1) - boundaries.at(absolute_block_idx);

        auto sc = current_block.get_dimension(0);
        ONEDAL_ASSERT(sc >= actual_rows_in_block);
        auto curr_k = std::min(actual_rows_in_block, kcount);
        auto actual_current_block = current_block.get_row_slice(0, actual_rows_in_block);

        if (ropts.test(result_options::responses)) {
            current_tresps = tresps_queue.front();
            ONEDAL_ASSERT(current_tresps.has_data());
            auto current_tresps_1d = pr::ndview<res_t, 1>::wrap(current_tresps.get_data(),
                                                                { current_tresps.get_count() });
            tresps_queue.pop_front();
            auto actual_current_tresps = current_tresps_1d.get_slice(0, actual_rows_in_block);
            callback.set_train_responses(actual_current_tresps);
        }

        callback.set_global_index_offset(boundaries.at(absolute_block_idx));

        if (relative_block_idx == block_count - 1) {
            callback.set_last_iteration(true);
        }
        if (is_cosine_distance) {
            using dst_t = pr::cosine_distance<Float>;
            using search_t = pr::search_engine<Float, dst_t, torder>;

            const dst_t dist{ queue };
            const search_t search{ queue, actual_current_block, tbcount, dist };
            next_event = search(query, callback, qbcount, curr_k, { next_event });
        }

        if (is_chebyshev_distance) {
            using dst_t = pr::chebyshev_distance<Float>;
            using search_t = pr::search_engine<Float, dst_t, torder>;

            const dst_t dist{ queue };
            const search_t search{ queue, actual_current_block, tbcount, dist };
            next_event = search(query, callback, qbcount, curr_k, { next_event });
        }

        if (is_euclidean_distance) {
            using dst_t = pr::squared_l2_distance<Float>;
            using search_t = pr::search_engine<Float, dst_t, torder>;

            callback.set_euclidean_distance(true);

            const dst_t dist{ queue };
            const search_t search{ queue, actual_current_block, tbcount, dist };
            next_event = search(query, callback, qbcount, curr_k, { next_event });
        }
        else if (is_minkowski_distance) {
            using met_t = pr::lp_metric<Float>;
            using dst_t = pr::lp_distance<Float>;
            using search_t = pr::search_engine<Float, dst_t, torder>;

            const dst_t dist{ queue, met_t(distance_impl->get_degree()) };
            const search_t search{ queue, actual_current_block, tbcount, dist };
            next_event = search(query, callback, qbcount, curr_k, { next_event });
        }

        if (relative_block_idx < block_count - 1) {
            ONEDAL_PROFILER_TASK(distributed_loop.sendrecv_replace, queue);
            auto send_count = current_block.get_count();
            ONEDAL_ASSERT(send_count >= 0);
            ONEDAL_ASSERT(send_count <= de::limits<int>::max());
            auto send_train_block = array<Float>::wrap(queue,
                                                       current_block.get_mutable_data(),
                                                       send_count,
                                                       { next_event });
            comm.sendrecv_replace(send_train_block, prev_node, next_node).wait();
            train_block_queue.emplace_back(current_block);
            if (ropts.test(result_options::responses)) {
                auto send_resps_block = array<res_t>::wrap(queue,
                                                           current_tresps.get_mutable_data(),
                                                           current_tresps.get_count(),
                                                           { next_event });
                comm.sendrecv_replace(send_resps_block, prev_node, next_node).wait();
                tresps_queue.emplace_back(current_tresps);
            }
        }
    }

    return next_event;
}
#define INSTANTIATE_DISTR(T, I, R, F, A)                                                    \
    template sycl::event bf_kernel_distr(sycl::queue&,                                      \
                                         bk::communicator<spmd::device_memory_access::usm>, \
                                         const descriptor_t<T>&,                            \
                                         const table&,                                      \
                                         const pr::ndview<F, 2, A>&,                        \
                                         const table&,                                      \
                                         pr::ndview<F, 2>&,                                 \
                                         pr::ndview<F, 2>&,                                 \
                                         pr::ndview<I, 2>&,                                 \
                                         pr::ndview<I, 2>&,                                 \
                                         pr::ndview<R, 1>&,                                 \
                                         pr::ndview<R, 2>&,                                 \
                                         pr::ndview<R, 2>&,                                 \
                                         const bk::event_vector&);

#define INSTANTIATE_A_DISTR(T, I, R, F)           \
    INSTANTIATE_DISTR(T, I, R, F, pr::ndorder::c) \
    INSTANTIATE_DISTR(T, I, R, F, pr::ndorder::f)

#define INSTANTIATE_T_DISTR(I, F)                                 \
    INSTANTIATE_A_DISTR(task::classification, I, std::int32_t, F) \
    INSTANTIATE_A_DISTR(task::regression, I, float, F)            \
    INSTANTIATE_A_DISTR(task::search, I, int, F)

#define INSTANTIATE_F_DISTR(I)    \
    INSTANTIATE_T_DISTR(I, float) \
    INSTANTIATE_T_DISTR(I, double)

INSTANTIATE_F_DISTR(std::int32_t)

} // namespace oneapi::dal::knn::backend
