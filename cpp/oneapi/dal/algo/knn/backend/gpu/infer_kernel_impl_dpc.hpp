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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/regression.hpp"
#include "oneapi/dal/backend/primitives/search.hpp"
#include "oneapi/dal/backend/primitives/selection.hpp"
#include "oneapi/dal/backend/primitives/voting.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::knn::backend {

template <typename T1, typename T2>
inline sycl::event copy_with_sqrt(sycl::queue& q,
                                  const pr::ndview<T2, 2>& src,
                                  pr::ndview<T1, 2>& dst,
                                  const bk::event_vector& deps = {}) {
    static_assert(de::is_floating_point<T1>());
    static_assert(de::is_floating_point<T2>());
    ONEDAL_ASSERT(src.has_data());
    ONEDAL_ASSERT(dst.has_mutable_data());
    const pr::ndshape<2> dst_shape = dst.get_shape();
    ONEDAL_ASSERT(dst_shape == src.get_shape());
    T1* const dst_ptr = dst.get_mutable_data();
    const T2* const src_ptr = src.get_data();
    const auto dst_stride = dst.get_leading_stride();
    const auto src_stride = src.get_leading_stride();
    const auto cp_range = bk::make_range_2d(dst_shape[0], dst_shape[1]);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(cp_range, [=](sycl::id<2> idx) {
            T1& dst_ref = *(dst_ptr + idx[0] * dst_stride + idx[1]);
            const T2& val_ref = *(src_ptr + idx[0] * src_stride + idx[1]);
            dst_ref = sycl::sqrt(val_ref);
        });
    });
}

template <typename Float, typename Task>
class knn_callback {
    using dst_t = Float;
    using idx_t = std::int32_t;
    using res_t = response_t<Task, Float>;
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;

    using uniform_voting_t = std::unique_ptr<pr::uniform_voting<res_t>>;
    using distance_voting_t = std::unique_ptr<pr::distance_voting<dst_t>>;
    using uniform_regression_t = std::unique_ptr<pr::uniform_regression<res_t>>;
    using distance_regression_t = std::unique_ptr<pr::distance_regression<dst_t>>;

public:
    knn_callback(sycl::queue& q,
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
              k_neighbors_(k_neighbors) {
        if (result_options_.test(result_options::responses)) {
            this->temp_resp_ = pr::ndarray<res_t, 2>::empty(q,
                                                            { query_block, k_neighbors },
                                                            sycl::usm::alloc::device);
        }
    }

    auto& set_euclidean_distance(bool is_euclidean_distance) {
        this->compute_sqrt_ = is_euclidean_distance;
        return *this;
    }

    auto& set_inp_responses(const pr::ndview<res_t, 1>& inp_responses) {
        if (result_options_.test(result_options::responses)) {
            this->inp_responses_ = inp_responses;
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

    auto& set_indices(const pr::ndview<idx_t, 2>& indices) {
        if (result_options_.test(result_options::indices)) {
            ONEDAL_ASSERT(indices.get_dimension(0) == query_length_);
            ONEDAL_ASSERT(indices.get_dimension(1) == k_neighbors_);
            this->indices_ = indices;
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

    // Note: `inp_distances` can be modified if
    // metric is Euclidean
    sycl::event operator()(std::int64_t qb_id,
                           pr::ndview<idx_t, 2>& inp_indices,
                           pr::ndview<Float, 2>& inp_distances,
                           const bk::event_vector& deps = {}) {
        ONEDAL_PROFILER_TASK(query_loop.callback, queue_);
        sycl::event copy_indices, copy_distances, comp_responses;

        const auto bounds = this->block_bounds(qb_id);

        if (result_options_.test(result_options::indices)) {
            copy_indices = this->output_indices(bounds, inp_indices, deps);
        }

        if (result_options_.test(result_options::distances)) {
            copy_distances = this->output_distances(bounds, inp_distances, deps);
        }

        if (result_options_.test(result_options::responses)) {
            using namespace bk;
            const auto ndeps = deps + copy_indices + copy_distances;
            comp_responses = this->output_responses(bounds, inp_indices, inp_distances, ndeps);
        }

        sycl::event::wait_and_throw({ copy_indices, copy_distances, comp_responses });
        return sycl::event();
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

    sycl::event output_distances(const std::pair<idx_t, idx_t>& bnds,
                                 const pr::ndview<dst_t, 2>& inp_dts,
                                 const bk::event_vector& deps = {}) {
        ONEDAL_ASSERT(inp_dts.has_data());
        ONEDAL_ASSERT(this->result_options_.test(result_options::distances));

        const auto& [first, last] = bnds;
        ONEDAL_ASSERT(last > first);
        auto& queue = this->queue_;

        auto out_dts = this->distances_.get_row_slice(first, last);
        ONEDAL_ASSERT((last - first) == inp_dts.get_dimension(0));
        ONEDAL_ASSERT((last - first) == out_dts.get_dimension(0));

        // Generally !csqrt is more probable
        const bool& csqrt = this->compute_sqrt_;
        if (!csqrt)
            return pr::copy(queue, out_dts, inp_dts, deps);
        else
            return copy_with_sqrt(queue, inp_dts, out_dts, deps);
    }

    sycl::event output_indices(const std::pair<idx_t, idx_t>& bnds,
                               const pr::ndview<idx_t, 2>& inp_ids,
                               const bk::event_vector& deps = {}) {
        ONEDAL_ASSERT(inp_ids.has_data());
        ONEDAL_ASSERT(this->result_options_.test(result_options::indices));

        const auto& [first, last] = bnds;
        ONEDAL_ASSERT(last > first);
        auto& queue = this->queue_;

        auto out_ids = this->indices_.get_row_slice(first, last);
        ONEDAL_ASSERT((last - first) == inp_ids.get_dimension(0));
        ONEDAL_ASSERT((last - first) == out_ids.get_dimension(0));
        ONEDAL_ASSERT(inp_ids.get_shape() == out_ids.get_shape());

        return pr::copy(queue, out_ids, inp_ids, deps);
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

        bk::event_vector ndeps{ deps.cbegin(), deps.cend() };
        if (this->compute_sqrt_) {
            auto sqrt_event = copy_with_sqrt(this->queue_, inp_dts, inp_dts, deps);
            ndeps.push_back(sqrt_event);
        }

        auto out_rps = this->responses_.get_slice(first, last);
        ONEDAL_ASSERT((last - first) == out_rps.get_count());
        return (*(this->distance_voting_))(tmp_rps, inp_dts, out_rps, ndeps);
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

        bk::event_vector ndeps{ deps.cbegin(), deps.cend() };
        if (this->compute_sqrt_) {
            auto sqrt_event = copy_with_sqrt(this->queue_, inp_dts, inp_dts, deps);
            ndeps.push_back(sqrt_event);
        }

        auto out_rps = this->responses_.get_slice(first, last);
        ONEDAL_ASSERT((last - first) == out_rps.get_count());
        return (*(this->distance_regression_))(tmp_rps, inp_dts, out_rps, ndeps);
    }

    sycl::event output_responses(const std::pair<idx_t, idx_t>& bnds,
                                 const pr::ndview<idx_t, 2>& inp_ids,
                                 pr::ndview<dst_t, 2>& inp_dts,
                                 const bk::event_vector& deps = {}) {
        ONEDAL_ASSERT(inp_ids.has_data());
        ONEDAL_ASSERT(this->result_options_.test(result_options::responses));

        const auto& [first, last] = bnds;
        const auto len = last - first;
        ONEDAL_ASSERT(last > first);
        auto& queue = this->queue_;

        auto tmp_rps = this->temp_resp_.get_row_slice(0, len);

        const auto& inp_rps = this->inp_responses_;
        auto s_evt = pr::select_indexed(queue, inp_ids, inp_rps, tmp_rps, deps);

        if constexpr (std::is_same_v<Task, task::classification>) {
            const auto ucls = bool(this->uniform_voting_);
            if (ucls)
                return this->do_ucls(bnds, tmp_rps, { s_evt });

            const auto dcls = bool(this->distance_voting_);
            if (dcls)
                return this->do_dcls(bnds, tmp_rps, inp_dts, { s_evt });
        }

        if constexpr (std::is_same_v<Task, task::regression>) {
            const auto ureg = bool(this->uniform_regression_);
            if (ureg)
                return this->do_ureg(bnds, tmp_rps, { s_evt });

            const auto dreg = bool(this->distance_regression_);
            if (dreg)
                return this->do_dreg(bnds, tmp_rps, inp_dts, { s_evt });
        }

        ONEDAL_ASSERT(false);
        return sycl::event();
    }

private:
    sycl::queue& queue_;
    comm_t comm_;
    const result_option_id result_options_;
    const std::int64_t query_block_, query_length_, k_neighbors_;
    pr::ndview<res_t, 1> inp_responses_;
    pr::ndarray<res_t, 2> temp_resp_;
    pr::ndview<res_t, 1> responses_;
    pr::ndview<Float, 2> distances_;
    pr::ndview<idx_t, 2> indices_;
    uniform_voting_t uniform_voting_;
    distance_voting_t distance_voting_;
    uniform_regression_t uniform_regression_;
    distance_regression_t distance_regression_;
    bool compute_sqrt_ = false;
};

template <typename Task, typename Float, pr::ndorder torder, pr::ndorder qorder, typename RespT>
sycl::event bf_kernel(sycl::queue& queue,
                      bk::communicator<spmd::device_memory_access::usm> comm,
                      const descriptor_t<Task>& desc,
                      const pr::ndview<Float, 2, torder>& train,
                      const pr::ndview<Float, 2, qorder>& query,
                      const pr::ndview<RespT, 1>& tresps,
                      pr::ndview<Float, 2>& distances,
                      pr::ndview<idx_t, 2>& indices,
                      pr::ndview<RespT, 1>& qresps,
                      const bk::event_vector& deps = {}) {
    using res_t = response_t<Task, Float>;

    // Input arrays test section
    ONEDAL_ASSERT(train.has_data());
    ONEDAL_ASSERT(query.has_data());
    [[maybe_unused]] const auto tcount = train.get_dimension(0);
    const auto qcount = query.get_dimension(0);
    const auto fcount = train.get_dimension(1);
    ONEDAL_ASSERT(fcount == query.get_dimension(1));

    // Output arrays test section
    const auto& ropts = desc.get_result_options();
    if (ropts.test(result_options::responses)) {
        ONEDAL_ASSERT(tresps.has_data());
        ONEDAL_ASSERT(qresps.has_mutable_data());
        ONEDAL_ASSERT(tcount == tresps.get_count());
        ONEDAL_ASSERT(qcount == qresps.get_count());
    }
    const auto kcount = desc.get_neighbor_count();
    if (ropts.test(result_options::indices)) {
        ONEDAL_ASSERT(indices.has_mutable_data());
        ONEDAL_ASSERT(qcount == indices.get_dimension(0));
        ONEDAL_ASSERT(kcount == indices.get_dimension(1));
    }
    if (ropts.test(result_options::distances)) {
        ONEDAL_ASSERT(distances.has_mutable_data());
        ONEDAL_ASSERT(qcount == distances.get_dimension(0));
        ONEDAL_ASSERT(kcount == distances.get_dimension(1));
    }
    const auto ccount = desc.get_class_count();

    // Callback preparation
    const auto qbcount = pr::propose_query_block<Float>(queue, fcount);
    const auto tbcount = pr::propose_train_block<Float>(queue, fcount);

    knn_callback<Float, Task> callback(queue, comm, ropts, qbcount, qcount, kcount);

    callback.set_inp_responses(tresps);
    callback.set_distances(distances);
    callback.set_responses(qresps);
    callback.set_indices(indices);

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

    // Actual search
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

    sycl::event search_event;

    if (is_cosine_distance) {
        using dst_t = pr::cosine_distance<Float>;
        using search_t = pr::search_engine<Float, dst_t, torder>;

        const dst_t dist{ queue };
        const search_t search{ queue, train, tbcount, dist };
        search_event = search(query, callback, qbcount, kcount);
    }

    if (is_chebyshev_distance) {
        using dst_t = pr::chebyshev_distance<Float>;
        using search_t = pr::search_engine<Float, dst_t, torder>;

        const dst_t dist{ queue };
        const search_t search{ queue, train, tbcount, dist };
        search_event = search(query, callback, qbcount, kcount);
    }

    if (is_euclidean_distance) {
        using dst_t = pr::squared_l2_distance<Float>;
        using search_t = pr::search_engine<Float, dst_t, torder>;

        callback.set_euclidean_distance(true);

        const dst_t dist{ queue };
        const search_t search{ queue, train, tbcount, dist };
        search_event = search(query, callback, qbcount, kcount);
    }
    else if (is_minkowski_distance) {
        using met_t = pr::lp_metric<Float>;
        using dst_t = pr::lp_distance<Float>;
        using search_t = pr::search_engine<Float, dst_t, torder>;

        const dst_t dist{ queue, met_t(distance_impl->get_degree()) };
        const search_t search{ queue, train, tbcount, dist };
        search_event = search(query, callback, qbcount, kcount);
    }

    return search_event;
}

#define INSTANTIATE(T, I, R, F, A, B)                                                 \
    template sycl::event bf_kernel(sycl::queue&,                                      \
                                   bk::communicator<spmd::device_memory_access::usm>, \
                                   const descriptor_t<T>&,                            \
                                   const pr::ndview<F, 2, A>&,                        \
                                   const pr::ndview<F, 2, B>&,                        \
                                   const pr::ndview<R, 1>&,                           \
                                   pr::ndview<F, 2>&,                                 \
                                   pr::ndview<I, 2>&,                                 \
                                   pr::ndview<R, 1>&,                                 \
                                   const bk::event_vector&);

#define INSTANTIATE_B(T, I, R, F, A)           \
    INSTANTIATE(T, I, R, F, A, pr::ndorder::c) \
    INSTANTIATE(T, I, R, F, A, pr::ndorder::f)

#define INSTANTIATE_A(T, I, R, F)             \
    INSTANTIATE_B(T, I, R, F, pr::ndorder::c) \
    INSTANTIATE_B(T, I, R, F, pr::ndorder::f)

#define INSTANTIATE_T(I, F)                                 \
    INSTANTIATE_A(task::classification, I, std::int32_t, F) \
    INSTANTIATE_A(task::regression, I, float, F)            \
    INSTANTIATE_A(task::search, I, int, F)

#define INSTANTIATE_F(I)    \
    INSTANTIATE_T(I, float) \
    INSTANTIATE_T(I, double)

INSTANTIATE_F(std::int32_t)

} // namespace oneapi::dal::knn::backend
