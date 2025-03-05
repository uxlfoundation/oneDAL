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

#pragma once

#include <iostream>

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/distance.hpp"
#include "oneapi/dal/backend/primitives/selection.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
std::int64_t propose_train_block(const sycl::queue& q, std::int64_t width);

template <typename Float>
std::int64_t propose_query_block(const sycl::queue& q, std::int64_t width);

template <typename Index>
sycl::event treat_indices(sycl::queue& q,
                          ndview<Index, 2>& indices,
                          std::int64_t start_index,
                          const event_vector& deps);

template <typename Float, typename Impl>
class callback_base {
public:
    using float_t = Float;
    using derived_t = Impl;

    sycl::event operator()(std::int64_t qb_id,
                           const ndview<std::int32_t, 2>& indices,
                           const ndview<Float, 2>& distances,
                           const event_vector& deps = {}) {
        return static_cast<Impl*>(this)->run(qb_id, indices, distances, deps);
    }
};

template <typename Float, typename Distance>
class search_temp_objects;

template <typename Float, typename Distance>
class search_temp_objects_deleter {
    using temp_t = search_temp_objects<Float, Distance>;
    using event_ptr_t = std::shared_ptr<sycl::event>;

public:
    search_temp_objects_deleter(event_ptr_t event);
    void operator()(temp_t* obj) const;

    event_ptr_t& get_last_event();

private:
    event_ptr_t last_event_;
};

template <typename Float, typename Distance>
std::shared_ptr<search_temp_objects<Float, Distance>> create_search_objects(
    sycl::queue& q,
    std::int64_t k,
    std::int64_t query_block,
    std::int64_t train_block,
    std::int64_t select_block,
    std::shared_ptr<sycl::event> dep);

template <typename Float, typename Distance, typename Impl, ndorder torder>
class search_engine_base {
protected:
    using temp_t = search_temp_objects<Float, Distance>;
    using temp_del_t = search_temp_objects_deleter<Float, Distance>;
    using temp_ptr_t = std::shared_ptr<temp_t>;
    using event_ptr_t = std::shared_ptr<sycl::event>;
    using selc_t = kselect_by_rows<Float>;
    using selc_ptr_t = std::shared_ptr<selc_t>;

    constexpr static inline std::int64_t selection_sub_blocks = 31;

public:
    search_engine_base(sycl::queue& queue, const ndview<Float, 2, torder>& train_data);

    search_engine_base(sycl::queue& queue,
                       const ndview<Float, 2, torder>& train_data,
                       std::int64_t train_block);

    search_engine_base(sycl::queue& queue,
                       const ndview<Float, 2, torder>& train_data,
                       std::int64_t train_block,
                       const Distance& distance_instance);

    template <ndorder qorder, typename CallbackImpl>
    sycl::event operator()(const ndview<Float, 2, qorder>& query_data,
                           CallbackImpl& callback,
                           std::int64_t k_neighbors = 1,
                           const event_vector& deps = {}) const {
        const auto query_block = propose_query_block<Float>(queue_, query_data.get_dimension(1));
        return this->operator()(query_data.callback, k_neighbors, query_block, deps);
    }

    template <ndorder qorder, typename CallbackImpl>
    sycl::event operator()(const ndview<Float, 2, qorder>& query_data,
                           CallbackImpl& callback,
                           std::int64_t query_block,
                           std::int64_t k_neighbors = 1,
                           const event_vector& deps = {}) const {
        auto last_event = std::make_shared<sycl::event>();
        const auto* const impl_ptr = static_cast<const Impl* const>(this);
        auto sel_objs = impl_ptr->create_selection_objects(query_block, k_neighbors);
        auto tmp_objs = impl_ptr->create_temporary_objects(query_block, k_neighbors, last_event);
        return this->
        operator()(query_data, callback, query_block, k_neighbors, tmp_objs, sel_objs, deps);
    }

    template <ndorder qorder, typename CallbackImpl>
    sycl::event operator()(const ndview<Float, 2, qorder>& query_data,
                           CallbackImpl& callback,
                           std::int64_t query_block,
                           std::int64_t k_neighbors,
                           temp_ptr_t temporary_obj,
                           selc_ptr_t selection_obj,
                           const event_vector& deps = {}) const {
        ONEDAL_PROFILER_TASK(search.query_loop, this->get_queue());
        using tmp_del_t = search_temp_objects_deleter<Float, Distance>;
        const auto* const impl_ptr = static_cast<const Impl* const>(this);
        const uniform_blocking query_blocking(query_data.get_dimension(0), query_block);
        auto& last_event = std::get_deleter<tmp_del_t>(temporary_obj)->get_last_event();
        for (std::int64_t qb_id = 0; qb_id < query_blocking.get_block_count(); ++qb_id) {
            const auto query_slice =
                query_data.get_row_slice(query_blocking.get_block_start_index(qb_id),
                                         query_blocking.get_block_end_index(qb_id));
            std::cerr << "do_search ";
            auto search_event = impl_ptr->do_search(query_slice,
                                                    k_neighbors,
                                                    temporary_obj,
                                                    selection_obj,
                                                    deps + *last_event);
            auto out_indices =
                get_indices(temporary_obj).get_row_slice(0, query_blocking.get_block_length(qb_id));
            auto out_distances = get_distances(temporary_obj)
                                     .get_row_slice(0, query_blocking.get_block_length(qb_id));
            std::cerr << "callback ";
            *last_event = callback(qb_id, out_indices, out_distances, { search_event });
        }
        return *last_event;
    }

    sycl::event reset(temp_ptr_t temp_obj, const event_vector& deps) const;

protected:
    template <ndorder qorder>
    sycl::event do_search(const ndview<Float, 2, qorder>& query,
                          std::int64_t k_neighbors,
                          temp_ptr_t temp_objs,
                          selc_ptr_t selt_objs,
                          const event_vector& deps) const;
    selc_ptr_t create_selection_objects(std::int64_t query_block, std::int64_t k_neighbors) const;
    temp_ptr_t create_temporary_objects(std::int64_t query_block,
                                        std::int64_t k_neighbors,
                                        event_ptr_t last_event) const;
    sycl::queue& get_queue() const;
    const Distance& get_distance_impl() const;
    const uniform_blocking& get_train_blocking() const;
    const uniform_blocking& get_selection_blocking() const;
    ndview<Float, 2, torder> get_train_block(std::int64_t idx) const;
    static ndview<Float, 2> get_distances(temp_ptr_t tmp_objs);
    static ndview<std::int32_t, 2> get_indices(temp_ptr_t tmp_objs);
    template <ndorder qorder>
    sycl::event distance(const ndview<Float, 2, qorder>& query,
                         const ndview<Float, 2, torder>& train,
                         ndview<Float, 2>& distances,
                         const event_vector& deps) const;
    sycl::event select_indexed(const ndview<std::int32_t, 2>& src,
                               ndview<std::int32_t, 2>& dst,
                               const event_vector& deps) const;

    sycl::queue& queue_;
    const Distance distance_instance_;
    const ndview<Float, 2, torder>& train_data_;
    const uniform_blocking train_blocking_;
    const uniform_blocking selection_blocking_;
};

template <typename Float, typename Distance, ndorder torder = ndorder::c>
class search_engine : public search_engine_base<Float,
                                                Distance,
                                                search_engine<Float, Distance, torder>,
                                                torder> {
    using base_t = search_engine_base<Float, Distance, search_engine, torder>;

public:
    search_engine(sycl::queue& queue,
                  const ndview<Float, 2, torder>& train_data,
                  std::int64_t train_block);

    search_engine(sycl::queue& queue,
                  const ndview<Float, 2, torder>& train_data,
                  std::int64_t train_block,
                  const Distance& distance_instance);

    template <ndorder qorder, typename CallbackImpl>
    sycl::event operator()(const ndview<Float, 2, qorder>& query_data,
                           CallbackImpl& callback,
                           std::int64_t query_block,
                           std::int64_t k_neighbors = 1,
                           const event_vector& deps = {}) const {
        return base_t::operator()(query_data, callback, query_block, k_neighbors, deps);
    }

    template <typename... Args>
    sycl::event operator()(Args&&... args) const {
        return base_t::operator()(std::forward<Args>(args)...);
    }
};

template <typename Float, ndorder torder>
class search_engine<Float, squared_l2_distance<Float>, torder>
        : public search_engine_base<Float,
                                    squared_l2_distance<Float>,
                                    search_engine<Float, squared_l2_distance<Float>, torder>,
                                    torder> {
    using base_t = search_engine_base<Float, squared_l2_distance<Float>, search_engine, torder>;
    using temp_t = search_temp_objects<Float, squared_l2_distance<Float>>;
    using temp_del_t = search_temp_objects_deleter<Float, squared_l2_distance<Float>>;
    using temp_ptr_t = std::shared_ptr<temp_t>;
    using event_ptr_t = std::shared_ptr<sycl::event>;
    using selc_t = kselect_by_rows<Float>;
    using selc_ptr_t = std::shared_ptr<selc_t>;

    friend class search_engine_base<Float, squared_l2_distance<Float>, search_engine, torder>;

public:
    search_engine(sycl::queue& queue,
                  const ndview<Float, 2, torder>& train_data,
                  std::int64_t train_block);

    search_engine(sycl::queue& queue,
                  const ndview<Float, 2, torder>& train_data,
                  std::int64_t train_block,
                  const squared_l2_distance<Float>& distance_instance);

    template <ndorder qorder, typename CallbackImpl>
    sycl::event operator()(const ndview<Float, 2, qorder>& query_data,
                           CallbackImpl& callback,
                           std::int64_t query_block,
                           std::int64_t k_neighbors = 1,
                           const event_vector& deps = {}) const {
        return base_t::operator()(query_data, callback, query_block, k_neighbors, deps);
    }

    template <typename... Args>
    sycl::event operator()(Args&&... args) const {
        return base_t::operator()(std::forward<Args>(args)...);
    }

protected:
    template <ndorder qorder>
    sycl::event do_search(const ndview<Float, 2, qorder>& query,
                          std::int64_t k_neighbors,
                          temp_ptr_t temp_objs,
                          selc_ptr_t select,
                          const event_vector& deps) const;
    template <ndorder qorder>
    sycl::event distance(const ndview<Float, 2, qorder>& query,
                         const ndview<Float, 2, torder>& train,
                         ndview<Float, 2>& distances,
                         const ndview<Float, 1>& query_norms,
                         const ndview<Float, 1>& train_norms,
                         const event_vector& deps) const;
};

template <typename Float, ndorder torder>
class search_engine<Float, cosine_distance<Float>, torder>
        : public search_engine_base<Float,
                                    cosine_distance<Float>,
                                    search_engine<Float, cosine_distance<Float>, torder>,
                                    torder> {
    using base_t = search_engine_base<Float, cosine_distance<Float>, search_engine, torder>;
    using temp_t = search_temp_objects<Float, cosine_distance<Float>>;
    using temp_del_t = search_temp_objects_deleter<Float, cosine_distance<Float>>;
    using temp_ptr_t = std::shared_ptr<temp_t>;
    using event_ptr_t = std::shared_ptr<sycl::event>;
    using selc_t = kselect_by_rows<Float>;
    using selc_ptr_t = std::shared_ptr<selc_t>;

    friend class search_engine_base<Float, cosine_distance<Float>, search_engine, torder>;

public:
    search_engine(sycl::queue& queue,
                  const ndview<Float, 2, torder>& train_data,
                  std::int64_t train_block);

    search_engine(sycl::queue& queue,
                  const ndview<Float, 2, torder>& train_data,
                  std::int64_t train_block,
                  const cosine_distance<Float>& distance_instance);

    template <ndorder qorder, typename CallbackImpl>
    sycl::event operator()(const ndview<Float, 2, qorder>& query_data,
                           CallbackImpl& callback,
                           std::int64_t query_block,
                           std::int64_t k_neighbors = 1,
                           const event_vector& deps = {}) const {
        return base_t::operator()(query_data, callback, query_block, k_neighbors, deps);
    }

    template <typename... Args>
    sycl::event operator()(Args&&... args) const {
        return base_t::operator()(std::forward<Args>(args)...);
    }

protected:
    template <ndorder qorder>
    sycl::event do_search(const ndview<Float, 2, qorder>& query,
                          std::int64_t k_neighbors,
                          temp_ptr_t temp_objs,
                          selc_ptr_t selt_objs,
                          const event_vector& deps) const;
    template <ndorder qorder>
    sycl::event distance(const ndview<Float, 2, qorder>& query,
                         const ndview<Float, 2, torder>& train,
                         ndview<Float, 2>& distances,
                         const ndview<Float, 1>& query_inv_norms,
                         const ndview<Float, 1>& train_inv_norms,
                         const event_vector& deps) const;
};

#endif

} // namespace oneapi::dal::backend::primitives
