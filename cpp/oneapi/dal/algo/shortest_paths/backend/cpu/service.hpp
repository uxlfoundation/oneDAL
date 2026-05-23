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

#include <atomic>

#include "oneapi/dal/algo/shortest_paths/common.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/detail/threading.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

namespace mode {
struct distances;
struct distances_predecessors;
} // namespace mode

template <typename EV, typename VT>
struct dist_pred {
    using distance_type = EV;
    using vertex_type = VT;
    dist_pred(const EV& dist_, const VT& pred_) : dist(dist_), pred(pred_) {}
    EV dist = 0;
    VT pred = 0;
};

template <class T1, class T2>
bool operator==(const dist_pred<T1, T2>& lhs, const dist_pred<T1, T2>& rhs) {
    return lhs.dist == rhs.dist && lhs.pred == rhs.pred;
}

template <class T1, class T2>
bool operator!=(const dist_pred<T1, T2>& lhs, const dist_pred<T1, T2>& rhs) {
    return !(lhs == rhs);
}

template <typename Cpu, typename Mode, typename Vertex, typename Value, typename AtomicT>
class data_to_relax_base {
public:
    static_assert(is_same_v<Mode, mode::distances>);
    using relaxing_data_type = Value;
    using atomic_type = std::atomic<AtomicT>;
    using atomic_value_type = AtomicT;
    using atomic_value_allocator_type = inner_alloc<atomic_type>;
    using vertex_type = Vertex;
    using mode_type = Mode;

    data_to_relax_base(std::int64_t vertex_count,
                       Vertex source,
                       relaxing_data_type max_dist,
                       byte_alloc_iface* alloc_ptr)
            : vertex_count(vertex_count),
              atomic_value_allocator(alloc_ptr) {
        relaxing_data = allocate(atomic_value_allocator, vertex_count);
        dal::detail::threader_for(vertex_count, 1024, [&](std::int64_t i) {
            new (relaxing_data + i) atomic_type(max_dist);
        });
        store_by_index(source, 0);
    }

    virtual ~data_to_relax_base() {
        deallocate(atomic_value_allocator, relaxing_data, vertex_count);
    }

    inline relaxing_data_type load_by_index(Vertex u) const {
        ONEDAL_ASSERT(static_cast<std::int64_t>(u) < vertex_count);
        return relaxing_data[u].load();
    }

    inline void store_by_index(Vertex u, relaxing_data_type value) {
        ONEDAL_ASSERT(static_cast<std::int64_t>(u) < vertex_count);
        relaxing_data[u].store(value);
    }

    inline bool compare_exchange_strong(Vertex u,
                                        relaxing_data_type& old_value,
                                        relaxing_data_type new_value) {
        ONEDAL_ASSERT(static_cast<std::int64_t>(u) < vertex_count);
        return relaxing_data[u].compare_exchange_strong(old_value, new_value);
    }

    inline relaxing_data_type operator[](Vertex u) const {
        return load_by_index(u);
    }

    inline auto get_distance(Vertex u) const {
        return load_by_index(u);
    }

    inline auto get_distances_ptr() const {
        return relaxing_data;
    }

private:
    data_to_relax_base(const data_to_relax_base&) = delete;
    data_to_relax_base(data_to_relax_base&&) = delete;
    const std::int64_t vertex_count;
    atomic_value_allocator_type atomic_value_allocator;
    atomic_type* relaxing_data;
};

template <typename Cpu, typename Mode, typename Vertex, typename Value, typename AtomicT>
class data_to_relax_int64_base {
public:
    static_assert(sizeof(Value) == sizeof(AtomicT));
    using relaxing_data_type = Value;
    using atomic_type = std::atomic<AtomicT>;
    using atomic_value_type = AtomicT;
    using atomic_value_allocator_type = inner_alloc<atomic_type>;
    using vertex_type = Vertex;
    using mode_type = Mode;

    template <typename EV = relaxing_data_type,
              typename M = Mode,
              std::enable_if_t<is_same_v<M, mode::distances>, bool> = true>
    data_to_relax_int64_base(std::int64_t vertex_count,
                             Vertex source,
                             relaxing_data_type max_dist,
                             byte_alloc_iface* alloc_ptr)
            : vertex_count(vertex_count),
              atomic_value_allocator(alloc_ptr) {
        relaxing_data = allocate(atomic_value_allocator, vertex_count);
        double* value_ptr = &max_dist;
        AtomicT value_int_representation = *reinterpret_cast<AtomicT*>(value_ptr);
        dal::detail::threader_for(vertex_count, 1024, [&](AtomicT i) {
            new (relaxing_data + i) atomic_type(value_int_representation);
        });
        store_by_index(source, 0);
    }

    template <typename EV,
              typename M = Mode,
              std::enable_if_t<is_same_v<M, mode::distances_predecessors>, bool> = true>
    data_to_relax_int64_base(std::int64_t vertex_count,
                             Vertex source,
                             EV max_dist,
                             byte_alloc_iface* alloc_ptr)
            : vertex_count(vertex_count),
              atomic_value_allocator(alloc_ptr) {
        relaxing_data = allocate(atomic_value_allocator, vertex_count);
        relaxing_data_type max_dp(max_dist, -1);
        AtomicT max_dp_int_representation = *reinterpret_cast<AtomicT*>(&max_dp);
        dal::detail::threader_for(vertex_count, 1024, [&](AtomicT i) {
            new (relaxing_data + i) atomic_type(max_dp_int_representation);
        });
        store_by_index(source, relaxing_data_type(0, -1));
    }

    virtual ~data_to_relax_int64_base() {
        deallocate(atomic_value_allocator, relaxing_data, vertex_count);
    }

    inline double load_by_index(Vertex u) const {
        ONEDAL_ASSERT(static_cast<std::int64_t>(u) < vertex_count);
        AtomicT a_int = relaxing_data[u].load();
        AtomicT* a_int_ptr = &a_int;
        return *reinterpret_cast<relaxing_data_type*>(a_int_ptr);
    }

    inline void store_by_index(Vertex u, relaxing_data_type value) {
        ONEDAL_ASSERT(static_cast<std::int64_t>(u) < vertex_count);
        relaxing_data_type* value_ptr = &value;
        AtomicT value_int_representation = *reinterpret_cast<AtomicT*>(value_ptr);
        relaxing_data[u].store(value_int_representation);
    }

    inline bool compare_exchange_strong(Vertex u,
                                        relaxing_data_type& old_value,
                                        relaxing_data_type new_value) {
        ONEDAL_ASSERT(static_cast<std::int64_t>(u) < vertex_count);
        relaxing_data_type* old_value_ptr = &old_value;
        relaxing_data_type* new_value_ptr = &new_value;
        AtomicT old_value_int_representation = *reinterpret_cast<AtomicT*>(old_value_ptr);
        AtomicT new_value_int_representation = *reinterpret_cast<AtomicT*>(new_value_ptr);
        return relaxing_data[u].compare_exchange_strong(old_value_int_representation,
                                                        new_value_int_representation);
    }

    inline relaxing_data_type operator[](Vertex u) const {
        return load_by_index(u);
    }

    template <typename M = Mode, std::enable_if_t<is_same_v<M, mode::distances>, bool> = true>
    inline auto get_distance(Vertex u) const {
        return load_by_index(u);
    }

    template <typename M = Mode,
              std::enable_if_t<is_same_v<M, mode::distances_predecessors>, bool> = true>
    inline auto get_distance(Vertex u) const {
        return load_by_index(u).dist;
    }

    template <typename M = Mode,
              std::enable_if_t<is_same_v<M, mode::distances_predecessors>, bool> = true>
    inline auto get_predecessor(Vertex u) const {
        return load_by_index(u).pred;
    }

    template <typename M = Mode, std::enable_if_t<is_same_v<M, mode::distances>, bool> = true>
    inline auto get_distances_ptr() const {
        return reinterpret_cast<relaxing_data_type*>(relaxing_data);
    }

private:
    data_to_relax_int64_base(const data_to_relax_int64_base&) = delete;
    data_to_relax_int64_base(data_to_relax_int64_base&&) = delete;
    const std::int64_t vertex_count;
    atomic_value_allocator_type atomic_value_allocator;
    atomic_type* relaxing_data;
};

//Specialisation for dist_pred atomic through lock, unlock mutex
template <typename Cpu, typename Vertex, typename Value>
class data_to_relax_base<Cpu,
                         mode::distances_predecessors,
                         Vertex,
                         dist_pred<Value, Vertex>,
                         dist_pred<Value, Vertex>> {
public:
    using vertex_type = Vertex;
    using distance_value_type = Value;
    using relaxing_data_type = dist_pred<distance_value_type, vertex_type>;
    using relaxing_data_allocator_type = inner_alloc<relaxing_data_type>;
    using mutex_type = oneapi::dal::detail::mutex;
    using mutex_allocator_type = inner_alloc<mutex_type>;
    using mode_type = mode::distances_predecessors;

    data_to_relax_base(std::int64_t vertex_count,
                       vertex_type source,
                       distance_value_type max_dist,
                       byte_alloc_iface* alloc_ptr)
            : vertex_count(vertex_count),
              relaxing_data_allocator(alloc_ptr),
              mutex_allocator(alloc_ptr) {
        relaxing_data = allocate(relaxing_data_allocator, vertex_count);
        mutexes = allocate(mutex_allocator, vertex_count);
        dal::detail::threader_for(vertex_count, 1024, [&](std::int64_t i) {
            new (relaxing_data + i) relaxing_data_type(max_dist, -1);
            new (mutexes + i) mutex_type();
        });
        relaxing_data[source] = relaxing_data_type(0, -1);
    }

    virtual ~data_to_relax_base() {
        deallocate(relaxing_data_allocator, relaxing_data, vertex_count);
        deallocate(mutex_allocator, mutexes, vertex_count);
    }

    inline relaxing_data_type load_by_index(vertex_type u) const {
        ONEDAL_ASSERT(static_cast<std::int64_t>(u) < vertex_count);
        const dal::detail::scoped_lock lock(mutexes[u]);
        relaxing_data_type relaxing_data_u = relaxing_data[u];
        return relaxing_data_u;
    }

    inline void store_by_index(vertex_type u, relaxing_data_type value) {
        ONEDAL_ASSERT(static_cast<std::int64_t>(u) < vertex_count);
        const dal::detail::scoped_lock lock(mutexes[u]);
        relaxing_data[u] = value;
    }

    inline bool compare_exchange_strong(vertex_type u,
                                        relaxing_data_type& old_value,
                                        relaxing_data_type new_value) {
        bool is_same = false;
        const dal::detail::scoped_lock lock(mutexes[u]);
        if (relaxing_data[u] == old_value) {
            is_same = true;
            relaxing_data[u] = new_value;
        }
        return is_same;
    }

    inline relaxing_data_type operator[](vertex_type u) const {
        return load_by_index(u);
    }

    inline auto get_distance(vertex_type u) const {
        return load_by_index(u).dist;
    }

    inline auto get_predecessor(vertex_type u) const {
        return load_by_index(u).pred;
    }

private:
    data_to_relax_base(const data_to_relax_base&) = delete;
    data_to_relax_base(data_to_relax_base&&) = delete;
    const std::int64_t vertex_count;
    relaxing_data_allocator_type relaxing_data_allocator;
    mutex_allocator_type mutex_allocator;
    relaxing_data_type* relaxing_data;
    mutex_type* mutexes;
};

template <typename Cpu, typename Mode, typename Vertex, typename Value>
class data_to_relax : public data_to_relax_base<Cpu, Mode, Vertex, Value, Value> {
    using data_to_relax_base<Cpu, Mode, Vertex, Value, Value>::data_to_relax_base;
    data_to_relax(const data_to_relax&) = delete;
    data_to_relax(data_to_relax&&) = delete;
};

template <typename Cpu, typename Mode, typename Vertex>
class data_to_relax<Cpu, Mode, Vertex, double>
        : public data_to_relax_int64_base<Cpu, Mode, Vertex, double, std::int64_t> {
    using data_to_relax_int64_base<Cpu, Mode, Vertex, double, std::int64_t>::
        data_to_relax_int64_base;
    data_to_relax(const data_to_relax&) = delete;
    data_to_relax(data_to_relax&&) = delete;
};

template <typename Cpu, typename Vertex, typename Value>
class data_to_relax<Cpu, mode::distances_predecessors, Vertex, dist_pred<Value, Vertex>>
        : public data_to_relax_base<Cpu,
                                    mode::distances_predecessors,
                                    Vertex,
                                    dist_pred<Value, Vertex>,
                                    dist_pred<Value, Vertex>> {
    using data_to_relax_base<Cpu,
                             mode::distances_predecessors,
                             Vertex,
                             dist_pred<Value, Vertex>,
                             dist_pred<Value, Vertex>>::data_to_relax_base;
    data_to_relax(const data_to_relax&) = delete;
    data_to_relax(data_to_relax&&) = delete;
};

template <typename Cpu>
class data_to_relax<Cpu, mode::distances_predecessors, std::int32_t, std::int32_t>
        : public data_to_relax_int64_base<Cpu,
                                          mode::distances_predecessors,
                                          std::int32_t,
                                          dist_pred<std::int32_t, std::int32_t>,
                                          std::int64_t> {
    using data_to_relax_int64_base<Cpu,
                                   mode::distances_predecessors,
                                   std::int32_t,
                                   dist_pred<std::int32_t, std::int32_t>,
                                   std::int64_t>::data_to_relax_int64_base;
    data_to_relax(const data_to_relax&) = delete;
    data_to_relax(data_to_relax&&) = delete;
};

template <typename Mode, typename Vertex, typename Value>
struct relaxing_data_type {};

template <typename Vertex, typename Value>
struct relaxing_data_type<mode::distances, Vertex, Value> {
    using value_type = Value;
};

template <typename Vertex, typename Value>
struct relaxing_data_type<mode::distances_predecessors, Vertex, Value> {
    using value_type = dist_pred<Value, Vertex>;
};

} // namespace oneapi::dal::preview::shortest_paths::backend
