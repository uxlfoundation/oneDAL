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

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/inner_alloc.hpp"
#include "oneapi/dal/table/column_accessor.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template <typename Cpu>
struct state {
    std::int64_t* core;
    std::int64_t core_length;

    state(inner_alloc a);
    state(std::int64_t length, inner_alloc a);
    state(state<Cpu>& parent_state, std::int64_t new_element, inner_alloc a);
    state(state<Cpu>* parent_state, std::int64_t new_element, inner_alloc a);
    void clear();
    ~state();

private:
    inner_alloc allocator;
};

template <typename Cpu>
class solution {
public:
    solution(const std::int64_t length, inner_alloc a);
    virtual ~solution();

    solution(solution<Cpu>&& sol);
    solution<Cpu>& operator=(solution<Cpu>&& sol);

    std::int64_t get_solution_count() const;
    void add(std::int64_t** state_core);
    void append(solution<Cpu>&& _solution);
    oneapi::dal::homogen_table export_as_table(std::int64_t* sorted_pattern_vertex_array,
                                               std::int64_t max_match_count) const;

private:
    inner_alloc allocator;

public:
    std::int64_t** data;

    std::int64_t solution_core_length;
    std::int64_t solution_count;
    std::int64_t max_solution_count;

    void increase_solutions_size();
    void delete_data();
};

template <typename Cpu>
state<Cpu>::state(inner_alloc a) : allocator(a) {
    core = nullptr;
    core_length = 0;
}

template <typename Cpu>
state<Cpu>::state(std::int64_t length, inner_alloc a) : allocator(a) {
    core_length = length;
    core = allocator.allocate<std::int64_t>(core_length);
}

template <typename Cpu>
state<Cpu>::state(state<Cpu>& parent_state, std::int64_t new_element, inner_alloc a)
        : state(parent_state.core_length + 1, a) {
    for (std::int64_t i = 0; i < parent_state.core_length; i++) {
        /* TODO replace by memset */
        core[i] = parent_state.core[i];
    }
    core[parent_state.core_length] = new_element;
}

template <typename Cpu>
state<Cpu>::state(state<Cpu>* parent_state, std::int64_t new_element, inner_alloc a)
        : state(parent_state->core_length + 1, a) {
    for (std::int64_t i = 0; i < parent_state->core_length; i++) {
        /* TODO replace by memset */
        core[i] = parent_state->core[i];
    }
    core[parent_state->core_length] = new_element;
}

template <typename Cpu>
void state<Cpu>::clear() {
    allocator.deallocate(core, core_length);
    core = nullptr;
    core_length = 0;
}

template <typename Cpu>
state<Cpu>::~state() {
    this->clear();
}

template <typename Cpu>
solution<Cpu>::solution(const std::int64_t length, inner_alloc a)
        : allocator(a),
          solution_count(0),
          max_solution_count(100) {
    data = allocator.allocate<std::int64_t*>(max_solution_count);
    for (std::int64_t i = 0; i < max_solution_count; i++) {
        data[i] = nullptr;
    }
    solution_core_length = length;
}

template <typename Cpu>
solution<Cpu>::~solution() {
    delete_data();
}

template <typename Cpu>
void solution<Cpu>::delete_data() {
    if (data != nullptr) {
        for (std::int64_t i = 0; i < max_solution_count; i++) {
            if (data[i] != nullptr) {
                allocator.deallocate(data[i], 0);
                data[i] = nullptr;
            }
        }
        allocator.deallocate(data, max_solution_count);
        data = nullptr;
    }
}

template <typename Cpu>
solution<Cpu>::solution(solution<Cpu>&& sol)
        : allocator(sol.allocator),
          data(sol.data),
          solution_core_length(sol.solution_core_length),
          solution_count(sol.solution_count),
          max_solution_count(sol.max_solution_count) {
    sol.data = nullptr;
    sol.solution_count = 0;
    sol.solution_core_length = 0;
    sol.max_solution_count = 100;
}

template <typename Cpu>
solution<Cpu>& solution<Cpu>::operator=(solution<Cpu>&& sol) {
    if (&sol == this) {
        return *this;
    }
    delete_data();
    solution_count = sol.solution_count;
    max_solution_count = sol.max_solution_count;
    solution_core_length = sol.solution_core_length;
    data = sol.data;

    sol.data = nullptr;
    sol.solution_count = 0;
    sol.solution_core_length = 0;
    sol.max_solution_count = 100;
    return *this;
}

template <typename Cpu>
std::int64_t solution<Cpu>::get_solution_count() const {
    return solution_count;
}

template <typename Cpu>
void solution<Cpu>::add(std::int64_t** state_core) {
    if (state_core != nullptr && *state_core != nullptr) {
        if (solution_count >= max_solution_count) {
            increase_solutions_size();
        }
        data[solution_count] = *state_core;
        *state_core = nullptr;
        solution_count++;
    }
}

template <typename Cpu>
void solution<Cpu>::append(solution<Cpu>&& _solution) {
    for (std::int64_t i = 0; i < _solution.get_solution_count(); i++) {
        add(&_solution.data[i]);
    }

    if (_solution.get_solution_count() > 0) {
        solution_core_length = _solution.solution_core_length;
    }

    if (_solution.data != nullptr) {
        allocator.deallocate(_solution.data, _solution.max_solution_count);
        _solution.data = nullptr;
    }

    _solution.solution_count = 0;
    _solution.solution_core_length = 0;
}

template <typename Cpu>
void solution<Cpu>::increase_solutions_size() {
    const std::int64_t new_max_solution_count = 2 * max_solution_count;
    std::int64_t** const new_data = allocator.allocate<std::int64_t*>(new_max_solution_count);
    for (std::int64_t i = 0; i < max_solution_count; i++) {
        new_data[i] = data[i];
    }
    for (std::int64_t i = max_solution_count; i < new_max_solution_count; i++) {
        new_data[i] = nullptr;
    }
    if (data != nullptr) {
        allocator.deallocate(data, max_solution_count);
    }
    max_solution_count = new_max_solution_count;
    data = new_data;
}

template <typename T>
inline T min(const T a, const T b) {
    return (a >= b) ? b : a;
}

template <typename Cpu>
oneapi::dal::homogen_table solution<Cpu>::export_as_table(std::int64_t* sorted_pattern_vertices,
                                                          std::int64_t max_match_count) const {
    if (solution_count == 0)
        return dal::homogen_table();

    auto export_solution_count = solution_count;
    if (max_match_count != 0) {
        export_solution_count = std::min(max_match_count, solution_count);
    }

    const auto begin = sorted_pattern_vertices;
    const auto end = &sorted_pattern_vertices[solution_core_length];

    auto mapping_array = dal::array<std::int64_t>::empty(solution_core_length);
    const auto mapping = mapping_array.get_mutable_data();
    for (std::int64_t j = 0; j < solution_core_length; ++j) {
        const auto p = std::find(begin, end, j);
        ONEDAL_ASSERT(p != end, "Index not found");
        mapping[j] = p - begin;
    }

    auto arr_solution = dal::array<int>::empty(solution_core_length * export_solution_count);
    const auto arr = arr_solution.get_mutable_data();

    constexpr std::int64_t block_size = 64;
    const std::int64_t block_count = (export_solution_count - 1 + block_size) / block_size;
    dal::detail::threader_for(block_count, 1, [&](int index) {
        const std::int64_t first = index * block_size;
        const std::int64_t last = min(first + block_size, export_solution_count);
        for (auto i = first; i != last; ++i) {
            for (std::int64_t j = 0; j < solution_core_length; ++j) {
                arr[i * solution_core_length + j] = data[i][mapping[j]];
            }
        }
    });

    return dal::detail::homogen_table_builder{}
        .reset(arr_solution, export_solution_count, solution_core_length)
        .build();
}
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
