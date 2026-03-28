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

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/sorter.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/stack.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/bit_vector.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"

#include <typeinfo> // for typeid

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template <typename Cpu>
class engine_bundle;

template <typename Cpu>
class matching_engine {
public:
    matching_engine(const graph<Cpu>* ppattern,
                    const graph<Cpu>* ptarget,
                    const std::int64_t* psorted_pattern_vertex,
                    const std::int64_t* ppredecessor,
                    const edge_direction* pdirection,
                    sconsistent_conditions<Cpu> const* pcconditions,
                    kind isomorphism_kind,
                    inner_alloc alloc);
    matching_engine(const matching_engine& _matching_engine,
                    stack<Cpu>& _local_stack,
                    inner_alloc alloc);
    virtual ~matching_engine();

    template <bool is_parallel>
    void run_and_wait(global_stack<Cpu>& gstack,
                      std::int64_t& busy_engine_count,
                      std::int64_t& current_match_count,
                      std::int64_t target_match_count,
                      bool main_engine);
    solution<Cpu> get_solution();
    std::int64_t get_match_count() const;

    std::int64_t state_exploration();
    std::int64_t state_exploration_bit(bool check_solution = true);
    std::int64_t state_exploration_list(bool check_solution = true);

    template <bool is_parallel>
    bool check_if_max_match_count_reached(std::int64_t& cumulative_match_count,
                                          std::int64_t delta,
                                          std::int64_t target_match_count);

    std::int64_t first_states_generator(dfs_stack<Cpu>& stack);

    void push_into_stack(const std::int64_t vertex_id);
    bool match_vertex(const std::int64_t pattern_vertex, const std::int64_t target_vertex) const;

    inner_alloc allocator;
    const graph<Cpu>* pattern;
    const graph<Cpu>* target;
    const std::int64_t* sorted_pattern_vertex;
    const std::int64_t* predecessor;
    const edge_direction* direction;
    const sconsistent_conditions<Cpu>* pconsistent_conditions;

    std::int64_t solution_length;
    bit_vector<Cpu> vertex_candidates;

    std::int64_t temporary_list_size;
    std::int64_t* temporary_list;

    std::uint8_t* pstart_byte;

    stack<Cpu> local_stack;
    dfs_stack<Cpu> hlocal_stack;
    solution<Cpu> engine_solutions;

    kind isomorphism_kind;

    std::int64_t extract_candidates(bool check_solution);
    bool check_vertex_candidate(bool check_solution, std::int64_t candidate);
    template <bool is_parallel>
    void set_not_busy(bool& is_busy_engine, std::int64_t& busy_engine_count);
};

template <bool is_parallel>
void increment_shared_value(std::int64_t& value, std::int64_t delta = 1) {
    if constexpr (is_parallel) {
        dal::detail::atomic_increment(value, delta);
    }
    else {
        value += delta;
    }
}

template <bool is_parallel>
void decrement_shared_value(std::int64_t& value, std::int64_t delta = 1) {
    if constexpr (is_parallel) {
        dal::detail::atomic_decrement(value, delta);
    }
    else {
        value -= delta;
    }
}

template <bool is_parallel>
std::int64_t load_shared_value(std::int64_t& value) {
    if constexpr (is_parallel) {
        return dal::detail::atomic_load(value);
    }
    else {
        return value;
    }
}

template <typename Cpu>
class engine_bundle {
public:
    stack<Cpu> exploration_stack;
    engine_bundle(const graph<Cpu>* ppattern,
                  const graph<Cpu>* ptarget,
                  const std::int64_t* psorted_pattern_vertex,
                  const std::int64_t* ppredecessor,
                  const edge_direction* pdirection,
                  sconsistent_conditions<Cpu> const* pcconditions,
                  float* ppattern_vertex_probability,
                  kind isomorphism_kind,
                  inner_alloc alloc);
    virtual ~engine_bundle();
    solution<Cpu> run(std::int64_t max_match_count);

    inner_alloc allocator;
    const graph<Cpu>* pattern;
    const graph<Cpu>* target;
    const std::int64_t* sorted_pattern_vertex;
    const std::int64_t* predecessor;
    const edge_direction* direction;
    const sconsistent_conditions<Cpu>* pconsistent_conditions;
    const float* pattern_vertex_probability;
    kind isomorphism_kind;

    solution<Cpu> combine_solutions(matching_engine<Cpu>* engine_array,
                                    std::uint64_t array_size,
                                    std::int64_t max_match_count);
};

template <typename Cpu>
matching_engine<Cpu>::~matching_engine() {
    pattern = nullptr;
    target = nullptr;
    sorted_pattern_vertex = nullptr;
    predecessor = nullptr;
    direction = nullptr;
    pconsistent_conditions = nullptr;

    allocator.deallocate(temporary_list, temporary_list_size);
    temporary_list = nullptr;
    temporary_list_size = 0;
}

template <typename Cpu>
matching_engine<Cpu>::matching_engine(const graph<Cpu>* ppattern,
                                      const graph<Cpu>* ptarget,
                                      const std::int64_t* psorted_pattern_vertex,
                                      const std::int64_t* ppredecessor,
                                      const edge_direction* pdirection,
                                      sconsistent_conditions<Cpu> const* pcconditions,
                                      kind isomor_kind,
                                      inner_alloc alloc)
        : allocator(alloc),
          vertex_candidates(bit_vector<Cpu>::bit_vector_size(ptarget->get_vertex_count()), alloc),
          local_stack(alloc),
          hlocal_stack(alloc),
          engine_solutions(ppattern->get_vertex_count(), alloc),
          isomorphism_kind(isomor_kind) {
    pattern = ppattern;
    target = ptarget;
    sorted_pattern_vertex = psorted_pattern_vertex;
    predecessor = ppredecessor;
    direction = pdirection;
    pconsistent_conditions = pcconditions;

    solution_length = pattern->get_vertex_count();

    std::int64_t target_vertex_count = target->get_vertex_count();

    pstart_byte = vertex_candidates.get_vector_pointer();

    std::int64_t max_neighbours_size = target->get_max_degree();
    std::int64_t max_degree = target->get_max_degree();
    if (max_neighbours_size < max_degree) {
        max_neighbours_size = max_degree;
    }

    hlocal_stack.init(solution_length - 1, target_vertex_count);

    if (target->bit_representation) {
        temporary_list = nullptr;
    }
    else {
        temporary_list_size = max_neighbours_size;
        temporary_list = allocator.allocate<std::int64_t>(temporary_list_size);
    }
}

template <typename Cpu>
matching_engine<Cpu>::matching_engine(const matching_engine& _matching_engine,
                                      stack<Cpu>& _local_stack,
                                      inner_alloc alloc)
        : matching_engine(_matching_engine.pattern,
                          _matching_engine.target,
                          _matching_engine.sorted_pattern_vertex,
                          _matching_engine.predecessor,
                          _matching_engine.direction,
                          _matching_engine.pconsistent_conditions,
                          _matching_engine.isomorphism_kind,
                          alloc) {
    local_stack = std::move(_local_stack);
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::state_exploration_bit(bool check_solution) {
    std::uint64_t current_level_index = hlocal_stack.get_current_level_index();
    std::int64_t divider = pconsistent_conditions[current_level_index].divider;

    if (isomorphism_kind != kind::non_induced) {
        ONEDAL_IVDEP
        for (std::int64_t j = 0; j < divider; j++) {
            or_equal<Cpu>(vertex_candidates.get_vector_pointer(),
                          target->p_edges_bit[hlocal_stack.top(
                              pconsistent_conditions[current_level_index].array[j])],
                          vertex_candidates.size());
        }
    }

    ~vertex_candidates;

    ONEDAL_IVDEP
    for (std::int64_t j = current_level_index; j >= divider; j--) { //j > divider - 1
        and_equal<Cpu>(vertex_candidates.get_vector_pointer(),
                       target->p_edges_bit[hlocal_stack.top(
                           pconsistent_conditions[current_level_index].array[j])],
                       vertex_candidates.size());
    }

    for (std::uint64_t i = 0; i <= current_level_index; i++) {
        vertex_candidates.get_vector_pointer()[bit_vector<Cpu>::byte(hlocal_stack.top(i))] &=
            ~bit_vector<Cpu>::bit(hlocal_stack.top(i));
    }

    return extract_candidates(check_solution);
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::extract_candidates(bool check_solution) {
    std::int64_t feasible_result_count = 0;
    std::int64_t size_in_dword = vertex_candidates.size() >> 3;
    std::uint64_t* ptr;
    std::int32_t popcnt;
    for (std::int64_t i = 0; i < size_in_dword; i++) {
        ptr = (std::uint64_t*)(pstart_byte + (i << 3));
        popcnt = ONEDAL_popcnt64<Cpu>(*ptr);
        ONEDAL_ASSERT(popcnt <= 64);
        for (std::int64_t j = 0; j < popcnt; j++) {
            std::int64_t candidate = 63 - ONEDAL_lzcnt_u64<Cpu>(*ptr);
            (*ptr) ^= (std::uint64_t)1 << candidate;
            candidate += (i << 6);
            feasible_result_count += check_vertex_candidate(check_solution, candidate);
        }
    }
    for (std::int64_t i = (size_in_dword << 3); i < vertex_candidates.size(); i++) {
        while (pstart_byte[i] > 0) {
            std::int64_t candidate = bit_vector<Cpu>::power_of_two(pstart_byte[i]);
            pstart_byte[i] ^= (1 << candidate);
            candidate += (i << 3);
            feasible_result_count += check_vertex_candidate(check_solution, candidate);
        }
    }

    hlocal_stack.update();

    return feasible_result_count;
}

template <typename Cpu>
bool matching_engine<Cpu>::check_vertex_candidate(bool check_solution, std::int64_t candidate) {
    std::uint64_t solution_length_unsigned = solution_length;
    if (match_vertex(sorted_pattern_vertex[hlocal_stack.get_current_level()], candidate)) {
        if (check_solution && hlocal_stack.get_current_level() + 1 == solution_length_unsigned) {
            std::int64_t* solution_core = allocator.allocate<std::int64_t>(solution_length);
            if (solution_core != nullptr) {
                hlocal_stack.fill_solution(solution_core, candidate);
                engine_solutions.add(&solution_core);
                return true;
            }
        }
        else {
            hlocal_stack.push_into_next_level(candidate);
        }
    }
    return false;
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::state_exploration_list(bool check_solution) {
    std::uint64_t current_level_index = hlocal_stack.get_current_level_index();
    std::int64_t divider = pconsistent_conditions[current_level_index].divider;

    if (isomorphism_kind != kind::non_induced) {
        ONEDAL_IVDEP
        for (std::int64_t j = 0; j < divider; j++) {
            or_equal<Cpu>(vertex_candidates.get_vector_pointer(),
                          target->p_edges_list[hlocal_stack.top(
                              pconsistent_conditions[current_level_index].array[j])],
                          target->p_degree[hlocal_stack.top(
                              pconsistent_conditions[current_level_index].array[j])]);
        }
    }

    ~vertex_candidates;

    ONEDAL_IVDEP
    for (std::int64_t j = current_level_index; j >= divider; j--) { //j > divider - 1
        and_equal<Cpu>(
            vertex_candidates.get_vector_pointer(),
            target->p_edges_list[hlocal_stack.top(
                pconsistent_conditions[current_level_index].array[j])],
            vertex_candidates.size(),
            target
                ->p_degree[hlocal_stack.top(pconsistent_conditions[current_level_index].array[j])],
            temporary_list);
    }

    for (std::uint64_t i = 0; i <= current_level_index; i++) {
        vertex_candidates.get_vector_pointer()[bit_vector<Cpu>::byte(hlocal_stack.top(i))] &=
            ~bit_vector<Cpu>::bit(hlocal_stack.top(i));
    }
    return extract_candidates(check_solution);
}

template <typename Cpu>
void matching_engine<Cpu>::push_into_stack(const std::int64_t vertex_id) {
    hlocal_stack.push_into_current_level(vertex_id);
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::first_states_generator(dfs_stack<Cpu>& stack) {
    std::int64_t degree = pattern->get_vertex_degree(sorted_pattern_vertex[0]);
    for (std::int64_t i = 0; i < target->get_vertex_count(); i++) {
        if (degree <= target->get_vertex_degree(i) &&
            pattern->get_vertex_attribute(sorted_pattern_vertex[0]) ==
                target->get_vertex_attribute(i)) {
            stack.push_into_current_level(i);
        }
    }

    return stack.get_current_level_fill_size();
}

template <typename Cpu>
bool matching_engine<Cpu>::match_vertex(const std::int64_t pattern_vertex,
                                        const std::int64_t target_vertex) const {
    if (target_vertex >= target->get_vertex_count())
        return false;
    ONEDAL_ASSERT(pattern_vertex < pattern->get_vertex_count());
    ONEDAL_ASSERT(target_vertex < target->get_vertex_count());
    return pattern->get_vertex_degree(pattern_vertex) <= target->get_vertex_degree(target_vertex) &&
           pattern->get_vertex_attribute(pattern_vertex) ==
               target->get_vertex_attribute(target_vertex);
}

template <typename Cpu>
solution<Cpu> matching_engine<Cpu>::get_solution() {
    return std::move(engine_solutions);
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::get_match_count() const {
    return engine_solutions.get_solution_count();
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::state_exploration() {
    if (target->bit_representation) {
        return state_exploration_bit();
    }
    else {
        return state_exploration_list();
    }
}

template <typename Cpu>
template <bool is_parallel>
bool matching_engine<Cpu>::check_if_max_match_count_reached(std::int64_t& cumulative_match_count,
                                                            std::int64_t delta,
                                                            std::int64_t target_match_count) {
    bool is_reached = false;
    if (delta > 0) {
        increment_shared_value<is_parallel>(cumulative_match_count, delta);
    }
    if (load_shared_value<is_parallel>(cumulative_match_count) >= target_match_count) {
        is_reached = true;
    }
    return is_reached;
}

template <typename Cpu>
template <bool is_parallel>
void matching_engine<Cpu>::set_not_busy(bool& is_busy_engine, std::int64_t& busy_engine_count) {
    if (is_busy_engine) {
        is_busy_engine = false;
        decrement_shared_value<is_parallel>(busy_engine_count);
    }
}

template <typename Cpu>
template <bool is_parallel>
void matching_engine<Cpu>::run_and_wait(global_stack<Cpu>& gstack,
                                        std::int64_t& busy_engine_count,
                                        std::int64_t& cumulative_match_count,
                                        std::int64_t target_match_count,
                                        bool main_engine) {
    if (main_engine) {
        first_states_generator(hlocal_stack);
    }
    bool is_busy_engine = true;
    std::int64_t current_match_count = 0;
    ONEDAL_ASSERT(pattern != nullptr);
    for (;;) {
        if (target_match_count > 0 &&
            load_shared_value<is_parallel>(cumulative_match_count) >= target_match_count) {
            set_not_busy<is_parallel>(is_busy_engine, busy_engine_count);
            break;
        }
        if (hlocal_stack.states_in_stack() > 0) {
            while ((hlocal_stack.states_in_stack() > 5) && gstack.push(hlocal_stack))
                ;
            ONEDAL_ASSERT(hlocal_stack.states_in_stack() > 0);
            const auto delta = state_exploration();
            current_match_count += delta;
            if (target_match_count > 0 &&
                check_if_max_match_count_reached<is_parallel>(cumulative_match_count,
                                                              delta,
                                                              target_match_count)) {
                set_not_busy<is_parallel>(is_busy_engine, busy_engine_count);
                break;
            }
        }
        else {
            gstack.pop(hlocal_stack);
            if (hlocal_stack.empty()) {
                set_not_busy<is_parallel>(is_busy_engine, busy_engine_count);
                if (target_match_count > 0 &&
                    load_shared_value<is_parallel>(cumulative_match_count) >= target_match_count) {
                    break;
                }
                if (load_shared_value<is_parallel>(busy_engine_count) == 0)
                    break;
            }
            else if (!is_busy_engine) {
                is_busy_engine = true;
                increment_shared_value<is_parallel>(busy_engine_count);
            }
        }
    }
    return;
}

template <typename Cpu>
engine_bundle<Cpu>::engine_bundle(const graph<Cpu>* ppattern,
                                  const graph<Cpu>* ptarget,
                                  const std::int64_t* psorted_pattern_vertex,
                                  const std::int64_t* ppredecessor,
                                  const edge_direction* pdirection,
                                  sconsistent_conditions<Cpu> const* pcconditions,
                                  float* ppattern_vertex_probability,
                                  kind isomor_kind,
                                  inner_alloc alloc)
        : exploration_stack(alloc),
          allocator(alloc),
          pattern(ppattern),
          target(ptarget),
          sorted_pattern_vertex(psorted_pattern_vertex),
          predecessor(ppredecessor),
          direction(pdirection),
          pconsistent_conditions(pcconditions),
          pattern_vertex_probability(ppattern_vertex_probability),
          isomorphism_kind(isomor_kind) {}

template <typename Cpu>
engine_bundle<Cpu>::~engine_bundle() {
    pattern = nullptr;
    target = nullptr;
    sorted_pattern_vertex = nullptr;
    predecessor = nullptr;
    direction = nullptr;
    pconsistent_conditions = nullptr;
}

template <typename Cpu>
solution<Cpu> engine_bundle<Cpu>::combine_solutions(matching_engine<Cpu>* engine_array,
                                                    std::uint64_t array_size,
                                                    std::int64_t max_match_count) {
    ONEDAL_ASSERT(engine_array != nullptr);
    solution<Cpu> bundle_solutions(pattern->get_vertex_count(), allocator);
    for (std::uint64_t k = 0; k < array_size; k++) {
        std::uint64_t engine_max_index = 0;
        std::uint64_t engine_max_match_count = 0;
        std::uint64_t total_combined_count = 0;
        for (std::uint64_t i = 0; i < array_size; i++) {
            std::uint64_t match_count = engine_array[i].get_match_count();
            if (match_count > engine_max_match_count) {
                engine_max_match_count = match_count;
                engine_max_index = i;
            }
        }
        if (engine_max_match_count != 0) {
            total_combined_count += engine_array[engine_max_index].get_match_count();
            bundle_solutions.append(engine_array[engine_max_index].get_solution());
        }
        else
            break;

        if (max_match_count != 0 &&
            total_combined_count >= static_cast<std::uint64_t>(max_match_count))
            break;
    }
    return bundle_solutions;
}

template <typename Cpu>
solution<Cpu> engine_bundle<Cpu>::run(std::int64_t max_match_count) {
    std::int64_t degree = pattern->get_vertex_degree(sorted_pattern_vertex[0]);

    std::uint64_t first_states_count =
        pattern_vertex_probability[0] * target->get_vertex_count() + 1;
    std::uint64_t max_threads_count = dal::detail::threader_get_max_threads();
    std::uint64_t possible_first_states_count_per_thread = first_states_count / max_threads_count;
    if (possible_first_states_count_per_thread < 1) {
        possible_first_states_count_per_thread = 1;
    }
    else {
        possible_first_states_count_per_thread +=
            static_cast<bool>(first_states_count % max_threads_count);
    }

    std::uint64_t array_size = (max_threads_count >= 64)   ? max_threads_count * 2 / 10
                               : (max_threads_count >= 24) ? max_threads_count * 4 / 10
                               : (max_threads_count >= 8)  ? 4
                               : (max_threads_count >= 4)  ? 2
                                                           : 1;
#if defined(TARGET_X86_64)
    if (typeid(Cpu) == typeid(oneapi::dal::backend::cpu_dispatch_avx2) ||
        typeid(Cpu) == typeid(oneapi::dal::backend::cpu_dispatch_sse42)) {
        // TODO: Workaround that disabled parallelism for SSE4.2 and AVX2 code paths
        //       due to observed timeouts in case of execution under emulator
        //       related to atomics usage in such configurations.
        array_size = 1;
    }
#endif
    auto engine_array_ptr = allocator.make_shared_memory<matching_engine<Cpu>>(array_size);
    matching_engine<Cpu>* engine_array = engine_array_ptr.get();

    for (std::uint64_t i = 0; i < array_size; ++i) {
        new (engine_array + i) matching_engine<Cpu>(pattern,
                                                    target,
                                                    sorted_pattern_vertex,
                                                    predecessor,
                                                    direction,
                                                    pconsistent_conditions,
                                                    isomorphism_kind,
                                                    allocator);
    }

    state<Cpu> null_state(allocator);
    std::uint64_t task_counter = 0, index = 0;
    for (std::int64_t i = 0; i < target->get_vertex_count(); ++i) {
        if (degree <= target->get_vertex_degree(i) &&
            pattern->get_vertex_attribute(sorted_pattern_vertex[0]) ==
                target->get_vertex_attribute(i)) {
            index = task_counter % array_size;
            engine_array[index].push_into_stack(i);

            if ((engine_array[index].hlocal_stack.states_in_stack() /
                 possible_first_states_count_per_thread) > 0) {
                task_counter++;
            }
        }
    }

    global_stack<Cpu> gstack(pattern->get_vertex_count(), allocator);
    std::int64_t busy_engine_count(array_size);
    std::int64_t cumulative_match_count(0);

    if (array_size == 1) {
        engine_array[0].template run_and_wait<false>(gstack,
                                                     busy_engine_count,
                                                     cumulative_match_count,
                                                     max_match_count,
                                                     false);
    }
    else {
        dal::detail::threader_for(array_size, array_size, [&](const int index) {
            engine_array[index].template run_and_wait<true>(gstack,
                                                            busy_engine_count,
                                                            cumulative_match_count,
                                                            max_match_count,
                                                            false);
        });
    }

    auto aggregated_solution = combine_solutions(engine_array, array_size, max_match_count);

    for (std::uint64_t i = 0; i < array_size; i++) {
        engine_array[i].~matching_engine();
    }
    return aggregated_solution;
}
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
