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
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/compiler_adapt.hpp"
#include "oneapi/dal/detail/threading.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template <typename Cpu>
class stack {
public:
    stack(inner_alloc alloc);
    virtual ~stack();
    stack(stack<Cpu>&& _stack);
    stack<Cpu>& operator=(stack<Cpu>&& _stack);
    state<Cpu>* operator[](std::int64_t index);

    void push(state<Cpu>* new_state);
    state<Cpu>* pop();
    std::int64_t size() const;
    void clear(bool direct = true);

private:
    inner_alloc allocator;
    std::int64_t max_stack_size;
    std::int64_t stack_size;
    state<Cpu>** data;

    void increase_stack_size();
    void delete_data();
};

template <typename Cpu>
class dfs_stack;

template <typename Cpu>
class global_stack;

template <typename Cpu>
class vertex_stack {
public:
    vertex_stack(inner_alloc alloc);
    vertex_stack(const std::uint64_t max_states_size, inner_alloc alloc);
    virtual ~vertex_stack();
    void clean();

    void push(const std::uint64_t vertex_id);
    std::int64_t pop();
    bool delete_vertex();
    std::uint64_t size() const;
    std::uint64_t max_size() const;

private:
    inner_alloc allocator;
    std::uint64_t stack_size;
    std::uint64_t* stack_data;
    std::uint64_t* ptop;
    void increase_stack_size();
    bool use_external_memory;
    std::uint64_t* bottom_;

    friend class dfs_stack<Cpu>;
    friend class global_stack<Cpu>;
};

template <typename Cpu>
class dfs_stack;

template <typename Cpu>
class global_stack {
public:
    global_stack(std::int64_t vertex_count, inner_alloc alloc)
            : allocator(alloc),
              vertex_count_(vertex_count) {}

    global_stack(const global_stack&) = delete;
    global_stack(global_stack&&) = delete;

    ~global_stack() {
        clear();
    }

    global_stack& operator=(const global_stack&) = delete;
    global_stack& operator=(global_stack&&) = delete;

    bool push(dfs_stack<Cpu>& s);
    void pop(dfs_stack<Cpu>& s);

private:
    void internal_push(dfs_stack<Cpu>& s, std::uint64_t level);
    void clear();
    void grow();

    static constexpr std::uint64_t null_vertex() {
        return static_cast<std::uint64_t>(-1);
    }

    std::int64_t size() const {
        return (bottom_ != nullptr && vertex_count_ != 0) ? (top_ - bottom_) / vertex_count_ : 0;
    }

    bool empty() const {
        return (size() == 0);
    }

    dal::detail::mutex mutex_;
    inner_alloc allocator;
    std::int64_t vertex_count_;
    std::uint64_t* bottom_{ nullptr };
    std::uint64_t* top_{ nullptr };
    std::int64_t capacity_{ 0 };
};

template <typename Cpu>
class dfs_stack {
public:
    dfs_stack(inner_alloc alloc);
    virtual ~dfs_stack();
    void init(const std::uint64_t levels);
    void init(const std::uint64_t levels, const std::uint64_t max_states_size);
    void init(const std::uint64_t levels, const std::uint64_t* max_states_size_per_level);

    void push_into_current_level(const std::uint64_t vertex_id);
    void push_into_next_level(const std::uint64_t vertex_id);
    void update();

    std::uint64_t states_in_stack() const;
    std::uint64_t size(const std::uint64_t level) const;
    std::uint64_t max_level_width(const std::uint64_t level) const;

    std::uint64_t get_current_level() const;
    std::uint64_t get_current_level_index() const;
    std::uint64_t get_current_level_fill_size() const;
    std::uint64_t top(const std::uint64_t level) const;

    state<Cpu> get_current_state() const;
    void fill_solution(std::int64_t* solution_core, const std::uint64_t last_vertex_id) const;

    void delete_current_state();

    bool empty() const;

protected:
    inner_alloc allocator;
    std::uint64_t max_level_size;
    vertex_stack<Cpu>* data_by_levels;

    std::uint64_t current_level;

    void increase_core_level();
    void decrease_core_level();

private:
    void delete_data();

    friend class global_stack<Cpu>;
};

template <typename Cpu>
stack<Cpu>::stack(inner_alloc alloc) : allocator(alloc) {
    max_stack_size = 100;
    stack_size = 0;
    data = allocator.allocate<state<Cpu>*>(max_stack_size);
    for (std::int64_t i = 0; i < max_stack_size; i++) {
        data[i] = nullptr;
    }
}

template <typename Cpu>
void stack<Cpu>::delete_data() {
    if (data != nullptr) {
        for (std::int64_t i = 0; i < max_stack_size; i++) {
            if (data[i] != nullptr) {
                data[i]->clear();
                allocator.deallocate(data[i], 0);
                data[i] = nullptr;
            }
        }
        allocator.deallocate(data, max_stack_size);
        data = nullptr;
    }
}

template <typename Cpu>
void stack<Cpu>::clear(bool direct) {
    for (std::int64_t i = 0; i < stack_size * direct; i++) {
        if (data[i] != nullptr) {
            data[i]->clear();
            allocator.deallocate(data[i], 0);
            data[i] = nullptr;
        }
    }
    stack_size = 0;
}

template <typename Cpu>
stack<Cpu>::stack(stack<Cpu>&& _stack) : allocator(_stack.allocator),
                                         data(_stack.data) {
    max_stack_size = _stack.max_stack_size;
    stack_size = _stack.stack_size;
    _stack.data = nullptr;
    _stack.max_stack_size = 0;
    _stack.stack_size = 0;
}

template <typename Cpu>
stack<Cpu>& stack<Cpu>::operator=(stack<Cpu>&& _stack) {
    if (&_stack == this) {
        return *this;
    }
    delete_data();
    max_stack_size = _stack.max_stack_size;
    stack_size = _stack.stack_size;
    data = _stack.data;
    _stack.data = nullptr;
    _stack.max_stack_size = 0;
    _stack.stack_size = 0;
    return *this;
}

template <typename Cpu>
stack<Cpu>::~stack() {
    if (data != nullptr) {
        for (std::int64_t i = 0; i < max_stack_size; i++) {
            if (data[i] != nullptr) {
                this->data[i]->clear();
                allocator.deallocate(data[i], 0);
                data[i] = nullptr;
            }
        }
        allocator.deallocate(data, max_stack_size);
        data = nullptr;
    }
    stack_size = 0;
}

template <typename Cpu>
void stack<Cpu>::push(state<Cpu>* new_state) {
    if (new_state != nullptr) {
        if (max_stack_size == 0 || stack_size >= max_stack_size) {
            increase_stack_size();
        }
        data[stack_size] = new_state;
        stack_size++;
    }
}

template <typename Cpu>
state<Cpu>* stack<Cpu>::pop() {
    state<Cpu>* pstate = nullptr;
    if (stack_size > 0) {
        stack_size--;
        pstate = data[stack_size];
        data[stack_size] = nullptr;
    }
    return pstate;
}

template <typename Cpu>
state<Cpu>* stack<Cpu>::operator[](std::int64_t index) {
    return data[index];
}

template <typename Cpu>
std::int64_t stack<Cpu>::size() const {
    return stack_size;
}

template <typename Cpu>
void stack<Cpu>::increase_stack_size() {
    const auto new_max_stack_size = (max_stack_size > 0) ? 2 * max_stack_size : 100;
    state<Cpu>** tmp_data = allocator.allocate<state<Cpu>*>(new_max_stack_size);
    for (std::int64_t i = 0; i < max_stack_size; i++) {
        tmp_data[i] = data[i];
        data[i] = nullptr;
    }
    for (std::int64_t i = max_stack_size; i < new_max_stack_size; i++) {
        tmp_data[i] = nullptr;
    }
    allocator.deallocate(data, max_stack_size);
    max_stack_size = new_max_stack_size;
    data = tmp_data;
    tmp_data = nullptr;
}

template <typename Cpu>
vertex_stack<Cpu>::vertex_stack(inner_alloc alloc) : allocator(alloc),
                                                     bottom_(nullptr) {
    stack_size = 0;
    stack_data = nullptr;
    ptop = nullptr;
    use_external_memory = false;
}

template <typename Cpu>
vertex_stack<Cpu>::vertex_stack(const std::uint64_t max_states_size, inner_alloc alloc)
        : allocator(alloc) {
    use_external_memory = false;
    stack_size = max_states_size;
    stack_data = allocator.allocate<std::uint64_t>(stack_size);
    ptop = stack_data;
    bottom_ = stack_data;
}

template <typename Cpu>
void vertex_stack<Cpu>::clean() {
    allocator.deallocate(stack_data, stack_size);
    stack_data = nullptr;
    ptop = nullptr;
    stack_size = 0;
}

template <typename Cpu>
vertex_stack<Cpu>::~vertex_stack() {
    this->clean();
}

template <typename Cpu>
void vertex_stack<Cpu>::push(const std::uint64_t vertex_id) {
    ONEDAL_ASSERT(ptop != nullptr);
    ONEDAL_ASSERT(stack_data != nullptr);
    if (static_cast<std::uint64_t>(ptop - stack_data) >= stack_size) {
        increase_stack_size();
    }
    ONEDAL_ASSERT(ptop != nullptr);
    ONEDAL_ASSERT(ptop >= bottom_);
    ONEDAL_ASSERT(ptop <= stack_data + stack_size);
    *ptop = vertex_id;
    ptop++;
}

template <typename Cpu>
std::int64_t vertex_stack<Cpu>::pop() {
    if (ptop != nullptr && ptop != bottom_) {
        ONEDAL_ASSERT(ptop >= bottom_);
        ONEDAL_ASSERT(ptop <= stack_data + stack_size);

        ptop--;
        return *ptop;
    }
    return -1;
}

template <typename Cpu>
bool vertex_stack<Cpu>::delete_vertex() {
    ONEDAL_ASSERT(ptop != nullptr);
    ONEDAL_ASSERT(ptop >= bottom_);
    ONEDAL_ASSERT(ptop <= stack_data + stack_size);

    ptop -= (ptop != nullptr) && (ptop != bottom_);
    return !(ptop - bottom_);
}

template <typename Cpu>
std::uint64_t vertex_stack<Cpu>::size() const {
    return ptop - bottom_;
}

template <typename Cpu>
std::uint64_t vertex_stack<Cpu>::max_size() const {
    return stack_size;
}

template <typename Cpu>
void vertex_stack<Cpu>::increase_stack_size() {
    std::uint64_t* tmp_data = allocator.allocate<std::uint64_t>(2 * stack_size);
    const auto skip_count = bottom_ - stack_data;
    for (std::uint64_t i = 0; i < stack_size - skip_count; i++) {
        tmp_data[i] = stack_data[i + skip_count];
    }
    allocator.deallocate(stack_data, stack_size);
    stack_size *= 2;
    ptop = size() + tmp_data;
    bottom_ = tmp_data;
    stack_data = tmp_data;
    tmp_data = nullptr;

    ONEDAL_ASSERT(ptop != nullptr);
    ONEDAL_ASSERT(ptop >= bottom_);
    ONEDAL_ASSERT(ptop <= stack_data + stack_size);
}

template <typename Cpu>
bool global_stack<Cpu>::push(dfs_stack<Cpu>& s) {
    for (auto level = s.get_current_level_index(); level > 0; --level) {
        if (s.data_by_levels[level].size() > 1) {
            internal_push(s, level);
            return true;
        }
    }

    if (s.data_by_levels[0].size() > 1) {
        internal_push(s, 0);
        return true;
    }

    return false;
}

template <typename Cpu>
void global_stack<Cpu>::pop(dfs_stack<Cpu>& s) {
    ONEDAL_ASSERT(s.empty());
    const dal::detail::scoped_lock lock(mutex_);
    if (!empty()) {
        // const auto& v = data_.top();
        const auto v = top_ - vertex_count_;
        ONEDAL_ASSERT(v >= bottom_);
        for (std::int64_t i = 0; i < vertex_count_ && v[i] != null_vertex(); ++i) {
            ONEDAL_ASSERT(i <= dal::detail::integral_cast<std::int64_t>(s.max_level_size));
            s.push_into_current_level(v[i]);
            if (i != vertex_count_ - 1 && v[i + 1] != null_vertex()) {
                s.increase_core_level();
            }
        }
        top_ = v;
    }
}

template <typename Cpu>
void global_stack<Cpu>::internal_push(dfs_stack<Cpu>& s, std::uint64_t level) {
    ONEDAL_ASSERT(vertex_count_ >= 0);
    // Collect state and push back
    {
        const auto v = allocator.allocate<std::uint64_t>(level + 1);

        for (std::uint64_t i = 0; i < level; ++i) {
            ONEDAL_ASSERT(i < s.max_level_size);
            ONEDAL_ASSERT(s.data_by_levels[i].ptop != nullptr);
            ONEDAL_ASSERT(s.data_by_levels[i].ptop != s.data_by_levels[i].bottom_);
            ONEDAL_ASSERT(s.data_by_levels[i].ptop >= s.data_by_levels[i].bottom_);
            ONEDAL_ASSERT(s.data_by_levels[i].ptop <=
                          s.data_by_levels[i].stack_data + s.data_by_levels[i].stack_size);
            v[i] = s.data_by_levels[i].ptop[-1];
        }

        ONEDAL_ASSERT(level < s.max_level_size);
        ONEDAL_ASSERT(s.data_by_levels[level].ptop != nullptr);
        ONEDAL_ASSERT(s.data_by_levels[level].bottom_ != nullptr);
        ONEDAL_ASSERT(s.data_by_levels[level].ptop != s.data_by_levels[level].bottom_);
        ONEDAL_ASSERT(s.data_by_levels[level].ptop >= s.data_by_levels[level].bottom_);
        ONEDAL_ASSERT(s.data_by_levels[level].ptop <=
                      s.data_by_levels[level].stack_data + s.data_by_levels[level].stack_size);
        v[level] = *(s.data_by_levels[level].bottom_);

        const dal::detail::scoped_lock lock(mutex_);
        if (size() >= capacity_) {
            grow();
        }

        ONEDAL_ASSERT(top_ + vertex_count_ <= bottom_ + capacity_ * vertex_count_);
        std::uint64_t j = 0;
        for (; j <= level; ++j) {
            *(top_++) = v[j];
        }
        for (; j < static_cast<std::uint64_t>(vertex_count_); ++j) {
            *(top_++) = null_vertex();
        }

        allocator.deallocate(v, level + 1);
    }

    // Remove state
    ++(s.data_by_levels[level].bottom_);
}

template <typename Cpu>
void global_stack<Cpu>::clear() {
    if (bottom_ != nullptr) {
        ONEDAL_ASSERT(top_ != nullptr);
        allocator.deallocate(bottom_,
                             (capacity_ * vertex_count_ > 0) ? capacity_ * vertex_count_ : 1);
        bottom_ = nullptr;
        top_ = nullptr;
        capacity_ = 0;
    }
}

template <typename Cpu>
void global_stack<Cpu>::grow() {
    const std::int64_t new_capacity = (capacity_ > 0) ? capacity_ * 2 : 1;
    const auto new_bottom = allocator.allocate<std::uint64_t>(
        (new_capacity * vertex_count_ > 0) ? new_capacity * vertex_count_ : 1);
    const auto new_top = new_bottom + size() * vertex_count_;

    PRAGMA_IVDEP
    for (auto dest = new_bottom, src = bottom_; dest != new_top;) {
        *(dest++) = *(src++);
    }

    clear();

    bottom_ = new_bottom;
    top_ = new_top;
    capacity_ = new_capacity;
}

template <typename Cpu>
dfs_stack<Cpu>::dfs_stack(inner_alloc alloc) : allocator(alloc) {
    max_level_size = 0;
    data_by_levels = nullptr;

    current_level = 0;
}

template <typename Cpu>
void dfs_stack<Cpu>::init(const std::uint64_t levels) {
    max_level_size = levels;
    current_level = 0;
    data_by_levels = allocator.allocate<vertex_stack<Cpu>>(max_level_size);
}

template <typename Cpu>
void dfs_stack<Cpu>::init(const std::uint64_t levels, const std::uint64_t max_states_size) {
    init(levels);

    for (std::uint64_t i = 0; i < max_level_size; ++i) {
        new (data_by_levels + i) vertex_stack<Cpu>(max_states_size, allocator);
    }
}

template <typename Cpu>
void dfs_stack<Cpu>::init(const std::uint64_t levels,
                          const std::uint64_t* max_states_size_per_level) {
    init(levels);

    for (std::uint64_t i = 0; i < max_level_size; ++i) {
        new (data_by_levels + i) vertex_stack<Cpu>(max_states_size_per_level[i], allocator);
    }
}

template <typename Cpu>
void dfs_stack<Cpu>::delete_data() {
    for (std::uint64_t i = 0; i < max_level_size; i++) {
        data_by_levels[i].clean();
    }
    allocator.deallocate(data_by_levels, max_level_size);
    data_by_levels = nullptr;

    max_level_size = 0;
    current_level = 0;
}

template <typename Cpu>
dfs_stack<Cpu>::~dfs_stack() {
    delete_data();
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::get_current_level() const {
    return current_level + 1;
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::get_current_level_fill_size() const {
    return data_by_levels[current_level].size();
}

template <typename Cpu>
void dfs_stack<Cpu>::push_into_current_level(const std::uint64_t vertex_id) {
    data_by_levels[current_level].push(vertex_id);
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::get_current_level_index() const {
    return current_level;
}

template <typename Cpu>
void dfs_stack<Cpu>::push_into_next_level(const std::uint64_t vertex_id) {
    data_by_levels[current_level + 1].push(vertex_id);
}

template <typename Cpu>
void dfs_stack<Cpu>::increase_core_level() {
    current_level++;
}

template <typename Cpu>
void dfs_stack<Cpu>::decrease_core_level() {
    current_level--;
}

template <typename Cpu>
void dfs_stack<Cpu>::update() {
    std::uint64_t new_level = current_level + 1;
    if (new_level < max_level_size && size(new_level) > 0) {
        current_level++;
    }
    else {
        delete_current_state();
    }
}

template <typename Cpu>
state<Cpu> dfs_stack<Cpu>::get_current_state() const {
    state<Cpu> result(current_level + 1, allocator);
    for (std::int64_t i = 0; i < result.core_length; i++) {
        result.core[i] = *(data_by_levels[i].ptop - 1);
    }
    return result;
}

template <typename Cpu>
void dfs_stack<Cpu>::fill_solution(std::int64_t* solution_core,
                                   const std::uint64_t last_vertex_id) const {
    for (std::uint64_t i = 0; i <= current_level; i++) {
        solution_core[i] = *(data_by_levels[i].ptop - 1);
    }
    solution_core[current_level + 1] = last_vertex_id;
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::top(const std::uint64_t level) const {
    return *(data_by_levels[level].ptop - 1);
}

template <typename Cpu>
void dfs_stack<Cpu>::delete_current_state() {
    while (data_by_levels[current_level].delete_vertex() && current_level > 0) {
        current_level--;
    }
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::states_in_stack() const {
    std::int64_t size = 0;
    for (std::uint64_t i = 0; i <= current_level; i++) {
        size += data_by_levels[i].size();
    }
    return size - current_level;
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::size(const std::uint64_t level) const {
    return data_by_levels[level].size();
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::max_level_width(const std::uint64_t level) const {
    return data_by_levels[level].max_size();
}

template <typename Cpu>
bool dfs_stack<Cpu>::empty() const {
    return (current_level == 0) && ((max_level_size == 0) || (data_by_levels[0].size() == 0));
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
