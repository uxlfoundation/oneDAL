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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/spmd/communicator.hpp"

namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::backend {

namespace de = dal::detail;

/// Implementation of SPMD communicator for one-rank system
template <typename MemoryAccessKind>
class fake_spmd_communicator : public spmd::communicator<MemoryAccessKind> {
public:
    fake_spmd_communicator();
};

/// Wrapper over public SPMD request.
/// The event-like object seems more natural to data-parallel algorithms
/// which are built on top of SYCL events.
class communicator_event {
public:
    communicator_event() = default;
    communicator_event(const spmd::request& req) : public_req_(req) {}
    communicator_event(spmd::request&& req) : public_req_(std::move(req)) {}

    void wait() {
        public_req_.wait();
    }

private:
    spmd::request public_req_;
};

/// Wrapper over public SPMD communicator.
/// The additional layer of abstraction is added to have more flexibility in changing the
/// communicator interface which algorithms depend on. For example, this can be used to add
/// collective operation overloading for internal classes such as `ndarray`.
template <typename MemoryAccessKind>
class communicator {
public:
    /// Creates communicator based on public SPMD interface
    communicator(const spmd::communicator<MemoryAccessKind>& comm)
            : public_comm_(comm),
              is_distributed_(true) {}

    /// Creates communicator for one-rank system
    communicator(const fake_spmd_communicator<MemoryAccessKind>& comm =
                     fake_spmd_communicator<MemoryAccessKind>{})
            : public_comm_(comm),
              is_distributed_(false) {}

    bool is_distributed() const {
        return is_distributed_;
    }

    std::int64_t get_rank() const {
        return public_comm_.get_rank();
    }

    std::int64_t get_rank_count() const {
        return public_comm_.get_rank_count();
    }

    std::int64_t get_default_root_rank() const {
        return public_comm_.get_default_root_rank();
    }

    template <typename... Args>
    bool is_root_rank(Args&&... args) const {
        return public_comm_.is_root_rank(std::forward<Args>(args)...);
    }

    void barrier() const {
        public_comm_.barrier();
    }

    // void is_init(int * flag) const {
    //     public_comm_.is_init(flag);
    // }

    template <typename... Args>
    communicator_event bcast(Args&&... args) const {
        return public_comm_.bcast(std::forward<Args>(args)...);
    }

    template <typename... Args>
    communicator_event allgatherv(Args&&... args) const {
        return public_comm_.allgatherv(std::forward<Args>(args)...);
    }

    template <typename... Args>
    communicator_event allgather(Args&&... args) const {
        return public_comm_.allgather(std::forward<Args>(args)...);
    }

    template <typename... Args>
    communicator_event allreduce(Args&&... args) const {
        return public_comm_.allreduce(std::forward<Args>(args)...);
    }
    template <typename... Args>
    communicator_event sendrecv_replace(Args&&... args) const {
        return public_comm_.sendrecv_replace(std::forward<Args>(args)...);
    }

    void set_active_exception(const std::exception_ptr& ex_ptr) const {
        public_comm_.set_active_exception(ex_ptr);
    }

    void wait_for_exception_handling() const {
        public_comm_.wait_for_exception_handling();
    }

private:
    spmd::communicator<MemoryAccessKind> public_comm_;
    bool is_distributed_;
};

} // namespace oneapi::dal::backend
