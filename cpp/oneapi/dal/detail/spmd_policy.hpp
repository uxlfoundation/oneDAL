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

#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/spmd/communicator.hpp"

namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::detail {
namespace v1 {

template <typename MemoryAccessKind>
class spmd_policy_impl;

template <typename MemoryAccessKind>
class ONEDAL_EXPORT spmd_policy_base : public base {
public:
    explicit spmd_policy_base(const spmd::communicator<MemoryAccessKind>& comm);

    const spmd::communicator<MemoryAccessKind>& get_communicator() const;

private:
    pimpl<spmd_policy_impl<MemoryAccessKind>> impl_;
};

template <typename LocalPolicy>
class spmd_policy : public spmd_policy_base<spmd::device_memory_access::none> {
    static_assert(is_execution_policy_v<LocalPolicy>, "Unknown local policy type");

public:
    explicit spmd_policy(const LocalPolicy& local_policy,
                         const spmd::communicator<spmd::device_memory_access::none>& comm)
            : spmd_policy_base(comm),
              local_policy_(local_policy) {}

    const LocalPolicy& get_local() const {
        return local_policy_;
    }

private:
    LocalPolicy local_policy_;
};

#ifdef ONEDAL_DATA_PARALLEL
template <>
class spmd_policy<data_parallel_policy> : public spmd_policy_base<spmd::device_memory_access::usm> {
public:
    explicit spmd_policy(const data_parallel_policy& local_policy,
                         const spmd::communicator<spmd::device_memory_access::usm>& comm)
            : spmd_policy_base(comm),
              local_policy_(local_policy) {}

    const data_parallel_policy& get_local() const {
        return local_policy_;
    }

private:
    data_parallel_policy local_policy_;
};
#endif

template <typename LocalPolicy>
struct is_execution_policy<spmd_policy<LocalPolicy>> : std::bool_constant<true> {};

template <typename LocalPolicy>
struct is_distributed_policy<spmd_policy<LocalPolicy>> : std::bool_constant<true> {};

template <typename LocalPolicy>
struct is_host_policy<spmd_policy<LocalPolicy>> : is_host_policy<LocalPolicy> {};

template <typename LocalPolicy>
struct is_data_parallel_policy<spmd_policy<LocalPolicy>> : is_data_parallel_policy<LocalPolicy> {};

using spmd_host_policy = spmd_policy<host_policy>;

#ifdef ONEDAL_DATA_PARALLEL
using spmd_data_parallel_policy = spmd_policy<data_parallel_policy>;
#endif

} // namespace v1

using v1::spmd_policy;
using v1::spmd_host_policy;

#ifdef ONEDAL_DATA_PARALLEL
using v1::spmd_data_parallel_policy;
#endif

} // namespace oneapi::dal::detail
