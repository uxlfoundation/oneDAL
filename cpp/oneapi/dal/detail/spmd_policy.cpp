/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/detail/spmd_policy.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename MemoryAccessKind>
class spmd_policy_impl {
public:
    explicit spmd_policy_impl(const spmd::communicator<MemoryAccessKind>& comm) : comm(comm) {}
    spmd::communicator<MemoryAccessKind> comm;
};

template <typename MemoryAccessKind>
spmd_policy_base<MemoryAccessKind>::spmd_policy_base(
    const spmd::communicator<MemoryAccessKind>& comm)
        : impl_(new spmd_policy_impl<MemoryAccessKind>{ comm }) {}

template <typename MemoryAccessKind>
const spmd::communicator<MemoryAccessKind>& spmd_policy_base<MemoryAccessKind>::get_communicator()
    const {
    return impl_->comm;
}

template class spmd_policy_base<spmd::device_memory_access::none>;
template class spmd_policy_base<spmd::device_memory_access::usm>;

} // namespace v1
} // namespace oneapi::dal::detail
