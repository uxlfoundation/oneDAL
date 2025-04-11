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

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif

#include "oneapi/dal/spmd/communicator.hpp"

namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::detail {
namespace v1 {

#ifdef ONEDAL_DATA_PARALLEL
/// Implementation of the low-level SPMD communicator interface
/// that uses host-only functions to exchange USM data
class ONEDAL_EXPORT spmd_communicator_via_host_impl : public spmd::communicator_iface {
public:
    // Explicitly declare all virtual functions with overloads to workaround Clang warning
    // https://stackoverflow.com/questions/18515183/c-overloaded-virtual-function-warning-by-clang
    using base_t = spmd::communicator_iface;
    using base_t::bcast;
    using base_t::allgatherv;
    using base_t::allreduce;
    using base_t::sendrecv_replace;

    explicit spmd_communicator_via_host_impl(sycl::queue& queue) : queue_(queue) {}

    sycl::queue get_queue() override {
        return queue_;
    }

    spmd::request_iface* bcast(sycl::queue& q,
                               byte_t* send_buf,
                               std::int64_t count,
                               const data_type& dtype,
                               const std::vector<sycl::event>& deps,
                               std::int64_t root) override;

    spmd::request_iface* allgatherv(sycl::queue& q,
                                    const byte_t* send_buf,
                                    std::int64_t send_count,
                                    byte_t* recv_buf,
                                    const std::int64_t* recv_counts_host,
                                    const std::int64_t* displs_host,
                                    const data_type& dtype,
                                    const std::vector<sycl::event>& deps) override;
    spmd::request_iface* allreduce(sycl::queue& q,
                                   const byte_t* send_buf,
                                   byte_t* recv_buf,
                                   std::int64_t count,
                                   const data_type& dtype,
                                   const spmd::reduce_op& op,
                                   const std::vector<sycl::event>& deps) override;
    spmd::request_iface* sendrecv_replace(sycl::queue& q,
                                          byte_t* buf,
                                          std::int64_t count,
                                          const data_type& dtype,
                                          std::int64_t destination_rank,
                                          std::int64_t source_rank,
                                          const std::vector<sycl::event>& deps) override;

private:
    sycl::queue queue_;
};
#endif

} // namespace v1

#ifdef ONEDAL_DATA_PARALLEL
using v1::spmd_communicator_via_host_impl;
#endif

template <typename MemoryAccessKind>
struct via_host_interface_selector {
    using type = spmd::communicator_iface_base;
};

#ifdef ONEDAL_DATA_PARALLEL
template <>
struct via_host_interface_selector<spmd::device_memory_access::usm> {
    using type = spmd_communicator_via_host_impl;
};
#endif

} // namespace oneapi::dal::detail
