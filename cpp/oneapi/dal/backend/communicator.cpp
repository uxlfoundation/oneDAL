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

#include "oneapi/dal/backend/communicator.hpp"
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend {

inline void copy_if_different_pointers(byte_t* dst,
                                       const byte_t* src,
                                       std::int64_t count,
                                       const data_type& dtype) {
    if (count == 0) {
        return;
    }

    ONEDAL_ASSERT(src);
    ONEDAL_ASSERT(dst);
    ONEDAL_ASSERT(count > 0);

    if (dst == src) {
        return;
    }

    const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
    const std::int64_t size = dal::detail::check_mul_overflow(dtype_size, count);
    memcpy(dst, src, size);
}

#ifdef ONEDAL_DATA_PARALLEL
inline void copy_if_different_pointers(sycl::queue& q,
                                       byte_t* dst,
                                       const byte_t* src,
                                       std::int64_t count,
                                       const data_type& dtype) {
    if (count == 0) {
        return;
    }

    ONEDAL_ASSERT(src);
    ONEDAL_ASSERT(dst);
    ONEDAL_ASSERT(count > 0);
    ONEDAL_ASSERT(is_known_usm(q, src));
    ONEDAL_ASSERT(is_known_usm(q, dst));

    if (dst == src) {
        return;
    }

    const std::int64_t dtype_size = dal::detail::get_data_type_size(dtype);
    const std::int64_t size = dal::detail::check_mul_overflow(dtype_size, count);
    memcpy(q, dst, src, size).wait_and_throw();
}
#endif

class fake_spmd_communicator_host_impl : public spmd::communicator_iface_base {
public:
    using base_t = spmd::communicator_iface_base;
    using request_t = spmd::request_iface;

    static constexpr std::int64_t root_rank = 0;
    static constexpr std::int64_t rank_count = 1;

    std::int64_t get_rank() override {
        return root_rank;
    }

    std::int64_t get_default_root_rank() override {
        return root_rank;
    }

    std::int64_t get_rank_count() override {
        return rank_count;
    }

    bool get_mpi_offload_support() override {
        return false;
    }

    bool use_sendrecv_replace_alternative() override {
        return false;
    }

    void barrier() override {}

    // void is_init(int * flag) override {}

    request_t* bcast(byte_t* send_buf,
                     std::int64_t count,
                     const data_type& dtype,
                     std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);
        return nullptr;
    }

    request_t* allgatherv(const byte_t* send_buf,
                          std::int64_t send_count,
                          byte_t* recv_buf,
                          const std::int64_t* recv_counts,
                          const std::int64_t* displs,
                          const data_type& dtype) override {
        ONEDAL_ASSERT(recv_counts);
        ONEDAL_ASSERT(displs);

        copy_if_different_pointers(recv_buf + displs[0], send_buf, send_count, dtype);

        return nullptr;
    }

    request_t* allreduce(const byte_t* send_buf,
                         byte_t* recv_buf,
                         std::int64_t count,
                         const data_type& dtype,
                         const spmd::reduce_op& op) override {
        copy_if_different_pointers(recv_buf, send_buf, count, dtype);
        return nullptr;
    }
    request_t* sendrecv_replace(byte_t* buf,
                                std::int64_t count,
                                const data_type& dtype,
                                std::int64_t destination_rank,
                                std::int64_t source_rank,
                                byte_t* recv_buf = nullptr) override {
        return nullptr;
    }
};

#ifdef ONEDAL_DATA_PARALLEL
class fake_spmd_communicator_device_impl : public spmd::communicator_iface {
public:
    using base_t = spmd::communicator_iface;
    using request_t = spmd::request_iface;

    static constexpr std::int64_t root_rank = 0;
    static constexpr std::int64_t rank_count = 1;

    std::int64_t get_rank() override {
        return root_rank;
    }

    std::int64_t get_default_root_rank() override {
        return root_rank;
    }

    std::int64_t get_rank_count() override {
        return rank_count;
    }

    bool get_mpi_offload_support() override {
        return false;
    }

    bool use_sendrecv_replace_alternative() override {
        return false;
    }

    void barrier() override {}

    // void is_init(int * flag) override {}

    request_t* bcast(byte_t* send_buf,
                     std::int64_t count,
                     const data_type& dtype,
                     std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);
        return nullptr;
    }

    request_t* bcast(sycl::queue& q,
                     byte_t* send_buf,
                     std::int64_t count,
                     const data_type& dtype,
                     const event_vector& deps,
                     std::int64_t root) override {
        ONEDAL_ASSERT(root == root_rank);
        sycl::event::wait_and_throw(deps);
        return nullptr;
    }

    request_t* allgatherv(const byte_t* send_buf,
                          std::int64_t send_count,
                          byte_t* recv_buf,
                          const std::int64_t* recv_counts,
                          const std::int64_t* displs,
                          const data_type& dtype) override {
        ONEDAL_ASSERT(recv_counts);
        ONEDAL_ASSERT(displs);
        ONEDAL_ASSERT(recv_counts[0] == send_count);

        copy_if_different_pointers(recv_buf + displs[0], send_buf, send_count, dtype);

        return nullptr;
    }

    request_t* allgatherv(sycl::queue& q,
                          const byte_t* send_buf,
                          std::int64_t send_count,
                          byte_t* recv_buf,
                          const std::int64_t* recv_counts,
                          const std::int64_t* displs,
                          const data_type& dtype,
                          const event_vector& deps) override {
        ONEDAL_ASSERT(recv_counts);
        ONEDAL_ASSERT(displs);

        sycl::event::wait_and_throw(deps);
        copy_if_different_pointers(q, recv_buf + displs[0], send_buf, send_count, dtype);

        return nullptr;
    }

    request_t* allreduce(const byte_t* send_buf,
                         byte_t* recv_buf,
                         std::int64_t count,
                         const data_type& dtype,
                         const spmd::reduce_op& op) override {
        copy_if_different_pointers(recv_buf, send_buf, count, dtype);
        return nullptr;
    }

    request_t* allreduce(sycl::queue& q,
                         const byte_t* send_buf,
                         byte_t* recv_buf,
                         std::int64_t count,
                         const data_type& dtype,
                         const spmd::reduce_op& op,
                         const event_vector& deps) override {
        sycl::event::wait_and_throw(deps);
        copy_if_different_pointers(recv_buf, send_buf, count, dtype);
        return nullptr;
    }
    request_t* sendrecv_replace(byte_t* buf,
                                std::int64_t count,
                                const data_type& dtype,
                                std::int64_t destination_rank,
                                std::int64_t source_rank,
                                byte_t* recv_buf = nullptr) override {
        return nullptr;
    }
    request_t* sendrecv_replace(sycl::queue& q,
                                byte_t* buf,
                                std::int64_t count,
                                const data_type& dtype,
                                std::int64_t destination_rank,
                                std::int64_t source_rank,
                                const event_vector& deps) override {
        sycl::event::wait_and_throw(deps);
        return nullptr;
    }

    sycl::queue get_queue() override {
        return sycl::queue();
    }
};
#endif

template <typename MemoryAccessKind>
struct comm_impl_selector {
    using comm_type = fake_spmd_communicator_host_impl;
};

#ifdef ONEDAL_DATA_PARALLEL
template <>
struct comm_impl_selector<spmd::device_memory_access::usm> {
    using comm_type = fake_spmd_communicator_device_impl;
};
#endif

template <typename MemoryAccessKind>
fake_spmd_communicator<MemoryAccessKind>::fake_spmd_communicator()
        : spmd::communicator<MemoryAccessKind>(
              new typename comm_impl_selector<MemoryAccessKind>::comm_type{}) {}

template class fake_spmd_communicator<spmd::device_memory_access::usm>;
template class fake_spmd_communicator<spmd::device_memory_access::none>;

} // namespace oneapi::dal::backend
