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

#include "oneapi/dal/spmd/communicator.hpp"
#include "oneapi/dal/spmd/detail/communicator_utils.hpp"

namespace oneapi::dal::preview::spmd {

namespace de = dal::detail;

template <typename MemoryAccessKind>
template <typename D>
request communicator<MemoryAccessKind>::bcast(const array<D>& ary, std::int64_t root) const {
    return de::bcast(*this, ary, root);
}

template <typename MemoryAccessKind>
template <typename D>
request communicator<MemoryAccessKind>::allgather(const array<D>& send,
                                                  const array<D>& recv) const {
    return de::allgather(*this, send, recv);
}

template <typename MemoryAccessKind>
template <typename D>
request communicator<MemoryAccessKind>::allgatherv(const array<D>& send,
                                                   const array<D>& recv,
                                                   const std::int64_t* recv_counts,
                                                   const std::int64_t* displs) const {
    return de::allgatherv(*this, send, recv, recv_counts, displs);
}

template <typename MemoryAccessKind>
template <typename D>
request communicator<MemoryAccessKind>::sendrecv_replace(const array<D>& buf,
                                                         std::int64_t destination_rank,
                                                         std::int64_t source_rank) const {
    return de::sendrecv_replace(*this, buf, destination_rank, source_rank);
}

template <typename MemoryAccessKind>
template <typename D>
request communicator<MemoryAccessKind>::allgather(const D& scalar, const array<D>& recv) const {
    return de::allgather(*this, scalar, recv);
}

template <typename MemoryAccessKind>
template <typename D>
request communicator<MemoryAccessKind>::allreduce(const array<D>& ary, const reduce_op& op) const {
    return de::allreduce(*this, ary, op);
}

template <typename MemoryAccessKind>
void communicator<MemoryAccessKind>::set_active_exception(const std::exception_ptr& ex_ptr) const {
    error_flag_ = 1;
    active_exception_ = ex_ptr;
}

template <typename MemoryAccessKind>
void communicator<MemoryAccessKind>::wait_for_exception_handling() const {
    auto com_error_flag = error_flag_;

    try {
        dal::detail::make_private<request>(
            impl_->allreduce(reinterpret_cast<byte_t*>(&com_error_flag),
                             reinterpret_cast<byte_t*>(&com_error_flag),
                             1,
                             dal::detail::make_data_type<decltype(com_error_flag)>(),
                             reduce_op::sum))
            .wait();
    }
    catch (...) {
        set_active_exception(std::current_exception());
    }

    if (com_error_flag) {
        if (error_flag_) {
            reset_error_flag();
            throw dal::preview::spmd::error_holder{ active_exception_ };
        }

        throw dal::preview::spmd::error_holder{ std::make_exception_ptr(
            spmd::coworker_error(oneapi::dal::detail::error_messages::spmd_coworker_failure())) };
    }
}

template <typename MemoryAccessKind>
void communicator<MemoryAccessKind>::reset_error_flag() const {
    error_flag_ = 0;
}

#define INSTANTIATE(M, D)                                                                        \
    template request communicator<M>::bcast<D>(const array<D>& ary, std::int64_t root) const;    \
    template request communicator<M>::allgather<D>(const array<D>& send, const array<D>& recv)   \
        const;                                                                                   \
    template request communicator<M>::allgather<D>(const D& scalar, const array<D>& recv) const; \
    template request communicator<M>::allgatherv<D>(const array<D>& send,                        \
                                                    const array<D>& recv,                        \
                                                    const std::int64_t* recv_counts,             \
                                                    const std::int64_t* displs) const;           \
    template request communicator<M>::allreduce<D>(const array<D>& ary, const reduce_op& op)     \
        const;                                                                                   \
    template request communicator<M>::sendrecv_replace<D>(const array<D>& ary,                   \
                                                          std::int64_t destination_rank,         \
                                                          std::int64_t) const;

#define INSTANTIATE_MEMORY_ACCESS(M)                                                             \
    template void communicator<M>::set_active_exception(const std::exception_ptr& ex_ptr) const; \
    template void communicator<M>::wait_for_exception_handling() const;                          \
    template void communicator<M>::reset_error_flag() const;                                     \
    INSTANTIATE(M, float)                                                                        \
    INSTANTIATE(M, double)                                                                       \
    INSTANTIATE(M, std::int8_t)                                                                  \
    INSTANTIATE(M, std::int16_t)                                                                 \
    INSTANTIATE(M, std::int32_t)                                                                 \
    INSTANTIATE(M, std::int64_t)                                                                 \
    INSTANTIATE(M, std::uint8_t)                                                                 \
    INSTANTIATE(M, std::uint16_t)                                                                \
    INSTANTIATE(M, std::uint32_t)                                                                \
    INSTANTIATE(M, std::uint64_t)

INSTANTIATE_MEMORY_ACCESS(device_memory_access::none)
INSTANTIATE_MEMORY_ACCESS(device_memory_access::usm)

} // namespace oneapi::dal::preview::spmd
