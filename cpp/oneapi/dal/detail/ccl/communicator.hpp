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

// TODO host version
#ifdef ONEDAL_DATA_PARALLEL

// IMPORTANT! This file should not be included to any other non-ccl header,
// otherwise it creates dependeny on CCL at the user's application compile time
// TODO: In the future this can be solved via __has_include C++17 feature

#include <oneapi/dal/array.hpp>
#include "oneapi/dal/detail/communicator.hpp"
#include <mpi.h>
#include <ccl.hpp>

namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::detail {
namespace v1 {

using preview::spmd::communication_error;

inline ccl::datatype make_ccl_data_type(const data_type& dtype) {
    switch (dtype) {
        case data_type::int8: return ccl::datatype::int8;
        case data_type::int16: return ccl::datatype::int16;
        case data_type::int32: return ccl::datatype::int32;
        case data_type::int64: return ccl::datatype::int64;
        case data_type::uint8: return ccl::datatype::uint8;
        case data_type::uint16: return ccl::datatype::uint16;
        case data_type::uint32: return ccl::datatype::uint32;
        case data_type::uint64: return ccl::datatype::uint64;
        case data_type::float32: return ccl::datatype::float32;
        case data_type::float64: return ccl::datatype::float64;
        case data_type::bfloat16: return ccl::datatype::uint16;
        default: throw communication_error(dal::detail::error_messages::invalid_data_type());
    }
}

inline ccl::reduction make_ccl_reduce_op(const spmd::reduce_op& op) {
    switch (op) {
        case spmd::reduce_op::sum: return ccl::reduction::sum;
        case spmd::reduce_op::min: return ccl::reduction::min;
        case spmd::reduce_op::max: return ccl::reduction::max;
        default: throw communication_error(dal::detail::error_messages::invalid_op());
    }
}

class ccl_request_impl : public spmd::request_iface {
public:
    explicit ccl_request_impl(ccl::event&& request) {
        event_ = std::move(request);
    }

    ccl_request_impl(const ccl_request_impl&) = delete;
    ccl_request_impl& operator=(const ccl_request_impl&) = delete;

    void wait() override {
        event_.wait();
    }

    bool test() override {
        return event_.test();
    }

private:
    ccl::event event_;
};

class ccl_comm_wrapper {
public:
    explicit ccl_comm_wrapper(ccl::communicator&& comm) : comm_(std::move(comm)) {}
    ccl::communicator& get_ref() {
        return comm_;
    }

private:
    ccl::communicator comm_;
};

class ccl_stream_wrapper {
public:
    explicit ccl_stream_wrapper(ccl::stream&& stream) : stream_(std::move(stream)) {}
    ccl::stream& get_ref() {
        return stream_;
    }

private:
    ccl::stream stream_;
};

/// Implementation of the low-level SPMD communicator interface via ccl
/// TODO: Currently message sizes are limited via `int` type.
///       Large message sizes should be handled on the communicator side in the future.
class ONEDAL_EXPORT ccl_device_communicator_impl : public spmd::communicator_iface {
public:
    // Explicitly declare all virtual functions with overloads to workaround Clang warning
    // https://stackoverflow.com/questions/18515183/c-overloaded-virtual-function-warning-by-clang
    using spmd::communicator_iface::bcast;
    using spmd::communicator_iface::allgatherv;
    using spmd::communicator_iface::allreduce;
    using spmd::communicator_iface::sendrecv_replace;

    template <typename Kvs>
    explicit ccl_device_communicator_impl(const sycl::queue& queue,
                                          std::shared_ptr<Kvs> kvs,
                                          std::int64_t rank,
                                          std::int64_t rank_count)
            : queue_(queue),
              rank_(rank),
              rank_count_(rank_count) {
        auto dev = ccl::create_device(queue_.get_device());
        auto ctx = ccl::create_context(queue_.get_context());
        device_comm_.reset(
            new ccl_comm_wrapper{ ccl::create_communicator(rank_count_, rank_, dev, ctx, kvs) });
        stream_.reset(new ccl_stream_wrapper{ ccl::create_stream(queue_) });
    }

    sycl::queue get_queue() override {
        return queue_;
    }

    spmd::request_iface* bcast(sycl::queue& q,
                               byte_t* send_buf,
                               std::int64_t count,
                               const data_type& dtype,
                               const std::vector<sycl::event>& deps,
                               std::int64_t root) override {
        preview::detail::check_if_pointer_matches_queue(queue_, send_buf);
        preview::detail::check_if_pointer_matches_queue(q, send_buf);
        ONEDAL_ASSERT(root >= 0);

        if (count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(count > 0);

        sycl::event::wait(deps);

        auto event = ccl::broadcast(send_buf,
                                    integral_cast<int>(count),
                                    make_ccl_data_type(dtype),
                                    integral_cast<int>(root),
                                    device_comm_->get_ref(),
                                    stream_->get_ref());
        return new ccl_request_impl{ std::move(event) };
    }

    /// `allgatherv` that accepts USM pointers
    spmd::request_iface* allgatherv(sycl::queue& q,
                                    const byte_t* send_buf,
                                    std::int64_t send_count,
                                    byte_t* recv_buf,
                                    const std::int64_t* recv_counts,
                                    const std::int64_t* displs_host,
                                    const data_type& dtype,
                                    const std::vector<sycl::event>& deps = {}) override {
        preview::detail::check_if_pointer_matches_queue(queue_, send_buf);
        preview::detail::check_if_pointer_matches_queue(queue_, recv_buf);

        ONEDAL_ASSERT(send_buf != nullptr);
        ONEDAL_ASSERT(recv_buf);

        sycl::event::wait(deps);

        std::vector<std::size_t> internal_recv_counts(this->get_rank_count());
        for (std::int64_t i = 0; i < this->get_rank_count(); ++i) {
            internal_recv_counts[i] = integral_cast<std::size_t>(recv_counts[i]);
        }

        auto event = ccl::allgatherv(send_buf,
                                     integral_cast<int>(send_count),
                                     recv_buf,
                                     internal_recv_counts,
                                     make_ccl_data_type(dtype),
                                     device_comm_->get_ref(),
                                     stream_->get_ref());
        return new ccl_request_impl{ std::move(event) };
    }

    /// `allreduce` that accepts USM pointers
    spmd::request_iface* allreduce(sycl::queue& q,
                                   const byte_t* send_buf,
                                   byte_t* recv_buf,
                                   std::int64_t count,
                                   const data_type& dtype,
                                   const spmd::reduce_op& op = spmd::reduce_op::sum,
                                   const std::vector<sycl::event>& deps = {}) override {
        preview::detail::check_if_pointer_matches_queue(queue_, send_buf);
        preview::detail::check_if_pointer_matches_queue(queue_, recv_buf);
        if (count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(recv_buf);

        sycl::event::wait(deps);

        auto event = ccl::allreduce(send_buf,
                                    recv_buf,
                                    integral_cast<int>(count),
                                    make_ccl_data_type(dtype),
                                    make_ccl_reduce_op(op),
                                    device_comm_->get_ref(),
                                    stream_->get_ref());
        return new ccl_request_impl{ std::move(event) };
    }
    /// `sendrecv_replace` that accepts USM pointers
    spmd::request_iface* sendrecv_replace(sycl::queue& q,
                                          byte_t* buf,
                                          std::int64_t count,
                                          const data_type& dtype,
                                          std::int64_t destination_rank,
                                          std::int64_t source_rank,
                                          const std::vector<sycl::event>& deps) override {
        ONEDAL_ASSERT(destination_rank >= 0);
        ONEDAL_ASSERT(source_rank >= 0);
        ONEDAL_ASSERT(destination_rank < rank_count_);
        ONEDAL_ASSERT(source_rank < rank_count_);

        if (count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(buf);
        ONEDAL_ASSERT(count > 0);

        sycl::event::wait(deps);

        std::vector<std::size_t> send_counts(rank_count_, std::size_t(0));
        send_counts.at(destination_rank) = dal::detail::integral_cast<std::size_t>(count);
        std::vector<std::size_t> recv_counts(rank_count_, std::size_t(0));
        recv_counts.at(source_rank) = dal::detail::integral_cast<std::size_t>(count);

        auto event = ccl::alltoallv(buf,
                                    send_counts,
                                    buf,
                                    recv_counts,
                                    make_ccl_data_type(dtype),
                                    device_comm_->get_ref(),
                                    stream_->get_ref());
        return new ccl_request_impl{ std::move(event) };
    }

private:
    std::unique_ptr<ccl_comm_wrapper> device_comm_;
    std::unique_ptr<ccl_stream_wrapper> stream_;
    sycl::queue queue_;
    std::int64_t rank_ = -1;
    std::int64_t rank_count_ = -1;
};
template <typename MemoryAccessKind>
struct ccl_interface_selector {
    using type = spmd::communicator_iface_base;
};

template <>
struct ccl_interface_selector<spmd::device_memory_access::usm> {
    using type = ccl_device_communicator_impl;
};

/// Implementation of the low-level SPMD communicator interface via MPI
/// TODO: Currently message sizes are limited via `int` type.
///       Large message sizes should be handled on the communicator side in the future.
template <typename MemoryAccessKind>
class ccl_communicator_impl : public ccl_interface_selector<MemoryAccessKind>::type {
public:
    // Explicitly declare all virtual functions with overloads to workaround Clang warning
    // https://stackoverflow.com/questions/18515183/c-overloaded-virtual-function-warning-by-clang
    using base_t = typename ccl_interface_selector<MemoryAccessKind>::type;
    using base_t::bcast;
    using base_t::allgatherv;
    using base_t::allreduce;
    using base_t::sendrecv_replace;

    explicit ccl_communicator_impl(ccl::shared_ptr_class<ccl::kvs> kvs,
                                   std::int64_t rank,
                                   std::int64_t rank_count,
                                   std::int64_t default_root = 0)
            : rank_(rank),
              rank_count_(rank_count),
              default_root_(default_root) {
        host_comm_.reset(new ccl_comm_wrapper{ ccl::create_communicator(rank_count_, rank_, kvs) });
    }

    //    template<typename T = MemoryAccessKind, spmd::enable_if_device_memory_accessible_t<T>>
    explicit ccl_communicator_impl(sycl::queue& queue,
                                   ccl::shared_ptr_class<ccl::kvs> kvs,
                                   std::int64_t rank,
                                   std::int64_t rank_count,
                                   std::int64_t default_root = 0)
            : base_t(queue, kvs, rank, rank_count),
              rank_(rank),
              rank_count_(rank_count),
              default_root_(default_root) {
        host_comm_.reset(new ccl_comm_wrapper{ ccl::create_communicator(rank_count_, rank_, kvs) });
    }

    std::int64_t get_rank() override {
        return rank_;
    }
    std::int64_t get_rank_count() override {
        return rank_count_;
    }

    std::int64_t get_default_root_rank() override {
        return default_root_;
    }

    bool get_mpi_offload_support() override {
        auto ccl_backend = ccl::get_library_version().cl_backend_name;
        if (ccl_backend == "DPCPP") {
            return true;
        }
        return false;
    }

    bool use_sendrecv_replace_alternative() override {
        return false;
    }

    void barrier() override {
        ccl::barrier(host_comm_->get_ref()).wait();
    }

    spmd::request_iface* bcast(byte_t* send_buf,
                               std::int64_t count,
                               const data_type& dtype,
                               std::int64_t root) override {
        ONEDAL_ASSERT(root >= 0);

        if (count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(count > 0);

        auto event = ccl::broadcast(send_buf,
                                    integral_cast<int>(count),
                                    make_ccl_data_type(dtype),
                                    integral_cast<int>(root),
                                    host_comm_->get_ref());
        return new ccl_request_impl{ std::move(event) };
    }

    spmd::request_iface* allgatherv(const byte_t* send_buf,
                                    std::int64_t send_count,
                                    byte_t* recv_buf,
                                    const std::int64_t* recv_counts,
                                    const std::int64_t* displs,
                                    const data_type& dtype) override {
        std::vector<std::size_t> internal_recv_counts(rank_count_);
        for (std::int64_t i = 0; i < rank_count_; i++) {
            internal_recv_counts[i] = integral_cast<std::size_t>(recv_counts[i]);
        }

        ONEDAL_ASSERT(recv_buf);

        auto event = ccl::allgatherv(send_buf,
                                     integral_cast<std::size_t>(send_count),
                                     recv_buf,
                                     internal_recv_counts,
                                     make_ccl_data_type(dtype),
                                     host_comm_->get_ref());
        return new ccl_request_impl{ std::move(event) };
    }

    spmd::request_iface* allreduce(const byte_t* send_buf,
                                   byte_t* recv_buf,
                                   std::int64_t count,
                                   const data_type& dtype,
                                   const spmd::reduce_op& op) override {
        if (count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(recv_buf);

        auto event = ccl::allreduce(send_buf,
                                    recv_buf,
                                    integral_cast<int>(count),
                                    make_ccl_data_type(dtype),
                                    make_ccl_reduce_op(op),
                                    host_comm_->get_ref());
        return new ccl_request_impl{ std::move(event) };
    }
    spmd::request_iface* sendrecv_replace(byte_t* buf,
                                          std::int64_t count,
                                          const data_type& dtype,
                                          std::int64_t destination_rank,
                                          std::int64_t source_rank,
                                          byte_t* recv_buf = nullptr) override {
        ONEDAL_ASSERT(destination_rank >= 0);
        ONEDAL_ASSERT(source_rank >= 0);
        ONEDAL_ASSERT(destination_rank < rank_count_);
        ONEDAL_ASSERT(source_rank < rank_count_);

        if (count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(buf);
        ONEDAL_ASSERT(count > 0);

        std::vector<std::size_t> send_counts(rank_count_, std::size_t(0));
        send_counts.at(destination_rank) = dal::detail::integral_cast<std::size_t>(count);
        std::vector<std::size_t> recv_counts(rank_count_, std::size_t(0));
        recv_counts.at(source_rank) = dal::detail::integral_cast<std::size_t>(count);

        auto event = ccl::alltoallv(buf,
                                    send_counts,
                                    buf,
                                    recv_counts,
                                    make_ccl_data_type(dtype),
                                    host_comm_->get_ref());
        return new ccl_request_impl{ std::move(event) };
    }

private:
    std::unique_ptr<ccl_comm_wrapper> host_comm_;
    std::int64_t rank_ = -1;
    std::int64_t rank_count_ = -1;
    std::int64_t default_root_ = -1;
};

template <typename MemoryAccessKind>
class ccl_communicator : public spmd::communicator<MemoryAccessKind> {
public:
    template <typename T = MemoryAccessKind,
              typename = spmd::enable_if_device_memory_accessible_t<T>>
    explicit ccl_communicator(sycl::queue& queue,
                              ccl::shared_ptr_class<ccl::kvs> kvs,
                              std::int64_t rank,
                              std::int64_t rank_count,
                              std::int64_t default_root = 0)
            : spmd::communicator<MemoryAccessKind>(
                  new ccl_communicator_impl<MemoryAccessKind>(queue,
                                                              kvs,
                                                              rank,
                                                              rank_count,
                                                              default_root)) {}
    explicit ccl_communicator(ccl::shared_ptr_class<ccl::kvs> kvs,
                              std::int64_t rank,
                              std::int64_t rank_count,
                              std::int64_t default_root = 0)
            : spmd::communicator<MemoryAccessKind>(
                  new ccl_communicator_impl<MemoryAccessKind>(kvs,
                                                              rank,
                                                              rank_count,
                                                              default_root)) {}
};

} // namespace v1

using v1::ccl_communicator;

} // namespace oneapi::dal::detail
#endif
