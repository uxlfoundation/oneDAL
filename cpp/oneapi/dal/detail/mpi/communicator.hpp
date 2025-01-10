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

// IMPORTANT! This file should not be included to any other non-mpi header,
// otherwise it creates dependeny on MPI at the user's application compile time
// TODO: In the future this can be solved via __has_include C++17 feature

#include <mpi.h>
#include <dlfcn.h>
#include <string>
#include <sstream>
#include <oneapi/dal/array.hpp>
#include "oneapi/dal/detail/communicator.hpp"

namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::detail {
namespace v1 {

using preview::spmd::communication_error;

inline void mpi_call(int mpi_status) {
    if (mpi_status == MPI_SUCCESS) {
        return;
    }
    switch (mpi_status) {
        case MPI_ERR_BUFFER:
            throw communication_error(dal::detail::error_messages::invalid_buffer());
        case MPI_ERR_COUNT: throw communication_error(dal::detail::error_messages::invalid_count());
        case MPI_ERR_TYPE:
            throw communication_error(dal::detail::error_messages::invalid_data_type());
        case MPI_ERR_OP: throw communication_error(dal::detail::error_messages::invalid_op());
        case MPI_ERR_COMM:
            throw communication_error(dal::detail::error_messages::invalid_mpi_comm());
        case MPI_ERR_ROOT: throw communication_error(dal::detail::error_messages::invalid_root());
        default: throw internal_error(dal::detail::error_messages::unknown_mpi_error());
    }
}

inline MPI_Datatype make_mpi_data_type(const data_type& dtype) {
    switch (dtype) {
        case data_type::int8: return MPI_INT8_T;
        case data_type::int16: return MPI_INT16_T;
        case data_type::int32: return MPI_INT32_T;
        case data_type::int64: return MPI_INT64_T;
        case data_type::uint8: return MPI_UINT8_T;
        case data_type::uint16: return MPI_UINT16_T;
        case data_type::uint32: return MPI_UINT32_T;
        case data_type::uint64: return MPI_UINT64_T;
        case data_type::float32: return MPI_FLOAT;
        case data_type::float64: return MPI_DOUBLE;
        case data_type::bfloat16: return MPI_UINT16_T;
        default: ONEDAL_ASSERT(!"Unknown data type");
    }
    return MPI_DATATYPE_NULL;
}

inline MPI_Op make_mpi_reduce_op(const preview::spmd::reduce_op& op) {
    switch (op) {
        case spmd::reduce_op::max: return MPI_MAX;
        case spmd::reduce_op::min: return MPI_MIN;
        case spmd::reduce_op::sum: return MPI_SUM;
        default: ONEDAL_ASSERT(!"Unknown reduce operation");
    }
    return MPI_OP_NULL;
}

class mpi_request_impl : public preview::spmd::request_iface {
public:
    explicit mpi_request_impl(const MPI_Request& request) : mpi_request_(request) {}

    mpi_request_impl(const mpi_request_impl&) = delete;
    mpi_request_impl& operator=(const mpi_request_impl&) = delete;

    void wait() override {
        mpi_call(MPI_Wait(&mpi_request_, MPI_STATUS_IGNORE));
    }

    bool test() override {
        int flag;
        mpi_call(MPI_Test(&mpi_request_, &flag, MPI_STATUS_IGNORE));
        return bool(flag != 0);
    }

private:
    MPI_Request mpi_request_;
};

/// Implementation of the low-level SPMD communicator interface via MPI
/// TODO: Currently message sizes are limited via `int` type.
///       Large message sizes should be handled on the communicator side in the future.
template <typename MemoryAccessKind>
class mpi_communicator_impl : public via_host_interface_selector<MemoryAccessKind>::type {
public:
    // Explicitly declare all virtual functions with overloads to workaround Clang warning
    // https://stackoverflow.com/questions/18515183/c-overloaded-virtual-function-warning-by-clang
    using base_t = typename via_host_interface_selector<MemoryAccessKind>::type;
    using base_t::bcast;
    using base_t::allgatherv;
    using base_t::allreduce;
    using base_t::sendrecv_replace;

    explicit mpi_communicator_impl(std::int64_t default_root = 0)
            : mpi_comm_(MPI_COMM_WORLD),
              default_root_(default_root) {}

#ifdef ONEDAL_DATA_PARALLEL
    //    template<typename T = MemoryAccessKind, spmd::enable_if_device_memory_accessible_t<T>>
    explicit mpi_communicator_impl(sycl::queue& queue, std::int64_t default_root = 0)
            : base_t(queue),
              mpi_comm_(MPI_COMM_WORLD),
              default_root_(default_root) {}
#endif

    std::int64_t get_rank() override {
        if (rank_ < 0) {
            int rank;
            mpi_call(MPI_Comm_rank(mpi_comm_, &rank));
            rank_ = rank;
        }
        return rank_;
    }

    std::int64_t get_rank_count() override {
        if (rank_count_ < 0) {
            int rank_count;
            mpi_call(MPI_Comm_size(mpi_comm_, &rank_count));
            rank_count_ = rank_count;
        }
        return rank_count_;
    }

    std::int64_t get_default_root_rank() override {
        return default_root_;
    }

    bool get_mpi_offload_support() override {
        // TODO: explore additional conditions (i.e. GPU type, data parallel, etc.)
        // TODO: make this function bool convert() and instead determine if conversion necessary?

        // Check libmpi.so for symbol
        void* handle = dlopen("libmpi.so", RTLD_LAZY);

        if (handle == nullptr) {
            return false;
        }

        void* sym = dlsym(handle, "MPIX_Query_ze_support");

        if (sym == nullptr) {
            dlclose(handle);

            // Check if Intel MPI without MPIX_Query_ze_support
            // Intel MPI 2021.1 and greater supports but not all have symbol
            char version[MPI_MAX_LIBRARY_VERSION_STRING];
            int len = 0;
            MPI_Get_library_version(version, &len);
            std::string version_str(version);
            if (version_str.compare(0, 5, "Intel") != 0) {
                return false;
            }

            // Extract major version based on location of period
            size_t period_pos = version_str.find('.');
            std::string major_str = version_str.substr(period_pos - 4, period_pos);

            // Convert the substring to an integer
            int major;
            std::istringstream(major_str) >> major;

            return (major >= 2021);
        }

        // Return status of MPI ze support using pointer to function
        typedef int (*MPIX_Query_ze_support_ptr)();
        MPIX_Query_ze_support_ptr query_ze_support_ptr = (MPIX_Query_ze_support_ptr)sym;

        bool result = query_ze_support_ptr();
        dlclose(handle);

        return result;
    }

    bool use_sendrecv_replace_alternative() override {
        char version[MPI_MAX_LIBRARY_VERSION_STRING];
        int len = 0;
        MPI_Get_library_version(version, &len);
        std::string version_str(version);
        if (version_str.compare(0, 5, "MPICH") == 0) {
            return true;
        }
        else {
            return false;
        }
    }

    void barrier() override {
        mpi_call(MPI_Barrier(mpi_comm_));
    }

    // void is_init(int * flag) override {
    //     mpi_call(MPI_Initialized(flag));
    // }

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

        // TODO replace with MPI_Ibcast
        mpi_call(MPI_Bcast(send_buf,
                           integral_cast<int>(count),
                           make_mpi_data_type(dtype),
                           integral_cast<int>(root),
                           mpi_comm_));
        return nullptr;
    }

    spmd::request_iface* allgatherv(const byte_t* send_buf,
                                    std::int64_t send_count,
                                    byte_t* recv_buf,
                                    const std::int64_t* recv_counts,
                                    const std::int64_t* displs,
                                    const data_type& dtype) override {
        ONEDAL_ASSERT(send_buf);
        ONEDAL_ASSERT(recv_counts);
        ONEDAL_ASSERT(displs);
        ONEDAL_ASSERT(recv_buf);

        array<int> recv_counts_int;
        array<int> displs_int;

        const std::int64_t rank_count = get_rank_count();
        recv_counts_int.reset(rank_count);
        displs_int.reset(rank_count);

        auto recv_counts_int_ptr = recv_counts_int.get_mutable_data();
        auto displs_int_ptr = displs_int.get_mutable_data();

        [[maybe_unused]] std::int64_t displs_counter = 0;
        for (std::int64_t i = 0; i < rank_count; ++i) {
            ONEDAL_ASSERT(recv_counts[i] > 0);
            ONEDAL_ASSERT(displs[i] >= displs_counter);
            displs_counter += recv_counts[i];

            recv_counts_int_ptr[i] = dal::detail::integral_cast<int>(recv_counts[i]);
            displs_int_ptr[i] = dal::detail::integral_cast<int>(displs[i]);
        }

        MPI_Request mpi_request;
        mpi_call(MPI_Iallgatherv(send_buf,
                                 integral_cast<int>(send_count),
                                 make_mpi_data_type(dtype),
                                 recv_buf,
                                 recv_counts_int.get_data(),
                                 displs_int.get_data(),
                                 make_mpi_data_type(dtype),
                                 mpi_comm_,
                                 &mpi_request));

        return new mpi_request_impl{ mpi_request };
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

        // Intel(R) MPI requires buffers to be not aliased
        // However, communicator interface allows aliased buffers
        // TODO: Implement correct aliasing check
        if (send_buf != recv_buf) {
            MPI_Request mpi_request;
            mpi_call(MPI_Iallreduce(send_buf,
                                    recv_buf,
                                    integral_cast<int>(count),
                                    make_mpi_data_type(dtype),
                                    make_mpi_reduce_op(op),
                                    mpi_comm_,
                                    &mpi_request));
            return new mpi_request_impl{ mpi_request };
        }
        else {
            // TODO Replace with MPI_Iallreduce
            mpi_call(MPI_Allreduce(MPI_IN_PLACE,
                                   recv_buf,
                                   integral_cast<int>(count),
                                   make_mpi_data_type(dtype),
                                   make_mpi_reduce_op(op),
                                   mpi_comm_));
            return nullptr;
        }
    }

    spmd::request_iface* sendrecv_replace(byte_t* buf,
                                          std::int64_t count,
                                          const data_type& dtype,
                                          std::int64_t destination_rank,
                                          std::int64_t source_rank,
                                          byte_t* recv_buf = nullptr) override {
        ONEDAL_ASSERT(destination_rank >= 0);
        ONEDAL_ASSERT(source_rank >= 0);

        if (count == 0) {
            return nullptr;
        }

        ONEDAL_ASSERT(buf);
        ONEDAL_ASSERT(count > 0);

        MPI_Status status;
        constexpr int zero_tag = 0;

        if (recv_buf) {
            // MPICH-specific workaround for GPU performance
            mpi_call(MPI_Sendrecv(buf,
                                  integral_cast<int>(count),
                                  make_mpi_data_type(dtype),
                                  integral_cast<int>(destination_rank),
                                  zero_tag,
                                  recv_buf,
                                  integral_cast<int>(count),
                                  make_mpi_data_type(dtype),
                                  integral_cast<int>(source_rank),
                                  zero_tag,
                                  mpi_comm_,
                                  &status));
        }
        else {
            // Standard call to sendrecv_replace of designated mpi backend
            mpi_call(MPI_Sendrecv_replace(buf,
                                          integral_cast<int>(count),
                                          make_mpi_data_type(dtype),
                                          integral_cast<int>(destination_rank),
                                          zero_tag,
                                          integral_cast<int>(source_rank),
                                          zero_tag,
                                          mpi_comm_,
                                          &status));
        }
        return nullptr;
    }

private:
    MPI_Comm mpi_comm_;
    std::int64_t default_root_;
    std::int64_t rank_ = -1;
    std::int64_t rank_count_ = -1;
};

template <typename MemoryAccessKind>
class mpi_communicator : public spmd::communicator<MemoryAccessKind> {
public:
#ifdef ONEDAL_DATA_PARALLEL
    template <typename T = MemoryAccessKind,
              typename = spmd::enable_if_device_memory_accessible_t<T>>
    explicit mpi_communicator(sycl::queue& queue, std::int64_t default_root = 0)
            : spmd::communicator<MemoryAccessKind>(
                  new mpi_communicator_impl<MemoryAccessKind>(queue, default_root)) {}
#endif
    explicit mpi_communicator(std::int64_t default_root = 0)
            : spmd::communicator<MemoryAccessKind>(
                  new mpi_communicator_impl<MemoryAccessKind>(default_root)) {}
};

} // namespace v1

using v1::mpi_communicator;

} // namespace oneapi::dal::detail
