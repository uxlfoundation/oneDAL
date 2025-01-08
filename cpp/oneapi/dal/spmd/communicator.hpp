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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/spmd/common.hpp"
#include "oneapi/dal/array.hpp"

namespace oneapi::dal::preview::spmd {

namespace v1 {

class request_iface {
public:
    virtual ~request_iface() = default;
    virtual void wait() = 0;
    virtual bool test() = 0;
};

class request : public base {
    friend dal::detail::pimpl_accessor;

public:
    request() : impl_(nullptr) {}

    void wait() {
        if (impl_) {
            impl_->wait();
        }
    }

    bool test() {
        if (impl_) {
            return impl_->test();
        }
        return true;
    }

private:
    explicit request(request_iface* impl) : impl_(impl) {}
    dal::detail::pimpl<request_iface> impl_;
};

class communicator_iface_base {
public:
    virtual ~communicator_iface_base() = default;

    virtual std::int64_t get_rank() = 0;
    virtual std::int64_t get_rank_count() = 0;
    virtual std::int64_t get_default_root_rank() = 0;
    virtual bool get_mpi_offload_support() = 0;
    virtual bool use_sendrecv_replace_alternative() = 0;

    virtual void barrier() = 0;

    virtual request_iface* bcast(byte_t* send_buf,
                                 std::int64_t count,
                                 const data_type& dtype,
                                 std::int64_t root) = 0;
    virtual request_iface* allgatherv(const byte_t* send_buf,
                                      std::int64_t send_count,
                                      byte_t* recv_buf,
                                      const std::int64_t* recv_counts_host,
                                      const std::int64_t* displs_host,
                                      const data_type& dtype) = 0;
    virtual request_iface* allreduce(const byte_t* send_buf,
                                     byte_t* recv_buf,
                                     std::int64_t count,
                                     const data_type& dtype,
                                     const reduce_op& op) = 0;
    virtual request_iface* sendrecv_replace(byte_t* buf,
                                            std::int64_t count,
                                            const data_type& dtype,
                                            std::int64_t destination_rank,
                                            std::int64_t source_rank,
                                            byte_t* recv_buf = nullptr) = 0;
};

template <typename MemoryAccessKind>
struct interface_selector {
    using type = communicator_iface_base;
};

#ifdef ONEDAL_DATA_PARALLEL
class communicator_iface : public communicator_iface_base {
public:
    using base_t = communicator_iface_base;
    using base_t::get_rank;
    using base_t::get_rank_count;
    using base_t::get_default_root_rank;
    using base_t::barrier;
    using base_t::bcast;
    using base_t::allgatherv;
    using base_t::allreduce;
    using base_t::sendrecv_replace;

    virtual request_iface* bcast(sycl::queue& q,
                                 byte_t* send_buf,
                                 std::int64_t count,
                                 const data_type& dtype,
                                 const std::vector<sycl::event>& deps,
                                 std::int64_t root) = 0;
    virtual request_iface* allgatherv(sycl::queue& q,
                                      const byte_t* send_buf,
                                      std::int64_t send_count,
                                      byte_t* recv_buf,
                                      const std::int64_t* recv_counts_host,
                                      const std::int64_t* displs_host,
                                      const data_type& dtype,
                                      const std::vector<sycl::event>& deps) = 0;
    virtual request_iface* allreduce(sycl::queue& q,
                                     const byte_t* send_buf,
                                     byte_t* recv_buf,
                                     std::int64_t count,
                                     const data_type& dtype,
                                     const reduce_op& op,
                                     const std::vector<sycl::event>& deps) = 0;
    virtual request_iface* sendrecv_replace(sycl::queue& q,
                                            byte_t* buf,
                                            std::int64_t count,
                                            const data_type& dtype,
                                            std::int64_t destination_rank,
                                            std::int64_t source_rank,
                                            const std::vector<sycl::event>& deps) = 0;
    virtual sycl::queue get_queue() = 0;
};

template <>
struct interface_selector<device_memory_access::usm> {
    using type = communicator_iface;
};

#endif

/// Low-level MPI-like communicator
template <typename MemoryAccessKind>
class communicator : public base {
private:
    template <typename T>
    static constexpr bool is_primitive_v = std::is_arithmetic_v<T>;

    template <typename T>
    using enable_if_primitive_t = std::enable_if_t<is_primitive_v<T>>;

    using interface_type = typename interface_selector<MemoryAccessKind>::type;

public:
    std::int64_t get_rank() const {
        return impl_->get_rank();
    }
    std::int64_t get_rank_count() const {
        return impl_->get_rank_count();
    }
    std::int64_t get_default_root_rank() const {
        return impl_->get_default_root_rank();
    }

    /// Returns `true` if the current rank is root
    bool is_root_rank(std::int64_t root = -1) const {
        return get_rank() == fix_root_rank(root);
    }
    std::int64_t fix_root_rank(std::int64_t root) const {
        if (root < 0) {
            return get_default_root_rank();
        }
        return root;
    }

    /// Blocks until all ranks in the communicator have reached this function
    void barrier() const {
        wait_for_exception_handling();
        impl_->barrier();
    }

    /// Broadcasts a message from the `root` rank to all other ranks
    ///
    /// @param buf   The buffer which content is broadcasted
    /// @param count The number of elements of `dtype` in `send_buf`
    /// @param dtype The type of elements in the passed buffers
    /// @param root  The rank of the broadcasting process, if the passed
    ///              rank is negative, the default root rank is used
    ///
    /// @return The object to track the progress of the operation
    request bcast(byte_t* buf,
                  std::int64_t count,
                  const data_type& dtype,
                  std::int64_t root = -1) const {
        wait_for_exception_handling();
        return dal::detail::make_private<request>(
            impl_->bcast(buf, count, dtype, fix_root_rank(root)));
    }
#ifdef ONEDAL_DATA_PARALLEL
    /// `bcast` that accepts USM pointers
    template <typename T = MemoryAccessKind, typename = enable_if_device_memory_accessible_t<T>>
    request bcast(sycl::queue& queue,
                  byte_t* buf,
                  std::int64_t count,
                  const data_type& dtype,
                  const std::vector<sycl::event>& deps = {},
                  std::int64_t root = -1) const {
        wait_for_exception_handling();
        return dal::detail::make_private<request>(
            impl_->bcast(queue, buf, count, dtype, deps, fix_root_rank(root)));
    }
#endif
    template <typename D, typename = enable_if_primitive_t<D>>
    request bcast(D* buf, std::int64_t count, std::int64_t root = -1) const {
        auto ret =
            bcast(reinterpret_cast<byte_t*>(buf), count, dal::detail::make_data_type<D>(), root);
        return ret;
    }
#ifdef ONEDAL_DATA_PARALLEL
    template <typename D,
              typename T = MemoryAccessKind,
              typename = std::enable_if_t<dal::detail::is_one_of_v<T, device_memory_access::usm> &&
                                          is_primitive_v<D>>>
    request bcast(sycl::queue& q,
                  D* buf,
                  std::int64_t count,
                  const std::vector<sycl::event>& deps = {},
                  std::int64_t root = -1) const {
        return bcast(q,
                     reinterpret_cast<byte_t*>(buf),
                     count,
                     dal::detail::make_data_type<D>(),
                     deps,
                     root);
    }
#endif
    template <typename D, typename = enable_if_primitive_t<D>>
    request bcast(D& value, std::int64_t root = -1) const {
        return bcast(&value, 1, root);
    }
    template <typename D>
    request bcast(const array<D>& ary, std::int64_t root = -1) const;

    /// Gathers data from all ranks and distributes the results back to all ranks
    ///
    /// @param send   The send buffer
    /// @param recv   The receiving buffer
    /// @tparam D     The type of elements in the passed buffers
    ///
    /// @return The object to track the progress of the operation
    template <typename D>
    request allgather(const array<D>& send, const array<D>& recv) const;

    /// Gathers data from all ranks and distributes the results back to all ranks
    ///
    /// @param scalar The send element
    /// @param recv   The receiving buffer
    /// @tparam D     The type of elements in send / recv
    ///
    /// @return The object to track the progress of the operation
    template <typename D>
    request allgather(const D& scalar, const array<D>& recv) const;
    /// Collects data from all the ranks within a communicator into a single buffer
    /// and redistribute to all ranks.
    /// The data size send by each rank may be different.
    ///
    /// @param send_buf   The send buffer
    /// @param send_count The number of elements of `dtype` in `send_buf`
    /// @param recv_buf   The receiveing buffer, must contain at least
    ///                   `rank_count * recv_count` elements,
    ///                   significant only at `root`
    /// @param recv_counts The number of elements of `dtype` received from
    ///                    each rank, must contain at least `rank_count` elements,
    ///                    significant only at `root`
    /// @param displs     Entry $i$ specifies the displacement relative to
    ///                   `recv_buf` at which to place the incoming data
    ///                   from process $i$, must contain at least `rank_count`
    ///                   elements, significant only at `root`
    /// @param dtype      The type of elements in the passed buffers
    ///
    /// @return The object to track the progress of the operation
    request allgatherv(const byte_t* send_buf,
                       std::int64_t send_count,
                       byte_t* recv_buf,
                       const std::int64_t* recv_counts,
                       const std::int64_t* displs,
                       const data_type& dtype) const {
        wait_for_exception_handling();
        return dal::detail::make_private<request>(
            impl_->allgatherv(send_buf, send_count, recv_buf, recv_counts, displs, dtype));
    }
#ifdef ONEDAL_DATA_PARALLEL
    /// `allgatherv` that accepts USM pointers
    template <typename T = MemoryAccessKind, typename = enable_if_device_memory_accessible_t<T>>
    request allgatherv(sycl::queue& queue,
                       const byte_t* send_buf,
                       std::int64_t send_count,
                       byte_t* recv_buf,
                       const std::int64_t* recv_counts,
                       const std::int64_t* displs,
                       const data_type& dtype,
                       const std::vector<sycl::event>& deps = {}) const {
        wait_for_exception_handling();
        return dal::detail::make_private<request>(impl_->allgatherv(queue,
                                                                    send_buf,
                                                                    send_count,
                                                                    recv_buf,
                                                                    recv_counts,
                                                                    displs,
                                                                    dtype,
                                                                    deps));
    }
#endif
    template <typename D, enable_if_primitive_t<D>* = nullptr>
    request allgatherv(const D* send_buf,
                       std::int64_t send_count,
                       D* recv_buf,
                       const std::int64_t* recv_counts,
                       const std::int64_t* displs) const {
        return allgatherv(reinterpret_cast<const byte_t*>(send_buf),
                          send_count,
                          reinterpret_cast<byte_t*>(recv_buf),
                          recv_counts,
                          displs,
                          dal::detail::make_data_type<D>());
    }
#ifdef ONEDAL_DATA_PARALLEL
    template <typename D,
              typename T = MemoryAccessKind,
              typename = std::enable_if_t<dal::detail::is_one_of_v<T, device_memory_access::usm> &&
                                          is_primitive_v<D>>>
    request allgatherv(sycl::queue& queue,
                       const D* send_buf,
                       std::int64_t send_count,
                       D* recv_buf,
                       const std::int64_t* recv_counts,
                       const std::int64_t* displs,
                       const std::vector<sycl::event>& deps = {}) const {
        return allgatherv(queue,
                          reinterpret_cast<const byte_t*>(send_buf),
                          send_count,
                          reinterpret_cast<byte_t*>(recv_buf),
                          recv_counts,
                          displs,
                          dal::detail::make_data_type<D>(),
                          deps);
    }
#endif
    template <typename D>
    request allgatherv(const array<D>& send,
                       const array<D>& recv,
                       const std::int64_t* recv_counts,
                       const std::int64_t* displs) const;
    /// Combines data from all ranks using reduction operation and
    /// distributes the result back to all ranks
    ///
    /// @param send_buf The send buffer
    /// @param recv_buf The receiving buffer
    /// @param count    The number of elements of `dtype` sent to and
    ///                 received from each rank
    /// @param dtype    The type of elements in the passed buffers
    /// @param op       The reduction operation
    ///
    /// @return The object to track the progress of the operation
    request allreduce(const byte_t* send_buf,
                      byte_t* recv_buf,
                      std::int64_t count,
                      const data_type& dtype,
                      const reduce_op& op) const {
        wait_for_exception_handling();
        return dal::detail::make_private<request>(
            impl_->allreduce(send_buf, recv_buf, count, dtype, op));
    }
#ifdef ONEDAL_DATA_PARALLEL
    /// `allreduce` that accepts USM pointers
    template <typename T = MemoryAccessKind, typename = enable_if_device_memory_accessible_t<T>>
    request allreduce(sycl::queue& queue,
                      const byte_t* send_buf,
                      byte_t* recv_buf,
                      std::int64_t count,
                      const data_type& dtype,
                      const reduce_op& op,
                      const std::vector<sycl::event>& deps = {}) const {
        wait_for_exception_handling();
        return dal::detail::make_private<request>(
            impl_->allreduce(queue, send_buf, recv_buf, count, dtype, op, deps));
    }
#endif
    template <typename D, enable_if_primitive_t<D>* = nullptr>
    request allreduce(const D* send_buf,
                      D* recv_buf,
                      std::int64_t count,
                      const reduce_op& op) const {
        return allreduce(reinterpret_cast<const byte_t*>(send_buf),
                         reinterpret_cast<byte_t*>(recv_buf),
                         count,
                         dal::detail::make_data_type<D>(),
                         op);
    }
#ifdef ONEDAL_DATA_PARALLEL
    template <typename D,
              typename T = MemoryAccessKind,
              typename = std::enable_if_t<dal::detail::is_one_of_v<T, device_memory_access::usm> &&
                                              is_primitive_v<D>,
                                          bool>>
    request allreduce(sycl::queue& queue,
                      const D* send_buf,
                      D* recv_buf,
                      std::int64_t count,
                      const reduce_op& op,
                      const std::vector<sycl::event>& deps = {}) const {
        return allreduce(queue,
                         reinterpret_cast<const byte_t*>(send_buf),
                         reinterpret_cast<byte_t*>(recv_buf),
                         count,
                         dal::detail::make_data_type<D>(),
                         op,
                         deps);
    }
#endif
    template <typename D, typename = enable_if_primitive_t<D>>
    request allreduce(D& scalar, const reduce_op& op = reduce_op::sum) const {
        return allreduce(&scalar, &scalar, 1, op);
    }
    template <typename D>
    request allreduce(const array<D>& ary, const reduce_op& op = reduce_op::sum) const;
    /// Shuffles data reusing the same buffer for send and receive operations
    ///
    /// @param buf                  The buffer
    /// @param count                The number of elements of `dtype` sent to and
    ///                             received from for each rank
    /// @param dtype                The type of elements in the passed buffers
    /// @param destination_rank     The rank to send data to.
    /// @param source_rank          The rank to receive data from.
    ///
    /// @return The object to track the progress of the operation
    request sendrecv_replace(byte_t* buf,
                             std::int64_t count,
                             const data_type& dtype,
                             std::int64_t destination_rank,
                             std::int64_t source_rank) const {
        wait_for_exception_handling();
        return dal::detail::make_private<request>(
            impl_->sendrecv_replace(buf, count, dtype, destination_rank, source_rank));
    }
#ifdef ONEDAL_DATA_PARALLEL
    /// `sendrecv_replace` that accepts USM pointers
    request sendrecv_replace(sycl::queue& q,
                             byte_t* buf,
                             std::int64_t count,
                             const data_type& dtype,
                             std::int64_t destination_rank,
                             std::int64_t source_rank,
                             const std::vector<sycl::event>& deps = {}) const {
        wait_for_exception_handling();
        return dal::detail::make_private<request>(
            impl_->sendrecv_replace(q, buf, count, dtype, destination_rank, source_rank, deps));
    }
#endif
    template <typename D, enable_if_primitive_t<D>* = nullptr>
    request sendrecv_replace(D* buf,
                             std::int64_t count,
                             std::int64_t destination_rank,
                             std::int64_t source_rank) const {
        return sendrecv_replace(reinterpret_cast<byte_t*>(buf),
                                count,
                                dal::detail::make_data_type<D>(),
                                destination_rank,
                                source_rank);
    }
#ifdef ONEDAL_DATA_PARALLEL
    template <typename D,
              typename T = MemoryAccessKind,
              typename = std::enable_if_t<dal::detail::is_one_of_v<T, device_memory_access::usm> &&
                                          is_primitive_v<D>>>
    request sendrecv_replace(sycl::queue& queue,
                             D* buf,
                             std::int64_t count,
                             std::int64_t destination_rank,
                             std::int64_t source_rank,
                             const std::vector<sycl::event>& deps = {}) const {
        return sendrecv_replace(queue,
                                reinterpret_cast<byte_t*>(buf),
                                count,
                                dal::detail::make_data_type<D>(),
                                destination_rank,
                                source_rank,
                                deps);
    }
#endif
    template <typename D>
    request sendrecv_replace(const array<D>& buf,
                             std::int64_t destination_rank,
                             std::int64_t source_rank) const;
#ifdef ONEDAL_DATA_PARALLEL
    template <typename T = MemoryAccessKind, typename = enable_if_device_memory_accessible_t<T>>
    sycl::queue get_queue() const {
        return impl_->get_queue();
    }
#endif

    void set_active_exception(const std::exception_ptr& ex_ptr) const;
    void wait_for_exception_handling() const;

protected:
    template <typename Impl>
    Impl& get_impl() const {
        return static_cast<Impl&>(*impl_);
    }
    explicit communicator(interface_type* impl) : impl_(impl) {}
    dal::detail::pimpl<interface_type> impl_;

private:
    void reset_error_flag() const;

    mutable std::int32_t error_flag_ = 0;
    mutable std::exception_ptr active_exception_;
};

} // namespace v1

using v1::request_iface;
using v1::request;
using v1::communicator_iface_base;
#ifdef ONEDAL_DATA_PARALLEL
using v1::communicator_iface;
#endif
using v1::communicator;

template <typename Backend>
communicator<device_memory_access::none> make_communicator() {
    static_assert(!std::is_same_v<Backend, Backend>, "Unsupported communicator backend");

    throw communication_error(dal::detail::error_messages::unsupported_communicator_backend());
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename Backend>
communicator<device_memory_access::usm> make_communicator(sycl::queue& queue) {
    static_assert(!std::is_same_v<Backend, Backend>, "Unsupported communicator backend");

    throw communication_error(dal::detail::error_messages::unsupported_communicator_backend());
}
#endif

} // namespace oneapi::dal::preview::spmd
