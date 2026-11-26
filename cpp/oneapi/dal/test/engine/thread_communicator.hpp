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

#include <cstdio>
#include <exception>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <condition_variable>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/communicator.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::test::engine {

class thread_communicator_context : public base {
public:
    explicit thread_communicator_context(std::int64_t thread_count) : thread_count_(thread_count) {
        ONEDAL_ASSERT(thread_count > 0);
        thread_pool_.reserve(thread_count);
        thread_id_map_.reserve(thread_count);
    }

    std::int64_t get_root_rank() const {
        return 0;
    }

    std::int64_t get_this_thread_rank() const {
        return map_thread_id_to_rank(std::this_thread::get_id());
    }

    std::int64_t map_thread_id_to_rank(const std::thread::id& thread_id) const {
        return thread_id_map_.at(thread_id);
    }

    std::int64_t get_thread_count() const {
        return thread_count_;
    }

    template <typename Body>
    void execute(const Body& body) {
        // An exception escaping a std::thread calls std::terminate and loses its message:
        // print it at once (a peer may then wait in a barrier), rethrow it after the join.
        std::vector<std::exception_ptr> errors(thread_count_);
        for (std::int64_t i = 0; i < thread_count_; i++) {
            thread_pool_.emplace_back([=, &errors]() {
                try {
                    init(i);

                    std::unique_lock lock(serializing_lock_);
                    body(i);
                }
                catch (const std::exception& e) {
                    std::fprintf(stderr, "rank %lld: %s\n", static_cast<long long>(i), e.what());
                    std::fflush(stderr);
                    errors[i] = std::current_exception();
                }
                catch (...) {
                    std::fprintf(stderr,
                                 "rank %lld: unknown exception\n",
                                 static_cast<long long>(i));
                    std::fflush(stderr);
                    errors[i] = std::current_exception();
                }
            });
        }

        for (auto& thread : thread_pool_) {
            thread.join();
        }
        for (const auto& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
    }

    template <typename Body>
    void exclusive(const Body& body) {
        std::unique_lock<std::mutex> lock(internal_lock_);
        body();
    }

    void enter_communication() {
        serializing_lock_.unlock();
    }

    void exit_communication() {
        serializing_lock_.lock();
    }

private:
    /// Blocks until all threads are mapped to ranks
    void init(std::int64_t rank);

    std::int64_t thread_count_;
    std::vector<std::thread> thread_pool_;
    std::unordered_map<std::thread::id, std::int64_t> thread_id_map_;
    std::mutex internal_lock_;
    std::mutex serializing_lock_;
    std::condition_variable cv_;
};

class thread_communicator_barrier {
public:
    explicit thread_communicator_barrier(thread_communicator_context& ctx)
            : ctx_(ctx),
              thread_counter_(ctx.get_thread_count()),
              generation_counter_(0) {}

    template <typename UniqueOp>
    void operator()(UniqueOp&& unique_op) {
        std::unique_lock<std::mutex> lock(mutex_);
        const std::uint64_t thread_generation = generation_counter_;

        if (--thread_counter_ == 0) {
            // This line may lead to overflow of the counter,
            // but this is a part of barrier design. Flip to zero
            // is safe here as counter is `uint64_t`.
            generation_counter_++;

            thread_counter_ = ctx_.get_thread_count();
            unique_op();
            cv_.notify_all();
        }
        else {
            cv_.wait(lock, [=] {
                return thread_generation != generation_counter_;
            });
        }
    }

    void operator()() {
        (*this)([]() {});
    }

private:
    thread_communicator_context& ctx_;
    std::int64_t thread_counter_;
    std::uint64_t generation_counter_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

class thread_communicator_bcast {
public:
    explicit thread_communicator_bcast(thread_communicator_context& ctx)
            : ctx_(ctx),
              barrier_(ctx),
              source_count_(0),
              source_buf_(nullptr) {}

    void operator()(byte_t* send_buf,
                    std::int64_t count,
                    const data_type& dtype,
                    std::int64_t root);

private:
    thread_communicator_context& ctx_;
    thread_communicator_barrier barrier_;
    std::int64_t source_count_;
    byte_t* source_buf_;
};

class thread_communicator_gather {
public:
    explicit thread_communicator_gather(thread_communicator_context& ctx)
            : ctx_(ctx),
              barrier_(ctx),
              recv_count_(0),
              recv_buf_(nullptr) {}

    void operator()(const byte_t* send_buf,
                    std::int64_t send_count,
                    byte_t* recv_buf,
                    std::int64_t recv_count,
                    const data_type& dtype,
                    std::int64_t root);

private:
    thread_communicator_context& ctx_;
    thread_communicator_barrier barrier_;
    std::int64_t recv_count_;
    byte_t* recv_buf_;
};

class thread_communicator_allgatherv {
public:
    struct buffer_info {
        const byte_t* buf = nullptr;
        std::int64_t count = 0;
    };

    explicit thread_communicator_allgatherv(thread_communicator_context& ctx)
            : ctx_(ctx),
              barrier_(ctx),
              send_buffers_(ctx_.get_thread_count()) {}

    void operator()(const byte_t* send_buf,
                    std::int64_t send_count,
                    byte_t* recv_buf,
                    const std::int64_t* recv_counts,
                    const std::int64_t* displs,
                    const data_type& dtype);

private:
    thread_communicator_context& ctx_;
    thread_communicator_barrier barrier_;
    std::vector<buffer_info> send_buffers_;
};

class thread_communicator_sendrecv_replace {
public:
    struct buffer_info {
        byte_t* buf = nullptr;
        std::int64_t count = 0;
    };

    explicit thread_communicator_sendrecv_replace(thread_communicator_context& ctx)
            : ctx_(ctx),
              barrier_(ctx),
              send_buffers_(ctx_.get_thread_count()) {}

    void operator()(byte_t* buf,
                    std::int64_t count,
                    const data_type& dtype,
                    std::int64_t destination_rank,
                    std::int64_t source_rank);

private:
    thread_communicator_context& ctx_;
    thread_communicator_barrier barrier_;
    std::vector<buffer_info> send_buffers_;
};

class thread_communicator_allreduce {
public:
    struct buffer_info {
        const byte_t* send_buf = nullptr;
        std::int64_t count = 0;
    };

    explicit thread_communicator_allreduce(thread_communicator_context& ctx)
            : ctx_(ctx),
              barrier_(ctx),
              send_buffers_(ctx_.get_thread_count()) {}

    void operator()(const byte_t* send_buf,
                    byte_t* recv_buf,
                    std::int64_t count,
                    const data_type& dtype,
                    const spmd::reduce_op& op);

private:
    thread_communicator_context& ctx_;
    thread_communicator_barrier barrier_;
    std::vector<buffer_info> send_buffers_;

    static void reduce(const byte_t* src,
                       byte_t* dst,
                       std::int64_t count,
                       const data_type& dtype,
                       const spmd::reduce_op& op);

    static void fill_by_op_id(byte_t* dst,
                              std::int64_t count,
                              const data_type& dtype,
                              const spmd::reduce_op& op_id);

    template <typename T>
    static void reduce_impl(const byte_t* src,
                            byte_t* dst,
                            std::int64_t count,
                            const spmd::reduce_op& op);

    template <typename T>
    static void fill_with_zeros_impl(byte_t* dst, std::int64_t count);

    template <typename T>
    static void fill_with_min_impl(byte_t* dst, std::int64_t count);

    template <typename T>
    static void fill_with_max_impl(byte_t* dst, std::int64_t count);

    static bool data_blocks_has_intersection(const byte_t* x, const byte_t* y, std::int64_t count) {
        const std::uintptr_t x_address = std::uintptr_t(x);
        const std::uintptr_t y_address = std::uintptr_t(y);
        const std::uintptr_t a = std::min(x_address, y_address);
        const std::uintptr_t b = std::max(x_address, y_address);
        return a + count > b;
    }
};

class thread_communicator_allgather {
public:
    struct buffer_info {
        const byte_t* buf = nullptr;
        std::int64_t count = 0;
    };

    explicit thread_communicator_allgather(thread_communicator_context& ctx)
            : ctx_(ctx),
              barrier_(ctx),
              send_buffers_(ctx_.get_thread_count()) {}

    void operator()(const byte_t* send_buf,
                    std::int64_t send_count,
                    byte_t* recv_buf,
                    std::int64_t recv_count,
                    const data_type& dtype);

private:
    thread_communicator_context& ctx_;
    thread_communicator_barrier barrier_;
    std::vector<buffer_info> send_buffers_;
};

template <typename MemoryAccessKind>
class thread_communicator_impl
        : public dal::detail::via_host_interface_selector<MemoryAccessKind>::type {
public:
    using base_t = typename dal::detail::via_host_interface_selector<MemoryAccessKind>::type;
    using request_t = spmd::request_iface;

    // Explicitly declare all virtual functions with overloads to workaround Clang warning
    // https://stackoverflow.com/questions/18515183/c-overloaded-virtual-function-warning-by-clang
    using base_t::bcast;
    using base_t::allgatherv;
    using base_t::allreduce;
    using base_t::sendrecv_replace;

    class collective_operation_guard {
    public:
        explicit collective_operation_guard(thread_communicator_context& ctx) : ctx_(ctx) {
            ctx_.enter_communication();
        }

        ~collective_operation_guard() {
            ctx_.exit_communication();
        }

    private:
        thread_communicator_context& ctx_;
    };
#ifndef ONEDAL_DATA_PARALLEL
    explicit thread_communicator_impl(std::int64_t thread_count)
            : ctx_(thread_count),
              barrier_(ctx_),
              bcast_(ctx_),
              gather_(ctx_),
              allgatherv_(ctx_),
              sendrecv_replace_(ctx_),
              allreduce_(ctx_),
              allgather_(ctx_) {}
#endif
#ifdef ONEDAL_DATA_PARALLEL
    template <typename T = MemoryAccessKind,
              typename = spmd::enable_if_device_memory_accessible_t<T>>
    explicit thread_communicator_impl(sycl::queue& queue, std::int64_t thread_count)
            : base_t(queue),
              ctx_(thread_count),
              barrier_(ctx_),
              bcast_(ctx_),
              gather_(ctx_),
              allgatherv_(ctx_),
              sendrecv_replace_(ctx_),
              allreduce_(ctx_),
              allgather_(ctx_) {}
#endif

    thread_communicator_context& get_context() {
        return ctx_;
    }

    std::int64_t get_rank() override {
        return ctx_.map_thread_id_to_rank(std::this_thread::get_id());
    }

    std::int64_t get_default_root_rank() override {
        return ctx_.get_root_rank();
    }

    std::int64_t get_rank_count() override {
        return ctx_.get_thread_count();
    }

    bool get_mpi_offload_support() override {
        return false;
    }

    bool use_sendrecv_replace_alternative() override {
        return false;
    }

    void barrier() override;

    request_t* bcast(byte_t* send_buf,
                     std::int64_t count,
                     const data_type& dtype,
                     std::int64_t root) override;

    request_t* allgatherv(const byte_t* send_buf,
                          std::int64_t send_count,
                          byte_t* recv_buf,
                          const std::int64_t* recv_counts,
                          const std::int64_t* displs,
                          const data_type& dtype) override;

    request_t* allreduce(const byte_t* send_buf,
                         byte_t* recv_buf,
                         std::int64_t count,
                         const data_type& dtype,
                         const spmd::reduce_op& op) override;

    request_t* sendrecv_replace(byte_t* buf,
                                std::int64_t count,
                                const data_type& dtype,
                                std::int64_t destination_rank,
                                std::int64_t source_rank,
                                byte_t* recv_buf = nullptr) override;

private:
    thread_communicator_context ctx_;
    thread_communicator_barrier barrier_;
    thread_communicator_bcast bcast_;
    thread_communicator_gather gather_;
    thread_communicator_allgatherv allgatherv_;
    thread_communicator_sendrecv_replace sendrecv_replace_;
    thread_communicator_allreduce allreduce_;
    thread_communicator_allgather allgather_;
};

template <typename MemoryAccessKind>
class thread_communicator : public spmd::communicator<MemoryAccessKind> {
public:
    using impl_t = thread_communicator_impl<MemoryAccessKind>;
#ifndef ONEDAL_DATA_PARALLEL
    explicit thread_communicator(std::int64_t thread_count)
            : spmd::communicator<MemoryAccessKind>(new impl_t{ thread_count }) {}
#endif
#ifdef ONEDAL_DATA_PARALLEL
    template <typename T = MemoryAccessKind,
              typename = spmd::enable_if_device_memory_accessible_t<T>>
    explicit thread_communicator(sycl::queue& queue, std::int64_t thread_count)
            : spmd::communicator<MemoryAccessKind>(new impl_t{ queue, thread_count }) {}
#endif
    template <typename Body>
    void execute(const Body& body) {
        this->template get_impl<impl_t>().get_context().execute(body);
    }

    template <typename Body>
    void exclusive(const Body& body) {
        this->template get_impl<impl_t>().get_context().exclusive(body);
    }

    template <typename Body>
    auto map(const Body& body) {
        using map_t = decltype(body(std::declval<std::int64_t>()));

        std::vector<map_t> results(this->get_rank_count());
        execute([&](std::int64_t rank) {
            results[rank] = body(rank);
        });
        return results;
    }
};

} // namespace oneapi::dal::test::engine
