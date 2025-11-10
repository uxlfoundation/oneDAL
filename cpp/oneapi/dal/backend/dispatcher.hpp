/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/detail/global_context.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/spmd_policy.hpp"
#include "oneapi/dal/detail/dtype_dispatcher.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/communicator.hpp"
#include "oneapi/dal/backend/dispatcher_cpu.hpp"

#define KERNEL_SPEC(spec, ...) ::oneapi::dal::backend::kernel_spec<spec, __VA_ARGS__>

#define KERNEL_SINGLE_NODE_CPU(...) \
    KERNEL_SPEC(::oneapi::dal::backend::single_node_cpu_kernel, __VA_ARGS__)

#define KERNEL_UNIVERSAL_SPMD_CPU(...) \
    KERNEL_SPEC(::oneapi::dal::backend::universal_spmd_cpu_kernel, __VA_ARGS__)

#define KERNEL_SINGLE_NODE_GPU(...) \
    KERNEL_SPEC(::oneapi::dal::backend::single_node_gpu_kernel, __VA_ARGS__)

#define KERNEL_UNIVERSAL_SPMD_GPU(...) \
    KERNEL_SPEC(::oneapi::dal::backend::universal_spmd_gpu_kernel, __VA_ARGS__)

namespace oneapi::dal::backend {

#if defined(TARGET_X86_64)
struct cpu_dispatch_sse2 {};
struct cpu_dispatch_sse42 {};
struct cpu_dispatch_avx2 {};
struct cpu_dispatch_avx512 {};
#elif defined(TARGET_ARM)
struct cpu_dispatch_sve {};
#elif defined(TARGET_RISCV64)
struct cpu_dispatch_rv64 {};
#endif

#if defined(TARGET_X86_64)
using cpu_dispatch_default = cpu_dispatch_sse2;

#define __CPU_TAG_SSE2__    oneapi::dal::backend::cpu_dispatch_sse2
#define __CPU_TAG_SSE42__   oneapi::dal::backend::cpu_dispatch_sse42
#define __CPU_TAG_AVX2__    oneapi::dal::backend::cpu_dispatch_avx2
#define __CPU_TAG_AVX512__  oneapi::dal::backend::cpu_dispatch_avx512
#define __CPU_TAG_DEFAULT__ oneapi::dal::backend::cpu_dispatch_default

#elif defined(TARGET_ARM)
using cpu_dispatch_default = cpu_dispatch_sve;

#define __CPU_TAG_ARMV8SVE__ oneapi::dal::backend::cpu_dispatch_sve

#elif defined(TARGET_RISCV64)
using cpu_dispatch_default = cpu_dispatch_rv64;

#define __CPU_TAG_RISCV64GC__ oneapi::dal::backend::cpu_dispatch_rv64
#endif

template <typename MemoryAccessKind>
class communicator_provider : public base {
public:
    communicator_provider() = default;
    communicator_provider(const communicator<MemoryAccessKind>& comm)
            : comm_(new communicator<MemoryAccessKind>{ comm }) {}

    const communicator<MemoryAccessKind>& get_communicator() const {
        if (!comm_) {
            comm_.reset(new communicator<MemoryAccessKind>{});
        }
        return *comm_;
    }

private:
    mutable std::unique_ptr<communicator<MemoryAccessKind>> comm_;
};

class context_cpu : public communicator_provider<spmd::device_memory_access::none> {
public:
    explicit context_cpu(const detail::host_policy& policy = detail::host_policy::get_default())
            : cpu_extensions_(policy.get_enabled_cpu_extensions()) {
        detail::global_context::get_global_context();
    }

    explicit context_cpu(const detail::spmd_host_policy& policy)
            : communicator_provider<spmd::device_memory_access::none>(policy.get_communicator()),
              cpu_extensions_(policy.get_local().get_enabled_cpu_extensions()) {
        detail::global_context::get_global_context();
    }

    explicit context_cpu(const spmd::communicator<spmd::device_memory_access::none>& comm)
            : communicator_provider<spmd::device_memory_access::none>(comm),
              cpu_extensions_(detail::host_policy::get_default().get_enabled_cpu_extensions()) {}

    detail::cpu_extension get_enabled_cpu_extensions() const {
        return cpu_extensions_;
    }

private:
    detail::cpu_extension cpu_extensions_;
};

#ifdef ONEDAL_DATA_PARALLEL
class context_gpu : public communicator_provider<spmd::device_memory_access::usm> {
public:
    explicit context_gpu(const detail::data_parallel_policy& policy) : queue_(policy.get_queue()) {}

    explicit context_gpu(const detail::spmd_data_parallel_policy& policy)
            : communicator_provider(policy.get_communicator()),
              queue_(policy.get_local().get_queue()) {}

    sycl::queue& get_queue() const {
        return queue_;
    }

private:
    sycl::queue& queue_;
};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <typename CpuBranch, typename GpuBranch>
inline auto dispatch_by_device(const detail::data_parallel_policy& policy,
                               CpuBranch&& cpu_branch,
                               GpuBranch&& gpu_branch) {
    const auto device = policy.get_queue().get_device();
    if (device.is_cpu()) {
        return cpu_branch();
    }
    else if (device.is_gpu()) {
        return gpu_branch();
    }
    else {
        throw unsupported_device{ dal::detail::error_messages::unsupported_device_type() };
    }
}
#endif

/// Tag that indicates CPU kernel for single-node
struct single_node_cpu_kernel {};

/// Tag that indicates universal CPU kernel for single-node and SPMD modes
struct universal_spmd_cpu_kernel {};

/// Tag that indicates GPU kernel for single-node
struct single_node_gpu_kernel {};

/// Tag that indicates universal GPU kernel for single-node and SPMD modes
struct universal_spmd_gpu_kernel {};

template <typename Tag, typename Kernel>
struct kernel_spec {};

template <typename... KernelSpecs>
struct kernel_dispatcher {};

template <typename Context, typename Kernel, typename... Args>
struct kernel_return_type {
    using type = decltype(std::declval<Kernel>()(std::declval<Context>(), std::declval<Args>()...));
};

template <typename... Args>
using cpu_kernel_return_t = typename kernel_return_type<context_cpu, Args...>::type;

#ifdef ONEDAL_DATA_PARALLEL
template <typename... Args>
using gpu_kernel_return_t = typename kernel_return_type<context_gpu, Args...>::type;
#endif

/// Dispatcher for the case of only CPU and single-node algorithm
template <typename CpuKernel>
struct kernel_dispatcher<kernel_spec<single_node_cpu_kernel, CpuKernel>> {
    template <typename... Args>
    auto operator()(const detail::host_policy& policy, Args&&... args) const {
        return CpuKernel{}(context_cpu{ policy }, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto operator()(const detail::spmd_host_policy& policy, Args&&... args) const
        -> cpu_kernel_return_t<CpuKernel, Args...> {
        // We have to specify return type for this function as compiler cannot
        // infer it from a body that consist of single `throw` expression
        using msg = detail::error_messages;
        throw unimplemented{ msg::spmd_version_of_algorithm_is_not_implemented() };
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename... Args>
    auto operator()(const detail::data_parallel_policy& policy, Args&&... args) const {
        return dispatch_by_device(
            policy,
            [&]() {
                return CpuKernel{}(context_cpu{}, std::forward<Args>(args)...);
            },
            [&]() -> cpu_kernel_return_t<CpuKernel, Args...> {
                // We have to specify return type for this lambda as compiler cannot
                // infer it from a body that consist of single `throw` expression
                using msg = detail::error_messages;
                throw unimplemented{ msg::algorithm_is_not_implemented_for_this_device() };
            });
    }
    template <typename... Args>
    auto operator()(const detail::spmd_data_parallel_policy& policy, Args&&... args) const
        -> cpu_kernel_return_t<CpuKernel, Args...> {
        // We have to specify return type for this function as compiler cannot
        // infer it from a body that consist of single `throw` expression
        using msg = detail::error_messages;
        throw unimplemented{ msg::spmd_version_of_algorithm_is_not_implemented_for_this_device() };
    }
#endif
};

/// Dispatcher for the case of multi-node CPU algorithm based on universal SPMD kernel
template <typename CpuKernel>
struct kernel_dispatcher<kernel_spec<universal_spmd_cpu_kernel, CpuKernel>> {
    template <typename... Args>
    auto operator()(const detail::host_policy& policy, Args&&... args) const {
        return CpuKernel{}(context_cpu{}, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto operator()(const detail::spmd_host_policy& policy, Args&&... args) const {
        return CpuKernel{}(context_cpu{ policy }, std::forward<Args>(args)...);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename... Args>
    auto operator()(const detail::data_parallel_policy& policy, Args&&... args) const {
        return dispatch_by_device(
            policy,
            [&]() {
                return CpuKernel{}(context_cpu{}, std::forward<Args>(args)...);
            },
            [&]() -> cpu_kernel_return_t<CpuKernel, Args...> {
                // We have to specify return type for this lambda as compiler cannot
                // infer it from a body that consist of single `throw` expression
                using msg = detail::error_messages;
                throw unimplemented{ msg::algorithm_is_not_implemented_for_this_device() };
            });
    }
    template <typename... Args>
    auto operator()(const detail::spmd_data_parallel_policy& policy, Args&&... args) const
        -> cpu_kernel_return_t<CpuKernel, Args...> {
        // We have to specify return type for this function as compiler cannot
        // infer it from a body that consist of single `throw` expression
        using msg = detail::error_messages;
        throw unimplemented{ msg::spmd_version_of_algorithm_is_not_implemented() };
    }
#endif
};

#ifdef ONEDAL_DATA_PARALLEL
/// Dispatcher for the case of single-node CPU and GPU algorithm
template <typename CpuKernel, typename GpuKernel>
struct kernel_dispatcher<kernel_spec<single_node_cpu_kernel, CpuKernel>,
                         kernel_spec<single_node_gpu_kernel, GpuKernel>> {
    template <typename... Args>
    auto operator()(const detail::data_parallel_policy& policy, Args&&... args) const {
        return dispatch_by_device(
            policy,
            [&]() {
                return CpuKernel{}(context_cpu{}, std::forward<Args>(args)...);
            },
            [&]() {
                return GpuKernel{}(context_gpu{ policy }, std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    auto operator()(const detail::spmd_data_parallel_policy& policy, Args&&... args) const
        -> cpu_kernel_return_t<CpuKernel, Args...> {
        // We have to specify return type for this function as compiler cannot
        // infer it from a body that consist of single `throw` expression
        using msg = detail::error_messages;
        throw unimplemented{ msg::spmd_version_of_algorithm_is_not_implemented() };
    }
};
#endif

#ifdef ONEDAL_DATA_PARALLEL
/// Dispatcher for the case of single-node CPU and multi-node
/// GPU algorithm based on universal SPMD kernel
template <typename CpuKernel, typename GpuKernel>
struct kernel_dispatcher<kernel_spec<single_node_cpu_kernel, CpuKernel>,
                         kernel_spec<universal_spmd_gpu_kernel, GpuKernel>> {
    template <typename... Args>
    auto operator()(const detail::data_parallel_policy& policy, Args&&... args) const {
        return dispatch_by_device(
            policy,
            [&]() {
                return CpuKernel{}(context_cpu{}, std::forward<Args>(args)...);
            },
            [&]() {
                return GpuKernel{}(context_gpu{ policy }, std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    auto operator()(const detail::spmd_data_parallel_policy& policy, Args&&... args) const {
        return dispatch_by_device(
            policy.get_local(),
            [&]() -> gpu_kernel_return_t<GpuKernel, Args...> {
                return CpuKernel{}(context_cpu{}, std::forward<Args>(args)...);
            },
            [&]() {
                return GpuKernel{}(context_gpu{ policy }, std::forward<Args>(args)...);
            });
    }
};
#endif

inline bool test_cpu_extension(detail::cpu_extension mask, detail::cpu_extension test) {
    return mask >= test;
}

template <typename Op>
inline constexpr auto dispatch_by_cpu(const context_cpu& ctx, Op&& op) {
    using detail::cpu_extension;

    [[maybe_unused]] const cpu_extension cpu_ex = ctx.get_enabled_cpu_extensions();

#if defined(TARGET_X86_64)
    ONEDAL_IF_CPU_DISPATCH_AVX512(if (test_cpu_extension(cpu_ex, cpu_extension::avx512)) {
        return op(cpu_dispatch_avx512{});
    })
    ONEDAL_IF_CPU_DISPATCH_AVX2(
        if (test_cpu_extension(cpu_ex, cpu_extension::avx2)) { return op(cpu_dispatch_avx2{}); })
    ONEDAL_IF_CPU_DISPATCH_SSE42(
        if (test_cpu_extension(cpu_ex, cpu_extension::sse42)) { return op(cpu_dispatch_sse42{}); })

#elif defined(TARGET_ARM)
    ONEDAL_IF_CPU_DISPATCH_A8SVE(
        if (test_cpu_extension(cpu_ex, cpu_extension::sve)) { return op(cpu_dispatch_sve{}); })

#elif defined(TARGET_RISCV64)
    ONEDAL_IF_CPU_DISPATCH_RV64(
        if (test_cpu_extension(cpu_ex, cpu_extension::rv64)) { return op(cpu_dispatch_rv64{}); })
#endif

    return op(cpu_dispatch_default{});
}

using detail::v1::dispatch_by_data_type;
using detail::v1::multi_dispatch_by_data_type;

} // namespace oneapi::dal::backend
