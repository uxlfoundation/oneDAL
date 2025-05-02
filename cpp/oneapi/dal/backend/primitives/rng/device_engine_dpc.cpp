/*******************************************************************************
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

#include "oneapi/dal/backend/primitives/rng/device_engine.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include <oneapi/mkl.hpp>

namespace oneapi::dal::backend::primitives {

namespace bk = oneapi::dal::backend;

template <typename Distribution, typename Type>
sycl::event generate_rng(Distribution& distr,
                         device_engine& engine_,
                         std::int64_t count,
                         Type* dst,
                         const event_vector& deps) {
    auto engine_type = engine_.get_device_engine_base_ptr()->get_engine_type();

    if (engine_type == engine_type_internal::philox4x32x10) {
        auto& device_engine =
            *(static_cast<gen_philox*>(engine_.get_device_engine_base_ptr().get()))->get();
        return oneapi::mkl::rng::generate(distr, device_engine, count, dst, deps);
    }
    else if (engine_type == engine_type_internal::mt19937) {
        auto& device_engine =
            *(static_cast<gen_mt19937*>(engine_.get_device_engine_base_ptr().get()))->get();
        return oneapi::mkl::rng::generate(distr, device_engine, count, dst, deps);
    }
    else if (engine_type == engine_type_internal::mrg32k3a) {
        auto& device_engine =
            *(static_cast<gen_mrg32k*>(engine_.get_device_engine_base_ptr().get()))->get();
        return oneapi::mkl::rng::generate(distr, device_engine, count, dst, deps);
    }
    else if (engine_type == engine_type_internal::mcg59) {
        auto& device_engine =
            *(static_cast<gen_mcg59*>(engine_.get_device_engine_base_ptr().get()))->get();
        return oneapi::mkl::rng::generate(distr, device_engine, count, dst, deps);
    }
    else if (engine_type == engine_type_internal::mt2203) {
        auto& device_engine =
            *(static_cast<gen_mt2203*>(engine_.get_device_engine_base_ptr().get()))->get();
        return oneapi::mkl::rng::generate(distr, device_engine, count, dst, deps);
    }
    else {
        throw std::runtime_error("Unsupported engine type in generate_rng");
    }
}

/// Generates uniformly distributed random numbers on the GPU.
/// @tparam Type The data type of the generated numbers.
/// @param[in] queue The SYCL queue for device execution.
/// @param[in] count The number of random numbers to generate.
/// @param[out] dst Pointer to the output buffer.
/// @param[in] engine_ Reference to the device engine.
/// @param[in] a The lower bound of the uniform distribution.
/// @param[in] b The upper bound of the uniform distribution.
/// @param[in] deps Dependencies for the SYCL event.
template <typename Type>
sycl::event uniform(sycl::queue& queue,
                    std::int64_t count,
                    Type* dst,
                    device_engine& engine_,
                    Type a,
                    Type b,
                    const event_vector& deps) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) == sycl::usm::alloc::host) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    oneapi::mkl::rng::uniform<Type> distr(a, b);
    engine_.skip_ahead_cpu(count);
    auto event = generate_rng(distr, engine_, count, dst, deps);
    return event;
}

template <typename Type>
sycl::event uniform_normalized(sycl::queue& queue,
                               std::int64_t count,
                               Type* dst,
                               device_engine& engine_,
                               Type a,
                               Type b,
                               const event_vector& deps = {}) {
    using Index = std::uint32_t;

    auto index_array = array<Index>::empty(queue, count);
    Index* index_ptr = index_array.get_mutable_data();

    auto gen_event =
        uniform<Index>(queue, count, index_ptr, engine_, 0, static_cast<Index>(count), deps);

    auto norm_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(gen_event);
        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
            dst[i] = a + (b - a) * static_cast<Type>(index_ptr[i]) / static_cast<Type>(count);
        });
    });

    return norm_event;
}

/// Generates a random permutation of elements without replacement on the GPU.
/// @tparam Type The data type of the elements.
/// @param[in] queue The SYCL queue for device execution.
/// @param[in] count The number of elements to generate.
/// @param[out] dst Pointer to the output buffer.
/// @param[out] buffer Temporary buffer used for computations.
/// @param[in] engine_ Reference to the device engine.
/// @param[in] a The lower bound of the range.
/// @param[in] b The upper bound of the range.
/// @param[in] deps Dependencies for the SYCL event.
template <typename Type>
sycl::event uniform_without_replacement(sycl::queue& queue,
                                        std::int64_t count,
                                        Type* dst,
                                        device_engine& engine_,
                                        Type a,
                                        Type b,
                                        const event_vector& deps) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    void* state = engine_.get_host_engine_state();
    engine_.skip_ahead_gpu(count);
    uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count, dst, state, a, b);
    auto event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
    });
    return event;
}

/// Shuffles an array using random swaps on the GPU.
/// @tparam Type The data type of the array elements.
/// @param[in] queue The SYCL queue for device execution.
/// @param[in] count The number of elements to shuffle.
/// @param[in, out] dst Pointer to the array to be shuffled.
/// @param[in] engine_ Reference to the device engine.
/// @param[in] deps Dependencies for the SYCL event.
template <typename Type>
sycl::event shuffle(sycl::queue& queue,
                    std::int64_t count,
                    Type* dst,
                    device_engine& engine_,
                    const event_vector& deps) {
    Type idx[2];
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    void* state = engine_.get_host_engine_state();
    engine_.skip_ahead_gpu(count);

    for (std::int64_t i = 0; i < count; ++i) {
        uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
        std::swap(dst[idx[0]], dst[idx[1]]);
    }
    auto event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
    });
    return event;
}

/// Partially shuffles the first `top` elements of an array using the Fisher-Yates algorithm.
/// @tparam Type The data type of the array elements.
/// @param[in] queue_ The SYCL queue for device execution.
/// @param[in, out] result_array The array to be partially shuffled.
/// @param[in] top The number of elements to shuffle.
/// @param[in] seed The seed for the engine.
/// @param[in] method The rng engine type. Defaults to `mt19937`.
/// @param[in] deps Dependencies for the SYCL event.
template <typename Type>
sycl::event partial_fisher_yates_shuffle(sycl::queue& queue_,
                                         ndview<Type, 1>& result_array,
                                         std::int64_t top,
                                         std::int64_t seed,
                                         device_engine& engine_,
                                         const event_vector& deps) {
    const auto casted_top = dal::detail::integral_cast<std::size_t>(top);
    const std::int64_t count = result_array.get_count();
    const auto casted_count = dal::detail::integral_cast<std::size_t>(count);
    ONEDAL_ASSERT(casted_count < casted_top);
    auto indices_ptr = result_array.get_mutable_data();

    std::size_t value = 0;
    auto state = engine_.get_host_engine_state();
    for (std::size_t i = 0; i < casted_count; i++) {
        uniform_dispatcher::uniform_by_cpu(1, &value, state, i, casted_top);
        for (std::size_t j = i; j > 0; j--) {
            if (value == dal::detail::integral_cast<std::size_t>(indices_ptr[j - 1])) {
                value = j - 1;
            }
        }
        if (value >= casted_top)
            continue;
        indices_ptr[i] = dal::detail::integral_cast<Type>(value);
    }
    auto event = queue_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
    });
    return event;
}

#define INSTANTIATE_UNIFORM(F)                                         \
    template ONEDAL_EXPORT sycl::event uniform(sycl::queue& queue,     \
                                               std::int64_t count_,    \
                                               F* dst,                 \
                                               device_engine& engine_, \
                                               F a,                    \
                                               F b,                    \
                                               const event_vector& deps);

INSTANTIATE_UNIFORM(float)
INSTANTIATE_UNIFORM(double)
INSTANTIATE_UNIFORM(std::int32_t)

#define INSTANTIATE_UNIFORM_NORMALIZED(F)                                         \
    template ONEDAL_EXPORT sycl::event uniform_normalized(sycl::queue& queue,     \
                                                          std::int64_t count_,    \
                                                          F* dst,                 \
                                                          device_engine& engine_, \
                                                          F a,                    \
                                                          F b,                    \
                                                          const event_vector& deps);

INSTANTIATE_UNIFORM_NORMALIZED(std::int32_t)

#define INSTANTIATE_UWR(F)                                                                 \
    template ONEDAL_EXPORT sycl::event uniform_without_replacement(sycl::queue& queue,     \
                                                                   std::int64_t count_,    \
                                                                   F* dst,                 \
                                                                   device_engine& engine_, \
                                                                   F a,                    \
                                                                   F b,                    \
                                                                   const event_vector& deps);

INSTANTIATE_UWR(std::int32_t)

#define INSTANTIATE_SHUFFLE(F)                                         \
    template ONEDAL_EXPORT sycl::event shuffle(sycl::queue& queue,     \
                                               std::int64_t count_,    \
                                               F* dst,                 \
                                               device_engine& engine_, \
                                               const event_vector& deps);

INSTANTIATE_SHUFFLE(std::int32_t)

#define INSTANTIATE_PARTIAL_SHUFFLE(F)                                                      \
    template ONEDAL_EXPORT sycl::event partial_fisher_yates_shuffle(sycl::queue& queue,     \
                                                                    ndview<F, 1>& a,        \
                                                                    std::int64_t top,       \
                                                                    std::int64_t seed,      \
                                                                    device_engine& engine_, \
                                                                    const event_vector& deps);

INSTANTIATE_PARTIAL_SHUFFLE(std::int32_t)
INSTANTIATE_PARTIAL_SHUFFLE(std::int64_t)

} // namespace oneapi::dal::backend::primitives
