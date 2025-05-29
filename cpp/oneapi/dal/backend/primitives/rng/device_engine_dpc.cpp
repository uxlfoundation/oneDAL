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
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) == sycl::usm::alloc::host) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    const std::int64_t n = b - a;
    if (count > n) {
        throw std::invalid_argument("Sample size exceeds population size");
    }

    // Allocate array for full population [a, b)
    auto full_array = array<Type>::empty(queue, n);
    Type* full_ptr = full_array.get_mutable_data();

    // Fill array with [a, a+1, ..., b-1]
    auto fill_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            full_ptr[i] = a + static_cast<Type>(i);
        });
    });

    // Generate random indices for Fisher-Yates shuffle
    auto rand_array = array<Type>::empty(queue, count);
    Type* rand_ptr = rand_array.get_mutable_data();

    oneapi::mkl::rng::uniform<Type> distr(0, n);
    engine_.skip_ahead_cpu(count);
    auto rand_event = generate_rng(distr, engine_, count, rand_ptr, { fill_event });

    // Perform Fisher-Yates shuffle for first 'count' elements
    auto shuffle_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(fill_event);
        cgh.single_task([=]() {
            for (std::int64_t idx = 0; idx < count; ++idx) {
                // Generate random index between idx and n-1 (inclusive)
                std::int64_t j = idx + rand_ptr[idx] % (n - idx);
                if (j != idx) {
                    Type tmp = full_ptr[idx];
                    full_ptr[idx] = full_ptr[j];
                    full_ptr[j] = tmp;
                }
                // Copy the shuffled element to output
                dst[idx] = full_ptr[idx];
            }
        });
    });

    return shuffle_event;
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
    // Check if destination pointer is on host, as device memory is not supported for dst here
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) == sycl::usm::alloc::host) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }

    engine_.skip_ahead_gpu(count);

    // Array to store random indices for shuffling
    auto idx_array = array<Type>::empty(queue, count);
    Type* idx_ptr = idx_array.get_mutable_data();

    // Generate random indices for shuffle
    oneapi::mkl::rng::uniform<Type> distr(0, count);
    auto rand_event = generate_rng(distr, engine_, count, idx_ptr, deps);

    // Perform the shuffle by swapping values in dst based on the generated random indices
    auto shuffle_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(rand_event); // Depends on the RNG generation step
        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
            // Randomly select another index to swap with the current one
            Type j = idx_ptr[i];
            if (j != i) {
                Type temp = dst[i];
                dst[i] = dst[j];
                dst[j] = temp;
            }
        });
    });

    return shuffle_event;
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
    // Validate inputs
    const auto casted_top = dal::detail::integral_cast<std::size_t>(top);
    const std::int64_t count = result_array.get_count();
    const auto casted_count = dal::detail::integral_cast<std::size_t>(count);
    ONEDAL_ASSERT(casted_count <= casted_top, "Sample size exceeds population size");

    // Check if result_array is in device memory
    if (sycl::get_pointer_type(result_array.get_mutable_data(), queue_.get_context()) ==
        sycl::usm::alloc::host) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }

    // Allocate temporary array for the full population [0, top)
    auto full_array = array<Type>::empty(queue_, casted_top);
    Type* full_ptr = full_array.get_mutable_data();

    // Initialize full_array with values [0, 1, ..., top-1]
    auto fill_event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>(casted_top), [=](sycl::id<1> i) {
            full_ptr[i] = static_cast<Type>(i);
        });
    });

    auto* result_ptr = result_array.get_mutable_data();

    // Allocate array for random indices
    auto rand_array = array<std::int32_t>::empty(queue_, casted_count);
    std::int32_t* rand_ptr = rand_array.get_mutable_data();

    // Generate random indices in range [0, top) for Fisher-Yates shuffle
    oneapi::mkl::rng::uniform<std::int32_t> distr(0, casted_top);
    engine_.skip_ahead_cpu(casted_count); // Adjust engine state
    auto rand_event = generate_rng(distr, engine_, casted_count, rand_ptr, { fill_event });

    // Perform Fisher-Yates shuffle for the first 'count' elements
    auto shuffle_event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(rand_event);
        cgh.single_task([=]() {
            for (std::size_t idx = 0; idx < casted_count; ++idx) {
                // Generate random index between idx and top-1 (inclusive)
                std::size_t j = idx + static_cast<std::size_t>(rand_ptr[idx]) % (casted_top - idx);
                if (j != idx) {
                    Type tmp = full_ptr[idx];
                    full_ptr[idx] = full_ptr[j];
                    full_ptr[j] = tmp;
                }
                // Copy the shuffled element to the output
                result_ptr[idx] = full_ptr[idx];
            }
        });
    });

    return shuffle_event;
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
