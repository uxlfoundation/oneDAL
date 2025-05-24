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

#pragma once

#include <daal/include/algorithms/engines/mt2203/mt2203.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/include/algorithms/engines/mrg32k3a/mrg32k3a.h>
#include <daal/include/algorithms/engines/philox4x32x10/philox4x32x10.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/rng/utils.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_types.hpp"

namespace oneapi::dal::backend::primitives {

/// A class that provides an interface for random number generation on the host (CPU) only.
///
/// This class serves as a wrapper for host-based random number generators (RNGs), supporting multiple engine
/// types for flexible and efficient random number generation on CPU. It abstracts the underlying engine
/// implementation and provides an interface to manage and retrieve the engine's state.
///
/// @note The class only supports host-based RNG and does not require a SYCL queue or device context.
class host_engine {
public:
    /// @param[in] seed    The initial seed for the random number generator. Defaults to `777`.
    /// @param[in] method  The engine method. Defaults to `engine_type_internal::mt2203`.
    host_engine(std::int64_t seed = 777,
                engine_type_internal method = engine_type_internal::mt2203) {
        switch (method) {
            case engine_type_internal::mt2203:
                host_engine_ = daal::algorithms::engines::mt2203::Batch<>::create(seed);
                break;
            case engine_type_internal::mcg59:
                host_engine_ = daal::algorithms::engines::mcg59::Batch<>::create(seed);
                break;
            case engine_type_internal::mrg32k3a:
                host_engine_ = daal::algorithms::engines::mrg32k3a::Batch<>::create(seed);
                break;
            case engine_type_internal::philox4x32x10:
                host_engine_ = daal::algorithms::engines::philox4x32x10::Batch<>::create(seed);
                break;
            case engine_type_internal::mt19937:
                host_engine_ = daal::algorithms::engines::mt19937::Batch<>::create(seed);
                break;
            default: throw std::invalid_argument("Unsupported engine type 1");
        }
        impl_ =
            dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(host_engine_.get());
        if (!impl_) {
            throw std::domain_error("RNG engine is not supported");
        }
    }

    explicit host_engine(const daal::algorithms::engines::EnginePtr& eng) : host_engine_(eng) {
        impl_ = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(eng.get());
        if (!impl_) {
            throw domain_error(dal::detail::error_messages::rng_engine_is_not_supported());
        }
    }

    /// Assignment operator.
    host_engine& operator=(const daal::algorithms::engines::EnginePtr& eng) {
        host_engine_ = eng;
        impl_ = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(eng.get());
        if (!impl_) {
            throw domain_error(dal::detail::error_messages::rng_engine_is_not_supported());
        }

        return *this;
    }

    // Copy constructor
    host_engine(const host_engine&) = default;

    // Copy assignment operator
    host_engine& operator=(const host_engine&) = default;

    // Move constructor
    host_engine(host_engine&& other) noexcept = default;

    // Move assignment operator
    host_engine& operator=(host_engine&& other) noexcept = default;

    /// Destructor.
    ~host_engine() = default;

    /// Retrieves the state of the host rng engine(DAAL).
    /// @return Pointer to the host engine state.
    void* get_host_engine_state() const {
        return impl_->getState();
    }

private:
    daal::algorithms::engines::EnginePtr host_engine_;
    daal::algorithms::engines::internal::BatchBaseImpl* impl_;
};

/// Generates uniformly distributed random numbers on the CPU.
/// @tparam Type The data type of the generated numbers.
/// @param[in] count The number of random numbers to generate.
/// @param[out] dst Pointer to the output buffer.
/// @param[in] engine_ Reference to the device engine.
/// @param[in] a The lower bound of the uniform distribution.
/// @param[in] b The upper bound of the uniform distribution.
template <typename Type>
void uniform(std::int64_t count, Type* dst, host_engine& host_engine, Type a, Type b) {
    auto state = host_engine.get_host_engine_state();
    uniform_dispatcher::uniform_by_cpu<Type>(count, dst, state, a, b);
}

/// Generates a random permutation of elements without replacement on the CPU.
/// @tparam Type The data type of the elements.
/// @param[in] count The number of elements to generate.
/// @param[out] dst Pointer to the output buffer.
/// @param[out] buffer Temporary buffer used for computations.
/// @param[in] engine_ Reference to the device engine.
/// @param[in] a The lower bound of the range.
/// @param[in] b The upper bound of the range.
template <typename Type>
void uniform_without_replacement(std::int64_t count,
                                 Type* dst,
                                 Type* buffer,
                                 host_engine host_engine,
                                 Type a,
                                 Type b) {
    auto state = host_engine.get_host_engine_state();
    uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count, dst, buffer, state, a, b);
}

/// Shuffles an array using random swaps on the CPU.
/// @tparam Type The data type of the array elements.
/// @param[in] count The number of elements to shuffle.
/// @param[in, out] dst Pointer to the array to be shuffled.
/// @param[in] engine_ Reference to the device engine.
template <typename Type, typename T = Type, typename = std::enable_if_t<std::is_integral_v<T>>>
void shuffle(std::int64_t count, Type* dst, host_engine host_engine) {
    auto state = host_engine.get_host_engine_state();
    Type idx[2];
    for (std::int64_t i = 0; i < count; ++i) {
        uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
        std::swap(dst[idx[0]], dst[idx[1]]);
    }
}

/// Shuffles an array using random swaps on the CPU.
/// @tparam Type The data type of the array elements.
/// @param[in] count The number of elements to shuffle.
/// @param[in, out] dst Pointer to the array to be shuffled.
/// @param[in] engine_ Reference to the device engine.
template <typename Type>
void partial_fisher_yates_shuffle(ndview<Type, 1>& result_array,
                                  std::int64_t top,
                                  std::int64_t seed,
                                  engine_type_internal method = engine_type_internal::mt19937) {
    host_engine eng_ = host_engine(seed, method);
    const auto casted_top = dal::detail::integral_cast<std::size_t>(top);
    const std::int64_t count = result_array.get_count();
    const auto casted_count = dal::detail::integral_cast<std::size_t>(count);
    ONEDAL_ASSERT(casted_count < casted_top);
    auto indices_ptr = result_array.get_mutable_data();

    std::int64_t k = 0;
    std::size_t value = 0;
    auto state = eng_.get_host_engine_state();
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
        k++;
    }
    ONEDAL_ASSERT(k == count);
}

} // namespace oneapi::dal::backend::primitives
