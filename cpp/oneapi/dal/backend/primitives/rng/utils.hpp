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

#include <daal/src/externals/service_rng.h>
#include <daal/src/algorithms/engines/engine_batch_impl.h>
#include <daal/src/algorithms/engines/engine_types_internal.h>

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"

namespace oneapi::dal::backend::primitives {

enum class rng_method {
    uniform,
    uniform_bits_32,
    bernoulli,
    gaussian,
    uniform_without_replacement
};

template <typename U, rng_method M, typename... Args>
struct uniform_functor {
    void operator()(Args... args) {
        std::int32_t res = 0;

        if constexpr (M == rng_method::uniform) {
            res = U{}.uniform(std::forward<Args>(args)...);
        }
        else if constexpr (M == rng_method::uniform_bits_32) {
            res = U{}.uniformBits32(std::forward<Args>(args)...);
        }
        else if constexpr (M == rng_method::bernoulli) {
            res = U{}.bernoulli(std::forward<Args>(args)...);
        }
        else if constexpr (M == rng_method::gaussian) {
            res = U{}.gaussian(std::forward<Args>(args)...);
        }
        else if constexpr (M == rng_method::uniform_without_replacement) {
            res = U{}.uniformWithoutReplacement(std::forward<Args>(args)...);
        }

        if (res) {
            using msg = dal::detail::error_messages;
            throw internal_error(msg::failed_to_generate_random_numbers());
        }
    }
};

template <typename Type = std::size_t, rng_method M = rng_method::uniform, typename... Args>
struct internal_dispatcher {
    explicit internal_dispatcher(Args&&... args) : args_(std::forward<Args>(args)...) {}
    template <typename CPU>
    void operator()(CPU cpu) {
        using uniform_type = daal::internal::
            RNGsInst<Type, oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value>;
        uniform_functor<uniform_type, M, Args...> f;
        std::apply(f, args_);
    }
    std::tuple<Args...> args_;
};

struct uniform_dispatcher {
    template <typename Type = std::size_t, typename... Args>
    static void uniform_by_cpu(Args&&... args) {
        internal_dispatcher<Type, rng_method::uniform, Args...> disp(std::forward<Args>(args)...);
        dispatch_by_cpu(context_cpu{}, disp);
    }

    template <typename Type = std::size_t, typename... Args>
    static void uniform_without_replacement_by_cpu(Args&&... args) {
        internal_dispatcher<Type, rng_method::uniform_without_replacement, Args...> disp(
            std::forward<Args>(args)...);
        dispatch_by_cpu(context_cpu{}, disp);
    }
};

/// Draws `count` distinct indices out of `[0, top)` with the partial Fisher-Yates algorithm.
///
/// @param[in]  state   The DAAL engine state the values are drawn from.
/// @param[out] indices The `count` drawn indices.
/// @param[in]  count   The number of indices to draw; must not exceed `top`.
/// @param[in]  top     The size of the population to draw from.
template <typename Type>
inline void partial_fisher_yates_draw(void* state,
                                      Type* indices,
                                      std::size_t count,
                                      std::size_t top) {
    std::size_t value = 0;
    for (std::size_t i = 0; i < count; ++i) {
        uniform_dispatcher::uniform_by_cpu(1, &value, state, i, top);
        indices[i] = dal::detail::integral_cast<Type>(value);
    }
    // Resolving step `i` needs the raw draws of all earlier steps, so the pass runs backwards and
    // every draw stays in place until the steps that read it are done.
    for (std::size_t i = count; i-- > 0;) {
        value = dal::detail::integral_cast<std::size_t>(indices[i]);
        for (std::size_t j = i; j > 0; j--) {
            if (value == dal::detail::integral_cast<std::size_t>(indices[j - 1])) {
                value = j - 1;
            }
        }
        indices[i] = dal::detail::integral_cast<Type>(value);
    }
}

} // namespace oneapi::dal::backend::primitives
