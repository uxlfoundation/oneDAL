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

#include "oneapi/dal/rng.hpp"
#pragma once

namespace oneapi::dal::backend::primitives {

/// Enum class representing different random number generation (RNG) engine methods.
///
/// This enumeration defines the available RNG engines supported by the library.
/// Each engine method corresponds to a specific algorithm for generating random numbers
/// @enum engine_type_internal
/// Enumeration of RNG engine methods:
/// - `mt2203`: Mersenne Twister engine with specific optimizations for parallel environments.
/// - `mcg59`: Multiplicative congruential generator with a modulus of \(2^{59}\).
/// - `mt19937`: Standard Mersenne Twister engine with a period of \(2^{19937} - 1\).
/// - `mrg32k3a`: Combined multiple recursive generator with a period of \(2^{191}\).
/// - `philox4x32x10`: Counter-based RNG engine optimized for parallel computations.
enum class engine_type_internal { mt2203, mcg59, mt19937, mrg32k3a, philox4x32x10 };

inline engine_type_internal convert_engine_method(engine_type method) {
    switch (method) {
        case engine_type::mt2203:
            return ::oneapi::dal::backend::primitives::engine_type_internal::mt2203;
        case engine_type::mcg59:
            return ::oneapi::dal::backend::primitives::engine_type_internal::mcg59;
        case engine_type::mrg32k3a:
            return ::oneapi::dal::backend::primitives::engine_type_internal::mrg32k3a;
        case engine_type::philox4x32x10:
            return ::oneapi::dal::backend::primitives::engine_type_internal::philox4x32x10;
        case engine_type::mt19937:
            return ::oneapi::dal::backend::primitives::engine_type_internal::mt19937;
        default: throw std::runtime_error("Unsupported engine type in generate_rng");
    }
}
} // namespace oneapi::dal::backend::primitives
