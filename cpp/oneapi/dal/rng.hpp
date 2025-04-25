/*******************************************************************************
# Copyright contributors to the oneDAL project
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

namespace oneapi::dal {

namespace v1 {

/// Enumeration of RNG engine methods
enum class engine_type {
    /// Mersenne Twister engine with specific optimizations for parallel environments.
    mt2203,
    /// Multiplicative congruential generator with a modulus of \(2^{59}\)
    mcg59,
    /// Counter-based RNG engine optimized for parallel computations
    philox4x32x10,
    /// Standard Mersenne Twister engine with a period of \(2^{19937} - 1\)
    mt19937,
    /// Combined multiple recursive generator with a period of \(2^{191}\)
    mrg32k3a
};

} // namespace v1

using v1::engine_type;
} // namespace oneapi::dal
