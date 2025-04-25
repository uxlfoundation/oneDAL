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

#include "oneapi/dal/algo/decision_forest/parameters/cpu/infer_parameters.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

#if defined(TARGET_X86_64)
#define CPU_EXTENSION dal::detail::cpu_extension::avx512
#elif defined(TARGET_ARM)
#define CPU_EXTENSION dal::detail::cpu_extension::sve
#elif defined(TARGET_RISCV64)
#define CPU_EXTENSION dal::detail::cpu_extension::rv64
#endif

namespace oneapi::dal::decision_forest::parameters {

using dal::backend::context_cpu;

std::int64_t ONEDAL_EXPORT propose_block_size(const context_cpu& ctx) {
    std::int64_t block_size = 22l;
    if (ctx.get_enabled_cpu_extensions() == CPU_EXTENSION) {
        /// Here if AVX512 extensions are available on CPU
        block_size = 32l;
    }
    return block_size;
}

using method::by_default;
using task::classification;

template struct ONEDAL_EXPORT infer_parameters_cpu<float, by_default, classification>;
template struct ONEDAL_EXPORT infer_parameters_cpu<double, by_default, classification>;

} // namespace oneapi::dal::decision_forest::parameters
