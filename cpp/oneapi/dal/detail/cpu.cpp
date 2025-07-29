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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/cpu.hpp"
#include <daal/src/services/service_defines.h>

namespace oneapi::dal::detail {
namespace v1 {

ONEDAL_EXPORT cpu_extension from_daal_cpu_type(int cpu_type) {
    daal::CpuType cpu = static_cast<daal::CpuType>(cpu_type);
    switch (cpu) {
#if defined(TARGET_X86_64)
        case daal::sse2: return cpu_extension::sse2;
        case daal::sse42: return cpu_extension::sse42;
        case daal::avx2: return cpu_extension::avx2;
        case daal::avx512: { std::cout << "system_parameters_impl, cpu_extension::avx512 " << std::endl << std::flush; return cpu_extension::avx512; }
#elif defined(TARGET_ARM)
        case daal::sve: return cpu_extension::sve;
#elif defined(TARGET_RISCV64)
        case daal::rv64: return cpu_extension::rv64;
#endif
    }
    std::cout << "system_parameters_impl, cpu_extension::none " << std::endl << std::flush;
    return cpu_extension::none;
}

ONEDAL_EXPORT cpu_extension detect_top_cpu_extension() {
    const auto daal_cpu = __daal_serv_cpu_detect(0);
    return from_daal_cpu_type(daal_cpu);
}

ONEDAL_EXPORT cpu_extension detect_onedal_cpu_extension() {
    const auto daal_cpu = daal_enabled_cpu_detect();
    return from_daal_cpu_type(daal_cpu);
}

uint64_t detect_cpu_features() {
    return daal_serv_cpu_feature_detect();
}

} // namespace v1
} // namespace oneapi::dal::detail
