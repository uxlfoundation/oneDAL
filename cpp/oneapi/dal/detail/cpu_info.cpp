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

#include "oneapi/dal/detail/cpu_info.hpp"

#if defined(TARGET_X86_64)
#include "oneapi/dal/detail/cpu_info_x86_impl.hpp"
#elif defined(TARGET_ARM)
#include "oneapi/dal/detail/cpu_info_arm_impl.hpp"
#elif defined(TARGET_RISCV64)
#include "oneapi/dal/detail/cpu_info_riscv64_impl.hpp"
#endif

#include <daal/src/services/service_defines.h>

namespace oneapi::dal::detail {
namespace v1 {

cpu_info::cpu_info()
        :
#if defined(TARGET_X86_64)
          impl_(new cpu_info_x86)
#elif defined(TARGET_ARM)
          impl_(new cpu_info_arm)
#elif defined(TARGET_RISCV64)
          impl_(new cpu_info_riscv64)
#endif
{
}

cpu_info::cpu_info(const cpu_extension cpu_extension_)
        :
#if defined(TARGET_X86_64)
          impl_(new cpu_info_x86(cpu_extension_))
#elif defined(TARGET_ARM)
          impl_(new cpu_info_arm(cpu_extension_))
#elif defined(TARGET_RISCV64)
          impl_(new cpu_info_riscv64(cpu_extension_))
#endif
{
}

detail::cpu_vendor cpu_info::get_cpu_vendor() const {
    return impl_->get_cpu_vendor();
}

detail::cpu_extension cpu_info::get_top_cpu_extension() const {
    return impl_->get_top_cpu_extension();
}

std::string cpu_info::dump() const {
    return impl_->dump();
}

} // namespace v1
} // namespace oneapi::dal::detail
