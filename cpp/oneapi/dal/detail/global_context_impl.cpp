/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/cpu.hpp"
#include "oneapi/dal/detail/global_context_impl.hpp"
#include <daal/src/services/service_defines.h>

namespace oneapi::dal::detail {
namespace v1 {

global_context_impl::global_context_impl() : cpu_info_(dal::detect_top_cpu_extension()) {
    using daal::services::Environment;
    // Call to `getCpuId` changes global settings, in particular,
    // changes default number of threads in the threading layer
    Environment::getInstance()->getCpuId();
}

} // namespace v1
} // namespace oneapi::dal::detail