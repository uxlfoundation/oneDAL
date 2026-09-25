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

#include "oneapi/dal/detail/cpu_info_impl.hpp"

#include <daal/src/services/service_defines.h>

namespace oneapi::dal::detail {
namespace v1 {

class cpu_info_x86 : public cpu_info_impl {
public:
    cpu_info_x86() {
        info_["top_cpu_extension"] = detect_top_cpu_extension();
        info_["onedal_cpu_extension"] = detect_onedal_cpu_extension();
        info_["vendor"] = (daal_check_is_intel_cpu() ? cpu_vendor::intel : cpu_vendor::amd);
        info_["cpu_features"] = detect_cpu_features();
    }

    explicit cpu_info_x86(const cpu_extension cpu_extension) {
        info_["top_cpu_extension"] = detect_top_cpu_extension();
        info_["onedal_cpu_extension"] = cpu_extension;
        info_["vendor"] = (daal_check_is_intel_cpu() ? cpu_vendor::intel : cpu_vendor::amd);
        info_["cpu_features"] = detect_cpu_features();
    }
};

} // namespace v1

namespace v2 {

/// The v1 implementation cannot be reused as a base class here: it implements
/// `v1::cpu_info_iface` only, while `v2::cpu_info` stores its implementation as a
/// `v2::cpu_info_iface`. The detection logic is therefore repeated on top of
/// `v2::cpu_info_impl`, which keeps the v1 class, and thus the ABI, untouched.
class cpu_info_x86 : public cpu_info_impl {
public:
    cpu_info_x86() {
        info_["top_cpu_extension"] = detect_top_cpu_extension();
        info_["onedal_cpu_extension"] = detect_onedal_cpu_extension();
        info_["vendor"] = (daal_check_is_intel_cpu() ? cpu_vendor::intel : cpu_vendor::amd);
        info_["cpu_features"] = detect_cpu_features();
        detect_cache_sizes();
    }

    explicit cpu_info_x86(const cpu_extension cpu_extension) {
        info_["top_cpu_extension"] = detect_top_cpu_extension();
        info_["onedal_cpu_extension"] = cpu_extension;
        info_["vendor"] = (daal_check_is_intel_cpu() ? cpu_vendor::intel : cpu_vendor::amd);
        info_["cpu_features"] = detect_cpu_features();
        detect_cache_sizes();
    }
};

} // namespace v2
using v2::cpu_info_x86;
} // namespace oneapi::dal::detail
