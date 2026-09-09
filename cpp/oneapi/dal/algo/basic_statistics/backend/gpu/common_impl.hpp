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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::basic_statistics::backend {

namespace be = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float>
inline array<Float> flatten_result_array(sycl::queue& q,
                                         const pr::ndarray<Float, 1>& ndarr,
                                         alloc_kind alloc,
                                         const be::event_vector& deps = {}) {
    if (alloc == alloc_kind::non_usm) {
        return ndarr.to_host(q, deps).flatten();
    }
    return ndarr.flatten(q, deps);
}

} // namespace oneapi::dal::basic_statistics::backend

#endif
