/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/atomic.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::decision_forest::backend {

using sycl::plus;
using sycl::minimum;
using sycl::maximum;

template <typename Data>
using local_accessor_rw_t = sycl::local_accessor<Data, 1>;

template <typename T, typename Index>
T* get_buf_ptr(byte_t** buf_ptr, Index elem_count) {
    T* res_ptr = reinterpret_cast<T*>(*buf_ptr);
    (*buf_ptr) += elem_count * sizeof(T);
    return res_ptr;
}

} // namespace oneapi::dal::decision_forest::backend

#endif
