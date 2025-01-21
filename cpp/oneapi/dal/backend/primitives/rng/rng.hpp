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

#ifdef ONEDAL_DATA_PARALLEL

#include "oneapi/dal/backend/primitives/rng/dpc_engine.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Type, typename Size, engine_method EngineType>
void uniform(Size count, Type* dst, dpc_engine<EngineType>& engine_, Type a, Type b) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    auto state = engine_.get_host_engine_state();
    uniform_dispatcher::uniform_by_cpu<Type>(count, dst, state, a, b);
    engine_.skip_ahead_gpu(count);
}

template <typename Type, typename Size, engine_method EngineType>
void uniform_without_replacement(Size count,
                                 Type* dst,
                                 Type* buffer,
                                 dpc_engine<EngineType>& engine_,
                                 Type a,
                                 Type b) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    void* state = engine_.get_host_engine_state();
    uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count, dst, buffer, state, a, b);
    engine_.skip_ahead_gpu(count);
}

template <typename Type,
          typename Size,
          engine_method EngineType,
          typename T = Type,
          typename = std::enable_if_t<std::is_integral_v<T>>>
void shuffle(Size count, Type* dst, dpc_engine<EngineType>& engine_) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    Type idx[2];
    void* state = engine_.get_host_engine_state();
    for (Size i = 0; i < count; ++i) {
        uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
        std::swap(dst[idx[0]], dst[idx[1]]);
    }
    engine_.skip_ahead_gpu(count);
}

template <typename Type, typename Size, engine_method EngineType>
void uniform(sycl::queue& queue,
             Size count,
             Type* dst,
             dpc_engine<EngineType>& engine_,
             Type a,
             Type b,
             const event_vector& deps = {});

template <typename Type, typename Size, engine_method EngineType>
void uniform_without_replacement(sycl::queue& queue,
                                 Size count,
                                 Type* dst,
                                 Type* buffer,
                                 dpc_engine<EngineType>& engine_,
                                 Type a,
                                 Type b,
                                 const event_vector& deps = {});

template <typename Type, typename Size, engine_method EngineType>
void shuffle(sycl::queue& queue,
             Size count,
             Type* dst,
             dpc_engine<EngineType>& engine_,
             const event_vector& deps = {});

}; // namespace oneapi::dal::backend::primitives

#endif
