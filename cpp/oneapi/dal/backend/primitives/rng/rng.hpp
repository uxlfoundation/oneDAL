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

#include "oneapi/dal/backend/primitives/rng/engine_cpu.hpp"

#ifdef ONEDAL_DATA_PARALLEL

#include "oneapi/dal/backend/primitives/rng/engine_gpu.hpp"

#endif

namespace oneapi::dal::backend::primitives {
template <typename Type, typename Size = std::int64_t>
class rng {
public:
    rng() = default;
    ~rng() = default;

    template <engine_list EngineType>
    void uniform_cpu(Size count, Type* dst, daal_engine<EngineType> daal_engine, Type a, Type b) {
        auto state = daal_engine.get_cpu_engine_state();
        uniform_dispatcher::uniform_by_cpu<Type>(count, dst, state, a, b);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <engine_list EngineType>
    void uniform_cpu(Size count, Type* dst, onedal_engine<EngineType>& engine_, Type a, Type b) {
        if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
            sycl::usm::alloc::device) {
            throw domain_error(dal::detail::error_messages::unsupported_data_type());
        }
        auto state = engine_.get_cpu_engine_state();
        engine_.skip_ahead_gpu(count);
        uniform_dispatcher::uniform_by_cpu<Type>(count, dst, state, a, b);
    }
#endif

    template <engine_list EngineType>
    void uniform_without_replacement_cpu(Size count,
                                         Type* dst,
                                         Type* buffer,
                                         daal_engine<EngineType> daal_engine,
                                         Type a,
                                         Type b) {
        auto state = daal_engine.get_cpu_engine_state();
        uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count,
                                                                     dst,
                                                                     buffer,
                                                                     state,
                                                                     a,
                                                                     b);
    }
#ifdef ONEDAL_DATA_PARALLEL
    template <engine_list EngineType>
    void uniform_without_replacement_cpu(Size count,
                                         Type* dst,
                                         Type* buffer,
                                         onedal_engine<EngineType>& engine_,
                                         Type a,
                                         Type b) {
        if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
            sycl::usm::alloc::device) {
            throw domain_error(dal::detail::error_messages::unsupported_data_type());
        }
        void* state = engine_.get_cpu_engine_state();
        engine_.skip_ahead_gpu(count);
        uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count,
                                                                     dst,
                                                                     buffer,
                                                                     state,
                                                                     a,
                                                                     b);
    }
#endif

    template <engine_list EngineType,
              typename T = Type,
              typename = std::enable_if_t<std::is_integral_v<T>>>
    void shuffle_cpu(Size count, Type* dst, daal_engine<EngineType> daal_engine) {
        Type idx[2];
        auto state = daal_engine.get_cpu_engine_state();
        for (Size i = 0; i < count; ++i) {
            uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
            std::swap(dst[idx[0]], dst[idx[1]]);
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <engine_list EngineType,
              typename T = Type,
              typename = std::enable_if_t<std::is_integral_v<T>>>
    void shuffle_cpu(Size count, Type* dst, onedal_engine<EngineType>& engine_) {
        Type idx[2];
        if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
            sycl::usm::alloc::device) {
            throw domain_error(dal::detail::error_messages::unsupported_data_type());
        }
        void* state = engine_.get_cpu_engine_state();
        engine_.skip_ahead_gpu(count);

        for (Size i = 0; i < count; ++i) {
            uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
            std::swap(dst[idx[0]], dst[idx[1]]);
        }
    }
#endif

#ifdef ONEDAL_DATA_PARALLEL
    template <engine_list EngineType>
    void uniform_gpu(sycl::queue& queue,
                     Size count,
                     Type* dst,
                     onedal_engine<EngineType>& engine_,
                     Type a,
                     Type b,
                     const event_vector& deps = {});

    template <engine_list EngineType>
    void uniform_without_replacement_gpu(sycl::queue& queue,
                                         Size count,
                                         Type* dst,
                                         Type* buffer,
                                         onedal_engine<EngineType>& engine_,
                                         Type a,
                                         Type b,
                                         const event_vector& deps = {});

    template <engine_list EngineType>
    void shuffle_gpu(sycl::queue& queue,
                     Size count,
                     Type* dst,
                     onedal_engine<EngineType>& engine_,
                     const event_vector& deps = {});
#endif
};

}; // namespace oneapi::dal::backend::primitives