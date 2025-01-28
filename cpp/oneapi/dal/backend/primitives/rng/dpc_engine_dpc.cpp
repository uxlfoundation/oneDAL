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

#include "oneapi/dal/backend/primitives/rng/dpc_engine.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include <oneapi/mkl.hpp>

namespace oneapi::dal::backend::primitives {

namespace bk = oneapi::dal::backend;

template <typename Type, engine_method EngineType>
void uniform(sycl::queue& queue,
             std::int64_t count,
             Type* dst,
             dpc_engine<EngineType>& engine_,
             Type a,
             Type b,
             const event_vector& deps) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) == sycl::usm::alloc::host) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    oneapi::mkl::rng::uniform<Type> distr(a, b);
    auto event = oneapi::mkl::rng::generate(distr, engine_.get_gpu_engine(), count, dst, { deps });
    event.wait_and_throw();
    engine_.skip_ahead_cpu(count);
}

//Currently only CPU impl
template <typename Type, engine_method EngineType>
void uniform_without_replacement(sycl::queue& queue,
                                 std::int64_t count,
                                 Type* dst,
                                 Type* buffer,
                                 dpc_engine<EngineType>& engine_,
                                 Type a,
                                 Type b,
                                 const event_vector& deps) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    void* state = engine_.get_host_engine_state();
    engine_.skip_ahead_gpu(count);
    uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count, dst, buffer, state, a, b);
}

//Currently only CPU impl
template <typename Type, engine_method EngineType>
void shuffle(sycl::queue& queue,
             std::int64_t count,
             Type* dst,
             dpc_engine<EngineType>& engine_,
             const event_vector& deps) {
    Type idx[2];
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    void* state = engine_.get_host_engine_state();
    engine_.skip_ahead_gpu(count);

    for (std::int64_t i = 0; i < count; ++i) {
        uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
        std::swap(dst[idx[0]], dst[idx[1]]);
    }
}

#define INSTANTIATE_UNIFORM(F, EngineType)                               \
    template ONEDAL_EXPORT void uniform(sycl::queue& queue,              \
                                        std::int64_t count_,             \
                                        F* dst,                          \
                                        dpc_engine<EngineType>& engine_, \
                                        F a,                             \
                                        F b,                             \
                                        const event_vector& deps);

INSTANTIATE_UNIFORM(float, engine_method::mt2203)
INSTANTIATE_UNIFORM(float, engine_method::mcg59)
INSTANTIATE_UNIFORM(float, engine_method::mrg32k3a)
INSTANTIATE_UNIFORM(float, engine_method::philox4x32x10)
INSTANTIATE_UNIFORM(float, engine_method::mt19937)
INSTANTIATE_UNIFORM(double, engine_method::mt2203)
INSTANTIATE_UNIFORM(double, engine_method::mcg59)
INSTANTIATE_UNIFORM(double, engine_method::mrg32k3a)
INSTANTIATE_UNIFORM(double, engine_method::philox4x32x10)
INSTANTIATE_UNIFORM(double, engine_method::mt19937)
INSTANTIATE_UNIFORM(std::int32_t, engine_method::mt2203)
INSTANTIATE_UNIFORM(std::int32_t, engine_method::mcg59)
INSTANTIATE_UNIFORM(std::int32_t, engine_method::mrg32k3a)
INSTANTIATE_UNIFORM(std::int32_t, engine_method::philox4x32x10)
INSTANTIATE_UNIFORM(std::int32_t, engine_method::mt19937)

#define INSTANTIATE_UWR(F, EngineType)                                                       \
    template ONEDAL_EXPORT void uniform_without_replacement(sycl::queue& queue,              \
                                                            std::int64_t count_,             \
                                                            F* dst,                          \
                                                            F* buff,                         \
                                                            dpc_engine<EngineType>& engine_, \
                                                            F a,                             \
                                                            F b,                             \
                                                            const event_vector& deps);

INSTANTIATE_UWR(float, engine_method::mt2203)
INSTANTIATE_UWR(float, engine_method::mcg59)
INSTANTIATE_UWR(float, engine_method::mrg32k3a)
INSTANTIATE_UWR(float, engine_method::philox4x32x10)
INSTANTIATE_UWR(float, engine_method::mt19937)
INSTANTIATE_UWR(double, engine_method::mt2203)
INSTANTIATE_UWR(double, engine_method::mcg59)
INSTANTIATE_UWR(double, engine_method::mrg32k3a)
INSTANTIATE_UWR(double, engine_method::philox4x32x10)
INSTANTIATE_UWR(double, engine_method::mt19937)
INSTANTIATE_UWR(std::int32_t, engine_method::mt2203)
INSTANTIATE_UWR(std::int32_t, engine_method::mcg59)
INSTANTIATE_UWR(std::int32_t, engine_method::mrg32k3a)
INSTANTIATE_UWR(std::int32_t, engine_method::philox4x32x10)
INSTANTIATE_UWR(std::int32_t, engine_method::mt19937)

#define INSTANTIATE_SHUFFLE(F, EngineType)                               \
    template ONEDAL_EXPORT void shuffle(sycl::queue& queue,              \
                                        std::int64_t count_,             \
                                        F* dst,                          \
                                        dpc_engine<EngineType>& engine_, \
                                        const event_vector& deps);

INSTANTIATE_SHUFFLE(std::int32_t, engine_method::mt2203)
INSTANTIATE_SHUFFLE(std::int32_t, engine_method::mcg59)
INSTANTIATE_SHUFFLE(std::int32_t, engine_method::mrg32k3a)
INSTANTIATE_SHUFFLE(std::int32_t, engine_method::philox4x32x10)
INSTANTIATE_SHUFFLE(std::int32_t, engine_method::mt19937)

} // namespace oneapi::dal::backend::primitives
