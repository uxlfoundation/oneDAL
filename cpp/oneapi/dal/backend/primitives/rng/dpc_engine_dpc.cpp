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

template <typename Distribution, typename Type>
sycl::event generate_rng(Distribution& distr,
                         dpc_engine& engine_,
                         std::int64_t count,
                         Type* dst,
                         const event_vector& deps) {
    switch (engine_.get_device_engine_base_ptr()->get_engine_method()) {
        case engine_method::mt2203: {
            auto& dpc_engine =
                *(dynamic_cast<gen_mt2203*>(engine_.get_device_engine_base_ptr().get()))->get();
            return oneapi::mkl::rng::generate(distr, dpc_engine, count, dst, deps);
        }
        case engine_method::mcg59: {
            auto& dpc_engine =
                *(dynamic_cast<gen_mcg59*>(engine_.get_device_engine_base_ptr().get()))->get();
            return oneapi::mkl::rng::generate(distr, dpc_engine, count, dst, deps);
        }
        case engine_method::mrg32k3a: {
            auto& dpc_engine =
                *(dynamic_cast<gen_mrg32k*>(engine_.get_device_engine_base_ptr().get()))->get();
            return oneapi::mkl::rng::generate(distr, dpc_engine, count, dst, deps);
        }
        case engine_method::philox4x32x10: {
            auto& dpc_engine =
                *(dynamic_cast<gen_philox*>(engine_.get_device_engine_base_ptr().get()))->get();
            return oneapi::mkl::rng::generate(distr, dpc_engine, count, dst, deps);
        }
        case engine_method::mt19937: {
            auto& dpc_engine =
                *(dynamic_cast<gen_mt19937*>(engine_.get_device_engine_base_ptr().get()))->get();
            return oneapi::mkl::rng::generate(distr, dpc_engine, count, dst, deps);
        }
        default: throw std::runtime_error("Unsupported engine type in generate_rng");
    }
}

template <typename Type>
void uniform(sycl::queue& queue,
             std::int64_t count,
             Type* dst,
             dpc_engine& engine_,
             Type a,
             Type b,
             const event_vector& deps) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) == sycl::usm::alloc::host) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    oneapi::mkl::rng::uniform<Type> distr(a, b);
    auto event = generate_rng(distr, engine_, count, dst, deps);
    event.wait_and_throw();
    engine_.skip_ahead_cpu(count);
}

//Currently only CPU impl
template <typename Type>
void uniform_without_replacement(sycl::queue& queue,
                                 std::int64_t count,
                                 Type* dst,
                                 Type* buffer,
                                 dpc_engine& engine_,
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
template <typename Type>
void shuffle(sycl::queue& queue,
             std::int64_t count,
             Type* dst,
             dpc_engine& engine_,
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

#define INSTANTIATE_UNIFORM(F)                               \
    template ONEDAL_EXPORT void uniform(sycl::queue& queue,  \
                                        std::int64_t count_, \
                                        F* dst,              \
                                        dpc_engine& engine_, \
                                        F a,                 \
                                        F b,                 \
                                        const event_vector& deps);

INSTANTIATE_UNIFORM(float)
INSTANTIATE_UNIFORM(double)
INSTANTIATE_UNIFORM(std::int32_t)

#define INSTANTIATE_UWR(F)                                                       \
    template ONEDAL_EXPORT void uniform_without_replacement(sycl::queue& queue,  \
                                                            std::int64_t count_, \
                                                            F* dst,              \
                                                            F* buff,             \
                                                            dpc_engine& engine_, \
                                                            F a,                 \
                                                            F b,                 \
                                                            const event_vector& deps);

INSTANTIATE_UWR(float)
INSTANTIATE_UWR(double)
INSTANTIATE_UWR(std::int32_t)

#define INSTANTIATE_SHUFFLE(F)                               \
    template ONEDAL_EXPORT void shuffle(sycl::queue& queue,  \
                                        std::int64_t count_, \
                                        F* dst,              \
                                        dpc_engine& engine_, \
                                        const event_vector& deps);

INSTANTIATE_SHUFFLE(std::int32_t)

} // namespace oneapi::dal::backend::primitives
