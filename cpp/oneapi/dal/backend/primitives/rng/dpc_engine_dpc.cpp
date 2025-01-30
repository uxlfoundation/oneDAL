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
    sycl::event event;
    switch (engine_.get_gpu_engine()->get_engine_method()) {
        case engine_method::mt2203: {
            Helper<engine_method::mt2203> helper;
            event = oneapi::mkl::rng::generate(distr,
                                               *(helper.getter(engine_.get_gpu_engine().get())),
                                               count,
                                               dst,
                                               { deps });
            break;
        }
        case engine_method::mcg59: {
            Helper<engine_method::mcg59> helper;
            event = oneapi::mkl::rng::generate(distr,
                                               *(helper.getter(engine_.get_gpu_engine().get())),
                                               count,
                                               dst,
                                               { deps });
            break;
        }
        case engine_method::mrg32k3a: {
            Helper<engine_method::mrg32k3a> helper;
            event = oneapi::mkl::rng::generate(distr,
                                               *(helper.getter(engine_.get_gpu_engine().get())),
                                               count,
                                               dst,
                                               { deps });
            break;
        }
        case engine_method::philox4x32x10: {
            Helper<engine_method::philox4x32x10> helper;
            event = oneapi::mkl::rng::generate(distr,
                                               *(helper.getter(engine_.get_gpu_engine().get())),
                                               count,
                                               dst,
                                               { deps });
            break;
        }
        case engine_method::mt19937: {
            Helper<engine_method::mt19937> helper;
            event = oneapi::mkl::rng::generate(distr,
                                               *(helper.getter(engine_.get_gpu_engine().get())),
                                               count,
                                               dst,
                                               { deps });
            break;
        }
        default: throw std::invalid_argument("Unsupported engine type 3");
    }
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
