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

#include "oneapi/dal/backend/primitives/rng/utils.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_types.hpp"

#include <daal/include/algorithms/engines/mt2203/mt2203.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/include/algorithms/engines/mrg32k3a/mrg32k3a.h>
#include <daal/include/algorithms/engines/philox4x32x10/philox4x32x10.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>

#include <oneapi/mkl.hpp>

namespace mkl = oneapi::mkl;
namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

class gen_base {
public:
    virtual ~gen_base() = default;
    virtual engine_method get_engine_method() const = 0;
    virtual void skip_ahead_gpu(std::int64_t nSkip) = 0;
};

class gen_mt2203 : public gen_base {
public:
    explicit gen_mt2203() = delete;
    gen_mt2203(sycl::queue queue, std::int64_t seed) : _gen(queue, seed, 0) {}

    engine_method get_engine_method() const override {
        return engine_method::mt2203;
    }
    void skip_ahead_gpu(std::int64_t nSkip) override {
        //skip;
    }
    oneapi::mkl::rng::mt2203* get() {
        return &_gen;
    }

protected:
    oneapi::mkl::rng::mt2203 _gen;
};

class gen_philox : public gen_base {
public:
    explicit gen_philox() = delete;
    gen_philox(sycl::queue queue, std::int64_t seed) : _gen(queue, seed) {}
    engine_method get_engine_method() const override {
        return engine_method::philox4x32x10;
    }
    void skip_ahead_gpu(std::int64_t nSkip) override {
        skip_ahead(_gen, nSkip);
    }
    oneapi::mkl::rng::philox4x32x10* get() {
        return &_gen;
    }

protected:
    oneapi::mkl::rng::philox4x32x10 _gen;
};

class gen_mrg32k : public gen_base {
public:
    explicit gen_mrg32k() = delete;
    gen_mrg32k(sycl::queue queue, std::int64_t seed) : _gen(queue, seed) {}
    engine_method get_engine_method() const override {
        return engine_method::mrg32k3a;
    }
    void skip_ahead_gpu(std::int64_t nSkip) override {
        skip_ahead(_gen, nSkip);
    }
    oneapi::mkl::rng::mrg32k3a* get() {
        return &_gen;
    }

protected:
    oneapi::mkl::rng::mrg32k3a _gen;
};

class gen_mt19937 : public gen_base {
public:
    explicit gen_mt19937() = delete;
    gen_mt19937(sycl::queue queue, std::int64_t seed) : _gen(queue, seed) {}
    engine_method get_engine_method() const override {
        return engine_method::mt19937;
    }
    void skip_ahead_gpu(std::int64_t nSkip) override {
        skip_ahead(_gen, nSkip);
    }
    oneapi::mkl::rng::mt19937* get() {
        return &_gen;
    }

protected:
    oneapi::mkl::rng::mt19937 _gen;
};

class gen_mcg59 : public gen_base {
public:
    explicit gen_mcg59() = delete;
    gen_mcg59(sycl::queue queue, std::int64_t seed) : _gen(queue, seed) {}
    engine_method get_engine_method() const override {
        return engine_method::mcg59;
    }
    void skip_ahead_gpu(std::int64_t nSkip) override {
        skip_ahead(_gen, nSkip);
    }
    oneapi::mkl::rng::mcg59* get() {
        return &_gen;
    }

protected:
    oneapi::mkl::rng::mcg59 _gen;
};

/// A class that provides a unified interface for random number generation on both CPU and GPU devices.
///
/// This class serves as a wrapper for random number generators (RNGs) that supports different engine types,
/// enabling efficient random number generation on heterogeneous platforms using SYCL. It integrates a host
/// (CPU) engine and a device (GPU) engine, allowing operations to be executed seamlessly on the appropriate
/// device.
///
/// The class provides functionality to skip ahead in the RNG sequence, retrieve engine states, and
/// manage host and device engines independently. Support for `skip_ahead` on GPU is currently limited for
/// some engine types.
class dpc_engine {
public:
    /// @param[in] queue The SYCL queue used to manage device operations.
    /// @param[in] seed  The initial seed for the random number generator. Defaults to `777`.
    explicit dpc_engine(sycl::queue& queue,
                        std::int64_t seed = 777,
                        engine_method method = engine_method::mt2203)
            : q(queue),
              host_engine_(initialize_host_engine(seed, method)),
              impl_(dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(
                  host_engine_.get())) {
        initialize_dpc_engine(queue, seed, method);
        if (!impl_) {
            throw std::domain_error("RNG engine is not supported");
        }
    }

    dpc_engine& operator=(const dpc_engine& other);

    virtual ~dpc_engine() = default;

    void* get_host_engine_state() const {
        return impl_->getState();
    }

    auto get_device_engine_base_ptr() {
        return engine_;
    }

    void skip_ahead_cpu(size_t nSkip) {
        host_engine_->skipAhead(nSkip);
    }

    void skip_ahead_gpu(size_t nSkip) {
        engine_->skip_ahead_gpu(nSkip);
    }

    sycl::queue& get_queue() {
        return q;
    }

private:
    daal::algorithms::engines::EnginePtr initialize_host_engine(std::int64_t seed,
                                                                engine_method method) {
        switch (method) {
            case engine_method::mt2203:
                return daal::algorithms::engines::mt2203::Batch<>::create(seed);
            case engine_method::mcg59:
                return daal::algorithms::engines::mcg59::Batch<>::create(seed);
            case engine_method::mrg32k3a:
                return daal::algorithms::engines::mrg32k3a::Batch<>::create(seed);
            case engine_method::philox4x32x10:
                return daal::algorithms::engines::philox4x32x10::Batch<>::create(seed);
            case engine_method::mt19937:
                return daal::algorithms::engines::mt19937::Batch<>::create(seed);
            default: throw std::invalid_argument("Unsupported engine type 2");
        }
    }

    void initialize_dpc_engine(sycl::queue& queue, std::int64_t seed, engine_method method) {
        switch (method) {
            case engine_method::mt2203: engine_ = std::make_shared<gen_mt2203>(queue, seed); break;
            case engine_method::mcg59: engine_ = std::make_shared<gen_mcg59>(queue, seed); break;
            case engine_method::mrg32k3a:
                engine_ = std::make_shared<gen_mrg32k>(queue, seed);
                break;
            case engine_method::philox4x32x10:
                engine_ = std::make_shared<gen_philox>(queue, seed);
                break;
            case engine_method::mt19937:
                engine_ = std::make_shared<gen_mt19937>(queue, seed);
                break;
            default: throw std::invalid_argument("Unsupported engine type 1");
        }
    }
    sycl::queue q;
    daal::algorithms::engines::EnginePtr host_engine_;
    std::shared_ptr<gen_base> engine_;
    daal::algorithms::engines::internal::BatchBaseImpl* impl_;
};

template <typename Type>
void uniform(std::int64_t count, Type* dst, dpc_engine& engine_, Type a, Type b) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    auto state = engine_.get_host_engine_state();
    uniform_dispatcher::uniform_by_cpu<Type>(count, dst, state, a, b);
    engine_.skip_ahead_gpu(count);
}

template <typename Type>
void uniform_without_replacement(std::int64_t count,
                                 Type* dst,
                                 Type* buffer,
                                 dpc_engine& engine_,
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

template <typename Type, typename T = Type, typename = std::enable_if_t<std::is_integral_v<T>>>
void shuffle(std::int64_t count, Type* dst, dpc_engine& engine_) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    Type idx[2];
    void* state = engine_.get_host_engine_state();
    for (std::int64_t i = 0; i < count; ++i) {
        uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
        std::swap(dst[idx[0]], dst[idx[1]]);
    }
    engine_.skip_ahead_gpu(count);
}

template <typename Type>
void uniform(sycl::queue& queue,
             std::int64_t count,
             Type* dst,
             dpc_engine& engine_,
             Type a,
             Type b,
             const event_vector& deps = {});

template <typename Type>
void uniform_without_replacement(sycl::queue& queue,
                                 std::int64_t count,
                                 Type* dst,
                                 Type* buffer,
                                 dpc_engine& engine_,
                                 Type a,
                                 Type b,
                                 const event_vector& deps = {});

template <typename Type>
void shuffle(sycl::queue& queue,
             std::int64_t count,
             Type* dst,
             dpc_engine& engine_,
             const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
