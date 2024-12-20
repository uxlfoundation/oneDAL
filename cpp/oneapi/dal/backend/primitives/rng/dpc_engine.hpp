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
#include <oneapi/mkl.hpp>

namespace mkl = oneapi::mkl;
namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <engine_method EngineType>
struct dpc_engine_type;

template <>
struct dpc_engine_type<engine_method::mt2203> {
    using type = oneapi::mkl::rng::mt2203;
};

template <>
struct dpc_engine_type<engine_method::mcg59> {
    using type = oneapi::mkl::rng::mcg59;
};

template <>
struct dpc_engine_type<engine_method::mt19937> {
    using type = oneapi::mkl::rng::mt19937;
};

template <>
struct dpc_engine_type<engine_method::mrg32k3a> {
    using type = oneapi::mkl::rng::mrg32k3a;
};

template <>
struct dpc_engine_type<engine_method::philox4x32x10> {
    using type = oneapi::mkl::rng::philox4x32x10;
};

/// A class that provides a unified interface for random number generation on both CPU and GPU devices.
///
/// This class serves as a wrapper for random number generators (RNGs) that supports different engine types,
/// enabling efficient random number generation on heterogeneous platforms using SYCL. It integrates a host
/// (CPU) engine and a device (GPU) engine, allowing operations to be executed seamlessly on the appropriate
/// device.
///
/// @tparam EngineType The RNG engine type to be used. Defaults to `engine_method::mt2203`.
///
/// @param[in] queue The SYCL queue used to manage device operations.
/// @param[in] seed  The initial seed for the random number generator. Defaults to `777`.
///
/// The class provides functionality to skip ahead in the RNG sequence, retrieve engine states, and
/// manage host and device engines independently. Support for `skip_ahead` on GPU is currently limited for
/// some engine types.
template <engine_method EngineType = engine_method::mt2203>
class dpc_engine {
public:
    using dpc_engine_t = typename dpc_engine_type<EngineType>::type;

    explicit dpc_engine(sycl::queue& queue, std::int64_t seed = 777)
            : q(queue),
              host_engine_(initialize_host_engine(seed)),
              dpc_engine_(initialize_dpc_engine(queue, seed)),
              impl_(dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(
                  host_engine_.get())) {
        if (!impl_) {
            throw std::domain_error("RNG engine is not supported");
        }
    }

    virtual ~dpc_engine() = default;

    void* get_host_engine_state() const {
        return impl_->getState();
    }

    auto& get_cpu_engine() {
        return host_engine_;
    }

    auto& get_gpu_engine() {
        return dpc_engine_;
    }

    void skip_ahead_cpu(size_t nSkip) {
        host_engine_->skipAhead(nSkip);
    }

    void skip_ahead_gpu(size_t nSkip) {
        // Will be supported in the next oneMKL release.
        if constexpr (EngineType == engine_method::mt2203) {
        }
        else {
            skip_ahead(dpc_engine_, nSkip);
        }
    }

    sycl::queue& get_queue() {
        return q;
    }

private:
    daal::algorithms::engines::EnginePtr initialize_host_engine(std::int64_t seed) {
        switch (EngineType) {
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
            default: throw std::invalid_argument("Unsupported engine type");
        }
    }

    dpc_engine_t initialize_dpc_engine(sycl::queue& queue, std::int64_t seed) {
        if constexpr (EngineType == engine_method::mt2203) {
            return dpc_engine_t(
                queue,
                seed,
                0); // Aligns CPU and GPU results for mt2203, impacts the performance.
        }
        else {
            return dpc_engine_t(queue, seed);
        }
    }
    sycl::queue q;
    daal::algorithms::engines::EnginePtr host_engine_;
    dpc_engine_t dpc_engine_;
    daal::algorithms::engines::internal::BatchBaseImpl* impl_;
};

#endif
} // namespace oneapi::dal::backend::primitives
