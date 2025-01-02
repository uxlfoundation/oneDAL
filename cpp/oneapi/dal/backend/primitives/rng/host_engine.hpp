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

#include "oneapi/dal/backend/primitives/rng/rng_types.hpp"
#include "oneapi/dal/backend/primitives/rng/rng.hpp"
#include "oneapi/dal/backend/primitives/rng/utils.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace oneapi::dal::backend::primitives {

/// A class that provides an interface for random number generation on the host (CPU) only.
///
/// This class serves as a wrapper for host-based random number generators (RNGs), supporting multiple engine
/// types for flexible and efficient random number generation on CPU. It abstracts the underlying engine
/// implementation and provides an interface to manage and retrieve the engine's state.
///
/// @tparam EngineType The RNG engine type to be used. Defaults to `engine_method::mt2203`.
///
/// @param[in] seed  The initial seed for the random number generator. Defaults to `777`.
///
/// @note The class only supports host-based RNG and does not require a SYCL queue or device context.
template <engine_method EngineType = engine_method::mt2203>
class host_engine {
public:
    explicit host_engine(std::int64_t seed = 777)
            : host_engine_(initialize_host_engine(seed)),
              impl_(dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(
                  host_engine_.get())) {
        if (!impl_) {
            throw std::domain_error("RNG engine is not supported");
        }
    }

    explicit host_engine(const daal::algorithms::engines::EnginePtr& eng) : host_engine_(eng) {
        impl_ = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(eng.get());
        if (!impl_) {
            throw domain_error(dal::detail::error_messages::rng_engine_is_not_supported());
        }
    }

    host_engine& operator=(const daal::algorithms::engines::EnginePtr& eng) {
        host_engine_ = eng;
        impl_ = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(eng.get());
        if (!impl_) {
            throw domain_error(dal::detail::error_messages::rng_engine_is_not_supported());
        }

        return *this;
    }

    virtual ~host_engine() = default;

    void* get_host_engine_state() const {
        return impl_->getState();
    }

    auto& get_host_engine() {
        return host_engine_;
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

    daal::algorithms::engines::EnginePtr host_engine_;
    daal::algorithms::engines::internal::BatchBaseImpl* impl_;
};

} // namespace oneapi::dal::backend::primitives
