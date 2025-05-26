/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/backend/primitives/rng/host_engine.hpp"

#include <vector>

namespace oneapi::dal::backend::primitives {

class host_engine_collection {
public:
    explicit host_engine_collection(std::int64_t count,
                                    std::int64_t seed = 777,
                                    engine_type_internal method = engine_type_internal::mt2203)
            : count_(count),
              engine_(initialize_host_engine(seed, method)),
              params_(count),
              technique_(daal::algorithms::engines::internal::family),
              daal_engine_list_(count) {}

    template <typename Op>
    std::vector<host_engine> operator()(Op&& op) {
        daal::services::Status status;
        for (std::int64_t i = 0; i < count_; ++i) {
            op(i, params_.nSkip[i]);
        }
        select_parallelization_technique(technique_);
        daal::algorithms::engines::internal::EnginesCollection<daal::sse2> host_engine_collection(
            engine_,
            technique_,
            params_,
            daal_engine_list_,
            &status);
        if (!status) {
            dal::backend::interop::status_to_exception(status);
        }

        std::vector<host_engine> engine_list(count_);
        for (std::int64_t i = 0; i < count_; ++i) {
            engine_list[i] = daal_engine_list_[i];
        }

        //copy elision
        return engine_list;
    }

private:
    daal::algorithms::engines::EnginePtr initialize_host_engine(std::int64_t seed,
                                                                engine_type_internal method) {
        switch (method) {
            case engine_type_internal::mt2203:
                return daal::algorithms::engines::mt2203::Batch<>::create(seed);
            case engine_type_internal::mcg59:
                return daal::algorithms::engines::mcg59::Batch<>::create(seed);
            case engine_type_internal::mrg32k3a:
                return daal::algorithms::engines::mrg32k3a::Batch<>::create(seed);
            case engine_type_internal::philox4x32x10:
                return daal::algorithms::engines::philox4x32x10::Batch<>::create(seed);
            case engine_type_internal::mt19937:
                return daal::algorithms::engines::mt19937::Batch<>::create(seed);
            default: throw std::invalid_argument("Unsupported engine type");
        }
    }

    void select_parallelization_technique(
        daal::algorithms::engines::internal::ParallelizationTechnique& technique) {
        auto daal_engine_impl =
            dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(engine_.get());

        daal::algorithms::engines::internal::ParallelizationTechnique techniques[] = {
            daal::algorithms::engines::internal::family,
            daal::algorithms::engines::internal::leapfrog,
            daal::algorithms::engines::internal::skipahead
        };

        for (auto& techn : techniques) {
            if (daal_engine_impl->hasSupport(techn)) {
                technique = techn;
                return;
            }
        }

        throw domain_error(
            dal::detail::error_messages::rng_engine_does_not_support_parallelization_techniques());
    }

private:
    std::int64_t count_;
    daal::algorithms::engines::EnginePtr engine_;
    daal::algorithms::engines::internal::Params<daal::sse2> params_;
    daal::algorithms::engines::internal::ParallelizationTechnique technique_;
    daal::services::internal::TArray<daal::algorithms::engines::EnginePtr, daal::sse2>
        daal_engine_list_;
};

} // namespace oneapi::dal::backend::primitives
