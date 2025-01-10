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

#include "oneapi/dal/backend/primitives/rng/rng.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include <vector>

#include "oneapi/dal/backend/primitives/rng/utils.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_types.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Size = std::int64_t>
class engine_collection {
public:
    explicit engine_collection(Size count, std::int64_t seed = 777)
            : count_(count),
              engine_(daal::algorithms::engines::mt2203::Batch<>::create(seed)),
              params_(count),
              technique_(daal::algorithms::engines::internal::family),
              host_engine_method_(count) {}

    template <typename Op>
    std::vector<host_engine<engine_method::mt2203>> operator()(Op&& op) {
        daal::services::Status status;
        for (Size i = 0; i < count_; ++i) {
            op(i, params_.nSkip[i]);
        }
        select_parallelization_technique(technique_);
        daal::algorithms::engines::internal::EnginesCollection<daal::sse2> engine_collection(
            engine_,
            technique_,
            params_,
            host_engine_method_,
            &status);
        if (!status) {
            dal::backend::interop::status_to_exception(status);
        }

        std::vector<host_engine<engine_method::mt2203>> engine_method(count_);
        for (Size i = 0; i < count_; ++i) {
            engine_method[i] = host_engine_method_[i];
        }

        //copy elision
        return engine_method;
    }

private:
    void select_parallelization_technique(
        daal::algorithms::engines::internal::ParallelizationTechnique& technique) {
        auto host_engine_impl =
            dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(engine_.get());

        daal::algorithms::engines::internal::ParallelizationTechnique techniques[] = {
            daal::algorithms::engines::internal::family,
            daal::algorithms::engines::internal::leapfrog,
            daal::algorithms::engines::internal::skipahead
        };

        for (auto& techn : techniques) {
            if (host_engine_impl->hasSupport(techn)) {
                technique = techn;
                return;
            }
        }

        throw domain_error(
            dal::detail::error_messages::rng_engine_does_not_support_parallelization_techniques());
    }

private:
    Size count_;
    daal::algorithms::engines::EnginePtr engine_;
    daal::algorithms::engines::internal::Params<daal::sse2> params_;
    daal::algorithms::engines::internal::ParallelizationTechnique technique_;
    daal::services::internal::TArray<daal::algorithms::engines::EnginePtr, daal::sse2>
        host_engine_method_;
};

template <typename Size = std::int64_t, engine_method EngineType = engine_method::mt2203>
class engine_collection_oneapi {
public:
    engine_collection_oneapi(sycl::queue& queue, Size count, std::int64_t seed = 777)
            : count_(count),
              seed_(seed) {
        engines_.reserve(count_);
        for (Size i = 0; i < count_; ++i) {
            engines_.push_back(dpc_engine<EngineType>(queue, seed_));
        }
    }

    std::vector<dpc_engine<EngineType>> get_engines() const {
        return engines_;
    }

private:
    Size count_;
    std::int64_t seed_;
    std::vector<dpc_engine<EngineType>> engines_;
};

#endif
} // namespace oneapi::dal::backend::primitives
