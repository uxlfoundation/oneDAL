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

#include "oneapi/dal/backend/primitives/rng/rng_dpc.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include <vector>

#include "oneapi/dal/backend/primitives/rng/utils.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_types.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

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
