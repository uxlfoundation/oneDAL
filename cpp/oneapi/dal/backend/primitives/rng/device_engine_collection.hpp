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

template <engine_type EngineType = engine_type::mt2203>
class device_host_engine_collection {
public:
    device_host_engine_collection(sycl::queue& queue, std::int64_t count, std::int64_t seed = 777)
            : count_(count),
              base_seed_(seed) {
        engines_.reserve(count_);
        for (std::int64_t i = 0; i < count_; ++i) {
            engines_.push_back(device_engine<EngineType>(queue, base_seed_ + i));
        }
    }

    std::vector<device_engine<EngineType>> get_engines() const {
        return engines_;
    }

private:
    std::int64_t count_;
    std::int64_t base_seed_;
    std::vector<device_engine<EngineType>> engines_;
};

#endif
} // namespace oneapi::dal::backend::primitives
