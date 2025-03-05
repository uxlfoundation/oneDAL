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

#include <limits>
#include <iostream>

#include <daal/src/externals/service_rng.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/src/algorithms/engines/engine_batch_impl.h>

#include "oneapi/dal/array.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

// Generates random sequence with required parameters
template <typename Float>
class rnd_seq {
public:
    rnd_seq() = delete;
    /// @param[in]  count Number of elements in random sequence
    /// @param[in]  a     Minimal value in the sequence
    /// @param[in]  b     Maximal value in the sequence
    rnd_seq(sycl::queue& queue, std::int64_t count, Float a = 0.0, Float b = 1.0) {
        ONEDAL_ASSERT(count > 0);
        seq_ = array<Float>::empty(queue, count);
        generate(queue, a, b);
    }
    std::int64_t get_count() {
        return seq_.get_count();
    }
    const Float* get_data() {
        return seq_.get_data();
    }

private:
    void generate(sycl::queue& queue, Float a, Float b) {
        auto engine = daal::algorithms::engines::mcg59::Batch<>::create();
        auto engine_impl =
            dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(&(*engine));
        ONEDAL_ASSERT(engine_impl != nullptr);

        auto* values = this->seq_.get_mutable_data();
        auto count = this->seq_.get_count();
        std::cerr << "count: " << count << std::endl;
        const auto count_as_size_t = dal::detail::integral_cast<std::size_t>(count);

        auto number_array = array<std::size_t>::empty(queue, count);
        daal::internal::RNGsInst<std::size_t, DAAL_BASE_CPU> rng;
        auto* number_ptr = number_array.get_mutable_data();

        rng.uniform(count_as_size_t, number_ptr, engine_impl->getState(), 0, count_as_size_t);
        std::transform(number_ptr, number_ptr + count, values, [=](std::size_t number) {
            return a + (b - a) * static_cast<Float>(number) / static_cast<Float>(count);
        });
    }
    array<Float> seq_;
};

#endif

} // namespace oneapi::dal::backend::primitives
