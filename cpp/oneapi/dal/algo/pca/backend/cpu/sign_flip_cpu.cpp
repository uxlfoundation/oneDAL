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

#include "oneapi/dal/algo/pca/backend/sign_flip.hpp"

namespace oneapi::dal::pca::backend {

template <typename Float>
ONEDAL_FORCEINLINE Float abs(Float x) {
    return (x >= Float(0)) ? x : -x;
}

template <typename Float>
ONEDAL_FORCEINLINE Float max_by_abs(Float* x, std::int64_t count) {
    ONEDAL_ASSERT(x);
    ONEDAL_ASSERT(count > 0);

    Float max = x[0];
    Float abs_max = abs(x[0]);

    for (std::int64_t i = 0; i < count; i++) {
        const Float x_abs = abs(x[i]);
        if (x_abs > abs_max) {
            abs_max = x_abs;
            max = x[i];
        }
    }

    return max;
}

template <typename Float>
ONEDAL_FORCEINLINE void sign_flip_vector(Float* x, std::int64_t count) {
    ONEDAL_ASSERT(x);
    ONEDAL_ASSERT(count > 0);

    const Float max_signed = max_by_abs(x, count);

    if (max_signed < 0) {
        for (std::int64_t i = 0; i < count; i++) {
            x[i] = -x[i];
        }
    }
}

template <typename Cpu, typename Float>
void sign_flip_impl(Float* eigvecs, std::int64_t row_count, std::int64_t column_count) {
    ONEDAL_ASSERT(eigvecs);
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(column_count > 0);
    ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, row_count, column_count);

    for (std::int64_t i = 0; i < row_count; i++) {
        sign_flip_vector(eigvecs + i * column_count, column_count);
    }
}

#define INSTANTIATE(Cpu, Float)                                      \
    template void sign_flip_impl<Cpu, Float>(Float * eigvecs,        \
                                             std::int64_t row_count, \
                                             std::int64_t column_count);

INSTANTIATE(__CPU_TAG__, float)
INSTANTIATE(__CPU_TAG__, double)

} // namespace oneapi::dal::pca::backend
