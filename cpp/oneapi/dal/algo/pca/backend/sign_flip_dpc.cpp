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
namespace bk = dal::backend;
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

template <typename Float>
sycl::event sign_flip_impl(sycl::queue q,
                           Float* eigvecs,
                           std::int64_t row_count,
                           std::int64_t column_count,
                           const bk::event_vector& deps) {
    ONEDAL_ASSERT(eigvecs);
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(column_count > 0);
    ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, row_count, column_count);

    auto update_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>(row_count), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            sign_flip_vector(eigvecs + i * column_count, column_count);
        });
    });
    return update_event;
}

#define INSTANTIATE(Float)                                                \
    template sycl::event sign_flip_impl<Float>(sycl::queue q,             \
                                               Float * eigvecs,           \
                                               std::int64_t row_count,    \
                                               std::int64_t column_count, \
                                               const bk::event_vector& deps);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::pca::backend
