/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <algorithm>

#include "oneapi/dal/algo/basic_statistics/backend/cpu/apply_weights.hpp"

namespace oneapi::dal::basic_statistics::backend {

template <typename Float>
void apply_weights_single_thread(const dal::backend::context_cpu& context,
                                 const pr::ndview<Float, 1>& weights,
                                 pr::ndview<Float, 2>& samples) {
    return dal::backend::dispatch_by_cpu(context, [&](auto cpu) {
        return apply_weights_single_thread<decltype(cpu), Float>(weights, samples);
    });
}

template <typename Float>
void apply_weights(const dal::backend::context_cpu& context,
                   const pr::ndview<Float, 1>& weights,
                   pr::ndview<Float, 2>& samples) {
    return dal::backend::dispatch_by_cpu(context, [&](auto cpu) {
        return apply_weights<decltype(cpu), Float>(weights, samples);
    });
}

template <typename Float>
void apply_weights_csr(const dal::backend::context_cpu& context,
                       const pr::ndview<Float, 1>& weights,
                       const pr::ndview<std::int64_t, 1>& row_offsets,
                       std::int64_t offset_shift,
                       const pr::ndview<Float, 1>& values,
                       pr::ndview<Float, 1>& scaled) {
    return dal::backend::dispatch_by_cpu(context, [&](auto cpu) {
        return apply_weights_csr<decltype(cpu), Float>(weights,
                                                       row_offsets,
                                                       offset_shift,
                                                       values,
                                                       scaled);
    });
}

#define INSTANTIATE(F)                                                          \
    template void apply_weights_single_thread(const dal::backend::context_cpu&, \
                                              const pr::ndview<F, 1>&,          \
                                              pr::ndview<F, 2>&);               \
    template void apply_weights(const dal::backend::context_cpu&,               \
                                const pr::ndview<F, 1>&,                        \
                                pr::ndview<F, 2>&);                             \
    template void apply_weights_csr(const dal::backend::context_cpu&,           \
                                    const pr::ndview<F, 1>&,                    \
                                    const pr::ndview<std::int64_t, 1>&,         \
                                    std::int64_t,                               \
                                    const pr::ndview<F, 1>&,                    \
                                    pr::ndview<F, 1>&);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::basic_statistics::backend
