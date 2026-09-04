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

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/util/common.hpp"

namespace oneapi::dal::basic_statistics::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Cpu, typename Float>
std::int64_t propose_threading_block(std::int64_t row_count, std::int64_t col_count);

template <typename Cpu, typename Float>
void apply_weights_single_thread(const pr::ndview<Float, 1>& weights,
                                 pr::ndview<Float, 2>& samples);

template <typename Cpu, typename Float>
void apply_weights(const pr::ndview<Float, 1>& weights, pr::ndview<Float, 2>& samples);

/// Scale the stored values of a CSR matrix by their row's weight, out of place.
///
/// Weighting in this algorithm is plain per-row scaling of the data, see
/// `apply_weights`. On a CSR matrix that is a scaling of the stored values alone,
/// because a structural zero stays zero under scaling (`0 * w == 0`), so the sparsity
/// pattern of the input is carried over unchanged and only `values` needs a new buffer.
///
/// @param weights      Row weights, one per row of the matrix.
/// @param row_offsets  CSR row offsets, `weights.get_count() + 1` of them.
/// @param offset_shift 1 for one-based indexing, 0 for zero-based.
/// @param values       Stored values of the input matrix.
/// @param scaled       Output buffer for the scaled values, same size as `values`.
template <typename Cpu, typename Float>
void apply_weights_csr(const pr::ndview<Float, 1>& weights,
                       const pr::ndview<std::int64_t, 1>& row_offsets,
                       std::int64_t offset_shift,
                       const pr::ndview<Float, 1>& values,
                       pr::ndview<Float, 1>& scaled);

template <typename Float>
void apply_weights_single_thread(const dal::backend::context_cpu& context,
                                 const pr::ndview<Float, 1>& weights,
                                 pr::ndview<Float, 2>& samples);

template <typename Float>
void apply_weights(const dal::backend::context_cpu& context,
                   const pr::ndview<Float, 1>& weights,
                   pr::ndview<Float, 2>& samples);

template <typename Float>
void apply_weights_csr(const dal::backend::context_cpu& context,
                       const pr::ndview<Float, 1>& weights,
                       const pr::ndview<std::int64_t, 1>& row_offsets,
                       std::int64_t offset_shift,
                       const pr::ndview<Float, 1>& values,
                       pr::ndview<Float, 1>& scaled);

} // namespace oneapi::dal::basic_statistics::backend
