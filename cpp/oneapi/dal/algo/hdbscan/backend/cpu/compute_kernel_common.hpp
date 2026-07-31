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

#include "oneapi/dal/algo/hdbscan/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/exceptions.hpp"

#include <daal/src/algorithms/service_kernel_math.h>

namespace oneapi::dal::hdbscan::backend {

using daal_pairwise_distance_t = daal::algorithms::internal::PairwiseDistanceType;

/// Convert a oneAPI HDBSCAN `distance_metric` value to the DAAL pairwise tag.
///
/// Used by the CPU backends to forward the user-facing oneAPI metric enum to
/// the shared DAAL `PairwiseDistanceType` consumed by `HDBSCANBatchKernel`.
/// Every enumerator in `distance_metric` is mapped explicitly; any value that
/// does not correspond to a known enumerator raises `invalid_argument` so a
/// new metric added to the header without updating this switch is surfaced as
/// a loud runtime error rather than silently routed to euclidean. The
/// method-specific compatibility rules (e.g. cosine is only valid with
/// `method::brute_force`) are enforced upstream in
/// `detail/compute_ops.hpp::check_preconditions`, so this converter is
/// reached only for combinations that have already passed that check.
///
/// @param[in] m oneAPI metric tag
///
/// @return Equivalent DAAL `PairwiseDistanceType`
inline daal_pairwise_distance_t convert_metric(distance_metric m) {
    switch (m) {
        case distance_metric::euclidean: return daal_pairwise_distance_t::euclidean;
        case distance_metric::manhattan: return daal_pairwise_distance_t::manhattan;
        case distance_metric::minkowski: return daal_pairwise_distance_t::minkowski;
        case distance_metric::chebyshev: return daal_pairwise_distance_t::chebyshev;
        case distance_metric::cosine: return daal_pairwise_distance_t::cosine;
    }
    throw invalid_argument(dal::detail::error_messages::unknown_distance_type());
}

} // namespace oneapi::dal::hdbscan::backend
