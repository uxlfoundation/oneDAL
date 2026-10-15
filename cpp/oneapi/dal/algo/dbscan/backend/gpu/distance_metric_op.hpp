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

#include <cmath>

#include "oneapi/dal/algo/dbscan/common.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::dbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

/// Per-pair accumulator of the metric operations; only cosine uses the two norms.
template <typename Float>
struct metric_accumulator {
    Float sum = Float(0);
    Float lhs_norm = Float(0);
    Float rhs_norm = Float(0);
};

// Metric operations of the neighborhood kernels. Each computes the unrooted distance and carries
// `epsilon` mapped into the same space as `threshold`, as `poweredEpsilon` does on CPU.
// `partial_is_lower_bound` allows the early exit on a partial sum; a partial cosine sum is no bound.
template <typename Float>
struct euclidean_metric_op {
    static constexpr bool partial_is_lower_bound = true;

    Float threshold;

    void step(metric_accumulator<Float>& acc, Float a, Float b) const {
        const Float diff = a - b;
        acc.sum += diff * diff;
    }

    template <typename SubGroup>
    Float reduce(const SubGroup& sg, const metric_accumulator<Float>& acc) const {
        return sycl::reduce_over_group(sg, acc.sum, sycl::plus<Float>());
    }

    Float finish(const metric_accumulator<Float>& acc) const {
        return acc.sum;
    }
};

template <typename Float>
struct manhattan_metric_op {
    static constexpr bool partial_is_lower_bound = true;

    Float threshold;

    void step(metric_accumulator<Float>& acc, Float a, Float b) const {
        acc.sum += sycl::fabs(a - b);
    }

    template <typename SubGroup>
    Float reduce(const SubGroup& sg, const metric_accumulator<Float>& acc) const {
        return sycl::reduce_over_group(sg, acc.sum, sycl::plus<Float>());
    }

    Float finish(const metric_accumulator<Float>& acc) const {
        return acc.sum;
    }
};

template <typename Float>
struct minkowski_metric_op {
    static constexpr bool partial_is_lower_bound = true;

    Float threshold;
    Float degree;

    void step(metric_accumulator<Float>& acc, Float a, Float b) const {
        acc.sum += sycl::pow(sycl::fabs(a - b), degree);
    }

    template <typename SubGroup>
    Float reduce(const SubGroup& sg, const metric_accumulator<Float>& acc) const {
        return sycl::reduce_over_group(sg, acc.sum, sycl::plus<Float>());
    }

    Float finish(const metric_accumulator<Float>& acc) const {
        return acc.sum;
    }
};

template <typename Float>
struct chebyshev_metric_op {
    static constexpr bool partial_is_lower_bound = true;

    Float threshold;

    void step(metric_accumulator<Float>& acc, Float a, Float b) const {
        acc.sum = sycl::fmax(acc.sum, sycl::fabs(a - b));
    }

    template <typename SubGroup>
    Float reduce(const SubGroup& sg, const metric_accumulator<Float>& acc) const {
        return sycl::reduce_over_group(sg, acc.sum, sycl::maximum<Float>());
    }

    Float finish(const metric_accumulator<Float>& acc) const {
        return acc.sum;
    }
};

template <typename Float>
struct cosine_metric_op {
    static constexpr bool partial_is_lower_bound = false;

    Float threshold;

    void step(metric_accumulator<Float>& acc, Float a, Float b) const {
        acc.sum += a * b;
        acc.lhs_norm += a * a;
        acc.rhs_norm += b * b;
    }

    template <typename SubGroup>
    Float reduce(const SubGroup& sg, const metric_accumulator<Float>& acc) const {
        metric_accumulator<Float> total;
        total.sum = sycl::reduce_over_group(sg, acc.sum, sycl::plus<Float>());
        total.lhs_norm = sycl::reduce_over_group(sg, acc.lhs_norm, sycl::plus<Float>());
        total.rhs_norm = sycl::reduce_over_group(sg, acc.rhs_norm, sycl::plus<Float>());
        return finish(total);
    }

    Float finish(const metric_accumulator<Float>& acc) const {
        const Float norm = sycl::sqrt(acc.lhs_norm) * sycl::sqrt(acc.rhs_norm);
        // As in scikit-learn, a zero row is at distance 1 from every other row.
        return (norm > Float(0)) ? Float(1) - acc.sum / norm : Float(1);
    }
};

/// Calls `body` with the metric operation for `metric`.
///
/// @tparam Float  Floating-point type
/// @tparam Body   Callable taking one metric operation
/// @param[in] metric  Distance metric
/// @param[in] epsilon Neighborhood radius in the units of `metric`
/// @param[in] degree  Minkowski degree; ignored by the other metrics
/// @param[in] body    Callable to invoke
/// @return The result of `body`
template <typename Float, typename Body>
inline auto dispatch_by_metric(distance_metric metric, double epsilon, double degree, Body&& body) {
    const Float eps = static_cast<Float>(epsilon);
    switch (metric) {
        case distance_metric::euclidean: return body(euclidean_metric_op<Float>{ eps * eps });
        case distance_metric::manhattan: return body(manhattan_metric_op<Float>{ eps });
        case distance_metric::minkowski:
            return body(minkowski_metric_op<Float>{ static_cast<Float>(std::pow(epsilon, degree)),
                                                    static_cast<Float>(degree) });
        case distance_metric::chebyshev: return body(chebyshev_metric_op<Float>{ eps });
        case distance_metric::cosine: return body(cosine_metric_op<Float>{ eps });
    }
    throw invalid_argument(dal::detail::error_messages::unknown_distance_type());
}

#endif

} // namespace oneapi::dal::dbscan::backend
