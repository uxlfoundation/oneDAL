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

/// Per-pair accumulator shared by the DBSCAN GPU metric operations.
///
/// `sum` carries the running distance term for every metric. Cosine additionally
/// needs the two row norms; the other metrics never touch those slots, so the
/// compiler drops them.
template <typename Float>
struct metric_accumulator {
    Float sum = Float(0);
    Float lhs_norm = Float(0);
    Float rhs_norm = Float(0);
};

/// Device-side metric operations for the DBSCAN brute-force GPU kernels.
///
/// The neighborhood search only ever compares a distance against `epsilon`, so
/// each operation evaluates the cheapest form that stays monotone in the true
/// distance and carries `epsilon` mapped into that same space as `threshold`.
/// Keeping the threshold inside the operation means exactly one place knows how
/// a metric is powered, so no caller can pass a raw radius by mistake.
///
/// `partial_is_lower_bound` says whether a partially accumulated value is a
/// lower bound on the final distance. The wide kernels use it to abandon a
/// candidate pair as soon as the running value crosses `threshold`; cosine has
/// no such bound, because its partial dot product can still move either way.
///
/// The operations are plain aggregates so they are trivially device-copyable and
/// can be captured straight into a SYCL kernel.
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
        // A zero row has no direction, so there is no angle between it and
        // anything else. Report maximum separation instead of dividing by zero.
        return norm > Float(0) ? Float(1) - acc.sum / norm : Float(1);
    }
};

/// Builds the metric operation matching `metric` and hands it to `body`.
///
/// Concentrates the enum-to-operation mapping in one place so every call site
/// only has to supply a generic callable. The operation arrives as a template
/// argument, which keeps the innermost distance loops free of metric branches.
///
/// @tparam Float  Floating-point type used to perform computations
/// @tparam Body   Callable accepting one metric operation by value
/// @param[in] metric  The requested metric
/// @param[in] epsilon The neighborhood radius, in the units of `metric`
/// @param[in] degree  The Minkowski degree; ignored by every other metric
/// @param[in] body    The generic callable to invoke with the built operation
/// @return Whatever `body` returns
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
