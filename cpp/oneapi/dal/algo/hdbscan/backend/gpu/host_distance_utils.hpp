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
#include <cstdint>

namespace oneapi::dal::hdbscan::backend {

template <typename FP>
struct HostEuclideanDist {
    FP pointDist(const FP* a, const FP* b, std::int32_t n) const {
        FP sum = FP(0);
        for (std::int32_t d = 0; d < n; d++) {
            FP diff = a[d] - b[d];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    FP bboxLowerBound(const FP* q, const FP* lo, const FP* hi, std::int32_t n) const {
        FP sum = FP(0);
        for (std::int32_t d = 0; d < n; d++) {
            FP excess = FP(0);
            if (q[d] < lo[d])
                excess = lo[d] - q[d];
            else if (q[d] > hi[d])
                excess = q[d] - hi[d];
            sum += excess * excess;
        }
        return std::sqrt(sum);
    }
};

template <typename FP>
struct HostManhattanDist {
    FP pointDist(const FP* a, const FP* b, std::int32_t n) const {
        FP sum = FP(0);
        for (std::int32_t d = 0; d < n; d++) {
            FP diff = a[d] - b[d];
            sum += (diff >= FP(0)) ? diff : -diff;
        }
        return sum;
    }
    FP bboxLowerBound(const FP* q, const FP* lo, const FP* hi, std::int32_t n) const {
        FP sum = FP(0);
        for (std::int32_t d = 0; d < n; d++) {
            FP excess = FP(0);
            if (q[d] < lo[d])
                excess = lo[d] - q[d];
            else if (q[d] > hi[d])
                excess = q[d] - hi[d];
            sum += excess;
        }
        return sum;
    }
};

template <typename FP>
struct HostMinkowskiDist {
    double p;
    double invp;
    HostMinkowskiDist(double deg) : p(deg), invp(1.0 / deg) {}
    FP pointDist(const FP* a, const FP* b, std::int32_t n) const {
        double sum = 0.0;
        for (std::int32_t d = 0; d < n; d++) {
            double diff = static_cast<double>(a[d] - b[d]);
            if (diff < 0.0)
                diff = -diff;
            sum += std::pow(diff, p);
        }
        return static_cast<FP>(std::pow(sum, invp));
    }
    FP bboxLowerBound(const FP* q, const FP* lo, const FP* hi, std::int32_t n) const {
        double sum = 0.0;
        for (std::int32_t d = 0; d < n; d++) {
            double excess = 0.0;
            if (q[d] < lo[d])
                excess = static_cast<double>(lo[d] - q[d]);
            else if (q[d] > hi[d])
                excess = static_cast<double>(q[d] - hi[d]);
            sum += std::pow(excess, p);
        }
        return static_cast<FP>(std::pow(sum, invp));
    }
};

template <typename FP>
struct HostChebyshevDist {
    FP pointDist(const FP* a, const FP* b, std::int32_t n) const {
        FP mx = FP(0);
        for (std::int32_t d = 0; d < n; d++) {
            FP diff = a[d] - b[d];
            if (diff < FP(0))
                diff = -diff;
            if (diff > mx)
                mx = diff;
        }
        return mx;
    }
    FP bboxLowerBound(const FP* q, const FP* lo, const FP* hi, std::int32_t n) const {
        FP mx = FP(0);
        for (std::int32_t d = 0; d < n; d++) {
            FP excess = FP(0);
            if (q[d] < lo[d])
                excess = lo[d] - q[d];
            else if (q[d] > hi[d])
                excess = q[d] - hi[d];
            if (excess > mx)
                mx = excess;
        }
        return mx;
    }
};

template <typename FP, typename BaseDist>
struct HostAlphaScaledDist {
    BaseDist base;
    FP inv_alpha;
    HostAlphaScaledDist(const BaseDist& b, double a)
            : base(b),
              inv_alpha(static_cast<FP>(1.0 / a)) {}
    FP pointDist(const FP* a, const FP* b, std::int32_t n) const {
        return base.pointDist(a, b, n) * inv_alpha;
    }
    FP bboxLowerBound(const FP* q, const FP* lo, const FP* hi, std::int32_t n) const {
        return base.bboxLowerBound(q, lo, hi, n) * inv_alpha;
    }
};

} // namespace oneapi::dal::hdbscan::backend
