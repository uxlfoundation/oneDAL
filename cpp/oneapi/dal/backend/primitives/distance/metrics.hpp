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

#include <cmath>

#include "oneapi/dal/backend/primitives/distance/distance.hpp"

namespace oneapi::dal::backend::primitives {

struct distance_metric_tag;

template <typename Float>
struct metric_base {
public:
    using tag_t = distance_metric_tag;
    template <typename InputIt1, typename InputIt2>
    Float operator()(InputIt1 first1, InputIt1 last1, InputIt2 first2) const;
};

template <typename Float>
struct lp_metric : public metric_base<Float> {
public:
    lp_metric(Float p = 1.0) : p_{ p } {}
    template <typename InputIt1, typename InputIt2>
    Float operator()(InputIt1 first1, InputIt1 last1, InputIt2 first2) const {
        Float acc = 0;
        auto it1 = first1;
        auto it2 = first2;
        for (; it1 != last1; ++it1, ++it2) {
            const Float adiff = std::abs(*it1 - *it2);
            acc += std::pow(adiff, get_p());
        }
        return std::pow(acc, Float(1) / get_p());
    }

    const Float& get_p() const {
        return p_;
    }

private:
    const Float p_;
};

template <typename Float>
struct squared_l2_metric : public metric_base<Float> {
public:
    squared_l2_metric() {}
    template <typename InputIt1, typename InputIt2>
    Float operator()(InputIt1 first1, InputIt1 last1, InputIt2 first2) const {
        Float acc = 0;
        auto it1 = first1;
        auto it2 = first2;
        for (; it1 != last1; ++it1, ++it2) {
            const Float diff = *it1 - *it2;
            acc += diff * diff;
        }
        return acc;
    }
};

template <typename Float>
struct cosine_metric : public metric_base<Float> {
public:
    cosine_metric() {}
    template <typename InputIt1, typename InputIt2>
    Float operator()(InputIt1 first1, InputIt1 last1, InputIt2 first2) const {
        constexpr Float zero = 0;
        constexpr Float one = 1;
        Float ip_acc = zero;
        Float n1_acc = zero;
        Float n2_acc = zero;
        auto it1 = first1;
        auto it2 = first2;
        for (; it1 != last1; ++it1, ++it2) {
            const Float v1 = *it1;
            const Float v2 = *it2;
            n1_acc += (v1 * v1);
            n2_acc += (v2 * v2);
            ip_acc += (v1 * v2);
        }
        const Float rsqn1 = one / std::sqrt(n1_acc);
        const Float rsqn2 = one / std::sqrt(n2_acc);
        return one - ip_acc * rsqn1 * rsqn2;
    }
};

template <typename Float>
struct chebyshev_metric : public metric_base<Float> {
public:
    chebyshev_metric() {}
    template <typename InputIt1, typename InputIt2>
    Float operator()(InputIt1 first1, InputIt1 last1, InputIt2 first2) const {
        Float max_difference = 0;
        auto it1 = first1;
        auto it2 = first2;
        for (; it1 != last1; ++it1, ++it2) {
            const auto diff = std::abs(*it1 - *it2);
            max_difference = std::max(max_difference, diff);
        }
        return max_difference;
    }
};

template <typename Float>
struct correlation_metric : public metric_base<Float> {
public:
    correlation_metric() {}
    template <typename InputIt1, typename InputIt2>
    Float operator()(InputIt1 first1, InputIt1 last1, InputIt2 first2) const {
        constexpr Float zero = 0;
        constexpr Float one = 1;
        Float ip_acc = zero;
        Float n1_acc = zero;
        Float n2_acc = zero;
        Float n1_sum = zero;
        Float n2_sum = zero;
        Float count = zero;

        // Calculate sum for input values
        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, it2++) {
            n1_sum += *it1;
            n2_sum += *it2;
            ++count;
        }

        // Ensure count for sum updates, otherwise return zero value
        if (count == zero)
            return Float(zero);

        // Calculate mean for input values
        const Float n1_mean = n1_sum / count;
        const Float n2_mean = n2_sum / count;

        // Compute dot product and squared norms for centered values
        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, ++it2) {
            const Float v1 = *it1 - n1_mean;
            const Float v2 = *it2 - n2_mean;
            n1_acc += (v1 * v1);
            n2_acc += (v2 * v2);
            ip_acc += (v1 * v2);
        }

        // Calculate correlation distance and return result
        const Float rsqn1 = one / std::sqrt(n1_acc);
        const Float rsqn2 = one / std::sqrt(n2_acc);
        return one - ip_acc * rsqn1 * rsqn2;
    }
};

} // namespace oneapi::dal::backend::primitives
