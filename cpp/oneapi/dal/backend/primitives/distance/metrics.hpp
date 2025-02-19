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
    correlation_metric(){}
    template <typename InputIt1, typename InputIt2>
    Float operator()(InputIt1 first1, InputIt1 last1, InputIt2 first2) const {
        constexpr Float zero = 0;
        constexpr Float one = 1;
        Float ip_acc = zero;
        Float n1_acc = zero;
        Float n2_acc = zero;
        Float n1_sum = zero;
        Float n1_sum = zero;
        Float count = zero;
	for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, it2++) {
            n1_sum += *it1;
            n2_sum += *it2;
	    ++count;
	}

	if (count == zero)
            return Float(zero);

	const Float n1_mean = n1_sum / count;
	const Float n2_mean = n2_sum / count;

        for (auto it1 = first1, it2 = first2; it1 != last1; ++it1, ++it2) {
            const Float v1 = *it1 - n1_mean;
            const Float v2 = *it2 - n2_mean;
            n1_acc += (v1 * v1);
            n2_acc += (v2 * v2);
            ip_acc += (v1 * v2);
        }
        const Float rsqn1 = one / std::sqrt(n1_acc);
        const Float rsqn2 = one / std::sqrt(n2_acc);
        return one - ip_acc * rsqn1 * rsqn2;
    }

    template <ndorder order>
    sycl::event operator()(sycl::queue& q,
                           const ndview<Float, 2, order>& u,
                           const ndview<Float, 2, order>& v,
                           ndview<Float, 2>& out,
                           const event_vector& deps = {}) const {
        const std::int64_t n = u.get_dimension(0);
        auto u_sum = ndarray<Float, 1>::empty({ 1 });
        auto v_sum = ndarray<Float, 1>::empty({ 1 });
        auto u_mean = ndarray<Float, 1>::empty({ 1 });
        auto v_mean = ndarray<Float, 1>::empty({ 1 });
        sycl::event evt1 = reduce_by_columns(q, u, u_sum, {}, {}, deps, true);
        sycl::event evt2 = reduce_by_columns(q, v, v_sum, {}, {}, { evt1 }, true);
        sycl::event evt3 = means(q, n, u_sum, u_mean, { evt2 });
        sycl::event evt4 = means(q, n, v_sum, v_mean, { evt3 });
        auto temp = ndarray<Float, 1>::empty({ 3 });
        q.fill(temp, Float(0)).wait();
        sycl::event evt5 = q.submit([&](sycl::handler& h) {
            h.depends_on({ evt4 });
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                const std::int64_t i = idx[0];
                const Float x = u.at(i, 0);
                const Float y = v.at(i, 0);
                const Float mu_x = u_mean.at(0);
                const Float mu_y = v_mean.at(0);
                const Float d1 = x - mu_x;
                const Float d2 = y - mu_y;
                sycl::atomic_ref<Float,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_dot(temp.get_mutable_data()[0]);
                sycl::atomic_ref<Float,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_norm1(temp.get_mutable_data()[1]);
                sycl::atomic_ref<Float,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_norm2(temp.get_mutable_data()[2]);
                atomic_dot.fetch_add(d1 * d2);
                atomic_norm1.fetch_add(d1 * d1);
                atomic_norm2.fetch_add(d2 * d2);
            });
        });
        evt5.wait_and_throw();
        std::array<Float, 3> host_temp;
        q.memcpy(host_temp.data(), temp.get_mutable_data(), 3 * sizeof(Float)).wait();
        const Float dot = host_temp[0];
        const Float norm1 = host_temp[1];
        const Float norm2 = host_temp[2];
        const Float corr =
            (norm1 > 0 && norm2 > 0) ? dot / (std::sqrt(norm1) * std::sqrt(norm2)) : Float(0);
        const Float distance = 1 - corr;
        out.get_mutable_data()[0] = distance;
        return evt5;
    }
};

} // namespace oneapi::dal::backend::primitives
