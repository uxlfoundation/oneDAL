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

#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
class naive_data_provider {
    using this_t = naive_data_provider<Float>;

public:
    naive_data_provider(std::int32_t wth, std::int32_t str, const Float* ptr)
            : wth_{ std::move(wth) },
              str_{ std::move(str) },
              ptr_{ std::move(ptr) } {}

    static inline this_t make(const ndview<Float, 2>& data) {
        ONEDAL_ASSERT(data.has_data());
        const auto* const ptr = data.get_data();
        const auto owth = data.get_dimension(1);
        const auto ostr = data.get_leading_stride();
        ONEDAL_ASSERT(owth <= ostr);
        const auto cwth = dal::detail::integral_cast<std::int32_t>(owth);
        const auto cstr = dal::detail::integral_cast<std::int32_t>(ostr);
        return this_t{ cwth, cstr, ptr };
    }

    const Float& at(std::int32_t row, std::int32_t col) const {
        return *(ptr_ + std::int64_t(row) * str_ + col);
    }

    const std::int32_t& get_width() const {
        return wth_;
    }

private:
    const std::int32_t wth_;
    const std::int32_t str_;
    const Float* const ptr_;
};

template <typename Float>
class sq_l2_distance_provider {
    using this_t = sq_l2_distance_provider<Float>;

public:
    sq_l2_distance_provider(std::int32_t ip_wth,
                            std::int32_t ip_str,
                            const Float* ip_ptr,
                            const Float* n1_ptr,
                            const Float* n2_ptr)
            : ip_wth_{ std::move(ip_wth) },
              ip_str_{ std::move(ip_str) },
              ip_ptr_{ std::move(ip_ptr) },
              n1_ptr_{ std::move(n1_ptr) },
              n2_ptr_{ std::move(n2_ptr) } {}

    static inline this_t make(const ndview<Float, 1>& n1,
                              const ndview<Float, 1>& n2,
                              const ndview<Float, 2>& ip) {
        ONEDAL_ASSERT(n1.has_data());
        ONEDAL_ASSERT(n2.has_data());
        ONEDAL_ASSERT(ip.has_data());
        ONEDAL_ASSERT(n1.get_count() == ip.get_dimension(0));
        ONEDAL_ASSERT(n2.get_count() == ip.get_dimension(1));
        const auto* const n1_ptr = n1.get_data();
        const auto* const n2_ptr = n2.get_data();
        const auto* const ip_ptr = ip.get_data();
        const auto ip_owth = ip.get_dimension(1);
        const auto ip_ostr = ip.get_leading_stride();
        const auto ip_cwth = dal::detail::integral_cast<std::int32_t>(ip_owth);
        const auto ip_cstr = dal::detail::integral_cast<std::int32_t>(ip_ostr);
        return this_t{ ip_cwth, ip_cstr, ip_ptr, n1_ptr, n2_ptr };
    }

    Float at(std::int32_t row, std::int32_t col) const {
        const auto& n1 = *(n1_ptr_ + row);
        const auto& n2 = *(n2_ptr_ + col);
        const auto& ip = *(ip_ptr_ + std::int64_t(row) * ip_str_ + col);
        // L2 distance = inner product + norms #1 + norms #2
        // inner product = -2 * qnorms * tnorms
        return n1 + n2 + ip;
    }

    const std::int32_t& get_width() const {
        return ip_wth_;
    }

private:
    const std::int32_t ip_wth_;
    const std::int32_t ip_str_;
    const Float* const ip_ptr_;
    const Float* const n1_ptr_;
    const Float* const n2_ptr_;
};

template <typename Float, bool sq_l2>
struct data_provider_map {};

template <typename Float>
struct data_provider_map<Float, true> {
    using type = sq_l2_distance_provider<Float>;
};

template <typename Float>
struct data_provider_map<Float, false> {
    using type = naive_data_provider<Float>;
};

template <typename Float, bool sq_l2 = false>
using data_provider_t = typename data_provider_map<Float, sq_l2>::type;

#endif

} // namespace oneapi::dal::backend::primitives
