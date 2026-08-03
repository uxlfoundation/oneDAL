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

#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::basic_statistics::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

enum class bs_list : std::uint64_t {
    max = 1UL << 0,
    min = 1UL << 1,
    sum = 1UL << 2,
    sum2 = 1UL << 3,
    sum2cent = 1UL << 4,
    mean = 1UL << 5,
    sorm = 1UL << 6,
    varc = 1UL << 7,
    stdev = 1UL << 8,
    vart = 1UL << 9
};

inline constexpr bs_list operator&(bs_list value_left, bs_list value_right) {
    return bitwise_and(value_left, value_right);
}

inline constexpr bs_list operator|(bs_list value_left, bs_list value_right) {
    return bitwise_or(value_left, value_right);
}

constexpr inline bs_list bs_mode_min_max = (bs_list::min | bs_list::max);
constexpr inline bs_list bs_mode_mean_variance = (bs_list::mean | bs_list::varc);
constexpr inline bs_list bs_mode_all =
    (bs_list::min | bs_list::max | bs_list::sum | bs_list::sum2 | bs_list::sum2cent |
     bs_list::mean | bs_list::sorm | bs_list::varc | bs_list::stdev | bs_list::vart);

constexpr inline bs_list sum2cent_based_stat =
    bs_list::sum2cent | bs_list::varc | bs_list::stdev | bs_list::vart;

template <typename Float, bs_list List>
class local_result {
    using alloc = sycl::usm::alloc;
    using own_t = local_result<Float, List>;

public:
    local_result() = default;

    static own_t empty(sycl::queue& q,
                       std::int64_t count,
                       alloc result_alloc_kind,
                       bool deffered_fin = false) {
        own_t res;
        ONEDAL_ASSERT(result_alloc_kind != alloc::unknown);
        if constexpr (check_mask_flag(bs_list::min, List)) {
            res.rmin_ = pr::ndarray<Float, 1>::empty(q, { count }, result_alloc_kind);
        }
        if constexpr (check_mask_flag(bs_list::max, List)) {
            res.rmax_ = pr::ndarray<Float, 1>::empty(q, { count }, result_alloc_kind);
        }
        if (check_mask_flag(bs_list::sum, List) ||
            (deffered_fin && check_mask_flag(bs_list::mean | sum2cent_based_stat, List))) {
            res.rsum_ = pr::ndarray<Float, 1>::empty(q, { count }, result_alloc_kind);
        }
        if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
            res.rsum2_ = pr::ndarray<Float, 1>::empty(q, { count }, result_alloc_kind);
        }
        if (check_mask_flag(bs_list::sum2cent, List) ||
            (deffered_fin &&
             check_mask_flag(bs_list::varc | bs_list::stdev | bs_list::vart, List))) {
            res.rsum2cent_ = pr::ndarray<Float, 1>::empty(q, { count }, result_alloc_kind);
        }
        if constexpr (check_mask_flag(bs_list::mean, List)) {
            res.rmean_ = pr::ndarray<Float, 1>::empty(q, { count }, result_alloc_kind);
        }
        if constexpr (check_mask_flag(bs_list::sorm, List)) {
            res.rsorm_ = pr::ndarray<Float, 1>::empty(q, { count }, result_alloc_kind);
        }
        if constexpr (check_mask_flag(bs_list::varc, List)) {
            res.rvarc_ = pr::ndarray<Float, 1>::empty(q, { count }, result_alloc_kind);
        }
        if constexpr (check_mask_flag(bs_list::stdev, List)) {
            res.rstdev_ = pr::ndarray<Float, 1>::empty(q, { count }, result_alloc_kind);
        }
        if constexpr (check_mask_flag(bs_list::vart, List)) {
            res.rvart_ = pr::ndarray<Float, 1>::empty(q, { count }, result_alloc_kind);
        }

        return res;
    }

    auto& get_min() const {
        return rmin_;
    }
    auto& get_max() const {
        return rmax_;
    }
    auto& get_sum() const {
        return rsum_;
    }
    auto& get_sum2() const {
        return rsum2_;
    }
    auto& get_sum2cent() const {
        return rsum2cent_;
    }
    auto& get_mean() const {
        return rmean_;
    }
    auto& get_sorm() const {
        return rsorm_;
    }
    auto& get_varc() const {
        return rvarc_;
    }
    auto& get_stdev() const {
        return rstdev_;
    }
    auto& get_vart() const {
        return rvart_;
    }

private:
    pr::ndarray<Float, 1> rmin_;
    pr::ndarray<Float, 1> rmax_;
    pr::ndarray<Float, 1> rsum_;
    pr::ndarray<Float, 1> rsum2_;
    pr::ndarray<Float, 1> rsum2cent_;
    pr::ndarray<Float, 1> rmean_;
    pr::ndarray<Float, 1> rsorm_;
    pr::ndarray<Float, 1> rvarc_;
    pr::ndarray<Float, 1> rstdev_;
    pr::ndarray<Float, 1> rvart_;
};

template <typename Float, bs_list List>
class local_buffer_list {
    using alloc = sycl::usm::alloc;
    using own_t = local_buffer_list<Float, List>;

public:
    static own_t empty(sycl::queue& q, std::int64_t count) {
        own_t res;

        if constexpr (check_mask_flag(bs_list::mean | sum2cent_based_stat, List)) {
            res.rrow_count_ = pr::ndarray<std::int64_t, 1>::empty(q, { count }, alloc::device);
        }
        if constexpr (check_mask_flag(bs_list::min, List)) {
            res.rmin_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        }
        if constexpr (check_mask_flag(bs_list::max, List)) {
            res.rmax_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        }
        if (check_mask_flag(bs_list::sum | bs_list::mean | sum2cent_based_stat, List)) {
            res.rsum_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        }
        if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
            res.rsum2_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        }
        if (check_mask_flag(sum2cent_based_stat, List)) {
            res.rsum2cent_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        }

        return res;
    }

    auto& get_rc_list() const {
        return rrow_count_;
    }
    auto& get_min() const {
        return rmin_;
    }
    auto& get_max() const {
        return rmax_;
    }
    auto& get_sum() const {
        return rsum_;
    }
    auto& get_sum2() const {
        return rsum2_;
    }
    auto& get_sum2cent() const {
        return rsum2cent_;
    }

private:
    local_buffer_list() = default;

    pr::ndarray<std::int64_t, 1> rrow_count_;

    pr::ndarray<Float, 1> rmin_;
    pr::ndarray<Float, 1> rmax_;
    pr::ndarray<Float, 1> rsum_;
    pr::ndarray<Float, 1> rsum2_;
    pr::ndarray<Float, 1> rsum2cent_;
};

} // namespace oneapi::dal::basic_statistics::backend
#endif // ONEDAL_DATA_PARALLEL
