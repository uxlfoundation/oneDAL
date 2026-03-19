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

#include <limits>
#include <algorithm>
#include <type_traits>
#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel_dense_impl.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::basic_statistics::backend {

#define ASSERT_IF(enable_condition, condition)                   \
    do {                                                         \
        if constexpr (check_mask_flag(enable_condition, List)) { \
            ONEDAL_ASSERT(condition);                            \
        }                                                        \
    } while (0)

#define DECLSET_IF(type, var, cond, value)       \
    type var = nullptr;                          \
    if constexpr (check_mask_flag(cond, List)) { \
        var = value;                             \
    }

#define SET_IF(var, cond, value)                 \
    if constexpr (check_mask_flag(cond, List)) { \
        var = value;                             \
    }

namespace de = dal::detail;
namespace be = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;

template <typename Data>
using local_accessor_rw_t = sycl::local_accessor<Data, 1>;

using comm_t = bk::communicator<spmd::device_memory_access::usm>;
using dal::backend::context_gpu;
using method_t = method::dense;
using task_t = task::compute;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float, bs_list List>
std::int64_t compute_kernel_dense_impl<Float, List>::get_row_block_count(std::int64_t row_count) {
    ONEDAL_ASSERT(row_count > 0);
    // TODO optimize the approach for row_block_count calculating

    std::int64_t row_block_count = 128;
    if (row_count < 5000)
        row_block_count = 1;
    else if (row_count < 10000)
        row_block_count = 8;
    else if (row_count < 20000)
        row_block_count = 16;
    else if (row_count < 50000)
        row_block_count = 32;
    else if (row_count < 100000)
        row_block_count = 64;

    return row_block_count;
}

template <typename Float, bs_list List>
std::int64_t compute_kernel_dense_impl<Float, List>::get_column_block_count(
    std::int64_t column_count) {
    ONEDAL_ASSERT(column_count > 0);

    const auto max_work_group_size = be::device_max_wg_size(q_);
    return (column_count + max_work_group_size - 1) / max_work_group_size;
}

template <typename Float, bs_list List>
result_t compute_kernel_dense_impl<Float, List>::get_result(const descriptor_t& desc,
                                                            const local_result<Float, List>& ndres,
                                                            std::int64_t column_count,
                                                            const bk::event_vector& deps) {
    ONEDAL_ASSERT(column_count > 0);
    result_t res;

    const auto res_op = desc.get_result_options();
    res.set_result_options(desc.get_result_options());

    if (res_op.test(result_options::min)) {
        ONEDAL_ASSERT(ndres.get_min().get_count() == column_count);
        res.set_min(homogen_table::wrap(ndres.get_min().flatten(q_, deps), 1, column_count));
    }
    if (res_op.test(result_options::max)) {
        ONEDAL_ASSERT(ndres.get_max().get_count() == column_count);
        res.set_max(homogen_table::wrap(ndres.get_max().flatten(q_, deps), 1, column_count));
    }
    if (res_op.test(result_options::sum)) {
        ONEDAL_ASSERT(ndres.get_sum().get_count() == column_count);
        res.set_sum(homogen_table::wrap(ndres.get_sum().flatten(q_, deps), 1, column_count));
    }
    if (res_op.test(result_options::sum_squares)) {
        ONEDAL_ASSERT(ndres.get_sum2().get_count() == column_count);
        res.set_sum_squares(
            homogen_table::wrap(ndres.get_sum2().flatten(q_, deps), 1, column_count));
    }
    if (res_op.test(result_options::sum_squares_centered)) {
        ONEDAL_ASSERT(ndres.get_sum2cent().get_count() == column_count);
        res.set_sum_squares_centered(
            homogen_table::wrap(ndres.get_sum2cent().flatten(q_, deps), 1, column_count));
    }
    if (res_op.test(result_options::mean)) {
        ONEDAL_ASSERT(ndres.get_mean().get_count() == column_count);
        res.set_mean(homogen_table::wrap(ndres.get_mean().flatten(q_, deps), 1, column_count));
    }
    if (res_op.test(result_options::second_order_raw_moment)) {
        ONEDAL_ASSERT(ndres.get_sorm().get_count() == column_count);
        res.set_second_order_raw_moment(
            homogen_table::wrap(ndres.get_sorm().flatten(q_, deps), 1, column_count));
    }
    if (res_op.test(result_options::variance)) {
        ONEDAL_ASSERT(ndres.get_varc().get_count() == column_count);
        res.set_variance(homogen_table::wrap(ndres.get_varc().flatten(q_, deps), 1, column_count));
    }
    if (res_op.test(result_options::standard_deviation)) {
        ONEDAL_ASSERT(ndres.get_stdev().get_count() == column_count);
        res.set_standard_deviation(
            homogen_table::wrap(ndres.get_stdev().flatten(q_, deps), 1, column_count));
    }
    if (res_op.test(result_options::variation)) {
        ONEDAL_ASSERT(ndres.get_vart().get_count() == column_count);
        res.set_variation(homogen_table::wrap(ndres.get_vart().flatten(q_, deps), 1, column_count));
    }
    return res;
}

template <typename Float, bs_list params, bool DefferedFin, bool weights = false>
struct singlepass_processor_kernel {
    struct empty {};

    std::conditional_t<weights, const Float*, empty> weights_ptr;

    constexpr static inline bool output_min = check_mask_flag(bs_list::min, params);
    std::conditional_t<output_min, Float*, empty> min_ptr;

    constexpr static inline bool output_max = check_mask_flag(bs_list::max, params);
    std::conditional_t<output_max, Float*, empty> max_ptr;

    constexpr static inline auto sum2_cond = bs_list::sum2 | bs_list::sorm;
    constexpr static inline bool output_sum2 = check_mask_flag(sum2_cond, params);
    std::conditional_t<output_sum2, Float*, empty> sum2_ptr;

    constexpr static inline bool output_sum =
        check_mask_flag(bs_list::sum, params) || //
        (DefferedFin && check_mask_flag(bs_list::mean | sum2cent_based_stat, params));
    std::conditional_t<output_sum, Float*, empty> sum_ptr;

    constexpr static inline bool output_sorm =
        !DefferedFin && check_mask_flag(bs_list::sorm, params);
    std::conditional_t<output_sorm, Float*, empty> sorm_ptr;

    constexpr static inline bool output_varc =
        !DefferedFin && check_mask_flag(bs_list::varc, params);
    std::conditional_t<output_varc, Float*, empty> varc_ptr;

    constexpr static inline bool output_vart =
        !DefferedFin && check_mask_flag(bs_list::vart, params);
    std::conditional_t<output_vart, Float*, empty> vart_ptr;

    constexpr static inline bool output_stdev =
        !DefferedFin && check_mask_flag(bs_list::stdev, params);
    std::conditional_t<output_stdev, Float*, empty> stdev_ptr;

    constexpr static inline bool output_sum2cent =
        check_mask_flag(bs_list::sum2cent, params) || //
        (DefferedFin && check_mask_flag(bs_list::varc | bs_list::stdev | bs_list::vart, params));
    std::conditional_t<output_sum2cent, Float*, empty> sum2cent_ptr;

    constexpr static inline bool output_mean = check_mask_flag(bs_list::mean, params);
    std::conditional_t<output_mean, Float*, empty> mean_ptr;

    constexpr static inline bool compute_min = output_min;
    constexpr static inline bool compute_max = output_max;
    constexpr static inline bool compute_vart = output_vart;
    constexpr static inline bool compute_sum2 = output_sum2;
    constexpr static inline bool compute_stdev = output_stdev;
    constexpr static inline bool compute_sum = output_sum || output_sorm;
    constexpr static inline bool compute_varc = output_varc || output_stdev;
    constexpr static inline bool compute_mean = output_mean || output_sum2cent;
    constexpr static inline bool compute_sum2cent = output_sum2cent || output_varc;

    singlepass_processor_kernel(const Float* data,
                                std::int64_t stride,
                                std::int64_t row_count,
                                std::int64_t column_count)
            : data_ptr{ data },
              stride{ stride },
              row_count{ row_count },
              column_count{ column_count } {}

    bool verify_pointers() const {
        if (data_ptr == nullptr)
            return false;
        if constexpr (weights) {
            if (weights_ptr == nullptr)
                return false;
        }
        if constexpr (output_max) {
            if (max_ptr == nullptr)
                return false;
        }
        if constexpr (output_min) {
            if (min_ptr == nullptr)
                return false;
        }
        if constexpr (output_sum) {
            if (sum_ptr == nullptr)
                return false;
        }
        if constexpr (output_sum2) {
            if (sum2_ptr == nullptr)
                return false;
        }
        if constexpr (output_mean) {
            if (mean_ptr == nullptr)
                return false;
        }
        if constexpr (output_vart) {
            if (vart_ptr == nullptr)
                return false;
        }
        if constexpr (output_varc) {
            if (varc_ptr == nullptr)
                return false;
        }
        if constexpr (output_stdev) {
            if (stdev_ptr == nullptr)
                return false;
        }
        if constexpr (output_sum2cent) {
            if (sum2cent_ptr == nullptr)
                return false;
        }

        return true;
    }

    void operator()(sycl::nd_item<1> id) const {
        constexpr Float zero = 0, one = 1;
        using limits_t = std::numeric_limits<Float>;
        constexpr Float maximum = limits_t::max();

        const auto col = id.get_global_linear_id();

        if (column_count <= std::int64_t(col))
            return;

        std::conditional_t<compute_min, Float, empty> min;
        if constexpr (compute_min)
            min = +maximum;

        std::conditional_t<compute_max, Float, empty> max;
        if constexpr (compute_max)
            max = -maximum;

        std::conditional_t<compute_sum, Float, empty> sum;
        if constexpr (compute_sum)
            sum = zero;

        std::conditional_t<compute_sum2, Float, empty> sum2;
        if constexpr (compute_sum2)
            sum2 = zero;

        std::conditional_t<compute_mean, Float, empty> mean;
        if constexpr (compute_mean)
            mean = zero;

        std::conditional_t<compute_sum2cent, Float, empty> sum2cent;
        if constexpr (compute_sum2cent)
            sum2cent = zero;

        for (std::int64_t row = 0; row < row_count; ++row) {
            Float val = data_ptr[row * stride + col];
            if constexpr (weights)
                val *= weights_ptr[row];

            if constexpr (compute_sum)
                sum += val;
            if constexpr (compute_sum2)
                sum2 += (val * val);
            if constexpr (compute_min)
                min = sycl::fmin(min, val);
            if constexpr (compute_max)
                max = sycl::fmax(max, val);

            if constexpr (compute_sum2cent || compute_mean) {
                const Float delta = val - mean;
                const Float inv_n = one / (row + one);

                if constexpr (compute_mean)
                    mean += delta * inv_n;
                if constexpr (compute_sum2cent)
                    sum2cent += delta * (val - mean);
            }
        }

        if constexpr (output_sum)
            sum_ptr[col] = sum;
        if constexpr (output_min)
            min_ptr[col] = min;
        if constexpr (output_max)
            max_ptr[col] = max;
        if constexpr (output_sum2)
            sum2_ptr[col] = sum2;
        if constexpr (output_mean)
            mean_ptr[col] = mean;
        if constexpr (output_sorm)
            sorm_ptr[col] = sum2 / row_count;
        if constexpr (output_sum2cent)
            sum2cent_ptr[col] = sum2cent;

        std::conditional_t<compute_varc, Float, empty> variance;
        if constexpr (compute_varc)
            variance = sum2cent / (row_count - one);

        std::conditional_t<compute_stdev, Float, empty> stdev;
        if constexpr (compute_stdev)
            stdev = sycl::sqrt(variance);

        if constexpr (output_vart)
            vart_ptr[col] = stdev / mean;
        if constexpr (output_varc)
            varc_ptr[col] = variance;
        if constexpr (output_stdev)
            stdev_ptr[col] = stdev;
    }

private:
    const Float* const data_ptr;
    const std::int64_t stride, row_count, column_count;
};

template <typename Float, bs_list params, bool weights = false>
struct block_processor_kernel {
    struct empty {};

    std::conditional_t<weights, const Float*, empty> weights_ptr;

    constexpr static bool compute_min = check_mask_flag(bs_list::min, params);
    std::conditional_t<compute_min, Float*, empty> min_ptr;

    constexpr static bool compute_max = check_mask_flag(bs_list::max, params);
    std::conditional_t<compute_max, Float*, empty> max_ptr;

    constexpr static auto sum2_cond = bs_list::sum2 | bs_list::sorm;
    constexpr static bool compute_sum2 = check_mask_flag(sum2_cond, params);
    std::conditional_t<compute_sum2, Float*, empty> sum2_ptr;

    constexpr static auto sum_cond = bs_list::sum | bs_list::mean | sum2cent_based_stat;
    constexpr static bool compute_sum = check_mask_flag(sum_cond, params);
    std::conditional_t<compute_sum, Float*, empty> sum_ptr;

    constexpr static bool compute_sum2cent = check_mask_flag(sum2cent_based_stat, params);
    std::conditional_t<compute_sum2cent, Float*, empty> sum2cent_ptr;

    constexpr static auto brc_cond = bs_list::mean | sum2cent_based_stat;
    constexpr static bool compute_brc = check_mask_flag(brc_cond, params);
    std::conditional_t<compute_brc, std::int64_t*, empty> brc_ptr;

    block_processor_kernel(const Float* data,
                           std::int64_t stride,
                           std::int64_t row_count,
                           std::int64_t column_count,
                           std::int64_t row_block_size)
            : data_ptr{ data },
              stride{ stride },
              row_count{ row_count },
              column_count{ column_count },
              row_block_size{ row_block_size } {
        ONEDAL_ASSERT(row_count >= row_block_size);
    }

    void operator()(sycl::nd_item<2> item) const {
        constexpr Float zero = 0, one = 1;
        using limits_t = std::numeric_limits<Float>;
        constexpr Float maximum = limits_t::max();

        const auto col = item.get_global_id(1);
        const auto row_block = item.get_global_id(0);

        if (column_count <= std::int64_t(col))
            return;

        std::conditional_t<compute_min, Float, empty> min;
        if constexpr (compute_min)
            min = +maximum;

        std::conditional_t<compute_max, Float, empty> max;
        if constexpr (compute_max)
            max = -maximum;

        std::conditional_t<compute_sum, Float, empty> sum;
        if constexpr (compute_sum)
            sum = zero;

        std::conditional_t<compute_sum2, Float, empty> sum2;
        if constexpr (compute_sum2)
            sum2 = zero;

        std::conditional_t<compute_sum2cent, Float, empty> mean;
        if constexpr (compute_sum2cent)
            mean = zero;

        std::conditional_t<compute_sum2cent, Float, empty> sum2cent;
        if constexpr (compute_sum2cent)
            sum2cent = zero;

        const std::int64_t f_row = row_block_size * row_block;
        const auto l_row = std::min(row_count, f_row + row_block_size);

        for (std::int64_t row = f_row; row < l_row; ++row) {
            Float val = data_ptr[row * stride + col];
            if constexpr (weights)
                val *= weights_ptr[row];

            if constexpr (compute_sum)
                sum += val;
            if constexpr (compute_sum2)
                sum2 += (val * val);
            if constexpr (compute_min)
                min = sycl::fmin(min, val);
            if constexpr (compute_max)
                max = sycl::fmax(max, val);

            if constexpr (compute_sum2cent) {
                const Float delta = val - mean;
                const Float rel_row = row - f_row;
                const Float inv_n = one / (rel_row + one);

                mean += delta * inv_n;
                sum2cent += delta * (val - mean);
            }
        }

        const auto row_block_count = row_count / row_block_size //
                                     + bool(row_count % row_block_size);
        const auto idx = col * row_block_count + row_block;
        if constexpr (compute_sum)
            sum_ptr[idx] = sum;
        if constexpr (compute_min)
            min_ptr[idx] = min;
        if constexpr (compute_max)
            max_ptr[idx] = max;
        if constexpr (compute_sum2)
            sum2_ptr[idx] = sum2;
        if constexpr (compute_brc)
            brc_ptr[idx] = (l_row - f_row);
        if constexpr (compute_sum2cent)
            sum2cent_ptr[idx] = sum2cent;
    }

private:
    const Float* const data_ptr;
    const std::int64_t stride, row_count, //
        column_count, row_block_size;
};

/* block processing kernel for device execution */
template <typename Float, bs_list List, bool DefferedFin>
inline void merge_blocks_kernel(sycl::nd_item<1> item,
                                const std::int64_t* brc_ptr,
                                const Float* bmin_ptr,
                                const Float* bmax_ptr,
                                const Float* bsum_ptr,
                                const Float* bsum2_ptr,
                                const Float* bsum2cent_ptr,
                                std::int64_t* lrc_ptr,
                                Float* lmin_ptr,
                                Float* lmax_ptr,
                                Float* lsum_ptr,
                                Float* lsum2_ptr,
                                Float* lsum2cent_ptr,
                                Float* lmean_ptr,
                                Float* rmin_ptr,
                                Float* rmax_ptr,
                                Float* rsum_ptr,
                                Float* rsum2_ptr,
                                Float* rsum2cent_ptr,
                                Float* rmean_ptr,
                                Float* rsorm_ptr,
                                Float* rvarc_ptr,
                                Float* rstdev_ptr,
                                Float* rvart_ptr,
                                std::int64_t id,
                                std::int64_t group_id,
                                std::int64_t local_size,
                                std::int64_t block_count) {
    Float mrgmin = Float(0);
    if constexpr (check_mask_flag(bs_list::min, List)) {
        mrgmin = bmin_ptr[group_id * block_count + id];
    }
    Float mrgmax = Float(0);
    if constexpr (check_mask_flag(bs_list::max, List)) {
        mrgmax = bmax_ptr[group_id * block_count + id];
    }
    Float mrgsum = Float(0);
    Float mrgsum2 = Float(0);
    Float mrgvectors = Float(0);
    Float mrgsum2cent = Float(0);
    Float mrgmean = Float(0);

    if constexpr (check_mask_flag(bs_list::mean | sum2cent_based_stat, List)) {
        lrc_ptr[id] = 0;
    }

    for (std::int64_t i = id; i < block_count; i += local_size) {
        std::int64_t offset = group_id * block_count + i;

        Float min = Float(0);
        if constexpr (check_mask_flag(bs_list::min, List)) {
            min = bmin_ptr[offset];
        }
        Float max = Float(0);
        if constexpr (check_mask_flag(bs_list::max, List)) {
            max = bmax_ptr[offset];
        }
        Float sum = Float(0);
        if constexpr (check_mask_flag(bs_list::sum | bs_list::mean | sum2cent_based_stat, List)) {
            sum = bsum_ptr[offset];
        }
        Float sum2 = Float(0);
        if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
            sum2 = bsum2_ptr[offset];
        }
        std::int64_t rcnt = 1;
        if constexpr (check_mask_flag(bs_list::mean | sum2cent_based_stat, List)) {
            rcnt = brc_ptr[offset];
        }
        Float sum2cent = Float(0);
        if constexpr (check_mask_flag(sum2cent_based_stat, List)) {
            sum2cent = bsum2cent_ptr[offset];
        }
        Float mean = sum / static_cast<Float>(rcnt);

        const Float sum_n1n2 = mrgvectors + static_cast<Float>(rcnt);
        const Float mul_n1n2 = mrgvectors * static_cast<Float>(rcnt);
        const Float delta_scale = mul_n1n2 / sum_n1n2;
        const Float mean_scale = Float(1) / sum_n1n2;
        const Float delta = mean - mrgmean;

        mrgmin = sycl::fmin(min, mrgmin);
        mrgmax = sycl::fmax(max, mrgmax);
        mrgsum += sum;
        mrgsum2 += sum2;
        mrgsum2cent = mrgsum2cent + sum2cent + delta * delta * delta_scale;
        mrgmean = (mrgmean * mrgvectors + mean * static_cast<Float>(rcnt)) * mean_scale;
        mrgvectors = sum_n1n2;

        if constexpr (check_mask_flag(bs_list::min, List)) {
            lmin_ptr[id] = mrgmin;
        }
        if constexpr (check_mask_flag(bs_list::max, List)) {
            lmax_ptr[id] = mrgmax;
        }
        if constexpr (check_mask_flag(bs_list::sum | bs_list::mean | sum2cent_based_stat, List)) {
            lsum_ptr[id] = mrgsum;
        }
        if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
            lsum2_ptr[id] = mrgsum2;
        }
        if constexpr (check_mask_flag(bs_list::mean | sum2cent_based_stat, List)) {
            lrc_ptr[id] += rcnt;
        }
        if constexpr (check_mask_flag(sum2cent_based_stat, List)) {
            lsum2cent_ptr[id] = mrgsum2cent;
        }
        if constexpr (check_mask_flag(bs_list::mean | sum2cent_based_stat, List)) {
            lmean_ptr[id] = mrgmean;
        }
    }

    for (std::int64_t stride = sycl::min(local_size, block_count) / 2; stride > 0; stride /= 2) {
        item.barrier(sycl::access::fence_space::local_space);

        if (stride > id) {
            std::int64_t offset = id + stride;

            Float min = Float(0);
            if constexpr (check_mask_flag(bs_list::min, List)) {
                min = lmin_ptr[offset];
            }
            Float max = Float(0);
            if constexpr (check_mask_flag(bs_list::max, List)) {
                max = lmax_ptr[offset];
            }
            Float sum = Float(0);
            if constexpr (check_mask_flag(bs_list::sum | bs_list::mean | sum2cent_based_stat,
                                          List)) {
                sum = lsum_ptr[offset];
            }
            Float sum2 = Float(0);
            if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
                sum2 = lsum2_ptr[offset];
            }
            std::int64_t rcnt = 1;
            if constexpr (check_mask_flag(bs_list::mean | sum2cent_based_stat, List)) {
                rcnt = lrc_ptr[offset];
            }
            Float sum2cent = Float(0);
            if constexpr (check_mask_flag(sum2cent_based_stat, List)) {
                sum2cent = lsum2cent_ptr[offset];
            }
            Float mean = Float(0);
            if constexpr (check_mask_flag(bs_list::mean | sum2cent_based_stat, List)) {
                mean = lmean_ptr[offset];
            }

            const Float sum_n1n2 = mrgvectors + static_cast<Float>(rcnt);
            const Float mul_n1n2 = mrgvectors * static_cast<Float>(rcnt);
            const Float delta_scale = mul_n1n2 / sum_n1n2;
            const Float mean_scale = Float(1) / sum_n1n2;
            const Float delta = mean - mrgmean;

            mrgmin = sycl::fmin(min, mrgmin);
            mrgmax = sycl::fmax(max, mrgmax);
            mrgsum += sum;
            mrgsum2 += sum2;
            mrgsum2cent = mrgsum2cent + sum2cent + delta * delta * delta_scale;
            mrgmean = (mrgmean * mrgvectors + mean * static_cast<Float>(rcnt)) * mean_scale;
            mrgvectors = sum_n1n2;

            // item 0 collects all results in private vars
            // but all others need to store it
            if (0 < id) {
                if constexpr (check_mask_flag(bs_list::min, List)) {
                    lmin_ptr[id] = mrgmin;
                }
                if constexpr (check_mask_flag(bs_list::max, List)) {
                    lmax_ptr[id] = mrgmax;
                }
                if constexpr (check_mask_flag(bs_list::sum | bs_list::mean | sum2cent_based_stat,
                                              List)) {
                    lsum_ptr[id] = mrgsum;
                }
                if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
                    lsum2_ptr[id] = mrgsum2;
                }
                if constexpr (check_mask_flag(bs_list::mean | sum2cent_based_stat, List)) {
                    lrc_ptr[id] += rcnt;
                }
                if constexpr (check_mask_flag(sum2cent_based_stat, List)) {
                    lsum2cent_ptr[id] = mrgsum2cent;
                }
                if constexpr (check_mask_flag(bs_list::mean | sum2cent_based_stat, List)) {
                    lmean_ptr[id] = mrgmean;
                }
            }
        }
    }

    if (0 == id) {
        if constexpr (check_mask_flag(bs_list::min, List)) {
            rmin_ptr[group_id] = mrgmin;
        }
        if constexpr (check_mask_flag(bs_list::max, List)) {
            rmax_ptr[group_id] = mrgmax;
        }
        if constexpr (check_mask_flag(bs_list::sum | bs_list::mean | sum2cent_based_stat, List)) {
            rsum_ptr[group_id] = mrgsum;
        }
        if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
            rsum2_ptr[group_id] = mrgsum2;
        }
        if constexpr (check_mask_flag(sum2cent_based_stat, List)) {
            rsum2cent_ptr[group_id] = mrgsum2cent;
        }
        if constexpr (check_mask_flag(bs_list::mean, List)) {
            rmean_ptr[group_id] = mrgmean;
        }

        if constexpr (!DefferedFin) {
            Float mrgvariance = mrgsum2cent / (mrgvectors - Float(1));
#if __SYCL_COMPILER_VERSION >= 20240715
            Float mrgstdev = (Float)sycl::sqrt(mrgvariance);
#else
            Float mrgstdev = (Float)sqrt(mrgvariance);
#endif

            if constexpr (check_mask_flag(bs_list::sorm, List)) {
                rsorm_ptr[group_id] = mrgsum2 / mrgvectors;
            }
            if constexpr (check_mask_flag(bs_list::varc, List)) {
                rvarc_ptr[group_id] = mrgvariance;
            }
            if constexpr (check_mask_flag(bs_list::stdev, List)) {
                rstdev_ptr[group_id] = mrgstdev;
            }
            if constexpr (check_mask_flag(bs_list::vart, List)) {
                rvart_ptr[group_id] = mrgstdev / mrgmean;
            }
        }
    }
}

template <typename Float, bs_list List>
std::tuple<local_result<Float, List>, sycl::event>
compute_kernel_dense_impl<Float, List>::merge_blocks(local_buffer_list<Float, List>&& ndbuf,
                                                     std::int64_t column_count,
                                                     std::int64_t block_count,
                                                     const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(merge_blocks, q_);

    ONEDAL_ASSERT(column_count > 0);
    ONEDAL_ASSERT(block_count > 0);

    const bool distr_mode = comm_.get_rank_count() > 1;
    auto ndres = local_result<Float, List>::empty(q_, column_count, distr_mode);

    // ndres asserts
    ASSERT_IF(bs_list::min, ndres.get_min().get_count() == column_count);
    ASSERT_IF(bs_list::max, ndres.get_max().get_count() == column_count);
    if (distr_mode) {
        ASSERT_IF(bs_list::mean | sum2cent_based_stat, ndres.get_sum().get_count() == column_count);
    }
    else {
        ASSERT_IF(bs_list::sum, ndres.get_sum().get_count() == column_count);
    }
    ASSERT_IF(bs_list::sum2 | bs_list::sorm, ndres.get_sum2().get_count() == column_count);
    if (distr_mode) {
        ASSERT_IF(bs_list::varc | bs_list::stdev | bs_list::vart,
                  ndres.get_sum2cent().get_count() == column_count);
    }
    else {
        ASSERT_IF(bs_list::sum2cent, ndres.get_sum2cent().get_count() == column_count);
    }
    ASSERT_IF(bs_list::mean, ndres.get_mean().get_count() == column_count);
    ASSERT_IF(bs_list::sorm, ndres.get_sorm().get_count() == column_count);
    ASSERT_IF(bs_list::varc, ndres.get_varc().get_count() == column_count);
    ASSERT_IF(bs_list::stdev, ndres.get_stdev().get_count() == column_count);
    ASSERT_IF(bs_list::vart, ndres.get_vart().get_count() == column_count);

    // ndbuf asserts
    ASSERT_IF(bs_list::mean | sum2cent_based_stat,
              ndbuf.get_rc_list().get_count() == block_count * column_count);
    ASSERT_IF(bs_list::min, ndbuf.get_min().get_count() == block_count * column_count);
    ASSERT_IF(bs_list::max, ndbuf.get_max().get_count() == block_count * column_count);
    ASSERT_IF(bs_list::sum | bs_list::mean | sum2cent_based_stat,
              ndbuf.get_sum().get_count() == block_count * column_count);
    ASSERT_IF(bs_list::sum2 | bs_list::sorm,
              ndbuf.get_sum2().get_count() == block_count * column_count);
    ASSERT_IF(sum2cent_based_stat, ndbuf.get_sum2cent().get_count() == block_count * column_count);

    const std::int64_t* brc_ptr = ndbuf.get_rc_list().get_data();
    const Float* bmin_ptr = ndbuf.get_min().get_data();
    const Float* bmax_ptr = ndbuf.get_max().get_data();
    const Float* bsum_ptr = ndbuf.get_sum().get_data();
    const Float* bsum2_ptr = ndbuf.get_sum2().get_data();
    const Float* bsum2cent_ptr = ndbuf.get_sum2cent().get_data();

    DECLSET_IF(Float*, rmin_ptr, bs_list::min, ndres.get_min().get_mutable_data())
    DECLSET_IF(Float*, rmax_ptr, bs_list::max, ndres.get_max().get_mutable_data())
    DECLSET_IF(Float*,
               rsum2_ptr,
               bs_list::sum2 | bs_list::sorm,
               ndres.get_sum2().get_mutable_data())

    Float* rsum_ptr = nullptr;
    if (distr_mode) {
        SET_IF(rsum_ptr, bs_list::mean | sum2cent_based_stat, ndres.get_sum().get_mutable_data())
    }
    else {
        SET_IF(rsum_ptr, bs_list::sum, ndres.get_sum().get_mutable_data())
    }

    Float* rsum2cent_ptr = nullptr;
    if (distr_mode) {
        SET_IF(rsum2cent_ptr,
               bs_list::varc | bs_list::stdev | bs_list::vart,
               ndres.get_sum2cent().get_mutable_data())
    }
    else {
        SET_IF(rsum2cent_ptr, bs_list::sum2cent, ndres.get_sum2cent().get_mutable_data())
    }

    DECLSET_IF(Float*, rmean_ptr, bs_list::mean, ndres.get_mean().get_mutable_data())
    DECLSET_IF(Float*, rsorm_ptr, bs_list::sorm, ndres.get_sorm().get_mutable_data())
    DECLSET_IF(Float*, rvarc_ptr, bs_list::varc, ndres.get_varc().get_mutable_data())
    DECLSET_IF(Float*, rstdev_ptr, bs_list::stdev, ndres.get_stdev().get_mutable_data())
    DECLSET_IF(Float*, rvart_ptr, bs_list::vart, ndres.get_vart().get_mutable_data())

    std::int64_t local_size = bk::device_max_sg_size(q_);
    auto global_size = de::check_mul_overflow(column_count, local_size);

    constexpr bool deffered_fin_true = true;
    constexpr bool deffered_fin_false = false;

    const sycl::nd_range<1> nd_range = bk::make_multiple_nd_range_1d(global_size, local_size);

    std::int64_t local_buffer_size = local_size;
    auto last_event = q_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<std::int64_t> lrc_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lmin_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lmax_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lsum_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lsum2_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lsum2cent_buf(local_buffer_size, cgh);
        local_accessor_rw_t<Float> lmean_buf(local_buffer_size, cgh);

        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            const std::int64_t local_size = item.get_local_range()[0];
            const std::int64_t id = item.get_local_id()[0];
            const std::int64_t group_id = item.get_group().get_group_id(0);
#if __SYCL_COMPILER_VERSION >= 20230828
            std::int64_t* lrc_ptr =
                lrc_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            Float* lmin_ptr =
                lmin_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            Float* lmax_ptr =
                lmax_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            Float* lsum_ptr =
                lsum_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            Float* lsum2_ptr =
                lsum2_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            Float* lsum2cent_ptr =
                lsum2cent_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            Float* lmean_ptr =
                lmean_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
#else
            std::int64_t* lrc_ptr = lrc_buf.get_pointer().get();
            Float* lmin_ptr = lmin_buf.get_pointer().get();
            Float* lmax_ptr = lmax_buf.get_pointer().get();
            Float* lsum_ptr = lsum_buf.get_pointer().get();
            Float* lsum2_ptr = lsum2_buf.get_pointer().get();
            Float* lsum2cent_ptr = lsum2cent_buf.get_pointer().get();
            Float* lmean_ptr = lmean_buf.get_pointer().get();
#endif
            if (distr_mode) {
                merge_blocks_kernel<Float, List, deffered_fin_true>(item,
                                                                    brc_ptr,
                                                                    bmin_ptr,
                                                                    bmax_ptr,
                                                                    bsum_ptr,
                                                                    bsum2_ptr,
                                                                    bsum2cent_ptr,
                                                                    lrc_ptr,
                                                                    lmin_ptr,
                                                                    lmax_ptr,
                                                                    lsum_ptr,
                                                                    lsum2_ptr,
                                                                    lsum2cent_ptr,
                                                                    lmean_ptr,
                                                                    rmin_ptr,
                                                                    rmax_ptr,
                                                                    rsum_ptr,
                                                                    rsum2_ptr,
                                                                    rsum2cent_ptr,
                                                                    rmean_ptr,
                                                                    rsorm_ptr,
                                                                    rvarc_ptr,
                                                                    rstdev_ptr,
                                                                    rvart_ptr,
                                                                    id,
                                                                    group_id,
                                                                    local_size,
                                                                    block_count);
            }
            else {
                merge_blocks_kernel<Float, List, deffered_fin_false>(item,
                                                                     brc_ptr,
                                                                     bmin_ptr,
                                                                     bmax_ptr,
                                                                     bsum_ptr,
                                                                     bsum2_ptr,
                                                                     bsum2cent_ptr,
                                                                     lrc_ptr,
                                                                     lmin_ptr,
                                                                     lmax_ptr,
                                                                     lsum_ptr,
                                                                     lsum2_ptr,
                                                                     lsum2cent_ptr,
                                                                     lmean_ptr,
                                                                     rmin_ptr,
                                                                     rmax_ptr,
                                                                     rsum_ptr,
                                                                     rsum2_ptr,
                                                                     rsum2cent_ptr,
                                                                     rmean_ptr,
                                                                     rsorm_ptr,
                                                                     rvarc_ptr,
                                                                     rstdev_ptr,
                                                                     rvart_ptr,
                                                                     id,
                                                                     group_id,
                                                                     local_size,
                                                                     block_count);
            }
        });
    });

    last_event.wait_and_throw();
    return std::make_tuple(std::move(ndres), std::move(last_event));
}

template <typename Float, bs_list List>
template <bool use_weights>
std::tuple<local_result<Float, List>, sycl::event>
compute_kernel_dense_impl<Float, List>::compute_single_pass(const pr::ndview<Float, 2>& data,
                                                            const pr::ndview<Float, 2>& weights) {
    ONEDAL_PROFILER_TASK(process_single_block, q_);

    ONEDAL_ASSERT(data.has_data());

    const auto row_count = data.get_dimension(0);
    const auto column_count = data.get_dimension(1);
    const auto stride = data.get_leading_stride();

    const bool distr_mode = comm_.get_rank_count() > 1;

    auto ndres = local_result<Float, List>::empty(q_, column_count, distr_mode);

    ASSERT_IF(bs_list::min, ndres.get_min().get_count() == column_count);
    ASSERT_IF(bs_list::max, ndres.get_max().get_count() == column_count);
    if (distr_mode) {
        ASSERT_IF(bs_list::mean | sum2cent_based_stat, ndres.get_sum().get_count() == column_count);
    }
    else {
        ASSERT_IF(bs_list::sum, ndres.get_sum().get_count() == column_count);
    }
    ASSERT_IF(bs_list::sum2 | bs_list::sorm, ndres.get_sum2().get_count() == column_count);
    if (distr_mode) {
        ASSERT_IF(bs_list::varc | bs_list::stdev | bs_list::vart,
                  ndres.get_sum2cent().get_count() == column_count);
    }
    else {
        ASSERT_IF(bs_list::sum2cent, ndres.get_sum2cent().get_count() == column_count);
    }
    ASSERT_IF(bs_list::mean, ndres.get_mean().get_count() == column_count);
    ASSERT_IF(bs_list::sorm, ndres.get_sorm().get_count() == column_count);
    ASSERT_IF(bs_list::varc, ndres.get_varc().get_count() == column_count);
    ASSERT_IF(bs_list::stdev, ndres.get_stdev().get_count() == column_count);
    ASSERT_IF(bs_list::vart, ndres.get_vart().get_count() == column_count);

    const auto* const data_ptr = data.get_data();

    DECLSET_IF(Float*, rmin_ptr, bs_list::min, ndres.get_min().get_mutable_data())
    DECLSET_IF(Float*, rmax_ptr, bs_list::max, ndres.get_max().get_mutable_data())
    DECLSET_IF(Float*,
               rsum2_ptr,
               bs_list::sum2 | bs_list::sorm,
               ndres.get_sum2().get_mutable_data())

    Float* rsum_ptr = nullptr;
    if (distr_mode) {
        SET_IF(rsum_ptr, bs_list::mean | sum2cent_based_stat, ndres.get_sum().get_mutable_data())
    }
    else {
        SET_IF(rsum_ptr, bs_list::sum, ndres.get_sum().get_mutable_data())
    }

    Float* rsum2cent_ptr = nullptr;
    if (distr_mode) {
        SET_IF(rsum2cent_ptr,
               bs_list::varc | bs_list::stdev | bs_list::vart,
               ndres.get_sum2cent().get_mutable_data())
    }
    else {
        SET_IF(rsum2cent_ptr, bs_list::sum2cent, ndres.get_sum2cent().get_mutable_data())
    }

    DECLSET_IF(Float*, rmean_ptr, bs_list::mean, ndres.get_mean().get_mutable_data())
    DECLSET_IF(Float*, rsorm_ptr, bs_list::sorm, ndres.get_sorm().get_mutable_data())
    DECLSET_IF(Float*, rvarc_ptr, bs_list::varc, ndres.get_varc().get_mutable_data())
    DECLSET_IF(Float*, rstdev_ptr, bs_list::stdev, ndres.get_stdev().get_mutable_data())
    DECLSET_IF(Float*, rvart_ptr, bs_list::vart, ndres.get_vart().get_mutable_data())

    const auto wg = bk::device_max_wg_size(q_);
    const auto nd_range = bk::make_multiple_nd_range_1d(column_count, wg);

    const auto setup_kernel = [=](auto& kernel) -> void {
        using kernel_t = std::remove_cv_t< //
            std::remove_reference_t<decltype(kernel)>>;

        if constexpr (use_weights) {
            ONEDAL_ASSERT(weights.has_data());
            ONEDAL_ASSERT(row_count == weights.get_dimension(0));
            ONEDAL_ASSERT(std::int64_t(1) == weights.get_dimension(1));

            kernel.weights_ptr = weights.get_data();
        }

        if constexpr (kernel_t::output_min)
            kernel.min_ptr = rmin_ptr;
        if constexpr (kernel_t::output_max)
            kernel.max_ptr = rmax_ptr;
        if constexpr (kernel_t::output_sum)
            kernel.sum_ptr = rsum_ptr;
        if constexpr (kernel_t::output_sum2)
            kernel.sum2_ptr = rsum2_ptr;
        if constexpr (kernel_t::output_sorm)
            kernel.sorm_ptr = rsorm_ptr;
        if constexpr (kernel_t::output_varc)
            kernel.varc_ptr = rvarc_ptr;
        if constexpr (kernel_t::output_vart)
            kernel.vart_ptr = rvart_ptr;
        if constexpr (kernel_t::output_mean)
            kernel.mean_ptr = rmean_ptr;
        if constexpr (kernel_t::output_stdev)
            kernel.stdev_ptr = rstdev_ptr;
        if constexpr (kernel_t::output_sum2cent)
            kernel.sum2cent_ptr = rsum2cent_ptr;
    };

    auto last_event = q_.submit([&](sycl::handler& cgh) {
        if (distr_mode) {
            using kernel_t = singlepass_processor_kernel<Float,
                                                         List, //
                                                         true,
                                                         use_weights>;
            kernel_t kernel(data_ptr, stride, row_count, column_count);

            setup_kernel(kernel);

            ONEDAL_ASSERT(kernel.verify_pointers());

            cgh.parallel_for(nd_range, kernel);
        }
        else {
            using kernel_t = singlepass_processor_kernel<Float,
                                                         List, //
                                                         false,
                                                         use_weights>;
            kernel_t kernel(data_ptr, stride, row_count, column_count);

            setup_kernel(kernel);

            ONEDAL_ASSERT(kernel.verify_pointers());

            cgh.parallel_for(nd_range, kernel);
        }
    });

    return std::make_tuple(std::move(ndres), last_event);
}

template <typename Float, bs_list List>
template <bool use_weights>
std::tuple<local_result<Float, List>, sycl::event>
compute_kernel_dense_impl<Float, List>::compute_by_blocks(const pr::ndview<Float, 2>& data,
                                                          std::int64_t row_block_count,
                                                          const pr::ndview<Float, 2>& weights) {
    ONEDAL_ASSERT(data.has_data());

    const auto row_count = data.get_dimension(0);
    const auto column_count = data.get_dimension(1);
    const auto stride = data.get_leading_stride();

    const auto aux_buf_size = de::check_mul_overflow(row_block_count, column_count);
    auto ndbuf = local_buffer_list<Float, List>::empty(q_, aux_buf_size);

    // ndbuf asserts
    ASSERT_IF(bs_list::mean | sum2cent_based_stat, ndbuf.get_rc_list().get_count() == aux_buf_size);
    ASSERT_IF(bs_list::min, ndbuf.get_min().get_count() == aux_buf_size);
    ASSERT_IF(bs_list::max, ndbuf.get_max().get_count() == aux_buf_size);
    ASSERT_IF(bs_list::sum | bs_list::mean | sum2cent_based_stat,
              ndbuf.get_sum().get_count() == aux_buf_size);
    ASSERT_IF(bs_list::sum2 | bs_list::sorm, ndbuf.get_sum2().get_count() == aux_buf_size);
    ASSERT_IF(sum2cent_based_stat, ndbuf.get_sum2cent().get_count() == aux_buf_size);

    DECLSET_IF(std::int64_t*,
               ablock_rc_ptr,
               bs_list::mean | sum2cent_based_stat,
               ndbuf.get_rc_list().get_mutable_data())
    DECLSET_IF(Float*, amin_ptr, bs_list::min, ndbuf.get_min().get_mutable_data())
    DECLSET_IF(Float*, amax_ptr, bs_list::max, ndbuf.get_max().get_mutable_data())
    DECLSET_IF(Float*,
               asum_ptr,
               bs_list::sum | bs_list::mean | sum2cent_based_stat,
               ndbuf.get_sum().get_mutable_data())
    DECLSET_IF(Float*,
               asum2_ptr,
               bs_list::sum2 | bs_list::sorm,
               ndbuf.get_sum2().get_mutable_data())
    DECLSET_IF(Float*, asum2cent_ptr, sum2cent_based_stat, ndbuf.get_sum2cent().get_mutable_data())

    const auto* data_ptr = data.get_data();
    const auto wg_size = be::device_max_wg_size(q_);
    const auto local_size = (wg_size < column_count) ? wg_size : column_count;

    const auto row_block_size = (row_count + row_block_count - 1) / row_block_count;

    const auto nd_range = bk::make_multiple_nd_range_2d( //
        { row_block_count, column_count },
        { 1l, local_size });

    sycl::event last_event;
    {
        ONEDAL_PROFILER_TASK(process_blocks, q_);

        last_event = q_.submit([&](sycl::handler& cgh) {
            using kernel_t = block_processor_kernel<Float, List, use_weights>;
            kernel_t kernel(data_ptr, stride, row_count, column_count, row_block_size);

            if constexpr (use_weights) {
                ONEDAL_ASSERT(weights.has_data());
                ONEDAL_ASSERT(row_count == weights.get_dimension(0));
                ONEDAL_ASSERT(std::int64_t(1) == weights.get_dimension(1));

                kernel.weights_ptr = weights.get_data();
            }

            if constexpr (kernel_t::compute_min)
                kernel.min_ptr = amin_ptr;
            if constexpr (kernel_t::compute_max)
                kernel.max_ptr = amax_ptr;
            if constexpr (kernel_t::compute_sum)
                kernel.sum_ptr = asum_ptr;
            if constexpr (kernel_t::compute_sum2)
                kernel.sum2_ptr = asum2_ptr;
            if constexpr (kernel_t::compute_brc)
                kernel.brc_ptr = ablock_rc_ptr;
            if constexpr (kernel_t::compute_sum2cent)
                kernel.sum2cent_ptr = asum2cent_ptr;

            cgh.parallel_for(nd_range, kernel);
        });
    }

    auto [ndres, merge_event] =
        merge_blocks(std::move(ndbuf), column_count, row_block_count, { last_event });

    return std::make_tuple(std::move(ndres), merge_event);
}

template <typename Float, bs_list List>
std::tuple<local_result<Float, List>, sycl::event> compute_kernel_dense_impl<Float, List>::finalize(
    local_result_t&& ndres,
    std::int64_t row_count,
    std::int64_t column_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(finalize, q_);

    const bool distr_mode = comm_.get_rank_count() > 1;
    // ndres asserts
    ASSERT_IF(bs_list::min, ndres.get_min().get_count() == column_count);
    ASSERT_IF(bs_list::max, ndres.get_max().get_count() == column_count);
    if (distr_mode) {
        ASSERT_IF(bs_list::mean | sum2cent_based_stat, ndres.get_sum().get_count() == column_count);
    }
    else {
        ASSERT_IF(bs_list::sum, ndres.get_sum().get_count() == column_count);
    }
    ASSERT_IF(bs_list::sum2 | bs_list::sorm, ndres.get_sum2().get_count() == column_count);
    if (distr_mode) {
        ASSERT_IF(bs_list::varc | bs_list::stdev | bs_list::vart,
                  ndres.get_sum2cent().get_count() == column_count);
    }
    else {
        ASSERT_IF(bs_list::sum2cent, ndres.get_sum2cent().get_count() == column_count);
    }

    sycl::event last_event;
    auto rows_count_global = row_count;
    if (distr_mode) {
        comm_.allreduce(rows_count_global, spmd::reduce_op::sum).wait();
        if constexpr (check_mask_flag(bs_list::min, List)) {
            ONEDAL_PROFILER_TASK(allreduce_min, q_);
            comm_.allreduce(ndres.get_min().flatten(q_, deps), spmd::reduce_op::min).wait();
        }
        if constexpr (check_mask_flag(bs_list::max, List)) {
            ONEDAL_PROFILER_TASK(allreduce_max, q_);
            comm_.allreduce(ndres.get_max().flatten(q_, deps), spmd::reduce_op::max).wait();
        }
        if constexpr (check_mask_flag(bs_list::sum2 | bs_list::sorm, List)) {
            ONEDAL_PROFILER_TASK(allreduce_sum2, q_);
            comm_.allreduce(ndres.get_sum2().flatten(q_, deps), spmd::reduce_op::sum).wait();
        }
        const Float* bsum_ptr = ndres.get_sum().get_data();
        const Float* bsum2_ptr = ndres.get_sum2().get_data();
        const Float* bsum2cent_ptr = ndres.get_sum2cent().get_data();
        const Float* bmean_ptr = ndres.get_mean().get_data();
        const sycl::range<1> range{ de::integral_cast<std::size_t>(column_count) };
        DECLSET_IF(Float*, rmean_ptr, bs_list::mean, ndres.get_mean().get_mutable_data())
        DECLSET_IF(Float*, rsorm_ptr, bs_list::sorm, ndres.get_sorm().get_mutable_data())
        DECLSET_IF(Float*, rvarc_ptr, bs_list::varc, ndres.get_varc().get_mutable_data())
        DECLSET_IF(Float*, rstdev_ptr, bs_list::stdev, ndres.get_stdev().get_mutable_data())
        DECLSET_IF(Float*, rvart_ptr, bs_list::vart, ndres.get_vart().get_mutable_data())

        if constexpr (check_mask_flag(bs_list::mean | sum2cent_based_stat, List)) {
            ONEDAL_PROFILER_TASK(allreduce_sum, q_);
            comm_.allreduce(ndres.get_sum().flatten(q_, deps), spmd::reduce_op::sum).wait();
        }
        DECLSET_IF(Float*,
                   rsum2cent_ptr,
                   bs_list::varc | bs_list::stdev | bs_list::vart,
                   ndres.get_sum2cent().get_mutable_data());
        const Float inv_n = Float(1.0 / double(rows_count_global));

        last_event = q_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(range, [=](sycl::id<1> id) {
                Float mrgsum2cent = bsum2cent_ptr[id];
                Float mrgmean = bsum_ptr[id] * inv_n;

                Float local_mean = bmean_ptr[id];

                Float delta = mrgmean - local_mean;

                mrgsum2cent += delta * delta * row_count;
                if constexpr (check_mask_flag(bs_list::mean, List)) {
                    rmean_ptr[id] = mrgmean;
                }
                if constexpr (check_mask_flag(bs_list::sum2cent, List)) {
                    rsum2cent_ptr[id] = mrgsum2cent;
                }
                if constexpr (check_mask_flag(bs_list::varc, List)) {
                    rvarc_ptr[id] = mrgsum2cent / (rows_count_global - 1);
                }
                if constexpr (check_mask_flag(bs_list::sorm, List)) {
                    rsorm_ptr[id] = bsum2_ptr[id] * inv_n;
                }
            });
        });
        last_event.wait_and_throw();
        if constexpr (check_mask_flag(bs_list::varc | bs_list::stdev | bs_list::vart, List)) {
            ONEDAL_PROFILER_TASK(allreduce_sum2cent, q_);
            comm_.allreduce(ndres.get_sum2cent().flatten(q_, deps), spmd::reduce_op::sum).wait();
        }
        if constexpr (check_mask_flag(bs_list::varc | bs_list::stdev | bs_list::vart, List)) {
            ONEDAL_PROFILER_TASK(allreduce_sum2cent, q_);
            comm_.allreduce(ndres.get_varc().flatten(q_, deps), spmd::reduce_op::sum).wait();
        }
        last_event = q_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(range, [=](sycl::id<1> id) {
                if constexpr (check_mask_flag(bs_list::stdev, List)) {
                    rstdev_ptr[id] = sycl::sqrt(rvarc_ptr[id]);
                }
                if constexpr (check_mask_flag(bs_list::vart, List)) {
                    rvart_ptr[id] = rstdev_ptr[id] / rmean_ptr[id];
                }
            });
        });
        last_event.wait_and_throw();
    }
    else {
        sycl::event::wait_and_throw(deps);
    }
    return std::make_tuple(std::forward<local_result_t>(ndres), std::move(last_event));
}

template <typename Float, bs_list List>
result_t compute_kernel_dense_impl<Float, List>::operator()(const descriptor_t& desc,
                                                            const input_t& input) {
    const auto& data = input.get_data();
    const auto& weights = input.get_weights();

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();

    const auto data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);

    const auto row_block_count = get_row_block_count(row_count);

    local_result<Float, List> ndres;
    sycl::event last_event;

    if (weights.has_data()) {
        const auto weights_nd = pr::table2ndarray<Float>(q_, weights, alloc::device);
        std::tie(ndres, last_event) =
            (row_block_count > 1) ? compute_by_blocks<true>(data_nd, row_block_count, weights_nd)
                                  : compute_single_pass<true>(data_nd, weights_nd);
    }
    else {
        std::tie(ndres, last_event) = (row_block_count > 1)
                                          ? compute_by_blocks<false>(data_nd, row_block_count)
                                          : compute_single_pass<false>(data_nd);
    }

    std::tie(ndres, last_event) =
        finalize(std::move(ndres), row_count, column_count, { last_event });

    return get_result(desc, std::move(ndres), column_count, { last_event })
        .set_result_options(desc.get_result_options());
}

#define INSTANTIATE(LIST)                                  \
    template class compute_kernel_dense_impl<float, LIST>; \
    template class compute_kernel_dense_impl<double, LIST>;

INSTANTIATE(bs_mode_min_max);
INSTANTIATE(bs_mode_mean_variance);
INSTANTIATE(bs_mode_all);

} // namespace oneapi::dal::basic_statistics::backend

#endif // ONEDAL_DATA_PARALLEL
