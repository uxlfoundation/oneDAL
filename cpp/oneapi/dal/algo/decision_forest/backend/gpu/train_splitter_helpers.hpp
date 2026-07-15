/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_helpers.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;
using address = sycl::access::address_space;

using sycl::plus;
using sycl::minimum;
using sycl::maximum;

template <typename T>
using enable_if_float_t = std::enable_if_t<detail::is_valid_float_v<T>>;

template <typename F>
struct float_accuracy;

template <>
struct float_accuracy<float> {
    static constexpr float val = float(1e-5);
};

template <>
struct float_accuracy<double> {
    static constexpr double val = double(1e-10);
};

bool device_has_enough_local_mem(const sycl::queue& queue, std::int64_t required_byte_size) {
    auto device = queue.get_device();
    auto has_local_mem =
        device.get_info<sycl::info::device::local_mem_type>() != sycl::info::local_mem_type::none;
    std::int64_t local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    return has_local_mem && (local_mem_size / 2 >= required_byte_size);
}

template <typename T, typename Index = std::int64_t>
inline T* fill_zero(T* dst, Index elem_count) {
    for (Index i = 0; i < elem_count; ++i) {
        dst[i] = T(0);
    }
    return dst;
}

template <typename T, typename Index = std::int64_t>
inline T* fill_with_group(sycl::nd_item<2>& item, T* dst, Index elem_count, T val) {
    const Index local_id = item.get_local_id(0);
    const Index local_size = item.get_local_range(0);
    for (Index i = local_id; i < elem_count; i += local_size) {
        dst[i] = val;
    }

    item.barrier(sycl::access::fence_space::local_space);

    return dst;
}

template <typename Float, typename Index>
inline void add_val_to_hist(
    typename task_types<Float, Index, task::classification>::hist_type_t* hist_ptr,
    Float val) {
    Index class_id = static_cast<Index>(val);
    hist_ptr[class_id] += 1;
}

template <typename Float, typename Index>
inline void add_val_to_hist(
    typename task_types<Float, Index, task::regression>::hist_type_t* hist_ptr,
    Float val) {
    hist_ptr[0] += Float(1);
    Float inv_n = Float(1) / hist_ptr[0];
    Float delta = val - hist_ptr[1]; // y[i] - mean
    hist_ptr[1] += delta * inv_n; // updated mean
    hist_ptr[2] += delta * (val - hist_ptr[1]); // updated sum2Cent
}

// merge src and dst stat
template <typename Float, typename Index, typename T = enable_if_float_t<Float>>
inline void merge_stat(Float* dst, const Float* src, Index) {
    if (Float(0) == src[0])
        return;

    Float sum_n1n2 = dst[0] + src[0];
    Float mul_n1n2 = dst[0] * src[0];
    Float delta_scl = mul_n1n2 / sum_n1n2;
    Float mean_scl = Float(1) / sum_n1n2;
    Float delta = src[1] - dst[1];

    dst[2] = dst[2] + src[2] + delta * delta * delta_scl;
    dst[1] = (dst[1] * dst[0] + src[1] * src[0]) * mean_scl;
    dst[0] = sum_n1n2;
}

template <typename Index>
inline void merge_stat(Index* dst, const Index* src, Index count) {
    for (Index i = 0; i < count; ++i) {
        dst[i] += src[i];
    }
}

template <typename Index>
inline void merge_stat(Index* dst, Index* accum, const Index* src, Index count) {
    for (Index i = 0; i < count; ++i) {
        *accum += src[i];
        dst[i] += src[i];
    }
}

template <typename Float, typename Index, typename T = enable_if_float_t<Float>>
inline void sub_stat(Float* dst, const Float* src, const Float* mrg, Index elem_count) {
    dst[0] = mrg[0] - src[0];

    dst[1] = (dst[0] >= Float(1)) ? (mrg[1] + (src[0] * (mrg[1] - src[1])) / dst[0]) : Float(0);

    Float sum_n1n2 = mrg[0];
    Float mul_n1n2 = src[0] * dst[0];
    Float delta_scl = mul_n1n2 / sum_n1n2;
    Float delta = src[1] - dst[1];

    dst[2] = (dst[0] >= Float(1)) ? ((mrg[2] - src[2] - delta * delta * delta_scl)) : Float(0);
}

template <typename Float>
inline bool float_eq(Float a, Float b) {
    return sycl::fabs(a - b) <= float_accuracy<Float>::val;
}

template <typename Float>
inline bool float_gt(Float a, Float b) {
    return (a - b) > float_accuracy<Float>::val;
}

template <typename Float, typename Index, typename Task>
struct split_scalar {
    using impl_const_t = impl_const<Index, Task>;
    Index ftr_id;
    Index ftr_bin;
    Index left_count;
    Index right_count;
    Float left_imp;
    Float right_imp;
    Float imp_dec;
    Float left_weight_sum;

    void clear() {
        ftr_id = impl_const_t::leaf_mark_;
        ftr_bin = impl_const_t::leaf_mark_;
        left_count = 0;
        right_count = 0;
        left_imp = Float(0);
        right_imp = Float(0);
        imp_dec = -de::limits<Float>::max();
        left_weight_sum = Float(0);
    }

    void copy(const split_scalar& other) {
        ftr_id = other.ftr_id;
        ftr_bin = other.ftr_bin;
        left_count = other.left_count;
        right_count = other.right_count;
        left_imp = other.left_imp;
        right_imp = other.right_imp;
        imp_dec = other.imp_dec;
        left_weight_sum = other.left_weight_sum;
    }
};

template <typename Float, typename Index, typename Task>
struct split_info {
    using task_t = Task;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;
    using byte_t = std::uint8_t;
    using split_scalar_t = split_scalar<Float, Index, Task>;
    static constexpr Index cache_buf_int_size = 3; // ftr_id, ftr_bin, left_count
    static constexpr Index cache_buf_float_size = 2; // left_imp, and imp_dec

    static constexpr Index get_cache_without_hist_byte_size() {
        return cache_buf_int_size * sizeof(Index) + cache_buf_float_size * sizeof(Float);
    }
    static constexpr Index get_cache_byte_size(Index hist_prop_count) {
        return get_cache_without_hist_byte_size() + hist_prop_count * sizeof(hist_type_t);
    }

    inline void init(Index in_hist_prop_count) {
        hist_prop_count = in_hist_prop_count;
    }

    inline void init(hist_type_t* in_left_hist, Index in_hist_prop_count) {
        left_hist = in_left_hist;
        hist_prop_count = in_hist_prop_count;
    }

    inline void init_clear(hist_type_t* in_left_hist, Index in_hist_prop_count) {
        init(in_left_hist, in_hist_prop_count);
        clear();
    }

    inline void clear_scalar() {
        scalars.clear();
    }

    inline void clear_hist() {
        for (Index i = 0; i < hist_prop_count; i++) {
            left_hist[i] = hist_type_t(0);
        }
    }

    inline void clear() {
        clear_scalar();
        clear_hist();
    }

    split_scalar_t scalars;
    hist_type_t* left_hist;
    Index hist_prop_count;
};

template <typename Float, typename Index, typename Task>
struct split_smp {
    using task_t = Task;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;
    using split_info_t = split_info<Float, Index, Task>;
    using split_scalar_t = split_scalar<Float, Index, Task>;

    // classififcation version
    inline void merge_bin_hist(Index& left_count_,
                               hist_type_t* left_class_hist_,
                               const hist_type_t* bin_hist_ptr,
                               Index class_count) {
        merge_stat(&left_class_hist_[0], &left_count_, bin_hist_ptr, class_count);
    }

    // regression version
    inline void merge_bin_hist(hist_type_t* left_hist_,
                               const hist_type_t* bin_hist_ptr,
                               Index elem_count) {
        merge_stat(&left_hist_[0], bin_hist_ptr, elem_count);
    }

    // classififcation version
    inline void calc_imp_dec(split_info_t& si,
                             const Index* node_ptr,
                             const Float* imp_list_ptr_,
                             const hist_type_t* node_class_hist_list_ptr_,
                             Index class_count,
                             Index node_id,
                             bool is_weighted) {
        Index node_row_count = node_ptr[impl_const_t::ind_grc];
        const Float* node_imp_ptr = imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float node_imp = node_imp_ptr[0];
        const Index* node_class_hist_ptr = node_class_hist_list_ptr_ + node_id * class_count;
        split_scalar_t& sc = si.scalars;

        sc.right_count = node_row_count - sc.left_count;

        const Float divL = (0 < sc.left_count)
                               ? Float(1) / (Float(sc.left_count) * Float(sc.left_count))
                               : Float(0);
        const Float divR = (0 < sc.right_count)
                               ? Float(1) / (Float(sc.right_count) * Float(sc.right_count))
                               : Float(0);

        sc.left_imp = Float(1);
        sc.right_imp = Float(1);

        for (Index class_id = 0; class_id < class_count; ++class_id) {
            sc.left_imp -= Float(si.left_hist[class_id]) * Float(si.left_hist[class_id]) * divL;
            sc.right_imp -= Float(node_class_hist_ptr[class_id] - si.left_hist[class_id]) *
                            Float(node_class_hist_ptr[class_id] - si.left_hist[class_id]) * divR;
        }

        sc.left_imp = sycl::max(sc.left_imp, Float(0));
        sc.right_imp = sycl::max(sc.right_imp, Float(0));

        sc.imp_dec =
            node_imp - (Float(sc.left_count) * sc.left_imp + Float(sc.right_count) * sc.right_imp) /
                           Float(node_row_count);
        if (sc.left_count > 0 && is_weighted) {
            sc.imp_dec /= (sc.left_weight_sum / sc.left_count);
        }
    }

    // Check if node impurity valid. True if valid
    inline bool is_valid_impurity(const Float* node_imp_list_ptr,
                                  Index node_id,
                                  Float imp_threshold,
                                  Index row_count) {
        Float node_imp = Float(0);
        if constexpr (std::is_same_v<task_t, task::classification>) {
            const Float* node_imp_ptr =
                node_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;
            node_imp = node_imp_ptr[0];
        }
        else {
            const Float* node_imp_ptr =
                node_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;
            node_imp = node_imp_ptr[1] / Float(row_count);
        }
        return !float_eq(node_imp, Float(0)) && node_imp >= imp_threshold;
    }

    // regression version
    inline void calc_imp_dec(split_info_t& si,
                             const Index* node_ptr,
                             const Float* imp_list_ptr,
                             Index node_id,
                             bool is_weighted) {
        constexpr Index buff_size = impl_const_t::node_imp_prop_count_ + 1;
        Index node_row_count = node_ptr[impl_const_t::ind_grc];
        split_scalar_t& sc = si.scalars;
        sc.left_count = static_cast<Index>(si.left_hist[0]);

        const Float* node_imp_ptr = imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;

        hist_type_t node_hist[buff_size] = { static_cast<Float>(node_row_count),
                                             node_imp_ptr[0],
                                             node_imp_ptr[1] };
        hist_type_t right_hist[buff_size] = { hist_type_t(0) };

        // getting hist for right part
        sub_stat<Float, Index, task_t>(&right_hist[0], &si.left_hist[0], &node_hist[0], buff_size);

        sc.right_count = node_row_count - sc.left_count;
        sc.imp_dec = node_imp_ptr[1] - (si.left_hist[2] + right_hist[2]);
        if (sc.left_count > 0 && is_weighted) {
            sc.imp_dec *= (sc.left_weight_sum / sc.left_count);
        }
    }

    inline bool test_split_is_best(const split_info_t& bs,
                                   const split_info_t& ts,
                                   Index min_observations_in_leaf_node) {
        const split_scalar_t& ts_sc = ts.scalars;
        const split_scalar_t& bs_sc = bs.scalars;
        return test_split_is_best(bs_sc, ts_sc, min_observations_in_leaf_node);
    }

    inline bool test_split_is_best(const split_scalar_t& bs_sc,
                                   const split_scalar_t& ts_sc,
                                   Index min_observations_in_leaf_node) {
        bool valid_imp_dec = Float(0) < ts_sc.imp_dec;
        bool valid_rcount = ts_sc.right_count >= min_observations_in_leaf_node;
        bool valid_lcount = ts_sc.left_count >= min_observations_in_leaf_node;
        bool bs_unassigned = bs_sc.ftr_bin == impl_const_t::leaf_mark_;
        bool bs_less_imp = float_gt(ts_sc.imp_dec, bs_sc.imp_dec);
        bool bs_eq_imp = float_eq(ts_sc.imp_dec, bs_sc.imp_dec);
        bool bs_greater_ftr = ts_sc.ftr_id < bs_sc.ftr_id;
        bool bs_eq_ftr = bs_sc.ftr_id == ts_sc.ftr_id;
        bool bs_greater_bin = ts_sc.ftr_bin < bs_sc.ftr_bin;
        bool is_best = bs_unassigned || bs_less_imp;
        is_best = is_best || (bs_eq_imp && (bs_greater_ftr || (bs_eq_ftr && bs_greater_bin)));
        return is_best && valid_imp_dec && valid_lcount && valid_rcount;
    }

    // universal
    inline void choose_best_split(split_info_t& bs,
                                  const split_info_t& ts,
                                  Index hist_elem_count,
                                  Index min_obs_in_leaf_node) {
        if (test_split_is_best(bs, ts, min_obs_in_leaf_node)) {
            bs.scalars.copy(ts.scalars);

            for (Index i = 0; i < hist_elem_count; ++i) {
                bs.left_hist[i] = ts.left_hist[i];
            }
        }
    }

    // classififcation version
    inline void update_left_child_imp(Float* left_imp_list_ptr,
                                      hist_type_t* left_class_hist_list_ptr,
                                      Float bs_left_imp_,
                                      const hist_type_t* bs_left_class_hist_,
                                      Index node_id,
                                      Index class_count) {
        Float* left_node_imp_ptr = left_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;

        left_node_imp_ptr[0] = bs_left_imp_;
        hist_type_t* left_node_class_hist_ptr = left_class_hist_list_ptr + node_id * class_count;

        for (Index class_id = 0; class_id < class_count; ++class_id) {
            left_node_class_hist_ptr[class_id] = bs_left_class_hist_[class_id];
        }
    }

    // regression version
    inline void update_left_child_imp(Float* left_imp_list_ptr,
                                      const hist_type_t* bs_left_hist,
                                      Index node_id) {
        Float* left_node_imp_ptr = left_imp_list_ptr + node_id * impl_const_t::node_imp_prop_count_;

        left_node_imp_ptr[0] = bs_left_hist[1];
        left_node_imp_ptr[1] = bs_left_hist[2];
    }

    inline void update_node_bs_info(const split_info_t& bs,
                                    Index* node_ptr,
                                    Float* node_imp_decr_ptr,
                                    Index node_id,
                                    Index index_max,
                                    bool update_imp_dec_required) {
        const split_scalar_t& bs_sc = bs.scalars;
        node_ptr[impl_const_t::ind_fid] =
            (bs_sc.ftr_id == index_max) ? impl_const_t::leaf_mark_ : bs_sc.ftr_id;
        node_ptr[impl_const_t::ind_bin] =
            (bs_sc.ftr_bin == index_max) ? impl_const_t::leaf_mark_ : bs_sc.ftr_bin;
        node_ptr[impl_const_t::ind_lch_grc] = bs_sc.left_count;

        if (update_imp_dec_required) {
            if constexpr (std::is_same_v<task_t, task::classification>) {
                node_imp_decr_ptr[node_id] = bs_sc.imp_dec;
            }
            else {
                node_imp_decr_ptr[node_id] = bs_sc.imp_dec / node_ptr[impl_const_t::ind_grc];
            }
        }
    }
};

} // namespace oneapi::dal::decision_forest::backend

#endif
