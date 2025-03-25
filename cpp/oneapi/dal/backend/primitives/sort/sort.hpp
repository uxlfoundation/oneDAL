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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
struct float2uint_map;

template <>
struct float2uint_map<float> {
    using integer_t = std::uint32_t;
};

template <>
struct float2uint_map<double> {
    using integer_t = std::uint64_t;
};

/// @tparam Float Floating-point type used for storing input values
/// @tparam Index Integer type used for storing input indices
template <typename Float, typename Index = std::uint32_t>
class radix_sort_indices_inplace {
    static_assert(std::is_same_v<float, Float> || std::is_same_v<double, Float>);
    using radix_integer_t = typename float2uint_map<Float>::integer_t;

public:
    /// Performs initialization of auxiliary variables and required auxiliary buffers
    ///
    /// @param[in]  queue The queue
    /// @param[in]  elem_count  The number of elements in input vector
    radix_sort_indices_inplace(const sycl::queue& queue);
    radix_sort_indices_inplace(const radix_sort_indices_inplace&) = delete;
    ~radix_sort_indices_inplace();
    radix_sort_indices_inplace& operator=(const radix_sort_indices_inplace&) = delete;

    /// Performs inplace radix sort of input vector and corresponding indices
    /// NOTE: auxiliary buffers and variables are reset in case if number of elements in val
    ///       differs from the number of elements provided in constructor
    ///
    /// @param[in, out]  val  The [n] input/output vector of values to sort out
    /// @param[in, out]  ind  The [n] input/output vector of corresponding indices
    sycl::event operator()(ndview<Float, 1>& val,
                           ndview<Index, 1>& ind,
                           const event_vector& deps = {});

private:
    void init(sycl::queue& queue, std::int64_t elem_count);
    sycl::event radix_scan(sycl::queue& queue,
                           const ndview<Float, 1>& val,
                           ndarray<Index, 1>& part_hist,
                           Index elem_count,
                           std::uint32_t bit_offset,
                           std::int64_t local_size,
                           std::int64_t local_hist_count,
                           sycl::event& deps);
    sycl::event radix_hist_scan(sycl::queue& queue,
                                const ndarray<Index, 1>& part_hist,
                                ndarray<Index, 1>& part_prefix_hist,
                                std::int64_t local_size,
                                std::int64_t local_hist_count,
                                sycl::event& deps);
    sycl::event radix_reorder(sycl::queue& queue,
                              const ndview<Float, 1>& val_in,
                              const ndview<Index, 1>& ind_in,
                              const ndview<Index, 1>& part_prefix_hist,
                              ndview<Float, 1>& val_out,
                              ndview<Index, 1>& ind_out,
                              Index elem_count,
                              std::uint32_t bit_offset,
                              std::int64_t local_size,
                              std::int64_t local_hist_count,
                              sycl::event& deps);

    sycl::queue queue_;
    sycl::event sort_event_;

    ndarray<Float, 1> val_buff_;
    ndarray<Index, 1> ind_buff_;

    ndarray<Index, 1> part_hist_;
    ndarray<Index, 1> part_prefix_hist_;

    std::uint32_t elem_count_;
    std::uint32_t local_size_;
    std::uint32_t local_hist_count_;
    std::uint32_t hist_buff_size_;

    static constexpr inline std::uint32_t radix_bits_ = 4;
    static constexpr inline std::uint32_t radix_range_ = (std::uint32_t)1 << radix_bits_;
    static constexpr inline std::uint32_t radix_range_1_ = radix_range_ - 1;

    static constexpr inline std::uint32_t byte_range_ = 8;
    static constexpr inline std::uint32_t max_local_hist_count_ = 1024;
    static constexpr inline std::uint32_t preferable_sbg_size_ = 16;
};

/// @tparam Integer Integer type used for storing input values
template <typename Integer>
class radix_sort {
public:
    /// Performs initialization of auxiliary variables and required auxiliary buffers
    ///
    /// @param[in]  queue The queue
    /// @param[in]  vector_count  The number of vectors (rows) in input array
    radix_sort(const sycl::queue& queue);
    radix_sort(const radix_sort&) = delete;
    ~radix_sort();
    radix_sort& operator=(const radix_sort&) = delete;

    /// Performs radix sort of batch of integer input vectors
    /// NOTE: only positive values are supported for now.
    ///       Auxiliary buffers and variables are reset in case if number of elements in val
    ///       differs from the number of elements provided in constructor
    ///
    /// @param[in]  val_in The [n x p] input array of vectors (row major format) to sort out,
    ///                    is also used for temporary data storage
    /// @param[out] val_out The [n x p] output array of sorted vectors (row major format)
    /// @param[in]  sorted_elem_count The number of elements to sort in each vector
    /// TODO: Extend interface with strided (not dense) input & output arrays
    sycl::event operator()(ndview<Integer, 2>& val_in,
                           ndview<Integer, 2>& val_out,
                           std::int64_t sorted_elem_count,
                           const event_vector& deps = {});

    sycl::event operator()(ndview<Integer, 2>& val_in,
                           ndview<Integer, 2>& val_out,
                           const event_vector& deps = {});

private:
    void init(sycl::queue& queue, std::int64_t vector_count);

    sycl::queue queue_;
    sycl::event sort_event_;

    ndarray<Integer, 2> buffer_;

    std::uint32_t vector_count_;

    static constexpr inline std::uint32_t preferable_wg_size_ = 32;
    static constexpr inline std::uint32_t radix_range_ = 256;
    static constexpr inline std::uint32_t radix_count_ = sizeof(Integer);
};

template <typename Float, typename Index = std::uint32_t>
sycl::event radix_sort_indices_inplace_dpl(sycl::queue& queue,
                                           ndview<Float, 1>& val,
                                           ndview<Index, 1>& ind,
                                           const event_vector& deps = {});

template <typename Integer>
sycl::event radix_sort_dpl(sycl::queue& queue,
                           ndview<Integer, 2>& val_in,
                           ndview<Integer, 2>& val_out,
                           std::int64_t sorted_elem_count,
                           const event_vector& deps = {});

template <typename Integer>
sycl::event radix_sort_dpl(sycl::queue& queue,
                           ndview<Integer, 2>& val_in,
                           ndview<Integer, 2>& val_out,
                           const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
