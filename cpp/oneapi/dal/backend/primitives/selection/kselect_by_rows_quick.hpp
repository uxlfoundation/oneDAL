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

#include <limits>

#include <daal/src/externals/service_rng.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/src/algorithms/engines/engine_batch_impl.h>

#include "oneapi/dal/backend/primitives/selection/row_partitioning_kernel.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_base.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/rng/device_engine.hpp"

#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL
namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;
// Performs k-selection using Quick Select algorithm which is based on row partitioning
template <typename Float>
class kselect_by_rows_quick : public kselect_by_rows_base<Float> {
    static constexpr Float max_float = oneapi::dal::detail::limits<Float>::max();

    using sq_l2_dp_t = data_provider_t<Float, true>;
    using naive_dp_t = data_provider_t<Float, false>;

public:
    kselect_by_rows_quick() = delete;
    kselect_by_rows_quick(sycl::queue& queue, const ndshape<2>& shape) {
        data_ = ndarray<Float, 2>::empty(queue, shape, sycl::usm::alloc::device);
        indices_ = ndarray<std::int32_t, 2>::empty(queue, shape, sycl::usm::alloc::device);
        rnd_seq_ = ndarray<Float, 1>::empty(queue,
                                            std::min(shape[1], max_rnd_seq_size_),
                                            sycl::usm::alloc::shared);
        auto* rnd_seq_ptr = rnd_seq_.get_mutable_data();

        pr::device_engine engine_gpu = ::oneapi::dal::backend::primitives::device_engine(
            queue,
            777,
            ::oneapi::dal::backend::primitives::engine_type_internal::philox4x32x10);
        auto generation_event = pr::uniform<Float>(queue,
                                                   std::min(shape[1], max_rnd_seq_size_),
                                                   rnd_seq_ptr,
                                                   engine_gpu,
                                                   0,
                                                   1);
    }
    ~kselect_by_rows_quick() {
        last_call_.wait_and_throw();
    }
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<Float, 2>& selection,
                           ndview<std::int32_t, 2>& indices,
                           const event_vector& deps) override {
        const auto ht = data.get_dimension(0);
        const auto dp = naive_dp_t::make(data);
        return select<true, true>(queue, dp, k, ht, selection, indices, deps);
    }
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<Float, 2>& selection,
                           const event_vector& deps) override {
        ndarray<std::int32_t, 2> dummy;
        const auto ht = data.get_dimension(0);
        const auto dp = naive_dp_t::make(data);
        return select<true, false>(queue, dp, k, ht, selection, dummy, deps);
    }
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<std::int32_t, 2>& indices,
                           const event_vector& deps) override {
        ndarray<Float, 2> dummy;
        const auto ht = data.get_dimension(0);
        const auto dp = naive_dp_t::make(data);
        return select<false, true>(queue, dp, k, ht, dummy, indices, deps);
    }

    sycl::event select_sq_l2(sycl::queue& queue,
                             const ndview<Float, 1>& n1,
                             const ndview<Float, 1>& n2,
                             const ndview<Float, 2>& ip,
                             std::int64_t k,
                             ndview<Float, 2>& selection,
                             ndview<std::int32_t, 2>& indices,
                             const event_vector& deps) override {
        const auto ht = ip.get_dimension(0);
        const auto dp = sq_l2_dp_t::make(n1, n2, ip);
        return select<true, true>(queue, dp, k, ht, selection, indices, deps);
    }

    sycl::event select_sq_l2(sycl::queue& queue,
                             const ndview<Float, 1>& n1,
                             const ndview<Float, 1>& n2,
                             const ndview<Float, 2>& ip,
                             std::int64_t k,
                             ndview<Float, 2>& selection,
                             const event_vector& deps) override {
        ndarray<std::int32_t, 2> dummy;
        const auto ht = ip.get_dimension(0);
        const auto dp = sq_l2_dp_t::make(n1, n2, ip);
        return select<true, false>(queue, dp, k, ht, selection, dummy, deps);
    }

    sycl::event select_sq_l2(sycl::queue& queue,
                             const ndview<Float, 1>& n1,
                             const ndview<Float, 1>& n2,
                             const ndview<Float, 2>& ip,
                             std::int64_t k,
                             ndview<std::int32_t, 2>& indices,
                             const event_vector& deps) override {
        ndarray<Float, 2> dummy;
        const auto ht = ip.get_dimension(0);
        const auto dp = sq_l2_dp_t::make(n1, n2, ip);
        return select<false, true>(queue, dp, k, ht, dummy, indices, deps);
    }

private:
    template <bool selection_out, bool indices_out, typename DataProvider>
    sycl::event select(sycl::queue& queue,
                       const DataProvider& dp,
                       std::int64_t k,
                       std::int64_t height,
                       ndview<Float, 2>& selection,
                       ndview<std::int32_t, 2>& indices,
                       const event_vector& deps) {
        ONEDAL_PROFILER_TASK(selection.kselect_by_rows_quick, queue);

        last_call_.wait_and_throw();
        const std::int64_t row_count = height;
        const std::int64_t col_count = dp.get_width();
        [[maybe_unused]] const std::int64_t out_ids_stride = indices.get_leading_stride();
        [[maybe_unused]] const std::int64_t out_dst_stride = selection.get_leading_stride();

        ONEDAL_ASSERT(!indices_out || indices.get_shape()[0] == row_count);
        ONEDAL_ASSERT(!indices_out || indices.get_shape()[1] == k);
        ONEDAL_ASSERT(!selection_out || selection.get_shape()[0] == row_count);
        ONEDAL_ASSERT(!selection_out || selection.get_shape()[1] == k);

        auto* data_tmp_ptr = data_.get_mutable_data();
        const auto data_tmp_str = data_.get_leading_stride();
        auto cpy_event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            const auto range = make_range_2d(row_count, col_count);
            cgh.parallel_for(range, [=](sycl::id<2> idx) {
                *(data_tmp_ptr + idx[0] * data_tmp_str + idx[1]) = dp.at(idx[0], idx[1]);
            });
        });

        auto indices_tmp_ptr = indices_.get_mutable_data();

        auto rnd_period = this->rnd_seq_.get_count();
        auto rnd_seq_ptr = this->rnd_seq_.get_data();

        [[maybe_unused]] Float* selection_ptr =
            selection_out ? selection.get_mutable_data() : nullptr;
        [[maybe_unused]] std::int32_t* indices_ptr =
            indices_out ? indices.get_mutable_data() : nullptr;

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.depends_on(cpy_event);
            cgh.parallel_for(make_multiple_nd_range_2d({ preffered_sg_size, row_count },
                                                       { preffered_sg_size, 1 }),
                             [=](sycl::nd_item<2> item) {
                                 kselect_by_rows_quick::kernel_select<selection_out, indices_out>(
                                     item,
                                     row_count,
                                     data_tmp_ptr,
                                     indices_tmp_ptr,
                                     selection_ptr,
                                     indices_ptr,
                                     rnd_seq_ptr,
                                     rnd_period,
                                     col_count,
                                     k,
                                     data_tmp_str,
                                     out_ids_stride,
                                     out_dst_stride);
                             });
        });
        last_call_ = event;
        return event;
    }

    static Float kernel_get_rnd(const Float* rnd_seq,
                                std::int32_t rnd_period,
                                std::int32_t* count) {
        Float ret = rnd_seq[(*count)++];
        if (*count >= rnd_period)
            *count = 0;
        return ret;
    }

    static void finalize(sycl::nd_item<2> item,
                         Float* values,
                         std::int32_t* indices,
                         std::int32_t partition_start,
                         std::int32_t partition_end,
                         std::int32_t remainder) {
        auto sg = item.get_sub_group();
        const std::int32_t local_id = sg.get_local_id()[0];
        const std::int32_t local_size = sg.get_local_range()[0];

        constexpr std::int32_t undefined_index = -1;
        auto partition_size = partition_end - partition_start;
        bool is_used = local_id < partition_size;
        auto offset = partition_start + local_id;
        Float val = is_used ? values[offset] : max_float;
        std::int32_t ind = is_used ? indices[offset] : undefined_index;
        std::int32_t pos = undefined_index;
        for (std::int32_t step = 0; step < remainder; step++) {
            Float min_val = sycl::reduce_over_group(sg,
                                                    pos < 0 ? val : max_float,
                                                    sycl::ext::oneapi::minimum<Float>());
            bool is_mine = min_val == val && pos == undefined_index;
            std::int32_t min_id =
                sycl::reduce_over_group(sg,
                                        is_mine ? local_id : local_size,
                                        sycl::ext::oneapi::minimum<std::int32_t>());
            pos = min_id == local_id ? step : pos;
        }
        if (pos > undefined_index) {
            values[partition_start + pos] = val;
            indices[partition_start + pos] = ind;
        }
    }

    template <bool selection_out, bool indices_out>
    static void kernel_select(sycl::nd_item<2> item,
                              std::int32_t num_rows,
                              Float* in_values,
                              std::int32_t* in_indices,
                              Float* out_values,
                              std::int32_t* out_indices,
                              const Float* rnd_seq,
                              std::int32_t rnd_period,
                              std::int32_t row_count,
                              std::int32_t k,
                              std::int32_t inp_stride,
                              std::int32_t out_ids_stride,
                              std::int32_t out_dst_stride) {
        auto sg = item.get_sub_group();
        const std::int32_t row_id =
            item.get_global_id(1) * sg.get_group_range()[0] + sg.get_group_id()[0];
        const std::int32_t local_id = sg.get_local_id()[0];
        const std::int32_t local_size = sg.get_local_range()[0];
        if (row_id >= num_rows)
            return;

        const std::int32_t offset_in = row_id * inp_stride;
        [[maybe_unused]] const std::int32_t offset_ids_out = row_id * out_ids_stride;
        [[maybe_unused]] const std::int32_t offset_dst_out = row_id * out_dst_stride;
        std::int32_t partition_start = 0;
        std::int32_t partition_end = row_count;
        std::int32_t rnd_count = 0;

        Float* values = &in_values[offset_in];
        std::int32_t* indices = &in_indices[offset_in];
        if constexpr (indices_out) {
            for (std::int32_t i = partition_start + local_id; i < partition_end; i += local_size) {
                indices[i] = i;
            }
        }
        std::int32_t iteration_count;
        for (iteration_count = 0; iteration_count < row_count; iteration_count++) {
            const Float rnd =
                kselect_by_rows_quick::kernel_get_rnd(rnd_seq, rnd_period, &rnd_count);
            std::int32_t pos = (std::int32_t)(rnd * (partition_end - partition_start - 1));
            pos = pos < 0 ? 0 : pos;
            const Float pivot = values[partition_start + pos];
            auto partition_size = partition_end - partition_start;
            if (partition_size > local_size) {
                std::int32_t split_index = row_partitioning_kernel(item,
                                                                   values,
                                                                   indices,
                                                                   partition_start,
                                                                   partition_end,
                                                                   pivot);
                if ((split_index) == k)
                    break;
                if (split_index > k)
                    partition_end = split_index;
                if (split_index < k)
                    partition_start = split_index;
            }
            else {
                auto remainder = k - partition_start;
                finalize(item, values, indices, partition_start, partition_end, remainder);
                break;
            }
        }
        for (std::int32_t i = local_id; i < k; i += local_size) {
            if constexpr (indices_out) {
                out_indices[offset_ids_out + i] = indices[i];
            }
            if constexpr (selection_out) {
                out_values[offset_dst_out + i] = values[i];
            }
        }
    }
    static constexpr std::uint32_t preffered_sg_size = 16;
    static constexpr std::int64_t max_rnd_seq_size_ = 1024;
    std::int64_t rnd_seq_size_ = max_rnd_seq_size_;
    ndarray<Float, 1> rnd_seq_;
    ndarray<Float, 2> data_;
    ndarray<std::int32_t, 2> indices_;
    sycl::event last_call_;
};

#endif

} // namespace oneapi::dal::backend::primitives
