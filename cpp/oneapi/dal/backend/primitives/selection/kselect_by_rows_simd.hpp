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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_base.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_data_provider.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

// Performs k-selection for small value of k that fits in Global Registry File width
template <typename Float, std::uint32_t simd_width>
class kselect_by_rows_simd : public kselect_by_rows_base<Float> {
    using sq_l2_dp_t = data_provider_t<Float, true>;
    using naive_dp_t = data_provider_t<Float, false>;

public:
    kselect_by_rows_simd() {}

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
                       const event_vector& deps = {}) {
        ONEDAL_PROFILER_TASK(selection.kselect_by_rows_simd, queue);

        const std::int64_t row_count = height;
        const std::int64_t col_count = dp.get_width();

        ONEDAL_ASSERT(!indices_out || indices.get_shape()[0] == row_count);
        ONEDAL_ASSERT(!indices_out || indices.get_shape()[1] == k);
        ONEDAL_ASSERT(!selection_out || selection.get_shape()[0] == row_count);
        ONEDAL_ASSERT(!selection_out || selection.get_shape()[1] == k);

        [[maybe_unused]] const std::int64_t out_ids_stride = indices.get_leading_stride();
        [[maybe_unused]] const std::int64_t out_dst_stride = selection.get_leading_stride();

        const std::int64_t wg_size =
            get_scaled_wg_size_per_row(queue, col_count, preffered_wg_size);

        [[maybe_unused]] Float* selection_ptr =
            selection_out ? selection.get_mutable_data() : nullptr;
        [[maybe_unused]] std::int32_t* indices_ptr =
            indices_out ? indices.get_mutable_data() : nullptr;
        const auto fp_max = detail::limits<Float>::max();

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(make_multiple_nd_range_2d({ wg_size, row_count }, { wg_size, 1 }),
                             [=](sycl::nd_item<2> item) {
                                 auto sg = item.get_sub_group();
                                 const std::uint32_t sg_id = sg.get_group_id()[0];
                                 const std::uint32_t wg_id = item.get_global_id(1);
                                 const std::uint32_t sg_num = sg.get_group_range()[0];

                                 const std::uint32_t sg_global_id = wg_id * sg_num + sg_id;
                                 if (sg_global_id >= row_count)
                                     return;

                                 [[maybe_unused]] const std::int32_t offset_ids_out =
                                     sg_global_id * out_ids_stride;
                                 [[maybe_unused]] const std::int32_t offset_dst_out =
                                     sg_global_id * out_dst_stride;

                                 const std::uint32_t local_id = sg.get_local_id()[0];
                                 const std::uint32_t local_range = sg.get_local_range()[0];

                                 Float values[simd_width];
                                 std::int32_t private_indices[simd_width];

                                 for (std::uint32_t i = 0; i < simd_width; i++) {
                                     values[i] = fp_max;
                                     private_indices[i] = -1;
                                 }
                                 for (std::uint32_t i = local_id; i < col_count; i += local_range) {
                                     const Float cur_val = dp.at(sg_global_id, i);
                                     std::int32_t index = i;
                                     std::int32_t pos = -1;

                                     pos = values[k - 1] > cur_val ? k - 1 : pos;
                                     for (std::int32_t j = k - 2; j >= 0; j--) {
                                         const bool do_shift = values[j] > cur_val;
                                         pos = do_shift ? j : pos;
                                         values[j + 1] = do_shift ? values[j] : values[j + 1];
                                         private_indices[j + 1] =
                                             do_shift ? private_indices[j] : private_indices[j + 1];
                                     }

                                     if (pos != -1) {
                                         values[pos] = cur_val;
                                         private_indices[pos] = index;
                                     }
                                 }
                                 sycl::group_barrier(sg);

                                 std::int32_t bias = 0;
                                 Float final_values[simd_width];
                                 std::int32_t final_indices[simd_width];
                                 for (std::uint32_t i = 0; i < k; i++) {
                                     const Float min_val = sycl::reduce_over_group(
                                         sg,
                                         values[bias],
                                         sycl::minimum<Float>());
                                     const bool present = (min_val == values[bias]);
                                     const std::int32_t pos = sycl::exclusive_scan_over_group(
                                         sg,
                                         present ? 1 : 0,
                                         sycl::plus<std::int32_t>());
                                     const bool owner = present && pos == 0;
                                     final_indices[i] = -sycl::reduce_over_group(
                                         sg,
                                         owner ? -private_indices[bias] : 1,
                                         sycl::minimum<std::int32_t>());
                                     final_values[i] = min_val;
                                     bias += owner ? 1 : 0;
                                 }
                                 if constexpr (indices_out) {
                                     for (std::uint32_t i = local_id; i < k; i += local_range) {
                                         indices_ptr[offset_ids_out + i] = final_indices[i];
                                     }
                                 }
                                 if constexpr (selection_out) {
                                     for (std::uint32_t i = local_id; i < k; i += local_range) {
                                         selection_ptr[offset_dst_out + i] = final_values[i];
                                     }
                                 }
                             });
        });
        return event;
    }
    static constexpr std::int64_t preffered_wg_size = 128;
};
#endif

} // namespace oneapi::dal::backend::primitives
