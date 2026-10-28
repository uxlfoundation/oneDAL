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

#include <type_traits>

#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_base.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
class kselect_by_rows : public kselect_by_rows_base<Float> {
public:
    kselect_by_rows(sycl::queue& queue, const ndshape<2>& shape, std::int64_t k);
    /// Performs K-selection on each row of a matrix
    ///
    /// @param[in]  queue           The queue
    /// @param[in]  data            The [n x m] matrix to be processed
    /// @param[in]  k               The number of minimal values to be selected in each row
    /// @param[out] selection       The [n x k] matrix of selected values
    /// @param[out] column_indices  The [n x k] matrix of indices of selected values
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<Float, 2>& selection,
                           ndview<std::int32_t, 2>& column_indices,
                           const event_vector& deps = {}) override {
        return base_->operator()(queue, data, k, selection, column_indices, deps);
    }

    /// Performs K-selection on each row of a matrix
    ///
    /// @param[in]  queue       The queue
    /// @param[in]  data        The [n x m] matrix to be processed
    /// @param[in]  k           The number of minimal values to be selected in each row
    /// @param[out] selection   The [n x k] matrix of selected values
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<Float, 2>& selection,
                           const event_vector& deps = {}) override {
        return base_->operator()(queue, data, k, selection, deps);
    }

    /// Performs K-selection on each row of a matrix
    ///
    /// @param[in]  queue           The queue
    /// @param[in]  data            The [n x m] matrix to be processed
    /// @param[in]  k               The number of minimal values to be selected in each row
    /// @param[out] column_indices  The [n x k] matrix of indices of selected values
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<std::int32_t, 2>& column_indices,
                           const event_vector& deps = {}) override {
        return base_->operator()(queue, data, k, column_indices, deps);
    }

    sycl::event select_sq_l2(sycl::queue& queue,
                             const ndview<Float, 1>& n1,
                             const ndview<Float, 1>& n2,
                             const ndview<Float, 2>& ip,
                             std::int64_t k,
                             ndview<Float, 2>& selection,
                             ndview<std::int32_t, 2>& column_indices,
                             const event_vector& deps = {}) override {
        return base_->select_sq_l2(queue, n1, n2, ip, k, selection, column_indices, deps);
    }

    sycl::event select_sq_l2(sycl::queue& queue,
                             const ndview<Float, 1>& n1,
                             const ndview<Float, 1>& n2,
                             const ndview<Float, 2>& ip,
                             std::int64_t k,
                             ndview<Float, 2>& selection,
                             const event_vector& deps = {}) override {
        return base_->select_sq_l2(queue, n1, n2, ip, k, selection, deps);
    }

    sycl::event select_sq_l2(sycl::queue& queue,
                             const ndview<Float, 1>& n1,
                             const ndview<Float, 1>& n2,
                             const ndview<Float, 2>& ip,
                             std::int64_t k,
                             ndview<std::int32_t, 2>& column_indices,
                             const event_vector& deps = {}) override {
        return base_->select_sq_l2(queue, n1, n2, ip, k, column_indices, deps);
    }

private:
    detail::unique<kselect_by_rows_base<Float>> base_;
};

/// Device memory `kselect_by_rows<Float>` allocates at construction.
///
/// @tparam Float Floating-point type of the values being selected
///
/// @param[in] queue The queue the selector would be constructed on
/// @param[in] shape Shape of the matrix the selector would be given
/// @param[in] k     Number of smallest values selected per row
///
/// @return Device bytes the constructor allocates, saturated at the largest `std::int64_t`
template <typename Float>
std::int64_t kselect_by_rows_scratch_size(const sycl::queue& queue,
                                          const ndshape<2>& shape,
                                          std::int64_t k);

#endif

} // namespace oneapi::dal::backend::primitives
