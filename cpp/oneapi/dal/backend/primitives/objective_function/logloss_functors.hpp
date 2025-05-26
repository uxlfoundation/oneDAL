/*******************************************************************************
* Copyright contributors to the oneDAL project
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
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/optimizers/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/communicator.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/handle.hpp"

namespace oneapi::dal::backend::primitives {

using comm_t = backend::communicator<spmd::device_memory_access::usm>;

template <typename Float>
class logloss_hessian_product : public base_matrix_operator<Float> {
    friend dal::detail::pimpl_accessor;

public:
    logloss_hessian_product(sycl::queue& q,
                            const table& data,
                            Float L2 = Float(0),
                            bool fit_intercept = true,
                            std::int64_t bsz = -1);
    logloss_hessian_product(sycl::queue& q,
                            comm_t comm,
                            const table& data,
                            Float L2 = Float(0),
                            bool fit_intercept = true,
                            std::int64_t bsz = -1);
    sycl::event operator()(const ndview<Float, 1>& vec,
                           ndview<Float, 1>& out,
                           const event_vector& deps) final;
    ndview<Float, 1>& get_raw_hessian();

private:
    void reserve_memory();

    sycl::event compute_with_fit_intercept(const ndview<Float, 1>& vec,
                                           ndview<Float, 1>& out,
                                           const event_vector& deps);
    sycl::event compute_without_fit_intercept(const ndview<Float, 1>& vec,
                                              ndview<Float, 1>& out,
                                              const event_vector& deps);
    sycl::queue& q_;
    comm_t comm_;
    const table data_;
    dal::detail::pimpl<sparse_matrix_handle> sp_handle_;
    ndarray<Float, 1> raw_hessian_;
    ndarray<Float, 1> buffer_;
    ndarray<Float, 1> tmp_gpu_;
    const std::int64_t n_;
    const std::int64_t p_;
    Float L2_;
    bool fit_intercept_;
    const std::int64_t bsz_;
};

template <typename Float>
class logloss_function : public base_function<Float> {
    friend dal::detail::pimpl_accessor;

public:
    logloss_function(sycl::queue& queue,
                     const table& data,
                     const ndview<std::int32_t, 1>& labels,
                     Float L2 = 0.0,
                     bool fit_intercept = true,
                     std::int64_t bsz = -1);
    logloss_function(sycl::queue& queue,
                     comm_t comm,
                     const table& data,
                     const ndview<std::int32_t, 1>& labels,
                     Float L2 = 0.0,
                     bool fit_intercept = true,
                     std::int64_t bsz = -1);
    Float get_value() final;
    ndview<Float, 1>& get_gradient() final;
    base_matrix_operator<Float>& get_hessian_product() final;

    event_vector update_x(const ndview<Float, 1>& x,
                          bool need_grad = true,
                          bool need_hessp = false,
                          const event_vector& deps = {}) final;

private:
    void reserve_memory();

    sycl::queue& q_;
    comm_t comm_;
    const table data_;
    dal::detail::pimpl<sparse_matrix_handle> sp_handle_;
    const ndview<std::int32_t, 1> labels_;
    ndarray<Float, 1> probabilities_;
    ndarray<Float, 1> gradient_;
    ndarray<Float, 1> buffer_;
    const std::int64_t n_;
    const std::int64_t p_;
    Float L2_;
    bool fit_intercept_;
    const std::int64_t bsz_;
    const std::int64_t dimension_;
    Float value_;
    logloss_hessian_product<Float> hessp_;
};

} // namespace oneapi::dal::backend::primitives
