/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event l1_norm(sycl::queue& queue,
                    const ndview<Float, 1>& x,
                    Float* res_gpu,
                    Float* res_host,
                    const event_vector& deps = {});

template <typename Float>
sycl::event dot_product(sycl::queue& queue,
                        const ndview<Float, 1>& x,
                        const ndview<Float, 1>& y,
                        Float* res_gpu,
                        Float* res_host,
                        const event_vector& deps = {});

template <typename Float>
sycl::event max_abs(sycl::queue& queue,
                    const ndview<Float, 1>& x,
                    Float* res_gpu,
                    Float* res_host,
                    const event_vector& deps = {});

template <typename Float>
class base_matrix_operator {
public:
    virtual ~base_matrix_operator() {}
    virtual sycl::event operator()(const ndview<Float, 1>& vec,
                                   ndview<Float, 1>& out,
                                   const event_vector& deps = {}) = 0;
};

template <typename Float>
class linear_matrix_operator : public base_matrix_operator<Float> {
public:
    linear_matrix_operator(sycl::queue& q, const ndview<Float, 2>& A);
    sycl::event operator()(const ndview<Float, 1>& vec,
                           ndview<Float, 1>& out,
                           const event_vector& deps) final;

private:
    sycl::queue q_;
    const ndview<Float, 2> A_;
};

template <typename Float>
class base_function {
public:
    virtual ~base_function() {}
    virtual Float get_value() = 0;
    virtual ndview<Float, 1>& get_gradient() = 0;
    virtual base_matrix_operator<Float>& get_hessian_product() = 0;
    virtual event_vector update_x(const ndview<Float, 1>& x,
                                  bool need_grad = true,
                                  bool need_hessp = false,
                                  const event_vector& deps = {}) = 0;
};

} // namespace oneapi::dal::backend::primitives
