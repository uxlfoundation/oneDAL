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

#ifdef ONEDAL_DATA_PARALLEL

#include <oneapi/mkl.hpp>

namespace oneapi::dal::detail {

namespace v1 {

namespace mkl = oneapi::mkl;

/// Class that hides the implementation details of the `backend::primitives::sparse_matrix_handle` class
class sparse_matrix_handle_impl {
public:
    sparse_matrix_handle_impl(sycl::queue& queue);

    virtual ~sparse_matrix_handle_impl();

    sparse_matrix_handle_impl(sparse_matrix_handle_impl& other) = delete;

    sparse_matrix_handle_impl& operator=(sparse_matrix_handle_impl& other) = delete;

    inline mkl::sparse::matrix_handle_t& get() {
        return handle_;
    }
    inline const mkl::sparse::matrix_handle_t& get() const {
        return handle_;
    }

private:
    mkl::sparse::matrix_handle_t handle_;
    sycl::queue& queue_;
};

} // namespace v1

using v1::sparse_matrix_handle_impl;

} // namespace oneapi::dal::detail

#endif // ifdef ONEDAL_DATA_PARALLEL
