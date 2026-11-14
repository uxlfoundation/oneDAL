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

#include "oneapi/dal/detail/sparse_matrix_handle_impl.hpp"

namespace oneapi::dal::detail {

namespace v1 {

#ifdef ONEDAL_DATA_PARALLEL

#ifdef ONEDAL_MATH_BACKEND_ONEMATH

// oneMath has no empty-handle-then-fill flow: its `init_csr_matrix` takes the
// CSR arrays and produces a handle in one step, so there is nothing to create
// here and nothing to release. The handle stays null, and every sparse entry
// point that would use it throws -- see the `sparse_blas` primitives.
sparse_matrix_handle_impl::sparse_matrix_handle_impl(sycl::queue& queue)
        : handle_(nullptr),
          queue_(queue) {}

sparse_matrix_handle_impl::~sparse_matrix_handle_impl() = default;

#else

sparse_matrix_handle_impl::sparse_matrix_handle_impl(sycl::queue& queue) : queue_(queue) {
    mkl::sparse::init_matrix_handle(&handle_);
}

sparse_matrix_handle_impl::~sparse_matrix_handle_impl() {
    mkl::sparse::release_matrix_handle(queue_, &handle_, {}).wait();
}

#endif // ONEDAL_MATH_BACKEND_ONEMATH

#endif // ONEDAL_DATA_PARALLEL

} // namespace v1
} // namespace oneapi::dal::detail
