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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/misc.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/handle.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

/// Analyzes the sparsity pattern of the matrix A and caches an execution plan for
/// the subsequent `gemv` calls that use the same matrix handle and the same
/// `transpose_a`.
///
/// oneMKL's sparse BLAS is a two-stage API on purpose: the inspection stage is
/// separated from the execution stage so that the cost of examining the sparsity
/// pattern -- choosing a kernel, balancing the rows across work-groups, and for
/// the transposed case building the transposed pattern -- is paid once and reused.
/// Calling this is optional; without it every `gemv` falls back to a generic
/// kernel and redoes that analysis internally. It therefore pays off wherever one
/// handle is multiplied repeatedly, for example across solver iterations, and is
/// not worth it for a single product.
///
/// In practice it is the transposed product that benefits, because that is the one
/// a plain CSR kernel cannot parallelize over rows without atomics. Measured on a
/// 100000 x 200 matrix, planning the transposed direction sped it up by 1.2x when
/// the rows carry equal numbers of non-zeros and by 11x when they do not, whereas
/// planning the non-transposed direction made that product up to 1.6x *slower*.
/// Prefer planning only the operations that are both transposed and repeated.
///
/// The plan lives inside the matrix handle, so it must be rebuilt after the
/// handle's data is replaced by another `set_csr_data`.
///
/// @param queue        The SYCL* queue object.
/// @param transpose_a  The operation the plan is built for. A plan built for
///                     `transpose::nontrans` does not speed up a transposed
///                     `gemv`; call this once per operation that is used.
/// @param a            Handle to object containing sparse matrix A.
/// @param dependencies Events indicating availability of the matrix A for reading.
///
/// @return A SYCL* event that can be used to track the completion of asynchronous
///         events that were enqueued during the API call.
ONEDAL_EXPORT sycl::event optimize_gemv(sycl::queue& queue,
                                        transpose transpose_a,
                                        sparse_matrix_handle& a,
                                        const event_vector& dependencies = {});

/// Computes a sparse matrix - dense vector product:
///         y = alpha * op(A) + beta * y
/// where `alpha` and `beta` are scalars, A - sparse matrix and x, y - dense vectors.
/// op(A) is an operator defining if the matrix A used as is in the computations
/// or is being transposed.
///
/// op(A) is `m` x `k` matrix.
///
/// @tparam Float   The type of elements in the matrix A and the vectors x and y.
///                 The `Float` type should be at least `float` or `double`.
///
/// @param queue        The SYCL* queue object.
/// @param transpose_a  Defines if the sparse matrix A transposed or not.
///                     If `transpose_a` == `transpose::notrans` then op(A) = A.
///                     If `transpose_a` == `transpose::trans` then op(A) = transpose(A).
/// @param a            Handle to object containing sparse matrix A.
/// @param x            Dense 1-dimensional input vector that has `k` elements.
/// @param y            Dense 1-dimensional resulting vector that has `m` elements.
/// @param alpha        Specifies the scalar `alpha`.
/// @param beta         Specifies the scalar `beta`.
/// @param dependencies Events indicating availability of the matrix A and the vectors x and y
///                     for reading or writing.
template <typename Float>
sycl::event gemv(sycl::queue& queue,
                 transpose transpose_a,
                 sparse_matrix_handle& a,
                 const ndview<Float, 1>& x,
                 ndview<Float, 1>& y,
                 const Float alpha,
                 const Float beta,
                 const event_vector& dependencies = {});

/// Computes a sparse matrix - dense vector product:
///         y = op(A) * x
/// A - sparse matrix and x, y - dense vectors.
/// op(A) is an operator defining if the matrix A used as is in the computations
/// or is being transposed.
///
/// op(A) is `m` x `k` matrix.
///
/// @tparam Float   The type of elements in the matrix A and the vectors x and y.
///                 The `Float` type should be at least `float` or `double`.
///
/// @param queue        The SYCL* queue object.
/// @param transpose_a  Defines if the sparse matrix A transposed or not.
///                     If `transpose_a` == `transpose::notrans` then op(A) = A.
///                     If `transpose_a` == `transpose::trans` then op(A) = transpose(A).
/// @param a            Handle to object containing sparse matrix A.
/// @param x            Dense 1-dimensional input vector that has `k` elements.
/// @param y            Dense 1-dimensional resulting vector that has `m` elements.
/// @param dependencies Events indicating availability of the matrices A, B and C for reading
///                     or writing.
template <typename Float>
sycl::event gemv(sycl::queue& queue,
                 transpose transpose_a,
                 sparse_matrix_handle& a,
                 const ndview<Float, 1>& x,
                 ndview<Float, 1>& y,
                 const event_vector& dependencies = {}) {
    return gemv<Float>(queue, transpose_a, a, x, y, Float(1), Float(0), dependencies);
}

#endif // ifdef ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
