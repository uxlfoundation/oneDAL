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

/// Computes a sparse-dense matrix product:
///         C = alpha * op(A) * B + beta * C
/// where `alpha` and `beta` are scalars, A - sparse matrix and B, C - dense matrices.
/// op(A) is an operator defining if the matrix A used as is in the computations
/// or is being transposed.
///
/// op(A) is `m` x `k` matrix;
/// B is `k` x `p` matrix;
/// C is `m` x `p` matrix.
///
/// @tparam Float   The type of elements in the matrices A, B and C.
///                 The `Float` type should be at least `float` or `double`.
/// @tparam bo      Data layout in the matrix B.
///                 If `bo` == `ndorder::c` then the data in the matrix B is stored
///                 in row-major order (C-style).
///                 If `bo` == `ndorder::f` then the data in the matrix B is stored
///                 in column-major order (Fortran-style).
/// @tparam co      Data layout in the matrix C.
///                 The `co` value should be `ndorder::c` or `ndorder::f`.
///
/// @param queue        The SYCL* queue object.
/// @param transpose_a  Defines if the sparse matrix A transposed or not.
///                     If `transpose_a` == `transpose::notrans` then op(A) = A.
///                     If `transpose_a` == `transpose::trans` then op(A) = transpose(A).
/// @param a            Handle to object containing sparse matrix A.
/// @param b            Dense input matrix that has `k` rows and `p` columns.
/// @param c            Dense output matrix that has `m` rows and `p` columns.
/// @param alpha        Specifies the scalar `alpha`
/// @param beta         Specifies the scalar `beta`
/// @param dependencies Events indicating availability of the matrices A, B and C for reading
///                     or writing.
template <typename Float, ndorder bo, ndorder co>
sycl::event gemm(sycl::queue& queue,
                 transpose transpose_a,
                 sparse_matrix_handle& a,
                 const ndview<Float, 2, bo>& b,
                 ndview<Float, 2, co>& c,
                 const Float alpha,
                 const Float beta,
                 const event_vector& dependencies = {});

/// Computes a sparse-dense matrix product:
///         C = op(A) * B
/// where A - sparse matrix and B, C - dense matrices.
/// op(A) is an operator defining if the matrix A used as is in the computations
/// or is being transposed.
///
/// op(A) is `m` x `k` matrix;
/// B is `k` x `p` matrix;
/// C is `m` x `p` matrix.
///
/// @tparam Float   The type of elements in the matrices A, B and C.
///                 The `Float` type should be at least `float` or `double`.
/// @tparam bo      Data layout in the matrix B.
///                 If `bo` == `ndorder::c` then the data in the matrix B is stored
///                 in row-major order (C-style).
///                 If `bo` == `ndorder::f` then the data in the matrix B is stored
///                 in column-major order (Fortran-style).
/// @tparam co      Data layout in the matrix C.
///                 The `co` value should be `ndorder::c` or `ndorder::f`.
///
/// @param queue        The SYCL* queue object.
/// @param transpose_a  Defines if the sparse matrix A transposed or not.
///                     If `transpose_a` == `transpose::notrans` then op(A) = A.
///                     If `transpose_a` == `transpose::trans` then op(A) = transpose(A).
/// @param a            Handle to object containing sparse matrix A.
/// @param b            Dense input matrix that has `k` rows and `p` columns.
/// @param c            Dense output matrix that has `m` rows and `p` columns.
/// @param dependencies Events indicating availability of the matrices A, B and C for reading
///                     or writing.
template <typename Float, ndorder bo, ndorder co>
sycl::event gemm(sycl::queue& queue,
                 transpose transpose_a,
                 sparse_matrix_handle& a,
                 const ndview<Float, 2, bo>& b,
                 ndview<Float, 2, co>& c,
                 const event_vector& dependencies = {}) {
    return gemm<Float>(queue, transpose_a, a, b, c, Float(1), Float(0), dependencies);
}

/// Analyzes the sparse matrix A for the later calls of `gemm` with the same operation and
/// layouts, so that oneMKL can pick a faster algorithm. The handle keeps the result.
///
/// @tparam bo      Data layout of the dense matrix B in the later `gemm` calls.
/// @tparam co      Data layout of the dense matrix C in the later `gemm` calls.
///
/// @param queue        The SYCL* queue object.
/// @param transpose_a  The operation on A in the later `gemm` calls.
/// @param a            Handle to object containing sparse matrix A.
/// @param columns      Number of columns of B and C in the later `gemm` calls.
/// @param dependencies Events indicating availability of the matrix A.
template <ndorder bo, ndorder co>
sycl::event optimize_gemm(sycl::queue& queue,
                          transpose transpose_a,
                          sparse_matrix_handle& a,
                          std::int64_t columns,
                          const event_vector& dependencies = {});

#endif // ifdef ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
