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

#include "oneapi/dal/backend/primitives/sparse_blas/test/fixture.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

using gemv_types = COMBINE_TYPES((float, double),
                                 (transpose_nontrans, transpose_trans),
                                 (c_order), /// not used in GEMV
                                 (c_order), /// not used in GEMV
                                 (indexing_zero_based, indexing_one_based));

TEMPLATE_LIST_TEST_M(sparse_blas_test, "ones matrix sparse CSR gemv", "[csr][gemv]", gemv_types) {
    // DPC++ Sparse GEMV from micro MKL libs is not supported on CPU
    SKIP_IF(this->get_policy().is_cpu());

    // Test takes too long time if HW emulates float64
    // Temporary workaround: skip tests on architectures that do not support native float64
    SKIP_IF(!this->get_policy().has_native_float64());

    this->generate_dimensions_gemv();
    this->test_gemv();
}

TEMPLATE_LIST_TEST_M(sparse_blas_test,
                     "ones matrix sparse CSR gemv with a cached plan",
                     "[csr][gemv]",
                     gemv_types) {
    // DPC++ Sparse GEMV from micro MKL libs is not supported on CPU
    SKIP_IF(this->get_policy().is_cpu());

    // Test takes too long time if HW emulates float64
    // Temporary workaround: skip tests on architectures that do not support native float64
    SKIP_IF(!this->get_policy().has_native_float64());

    this->generate_dimensions_gemv();
    this->test_gemv_optimized();
}

} // namespace oneapi::dal::backend::primitives::test
