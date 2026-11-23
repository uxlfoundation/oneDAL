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

#include "oneapi/dal/test/engine/common.hpp"

// service_spblas.h relies on the headers that k-means includes before it.
#include "daal/src/data_management/service_numeric_table.h"
#include "daal/src/externals/service_spblas.h"

namespace oneapi::dal::kmeans::test {

using spblas_t = daal::internal::SpBlasInst<double, daal::internal::firstCpuType>;

// A 2 x 2 one-based CSR matrix [[1, 0], [2, 3]].
static const double values[] = { 1.0, 2.0, 3.0 };
static const DAAL_INT columns[] = { 1, 1, 2 };
static const DAAL_INT rows[] = { 1, 2, 4 };
static const char matdescra[6] = { 'G', 0, 0, 'F', 0, 0 };

TEST("DAAL sparse BLAS wrappers compute a valid CSR product", "[kmeans][spblas]") {
    const char trans = 'n';
    const DAAL_INT m = 2, n = 1, k = 2, ld = 2;
    const double alpha = 1.0, beta = 0.0;
    const double b[] = { 1.0, 1.0 };
    double c[] = { 0.0, 0.0 };
    const auto status = spblas_t::xxcsrmm(&trans,
                                          &m,
                                          &n,
                                          &k,
                                          &alpha,
                                          matdescra,
                                          values,
                                          columns,
                                          rows,
                                          b,
                                          &ld,
                                          &beta,
                                          c,
                                          &ld);
    REQUIRE(status.ok());
    REQUIRE(c[0] == 1.0);
    REQUIRE(c[1] == 5.0);
}

// The reference backend does not validate its arguments, so only MKL reports these.
TEST("DAAL sparse BLAS wrappers report an invalid CSR matrix", "[kmeans][spblas]") {
#ifdef DAAL_REF
    SKIP("the reference sparse BLAS does not validate its arguments");
#endif
    const char trans = 'n';
    const DAAL_INT bad_rows = -1, n = 1, k = 2, ld = 2;
    const double alpha = 1.0, beta = 0.0;
    const double b[] = { 1.0, 1.0 };
    double c[] = { 0.0, 0.0 };
    REQUIRE_FALSE(spblas_t::xxcsrmm(&trans,
                                    &bad_rows,
                                    &n,
                                    &k,
                                    &alpha,
                                    matdescra,
                                    values,
                                    columns,
                                    rows,
                                    b,
                                    &ld,
                                    &beta,
                                    c,
                                    &ld)
                      .ok());
    REQUIRE_FALSE(spblas_t::xcsrmv(&trans,
                                   &bad_rows,
                                   &k,
                                   &alpha,
                                   matdescra,
                                   values,
                                   columns,
                                   rows,
                                   rows + 1,
                                   b,
                                   &beta,
                                   c)
                      .ok());
    double cross[4] = {};
    DAAL_INT ldc = 2;
    const char t = 't';
    REQUIRE_FALSE(spblas_t::xcsrmultd(&t,
                                      &bad_rows,
                                      &k,
                                      &k,
                                      const_cast<double*>(values),
                                      const_cast<DAAL_INT*>(columns),
                                      const_cast<DAAL_INT*>(rows),
                                      const_cast<double*>(values),
                                      const_cast<DAAL_INT*>(columns),
                                      const_cast<DAAL_INT*>(rows),
                                      cross,
                                      &ldc)
                      .ok());
}

} // namespace oneapi::dal::kmeans::test
