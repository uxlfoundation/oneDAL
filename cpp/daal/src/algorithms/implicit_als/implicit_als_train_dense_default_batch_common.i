/* file: implicit_als_train_dense_default_batch_common.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/*
//++
//  Implementation of common computational kernels of impicit ALS training algorithm
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_DENSE_DEFAULT_BATCH_COMMON_I__
#define __IMPLICIT_ALS_TRAIN_DENSE_DEFAULT_BATCH_COMMON_I__

#include "src/externals/service_blas.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{
using namespace daal::internal;

template <typename algorithmFPType, CpuType cpu>
void computeXtX(size_t * nRows, size_t * nCols, algorithmFPType * beta, algorithmFPType * x, size_t * ldx, algorithmFPType * xtx, size_t * ldxtx)
{
    /* SYRK parameters */
    char uplo             = 'U';
    char trans            = 'N';
    algorithmFPType alpha = 1.0;

    daal::internal::BlasInst<algorithmFPType, cpu>::xsyrk(&uplo, &trans, (DAAL_INT *)nCols, (DAAL_INT *)nRows, &alpha, x, (DAAL_INT *)ldx, beta, xtx,
                                                          (DAAL_INT *)ldxtx);
}

} // namespace internal
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
