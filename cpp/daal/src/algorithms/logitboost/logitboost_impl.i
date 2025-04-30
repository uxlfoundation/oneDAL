/* file: logitboost_impl.i */
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
//  Functuons that are used in Logit Boost algorithm at training
//  and prediction stage
//--
*/

#ifndef __LOGITBOOST_IMPL_I__
#define __LOGITBOOST_IMPL_I__

#include <cmath>
#include "src/externals/service_math.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_utils.h"

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace internal
{
/**
 *  \brief Update additive function's F values.
 *         Step 2.b) of the Algorithm 6 from [1] (page 356).
 *
 *  \param dim[in]          Number of features
 *  \param n[in]            Number of observations
 *  \param nc[in]           Number of classes
 *  \param x[in]            Input dataset
 *  \param xbm[in]          Buffer manager associated with input dataset
 *  \param splitFeature[in] Indices of the split features
 *  \param splitPoint[in]   Split points
 *  \param lMean[in]        "left" average sof weighted responses
 *  \param rMean[in]        "right" averages of weighted responses
 *  \param F[out]           Additive function's values (column-major format:
 *                          values for the first sample come first,
 *                          for the second - second, etc)
 */
template <typename algorithmFPType, CpuType cpu>
void UpdateF(size_t dim, size_t n, size_t nc, const algorithmFPType * pred, algorithmFPType * F)
{
    const algorithmFPType inv_nc = 1.0 / (algorithmFPType)nc;
    const algorithmFPType coef   = (algorithmFPType)(nc - 1) / (algorithmFPType)nc;

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < nc; j++)
        {
            algorithmFPType rj = pred[j * n + i];
            algorithmFPType s  = rj;

            for (size_t k = 0; k < j; k++)
            {
                algorithmFPType r = pred[k * n + i];
                s += r;
            }

            for (size_t k = j + 1; k < nc; k++)
            {
                algorithmFPType r = pred[k * n + i];
                s += r;
            }

            F[i * nc + j] += coef * (rj - s * inv_nc);
        }
    }
}

/**
 *  \brief Update probailities matrix
 *
 *  \param nc[in]   Number of classes
 *  \param n[in]    Number of observations
 *  \param F[in]    Values of additive function
 *  \param P[out]   Probailities matrix of size nc x n
 */
template <typename algorithmFPType, CpuType cpu>
void UpdateP(size_t nc, size_t n, algorithmFPType * F, algorithmFPType * P, algorithmFPType * Fbuf)
{
    for (size_t i = 0; i < n; i++)
    {
        daal::internal::MathInst<algorithmFPType, cpu>::vExp(nc, F + i * nc, Fbuf);
        algorithmFPType s = 0.0;

        for (size_t j = 0; j < nc; j++)
        {
            // if low accuracy exp() returns NaN\Inf - convert it to some positive big value
            Fbuf[j] = services::internal::infToBigValue<cpu>(Fbuf[j]);
            s += Fbuf[j];
        }

        algorithmFPType invs = (algorithmFPType)1.0 / s;

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nc; j++)
        {
            // Normalize probabilities
            P[j * n + i] = Fbuf[j] * invs;
        }
    }
}

} // namespace internal
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
