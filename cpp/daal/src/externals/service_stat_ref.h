/* file: service_stat_ref.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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

/*
//++
//  Template wrappers for common statistic functions.
//--
*/

#ifndef __SERVICE_STAT_REF_H__
#define __SERVICE_STAT_REF_H__

#include "src/externals/service_memory.h"
#include "src/externals/service_blas_ref.h"

typedef void (*func_type)(DAAL_INT, DAAL_INT, DAAL_INT, void *);
extern "C"
{
#define __DAAL_VSL_SS_MATRIX_STORAGE_COLS           0x00020000
#define __DAAL_VSL_SS_MATRIX_STORAGE_FULL           0x00000000
#define __DAAL_VSL_SS_ED_WEIGHTS                    6
#define __DAAL_VSL_SS_ED_MIN                        16
#define __DAAL_VSL_SS_ED_MAX                        17
#define __DAAL_VSL_SS_ED_SUM                        67
#define __DAAL_VSL_SS_ED_MEAN                       7
#define __DAAL_VSL_SS_ED_2R_MOM                     8
#define __DAAL_VSL_SS_ED_2C_MOM                     11
#define __DAAL_VSL_SS_ED_2C_SUM                     71
#define __DAAL_VSL_SS_ED_VARIATION                  18
#define __DAAL_VSL_SS_ED_CP                         74
#define __DAAL_VSL_SS_ED_CP_STORAGE                 75
#define __DAAL_VSL_SS_CP                            0x0000000100000000
#define __DAAL_VSL_SS_METHOD_FAST                   0x00000001
#define __DAAL_VSL_SS_METHOD_1PASS                  0x00000002
#define __DAAL_VSL_SS_METHOD_FAST_USER_MEAN         0x00000100
#define __DAAL_VSL_SS_MIN                           0x0000000000000400
#define __DAAL_VSL_SS_MAX                           0x0000000000000800
#define __DAAL_VSL_SS_SUM                           0x0000000002000000
#define __DAAL_VSL_SS_MEAN                          0x0000000000000001
#define __DAAL_VSL_SS_2R_MOM                        0x0000000000000002
#define __DAAL_VSL_SS_2C_MOM                        0x0000000000000010
#define __DAAL_VSL_SS_2C_SUM                        0x0000000020000000
#define __DAAL_VSL_SS_VARIATION                     0x0000000000000200
#define __DAAL_VSL_SS_ED_ACCUM_WEIGHT               23
#define __DAAL_VSL_SS_METHOD_FAST_USER_MEAN         0x00000100
#define __DAAL_VSL_SS_METHOD_BACON_MEDIAN_INIT      0x00000002
#define __DAAL_VSL_SS_METHOD_BACON_MAHALANOBIS_INIT 0x00000001
#define __DAAL_VSL_SS_OUTLIERS                      0x0000000000080000
#define __DAAL_VSL_SS_METHOD_BACON                  0x00000020
#define __DAAL_VSL_SS_QUANTS                        0x0000000000010000
#define __DAAL_VSL_SS_ED_QUANT_ORDER_N              24
#define __DAAL_VSL_SS_ED_QUANT_ORDER                25
#define __DAAL_VSL_SS_ED_QUANT_QUANTILES            26
#define __DAAL_VSL_SS_SORTED_OBSERV                 0x0000008000000000
#define __DAAL_VSL_SS_ED_SORTED_OBSERV              78
#define __DAAL_VSL_SS_ED_SORTED_OBSERV_STORAGE      79
#define __DAAL_VSL_SS_METHOD_RADIX                  0x00100000

#define __DAAL_VSL_SS_ERROR_BAD_QUANT_ORDER       -4022
#define __DAAL_VSL_SS_ERROR_INDICES_NOT_SUPPORTED -4085
}

namespace daal
{
namespace internal
{
namespace ref
{
template <typename fpType, CpuType cpu>
struct RefStatistics
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct RefStatistics<double, cpu>
{
    typedef __int64 SizeType;
    typedef __int64 MethodType;
    typedef int ErrorType;

    static int xcp(double * data, __int64 nFeatures, __int64 nVectors, double * nPreviousObservations, double * sum, double * crossProduct,
                   __int64 method)
    {
        int errcode = 0;
        daal::internal::ref::OpenBlas<double, cpu> blasInst;
        const double accWtOld  = *nPreviousObservations;
        const double accWt     = *nPreviousObservations + nVectors;
        constexpr DAAL_INT one = 1;
        if (accWtOld != 0)
        {
            double * const sumOld = daal::services::internal::service_malloc<double, cpu>(nFeatures, sizeof(double));
            DAAL_CHECK_MALLOC(sumOld);
            for (DAAL_INT i = 0; i < nFeatures; ++i)
            {
                sumOld[i] = sum[i];
            }
            // S_old S_old^t/accWtOld
            const double alpha    = 1.0 / accWtOld;
            const double beta     = 1.0;
            constexpr char transa = 'N';
            constexpr char transb = 'N';
            blasInst.xgemm(&transa, &transb, &nFeatures, &nFeatures, &one, &alpha, sumOld, &nFeatures, sumOld, &one, &beta, crossProduct, &nFeatures);
            daal::services::daal_free(sumOld);
        }
        for (DAAL_INT i = 0; i < nVectors; ++i)
        {
            for (DAAL_INT j = 0; j < nFeatures; ++j) // if accWtOld = 0, overwrite sum
            {
                if (accWtOld != 0)
                {
                    sum[j] += data[i * nFeatures + j];
                }
                else
                {
                    if (i == 0)
                        sum[j] = data[i * nFeatures + j]; //overwrite the current sum
                    else
                        sum[j] += data[i * nFeatures + j];
                }
            }
        }

        // -S S^t/accWt
        {
            const double alpha    = -1.0 / accWt;
            const double beta     = accWtOld != 0 ? 1.0 : 0.0;
            constexpr char transa = 'N';
            constexpr char transb = 'N';
            blasInst.xgemm(&transa, &transb, &nFeatures, &nFeatures, &one, &alpha, sum, &nFeatures, sum, &one, &beta, crossProduct, &nFeatures);
        }

        // X X^t
        {
            constexpr double alpha = 1.0;
            constexpr double beta  = 1.0;
            constexpr char transa  = 'N';
            constexpr char transb  = 'T';
            blasInst.xgemm(&transa, &transb, &nFeatures, &nFeatures, &nVectors, &alpha, data, &nFeatures, data, &nFeatures, &beta, crossProduct,
                           &nFeatures);
        }

        return errcode;
    }

    static int xxcp_weight(double * data, __int64 nFeatures, __int64 nVectors, double * weight, double * accumWeight, double * mean,
                           double * crossProduct, __int64 method)
    {
        int errcode = 0;

        return errcode;
    }

    static int xxvar_weight(double * data, __int64 nFeatures, __int64 nVectors, double * weight, double * accumWeight, double * mean,
                            double * sampleVariance, __int64 method)
    {
        int errcode = 0;

        return errcode;
    }

    static int x2c_mom(const double * data, const __int64 nFeatures, const __int64 nVectors, double * variance, const __int64 method)
    {
        // E(x-\mu)^2 = E(x^2) - \mu^2
        int errcode  = 0;
        double * sum = (double *)daal::services::internal::service_calloc<double, cpu>(nFeatures, sizeof(double));
        DAAL_CHECK_MALLOC(sum);
        daal::services::internal::service_memset<double, cpu>(variance, double(0), nFeatures);
        DAAL_INT feature_ptr, vec_ptr;
        double wtInv      = (double)1 / nVectors;
        double wtInvMinus = (double)1 / (nVectors - 1);
        double pt         = 0;
        for (vec_ptr = 0; vec_ptr < nVectors; ++vec_ptr)
        {
#pragma omp simd
            for (feature_ptr = 0; feature_ptr < nFeatures; ++feature_ptr)
            {
                pt = data[vec_ptr * nFeatures + feature_ptr];
                sum[feature_ptr] += pt;
                variance[feature_ptr] += (pt * pt); // 2RSum
            }
        }
        double sumSqDivN; // S^2/n = n*\mu^2
#pragma omp simd
        for (feature_ptr = 0; feature_ptr < nFeatures; ++feature_ptr)
        {
            sumSqDivN = sum[feature_ptr];
            sumSqDivN *= sumSqDivN;
            sumSqDivN *= wtInv;
            variance[feature_ptr] -= sumSqDivN; // (2RSum-S^2/n)
            variance[feature_ptr] *= wtInvMinus;
        }
        daal::services::internal::service_free<double, cpu>(sum);
        sum = NULL;
        return errcode;
    }

    static int xoutlierdetection(const double * data, const __int64 nFeatures, const __int64 nVectors, const __int64 nParams,
                                 const double * baconParams, double * baconWeights)
    {
        int errcode = 0;

        return errcode;
    }

    static int xLowOrderMoments(double * data, __int64 nFeatures, __int64 nVectors, __int64 method, double * sum, double * mean,
                                double * secondOrderRawMoment, double * variance, double * variation)
    {
        int errcode = 0;

        return errcode;
    }

    static int xSumAndVariance(double * data, __int64 nFeatures, __int64 nVectors, double * nPreviousObservations, __int64 method, double * sum,
                               double * mean, double * secondOrderRawMoment, double * variance)
    {
        int errcode = 0;

        return errcode;
    }

    static int xQuantiles(const double * data, const __int64 nFeatures, const __int64 nVectors, const __int64 quantOrderN, const double * quantOrder,
                          double * quants)
    {
        int errcode = 0;

        return errcode;
    }

    static int xSort(double * data, __int64 nFeatures, __int64 nVectors, double * sortedData)
    {
        int errcode = 0;

        return errcode;
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct RefStatistics<float, cpu>
{
    typedef __int64 SizeType;
    typedef __int64 MethodType;
    typedef int ErrorType;

    static int xcp(float * data, __int64 nFeatures, __int64 nVectors, float * nPreviousObservations, float * sum, float * crossProduct,
                   __int64 method)
    {
        int errcode = 0;
        daal::internal::ref::OpenBlas<float, cpu> blasInst;
        const float accWtOld   = *nPreviousObservations;
        const float accWt      = *nPreviousObservations + nVectors;
        constexpr DAAL_INT one = 1;
        if (accWtOld != 0)
        {
            float * const sumOld = daal::services::internal::service_malloc<float, cpu>(nFeatures, sizeof(float));
            DAAL_CHECK_MALLOC(sumOld);
            for (DAAL_INT i = 0; i < nFeatures; ++i)
            {
                sumOld[i] = sum[i];
            }
            // S_old S_old^t/accWtOld
            const float alpha     = 1.0 / accWtOld;
            const float beta      = 1.0;
            constexpr char transa = 'N';
            constexpr char transb = 'N';
            blasInst.xgemm(&transa, &transb, &nFeatures, &nFeatures, &one, &alpha, sumOld, &nFeatures, sumOld, &one, &beta, crossProduct, &nFeatures);
            daal::services::daal_free(sumOld);
        }
        for (DAAL_INT i = 0; i < nVectors; ++i)
        {
            for (DAAL_INT j = 0; j < nFeatures; ++j) // if accWtOld = 0, overwrite sum
            {
                if (accWtOld != 0)
                {
                    sum[j] += data[i * nFeatures + j];
                }
                else
                {
                    if (i == 0)
                        sum[j] = data[i * nFeatures + j]; //overwrite the current sum
                    else
                        sum[j] += data[i * nFeatures + j];
                }
            }
        }

        // -S S^t/accWt
        {
            const float alpha     = -1.0 / accWt;
            const float beta      = accWtOld != 0 ? 1.0 : 0.0;
            constexpr char transa = 'N';
            constexpr char transb = 'N';
            blasInst.xgemm(&transa, &transb, &nFeatures, &nFeatures, &one, &alpha, sum, &nFeatures, sum, &one, &beta, crossProduct, &nFeatures);
        }

        // X X^t
        {
            constexpr float alpha = 1.0;
            constexpr float beta  = 1.0;
            constexpr char transa = 'N';
            constexpr char transb = 'T';
            blasInst.xgemm(&transa, &transb, &nFeatures, &nFeatures, &nVectors, &alpha, data, &nFeatures, data, &nFeatures, &beta, crossProduct,
                           &nFeatures);
        }

        return errcode;
    }

    static int xxcp_weight(float * data, __int64 nFeatures, __int64 nVectors, float * weight, float * accumWeight, float * mean, float * crossProduct,
                           __int64 method)
    {
        int errcode = 0;

        return errcode;
    }

    static int xxvar_weight(float * data, __int64 nFeatures, __int64 nVectors, float * weight, float * accumWeight, float * mean,
                            float * sampleVariance, __int64 method)
    {
        int errcode = 0;

        return errcode;
    }

    static int x2c_mom(const float * data, const __int64 nFeatures, const __int64 nVectors, float * variance, const __int64 method)
    {
        // E(x-\mu)^2 = E(x^2) - \mu^2
        int errcode = 0;
        float * sum = (float *)daal::services::internal::service_calloc<float, cpu>(nFeatures, sizeof(float));
        DAAL_CHECK_MALLOC(sum);
        daal::services::internal::service_memset<float, cpu>(variance, float(0), nFeatures);
        DAAL_INT feature_ptr, vec_ptr;
        float wtInv      = (float)1 / nVectors;
        float wtInvMinus = (float)1 / (nVectors - 1);
        float pt         = 0;
        for (vec_ptr = 0; vec_ptr < nVectors; ++vec_ptr)
        {
#pragma omp simd
            for (feature_ptr = 0; feature_ptr < nFeatures; ++feature_ptr)
            {
                pt = data[vec_ptr * nFeatures + feature_ptr];
                sum[feature_ptr] += pt;
                variance[feature_ptr] += (pt * pt); // 2RSum
            }
        }
        float sumSqDivN; // S^2/n = n*\mu^2
#pragma omp simd
        for (feature_ptr = 0; feature_ptr < nFeatures; ++feature_ptr)
        {
            sumSqDivN = sum[feature_ptr];
            sumSqDivN *= sumSqDivN;
            sumSqDivN *= wtInv;
            variance[feature_ptr] -= sumSqDivN; // (2RSum-S^2/n)
            variance[feature_ptr] *= wtInvMinus;
        }
        daal::services::internal::service_free<float, cpu>(sum);
        sum = NULL;
        return errcode;
    }

    static int xoutlierdetection(const float * data, const __int64 nFeatures, const __int64 nVectors, const __int64 nParams,
                                 const float * baconParams, float * baconWeights)
    {
        int errcode = 0;

        return errcode;
    }

    static int xLowOrderMoments(float * data, __int64 nFeatures, __int64 nVectors, __int64 method, float * sum, float * mean,
                                float * secondOrderRawMoment, float * variance, float * variation)
    {
        int errcode = 0;

        return errcode;
    }

    static int xSumAndVariance(float * data, __int64 nFeatures, __int64 nVectors, float * nPreviousObservations, __int64 method, float * sum,
                               float * mean, float * secondOrderRawMoment, float * variance)
    {
        int errcode = 0;

        return errcode;
    }

    static int xQuantiles(const float * data, const __int64 nFeatures, const __int64 nVectors, const __int64 quantOrderN, const float * quantOrder,
                          float * quants)
    {
        int errcode = 0;

        return errcode;
    }

    static int xSort(float * data, __int64 nFeatures, __int64 nVectors, float * sortedData)
    {
        int errcode = 0;

        return errcode;
    }
};

} // namespace ref
} // namespace internal
} // namespace daal

#endif
