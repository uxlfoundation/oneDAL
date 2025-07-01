/* file: service_stat_mkl.h */
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
//  Template wrappers for common Intel(R) MKL functions.
//--
*/

#ifndef __SERVICE_STAT_MKL_H__
#define __SERVICE_STAT_MKL_H__

#include <mkl.h>
#include <mkl_vsl_functions.h>
#include "src/externals/service_memory.h"
#include "src/externals/service_stat_rng_mkl.h"

typedef void (*func_type)(DAAL_INT, DAAL_INT, DAAL_INT, void *);

#undef __DAAL_VSLFN_CALL
#define __DAAL_VSLFN_CALL(f_name, f_args, errcode) errcode = f_name f_args;

#if defined(_WIN64) || defined(__x86_64__)
    #define __SS_ILP_FLAG__ 1
#else
    #define __SS_ILP_FLAG__ 0
#endif

extern "C"
{
#define __DAAL_VSL_SS_MATRIX_STORAGE_COLS           VSL_SS_MATRIX_STORAGE_COLS
#define __DAAL_VSL_SS_MATRIX_STORAGE_FULL           VSL_SS_MATRIX_STORAGE_FULL
#define __DAAL_VSL_SS_ED_WEIGHTS                    VSL_SS_ED_WEIGHTS
#define __DAAL_VSL_SS_ED_MIN                        VSL_SS_ED_MIN
#define __DAAL_VSL_SS_ED_MAX                        VSL_SS_ED_MAX
#define __DAAL_VSL_SS_ED_SUM                        VSL_SS_ED_SUM
#define __DAAL_VSL_SS_ED_MEAN                       VSL_SS_ED_MEAN
#define __DAAL_VSL_SS_ED_2R_MOM                     VSL_SS_ED_2R_MOM
#define __DAAL_VSL_SS_ED_2C_MOM                     VSL_SS_ED_2C_MOM
#define __DAAL_VSL_SS_ED_2C_SUM                     VSL_SS_ED_2C_SUM
#define __DAAL_VSL_SS_ED_VARIATION                  VSL_SS_ED_VARIATION
#define __DAAL_VSL_SS_ED_CP                         VSL_SS_ED_CP
#define __DAAL_VSL_SS_ED_CP_STORAGE                 VSL_SS_ED_CP_STORAGE
#define __DAAL_VSL_SS_CP                            VSL_SS_CP
#define __DAAL_VSL_SS_METHOD_FAST                   VSL_SS_METHOD_FAST
#define __DAAL_VSL_SS_METHOD_1PASS                  VSL_SS_METHOD_1PASS
#define __DAAL_VSL_SS_METHOD_FAST_USER_MEAN         VSL_SS_METHOD_FAST_USER_MEAN
#define __DAAL_VSL_SS_MIN                           VSL_SS_MIN
#define __DAAL_VSL_SS_MAX                           VSL_SS_MAX
#define __DAAL_VSL_SS_SUM                           VSL_SS_SUM
#define __DAAL_VSL_SS_MEAN                          VSL_SS_MEAN
#define __DAAL_VSL_SS_2R_MOM                        VSL_SS_2R_MOM
#define __DAAL_VSL_SS_2C_MOM                        VSL_SS_2C_MOM
#define __DAAL_VSL_SS_2C_SUM                        VSL_SS_2C_SUM
#define __DAAL_VSL_SS_VARIATION                     VSL_SS_VARIATION
#define __DAAL_VSL_SS_ED_ACCUM_WEIGHT               VSL_SS_ED_ACCUM_WEIGHT
#define __DAAL_VSL_SS_METHOD_BACON_MEDIAN_INIT      VSL_SS_METHOD_BACON_MEDIAN_INIT
#define __DAAL_VSL_SS_METHOD_BACON_MAHALANOBIS_INIT VSL_SS_METHOD_BACON_MAHALANOBIS_INIT
#define __DAAL_VSL_SS_OUTLIERS                      VSL_SS_OUTLIERS
#define __DAAL_VSL_SS_METHOD_BACON                  VSL_SS_METHOD_BACON
#define __DAAL_VSL_SS_QUANTS                        VSL_SS_QUANTS
#define __DAAL_VSL_SS_ED_QUANT_ORDER_N              VSL_SS_ED_QUANT_ORDER_N
#define __DAAL_VSL_SS_ED_QUANT_ORDER                VSL_SS_ED_QUANT_ORDER
#define __DAAL_VSL_SS_ED_QUANT_QUANTILES            VSL_SS_ED_QUANT_QUANTILES
#define __DAAL_VSL_SS_SORTED_OBSERV                 VSL_SS_SORTED_OBSERV
#define __DAAL_VSL_SS_ED_SORTED_OBSERV              VSL_SS_ED_SORTED_OBSERV
#define __DAAL_VSL_SS_ED_SORTED_OBSERV_STORAGE      VSL_SS_ED_SORTED_OBSERV_STORAGE
#define __DAAL_VSL_SS_METHOD_RADIX                  VSL_SS_METHOD_RADIX

#define __DAAL_VSL_SS_ERROR_BAD_QUANT_ORDER       VSL_SS_ERROR_BAD_QUANT_ORDER
#define __DAAL_VSL_SS_ERROR_INDICES_NOT_SUPPORTED VSL_SS_ERROR_INDICES_NOT_SUPPORTED
}

namespace daal
{
namespace internal
{
namespace mkl
{
template <typename fpType, CpuType cpu>
struct MklStatistics
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct MklStatistics<double, cpu>
{
    typedef __int64 SizeType;
    typedef __int64 MethodType;
    typedef int ErrorType;

    static int xmeansOnePass(const double * data, __int64 nFeatures, __int64 nVectors, double * means)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __DAAL_VSLFN_CALL(vsldSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, means), errcode);
        __DAAL_VSLFN_CALL(vsldSSCompute, (task, __DAAL_VSL_SS_MEAN, VSL_SS_METHOD_1PASS), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);

        return errcode;
    }

    static int xcp(double * data, __int64 nFeatures, __int64 nVectors, double * nPreviousObservations, double * sum, double * crossProduct,
                   __int64 method)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __int64 cpStorage   = __DAAL_VSL_SS_MATRIX_STORAGE_FULL;

        double * mean = (double *)daal::services::daal_malloc(nFeatures * sizeof(double));

        if (method == __DAAL_VSL_SS_METHOD_FAST_USER_MEAN)
        {
            double invNVectors = 1.0 / (double)nVectors;
            for (size_t i = 0; i < nFeatures; i++)
            {
                mean[i] = sum[i] * invNVectors;
            }
        }

        double weight[2] = { *nPreviousObservations, *nPreviousObservations };

        __DAAL_VSLFN_CALL(vsldSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_CP, crossProduct), errcode);
        __DAAL_VSLFN_CALL(vsliSSEditTask, (task, __DAAL_VSL_SS_ED_CP_STORAGE, (const MKL_INT *)&cpStorage), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, weight), errcode);
        __DAAL_VSLFN_CALL(vsldSSCompute, (task, __DAAL_VSL_SS_CP | __DAAL_VSL_SS_SUM, method), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);

        daal::services::daal_free(mean);
        mean = NULL;
        return errcode;
    }

    static int xxcp_weight(double * data, __int64 nFeatures, __int64 nVectors, double * weight, double * accumWeight, double * mean,
                           double * crossProduct, __int64 method)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __int64 cpStorage   = __DAAL_VSL_SS_MATRIX_STORAGE_FULL;

        double * sum = (double *)daal::services::daal_malloc(nFeatures * sizeof(double));

        double accumWeightsAll[2] = { 0, 0 };

        __DAAL_VSLFN_CALL(vsldSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_WEIGHTS, weight), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_CP, crossProduct), errcode);
        __DAAL_VSLFN_CALL(vsliSSEditTask, (task, __DAAL_VSL_SS_ED_CP_STORAGE, (const MKL_INT *)&cpStorage), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, accumWeightsAll), errcode);
        __DAAL_VSLFN_CALL(vsldSSCompute, (task, __DAAL_VSL_SS_CP | __DAAL_VSL_SS_MEAN, method), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);

        *accumWeight = accumWeightsAll[0];

        daal::services::daal_free(sum);
        sum = NULL;
        return errcode;
    }

    static int xxvar_weight(double * data, __int64 nFeatures, __int64 nVectors, double * weight, double * accumWeight, double * mean,
                            double * sampleVariance, __int64 method)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        double * sum       = daal::services::internal::service_scalable_malloc<double, cpu>(nFeatures);
        double * rawSecond = daal::services::internal::service_scalable_malloc<double, cpu>(nFeatures);

        double accumWeightsAll[2] = { 0, 0 };

        __DAAL_VSLFN_CALL(vsldSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_WEIGHTS, weight), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2C_SUM, sampleVariance), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM, rawSecond), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, accumWeightsAll), errcode);
        __DAAL_VSLFN_CALL(vsldSSCompute, (task, __DAAL_VSL_SS_2C_SUM | __DAAL_VSL_SS_MEAN, method), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);

        *accumWeight = accumWeightsAll[0];

        daal::services::internal::service_scalable_free<double, cpu>(sum);
        daal::services::internal::service_scalable_free<double, cpu>(rawSecond);

        return errcode;
    }

    static int x2c_mom(const double * data, const __int64 nFeatures, const __int64 nVectors, double * variance, const __int64 method)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        double * mean                 = (double *)daal::services::daal_malloc(nFeatures * sizeof(double));
        double * secondOrderRawMoment = (double *)daal::services::daal_malloc(nFeatures * sizeof(double));

        __DAAL_VSLFN_CALL(vsldSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM, secondOrderRawMoment), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2C_MOM, variance), errcode);
        __DAAL_VSLFN_CALL(vsldSSCompute, (task, __DAAL_VSL_SS_2C_MOM, method), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);

        daal::services::daal_free(mean);
        daal::services::daal_free(secondOrderRawMoment);
        mean                 = NULL;
        secondOrderRawMoment = NULL;
        return errcode;
    }

    static int xoutlierdetection(const double * data, const __int64 nFeatures, const __int64 nVectors, const __int64 nParams,
                                 const double * baconParams, double * baconWeights)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(vsldSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        __DAAL_VSLFN_CALL(vsldSSEditOutliersDetection, (task, (const MKL_INT *)&nParams, baconParams, baconWeights), errcode);
        __DAAL_VSLFN_CALL(vsldSSCompute, (task, __DAAL_VSL_SS_OUTLIERS, __DAAL_VSL_SS_METHOD_BACON), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xLowOrderMoments(double * data, __int64 nFeatures, __int64 nVectors, __int64 method, double * sum, double * mean,
                                double * secondOrderRawMoment, double * variance, double * variation)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(vsldSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);

        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM, secondOrderRawMoment), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2C_MOM, variance), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_VARIATION, variation), errcode);
        __DAAL_VSLFN_CALL(
            vsldSSCompute,
            (task, __DAAL_VSL_SS_SUM | __DAAL_VSL_SS_MEAN | __DAAL_VSL_SS_2R_MOM | __DAAL_VSL_SS_2C_MOM | __DAAL_VSL_SS_VARIATION, method), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xSumAndVariance(double * data, __int64 nFeatures, __int64 nVectors, double * nPreviousObservations, __int64 method, double * sum,
                               double * mean, double * secondOrderRawMoment, double * variance)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(vsldSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);

        double weight[2] = { *nPreviousObservations, *nPreviousObservations };

        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM, secondOrderRawMoment), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2C_MOM, variance), errcode);
        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, weight), errcode);
        __DAAL_VSLFN_CALL(vsldSSCompute, (task, __DAAL_VSL_SS_SUM | __DAAL_VSL_SS_MEAN | __DAAL_VSL_SS_2R_MOM | __DAAL_VSL_SS_2C_MOM, method),
                          errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xQuantiles(const double * data, const __int64 nFeatures, const __int64 nVectors, const __int64 quantOrderN, const double * quantOrder,
                          double * quants)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(vsldSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        if (errcode)
        {
            return errcode;
        }

        __DAAL_VSLFN_CALL(vsliSSEditTask, (task, __DAAL_VSL_SS_ED_QUANT_ORDER_N, (const MKL_INT *)&quantOrderN), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_QUANT_ORDER, quantOrder), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_QUANT_QUANTILES, quants), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }
        __DAAL_VSLFN_CALL(vsldSSCompute, (task, __DAAL_VSL_SS_QUANTS, __DAAL_VSL_SS_METHOD_FAST), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xSort(double * data, __int64 nFeatures, __int64 nVectors, double * sortedData)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 inputStorage  = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __int64 outputStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(vsldSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&inputStorage, data, 0, 0),
                          errcode);
        if (errcode)
        {
            return errcode;
        }

        __DAAL_VSLFN_CALL(vsldSSEditTask, (task, __DAAL_VSL_SS_ED_SORTED_OBSERV, sortedData), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vsliSSEditTask, (task, __DAAL_VSL_SS_ED_SORTED_OBSERV_STORAGE, (const MKL_INT *)&outputStorage), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vsldSSCompute, (task, __DAAL_VSL_SS_SORTED_OBSERV, __DAAL_VSL_SS_METHOD_RADIX), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
        return errcode;
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct MklStatistics<float, cpu>
{
    typedef __int64 SizeType;
    typedef __int64 MethodType;
    typedef int ErrorType;

    static int xmeansOnePass(const float * data, __int64 nFeatures, __int64 nVectors, float * means)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __DAAL_VSLFN_CALL(vslsSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, means), errcode);
        __DAAL_VSLFN_CALL(vslsSSCompute, (task, __DAAL_VSL_SS_MEAN, VSL_SS_METHOD_1PASS), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);

        return errcode;
    }

    static int xcp(float * data, __int64 nFeatures, __int64 nVectors, float * nPreviousObservations, float * sum, float * crossProduct,
                   __int64 method)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __int64 cpStorage   = __DAAL_VSL_SS_MATRIX_STORAGE_FULL;

        float * mean = (float *)daal::services::daal_malloc(nFeatures * sizeof(float));

        if (method == __DAAL_VSL_SS_METHOD_FAST_USER_MEAN)
        {
            float invNVectors = 1.0 / (float)nVectors;
            for (size_t i = 0; i < nFeatures; i++)
            {
                mean[i] = sum[i] * invNVectors;
            }
        }

        float weight[2] = { *nPreviousObservations, *nPreviousObservations };

        __DAAL_VSLFN_CALL(vslsSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_CP, crossProduct), errcode);
        __DAAL_VSLFN_CALL(vsliSSEditTask, (task, __DAAL_VSL_SS_ED_CP_STORAGE, (const MKL_INT *)&cpStorage), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, weight), errcode);
        __DAAL_VSLFN_CALL(vslsSSCompute, (task, __DAAL_VSL_SS_CP | __DAAL_VSL_SS_SUM, method), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);

        daal::services::daal_free(mean);
        mean = NULL;
        return errcode;
    }

    static int xxcp_weight(float * data, __int64 nFeatures, __int64 nVectors, float * weight, float * accumWeight, float * mean, float * crossProduct,
                           __int64 method)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __int64 cpStorage   = __DAAL_VSL_SS_MATRIX_STORAGE_FULL;

        float * sum = (float *)daal::services::daal_malloc(nFeatures * sizeof(float));

        float accumWeightsAll[2] = { 0, 0 };

        __DAAL_VSLFN_CALL(vslsSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_WEIGHTS, weight), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_CP, crossProduct), errcode);
        __DAAL_VSLFN_CALL(vsliSSEditTask, (task, __DAAL_VSL_SS_ED_CP_STORAGE, (const MKL_INT *)&cpStorage), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, accumWeightsAll), errcode);
        __DAAL_VSLFN_CALL(vslsSSCompute, (task, __DAAL_VSL_SS_CP | __DAAL_VSL_SS_MEAN, method), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);

        *accumWeight = accumWeightsAll[0];

        daal::services::daal_free(sum);
        sum = NULL;
        return errcode;
    }

    static int xxvar_weight(float * data, __int64 nFeatures, __int64 nVectors, float * weight, float * accumWeight, float * mean,
                            float * sampleVariance, __int64 method)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        float * sum       = daal::services::internal::service_scalable_malloc<float, cpu>(nFeatures);
        float * rawSecond = daal::services::internal::service_scalable_malloc<float, cpu>(nFeatures);

        float accumWeightsAll[2] = { 0, 0 };

        __DAAL_VSLFN_CALL(vslsSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_WEIGHTS, weight), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2C_SUM, sampleVariance), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM, rawSecond), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, accumWeightsAll), errcode);
        __DAAL_VSLFN_CALL(vslsSSCompute, (task, __DAAL_VSL_SS_2C_SUM | __DAAL_VSL_SS_MEAN, method), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);

        *accumWeight = accumWeightsAll[0];

        daal::services::internal::service_scalable_free<float, cpu>(sum);
        daal::services::internal::service_scalable_free<float, cpu>(rawSecond);
        return errcode;
    }

    static int x2c_mom(const float * data, const __int64 nFeatures, const __int64 nVectors, float * variance, const __int64 method)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        float * mean                 = (float *)daal::services::daal_malloc(nFeatures * sizeof(float));
        float * secondOrderRawMoment = (float *)daal::services::daal_malloc(nFeatures * sizeof(float));

        __DAAL_VSLFN_CALL(vslsSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM, secondOrderRawMoment), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2C_MOM, variance), errcode);
        __DAAL_VSLFN_CALL(vslsSSCompute, (task, __DAAL_VSL_SS_2C_MOM, method), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);

        daal::services::daal_free(mean);
        daal::services::daal_free(secondOrderRawMoment);
        mean                 = NULL;
        secondOrderRawMoment = NULL;

        return errcode;
    }

    static int xoutlierdetection(const float * data, const __int64 nFeatures, const __int64 nVectors, const __int64 nParams,
                                 const float * baconParams, float * baconWeights)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(vslsSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        __DAAL_VSLFN_CALL(vslsSSEditOutliersDetection, (task, (const MKL_INT *)&nParams, baconParams, baconWeights), errcode);
        __DAAL_VSLFN_CALL(vslsSSCompute, (task, __DAAL_VSL_SS_OUTLIERS, __DAAL_VSL_SS_METHOD_BACON), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xLowOrderMoments(float * data, __int64 nFeatures, __int64 nVectors, __int64 method, float * sum, float * mean,
                                float * secondOrderRawMoment, float * variance, float * variation)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(vslsSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);

        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM, secondOrderRawMoment), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2C_MOM, variance), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_VARIATION, variation), errcode);
        __DAAL_VSLFN_CALL(
            vslsSSCompute,
            (task, __DAAL_VSL_SS_SUM | __DAAL_VSL_SS_MEAN | __DAAL_VSL_SS_2R_MOM | __DAAL_VSL_SS_2C_MOM | __DAAL_VSL_SS_VARIATION, method), errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xSumAndVariance(float * data, __int64 nFeatures, __int64 nVectors, float * nPreviousObservations, __int64 method, float * sum,
                               float * mean, float * secondOrderRawMoment, float * variance)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(vslsSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);

        float weight[2] = { *nPreviousObservations, *nPreviousObservations };

        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM, secondOrderRawMoment), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2C_MOM, variance), errcode);
        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, weight), errcode);
        __DAAL_VSLFN_CALL(vslsSSCompute, (task, __DAAL_VSL_SS_SUM | __DAAL_VSL_SS_MEAN | __DAAL_VSL_SS_2R_MOM | __DAAL_VSL_SS_2C_MOM, method),
                          errcode);
        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xQuantiles(const float * data, const __int64 nFeatures, const __int64 nVectors, const __int64 quantOrderN, const float * quantOrder,
                          float * quants)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(vslsSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&dataStorage, data, 0, 0),
                          errcode);
        if (errcode)
        {
            return errcode;
        }

        __DAAL_VSLFN_CALL(vsliSSEditTask, (task, __DAAL_VSL_SS_ED_QUANT_ORDER_N, (const MKL_INT *)&quantOrderN), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_QUANT_ORDER, quantOrder), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_QUANT_QUANTILES, quants), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vslsSSCompute, (task, __DAAL_VSL_SS_QUANTS, __DAAL_VSL_SS_METHOD_FAST), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xSort(float * data, __int64 nFeatures, __int64 nVectors, float * sortedData)
    {
        VSLSSTaskPtr task;
        int errcode = 0;

        __int64 inputStorage  = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __int64 outputStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(vslsSSNewTask, (&task, (const MKL_INT *)&nFeatures, (const MKL_INT *)&nVectors, (const MKL_INT *)&inputStorage, data, 0, 0),
                          errcode);
        if (errcode)
        {
            return errcode;
        }

        __DAAL_VSLFN_CALL(vslsSSEditTask, (task, __DAAL_VSL_SS_ED_SORTED_OBSERV, sortedData), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vsliSSEditTask, (task, __DAAL_VSL_SS_ED_SORTED_OBSERV_STORAGE, (const MKL_INT *)&outputStorage), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vslsSSCompute, (task, __DAAL_VSL_SS_SORTED_OBSERV, __DAAL_VSL_SS_METHOD_RADIX), errcode);
        if (errcode)
        {
            __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
            return errcode;
        }

        __DAAL_VSLFN_CALL(vslSSDeleteTask, (&task), errcode);
        return errcode;
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
