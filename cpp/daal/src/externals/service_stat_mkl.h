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

#include <type_traits>

#include <mkl.h>
#include <mkl_vsl_functions.h>
#include "services/error_indexes.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_stat_rng_mkl.h"

#undef __DAAL_VSLFN_CALL
#define __DAAL_VSLFN_CALL(f_name, f_args, errcode) errcode = f_name f_args;

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

const MKL_INT storageCols = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
const MKL_INT storageFull = __DAAL_VSL_SS_MATRIX_STORAGE_FULL;

template <class fpType>
struct MKLTaskInterfacer
{
public:
    VSLSSTaskPtr task;
    int errcode = 0;

    MKLTaskInterfacer(const __int64 * nFeatures, const __int64 * nVectors, const double * data)
    {
        __DAAL_VSLFN_CALL(vsldSSNewTask, (&this->task, (const MKL_INT *)nFeatures, (const MKL_INT *)nVectors, &storageCols, data, 0, 0),
                          this->errcode);
    }

    MKLTaskInterfacer(const __int64 * nFeatures, const __int64 * nVectors, const float * data)
    {
        __DAAL_VSLFN_CALL(vslsSSNewTask, (&this->task, (const MKL_INT *)nFeatures, (const MKL_INT *)nVectors, &storageCols, data, 0, 0),
                          this->errcode);
    }

    ~MKLTaskInterfacer() { __DAAL_VSLFN_CALL(vslSSDeleteTask, (&this->task), this->errcode); }

    void edit(const MKL_INT taskId, const double * data) { __DAAL_VSLFN_CALL(vsldSSEditTask, (this->task, taskId, data), this->errcode); }

    void edit(const MKL_INT taskId, const float * data) { __DAAL_VSLFN_CALL(vslsSSEditTask, (this->task, taskId, data), this->errcode); }

    void edit_i(const MKL_INT taskId, const __int64 * data) { __DAAL_VSLFN_CALL(vsliSSEditTask, (this->task, taskId, data), this->errcode); }

    void edit_outliers_detection(const __int64 * nParams, const double * params, double * w)
    {
        __DAAL_VSLFN_CALL(vsldSSEditOutliersDetection, (this->task, (const MKL_INT *)nParams, params, w), this->errcode);
    }

    void edit_outliers_detection(const __int64 * nParams, const float * params, float * w)
    {
        __DAAL_VSLFN_CALL(vslsSSEditOutliersDetection, (this->task, (const MKL_INT *)nParams, params, w), this->errcode);
    }

    void compute(const MKL_INT taskId, const MKL_INT method)
    {
        __DAAL_VSLFN_CALL((std::is_same<fpType, double>::value ? vsldSSCompute : vslsSSCompute), (this->task, taskId, method), this->errcode);
    }
};

#define CALL_AND_CHECK(x) \
    x;                    \
    if (task.errcode) return task.errcode;

template <class fpType>
class DaalManagedMalloc
{
public:
    fpType * ptr = nullptr;

    DaalManagedMalloc(const size_t num_elts) { this->ptr = static_cast<fpType *>(daal::services::daal_malloc(num_elts * sizeof(fpType))); }

    ~DaalManagedMalloc() { daal::services::daal_free(this->ptr); }
};

template <class fpType, CpuType cpu>
class DaalManagedScalableMalloc
{
public:
    fpType * ptr = nullptr;

    DaalManagedScalableMalloc(const size_t num_elts)
    {
        this->ptr = static_cast<fpType *>(daal::services::internal::service_scalable_malloc<fpType, cpu>(num_elts));
    }

    ~DaalManagedScalableMalloc() { daal::services::internal::service_scalable_free<fpType, cpu>(this->ptr); }
};

template <class fpType, CpuType cpu>
struct MklStatistics
{
    typedef __int64 SizeType;
    typedef __int64 MethodType;
    typedef int ErrorType;

    static int xmeansOnePass(const fpType * data, __int64 nFeatures, __int64 nVectors, fpType * means)
    {
        CALL_AND_CHECK(MKLTaskInterfacer<fpType> task(&nFeatures, &nVectors, data));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_MEAN, means));
        CALL_AND_CHECK(task.compute(__DAAL_VSL_SS_MEAN, VSL_SS_METHOD_1PASS));
        return 0;
    }

    static int xcp(const fpType * data, __int64 nFeatures, __int64 nVectors, fpType * nPreviousObservations, fpType * sum, fpType * crossProduct,
                   __int64 method)
    {
        DaalManagedMalloc<fpType> meanHolder(nFeatures);
        fpType * mean = meanHolder.ptr;
        if (!mean) return services::ErrorMemoryAllocationFailed;

        if (method == __DAAL_VSL_SS_METHOD_FAST_USER_MEAN)
        {
            fpType invNVectors = fpType(1) / static_cast<fpType>(nVectors);
            for (size_t i = 0; i < nFeatures; i++)
            {
                mean[i] = sum[i] * invNVectors;
            }
        }
        CALL_AND_CHECK(MKLTaskInterfacer<fpType> task(&nFeatures, &nVectors, data));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_SUM, sum));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_MEAN, mean));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_CP, crossProduct));
        CALL_AND_CHECK(task.edit_i(__DAAL_VSL_SS_ED_CP_STORAGE, &storageFull));
        fpType weight[2] = { *nPreviousObservations, *nPreviousObservations };
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_ACCUM_WEIGHT, weight));
        CALL_AND_CHECK(task.compute(__DAAL_VSL_SS_CP | __DAAL_VSL_SS_SUM, method));

        return 0;
    }

    static int xxcp_weight(const fpType * data, __int64 nFeatures, __int64 nVectors, fpType * weight, fpType * accumWeight, fpType * mean,
                           fpType * crossProduct, __int64 method)
    {
        DaalManagedScalableMalloc<fpType, cpu> sumHolder(nFeatures);
        fpType * sum = sumHolder.ptr;
        if (!sum) return services::ErrorMemoryAllocationFailed;

        CALL_AND_CHECK(MKLTaskInterfacer<fpType> task(&nFeatures, &nVectors, data));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_SUM, sum));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_MEAN, mean));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_WEIGHTS, weight));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_CP, crossProduct));
        CALL_AND_CHECK(task.edit_i(__DAAL_VSL_SS_ED_CP_STORAGE, &storageFull));
        fpType accumWeightsAll[2] = { 0, 0 };
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_ACCUM_WEIGHT, accumWeightsAll));
        CALL_AND_CHECK(task.compute(__DAAL_VSL_SS_CP | __DAAL_VSL_SS_SUM, method));
        *accumWeight = accumWeightsAll[0];
        return 0;
    }

    static int xxvar_weight(const fpType * data, __int64 nFeatures, __int64 nVectors, fpType * weight, fpType * accumWeight, fpType * mean,
                            fpType * sampleVariance, __int64 method)
    {
        DaalManagedScalableMalloc<fpType, cpu> sumHolder(nFeatures);
        DaalManagedScalableMalloc<fpType, cpu> rawSecondHolder(nFeatures);
        fpType * sum       = sumHolder.ptr;
        fpType * rawSecond = rawSecondHolder.ptr;
        if (!sum || !rawSecond) return services::ErrorMemoryAllocationFailed;

        fpType accumWeightsAll[2] = { 0, 0 };

        CALL_AND_CHECK(MKLTaskInterfacer<fpType> task(&nFeatures, &nVectors, data));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_SUM, sum));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_MEAN, mean));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_WEIGHTS, weight));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_2C_SUM, sampleVariance));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_2R_MOM, rawSecond));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_ACCUM_WEIGHT, accumWeightsAll));
        CALL_AND_CHECK(task.compute(__DAAL_VSL_SS_2C_SUM | __DAAL_VSL_SS_MEAN, method));

        *accumWeight = accumWeightsAll[0];
        return 0;
    }

    static int x2c_mom(const fpType * data, const __int64 nFeatures, const __int64 nVectors, fpType * variance, const __int64 method)
    {
        DaalManagedMalloc<fpType> meanHolder(nFeatures);
        DaalManagedMalloc<fpType> secondOrderRawMomentHolder(nFeatures);
        fpType * mean                 = meanHolder.ptr;
        fpType * secondOrderRawMoment = secondOrderRawMomentHolder.ptr;
        if (!mean || !secondOrderRawMoment) return services::ErrorMemoryAllocationFailed;

        CALL_AND_CHECK(MKLTaskInterfacer<fpType> task(&nFeatures, &nVectors, data));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_MEAN, mean));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_2R_MOM, secondOrderRawMoment));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_2C_MOM, variance));
        CALL_AND_CHECK(task.compute(__DAAL_VSL_SS_2C_MOM, method));
        return 0;
    }

    static int xoutlierdetection(const fpType * data, const __int64 nFeatures, const __int64 nVectors, const __int64 nParams,
                                 const fpType * baconParams, fpType * baconWeights)
    {
        CALL_AND_CHECK(MKLTaskInterfacer<fpType> task(&nFeatures, &nVectors, data));
        CALL_AND_CHECK(task.edit_outliers_detection(&nParams, baconParams, baconWeights));
        CALL_AND_CHECK(task.compute(__DAAL_VSL_SS_OUTLIERS, __DAAL_VSL_SS_METHOD_BACON));
        return 0;
    }

    static int xLowOrderMoments(const fpType * data, __int64 nFeatures, __int64 nVectors, __int64 method, fpType * sum, fpType * mean,
                                fpType * secondOrderRawMoment, fpType * variance, fpType * variation)
    {
        CALL_AND_CHECK(MKLTaskInterfacer<fpType> task(&nFeatures, &nVectors, data));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_SUM, sum));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_MEAN, mean));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_2R_MOM, secondOrderRawMoment));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_2C_MOM, variance));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_VARIATION, variation));
        CALL_AND_CHECK(
            task.compute(__DAAL_VSL_SS_SUM | __DAAL_VSL_SS_MEAN | __DAAL_VSL_SS_2R_MOM | __DAAL_VSL_SS_2C_MOM | __DAAL_VSL_SS_VARIATION, method));
        return 0;
    }

    static int xSumAndVariance(fpType * data, __int64 nFeatures, __int64 nVectors, fpType * nPreviousObservations, __int64 method, fpType * sum,
                               fpType * mean, fpType * secondOrderRawMoment, fpType * variance)
    {
        CALL_AND_CHECK(MKLTaskInterfacer<fpType> task(&nFeatures, &nVectors, data));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_SUM, sum));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_MEAN, mean));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_2R_MOM, secondOrderRawMoment));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_2C_MOM, variance));
        fpType weight[2] = { *nPreviousObservations, *nPreviousObservations };
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_ACCUM_WEIGHT, weight));
        CALL_AND_CHECK(task.compute(__DAAL_VSL_SS_SUM | __DAAL_VSL_SS_MEAN | __DAAL_VSL_SS_2R_MOM | __DAAL_VSL_SS_2C_MOM, method));
        return 0;
    }

    static int xQuantiles(const fpType * data, const __int64 nFeatures, const __int64 nVectors, const __int64 quantOrderN, const fpType * quantOrder,
                          fpType * quants)
    {
        CALL_AND_CHECK(MKLTaskInterfacer<fpType> task(&nFeatures, &nVectors, data));
        CALL_AND_CHECK(task.edit_i(__DAAL_VSL_SS_ED_QUANT_ORDER_N, (const MKL_INT *)&quantOrderN));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_QUANT_ORDER, quantOrder));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_QUANT_QUANTILES, quants));
        CALL_AND_CHECK(task.compute(__DAAL_VSL_SS_QUANTS, __DAAL_VSL_SS_METHOD_FAST));
        return 0;
    }

    static int xSort(const fpType * data, __int64 nFeatures, __int64 nVectors, fpType * sortedData)
    {
        CALL_AND_CHECK(MKLTaskInterfacer<fpType> task(&nFeatures, &nVectors, data));
        CALL_AND_CHECK(task.edit(__DAAL_VSL_SS_ED_SORTED_OBSERV, sortedData));
        CALL_AND_CHECK(task.edit_i(__DAAL_VSL_SS_ED_SORTED_OBSERV_STORAGE, &storageCols));
        CALL_AND_CHECK(task.compute(__DAAL_VSL_SS_SORTED_OBSERV, __DAAL_VSL_SS_METHOD_RADIX));
        return 0;
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
