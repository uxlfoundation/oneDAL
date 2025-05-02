/* file: svm_train_result.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  SVM save result structure implementation
//--
*/

#ifndef __SVM_TRAIN_RESULT_H__
#define __SVM_TRAIN_RESULT_H__

#include "services/daal_defines.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_utils.h"
#include "src/services/service_data_utils.h"
#include "src/algorithms/svm/svm_train_cache.h"
#include "src/algorithms/svm/svm_train_common.h"
#include "src/externals/service_profiler.h"
#include "src/externals/service_math.h"
#include "src/algorithms/svm/svm_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
using namespace daal::internal;

/**
* \brief Write support vectors and coefficients into output model
*/
template <typename algorithmFPType, CpuType cpu>
class SaveResultTask
{
public:
    SaveResultTask(const size_t nVectors, const algorithmFPType * y, algorithmFPType * alpha, const algorithmFPType * grad, const SvmType task,
                   SVMCacheCommonIface<algorithmFPType, cpu> * cache)
        : _nVectors(nVectors), _y(y), _alpha(alpha), _grad(grad), _task(task), _cache(cache)
    {}

    services::Status compute(const NumericTablePtr & xTable, Model & model, const algorithmFPType * cw) const
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(saveResult);

        services::Status s;

        /* Calculate bias and write it into model */
        model.setBias(double(calculateBias(cw)));

        if (_task == SvmType::regression || _task == SvmType::nu_regression)
        {
            for (size_t i = 0; i < _nVectors; ++i)
            {
                _alpha[i] = _alpha[i] - _alpha[i + _nVectors];
            }
        }
        else
        {
            for (size_t i = 0; i < _nVectors; ++i)
            {
                _alpha[i] = _alpha[i] * _y[i];
            }
        }

        const algorithmFPType zero(0.0);
        size_t nSV = 0;
        for (size_t i = 0; i < _nVectors; ++i)
        {
            // Here and below, a bitwise comparison is necessary. For small c, any deviation from 0 means that
            // indices where coefficients not equal 0 will be support vectors.
            if (_alpha[i] != zero)
            {
                ++nSV;
            }
        }

        model.setNFeatures(xTable->getNumberOfColumns());
        if (nSV == 0) return s;
        DAAL_CHECK_STATUS(s, setSVCoefficients(nSV, model));
        DAAL_CHECK_STATUS(s, setSVIndices(nSV, model));
        DAAL_CHECK_STATUS(s, setSVByIndices(xTable.get(), model.getSupportIndices(), model.getSupportVectors()));

        return s;
    }

    static services::Status setSVByIndices(const NumericTable * xTable, const NumericTablePtr & svIndicesTable, NumericTablePtr svTable)
    {
        services::Status s;
        if (xTable->getDataLayout() == NumericTableIface::csrArray)
        {
            DAAL_CHECK_STATUS(s, setSVCSRByIndices(xTable, svIndicesTable, svTable));
        }
        else
        {
            DAAL_CHECK_STATUS(s, setSVDenseByIndices(xTable, svIndicesTable, svTable));
        }
        return s;
    }

protected:
    /**
     * \brief Write classification coefficients into resulting model
     *template <typename algorithmFPType, typename ParameterType, CpuType cpu>
     * \param[in]  nSV          Number of support vectors
     * \param[out] model        Resulting model
     */
    services::Status setSVCoefficients(size_t nSV, Model & model) const
    {
        const algorithmFPType zero(0.0);
        NumericTablePtr svCoeffTable = model.getClassificationCoefficients();
        services::Status s;
        DAAL_CHECK_STATUS(s, svCoeffTable->resize(nSV));

        WriteOnlyRows<algorithmFPType, cpu> mtSvCoeff(*svCoeffTable, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSvCoeff);
        algorithmFPType * const svCoeff = mtSvCoeff.get();

        for (size_t i = 0, iSV = 0; i < _nVectors; ++i)
        {
            if (_alpha[i] != zero)
            {
                svCoeff[iSV] = _alpha[i];
                ++iSV;
            }
        }
        return s;
    }

    /**
     * \brief Write indices of the support vectors into resulting model
     *
     * \param[in]  nSV          Number of support vectors
     * \param[out] model        Resulting model
     */
    services::Status setSVIndices(size_t nSV, Model & model) const
    {
        NumericTablePtr svIndicesTable = model.getSupportIndices();
        services::Status s;
        DAAL_CHECK_STATUS(s, svIndicesTable->resize(nSV));

        WriteOnlyRows<int, cpu> mtSvIndices(*svIndicesTable, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSvIndices);
        int * const svIndices = mtSvIndices.get();

        const algorithmFPType zero(0.0);
        for (size_t i = 0, iSV = 0; i < _nVectors; ++i)
        {
            if (_alpha[i] != zero)
            {
                DAAL_ASSERT(_cache->getDataRowIndex(i) < _nVectors)
                DAAL_ASSERT(_cache->getDataRowIndex(i) <= services::internal::MaxVal<int>::get())
                svIndices[iSV] = static_cast<int>(_cache->getDataRowIndex(i));
                ++iSV;
            }
        }
        return s;
    }

    static services::Status setSVDenseByIndices(const NumericTable * xTable, const NumericTablePtr & svIndicesTable, NumericTablePtr svTable)
    {
        services::Status s;
        const size_t nSV = svIndicesTable->getNumberOfRows();
        if (nSV == 0) return s;

        DAAL_CHECK_STATUS(s, svTable->resize(nSV));
        const size_t p = xTable->getNumberOfColumns();

        ReadRows<int, cpu> mtSvIndices(svIndicesTable.get(), 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSvIndices);
        const int * const svIndices = mtSvIndices.get();

        WriteOnlyRows<algorithmFPType, cpu> mtSv(*svTable, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSv);
        algorithmFPType * const sv = mtSv.get();

        SafeStatus safeStat;
        daal::threader_for(nSV, nSV, [&](const size_t iBlock) {
            const size_t iRows = svIndices[iBlock];
            ReadRows<algorithmFPType, cpu> mtX(const_cast<NumericTable *>(xTable), iRows, 1);
            DAAL_CHECK_BLOCK_STATUS_THR(mtX);
            const algorithmFPType * const dataIn = mtX.get();
            algorithmFPType * dataOut            = sv + iBlock * p;
            DAAL_CHECK_THR(!services::internal::daal_memcpy_s(dataOut, p * sizeof(algorithmFPType), dataIn, p * sizeof(algorithmFPType)),
                           services::ErrorMemoryCopyFailedInternal);
        });

        return safeStat.detach();
    }

    static services::Status setSVCSRByIndices(const NumericTable * xTable, const NumericTablePtr & svIndicesTable, NumericTablePtr svTable)
    {
        services::Status s;
        const size_t nSV = svIndicesTable->getNumberOfRows();
        if (nSV == 0) return s;

        TArray<size_t, cpu> aSvRowOffsets(nSV + 1);
        DAAL_CHECK_MALLOC(aSvRowOffsets.get());
        size_t * const svRowOffsetsBuffer = aSvRowOffsets.get();

        CSRNumericTableIface * const csrIface = dynamic_cast<CSRNumericTableIface * const>(const_cast<NumericTable *>(xTable));
        DAAL_CHECK(csrIface, services::ErrorEmptyCSRNumericTable);

        ReadRowsCSR<algorithmFPType, cpu> mtX;

        ReadRows<int, cpu> mtSvIndices(svIndicesTable.get(), 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSvIndices);
        const int * const svIndices = mtSvIndices.get();

        /* Calculate row offsets for the table that stores support vectors */
        svRowOffsetsBuffer[0] = 1;
        for (size_t iSV = 0; iSV < nSV; ++iSV)
        {
            const size_t rowIndex = svIndices[iSV];
            mtX.set(csrIface, rowIndex, 1);
            DAAL_CHECK_BLOCK_STATUS(mtX);
            svRowOffsetsBuffer[iSV + 1] = svRowOffsetsBuffer[iSV] + (mtX.rows()[1] - mtX.rows()[0]);
        }

        /* Allocate memory for storing support vectors and coefficients */
        CSRNumericTablePtr svCsrTable = services::staticPointerCast<CSRNumericTable, NumericTable>(svTable);
        DAAL_CHECK_STATUS(s, svCsrTable->resize(nSV));

        const size_t svDataSize = svRowOffsetsBuffer[nSV] - svRowOffsetsBuffer[0];
        /* If matrix is zeroes -> svDataSize will be equal 0.
           So for correct works we added 1 in this case. */
        DAAL_CHECK_STATUS(s, svCsrTable->allocateDataMemory(svDataSize ? svDataSize : svDataSize + 1));

        /* Copy row offsets into the table */
        size_t * svRowOffsets = nullptr;
        svCsrTable->getArrays<algorithmFPType>(NULL, NULL, &svRowOffsets);
        for (size_t i = 0; i < nSV + 1; i++)
        {
            svRowOffsets[i] = svRowOffsetsBuffer[i];
        }

        const algorithmFPType zero(0.0);

        WriteOnlyRowsCSR<algorithmFPType, cpu> mtSv(*svCsrTable, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSv);
        algorithmFPType * sv        = mtSv.values();
        size_t * const svColIndices = mtSv.cols();
        for (size_t iSV = 0, svOffset = 0; iSV < nSV; ++iSV)
        {
            const size_t rowIndex = svIndices[iSV];
            mtX.set(csrIface, rowIndex, 1);
            DAAL_CHECK_BLOCK_STATUS(mtX);
            const algorithmFPType * const xi  = mtX.values();
            const size_t * const xiColIndices = mtX.cols();
            const size_t nNonZeroValuesInRow  = mtX.rows()[1] - mtX.rows()[0];
            for (size_t j = 0; j < nNonZeroValuesInRow; j++, svOffset++)
            {
                sv[svOffset]           = xi[j];
                svColIndices[svOffset] = xiColIndices[j];
            }
        }

        return s;
    }

    /**
     * \brief Calculate SVM model bias
     *
     * \param[in]  C        Upper bound in constraints of the quadratic optimization problem
     * \return Bias for the SVM model
     */
    algorithmFPType calculateBiasImpl(const algorithmFPType * cw, SignNuType signNuType = SignNuType::none) const
    {
        algorithmFPType bias    = algorithmFPType(0.0);
        size_t nGrad            = 0;
        algorithmFPType sumGrad = algorithmFPType(0.0);

        const algorithmFPType fpMax = MaxVal<algorithmFPType>::get();
        algorithmFPType ub          = fpMax;
        algorithmFPType lb          = -fpMax;

        const size_t nTrainVectors = (_task == SvmType::regression || _task == SvmType::nu_regression) ? _nVectors * 2 : _nVectors;
        for (size_t i = 0; i < nTrainVectors; ++i)
        {
            const algorithmFPType gradi = _grad[i];
            const algorithmFPType yi    = _y[i];
            const algorithmFPType cwi   = cw[i];
            const algorithmFPType ai    = _alpha[i];

            /* free SV: (0 < alpha < C)*/
            if (HelperTrainSVM<algorithmFPType, cpu>::checkLabel(yi, signNuType) && 0 < ai && ai < cwi)
            {
                sumGrad += gradi;
                ++nGrad;
            }
            if (HelperTrainSVM<algorithmFPType, cpu>::isUpper(yi, ai, cwi, signNuType))
            {
                ub = services::internal::min<cpu, algorithmFPType>(ub, gradi);
            }
            if (HelperTrainSVM<algorithmFPType, cpu>::isLower(yi, ai, cwi, signNuType))
            {
                lb = services::internal::max<cpu, algorithmFPType>(lb, gradi);
            }
        }

        const double sign = (signNuType == SignNuType::positive) ? 1.0 : -1.0;
        if (nGrad == 0)
        {
            bias = sign * 0.5 * (ub + lb);
        }
        else
        {
            bias = sign * sumGrad / algorithmFPType(nGrad);
        }

        return bias;
    }

    algorithmFPType calculateBias(const algorithmFPType * cw) const
    {
        algorithmFPType bias = 0;

        if (_task == SvmType::classification || _task == SvmType::regression)
        {
            bias = calculateBiasImpl(cw);
        }
        else if (_task == SvmType::nu_classification || _task == SvmType::nu_regression)
        {
            const algorithmFPType biasPos = calculateBiasImpl(cw, SignNuType::positive);
            const algorithmFPType biasNeg = calculateBiasImpl(cw, SignNuType::negative);
            bias                          = (biasNeg - biasPos) / algorithmFPType(2);

            if (_task == SvmType::nu_classification)
            {
                const algorithmFPType r = (biasPos + biasNeg) / algorithmFPType(2);

                for (size_t i = 0; i < _nVectors; ++i)
                {
                    _alpha[i] /= r;
                }
                bias /= r;
            }
        }

        return bias;
    }

private:
    const size_t _nVectors;                             //Number of observations in the input data set
    const algorithmFPType * _y;                         //Array of labels
    algorithmFPType * _alpha;                           //Array of coefficients
    const algorithmFPType * _grad;                      //Array of coefficients
    const SvmType _task;                                //Classification or regression task
    SVMCacheCommonIface<algorithmFPType, cpu> * _cache; //Caches matrix Q (kernel(x[i], x[j])) values
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
