/* file: multiclassclassifier_train_oneagainstone_impl.i */
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
//  Implementation of One-Against-One method for Multi-class classifier
//  training algorithm for CSR input data.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_TRAIN_ONEAGAINSTONE_IMPL_I__
#define __MULTICLASSCLASSIFIER_TRAIN_ONEAGAINSTONE_IMPL_I__

#include "algorithms/multi_class_classifier/multi_class_classifier_model.h"
#include "algorithms/svm/svm_model.h"

#include "src/threading/threading.h"
#include "src/algorithms/service_sort.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_blas.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace training
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::data_management;
using namespace daal::services;
using namespace multi_class_classifier::internal;
using namespace svm::training::internal;

template <typename algorithmFPType, CpuType cpu>
services::Status MultiClassClassifierTrainKernel<oneAgainstOne, algorithmFPType, cpu>::compute(const NumericTable * xTable,
                                                                                               const NumericTable * yTable,
                                                                                               const NumericTable * wTable,
                                                                                               daal::algorithms::Model * m, SvmModel * svmModel,
                                                                                               const KernelParameter & par)
{
    Status s;

    const bool isOutSvmModel = svmModel != nullptr;
    Model * model            = static_cast<Model *>(m);

    const size_t nVectors = xTable->getNumberOfRows();
    ReadColumns<algorithmFPType, cpu> mtY(*const_cast<NumericTable *>(yTable), 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtY);
    const algorithmFPType * y = mtY.get();

    ReadColumns<algorithmFPType, cpu> mtW(const_cast<NumericTable *>(wTable), 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtW);
    const algorithmFPType * weights = mtW.get();

    const size_t nFeatures = xTable->getNumberOfColumns();
    model->setNFeatures(nFeatures);
    auto simpleTrainingInit = par.training->clone();

    const size_t nClasses = par.nClasses;
    /* Compute data size needed to store the largest subset of input tables */
    size_t nSubsetVectors, dataSize;
    DAAL_CHECK_STATUS(s, computeDataSize(nVectors, nFeatures, nClasses, xTable, y, nSubsetVectors, dataSize));

    typedef SubTask<algorithmFPType, cpu> TSubTask;
    /* Allocate memory for storing subsets of input data */
    daal::ls<TSubTask *> lsTask([=, &simpleTrainingInit]() {
        if (xTable->getDataLayout() == NumericTableIface::csrArray)
            return (TSubTask *)SubTaskCSR<algorithmFPType, cpu>::create(nFeatures, nSubsetVectors, dataSize, xTable, weights, simpleTrainingInit);
        return (TSubTask *)SubTaskDense<algorithmFPType, cpu>::create(nFeatures, nSubsetVectors, dataSize, xTable, weights, simpleTrainingInit);
    });

    SafeStatus safeStat;

    TArray<bool, cpu> isSV;
    bool * isSVData = nullptr;
    if (isOutSvmModel)
    {
        isSV.reset(nVectors);
        DAAL_CHECK_MALLOC(isSV.get());
        isSVData = isSV.get();
        services::internal::service_memset<bool, cpu>(isSVData, false, nVectors);
    }

    const size_t nModels = (nClasses * (nClasses - 1)) >> 1;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nModels, 2);
    TArray<size_t, cpu> classIndices(nModels * 2);
    DAAL_CHECK_MALLOC(classIndices.get());
    size_t * classIndicesData = classIndices.get();
    if (isOutSvmModel)
    {
        for (size_t iClass = 0, imodel = 0; iClass < nClasses; ++iClass)
        {
            for (size_t jClass = iClass + 1; jClass < nClasses; ++jClass, ++imodel)
            {
                classIndicesData[imodel]           = iClass;
                classIndicesData[imodel + nModels] = jClass;
            }
        }
    }
    else
    {
        for (size_t iClass = 1, imodel = 0; iClass < nClasses; ++iClass)
        {
            for (size_t jClass = 0; jClass < iClass; ++jClass, ++imodel)
            {
                classIndicesData[imodel]           = iClass;
                classIndicesData[imodel + nModels] = jClass;
            }
        }
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nModels, nSubsetVectors);
    TArrayScalable<size_t, cpu> originalIndicesMap(nModels * nSubsetVectors);
    DAAL_CHECK_MALLOC(originalIndicesMap.get());
    size_t * const originalIndicesMapData = originalIndicesMap.get();

    daal::threader_for(nModels, nModels, [&](size_t imodel) {
        const size_t iClass = classIndicesData[imodel];
        const size_t jClass = classIndicesData[imodel + nModels];

        TSubTask * local = lsTask.local();
        if (!local)
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
            return;
        }
        DAAL_LS_RELEASE(TSubTask, lsTask, local); //releases local storage when leaving this scope

        size_t nRowsInSubset                   = 0;
        size_t * const originalIndicesMapLocal = originalIndicesMapData + imodel * nSubsetVectors;
        s |= local->getDataSubset(nFeatures, nVectors, iClass, jClass, y, originalIndicesMapLocal, nRowsInSubset);
        DAAL_CHECK_STATUS_THR(s);
        classifier::ModelPtr pModel;
        if (nRowsInSubset)
        {
            s |= local->trainSimpleClassifier(nRowsInSubset);
            if (!s)
            {
                safeStat |= s;
                safeStat.add(services::ErrorMultiClassFailedToTrainTwoClassClassifier);
                return;
            }
            pModel = local->getModel();
        }
        model->setTwoClassClassifierModel(imodel, pModel);
        if (svmModel)
        {
            auto svmModelPtr   = daal::services::staticPointerCast<svm::Model>(pModel);
            auto twoClassSvInd = svmModelPtr->getSupportIndices();
            const size_t nSV   = twoClassSvInd->getNumberOfRows();

            ReadColumns<int, cpu> mtSvIndex(twoClassSvInd.get(), 0, 0, nSV);
            DAAL_CHECK_BLOCK_STATUS_THR(mtSvIndex);
            const int * twoClassSvIndData = mtSvIndex.get();

            for (size_t svId = 0; svId < nSV; ++svId)
            {
                DAAL_ASSERT(twoClassSvIndData[svId] < nRowsInSubset);
                const size_t originalIndex = originalIndicesMapLocal[twoClassSvIndData[svId]];
                isSVData[originalIndex]    = true;
            }
            auto biasesTable = svmModel->getBiases();
            WriteOnlyColumns<algorithmFPType, cpu> mtBiases(biasesTable.get(), 0, imodel, 1);
            DAAL_CHECK_BLOCK_STATUS_THR(mtBiases);
            *mtBiases.get() = svmModelPtr->getBias();
        }
    });
    lsTask.reduce([=, &safeStat](TSubTask * local) { delete local; });

    if (svmModel)
    {
        TArray<size_t, cpu> svCounts(nClasses);
        DAAL_CHECK_MALLOC(svCounts.get());
        size_t * const svCountsData = svCounts.get();
        size_t nSV                  = 0;
        for (size_t iClass = 0; iClass < nClasses; ++iClass)
        {
            svCountsData[iClass] = 0;
            for (size_t j = 0; j < nVectors; ++j)
            {
                const size_t label = size_t(y[j]);
                if (isSVData[j] && (label == iClass))
                {
                    ++svCountsData[iClass];
                }
            }
            nSV += svCountsData[iClass];
        }

        NumericTablePtr supportIndicesTable = svmModel->getSupportIndices();
        DAAL_CHECK_STATUS(s, supportIndicesTable->resize(nSV));

        TArray<size_t, cpu> svGroupByClassArray(nSV);
        DAAL_CHECK_MALLOC(svGroupByClassArray.get());
        size_t * const svGroupByClass = svGroupByClassArray.get();

        TArray<size_t, cpu> svIndMappingArray(nVectors);
        DAAL_CHECK_MALLOC(svIndMappingArray.get());
        size_t * const svIndMapping = svIndMappingArray.get();

        {
            WriteOnlyColumns<int, cpu> mtSupportIndices(supportIndicesTable.get(), 0, 0, nSV);
            DAAL_CHECK_BLOCK_STATUS(mtSupportIndices);
            int * supportIndices = mtSupportIndices.get();

            size_t inxSV = 0;
            for (size_t iClass = 0; iClass < nClasses; ++iClass)
            {
                for (size_t j = 0; j < nVectors; ++j)
                {
                    const size_t label = static_cast<size_t>(y[j]);
                    if (isSVData[j] && (label == iClass))
                    {
                        supportIndices[inxSV] = j;
                        svIndMapping[j]       = inxSV;
                        ++inxSV;
                    }
                }
            }
            DAAL_ASSERT(inxSV == nSV);
        }
        NumericTablePtr coeffOutTable = svmModel->getCoefficients();
        DAAL_CHECK_STATUS(s, coeffOutTable->resize(nSV));

        using SvmResultTask = SaveResultTask<algorithmFPType, cpu>;
        DAAL_CHECK_STATUS(s, SvmResultTask::setSVByIndices(xTable, supportIndicesTable, svmModel->getSupportVectors()));
        WriteOnlyRows<algorithmFPType, cpu> mtCoefficientsOut(coeffOutTable.get(), 0, coeffOutTable->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(mtCoefficientsOut);
        algorithmFPType * const coefficientsOut = mtCoefficientsOut.get();
        services::internal::service_memset<algorithmFPType, cpu>(coefficientsOut, algorithmFPType(0),
                                                                 coeffOutTable->getNumberOfRows() * coeffOutTable->getNumberOfColumns());
        daal::threader_for(nModels, nModels, [&](size_t imodel) {
            const size_t iClass = classIndicesData[imodel];
            const size_t jClass = classIndicesData[imodel + nModels];

            auto svmModelId   = daal::services::staticPointerCast<svm::Model>(model->getTwoClassClassifierModel(imodel));
            auto coeffInTable = svmModelId->getClassificationCoefficients();
            ReadRows<algorithmFPType, cpu> mtCoefficientsIn(coeffInTable.get(), 0, coeffInTable->getNumberOfRows());
            DAAL_CHECK_BLOCK_STATUS_THR(mtCoefficientsIn);
            const algorithmFPType * const coeffIn = mtCoefficientsIn.get();

            auto twoClassSvIndTable = svmModelId->getSupportIndices();
            const size_t svCounts   = twoClassSvIndTable->getNumberOfRows();
            ReadColumns<int, cpu> mtSvIndex(twoClassSvIndTable.get(), 0, 0, svCounts);
            DAAL_CHECK_BLOCK_STATUS_THR(mtSvIndex);
            const int * twoClassSvInd        = mtSvIndex.get();
            size_t * originalIndicesMapLocal = originalIndicesMapData + imodel * nSubsetVectors;

            for (size_t sv = 0; sv < svCounts; ++sv)
            {
                const size_t originalInd = originalIndicesMapLocal[twoClassSvInd[sv]];
                const size_t label       = static_cast<size_t>(y[originalInd]);
                const size_t colInd      = svIndMapping[originalInd];

                if (label == iClass)
                {
                    coefficientsOut[colInd * (nClasses - 1) + (jClass - 1)] = coeffIn[sv];
                }
                else if (label == jClass)
                {
                    coefficientsOut[colInd * (nClasses - 1) + (iClass)] = coeffIn[sv];
                }
            }
        });
    }
    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
Status MultiClassClassifierTrainKernel<oneAgainstOne, algorithmFPType, cpu>::computeDataSize(size_t nVectors, size_t nFeatures, size_t nClasses,
                                                                                             const NumericTable * xTable, const algorithmFPType * y,
                                                                                             size_t & nSubsetVectors, size_t & dataSize)
{
    TArray<size_t, cpu> buffer(2 * nClasses);
    DAAL_CHECK_MALLOC(buffer.get());
    daal::services::internal::service_memset_seq<size_t, cpu>(buffer.get(), 0, buffer.size());
    size_t * classLabelsCount        = buffer.get();
    size_t * classNonZeroValuesCount = buffer.get() + nClasses;
    for (size_t i = 0; i < nVectors; ++i)
    {
        ++classLabelsCount[size_t(y[i])];
    }
    if (xTable->getDataLayout() == NumericTableIface::csrArray)
    {
        CSRNumericTableIface * csrIface = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(xTable));
        DAAL_CHECK(csrIface, ErrorEmptyCSRNumericTable);
        ReadRowsCSR<algorithmFPType, cpu> _mtX(*csrIface, 0, nVectors);
        DAAL_CHECK_BLOCK_STATUS(_mtX);
        const size_t * rowOffsets = _mtX.rows();
        /* Compute data size needed to store the largest subset of input tables */
        for (size_t i = 0; i < nVectors; ++i)
        {
            classNonZeroValuesCount[size_t(y[i])] += (rowOffsets[i + 1] - rowOffsets[i]);
        }

        daal::algorithms::internal::qSort<size_t, cpu>(nClasses, classNonZeroValuesCount);
        dataSize = classNonZeroValuesCount[nClasses - 1] + classNonZeroValuesCount[nClasses - 2];
        daal::algorithms::internal::qSort<size_t, cpu>(nClasses, classLabelsCount);
        nSubsetVectors = classLabelsCount[nClasses - 1] + classLabelsCount[nClasses - 2];
    }
    else
    {
        daal::algorithms::internal::qSort<size_t, cpu>(nClasses, classLabelsCount);
        nSubsetVectors = classLabelsCount[nClasses - 1] + classLabelsCount[nClasses - 2];
        dataSize       = nFeatures * nSubsetVectors;
    }

    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status SubTaskDense<algorithmFPType, cpu>::copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label,
                                                                const algorithmFPType * y, size_t * originalIndicesMap, size_t & nRows)
{
    for (size_t ix = 0; ix < nVectors; ix++)
    {
        if (size_t(y[ix]) != classIdx) continue;
        originalIndicesMap[nRows] = ix;
        _mtX.next(ix, 1);
        DAAL_CHECK_BLOCK_STATUS(_mtX);
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t jx = 0; jx < nFeatures; jx++) this->_subsetX.get()[nRows * nFeatures + jx] = _mtX.get()[jx];
        this->_subsetY[nRows] = label;
        if (this->_weights)
        {
            this->_subsetW[nRows] = this->_weights[ix];
        }
        ++nRows;
    }
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status SubTaskCSR<algorithmFPType, cpu>::copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label,
                                                              const algorithmFPType * y, size_t * originalIndicesMap, size_t & nRows)
{
    _rowOffsetsX[0]  = 1;
    size_t dataIndex = (nRows ? _rowOffsetsX[nRows] - _rowOffsetsX[0] : 0);
    for (size_t ix = 0; ix < nVectors; ix++)
    {
        if (size_t(y[ix]) != classIdx) continue;
        originalIndicesMap[nRows] = ix;
        _mtX.next(ix, 1);
        DAAL_CHECK_BLOCK_STATUS(_mtX);
        const size_t nNonZeroValuesInRow  = _mtX.rows()[1] - _mtX.rows()[0];
        const size_t * colIndices         = _mtX.cols();
        const algorithmFPType * mtXValues = _mtX.values();
        algorithmFPType * subsetXData     = this->_subsetX.get();
        for (size_t jx = 0; jx < nNonZeroValuesInRow; ++jx, ++dataIndex)
        {
            subsetXData[dataIndex]  = mtXValues[jx];
            _colIndicesX[dataIndex] = colIndices[jx];
        }
        _rowOffsetsX[nRows + 1] = _rowOffsetsX[nRows] + nNonZeroValuesInRow;
        this->_subsetY[nRows]   = label;
        if (this->_weights)
        {
            this->_subsetW[nRows] = this->_weights[ix];
        }

        ++nRows;
    }
    return Status();
}

} // namespace internal
} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
