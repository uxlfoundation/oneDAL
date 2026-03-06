/* file: multiclassclassifier_train_oneagainstone_kernel.h */
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
//  Declaration of template structs for One-Against-One method for Multi-class classifier
//  training algorithm for CSR input data.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_TRAIN_ONEAGAINSTONE_KERNEL_H__
#define __MULTICLASSCLASSIFIER_TRAIN_ONEAGAINSTONE_KERNEL_H__

#include "algorithms/multi_class_classifier/multi_class_classifier_model.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_train.h"

#include "src/algorithms/service_sort.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/multiclassclassifier/multiclassclassifier_train_kernel.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::data_management;

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
//Base class for binary classification subtask
template <typename algorithmFPType, CpuType cpu>
class SubTask
{
public:
    DAAL_NEW_DELETE();
    virtual ~SubTask() {}

    services::Status getDataSubset(size_t nFeatures, size_t nVectors, int classIdxPositive, int classIdxNegative, const algorithmFPType * y,
                                   size_t * originalIndicesMap, size_t & nRows)
    {
        nRows = 0;

        /* Prepare "positive" observations of the training subset */
        services::Status s = copyDataIntoSubtable(nFeatures, nVectors, classIdxPositive, 1, y, originalIndicesMap, nRows);
        if (s) /* Prepare "negative" observations of the training subset */
            s = copyDataIntoSubtable(nFeatures, nVectors, classIdxNegative, -1, y, originalIndicesMap, nRows);
        return s;
    }

    services::Status trainSimpleClassifier(size_t nRowsInSubset)
    {
        _subsetXTable->resize(nRowsInSubset);
        _subsetYTable->resize(nRowsInSubset);

        auto input = _simpleTraining->getInput();
        DAAL_CHECK(input, services::ErrorNullInput);
        input->set(classifier::training::data, _subsetXTable);
        input->set(classifier::training::labels, _subsetYTable);
        if (_weights)
        {
            _subsetWTable->resize(nRowsInSubset);
            input->set(classifier::training::weights, _subsetWTable);
        }
        services::Status s;
        DAAL_CHECK_STATUS(s, _simpleTraining->resetResult());
        return _simpleTraining->computeNoThrow();
    }

    classifier::ModelPtr getModel() { return _simpleTraining->getResult()->get(classifier::training::model); }

protected:
    typedef HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;

    SubTask(size_t nSubsetVectors, size_t dataSize, const algorithmFPType * weights,
            const services::SharedPtr<classifier::training::Batch> & training)
        : _subsetX(dataSize + nSubsetVectors), _subsetY(nullptr), _weights(weights)
    {
        services::Status status;
        if (!_subsetX.get()) return;
        _subsetY      = _subsetX.get() + dataSize;
        _subsetYTable = HomogenNT::create(_subsetY, 1, nSubsetVectors, &status);
        if (_weights)
        {
            _subsetW.reset(nSubsetVectors);
            _subsetWTable = HomogenNT::create(_subsetW.get(), 1, nSubsetVectors, &status);
        }

        if (!status) return;
        _simpleTraining = training->clone();
    }

    bool isValid() const { return _subsetX.get() && _subsetYTable.get() && _simpleTraining.get(); }

    virtual services::Status copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label, const algorithmFPType * y,
                                                  size_t * originalIndicesMap, size_t & nRows) = 0;

protected:
    TArray<algorithmFPType, cpu> _subsetX;
    algorithmFPType * _subsetY;
    const algorithmFPType * _weights;
    TArray<algorithmFPType, cpu> _subsetW;
    NumericTablePtr _subsetYTable;
    NumericTablePtr _subsetXTable;
    NumericTablePtr _subsetWTable;
    services::SharedPtr<classifier::training::Batch> _simpleTraining;
};

template <typename algorithmFPType, CpuType cpu>
class SubTaskCSR : public SubTask<algorithmFPType, cpu>
{
public:
    virtual ~SubTaskCSR() override {}

    typedef SubTask<algorithmFPType, cpu> super;
    static SubTaskCSR * create(size_t nFeatures, size_t nSubsetVectors, size_t dataSize, const NumericTable * xTable, const algorithmFPType * weights,
                               const services::SharedPtr<classifier::training::Batch> & st)
    {
        auto val = new SubTaskCSR(nFeatures, nSubsetVectors, dataSize, dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(xTable)),
                                  weights, st);
        if (val && val->isValid()) return val;
        delete val;
        val = nullptr;
        return nullptr;
    }

private:
    bool isValid() const { return super::isValid() && _colIndicesX.get() && this->_subsetXTable.get(); }

    SubTaskCSR(size_t nFeatures, size_t nSubsetVectors, size_t dataSize, CSRNumericTableIface * xTable, const algorithmFPType * weights,
               const services::SharedPtr<classifier::training::Batch> & st)
        : super(nSubsetVectors, dataSize, weights, st), _mtX(xTable), _colIndicesX(dataSize + nSubsetVectors + 1), _rowOffsetsX(nullptr)
    {
        if (_colIndicesX.get())
        {
            _rowOffsetsX = _colIndicesX.get() + dataSize;
            services::Status s;
            this->_subsetXTable = CSRNumericTable::create(this->_subsetX.get(), _colIndicesX.get(), _rowOffsetsX, nFeatures, 0,
                                                          CSRNumericTableIface::CSRIndexing::oneBased, &s);
            if (!s) return;
        }
    }

    virtual services::Status copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label, const algorithmFPType * y,
                                                  size_t * originalIndicesMap, size_t & nRows) override;

private:
    TArray<size_t, cpu> _colIndicesX;
    size_t * _rowOffsetsX;
    ReadRowsCSR<algorithmFPType, cpu> _mtX;
};

template <typename algorithmFPType, CpuType cpu>
class SubTaskDense : public SubTask<algorithmFPType, cpu>
{
public:
    virtual ~SubTaskDense() override {}

    typedef SubTask<algorithmFPType, cpu> super;
    static SubTaskDense * create(size_t nFeatures, size_t nSubsetVectors, size_t dataSize, const NumericTable * xTable,
                                 const algorithmFPType * weights, const services::SharedPtr<classifier::training::Batch> & st)
    {
        auto val = new SubTaskDense(nFeatures, nSubsetVectors, dataSize, xTable, weights, st);
        if (val && val->isValid()) return val;
        delete val;
        val = nullptr;
        return nullptr;
    }

private:
    typedef HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    bool isValid() const { return super::isValid() && this->_subsetXTable.get(); }

    SubTaskDense(size_t nFeatures, size_t nSubsetVectors, size_t dataSize, const NumericTable * xTable, const algorithmFPType * weights,
                 const services::SharedPtr<classifier::training::Batch> & st)
        : super(nSubsetVectors, dataSize, weights, st), _mtX(const_cast<NumericTable *>(xTable))
    {
        services::Status status;
        if (this->_subsetX.get()) this->_subsetXTable = HomogenNT::create(this->_subsetX.get(), nFeatures, nSubsetVectors, &status);
        if (!status) return;
    }

    virtual services::Status copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label, const algorithmFPType * y,
                                                  size_t * originalIndicesMap, size_t & nRows) override;

private:
    ReadRows<algorithmFPType, cpu> _mtX;
};

template <typename algorithmFPType, CpuType cpu>
class MultiClassClassifierTrainKernel<oneAgainstOne, algorithmFPType, cpu> : public Kernel
{
public:
    services::Status compute(const NumericTable * xTable, const NumericTable * yTable, const NumericTable * wTable, daal::algorithms::Model * m,
                             multi_class_classifier::internal::SvmModel * svmModel, const KernelParameter & par);

protected:
    services::Status computeDataSize(size_t nVectors, size_t nFeatures, size_t nClasses, const NumericTable * xTable, const algorithmFPType * y,
                                     size_t & nSubsetVectors, size_t & dataSize);
};

} // namespace internal
} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
