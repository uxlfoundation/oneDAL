/* file: multiclassclassifier_svm_model.h */
/*******************************************************************************
* Copyright 2021 Intel Corporation
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
//  Implementation of the class defining the SVM model.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_SVM_MODEL_H__
#define __MULTICLASSCLASSIFIER_SVM_MODEL_H__

#include "src/algorithms/svm/svm_train_result.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace internal
{
class DAAL_EXPORT SvmModel : public classifier::Model
{
public:
    template <typename modelFPType>
    DAAL_EXPORT static services::SharedPtr<SvmModel> create(
        const size_t nClasses, const size_t nColumns,
        const data_management::NumericTableIface::StorageLayout layout = data_management::NumericTableIface::aos, services::Status * stat = NULL)

    {
        DAAL_DEFAULT_CREATE_IMPL_EX(SvmModel, (modelFPType)0.0, nClasses, nColumns, layout);
    }

    virtual ~SvmModel() {}

    data_management::NumericTablePtr getSupportVectors() { return _SV; }

    data_management::NumericTablePtr getSupportIndices() { return _SVIndices; }

    data_management::NumericTablePtr getCoefficients() { return _SVCoeff; }

    data_management::NumericTablePtr getBiases() { return _biases; }

    void setSupportVectors(data_management::NumericTablePtr sv) { _SV = sv; }
    void setSupportIndices(data_management::NumericTablePtr svIndices) { _SVIndices = svIndices; }
    void setCoefficients(data_management::NumericTablePtr coeffs) { _SVCoeff = coeffs; }
    void setBiases(data_management::NumericTablePtr biases) { _biases = biases; }

    size_t getNumberOfFeatures() const override { return (_SV ? _SV->getNumberOfColumns() : 0); }

protected:
    data_management::NumericTablePtr _SV;
    data_management::NumericTablePtr _SVCoeff;
    data_management::NumericTablePtr _biases;
    data_management::NumericTablePtr _SVIndices;

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        services::Status st = classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        if (!st) return st;
        arch->setSharedPtrObj(_SV);
        arch->setSharedPtrObj(_SVCoeff);
        arch->setSharedPtrObj(_biases);
        arch->setSharedPtrObj(_SVIndices);
        return st;
    }

    template <typename modelFPType>
    SvmModel(modelFPType dummy, const size_t nClasses, const size_t nColumns, const data_management::NumericTableIface::StorageLayout layout,
             services::Status & st)
    {
        using namespace data_management;
        const size_t nModels = (nClasses * (nClasses - 1)) >> 1;
        if (layout == NumericTableIface::csrArray)
        {
            _SV = CSRNumericTable::create<modelFPType>(nullptr, nullptr, nullptr, nColumns, size_t(0), CSRNumericTable::oneBased, &st);
        }
        else
        {
            _SV = HomogenNumericTable<modelFPType>::create(nColumns, 0, NumericTable::doNotAllocate, &st);
        }
        if (!st) return;
        _SVCoeff = HomogenNumericTable<modelFPType>::create(nClasses - 1, 0, NumericTable::doNotAllocate, &st);
        if (!st) return;
        _SVIndices = HomogenNumericTable<int>::create(1, 0, NumericTable::doNotAllocate, &st);
        if (!st) return;
        _biases = HomogenNumericTable<modelFPType>::create(1, nModels, NumericTable::doAllocate, &st);
    }
};

using SvmModelPtr = services::SharedPtr<SvmModel>;

} // namespace internal
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
