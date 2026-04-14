/* file: svm_model_fpt.cpp */
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
//  Implementation of the constructor of the class defining the SVM model.
//--
*/

#include "src/algorithms/svm/svm_model_impl.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/csr_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace internal
{
namespace dm = daal::data_management;

template <typename modelFPType>
ModelInternal::ModelInternal(modelFPType dummy, size_t nClasses, size_t nColumns, data_management::NumericTableIface::StorageLayout layout,
                             services::Status & st)
{
    if (nClasses < 2)
    {
        st.add(services::ErrorIncorrectNumberOfClasses);
        return;
    }
    if (layout == dm::NumericTableIface::csrArray)
    {
        _SV = dm::CSRNumericTable::create<modelFPType>(NULL, NULL, NULL, nColumns, 0, dm::CSRNumericTable::oneBased, &st);
    }
    else
    {
        _SV = dm::HomogenNumericTable<modelFPType>::create(NULL, nColumns, 0, &st);
    }
    if (!st) return;
    _SVCoeff = dm::HomogenNumericTable<modelFPType>::create(NULL, nClasses - 1, 0, &st);
    if (!st) return;
    _SVIndices = dm::HomogenNumericTable<int>::create(NULL, 1, 0, &st);
    if (!st) return;
    bool overflow = false;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_BOOL(size_t, nClasses, (nClasses - 1) / 2, overflow);
    if (overflow)
    {
        st.add(services::ErrorBufferSizeIntegerOverflow);
        return;
    }
    const size_t nModels = nClasses * (nClasses - 1) / 2;
    _biases              = dm::HomogenNumericTable<double>::create(1, nModels, dm::NumericTable::doAllocate, &st);
    if (!st) return;
    _nIterations = dm::HomogenNumericTable<int>::create(1, nModels, dm::NumericTable::doAllocate, &st);

    return;
}

template <typename modelFPType>
ModelInternal::ModelInternal(modelFPType dummy, size_t nColumns, data_management::NumericTableIface::StorageLayout layout, services::Status & st)
    : ModelInternal(dummy, 2, nColumns, layout, st)
{}

template DAAL_EXPORT ModelInternal::ModelInternal(DAAL_FPTYPE, size_t, size_t, dm::NumericTableIface::StorageLayout, services::Status &);
template DAAL_EXPORT ModelInternal::ModelInternal(DAAL_FPTYPE, size_t, dm::NumericTableIface::StorageLayout, services::Status &);

} // namespace internal
} // namespace svm
} // namespace algorithms
} // namespace daal
