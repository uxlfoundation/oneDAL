/* file: svm_model_builder_fpt.cpp */
/*******************************************************************************
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

#include "algorithms/svm/svm_model_builder.h"
#include "src/algorithms/svm/svm_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace interface1
{
template <typename modelFPType>
ModelBuilder<modelFPType>::ModelBuilder(size_t nFeatures, size_t nSupportVectors)
        : _modelPtr(new svm::internal::ModelImpl(modelFPType(0), 2, nFeatures, data_management::NumericTable::soa, _s)), _nFeatures(nFeatures), _nSupportVectors(nSupportVectors)
{
    _supportV  = _modelPtr->getSupportVectors();
    _supportI  = _modelPtr->getSupportIndices();
    _supportCC = _modelPtr->getClassificationCoefficients();
    _supportV->resize(nSupportVectors);
    _supportI->resize(nSupportVectors);
    _supportCC->resize(nSupportVectors);
}

template <typename modelFPType>
void ModelBuilder<modelFPType>::setBias(modelFPType bias)
{
    _modelPtr->setBias(bias);
}

template DAAL_EXPORT ModelBuilder<DAAL_FPTYPE>::ModelBuilder(size_t, size_t);

} // namespace interface1
} // namespace svm
} // namespace algorithms
} // namespace daal
