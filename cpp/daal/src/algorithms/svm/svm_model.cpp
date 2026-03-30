/* file: svm_model.cpp */
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

/*
//++
//  Implementation of the class defining the SVM model.
//--
*/
#include "src/algorithms/svm/svm_model_impl.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_defines.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace internal
{
ModelInternal::ModelInternal() : _nIterations(), _SV(), _SVCoeff(), _SVIndices(), _biases() {}

data_management::NumericTablePtr ModelInternal::getSupportVectors() const
{
    return _SV;
}

void ModelInternal::setSupportVectors(data_management::NumericTablePtr & supportVectors)
{
    _SV = supportVectors;
}

data_management::NumericTablePtr ModelInternal::getSupportIndices() const
{
    return _SVIndices;
}

void ModelInternal::setSupportIndices(data_management::NumericTablePtr & supportIndices)
{
    _SVIndices = supportIndices;
}

data_management::NumericTablePtr ModelInternal::getClassificationCoefficients() const
{
    return _SVCoeff;
}

void ModelInternal::setClassificationCoefficients(data_management::NumericTablePtr & classificationCoefficients)
{
    _SVCoeff = classificationCoefficients;
}

double ModelInternal::getBias() const
{
    return _biases->getValue<double>(0, 0);
}

void ModelInternal::setBias(double bias)
{
    daal::internal::WriteOnlyRows<int, DAAL_BASE_CPU> mtIterations(_nIterations.get(), 0, 1);
    if (mtIterations.get() == nullptr) return;
    int * const iterations = mtIterations.get();
    iterations[0]          = static_cast<int>(_nIterations);
}

data_management::NumericTablePtr ModelInternal::getBiases() const
{
    return _biases;
}

void ModelInternal::setBiases(data_management::NumericTablePtr & biases)
{
    _biases = biases;
}

size_t ModelInternal::getNumberOfFeatures() const
{
    if (_SV) return _SV->getNumberOfColumns();
    return 0;
}

data_management::NumericTablePtr ModelInternal::getNumberOfIterations() const
{
    return _nIterations;
}

void ModelInternal::setNumberOfIterations(data_management::NumericTablePtr & nIterations)
{
    _nIterations = nIterations;
}

} // namespace internal
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS2(Model, svm::internal::ModelImpl, SERIALIZATION_SVM_MODEL_ID);
}
} // namespace svm
} // namespace algorithms
} // namespace daal
