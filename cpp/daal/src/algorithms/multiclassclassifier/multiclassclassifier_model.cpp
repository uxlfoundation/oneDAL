/* file: multiclassclassifier_model.cpp */
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
//  Implementation of multi class classifier model.
//--
*/

#include "algorithms/multi_class_classifier/multi_class_classifier_model.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_MULTI_CLASS_CLASSIFIER_MODEL_ID);

Model::Model(size_t nFeatures, const ParameterBase * par)
    : _modelsArray(nullptr), _models(new data_management::DataCollection(par->nClasses * (par->nClasses - 1) / 2)), _nFeatures(nFeatures)
{}

Model::Model(size_t nFeatures, const multi_class_classifier::interface2::ParameterBase * par, services::Status & st)
    : _modelsArray(nullptr), _nFeatures(nFeatures)
{
    _models.reset(new data_management::DataCollection(par->nClasses * (par->nClasses - 1) / 2));
    if (!_models) st.add(services::ErrorMemoryAllocationFailed);
}

Model::Model(const Model & other) :
    _nFeatures(other._nFeatures),
    _models(new data_management::DataCollection(other.getNumberOfTwoClassClassifierModels())),
    _modelsArray(nullptr)
{
    const size_t nModels = other.getNumberOfTwoClassClassifierModels();
    for (size_t i = 0; i < nModels; ++i)
    {
        setTwoClassClassifierModel(i, other.getTwoClassClassifierModel(i));
    }
}

Model & Model::operator=(const Model & other)
{
    if (this != &other)
    {
        _nFeatures = other._nFeatures;
        const size_t nModels = other.getNumberOfTwoClassClassifierModels();
        _models = data_management::DataCollectionPtr(new data_management::DataCollection(nModels));
        for (size_t i = 0; i < nModels; ++i)
        {
            setTwoClassClassifierModel(i, other.getTwoClassClassifierModel(i));
        }
    }
    return *this;
}

Model::Model() : _nFeatures(0), _models(new data_management::DataCollection()), _modelsArray(nullptr) {}

ModelPtr Model::create(size_t nFeatures, const multi_class_classifier::interface2::ParameterBase * par, services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, nFeatures, par);
}

Model::~Model()
{
    delete[] _modelsArray;
}

classifier::ModelPtr * Model::getTwoClassClassifierModels()
{
    if (!_modelsArray)
    {
        _modelsArray = new classifier::ModelPtr[_models->size()];
        if (!_modelsArray) return nullptr;
        for (size_t i = 0; i < _models->size(); i++)
        {
            _modelsArray[i] = services::staticPointerCast<classifier::Model, data_management::SerializationIface>((*_models)[i]);
        }
    }
    return _modelsArray;
}

void Model::setTwoClassClassifierModel(size_t idx, const classifier::ModelPtr & model)
{
    (*_models)[idx] = model;
}

classifier::ModelPtr Model::getTwoClassClassifierModel(size_t idx) const
{
    if (idx < _models->size())
    {
        return services::staticPointerCast<classifier::Model, data_management::SerializationIface>((*_models)[idx]);
    }
    return classifier::ModelPtr();
}

} // namespace interface1
namespace interface2
{
services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, ParameterBase::check());
    DAAL_CHECK_EX((accuracyThreshold > 0) && (accuracyThreshold < 1), services::ErrorIncorrectParameter, services::ParameterName,
                  accuracyThresholdStr());
    DAAL_CHECK_EX(maxIterations, services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr());
    return s;
}
} // namespace interface2
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
