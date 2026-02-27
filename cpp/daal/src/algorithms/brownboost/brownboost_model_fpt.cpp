/* file: brownboost_model_fpt.cpp */
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
//  Implementation of class defining Brown Boost model.
//--
*/

#include "algorithms/boosting/brownboost_model.h"

namespace daal
{
namespace algorithms
{
namespace brownboost
{
/**
 *  Constructs the BrownBoost %Model
 * \tparam modelFPType  Data type to store BrownBoost model data, double or float
 * \param[in] dummy     Dummy variable for the templated constructor
 * \DAAL_DEPRECATED_USE{ Model::create }
 */
template <typename modelFPType>
DAAL_EXPORT Model::Model(size_t nFeatures, modelFPType dummy) : _nFeatures(nFeatures), _models(new data_management::DataCollection())
{
    _alpha = data_management::NumericTablePtr(new data_management::HomogenNumericTable<modelFPType>(NULL, 1, 0));
}

template <typename modelFPType>
DAAL_EXPORT Model::Model(size_t nFeatures, modelFPType dummy, services::Status & st)
    : _nFeatures(nFeatures), _models(new data_management::DataCollection())
{
    if (!_models)
    {
        st.add(services::ErrorMemoryAllocationFailed);
    }
    _alpha = data_management::HomogenNumericTable<modelFPType>::create(NULL, 1, 0, &st);
}

/**
 * Constructs the BrownBoost model
 * \tparam modelFPType  Data type to store BrownBoost model data, double or float
 * \param[in]  nFeatures Number of features in the dataset
 * \param[out] stat      Status of the model construction
 */
template <typename modelFPType>
DAAL_EXPORT ModelPtr Model::create(size_t nFeatures, services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, nFeatures, (modelFPType)0);
}

template DAAL_EXPORT Model::Model(size_t, DAAL_FPTYPE);
template DAAL_EXPORT Model::Model(size_t, DAAL_FPTYPE, services::Status &);
template DAAL_EXPORT ModelPtr Model::create<DAAL_FPTYPE>(size_t, services::Status *);

} // namespace brownboost
} // namespace algorithms
} // namespace daal
