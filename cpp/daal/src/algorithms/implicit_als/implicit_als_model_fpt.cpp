/* file: implicit_als_model_fpt.cpp */
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
//  Implementation of the class defining the implicit als model
//--
*/

#include "algorithms/implicit_als/implicit_als_model.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
template <typename modelFPType>
DAAL_EXPORT Model::Model(size_t nUsers, size_t nItems, const Parameter & parameter, modelFPType dummy)
{
    const size_t nFactors = parameter.nFactors;
    _usersFactors.reset(new data_management::HomogenNumericTable<modelFPType>(nFactors, nUsers, data_management::NumericTableIface::doAllocate, 0));
    _itemsFactors.reset(new data_management::HomogenNumericTable<modelFPType>(nFactors, nItems, data_management::NumericTableIface::doAllocate, 0));
}

template <typename modelFPType>
DAAL_EXPORT Model::Model(size_t nUsers, size_t nItems, const Parameter & parameter, modelFPType dummy, services::Status & st)
{
    using namespace daal::data_management;
    const size_t nFactors = parameter.nFactors;

    _usersFactors = HomogenNumericTable<modelFPType>::create(nFactors, nUsers, NumericTableIface::doAllocate, 0, &st);
    if (!st)
    {
        return;
    }

    _itemsFactors = HomogenNumericTable<modelFPType>::create(nFactors, nItems, NumericTableIface::doAllocate, 0, &st);
    if (!st)
    {
        return;
    }
}

/**
 * Constructs the implicit ALS model
 * \param[in]  nUsers    Number of users in the input data set
 * \param[in]  nItems    Number of items in the input data set
 * \param[in]  parameter Implicit ALS parameters
 * \param[out] stat      Status of the model construction
 */
template <typename modelFPType>
DAAL_EXPORT ModelPtr Model::create(size_t nUsers, size_t nItems, const Parameter & parameter, services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, nUsers, nItems, parameter, (modelFPType)0 /* dummy */);
}

template DAAL_EXPORT Model::Model(size_t, size_t, const Parameter &, DAAL_FPTYPE);
template DAAL_EXPORT Model::Model(size_t, size_t, const Parameter &, DAAL_FPTYPE, services::Status &);
template DAAL_EXPORT ModelPtr Model::create<DAAL_FPTYPE>(size_t, size_t, const Parameter &, services::Status *);

} // namespace implicit_als
} // namespace algorithms
} // namespace daal
