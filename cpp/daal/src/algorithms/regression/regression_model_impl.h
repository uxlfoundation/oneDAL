/* file: regression_model_impl.h */
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
//  Implementation of the class defining the regression model
//--
*/

#ifndef __REGRESSION_MODEL_IMPL_H__
#define __REGRESSION_MODEL_IMPL_H__

#include "algorithms/regression/regression_model.h"

namespace daal
{
namespace algorithms
{
namespace regression
{
namespace internal
{
class ModelInternal
{
public:
    ModelInternal(size_t nFeatures = 0) : _nFeatures(nFeatures) {}

    void setNumberOfFeatures(size_t nFeatures) { _nFeatures = nFeatures; }
    virtual size_t getNumberOfFeatures() const { return _nFeatures; }

protected:
    size_t _nFeatures;

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->set(_nFeatures);

        return services::Status();
    }
};

class ModelImpl : public regression::Model, public ModelInternal
{
public:
    typedef ModelInternal ImplType;

    ModelImpl(size_t nFeatures = 0) : ImplType(nFeatures) {}

    void setNumberOfFeatures(size_t nFeatures) { ImplType::setNumberOfFeatures(nFeatures); }
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return ImplType::getNumberOfFeatures(); }

protected:
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        regression::Model::serialImpl<Archive, onDeserialize>(arch);
        arch->set(ImplType::_nFeatures);

        return services::Status();
    }
};

} // namespace internal
} // namespace regression
} // namespace algorithms
} // namespace daal

#endif
