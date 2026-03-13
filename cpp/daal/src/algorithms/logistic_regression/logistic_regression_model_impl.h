/* file: logistic_regression_model_impl.h */
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
//  Implementation of the class defining the logistic regression model
//--
*/

#ifndef __LOGISTIC_REGRESSION_MODEL_IMPL__
#define __LOGISTIC_REGRESSION_MODEL_IMPL__

#include "algorithms/logistic_regression/logistic_regression_model.h"
#include "src/algorithms/classifier/classifier_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace internal
{
class ModelImpl : public logistic_regression::Model, public algorithms::classifier::internal::ModelInternal
{
public:
    typedef algorithms::classifier::internal::ModelInternal ClassificationImplType;

    ModelImpl(size_t nFeatures = 0, bool interceptFlag = true);
    template <typename modelFPType>
    ModelImpl(size_t nFeatures, bool interceptFlag, size_t nClasses, modelFPType dummy, services::Status * st);
    ~ModelImpl() {}

    virtual size_t getNumberOfFeatures() const override { return ClassificationImplType::getNumberOfFeatures(); }

    //Implementation of classification::Model
    virtual size_t getNumberOfBetas() const override;
    virtual bool getInterceptFlag() const override;
    virtual data_management::NumericTablePtr getBeta() override;
    virtual const data_management::NumericTablePtr getBeta() const override;

    virtual services::Status serializeImpl(data_management::InputDataArchive * arch) override;
    virtual services::Status deserializeImpl(const data_management::OutputDataArchive * arch) override;

    services::Status reset(bool interceptFlag);
    static logistic_regression::ModelPtr create(size_t nFeatures, bool interceptFlag, services::Status * stat = nullptr);

protected:
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        auto st = classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        if (!st) return st;
        arch->set(ClassificationImplType::_nFeatures);
        arch->set(_interceptFlag);
        arch->setSharedPtrObj(_beta);
        return services::Status();
    }

protected:
    bool _interceptFlag;                    // True if the model contains the intercept term false otherwise
    data_management::NumericTablePtr _beta; // Model coefficients
};

} // namespace internal
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal

#endif
