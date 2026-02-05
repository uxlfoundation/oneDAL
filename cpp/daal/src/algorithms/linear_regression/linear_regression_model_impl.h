/* file: linear_regression_model_impl.h */
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
//  Declaration of the linear regression model class that implements the linear regression model
//--
*/

#ifndef __LINEAR_REGRESSION_MODEL_IMPL_H__
#define __LINEAR_REGRESSION_MODEL_IMPL_H__

#include "algorithms/linear_regression/linear_regression_model.h"
#include "src/algorithms/linear_model/linear_model_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::services;

class ModelInternal : public linear_model::internal::ModelInternal
{
public:
    typedef linear_model::internal::ModelInternal super;

    /**
     * Constructs the linear regression model
     * \param[in] featnum Number of features in the training data set
     * \param[in] nrhs    Number of responses in the training data
     * \param[in] par     Parameters of linear regression model-based training
     * \param[in] dummy   Dummy variable for the templated constructor
     */
    template <typename modelFPType>
    ModelInternal(size_t featnum, size_t nrhs, const linear_regression::Parameter & par, modelFPType dummy) : super(featnum, nrhs, par, dummy)
    {}

    ModelInternal() {}

    virtual ~ModelInternal() {}
};

class ModelImpl : public linear_regression::Model, public ModelInternal
{
public:
    typedef ModelInternal ImplType;

    /**
     * Constructs the linear regression model
     * \param[in] featnum Number of features in the training data set
     * \param[in] nrhs    Number of responses in the training data
     * \param[in] par     Parameters of linear regression model-based training
     * \param[in] dummy   Dummy variable for the templated constructor
     */
    template <typename modelFPType>
    ModelImpl(size_t featnum, size_t nrhs, const linear_regression::Parameter & par, modelFPType dummy) : ImplType(featnum, nrhs, par, dummy)
    {}

    ModelImpl() {}

    virtual ~ModelImpl() {}

    /**
    * Initializes the linear regression model
    */
    Status initialize() override { return ImplType::initialize(); }

    /**
     * Returns the number of regression coefficients
     * \return Number of regression coefficients
     */
    size_t getNumberOfBetas() const override { return ImplType::getNumberOfBetas(); }

    /**
     * Returns the number of responses in the training data set
     * \return Number of responses in the training data set
     */
    size_t getNumberOfResponses() const override { return ImplType::getNumberOfResponses(); }

    /**
     * Returns true if the regression model contains the intercept term, and false otherwise
     * \return True if the regression model contains the intercept term, and false otherwise
     */
    bool getInterceptFlag() const override { return ImplType::getInterceptFlag(); }

    /**
     * Returns the number of features in the training data set
     * \return Number of features in the training data set
     */
    size_t getNumberOfFeatures() const override { return ImplType::getNumberOfFeatures(); }

    /**
     * Returns the numeric table that contains regression coefficients
     * \return Table that contains regression coefficients
     */
    data_management::NumericTablePtr getBeta() override { return ImplType::getBeta(); }

    void setInterceptFlag(bool interceptFlag) { ImplType::_interceptFlag = interceptFlag; }

protected:
    services::Status serializeImpl(InputDataArchive * arch) override
    {
        ImplType::serialImpl<InputDataArchive, false>(arch);

        return services::Status();
    }

    services::Status deserializeImpl(const OutputDataArchive * arch) override
    {
        ImplType::serialImpl<const OutputDataArchive, true>(arch);

        return services::Status();
    }
};

} // namespace internal
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
#endif
