/* file: linear_model_model_impl.h */
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
//  Declaration of the class that implements the linear model
//--
*/

#ifndef __LINEAR_MODEL_MODEL_IMPL_H__
#define __LINEAR_MODEL_MODEL_IMPL_H__

#include "algorithms/linear_model/linear_model_model.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace internal
{
using namespace data_management;

/**
 * \brief Class that implements the methods for models trained with the linear model algorithm
 */
class ModelInternal
{
public:
    /**
     * Constructs the linear model
     * \param[in] nFeatures  Number of features in the training data
     * \param[in] nResponses Number of responses in the training data
     * \param[in] par        Parameters of the linear model
     * \param[in] dummy      Dummy variable for the templated constructor
     */
    template <typename modelFPType>
    ModelInternal(size_t nFeatures, size_t nResponses, const linear_model::Parameter & par, modelFPType dummy);

    /**
     * Constructs the linear model
     * \param[in] beta  Numeric table that contains the linear model coefficients
     * \param[in] par   Parameters of the linear model
     */
    ModelInternal(const data_management::NumericTablePtr & beta, const linear_model::Parameter & par = Parameter());

    ModelInternal();

    virtual ~ModelInternal() {}

    /**
     * Initializes the coefficients of the linear model
     */
    services::Status initialize();

    /**
     * Returns the number of regression coefficients
     * \return Number of regression coefficients
     */
    size_t getNumberOfBetas() const;

    /**
     * Returns the number of responses in the training data set
     * \return Number of responses in the training data set
     */
    size_t getNumberOfResponses() const;

    /**
     * Returns true if the regression model contains the intercept term, and false otherwise
     * \return True if the regression model contains the intercept term, and false otherwise
     */
    bool getInterceptFlag() const;

    /**
     * Returns the number of features in the training data set
     * \return Number of features in the training data set
     */
    size_t getNumberOfFeatures() const;
    /**
     * Returns the numeric table that contains regression coefficients
     * \return Table that contains regression coefficients
     */
    data_management::NumericTablePtr getBeta();

protected:
    bool _interceptFlag;                    /* Flag. True if the ridge regression model contains the intercept term;
                                false otherwise. */
    data_management::NumericTablePtr _beta; /* Table that contains resulting coefficients */

    services::Status setToZero(data_management::NumericTable & table);

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->set(_interceptFlag);

        arch->setSharedPtrObj(_beta);

        return services::Status();
    }
};

/**
 * \brief Class that connects the interface and implementation
 */
class ModelImpl : public linear_model::Model, public ModelInternal
{
public:
    typedef ModelInternal ImplType;

    /**
     * Constructs the linear model
     * \param[in] nFeatures  Number of features in the training data
     * \param[in] nResponses Number of responses in the training data
     * \param[in] par        Parameters of the linear model
     * \param[in] dummy      Dummy variable for the templated constructor
     */
    template <typename modelFPType>
    ModelImpl(size_t nFeatures, size_t nResponses, const linear_model::Parameter & par, modelFPType dummy)
        : ImplType(nFeatures, nResponses, par, dummy)
    {}

    /**
     * Constructs the linear model
     * \param[in] beta  Numeric table that contains the linear model coefficients
     * \param[in] par   Parameters of the linear model
     */
    ModelImpl(const data_management::NumericTablePtr & beta, const linear_model::Parameter & par = Parameter()) : ImplType(beta, par) {}

    /**
     * Initializes the coefficients of the linear model
     */
    services::Status initialize() override { return ImplType::initialize(); }

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
};

} // namespace internal
} // namespace linear_model
} // namespace algorithms
} // namespace daal
#endif
