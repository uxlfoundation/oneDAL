/* file: logistic_regression_model_builder.h */
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
//  Implementation of the class defining the logistic regression model builder
//--
*/

#ifndef __LOGISTIC_REGRESSION_MODEL_BUILDER_H__
#define __LOGISTIC_REGRESSION_MODEL_BUILDER_H__

#include "algorithms/logistic_regression/logistic_regression_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup logistic_regression Logistic regression
 * \copydoc daal::algorithms::logistic_regression
 * @ingroup classification
 */
/**
 * \brief Contains classes for the logistic regression algorithm
 */
namespace logistic_regression
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup logistic_regression
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGISTIC_REGRESSION__MODEL__BUILDER"></a>
 * \brief %Class for building model of the logistic regression algorithm
 *
 * \tparam modelFPType  Data type to store logistic regression model data, double or float
 *
 */
template <typename modelFPType = DAAL_ALGORITHM_FP_TYPE>
class DAAL_EXPORT ModelBuilder
{
public:
    /**
     * Constructs the Logistic Regression model builder
     * \param[in] nFeatures      Number of features in training data
     * \param[in] nClasses       Number of classes in training data
     */
    ModelBuilder(size_t nFeatures, size_t nClasses);

    /**
     * Empty constructor for deserialization
     **/
    ModelBuilder();

    /**
     *  Method to set betas to model via NumericTablePtr, size of NumericTable have to be equal to (_nFeatures)*_nClasses
     *  in case when intercept flag is suppose to be false and (_nFeatures + 1)*_nClasses when intercept flag is true
     *  \param[in] beta       NumericTablePtr represent support vectors
     */
    void setBeta(const data_management::NumericTablePtr & beta)
    {
        data_management::BlockDescriptor<modelFPType> resBeta, inBeta;
        const size_t nVectorsBeta = _nClasses == 2 ? 1 : _nClasses;

        if (beta->getNumberOfColumns() == _nFeatures)
        {
            setInterceptFlag(false);
            for (size_t i = 0; i < nVectorsBeta; ++i)
            {
                _modelPtr->getBeta()->getBlockOfRows(i, 1, data_management::writeOnly, resBeta);
                beta->getBlockOfRows(i, 1, data_management::readOnly, inBeta);
                modelFPType * const resBetaData      = resBeta.getBlockPtr();
                const modelFPType * const inBetaData = inBeta.getBlockPtr();
                resBetaData[0]                       = modelFPType(0);
                for (size_t j = 0; j < _nFeatures; ++j)
                {
                    resBetaData[j + 1] = inBetaData[j];
                }
                _modelPtr->getBeta()->releaseBlockOfRows(resBeta);
                beta->releaseBlockOfRows(inBeta);
            }
        }
        else if (beta->getNumberOfColumns() == (_nFeatures + 1))
        {
            setInterceptFlag(true);
            _modelPtr->getBeta()->getBlockOfRows(0, nVectorsBeta, data_management::writeOnly, resBeta);
            beta->getBlockOfRows(0, nVectorsBeta, data_management::readOnly, inBeta);
            modelFPType * const resBetaData      = resBeta.getBlockPtr();
            const modelFPType * const inBetaData = inBeta.getBlockPtr();
            const size_t betaSize                = beta->getNumberOfColumns() * beta->getNumberOfRows();
            for (size_t i = 0; i < betaSize; ++i)
            {
                resBetaData[i] = inBetaData[i];
            }
            _modelPtr->getBeta()->releaseBlockOfRows(resBeta);
            beta->releaseBlockOfRows(inBeta);
        }
        else
        {
            _s = services::Status(services::ErrorIncorrectParameter);
            services::throwIfPossible(_s);
        }
    }

    /**
     *  Method to set betas to model via random access iterator, last - first value have to be equal to (_nFeatures)*_nClasses
     *  in case when intercept flag is suppose to be false and (_nFeatures + 1)*_nClasses when intercept flag is true
     * \tparam RandomIterator Random access iterator type for access to values of support vectors
     *  \param[in] first      Iterator which point to first element of support vectors
     *  \param[in] last       Iterator which point to last element of support vectors
     */
    template <typename RandomIterator>
    void setBeta(RandomIterator first, RandomIterator last)
    {
        data_management::BlockDescriptor<modelFPType> pBlock;
        const size_t nVectorsBeta = _nClasses == 2 ? 1 : _nClasses;
        _modelPtr->getBeta()->getBlockOfRows(0, nVectorsBeta, data_management::readWrite, pBlock);
        modelFPType * sp = pBlock.getBlockPtr();
        if (((size_t)(last - first) == ((_nFeatures)*nVectorsBeta)) && (last > first))
        {
            setInterceptFlag(false);
            size_t i = 0;
            while (first != last)
            {
                if ((i % (_nFeatures + 1)) == 0)
                {
                    sp[i] = 0;
                    ++i;
                }
                sp[i] = *first;
                ++first;
                ++i;
            }
        }
        else if (((size_t)(last - first) == ((_nFeatures + 1) * nVectorsBeta)) && (last > first))
        {
            setInterceptFlag(true);
            while (first != last)
            {
                *sp = *first;
                ++first;
                ++sp;
            }
        }
        else
        {
            _s = services::Status(services::ErrorIncorrectParameter);
            _modelPtr->getBeta()->releaseBlockOfRows(pBlock);
            services::throwIfPossible(_s);
            return;
        }
        _modelPtr->getBeta()->releaseBlockOfRows(pBlock);
    }

    /**
     *  Get built model
     *  \return Model pointer
     */
    ModelPtr getModel() { return _modelPtr; }

    /**
     *  Get status of model building
     *  \return Status
     */
    services::Status getStatus() { return _s; }

private:
    ModelPtr _modelPtr;
    services::Status _s;
    size_t _nFeatures;
    size_t _nClasses;

    void setInterceptFlag(bool interceptFlag);
};

/** @} */
} // namespace interface1
using interface1::ModelBuilder;

} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
#endif
