/* file: svm_model_builder.h */
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
//  Implementation of the class defining the SVM model builder.
//--
*/

#ifndef __SVM_MODEL_BUILDER_H__
#define __SVM_MODEL_BUILDER_H__

#include "algorithms/svm/svm_model.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup svm Support Vector Machine Classifier
 * \copydoc daal::algorithms::svm
 * @ingroup classification
 * @{
 */
/**
 * \brief Contains classes to work with the support vector machine classifier
 */
namespace svm
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup svm
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__MODEL_BUILDER"></a>
 * \brief %Model Builder class for class SVM Model.
 *
 * \par References
 *      - \ref interface1::Model "Model" class
 */
template <typename modelFPType = DAAL_ALGORITHM_FP_TYPE>
class DAAL_EXPORT ModelBuilder
{
public:
    /**
     * Constructs the SVM model builder
     * \tparam modelFPType         Data type to store SVM model data, double or float
     * \param[in] nFeatures        Number of features in training data
     * \param[in] nSupportVectors  Number of support vectors in model
     */
    ModelBuilder(size_t nFeatures, size_t nSupportVectors)
        : _modelPtr(Model::create<modelFPType>(nFeatures)), _nFeatures(nFeatures), _nSupportVectors(nSupportVectors)
    {
        _supportV  = _modelPtr->getSupportVectors();
        _supportI  = _modelPtr->getSupportIndices();
        _supportCC = _modelPtr->getClassificationCoefficients();
        _supportV->resize(nSupportVectors);
        _supportI->resize(nSupportVectors);
        _supportCC->resize(nSupportVectors);
    }

    /**
     *  Method to set support vectors to model via random access iterator
     * \tparam RandomIterator       Random access iterator type for access to values of support vectors
     *  \param[in] first            Iterator which point to first element of support vectors
     *  \param[in] last             Iterator which point to last element of support vectors
     */
    template <typename RandomIterator>
    void setSupportVectors(RandomIterator first, RandomIterator last)
    {
        if (((size_t)(last - first) != _nSupportVectors * _nFeatures) || (last < first))
        {
            services::throwIfPossible(services::Status(services::ErrorIncorrectParameter));
        }
        commonSetter<RandomIterator>(_supportV, first, last);
    }

    /**
     *  Method to set support indices to model via random access iterator
     * \tparam RandomIterator       Random access iterator type for access to values of support indices
     *  \param[in] first            Iterator which point to first element of support indices
     *  \param[in] last             Iterator which point to last element of support indices
     */
    template <typename RandomIterator>
    void setSupportIndices(RandomIterator first, RandomIterator last)
    {
        if (((size_t)(last - first) != _nSupportVectors) || (last < first))
        {
            services::throwIfPossible(services::Status(services::ErrorIncorrectParameter));
        }
        commonSetter<RandomIterator>(_supportI, first, last);
    }

    /**
     *  Method to set classification coefficients to model via random access iterator
     * \tparam RandomIterator       Random access iterator type for access to values of classification coefficients
     *  \param[in] first            Iterator which point to first element of classification coefficients
     *  \param[in] last             Iterator which point to last element of classification coefficients
     */
    template <typename RandomIterator>
    void setClassificationCoefficients(RandomIterator first, RandomIterator last)
    {
        if (((size_t)(last - first) != _nSupportVectors) || (last < first))
        {
            services::throwIfPossible(services::Status(services::ErrorIncorrectParameter));
        }
        commonSetter<RandomIterator>(_supportCC, first, last);
    }

    /**
     *  Method to set bias term to model
     *  \param[in] bias The value to be set
     */
    void setBias(modelFPType bias) { _modelPtr->setBias(bias); }

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
    size_t _nSupportVectors;
    data_management::NumericTablePtr _supportV;  /*!< \private Support vectors */
    data_management::NumericTablePtr _supportCC; /*!< \private Classification coefficients */
    data_management::NumericTablePtr _supportI;  /*!< \private Indices of the support vectors in training data set */

    template <typename RandomIterator>
    services::Status commonSetter(data_management::NumericTablePtr & p, RandomIterator first, RandomIterator last)
    {
        services::Status s;

        data_management::BlockDescriptor<modelFPType> pBlock;
        p->getBlockOfRows(0, _nSupportVectors, data_management::readWrite, pBlock);
        modelFPType * sp = pBlock.getBlockPtr();
        while (first != last)
        {
            *sp = *first;
            ++first;
            ++sp;
        }
        p->releaseBlockOfRows(pBlock);
        return s;
    }
};

/** @} */
} // namespace interface1
using interface1::ModelBuilder;

} // namespace svm
} // namespace algorithms
} // namespace daal
#endif
