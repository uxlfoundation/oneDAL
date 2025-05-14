/* file: multi_class_classifier_model_builder.h */
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

#ifndef __MULTI_CLASS_CLASSIFIER_MODEL_BUILDER_H__
#define __MULTI_CLASS_CLASSIFIER_MODEL_BUILDER_H__

#include "algorithms/multi_class_classifier/multi_class_classifier_model.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_train_types.h"
namespace daal
{
namespace algorithms
{
/**
 * @defgroup multi_class_classifier Multi-class Classifier
 * \copydoc daal::algorithms::multi_class_classifier
 * @ingroup classification
 * @{
 */
/**
 * \brief Contains classes for computing the results of the multi-class classifier algorithm
 */
namespace multi_class_classifier
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup multi_class_classifier
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__MODEL_BUILDER"></a>
 * \brief Builder for Model of the classifier trained by the multi_class_classifier::training::Batch algorithm.
 *
 * \tparam method           Computation method for the algorithm, \ref daal::algorithms::multi_class_classifier::training::Method
 *
 */
template <training::Method method = training::oneAgainstOne>
class DAAL_EXPORT ModelBuilder
{
public:
    /**
     * Constructs the multi class classifier model builder
     * \param[in] nFeatures  Number of features in training data
     * \param[in] nClasses   Number of classes in training dataset
     */
    ModelBuilder(size_t nFeatures, size_t nClasses) : _nFeatures(nFeatures), _nClasses(nClasses)
    {
        _par = services::SharedPtr<ParameterBase>(new ParameterBase(_nClasses));
        if (_par.get())
        {
            _modelPtr = Model::create(_nFeatures, _par.get(), &_s);
        }
        else
        {
            _s = services::Status(services::ErrorMemoryAllocationFailed);
        }
    }

    /**
     * Set two-class classifier model into a multi-class classifier model
     * \param[in] negativeClassIdx Index of negative class for one vs one classification algorithm
     * \param[in] positiveClassIdx Index of positive class for one vs one classification algorithm
     * \param[in] model  Two-class classifier model to add into collection
     */
    void setTwoClassClassifierModel(size_t negativeClassIdx, size_t positiveClassIdx, const classifier::ModelPtr & model)
    {
        if (negativeClassIdx >= positiveClassIdx)
        {
            _s |= services::Status(services::ErrorIncorrectParameter);
        }

        if (!_s)
        {
            services::throwIfPossible(_s);
            return;
        }
        size_t imodel = positiveClassIdx * (positiveClassIdx - 1) / 2 + negativeClassIdx;

        _modelPtr->setTwoClassClassifierModel(imodel, model);
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

protected:
    ModelPtr _modelPtr;
    services::SharedPtr<ParameterBase> _par;
    services::Status _s;
    size_t _nFeatures;
    size_t _nClasses;
};

} // namespace interface1

using interface1::ModelBuilder;

} // namespace multi_class_classifier
/** @} */
} // namespace algorithms
} // namespace daal
#endif
