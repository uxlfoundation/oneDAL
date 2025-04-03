/* file: svm_train_types.h */
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
//  SVM parameter structure
//--
*/

#ifndef __SVM_TRAIN_TYPES_H__
#define __SVM_TRAIN_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/kernel_function/kernel_function.h"
#include "algorithms/svm/svm_model.h"
#include "algorithms/classifier/classifier_training_types.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
/**
 * @defgroup svm_training Training
 * \copydoc daal::algorithms::svm::training
 * @ingroup svm
 * @{
 */
/**
 * \brief Contains classes to train the SVM model
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__SVM__TRAINING__METHOD"></a>
 * Available methods to train the SVM model
 */
enum Method
{
    defaultDense = 0, /*!< Default method.                 */
    boser        = 0, /*!< Method proposed by Boser et al. */
    thunder      = 1  /*!< Method proposed by ThunderSVM. */
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the
 *        SVM training algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    virtual ~Result() {}

    /**
     * Returns the model trained with the SVM algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the SVM algorithm
     */
    daal::algorithms::svm::ModelPtr get(classifier::training::ResultId id) const;

    /**
     * Allocates memory for storing SVM training results
     * \param[in] input     Pointer to input structure
     * \param[in] parameter Pointer to parameter structure
     * \param[in] method    Algorithm method
     *
     * \return Status of computation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface1
using interface1::Result;
using interface1::ResultPtr;

} // namespace training
/** @} */
} // namespace svm
} // namespace algorithms
} // namespace daal
#endif
