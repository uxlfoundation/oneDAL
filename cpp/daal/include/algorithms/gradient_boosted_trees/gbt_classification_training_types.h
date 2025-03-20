/* file: gbt_classification_training_types.h */
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
//  Implementation of gradient boosted trees classification training algorithm interface.
//--
*/

#ifndef __GBT_CLASSIFICATION_TRAINING_TYPES_H__
#define __GBT_CLASSIFICATION_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/gradient_boosted_trees/gbt_classification_model.h"
#include "algorithms/gradient_boosted_trees/gbt_training_parameter.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
/**
 * @defgroup gbt_classification_training Training
 * \copydoc daal::algorithms::gbt::classification::training
 * @ingroup gbt_classification
 * @{
 */
/**
 * \brief Contains classes for Gradient Boosted Trees models training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__CLASSIFICATION__TRAINING__METHOD"></a>
 * \brief Computation methods for gradient boosted trees classification model-based training
 */
enum Method
{
    xboost       = 0, /*!< Extreme boosting (second-order approximation of objective function,
                           regularization on number of leaves and their weights), Chen et al. */
    defaultDense = 0  /*!< Default training method */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__GBT__CLASSIFICATION__TRAINING__LOSS_FUNCTION_TYPE"></a>
* \brief Loss function type
*/
enum LossFunctionType
{
    crossEntropy, /* Multinomial deviance */
    custom        /* custom function type */
};

enum ResultNumericTableId
{
    variableImportanceByWeight = classifier::training::lastResultId + 1,
    variableImportanceByTotalCover,
    variableImportanceByCover,
    variableImportanceByTotalGain,
    variableImportanceByGain,
    lastResultNumericTableId = variableImportanceByGain
};

/**
 * \brief Contains version 2.0 of oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__GBT__CLASSIFICATION__TRAINING__PARAMETER"></a>
 * \brief Gradient Boosted Trees algorithm parameters
 *
 * \snippet gradient_boosted_trees/gbt_classification_training_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::Parameter, public daal::algorithms::gbt::training::Parameter
{
    /** Default constructor */
    Parameter(size_t nClasses) : classifier::Parameter(nClasses), loss(crossEntropy), varImportance(0) {}
    services::Status check() const DAAL_C11_OVERRIDE;
    LossFunctionType loss;     /*!< Loss function type */
    DAAL_UINT64 varImportance; /*!< 64 bit integer flag VariableImportanceModes that indicates the variable importance computation modes */
};
/* [Parameter source code] */
} // namespace interface2

namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__CLASSIFICATION__TRAINING__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of model-based training
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)

    Result();
    virtual ~Result() {}

    /**
     * Returns the model trained with the LogitBoost algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the LogitBoost algorithm
     */
    ModelPtr get(classifier::training::ResultId id) const;

    /**
    * Sets the result of model-based training
    * \param[in] id      Identifier of the result
    * \param[in] value   Result
    */
    void set(classifier::training::ResultId id, const ModelPtr & value);

    /**
     * Returns the result of model-based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultNumericTableId id) const;

    /**
     * Sets the result of model-based training
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultNumericTableId id, const data_management::NumericTablePtr & value);

    /**
     * Allocates memory to store final results of the LogitBoost training algorithm
     * \param[in] input         %Input of the LogitBoost training algorithm
     * \param[in] parameter     Parameters of the algorithm
     * \param[in] method        LogitBoost computation method
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**s
    * Checks the result of model-based training
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method
    * \return Status of checking
    */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

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
using interface2::Parameter;
using interface1::Result;
using interface1::ResultPtr;

} // namespace training
/** @} */
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal
#endif // __GBT_CLASSIFICATION_TRAINING_TYPES_H__
