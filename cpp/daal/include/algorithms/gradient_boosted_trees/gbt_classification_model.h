/* file: gbt_classification_model.h */
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
//  Implementation of class defining gradient boosted trees classification model.
//--
*/

#ifndef __GBT_CLASSIFICATION_MODEL_H__
#define __GBT_CLASSIFICATION_MODEL_H__

#include "algorithms/classifier/classifier_model.h"
#include "algorithms/regression/tree_traverse.h"
#include "algorithms/tree_utils/tree_utils_regression.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
/**
 * @defgroup gbt_classification Gradient Boosted Trees Classification
 * \copydoc daal::algorithms::gbt::classification
 * @ingroup classification
 */
/**
 * \brief Contains classes for the gbt classification algorithm
 */
namespace classification
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup gbt_classification
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__CLASSIFICATION__MODEL"></a>
 * \brief %Model of the classifier trained by the gbt::training::Batch algorithm.
 *
 * \par References
 *      - \ref classification::training::interface2::Batch "training::Batch" class
 *      - \ref classification::prediction::interface2::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public classifier::Model
{
public:
    DECLARE_MODEL(Model, classifier::Model)

    /**
     * Constructs the model
     * \param[in]  nFeatures Number of features in the dataset
     * \param[out] stat      Status of the model construction
     */
    static services::SharedPtr<Model> create(size_t nFeatures, services::Status * stat = NULL);

    /**
     *  Gets number of trees in the model
     *  \return number of trees
     *  \DAAL_DEPRECATED_USE{ Model::getNumberOfTrees }
     */
    virtual size_t numberOfTrees() const = 0;

    /**
     *  Performs Depth First Traversal of i-th tree
     *  \param[in] iTree    Index of the tree to traverse
     *  \param[in] visitor  This object gets notified when tree nodes are visited
     *  \DAAL_DEPRECATED_USE{ Model::traverseDFS }
     */
    /* regression traversing is used as classification model is similar to regression in gbt algorithms */
    virtual void traverseDF(size_t iTree, daal::algorithms::regression::TreeNodeVisitor & visitor) const = 0;

    /**
     *  Performs Breadth First Traversal of i-th tree
     *  \param[in] iTree    Index of the tree to traverse
     *  \param[in] visitor  This object gets notified when tree nodes are visited
     *  \DAAL_DEPRECATED_USE{ Model::traverseBFS }
     */
    /* regression traversing is used as classification model is similar to regression in gbt algorithms */
    virtual void traverseBF(size_t iTree, daal::algorithms::regression::TreeNodeVisitor & visitor) const = 0;

    /**
     *  Removes all trees from the model
     */
    virtual void clear() = 0;

    /**
    *  Perform Depth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    virtual void traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const = 0;

    /**
    *  Perform Breadth First Traversal of i-th tree
    *  \param[in] iTree    Index of the tree to traverse
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    virtual void traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const = 0;

    /**
     *  Gets number of trees in the model
     *  \return number of trees
     */
    virtual size_t getNumberOfTrees() const = 0;

    /**
     * \brief Set the Prediction Bias term
     *
     * \param value global prediction bias
     */
    virtual void setPredictionBias(double value) = 0;

    /**
     * \brief Get the Prediction Bias term
     *
     * \return double prediction bias
     */
    virtual double getPredictionBias() const = 0;

protected:
    Model() : classifier::Model() {}
};
/** @} */
typedef services::SharedPtr<Model> ModelPtr;
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;

} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal
#endif
