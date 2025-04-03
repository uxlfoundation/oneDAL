/* file: decision_tree_classification_model.h */
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
//  Implementation of the class defining the Decision tree classification model
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_MODEL_H__
#define __DECISION_TREE_CLASSIFICATION_MODEL_H__

#include "algorithms/classifier/classifier_model.h"
#include "algorithms/classifier/tree_traverse.h"
#include "data_management/data/aos_numeric_table.h"
#include "data_management/data/soa_numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/decision_tree/decision_tree_model.h"
#include "algorithms/tree_utils/tree_utils_classification.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup decision_tree_classification Decision Tree Classification
 * \copydoc daal::algorithms::decision_tree::classification
 * @ingroup classification
 */

/**
 * \brief Contains classes for Decision tree algorithm
 */
namespace decision_tree
{
/**
 * \brief Contains classes for Decision tree classification algorithm
 */
namespace classification
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__CLASSIFICATION__SPLITCRITERION"></a>
 * \brief Split criterion for Decision tree classification algorithm
 */
enum SplitCriterion
{
    gini     = 0, /*!< Gini index */
    infoGain = 1  /*!< Information gain */
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__MODEL"></a>
 * \brief %Base class for models trained with the Decision tree algorithm
 *
 * \par References
 *      - Parameter class
 *      - \ref training::interface2::Batch "training::Batch" class
 *      - \ref prediction::interface2::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public daal::algorithms::classifier::Model
{
public:
    DECLARE_MODEL_IFACE(Model, classifier::Model);

    /**
     * Constructs the model trained with the Decision tree algorithm
     * \param[in] nFeatures Number of features in the dataset
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    Model(size_t nFeatures = 0);

    /**
     * Constructs the model trained with the boosting algorithm
     * \param[in]  nFeatures Number of features in the dataset
     * \param[out] stat      Status of the model construction
     */
    static services::SharedPtr<Model> create(size_t nFeatures = 0, services::Status * stat = NULL);

    virtual ~Model();

    class ModelImpl;
    typedef services::SharedPtr<ModelImpl> ModelImplPtr;

    /**
     * Returns actual model implementation
     * \return Model implementation
     */
    const ModelImpl * impl() const { return _impl.get(); }

    /**
     * Returns actual model implementation
     * \return Model implementation
     */
    ModelImpl * impl() { return _impl.get(); }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
    *  Perform Depth First Traversal of tree
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    *  \DAAL_DEPRECATED_USE{ Model::traverseDFS }
    */
    void traverseDF(classifier::TreeNodeVisitor & visitor) const;

    /**
    *  Perform Breadth First Traversal of tree
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    *  \DAAL_DEPRECATED_USE{ Model::traverseBFS }
    */
    void traverseBF(classifier::TreeNodeVisitor & visitor) const;

    /**
    *  Perform Depth First Traversal of tree
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    void traverseDFS(tree_utils::classification::TreeNodeVisitor & visitor) const;

    /**
    *  Perform Breadth First Traversal of tree
    *  \param[in] visitor  This object gets notified when tree nodes are visited
    */
    void traverseBFS(tree_utils::classification::TreeNodeVisitor & visitor) const;

protected:
    Model(size_t nFeatures, services::Status & st);

    services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;

    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;

private:
    ModelImplPtr _impl; /*!< Model implementation */
};

typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;

} // namespace interface1

using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;

/**
 * \brief Contains version 2.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * @ingroup decision_tree_classification
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__DECISION_TREE__CLASSIFICATION__PARAMETER"></a>
 * \brief Decision tree algorithm parameters
 *
 * \snippet decision_tree/decision_tree_classification_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::classifier::Parameter
{
    /**
     *  Main constructor
     *  \param[in] nClasses                         Number of classes
     */
    Parameter(size_t nClasses = 2)
        : daal::algorithms::classifier::Parameter(nClasses),
          pruning(reducedErrorPruning),
          maxTreeDepth(0),
          minObservationsInLeafNodes(1),
          nBins(1),
          splitCriterion(infoGain)
    {}

    /**
     * Checks a parameter of the Decision tree algorithm
     */
    services::Status check() const DAAL_C11_OVERRIDE;

    Pruning pruning;                   /*!< Pruning method for Decision tree */
    size_t maxTreeDepth;               /*!< Maximum tree depth. 0 means unlimited depth. */
    size_t minObservationsInLeafNodes; /*!< Minimum number of observations in the leaf node. Can be any positive number. */
    size_t nBins;                      /*!< The number of bins used to compute probabilities of the observations belonging to the class.
                                             The only supported value for current version of the library is 1. */
    SplitCriterion splitCriterion;     /*!< Split criterion for Decision tree classification */
};
/* [Parameter source code] */
} // namespace interface2
using interface2::Parameter;

/** @} */
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
