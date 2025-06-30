/* file: df_classification_model_impl.h */
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
//  Implementation of the class defining the decision forest model
//--
*/

#ifndef __DTREES_CLASSIFICATION_MODEL_IMPL__
#define __DTREES_CLASSIFICATION_MODEL_IMPL__

#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"
#include "src/algorithms/classifier/classifier_model_impl.h"
#include "algorithms/decision_forest/decision_forest_classification_model_builder.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace internal
{
class DAAL_EXPORT ModelImpl : public decision_forest::classification::Model,
                              public algorithms::classifier::internal::ModelInternal,
                              public dtrees::internal::ModelImpl
{
public:
    friend class decision_forest::classification::interface2::ModelBuilder;
    typedef dtrees::internal::ModelImpl ImplType;
    typedef algorithms::classifier::internal::ModelInternal ClassifierImplType;
    typedef dtrees::internal::TreeImpClassification<> TreeType;
    ModelImpl(size_t nFeatures = 0) : ClassifierImplType(nFeatures) {}
    ~ModelImpl() DAAL_C11_OVERRIDE {}

    //Implementation of classifier::Model
    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return ClassifierImplType::getNumberOfFeatures(); }
    virtual size_t getNFeatures() const DAAL_C11_OVERRIDE { return getNumberOfFeatures(); }
    virtual void setNFeatures(size_t nFeatures) DAAL_C11_OVERRIDE { ClassifierImplType::setNumberOfFeatures(nFeatures); }
    virtual size_t getNumberOfClasses() const DAAL_C11_OVERRIDE { return getNumClasses(); }

    //Implementation of decision_forest::classification::Model
    virtual size_t numberOfTrees() const DAAL_C11_OVERRIDE;
    virtual void traverseDF(size_t iTree, classifier::TreeNodeVisitor & visitor) const DAAL_C11_OVERRIDE;
    virtual void traverseBF(size_t iTree, classifier::TreeNodeVisitor & visitor) const DAAL_C11_OVERRIDE;
    virtual void clear() DAAL_C11_OVERRIDE { ImplType::clear(); }

    virtual services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;
    virtual services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;

    void copyModelCls(const ModelImpl & other, size_t idx, size_t global_count);
    bool add(const TreeType & tree, size_t nClasses, size_t iTree);

    virtual size_t getNumberOfTrees() const DAAL_C11_OVERRIDE;

    virtual void traverseDFS(size_t iTree, tree_utils::classification::TreeNodeVisitor & visitor) const DAAL_C11_OVERRIDE;
    virtual void traverseBFS(size_t iTree, tree_utils::classification::TreeNodeVisitor & visitor) const DAAL_C11_OVERRIDE;
};

} // namespace internal
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
