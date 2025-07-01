/* file: df_regression_model_impl.h */
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

#ifndef __DF_REGRESSION_MODEL_IMPL__
#define __DF_REGRESSION_MODEL_IMPL__

#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "algorithms/decision_forest/decision_forest_regression_model.h"
#include "src/algorithms/regression/regression_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace internal
{
class DAAL_EXPORT ModelImpl : public daal::algorithms::decision_forest::regression::Model,
                              public algorithms::regression::internal::ModelInternal,
                              public daal::algorithms::dtrees::internal::ModelImpl
{
public:
    typedef dtrees::internal::ModelImpl ImplType;
    typedef algorithms::regression::internal::ModelInternal RegressionImplType;
    typedef dtrees::internal::TreeImpRegression<> TreeType;

    ModelImpl(size_t nFeatures = 0) : RegressionImplType(nFeatures) {}
    ~ModelImpl() {}

    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return RegressionImplType::getNumberOfFeatures(); }

    //Implementation of decision_forest::regression::Model
    virtual size_t numberOfTrees() const DAAL_C11_OVERRIDE;
    virtual void traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const DAAL_C11_OVERRIDE;
    virtual void traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const DAAL_C11_OVERRIDE;
    virtual void clear() DAAL_C11_OVERRIDE { ImplType::clear(); }

    virtual void traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const DAAL_C11_OVERRIDE;
    virtual void traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const DAAL_C11_OVERRIDE;

    virtual services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;
    virtual services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;

    void copyModelReg(const ModelImpl & other, size_t idx, size_t global_count);
    bool add(const TreeType & tree, size_t nClasses, size_t iTree);

    virtual size_t getNumberOfTrees() const DAAL_C11_OVERRIDE;
};

} // namespace internal
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
