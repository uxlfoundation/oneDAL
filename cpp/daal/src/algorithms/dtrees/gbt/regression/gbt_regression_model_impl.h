/* file: gbt_regression_model_impl.h */
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
//  Implementation of the class defining the gradient boosted trees model
//--
*/

#ifndef __GBT_REGRESSION_MODEL_IMPL__
#define __GBT_REGRESSION_MODEL_IMPL__

#include "src/algorithms/dtrees/gbt/gbt_model_impl.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_model.h"
#include "algorithms/gradient_boosted_trees/gbt_regression_model_builder.h"
#include "src/algorithms/regression/regression_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace internal
{
class ModelImpl : public daal::algorithms::gbt::regression::Model,
                  public algorithms::regression::internal::ModelInternal,
                  public daal::algorithms::gbt::internal::ModelImpl
{
public:
    friend class gbt::regression::ModelBuilder;
    typedef gbt::internal::ModelImpl ImplType;
    typedef algorithms::regression::internal::ModelInternal RegressionImplType;

    ModelImpl(size_t nFeatures = 0) : RegressionImplType(nFeatures), _predictionBias(0.0f) {}
    ~ModelImpl() override {}

    virtual size_t getNumberOfFeatures() const override { return RegressionImplType::getNumberOfFeatures(); }

    //Implementation of regression::Model
    virtual size_t numberOfTrees() const override;
    virtual void traverseDF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const override;
    virtual void traverseBF(size_t iTree, algorithms::regression::TreeNodeVisitor & visitor) const override;
    virtual void clear() override { ImplType::clear(); }
    virtual void traverseDFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const override;
    virtual void traverseBFS(size_t iTree, tree_utils::regression::TreeNodeVisitor & visitor) const override;

    virtual void setPredictionBias(double value) override;
    virtual double getPredictionBias() const override;

    virtual services::Status serializeImpl(data_management::InputDataArchive * arch) override;
    virtual services::Status deserializeImpl(const data_management::OutputDataArchive * arch) override;

    virtual size_t getNumberOfTrees() const override;

private:
    /* global bias applied to predictions*/
    double _predictionBias;
};

} // namespace internal
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
