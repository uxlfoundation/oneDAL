/* file: df_classification_model_builder.cpp */
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
//  Implementation of the class defining the decision forest classification model builder
//--
*/

#include "algorithms/decision_forest/decision_forest_classification_model_builder.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"
#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/forest/classification/df_classification_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::dtrees::internal;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace interface2
{

ModelBuilder::ModelBuilder() : _nClasses(0), _model(new decision_forest::classification::internal::ModelImpl()) {}

services::Status ModelBuilder::initialize(const size_t nClasses, const size_t nTrees)
{
    auto modelImpl = new decision_forest::classification::internal::ModelImpl();
    DAAL_CHECK_MALLOC(modelImpl)
    _model.reset(modelImpl);
    decision_forest::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl, ModelPtr>(_model);

    modelImplRef.resize(nTrees);
    modelImplRef._impurityTables.reset();
    modelImplRef._nNodeSampleTables.reset();
    modelImplRef._nTree.set(nTrees);
    return Status();
}

services::Status ModelBuilder::createTreeInternal(const size_t nNodes, TreeId & resId)
{
    decision_forest::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl, ModelPtr>(_model);
    services::Status status = daal::algorithms::dtrees::internal::createTreeInternal(modelImplRef._serializationData, nNodes, resId);
    if (status.ok())
    {
        const auto probTbl                = HomogenNumericTable<double>::create(nNodes, _nClasses, NumericTable::doAllocate);
        (*(modelImplRef._probTbl))[resId] = probTbl;
    }
    return status;
}

services::Status ModelBuilder::addLeafNodeInternal(const TreeId treeId, const NodeId parentId, const size_t position, const size_t classLabel,
                                                   const double cover, NodeId & res)
{
    decision_forest::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl, ModelPtr>(_model);
    if (classLabel > _nClasses)
    {
        return services::Status(services::ErrorID::ErrorIncorrectParameter);
    }
    return daal::algorithms::dtrees::internal::addLeafNodeInternal<size_t>(modelImplRef._serializationData, treeId, parentId, position, classLabel,
                                                                           cover, res, modelImplRef._probTbl);
}

bool checkProba(const double * const proba, const size_t nClasses)
{
    double acc           = 0.0;
    const double epsilon = 10e-6;
    for (size_t index = 0; index < nClasses; ++index)
    {
        if (proba[index] < 0.0 || proba[index] > 1.0)
        {
            return false;
        }
        acc += proba[index];
    }
    if (acc < 1.0 - epsilon || acc > 1.0 + epsilon)
    {
        return false;
    }
    return true;
}

services::Status ModelBuilder::addLeafNodeByProbaInternal(const TreeId treeId, const NodeId parentId, const size_t position,
                                                          const double * const proba, const double cover, NodeId & res)
{
    decision_forest::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl, ModelPtr>(_model);
    if (!checkProba(proba, _nClasses))
    {
        return services::Status(services::ErrorID::ErrorIncorrectParameter);
    }
    return daal::algorithms::dtrees::internal::addLeafNodeInternal<size_t>(modelImplRef._serializationData, treeId, parentId, position, 0, cover, res,
                                                                           modelImplRef._probTbl, proba, _nClasses);
}

services::Status ModelBuilder::addSplitNodeInternal(const TreeId treeId, const NodeId parentId, const size_t position, const size_t featureIndex,
                                                    const double featureValue, const int defaultLeft, const double cover, NodeId & res)
{
    decision_forest::classification::internal::ModelImpl & modelImplRef =
        daal::algorithms::dtrees::internal::getModelRef<decision_forest::classification::internal::ModelImpl, ModelPtr>(_model);
    return daal::algorithms::dtrees::internal::addSplitNodeInternal(modelImplRef._serializationData, treeId, parentId, position, featureIndex,
                                                                    featureValue, defaultLeft, cover, res);
}

} // namespace interface2
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
