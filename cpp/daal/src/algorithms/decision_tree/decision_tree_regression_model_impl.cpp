/* file: decision_tree_regression_model_impl.cpp */
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
//  Implementation of the class defining the Decision tree model
//--
*/

#include "src/algorithms/decision_tree/decision_tree_regression_model_impl.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace interface1
{
using namespace daal::data_management;
using namespace daal::services;

__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_DECISION_TREE_REGRESSION_MODEL_ID);

Model::Model() : _impl(new ModelImpl()) {}

Model::~Model() {}

Model::Model(services::Status & st) : _impl(new ModelImpl())
{
    DAAL_CHECK_COND_ERROR(_impl, st, services::ErrorMemoryAllocationFailed);
}

services::SharedPtr<Model> Model::create(services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL(Model);
}

size_t Model::getNumberOfFeatures() const
{
    return _impl->getNumberOfFeatures();
}

services::Status Model::serializeImpl(data_management::InputDataArchive * arch)
{
    algorithms::regression::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    _impl->serialImpl<data_management::InputDataArchive, false>(arch);

    return services::Status();
}

services::Status Model::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    algorithms::regression::Model::serialImpl<const data_management::OutputDataArchive, true>(arch);
    _impl->serialImpl<const data_management::OutputDataArchive, true>(
        arch, COMPUTE_DAAL_VERSION(arch->getMajorVersion(), arch->getMinorVersion(), arch->getUpdateVersion()));

    return services::Status();
}

void Model::traverseDF(algorithms::regression::TreeNodeVisitor & visitor) const
{
    _impl->traverseDF(visitor);
}

void Model::traverseBF(algorithms::regression::TreeNodeVisitor & visitor) const
{
    _impl->traverseBF(visitor);
}

void Model::traverseDFS(tree_utils::regression::TreeNodeVisitor & visitor) const
{
    _impl->traverseDFS(visitor);
}

void Model::traverseBFS(tree_utils::regression::TreeNodeVisitor & visitor) const
{
    _impl->traverseBFS(visitor);
}

services::Status Parameter::check() const
{
    services::Status s;
    // Inherited.
    DAAL_CHECK_STATUS(s, daal::algorithms::Parameter::check());

    DAAL_CHECK_EX(minObservationsInLeafNodes >= 1, services::ErrorIncorrectParameter, services::ParameterName, minObservationsInLeafNodesStr());
    return s;
}

} // namespace interface1
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
