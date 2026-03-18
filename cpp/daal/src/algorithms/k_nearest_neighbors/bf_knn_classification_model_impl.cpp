/* file: bf_knn_classification_model_impl.cpp */
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

#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_model_impl.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "src/services/service_data_utils.h"
#include "src/algorithms/engines/mcg59/mcg59.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_K_NEAREST_NEIGHBOR_BF_MODEL_ID);

Model::Model(size_t nFeatures) : daal::algorithms::classifier::Model(), _impl(new ModelImpl(nFeatures)) {}

Model::~Model()
{
    delete _impl;
    _impl = nullptr;
}

Model::Model(size_t nFeatures, services::Status & st) : _impl(new ModelImpl(nFeatures))
{
    DAAL_CHECK_COND_ERROR(_impl, st, services::ErrorMemoryAllocationFailed);
}

services::Status Model::serializeImpl(data_management::InputDataArchive * arch)
{
    daal::algorithms::classifier::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    return _impl->serialImpl<data_management::InputDataArchive, false>(arch);
}

services::Status Model::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    daal::algorithms::classifier::Model::serialImpl<const data_management::OutputDataArchive, true>(arch);
    return _impl->serialImpl<const data_management::OutputDataArchive, true>(arch);
}

size_t Model::getNumberOfFeatures() const
{
    return _impl->getNumberOfFeatures();
}

Parameter::Parameter(size_t nClasses, size_t nNeighbors, DataUseInModel dataUse, DAAL_UINT64 resToCompute, DAAL_UINT64 resToEvaluate,
                     VoteWeights vote)
    : daal::algorithms::classifier::Parameter(nClasses),
      k(nNeighbors),
      dataUseInModel(dataUse),
      resultsToCompute(resToCompute),
      voteWeights(vote),
      engine(engines::mcg59::Batch<>::create())
{
    this->resultsToEvaluate = resToEvaluate;
}

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(this->nClasses > 1 && this->nClasses < static_cast<size_t>(services::internal::MaxVal<int>::get()),
                  services::ErrorIncorrectParameter, services::ParameterName, nClassesStr());
    DAAL_CHECK_EX(this->k > 0 && this->k <= static_cast<size_t>(services::internal::MaxVal<int>::get()), services::ErrorIncorrectParameter,
                  services::ParameterName, kStr());
    return services::Status();
}

} // namespace interface1
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal
