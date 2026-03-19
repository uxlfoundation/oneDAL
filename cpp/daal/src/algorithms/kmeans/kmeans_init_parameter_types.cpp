/* file: kmeans_init_parameter_types.cpp */
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
//  Implementation of kmeans classes.
//--
*/

#include "algorithms/kmeans/kmeans_init_types.h"
#include "services/daal_defines.h"
#include "src/services/daal_strings.h"
#include "src/algorithms/engines/mt19937/mt19937.h"
using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace interface1
{
Parameter::Parameter(size_t _nClusters, size_t _offset, size_t _seed)
    : nClusters(_nClusters),
      offset(_offset),
      nRowsTotal(0),
      seed(_seed),
      oversamplingFactor(0.5),
      nRounds(5),
      engine(engines::engine_type::mt19937)
{}

/**
 * Constructs parameters of the algorithm that computes initial clusters for the K-Means algorithm
 * by copying another parameters object
 * \param[in] other    Parameters of the K-Means algorithm
 */
Parameter::Parameter(const Parameter & other)
    : nClusters(other.nClusters),
      offset(other.offset),
      nRowsTotal(other.nRowsTotal),
      seed(other.seed),
      oversamplingFactor(other.oversamplingFactor),
      nRounds(other.nRounds),
      engine(other.engine)
{}

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(nClusters > 0, ErrorIncorrectParameter, ParameterName, nClustersStr());
    return services::Status();
}

DistributedStep2LocalPlusPlusParameter::DistributedStep2LocalPlusPlusParameter(size_t _nClusters, bool bFirstIteration)
    : Parameter(_nClusters), outputForStep5Required(false), firstIteration(bFirstIteration)
{}

DistributedStep2LocalPlusPlusParameter::DistributedStep2LocalPlusPlusParameter(const DistributedStep2LocalPlusPlusParameter & other)
    : Parameter(other), outputForStep5Required(other.outputForStep5Required), firstIteration(other.firstIteration)
{}

services::Status DistributedStep2LocalPlusPlusParameter::check() const
{
    return services::Status();
}

} // namespace interface1

namespace interface2
{
Parameter::Parameter(size_t nClusters, size_t offset, size_t seed) : interface1::Parameter(nClusters, offset, seed), nTrials(1) {}

/**
 * Constructs parameters of the algorithm that computes initial clusters for the K-Means algorithm
 * by copying another parameters object
 * \param[in] other    Parameters of the K-Means algorithm
 */
Parameter::Parameter(const Parameter & other) : interface1::Parameter(other), nTrials(other.nTrials) {}

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(nTrials > 0, ErrorIncorrectParameter, ParameterName, nTrialsStr());

    return interface1::Parameter::check();
}

} // namespace interface2

} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal
