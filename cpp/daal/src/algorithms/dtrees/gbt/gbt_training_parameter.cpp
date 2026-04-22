/* file: gbt_training_parameter.cpp */
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
//  Implementation of gradient boosted trees training parameter class
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_training_parameter.h"
#include "src/algorithms/dtrees/gbt/gbt_training_parameter.h"
#include "src/services/daal_strings.h"
#include "src/algorithms/dtrees/gbt/gbt_internal.h"
#include "algorithms/engines/engine.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
using namespace daal::services;

Parameter::Parameter()
    : splitMethod(defaultSplit),
      maxIterations(50),
      maxTreeDepth(6),
      shrinkage(0.3),
      minSplitLoss(0.),
      lambda(1.),
      observationsPerTreeFraction(1.),
      featuresPerNode(0),
      minObservationsInLeafNode(5),
      memorySavingMode(false),
      engine(engines::createEngine(engines::mt19937Engine)),
      minBinSize(5),
      maxBins(256),
      internalOptions(gbt::internal::parallelAll)
{}

Status checkImpl(const gbt::training::Parameter & prm)
{
    DAAL_CHECK_EX(prm.maxIterations, ErrorIncorrectParameter, ParameterName, maxIterationsStr());
    DAAL_CHECK_EX((prm.shrinkage > 0) && (prm.shrinkage <= 1), ErrorIncorrectParameter, ParameterName, shrinkageStr());
    DAAL_CHECK_EX((prm.minSplitLoss >= 0), ErrorIncorrectParameter, ParameterName, minSplitLossStr());
    DAAL_CHECK_EX((prm.lambda >= 0), ErrorIncorrectParameter, ParameterName, lambdaStr());
    DAAL_CHECK_EX((prm.observationsPerTreeFraction > 0) && (prm.observationsPerTreeFraction <= 1), ErrorIncorrectParameter, ParameterName,
                  observationsPerTreeFractionStr());
    DAAL_CHECK_EX(prm.minObservationsInLeafNode, ErrorIncorrectParameter, ParameterName, minObservationsInLeafNodeStr());
    if (prm.splitMethod == inexact)
    {
        DAAL_CHECK_EX((prm.maxBins >= 2), ErrorIncorrectParameter, ParameterName, maxBinsStr());
        DAAL_CHECK_EX((prm.minBinSize >= 1), ErrorIncorrectParameter, ParameterName, minBinSizeStr());
    }
    return Status();
}

} // namespace training
} // namespace gbt
} // namespace algorithms
} // namespace daal
